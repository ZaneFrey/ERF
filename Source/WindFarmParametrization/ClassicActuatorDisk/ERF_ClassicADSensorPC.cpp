#include <ERF_ClassicADSensorPC.H>

#if defined(ERF_USE_PARTICLES) && defined(ERF_USE_WINDFARM)

#include <AMReX_GpuAtomic.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_TracerParticle_mod_K.H>

#include <array>
#include <cmath>
#include <limits>

using namespace amrex;

namespace {

constexpr Real classic_pi = 3.141592653589793238462643383279502884;

} // namespace

void
ClassicADSensorPC::rebuild ()
{
    clearParticles();
    const auto& xloc = m_model.x_locations();
    const auto& yloc = m_model.y_locations();
    const auto& yaw = m_model.yaw_state();
    const int nturb = static_cast<int>(xloc.size());
    AMREX_ALWAYS_ASSERT(static_cast<int>(yloc.size()) == nturb);
    AMREX_ALWAYS_ASSERT(static_cast<int>(yaw.size()) == nturb);

    Gpu::HostVector<ParticleType> host_particles;
    std::array<Gpu::HostVector<ParticleReal>, ClassicADSensorRealIdx::ncomps> host_real;
    std::array<Gpu::HostVector<int>, ClassicADSensorIntIdx::ncomps> host_int;

    if (ParallelDescriptor::IOProcessor()) {
        const Real radius = m_model.rotor_radius();
        const Real target_spacing = m_model.spacing_was_supplied()
                                  ? m_model.actuator_spacing()
                                  : Geom(0).CellSize(2);
        const int nr = amrex::max(1, static_cast<int>(std::llround(radius/target_spacing)));
        const Real dr = radius/static_cast<Real>(nr);
        const Real upstream = 2.0*radius*m_model.yaw_sensor_distance_by_D();
        const Real zhub = Geom(0).ProbLo(2)+m_model.hub_height();

        for (int it = 0; it < nturb; ++it) {
            const Real psi = yaw[it].psi_ref;
            const Real nx = std::cos(psi);
            const Real ny = std::sin(psi);
            const Real e1x = -ny;
            const Real e1y = nx;
            const Real xcenter = xloc[it]-upstream*nx;
            const Real ycenter = yloc[it]-upstream*ny;
            int elem = 0;
            for (int j = 0; j < nr; ++j) {
                const Real r = (static_cast<Real>(j)+0.5)*dr;
                const int ntheta = amrex::max(
                    1, static_cast<int>(std::llround(2.0*classic_pi*r/target_spacing)));
                const Real area = classic_pi*static_cast<Real>((j+1)*(j+1)-j*j)*dr*dr /
                                  static_cast<Real>(ntheta);
                for (int m = 0; m < ntheta; ++m, ++elem) {
                    const Real theta = 2.0*classic_pi*(static_cast<Real>(m)+0.5) /
                                       static_cast<Real>(ntheta);
                    const Real er1 = r*std::cos(theta);
                    ParticleType p;
                    p.id() = ParticleType::NextID();
                    p.cpu() = ParallelDescriptor::MyProc();
                    p.pos(0) = xcenter+er1*e1x;
                    p.pos(1) = ycenter+er1*e1y;
                    p.pos(2) = zhub+r*std::sin(theta);
                    host_particles.push_back(p);
                    host_real[ClassicADSensorRealIdx::radius].push_back(r);
                    host_real[ClassicADSensorRealIdx::theta].push_back(theta);
                    host_real[ClassicADSensorRealIdx::area].push_back(area);
                    host_int[ClassicADSensorIntIdx::turbine].push_back(it);
                    host_int[ClassicADSensorIntIdx::ring].push_back(j);
                    host_int[ClassicADSensorIntIdx::element].push_back(elem);
                }
            }
        }
    }

    ParticleTileType incoming;
    incoming.resize(host_particles.size());
    if (!host_particles.empty()) {
        Gpu::copyAsync(Gpu::hostToDevice, host_particles.begin(), host_particles.end(),
                       incoming.GetArrayOfStructs().begin());
        auto& soa = incoming.GetStructOfArrays();
        for (int n = 0; n < ClassicADSensorRealIdx::ncomps; ++n) {
            Gpu::copyAsync(Gpu::hostToDevice, host_real[n].begin(), host_real[n].end(),
                           soa.GetRealData(n).begin());
        }
        for (int n = 0; n < ClassicADSensorIntIdx::ncomps; ++n) {
            Gpu::copyAsync(Gpu::hostToDevice, host_int[n].begin(), host_int[n].end(),
                           soa.GetIntData(n).begin());
        }
        Gpu::streamSynchronize();
    }
    AddParticlesAtLevel(incoming, 0);
}

void
ClassicADSensorPC::update_positions ()
{
    const int nturb = static_cast<int>(m_model.x_locations().size());
    Vector<Real> psi_h(nturb);
    for (int it = 0; it < nturb; ++it) { psi_h[it] = m_model.yaw_state()[it].psi_ref; }
    Gpu::DeviceVector<Real> d_xloc(nturb), d_yloc(nturb), d_psi(nturb);
    Gpu::copyAsync(Gpu::hostToDevice, m_model.x_locations().begin(),
                   m_model.x_locations().end(), d_xloc.begin());
    Gpu::copyAsync(Gpu::hostToDevice, m_model.y_locations().begin(),
                   m_model.y_locations().end(), d_yloc.begin());
    Gpu::copyAsync(Gpu::hostToDevice, psi_h.begin(), psi_h.end(), d_psi.begin());
    const Real* xloc = d_xloc.data();
    const Real* yloc = d_yloc.data();
    const Real* psi = d_psi.data();
    const Real upstream = 2.0*m_model.rotor_radius()*m_model.yaw_sensor_distance_by_D();
    const Real zhub = Geom(0).ProbLo(2)+m_model.hub_height();

    for (ParIterType pti(*this, 0); pti.isValid(); ++pti) {
        auto& tile = ParticlesAt(0, pti);
        auto& aos = tile.GetArrayOfStructs();
        auto& soa = tile.GetStructOfArrays();
        const int np = aos.numParticles();
        auto* particles = aos().data();
        const auto* radius = soa.GetRealData(ClassicADSensorRealIdx::radius).data();
        const auto* theta = soa.GetRealData(ClassicADSensorRealIdx::theta).data();
        const auto* turb = soa.GetIntData(ClassicADSensorIntIdx::turbine).data();
        ParallelFor(np, [=] AMREX_GPU_DEVICE (int ip) noexcept {
            const int it = turb[ip];
            const Real nx = std::cos(psi[it]);
            const Real ny = std::sin(psi[it]);
            const Real e1x = -ny;
            const Real e1y = nx;
            const Real er1 = radius[ip]*std::cos(theta[ip]);
            particles[ip].pos(0) = xloc[it]-upstream*nx+er1*e1x;
            particles[ip].pos(1) = yloc[it]-upstream*ny+er1*e1y;
            particles[ip].pos(2) = zhub+radius[ip]*std::sin(theta[ip]);
        });
    }
    Gpu::streamSynchronize();
}

void
ClassicADSensorPC::sample_uv (int lev, const MultiFab& u, const MultiFab& v,
                              const MultiFab& w, Vector<Real>& u_mean,
                              Vector<Real>& v_mean) const
{
    const int nturb = static_cast<int>(m_model.x_locations().size());
    u_mean.assign(nturb, 0.0);
    v_mean.assign(nturb, 0.0);
    if (nturb == 0) { return; }

    const Geometry& geom = Geom(lev);
    const auto plo = geom.ProbLoArray();
    const auto phi = geom.ProbHiArray();
    const auto dxi = geom.InvCellSizeArray();
    const auto periodic = geom.isPeriodicArray();
    const GpuArray<Real,AMREX_SPACEDIM> length{{
        AMREX_D_DECL(phi[0]-plo[0], phi[1]-plo[1], phi[2]-plo[2])}};
    Vector<Real> psi_h(nturb);
    for (int it = 0; it < nturb; ++it) { psi_h[it] = m_model.yaw_state()[it].psi_ref; }
    Gpu::DeviceVector<Real> d_xloc(nturb), d_yloc(nturb), d_psi(nturb);
    Gpu::copyAsync(Gpu::hostToDevice, m_model.x_locations().begin(),
                   m_model.x_locations().end(), d_xloc.begin());
    Gpu::copyAsync(Gpu::hostToDevice, m_model.y_locations().begin(),
                   m_model.y_locations().end(), d_yloc.begin());
    Gpu::copyAsync(Gpu::hostToDevice, psi_h.begin(), psi_h.end(), d_psi.begin());
    const Real* xloc = d_xloc.data();
    const Real* yloc = d_yloc.data();
    const Real* psi = d_psi.data();
    const Real upstream = 2.0*m_model.rotor_radius()*m_model.yaw_sensor_distance_by_D();
    const Real zhub = plo[2]+m_model.hub_height();
    Gpu::DeviceVector<Real> d_u_sum(nturb, 0.0), d_v_sum(nturb, 0.0),
                            d_area(nturb, 0.0), d_geometry_error(nturb, 0.0);
    Real* u_sum = d_u_sum.data();
    Real* v_sum = d_v_sum.data();
    Real* area_sum = d_area.data();
    Real* geometry_error = d_geometry_error.data();

    for (ParConstIterType pti(*this, lev); pti.isValid(); ++pti) {
        const int grid = pti.index();
        const auto& tile = ParticlesAt(lev, pti);
        const auto& aos = tile.GetArrayOfStructs();
        const auto& soa = tile.GetStructOfArrays();
        const int np = aos.numParticles();
        const auto* particles = aos().data();
        const auto* area = soa.GetRealData(ClassicADSensorRealIdx::area).data();
        const auto* radius = soa.GetRealData(ClassicADSensorRealIdx::radius).data();
        const auto* turb = soa.GetIntData(ClassicADSensorIntIdx::turbine).data();
        const GpuArray<Array4<const Real>, AMREX_SPACEDIM> vel_arr{{
            u[grid].const_array(), v[grid].const_array(), w[grid].const_array()}};
        ParallelFor(np, [=] AMREX_GPU_DEVICE (int ip) noexcept {
            ParticleReal vel[3];
            mac_interpolate(particles[ip], plo, dxi, vel_arr, vel);
            Gpu::Atomic::Add(&u_sum[turb[ip]], vel[0]*area[ip]);
            Gpu::Atomic::Add(&v_sum[turb[ip]], vel[1]*area[ip]);
            Gpu::Atomic::Add(&area_sum[turb[ip]], area[ip]);
            const int it = turb[ip];
            const Real nx = std::cos(psi[it]);
            const Real ny = std::sin(psi[it]);
            Real rx = particles[ip].pos(0)-(xloc[it]-upstream*nx);
            Real ry = particles[ip].pos(1)-(yloc[it]-upstream*ny);
            Real rz = particles[ip].pos(2)-zhub;
            if (periodic[0]) {
                if (rx >  0.5*length[0]) { rx -= length[0]; }
                if (rx < -0.5*length[0]) { rx += length[0]; }
            }
            if (periodic[1]) {
                if (ry >  0.5*length[1]) { ry -= length[1]; }
                if (ry < -0.5*length[1]) { ry += length[1]; }
            }
            const Real plane_error = std::abs(rx*nx+ry*ny);
            const Real radial_error = std::abs(std::sqrt(rx*rx+ry*ry+rz*rz)-radius[ip]);
            Gpu::Atomic::Add(&geometry_error[it], area[ip]*(plane_error+radial_error));
        });
    }
    Gpu::streamSynchronize();

    Vector<Real> u_sum_h(nturb), v_sum_h(nturb), area_h(nturb), geometry_error_h(nturb);
    Gpu::copy(Gpu::deviceToHost, d_u_sum.begin(), d_u_sum.end(), u_sum_h.begin());
    Gpu::copy(Gpu::deviceToHost, d_v_sum.begin(), d_v_sum.end(), v_sum_h.begin());
    Gpu::copy(Gpu::deviceToHost, d_area.begin(), d_area.end(), area_h.begin());
    Gpu::copy(Gpu::deviceToHost, d_geometry_error.begin(), d_geometry_error.end(),
              geometry_error_h.begin());
    ParallelAllReduce::Sum(u_sum_h.data(), u_sum_h.size(), ParallelContext::CommunicatorAll());
    ParallelAllReduce::Sum(v_sum_h.data(), v_sum_h.size(), ParallelContext::CommunicatorAll());
    ParallelAllReduce::Sum(area_h.data(), area_h.size(), ParallelContext::CommunicatorAll());
    ParallelAllReduce::Sum(geometry_error_h.data(), geometry_error_h.size(),
                           ParallelContext::CommunicatorAll());

    const Real rotor_area = classic_pi*m_model.rotor_radius()*m_model.rotor_radius();
    for (int it = 0; it < nturb; ++it) {
        if (area_h[it] <= std::numeric_limits<Real>::epsilon()*rotor_area) {
            Abort("ClassicAD yaw sensor has no globally owned particle area");
        }
        if (std::abs(area_h[it]-rotor_area)/rotor_area > 1.0e-11) {
            Abort("ClassicAD yaw sensor particle areas do not sum to the rotor area");
        }
        if (geometry_error_h[it]/area_h[it] >
            1.0e-10*amrex::max(m_model.rotor_radius(), 1.0)) {
            Abort("ClassicAD yaw sensor particles do not form a rigid planar disk");
        }
        u_mean[it] = u_sum_h[it]/area_h[it];
        v_mean[it] = v_sum_h[it]/area_h[it];
    }
}

#endif
