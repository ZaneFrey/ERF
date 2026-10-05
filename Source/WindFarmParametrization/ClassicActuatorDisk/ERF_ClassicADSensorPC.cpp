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
    const auto& ground = m_model.ground_elevations();
    const auto& yaw = m_model.yaw_state();
    const int nturb = static_cast<int>(xloc.size());
    AMREX_ALWAYS_ASSERT(static_cast<int>(yloc.size()) == nturb);
    AMREX_ALWAYS_ASSERT(static_cast<int>(ground.size()) == nturb);
    AMREX_ALWAYS_ASSERT(static_cast<int>(yaw.size()) == nturb);

    Gpu::HostVector<ParticleType> host_particles;
    std::array<Gpu::HostVector<ParticleReal>,
               ClassicADSensorRealIdx::ncomps> host_real;
    std::array<Gpu::HostVector<int>,
               ClassicADSensorIntIdx::ncomps> host_int;

    if (ParallelDescriptor::IOProcessor()) {
        const Real rotor_radius = m_model.rotor_radius();
        const Real target_spacing = m_model.spacing_was_supplied()
            ? m_model.actuator_spacing()
            : Geom(0).CellSize(2);
        const int nr = amrex::max(
            1, static_cast<int>(std::llround(
                rotor_radius / target_spacing)));
        const Real dr = rotor_radius / static_cast<Real>(nr);
        const Real upstream_distance = Real(2.0) * rotor_radius *
            m_model.yaw_sensor_distance_by_D();

        for (int turbine_id = 0; turbine_id < nturb; ++turbine_id) {
            const Real psi = yaw[turbine_id].psi_ref;
            const Real nx = std::cos(psi);
            const Real ny = std::sin(psi);
            const Real e1x = -ny;
            const Real e1y = nx;
            const Real xcenter =
                xloc[turbine_id] - upstream_distance*nx;
            const Real ycenter =
                yloc[turbine_id] - upstream_distance*ny;
            const Real zhub =
                ground[turbine_id] + m_model.hub_height();
            int element_id = 0;

            for (int ring_id = 0; ring_id < nr; ++ring_id) {
                const Real radius =
                    (static_cast<Real>(ring_id) + Real(0.5)) * dr;
                const int ntheta = amrex::max(
                    1, static_cast<int>(std::llround(
                        Real(2.0)*classic_pi*radius/target_spacing)));
                const Real area = classic_pi *
                    static_cast<Real>((ring_id+1)*(ring_id+1) -
                                      ring_id*ring_id) *
                    dr*dr / static_cast<Real>(ntheta);

                for (int azimuth_id = 0;
                     azimuth_id < ntheta;
                     ++azimuth_id, ++element_id) {
                    const Real theta = Real(2.0)*classic_pi *
                        (static_cast<Real>(azimuth_id) + Real(0.5)) /
                        static_cast<Real>(ntheta);
                    const Real radial_in_plane = radius*std::cos(theta);

                    ParticleType particle;
                    particle.id() = ParticleType::NextID();
                    particle.cpu() = ParallelDescriptor::MyProc();
                    particle.pos(0) = xcenter + radial_in_plane*e1x;
                    particle.pos(1) = ycenter + radial_in_plane*e1y;
                    particle.pos(2) = zhub + radius*std::sin(theta);
                    host_particles.push_back(particle);

                    host_real[ClassicADSensorRealIdx::radius].push_back(radius);
                    host_real[ClassicADSensorRealIdx::theta].push_back(theta);
                    host_real[ClassicADSensorRealIdx::area].push_back(area);
                    host_int[ClassicADSensorIntIdx::turbine].push_back(turbine_id);
                    host_int[ClassicADSensorIntIdx::ring].push_back(ring_id);
                    host_int[ClassicADSensorIntIdx::element].push_back(element_id);
                }
            }
        }
    }

    ParticleTileType incoming;
    incoming.resize(host_particles.size());
    if (!host_particles.empty()) {
        Gpu::copyAsync(Gpu::hostToDevice,
                       host_particles.begin(), host_particles.end(),
                       incoming.GetArrayOfStructs().begin());
        auto& soa = incoming.GetStructOfArrays();
        for (int component = 0;
             component < ClassicADSensorRealIdx::ncomps;
             ++component) {
            Gpu::copyAsync(Gpu::hostToDevice,
                           host_real[component].begin(),
                           host_real[component].end(),
                           soa.GetRealData(component).begin());
        }
        for (int component = 0;
             component < ClassicADSensorIntIdx::ncomps;
             ++component) {
            Gpu::copyAsync(Gpu::hostToDevice,
                           host_int[component].begin(),
                           host_int[component].end(),
                           soa.GetIntData(component).begin());
        }
        Gpu::streamSynchronize();
    }
    AddParticlesAtLevel(incoming, 0);
}

void
ClassicADSensorPC::update_positions ()
{
    const int nturb = static_cast<int>(m_model.x_locations().size());
    Vector<Real> psi_host(nturb);
    for (int turbine_id = 0; turbine_id < nturb; ++turbine_id) {
        psi_host[turbine_id] = m_model.yaw_state()[turbine_id].psi_ref;
    }

    Gpu::DeviceVector<Real> d_xloc(nturb);
    Gpu::DeviceVector<Real> d_yloc(nturb);
    Gpu::DeviceVector<Real> d_ground(nturb);
    Gpu::DeviceVector<Real> d_psi(nturb);
    Gpu::copyAsync(Gpu::hostToDevice,
                   m_model.x_locations().begin(),
                   m_model.x_locations().end(), d_xloc.begin());
    Gpu::copyAsync(Gpu::hostToDevice,
                   m_model.y_locations().begin(),
                   m_model.y_locations().end(), d_yloc.begin());
    Gpu::copyAsync(Gpu::hostToDevice,
                   m_model.ground_elevations().begin(),
                   m_model.ground_elevations().end(), d_ground.begin());
    Gpu::copyAsync(Gpu::hostToDevice,
                   psi_host.begin(), psi_host.end(), d_psi.begin());
    const Real* xloc = d_xloc.data();
    const Real* yloc = d_yloc.data();
    const Real* ground = d_ground.data();
    const Real* psi = d_psi.data();
    const Real upstream_distance = Real(2.0)*m_model.rotor_radius() *
        m_model.yaw_sensor_distance_by_D();
    const Real hub_height = m_model.hub_height();

    for (ParIterType pti(*this, 0); pti.isValid(); ++pti) {
        auto& tile = ParticlesAt(0, pti);
        auto& aos = tile.GetArrayOfStructs();
        auto& soa = tile.GetStructOfArrays();
        const int np = aos.numParticles();
        auto* particles = aos().data();
        const auto* radius =
            soa.GetRealData(ClassicADSensorRealIdx::radius).data();
        const auto* theta =
            soa.GetRealData(ClassicADSensorRealIdx::theta).data();
        const auto* turbine =
            soa.GetIntData(ClassicADSensorIntIdx::turbine).data();

        ParallelFor(np, [=] AMREX_GPU_DEVICE (int ip) noexcept {
            const int turbine_id = turbine[ip];
            const Real nx = std::cos(psi[turbine_id]);
            const Real ny = std::sin(psi[turbine_id]);
            const Real e1x = -ny;
            const Real e1y = nx;
            const Real radial_in_plane = radius[ip]*std::cos(theta[ip]);
            particles[ip].pos(0) = xloc[turbine_id] -
                upstream_distance*nx + radial_in_plane*e1x;
            particles[ip].pos(1) = yloc[turbine_id] -
                upstream_distance*ny + radial_in_plane*e1y;
            particles[ip].pos(2) = ground[turbine_id] + hub_height +
                radius[ip]*std::sin(theta[ip]);
        });
    }
    Gpu::streamSynchronize();
}

void
ClassicADSensorPC::sample_uv (int lev,
                              const MultiFab& u,
                              const MultiFab& v,
                              const MultiFab& w,
                              Vector<Real>& u_mean,
                              Vector<Real>& v_mean) const
{
    AMREX_ALWAYS_ASSERT(lev == 0);
    const int nturb = static_cast<int>(m_model.x_locations().size());
    u_mean.assign(nturb, Real(0.0));
    v_mean.assign(nturb, Real(0.0));
    if (nturb == 0) {
        return;
    }

    const Geometry& geom = Geom(lev);
    const auto prob_lo = geom.ProbLoArray();
    const auto prob_hi = geom.ProbHiArray();
    const auto inverse_cell_size = geom.InvCellSizeArray();
    const auto periodic = geom.isPeriodicArray();
    const GpuArray<Real,AMREX_SPACEDIM> domain_length{{
        AMREX_D_DECL(prob_hi[0]-prob_lo[0],
                     prob_hi[1]-prob_lo[1],
                     prob_hi[2]-prob_lo[2])}};

    Vector<Real> psi_host(nturb);
    for (int turbine_id = 0; turbine_id < nturb; ++turbine_id) {
        psi_host[turbine_id] = m_model.yaw_state()[turbine_id].psi_ref;
    }
    Gpu::DeviceVector<Real> d_xloc(nturb);
    Gpu::DeviceVector<Real> d_yloc(nturb);
    Gpu::DeviceVector<Real> d_ground(nturb);
    Gpu::DeviceVector<Real> d_psi(nturb);
    Gpu::copyAsync(Gpu::hostToDevice,
                   m_model.x_locations().begin(),
                   m_model.x_locations().end(), d_xloc.begin());
    Gpu::copyAsync(Gpu::hostToDevice,
                   m_model.y_locations().begin(),
                   m_model.y_locations().end(), d_yloc.begin());
    Gpu::copyAsync(Gpu::hostToDevice,
                   m_model.ground_elevations().begin(),
                   m_model.ground_elevations().end(), d_ground.begin());
    Gpu::copyAsync(Gpu::hostToDevice,
                   psi_host.begin(), psi_host.end(), d_psi.begin());
    const Real* xloc = d_xloc.data();
    const Real* yloc = d_yloc.data();
    const Real* ground = d_ground.data();
    const Real* psi = d_psi.data();
    const Real upstream_distance = Real(2.0)*m_model.rotor_radius() *
        m_model.yaw_sensor_distance_by_D();
    const Real hub_height = m_model.hub_height();

    Gpu::DeviceVector<Real> d_u_sum(nturb, Real(0.0));
    Gpu::DeviceVector<Real> d_v_sum(nturb, Real(0.0));
    Gpu::DeviceVector<Real> d_area_sum(nturb, Real(0.0));
    Gpu::DeviceVector<Real> d_geometry_error(nturb, Real(0.0));
    Real* u_sum_device = d_u_sum.data();
    Real* v_sum_device = d_v_sum.data();
    Real* area_sum_device = d_area_sum.data();
    Real* geometry_error_device = d_geometry_error.data();

    for (ParConstIterType pti(*this, lev); pti.isValid(); ++pti) {
        const int grid = pti.index();
        const auto& tile = ParticlesAt(lev, pti);
        const auto& aos = tile.GetArrayOfStructs();
        const auto& soa = tile.GetStructOfArrays();
        const int np = aos.numParticles();
        const auto* particles = aos().data();
        const auto* area =
            soa.GetRealData(ClassicADSensorRealIdx::area).data();
        const auto* radius =
            soa.GetRealData(ClassicADSensorRealIdx::radius).data();
        const auto* turbine =
            soa.GetIntData(ClassicADSensorIntIdx::turbine).data();
        const GpuArray<Array4<const Real>, AMREX_SPACEDIM> velocity_arrays{{
            u[grid].const_array(),
            v[grid].const_array(),
            w[grid].const_array()}};

        ParallelFor(np, [=] AMREX_GPU_DEVICE (int ip) noexcept {
            ParticleReal velocity[AMREX_SPACEDIM];
            mac_interpolate(particles[ip], prob_lo, inverse_cell_size,
                            velocity_arrays, velocity);
            const int turbine_id = turbine[ip];
            Gpu::Atomic::Add(&u_sum_device[turbine_id],
                             velocity[0]*area[ip]);
            Gpu::Atomic::Add(&v_sum_device[turbine_id],
                             velocity[1]*area[ip]);
            Gpu::Atomic::Add(&area_sum_device[turbine_id], area[ip]);

            const Real nx = std::cos(psi[turbine_id]);
            const Real ny = std::sin(psi[turbine_id]);
            Real rx = particles[ip].pos(0) -
                (xloc[turbine_id] - upstream_distance*nx);
            Real ry = particles[ip].pos(1) -
                (yloc[turbine_id] - upstream_distance*ny);
            const Real rz = particles[ip].pos(2) -
                (ground[turbine_id] + hub_height);
            if (periodic[0]) {
                if (rx > Real(0.5)*domain_length[0]) {
                    rx -= domain_length[0];
                }
                if (rx < -Real(0.5)*domain_length[0]) {
                    rx += domain_length[0];
                }
            }
            if (periodic[1]) {
                if (ry > Real(0.5)*domain_length[1]) {
                    ry -= domain_length[1];
                }
                if (ry < -Real(0.5)*domain_length[1]) {
                    ry += domain_length[1];
                }
            }
            const Real plane_error = std::abs(rx*nx + ry*ny);
            const Real radial_error = std::abs(
                std::sqrt(rx*rx + ry*ry + rz*rz) - radius[ip]);
            Gpu::Atomic::Add(&geometry_error_device[turbine_id],
                             area[ip]*(plane_error + radial_error));
        });
    }
    Gpu::streamSynchronize();

    Vector<Real> u_sum(nturb);
    Vector<Real> v_sum(nturb);
    Vector<Real> area_sum(nturb);
    Vector<Real> geometry_error(nturb);
    Gpu::copy(Gpu::deviceToHost,
              d_u_sum.begin(), d_u_sum.end(), u_sum.begin());
    Gpu::copy(Gpu::deviceToHost,
              d_v_sum.begin(), d_v_sum.end(), v_sum.begin());
    Gpu::copy(Gpu::deviceToHost,
              d_area_sum.begin(), d_area_sum.end(), area_sum.begin());
    Gpu::copy(Gpu::deviceToHost,
              d_geometry_error.begin(), d_geometry_error.end(),
              geometry_error.begin());
    ParallelAllReduce::Sum(u_sum.data(), u_sum.size(),
                           ParallelContext::CommunicatorAll());
    ParallelAllReduce::Sum(v_sum.data(), v_sum.size(),
                           ParallelContext::CommunicatorAll());
    ParallelAllReduce::Sum(area_sum.data(), area_sum.size(),
                           ParallelContext::CommunicatorAll());
    ParallelAllReduce::Sum(geometry_error.data(), geometry_error.size(),
                           ParallelContext::CommunicatorAll());

    const Real rotor_area = classic_pi *
        m_model.rotor_radius() * m_model.rotor_radius();
    for (int turbine_id = 0; turbine_id < nturb; ++turbine_id) {
        if (area_sum[turbine_id] <=
            std::numeric_limits<Real>::epsilon()*rotor_area) {
            Abort("ClassicAD yaw sensor has no globally owned particle area");
        }
        if (std::abs(area_sum[turbine_id] - rotor_area) / rotor_area >
            Real(1.0e-11)) {
            Abort("ClassicAD yaw sensor particle areas do not sum to the rotor area");
        }
        if (geometry_error[turbine_id] / area_sum[turbine_id] >
            Real(1.0e-10)*amrex::max(m_model.rotor_radius(), Real(1.0))) {
            Abort("ClassicAD yaw sensor particles do not form a rigid planar disk");
        }
        u_mean[turbine_id] = u_sum[turbine_id] / area_sum[turbine_id];
        v_mean[turbine_id] = v_sum[turbine_id] / area_sum[turbine_id];
    }
}

#endif
