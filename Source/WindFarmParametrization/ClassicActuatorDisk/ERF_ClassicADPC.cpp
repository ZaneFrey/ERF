#include <ERF_ClassicADPC.H>

#if defined(ERF_USE_PARTICLES) && defined(ERF_USE_WINDFARM)

#include <ERF_IndexDefines.H>

#include <AMReX_GpuAtomic.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_TracerParticle_mod_K.H>

#include <array>
#include <cmath>
#include <limits>

using namespace amrex;

namespace {

constexpr Real classic_pi = 3.141592653589793238462643383279502884;

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void rotor_basis (Real face_angle_deg,
                  Real& nx, Real& ny,
                  Real& e1x, Real& e1y) noexcept
{
    const Real psi = (face_angle_deg - 90.0) * classic_pi / 180.0;
    nx = std::cos(psi);
    ny = std::sin(psi);
    e1x = -ny;
    e1y = nx;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real gaussian_kernel (Real x, Real y, Real z,
                      Real px, Real py, Real pz,
                      Real nx, Real ny, Real e1x, Real e1y,
                      Real epsn, Real eps1, Real eps2) noexcept
{
    const Real dx = x - px;
    const Real dy = y - py;
    const Real dz = z - pz;
    const Real dn = dx*nx + dy*ny;
    const Real d1 = dx*e1x + dy*e1y;
    const Real d2 = dz;
    if (std::abs(dn) > 3.0*epsn ||
        std::abs(d1) > 3.0*eps1 ||
        std::abs(d2) > 3.0*eps2) {
        return 0.0;
    }
    return std::exp(-0.5*(dn*dn/(epsn*epsn) +
                          d1*d1/(eps1*eps1) +
                          d2*d2/(eps2*eps2)));
}

} // namespace

void
ClassicADPC::rebuild (const Vector<int>& owner_levels)
{
    clearParticles();
    m_owner_levels = owner_levels;

    const auto& xloc = m_model.x_locations();
    const auto& yloc = m_model.y_locations();
    const auto& ground = m_model.ground_elevations();
    const auto& face_angles = m_model.disk_face_angles_deg();
    const int nturb = static_cast<int>(xloc.size());
    AMREX_ALWAYS_ASSERT(static_cast<int>(yloc.size()) == nturb);
    AMREX_ALWAYS_ASSERT(static_cast<int>(ground.size()) == nturb);
    AMREX_ALWAYS_ASSERT(static_cast<int>(face_angles.size()) == nturb);
    AMREX_ALWAYS_ASSERT(static_cast<int>(owner_levels.size()) == nturb);

    m_model.initialize_turbine_state(nturb);

    const Real radius = m_model.rotor_radius();
    const int nlev = finestLevel() + 1;

    for (int lev = 0; lev < nlev; ++lev) {
        Gpu::HostVector<ParticleType> host_particles;
        std::array<Gpu::HostVector<ParticleReal>, ClassicADRealIdx::ncomps> host_real;
        std::array<Gpu::HostVector<int>, ClassicADIntIdx::ncomps> host_int;

        if (ParallelDescriptor::IOProcessor()) {
            const Real target_spacing = m_model.spacing_was_supplied()
                                      ? m_model.actuator_spacing()
                                      : Geom(lev).CellSize(2);
            const int nr = amrex::max(1, static_cast<int>(std::llround(radius/target_spacing)));
            const Real dr = radius / static_cast<Real>(nr);

            for (int it = 0; it < nturb; ++it) {
                if (owner_levels[it] != lev) { continue; }

                Real nx, ny, e1x, e1y;
                rotor_basis(face_angles[it], nx, ny, e1x, e1y);
                const Real zhub = Geom(0).ProbLo(2) + m_model.hub_height();
                int elem = 0;

                for (int j = 0; j < nr; ++j) {
                    const Real r = (static_cast<Real>(j) + 0.5) * dr;
                    const int ntheta = amrex::max(
                        1, static_cast<int>(std::llround(2.0*classic_pi*r/target_spacing)));
                    const Real area = classic_pi *
                        (static_cast<Real>((j+1)*(j+1) - j*j)) * dr*dr /
                        static_cast<Real>(ntheta);

                    for (int m = 0; m < ntheta; ++m, ++elem) {
                        const Real theta = 2.0*classic_pi*(static_cast<Real>(m)+0.5) /
                                           static_cast<Real>(ntheta);
                        const Real er1 = r*std::cos(theta);
                        const Real erz = r*std::sin(theta);

                        ParticleType p;
                        p.id() = ParticleType::NextID();
                        p.cpu() = ParallelDescriptor::MyProc();
                        p.pos(0) = xloc[it] + er1*e1x;
                        p.pos(1) = yloc[it] + er1*e1y;
                        p.pos(2) = zhub + erz;
                        host_particles.push_back(p);

                        host_real[ClassicADRealIdx::radius].push_back(r);
                        host_real[ClassicADRealIdx::theta].push_back(theta);
                        host_real[ClassicADRealIdx::area].push_back(area);
                        host_real[ClassicADRealIdx::force_x].push_back(0.0);
                        host_real[ClassicADRealIdx::force_y].push_back(0.0);
                        host_real[ClassicADRealIdx::force_z].push_back(0.0);
                        host_int[ClassicADIntIdx::turbine].push_back(it);
                        host_int[ClassicADIntIdx::ring].push_back(j);
                        host_int[ClassicADIntIdx::element].push_back(elem);
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
            for (int n = 0; n < ClassicADRealIdx::ncomps; ++n) {
                Gpu::copyAsync(Gpu::hostToDevice, host_real[n].begin(), host_real[n].end(),
                               soa.GetRealData(n).begin());
            }
            for (int n = 0; n < ClassicADIntIdx::ncomps; ++n) {
                Gpu::copyAsync(Gpu::hostToDevice, host_int[n].begin(), host_int[n].end(),
                               soa.GetIntData(n).begin());
            }
            Gpu::streamSynchronize();
        }
        AddParticlesAtLevel(incoming, lev);
    }

}

void
ClassicADPC::compute_sources (int lev, Real time, Real dt,
                              Real start_time, Real ramp_time,
                              const MultiFab& cons,
                              const MultiFab& u,
                              const MultiFab& v,
                              const MultiFab& w,
                              MultiFab& xmom_src,
                              MultiFab& ymom_src,
                              MultiFab& zmom_src)
{
    xmom_src.setVal(0.0);
    ymom_src.setVal(0.0);
    zmom_src.setVal(0.0);

    const Real eps_time = 1.0e-12;
    if (time + eps_time < start_time) { return; }

    const int nturb = static_cast<int>(m_model.x_locations().size());
    if (nturb == 0) { return; }

    const Geometry& geom = Geom(lev);
    const auto plo = geom.ProbLoArray();
    const auto dxi = geom.InvCellSizeArray();
    const auto dx = geom.CellSizeArray();
    const Real cell_volume = dx[0]*dx[1]*dx[2];

    Gpu::DeviceVector<Real> d_sum_u(nturb, 0.0);
    Gpu::DeviceVector<Real> d_sum_rho(nturb, 0.0);
    Gpu::DeviceVector<Real> d_sum_area(nturb, 0.0);

    Real* sum_u = d_sum_u.data();
    Real* sum_rho = d_sum_rho.data();
    Real* sum_area = d_sum_area.data();
    const auto& sample_angles_h = m_model.disk_face_angles_deg();
    Gpu::DeviceVector<Real> d_sample_angles(sample_angles_h.size());
    Gpu::copyAsync(Gpu::hostToDevice, sample_angles_h.begin(), sample_angles_h.end(),
                   d_sample_angles.begin());
    const Real* sample_angles = d_sample_angles.data();

    for (ParIterType pti(*this, lev); pti.isValid(); ++pti) {
        const int grid = pti.index();
        auto& tile = ParticlesAt(lev, pti);
        auto& aos = tile.GetArrayOfStructs();
        auto& soa = tile.GetStructOfArrays();
        const int np = aos.numParticles();
        auto* particles = aos().data();
        const auto* area = soa.GetRealData(ClassicADRealIdx::area).data();
        const auto* turb = soa.GetIntData(ClassicADIntIdx::turbine).data();

        const GpuArray<Array4<const Real>, AMREX_SPACEDIM> vel_arr{{
            u[grid].const_array(), v[grid].const_array(), w[grid].const_array()}};
        const auto rho_arr = cons[grid].const_array();

        ParallelFor(np, [=] AMREX_GPU_DEVICE (int ip) noexcept {
            ParticleReal vel[3];
            ParticleReal rho[1];
            mac_interpolate(particles[ip], plo, dxi, vel_arr, vel);
            cic_interpolate(particles[ip], plo, dxi, rho_arr, rho, 1);
            Real nx, ny, e1x, e1y;
            rotor_basis(sample_angles[turb[ip]], nx, ny, e1x, e1y);
            const Real un = vel[0]*nx + vel[1]*ny;
            Gpu::Atomic::Add(&sum_u[turb[ip]], un*area[ip]);
            Gpu::Atomic::Add(&sum_rho[turb[ip]], rho[0]*area[ip]);
            Gpu::Atomic::Add(&sum_area[turb[ip]], area[ip]);
        });
    }
    Gpu::streamSynchronize();

    Vector<Real> area_sum(nturb), velocity_sum(nturb), density_sum(nturb);
    Gpu::copy(Gpu::deviceToHost, d_sum_area.begin(), d_sum_area.end(), area_sum.begin());
    Gpu::copy(Gpu::deviceToHost, d_sum_u.begin(), d_sum_u.end(), velocity_sum.begin());
    Gpu::copy(Gpu::deviceToHost, d_sum_rho.begin(), d_sum_rho.end(), density_sum.begin());
    ParallelAllReduce::Sum(area_sum.data(), area_sum.size(), ParallelContext::CommunicatorAll());
    ParallelAllReduce::Sum(velocity_sum.data(), velocity_sum.size(), ParallelContext::CommunicatorAll());
    ParallelAllReduce::Sum(density_sum.data(), density_sum.size(), ParallelContext::CommunicatorAll());

    const Real gamma_unclamped = (ramp_time > 0.0)
                               ? 1.0 - std::exp(-amrex::max(time-start_time, 0.0)/ramp_time)
                               : 1.0;
    const Real gamma = amrex::max(0.0, amrex::min(1.0, gamma_unclamped));
    const Real rotor_area = classic_pi*m_model.rotor_radius()*m_model.rotor_radius();
    auto& state = m_model.turbine_state();

    Vector<Real> thrust_applied(nturb, 0.0);
    Vector<Real> torque_applied(nturb, 0.0);
    Vector<Real> area_normalization(nturb, rotor_area);
    Vector<Real> face_angles_h = m_model.disk_face_angles_deg();

    for (int it = 0; it < nturb; ++it) {
        if (m_owner_levels[it] != lev) { continue; }
        if (area_sum[it] <= std::numeric_limits<Real>::epsilon()*rotor_area) {
            Abort("ClassicAD turbine has no locally/global-owned actuator area");
        }
        const Real rel_area_err = std::abs(area_sum[it]-rotor_area)/rotor_area;
        if (rel_area_err > 1.0e-11) {
            Abort("ClassicAD actuator-element areas do not sum to the rotor area");
        }
        area_normalization[it] = area_sum[it];
        auto& s = state[it];
        s.actuator_area = area_sum[it];
        s.disk_velocity_raw = velocity_sum[it]/area_sum[it];
        s.disk_density = density_sum[it]/area_sum[it];

        if (!s.memory_initialized || m_model.memory_time() == 0.0) {
            s.disk_velocity_filtered = s.disk_velocity_raw;
            s.memory_initialized = true;
        } else {
            const Real alpha = 1.0 - std::exp(-dt/m_model.memory_time());
            s.disk_velocity_filtered += alpha *
                (s.disk_velocity_raw-s.disk_velocity_filtered);
        }

        const Real ud = s.disk_velocity_filtered;
        s.thrust_target = 0.5*s.disk_density*rotor_area*m_model.ctprime()*ud*ud;
        s.thrust_applied = gamma*s.thrust_target;
        s.axial_power = s.thrust_target*ud;
        s.rotor_power = 0.0;
        s.omega = 0.0;
        s.torque_target = 0.0;
        s.torque_applied = 0.0;
        s.les_torque = 0.0;
        if (m_model.wake_rotation_enabled() && std::abs(ud) > 1.0e-12) {
            s.omega = m_model.tsr()*ud/m_model.rotor_radius();
            s.rotor_power = 0.5*s.disk_density*rotor_area*m_model.cpprime()*ud*ud*ud;
            s.torque_target = 0.5*s.disk_density*rotor_area*m_model.cpprime()*ud*ud*
                              m_model.rotor_radius()/m_model.tsr();
            s.torque_applied = gamma*s.torque_target;
        }
        thrust_applied[it] = s.thrust_applied;
        torque_applied[it] = s.torque_applied;
    }

    Gpu::DeviceVector<Real> d_thrust(nturb);
    Gpu::DeviceVector<Real> d_torque(nturb);
    Gpu::DeviceVector<Real> d_area_norm(nturb);
    Gpu::DeviceVector<Real> d_face_angles(nturb);
    Gpu::copyAsync(Gpu::hostToDevice, thrust_applied.begin(), thrust_applied.end(), d_thrust.begin());
    Gpu::copyAsync(Gpu::hostToDevice, torque_applied.begin(), torque_applied.end(), d_torque.begin());
    Gpu::copyAsync(Gpu::hostToDevice, area_normalization.begin(), area_normalization.end(), d_area_norm.begin());
    Gpu::copyAsync(Gpu::hostToDevice, face_angles_h.begin(), face_angles_h.end(), d_face_angles.begin());
    Gpu::DeviceVector<Real> d_raw_torque(nturb, 0.0);
    Real* raw_torque = d_raw_torque.data();

    for (ParIterType pti(*this, lev); pti.isValid(); ++pti) {
        auto& tile = ParticlesAt(lev, pti);
        auto& aos = tile.GetArrayOfStructs();
        auto& soa = tile.GetStructOfArrays();
        const int np = aos.numParticles();
        const auto* radius = soa.GetRealData(ClassicADRealIdx::radius).data();
        const auto* area = soa.GetRealData(ClassicADRealIdx::area).data();
        const auto* turb = soa.GetIntData(ClassicADIntIdx::turbine).data();
        const Real* qapp = d_torque.data();
        const Real* anorm = d_area_norm.data();
        ParallelFor(np, [=] AMREX_GPU_DEVICE (int ip) noexcept {
            if (qapp[turb[ip]] != 0.0) {
                const Real fraw = qapp[turb[ip]]*area[ip]/(anorm[turb[ip]]*radius[ip]);
                Gpu::Atomic::Add(&raw_torque[turb[ip]], radius[ip]*fraw);
            }
        });
    }
    Gpu::streamSynchronize();

    Vector<Real> raw_torque_h(nturb, 0.0);
    Gpu::copy(Gpu::deviceToHost, d_raw_torque.begin(), d_raw_torque.end(), raw_torque_h.begin());
    ParallelAllReduce::Sum(raw_torque_h.data(), raw_torque_h.size(), ParallelContext::CommunicatorAll());
    Vector<Real> torque_scale(nturb, 0.0);
    for (int it = 0; it < nturb; ++it) {
        if (std::abs(raw_torque_h[it]) > 1.0e-30) {
            torque_scale[it] = torque_applied[it]/raw_torque_h[it];
        }
    }
    Gpu::DeviceVector<Real> d_torque_scale(nturb);
    Gpu::copyAsync(Gpu::hostToDevice, torque_scale.begin(), torque_scale.end(), d_torque_scale.begin());

    for (ParIterType pti(*this, lev); pti.isValid(); ++pti) {
        auto& tile = ParticlesAt(lev, pti);
        auto& aos = tile.GetArrayOfStructs();
        auto& soa = tile.GetStructOfArrays();
        const int np = aos.numParticles();
        const auto* radius = soa.GetRealData(ClassicADRealIdx::radius).data();
        const auto* theta = soa.GetRealData(ClassicADRealIdx::theta).data();
        const auto* area = soa.GetRealData(ClassicADRealIdx::area).data();
        auto* fx = soa.GetRealData(ClassicADRealIdx::force_x).data();
        auto* fy = soa.GetRealData(ClassicADRealIdx::force_y).data();
        auto* fz = soa.GetRealData(ClassicADRealIdx::force_z).data();
        const auto* turb = soa.GetIntData(ClassicADIntIdx::turbine).data();
        const Real* tapp = d_thrust.data();
        const Real* qapp = d_torque.data();
        const Real* anorm = d_area_norm.data();
        const Real* angles = d_face_angles.data();
        const Real* qscale = d_torque_scale.data();

        ParallelFor(np, [=] AMREX_GPU_DEVICE (int ip) noexcept {
            const int it = turb[ip];
            Real nx, ny, e1x, e1y;
            rotor_basis(angles[it], nx, ny, e1x, e1y);
            const Real axial = -tapp[it]*area[ip]/anorm[it];
            Real tx = 0.0, ty = 0.0, tz = 0.0;
            if (qapp[it] != 0.0) {
                const Real mag = qscale[it]*qapp[it]*area[ip]/(anorm[it]*radius[ip]);
                const Real st = std::sin(theta[ip]);
                const Real ct = std::cos(theta[ip]);
                // -e_theta, with e_theta = n x e_r.
                tx = -ny*st*mag;
                ty =  nx*st*mag;
                tz = -ct*mag;
            }
            fx[ip] = axial*nx + tx;
            fy[ip] = axial*ny + ty;
            fz[ip] = tz;
        });
    }
    Gpu::streamSynchronize();

    // Validate the finite actuator-element loading before regularization.
    Gpu::DeviceVector<Real> d_force_sum(3*nturb, 0.0);
    Gpu::DeviceVector<Real> d_force_abs_sum(3*nturb, 0.0);
    Gpu::DeviceVector<Real> d_axial_sum(nturb, 0.0);
    Gpu::DeviceVector<Real> d_torque_sum(nturb, 0.0);
    Gpu::DeviceVector<Real> d_tangent_abs(nturb, 0.0);
    Real* force_sum = d_force_sum.data();
    Real* force_abs_sum = d_force_abs_sum.data();
    Real* axial_sum = d_axial_sum.data();
    Real* torque_sum = d_torque_sum.data();
    Real* tangent_abs = d_tangent_abs.data();

    for (ParIterType pti(*this, lev); pti.isValid(); ++pti) {
        auto& tile = ParticlesAt(lev, pti);
        auto& aos = tile.GetArrayOfStructs();
        auto& soa = tile.GetStructOfArrays();
        const int np = aos.numParticles();
        const auto* radius = soa.GetRealData(ClassicADRealIdx::radius).data();
        const auto* theta = soa.GetRealData(ClassicADRealIdx::theta).data();
        const auto* fx = soa.GetRealData(ClassicADRealIdx::force_x).data();
        const auto* fy = soa.GetRealData(ClassicADRealIdx::force_y).data();
        const auto* fz = soa.GetRealData(ClassicADRealIdx::force_z).data();
        const auto* turb = soa.GetIntData(ClassicADIntIdx::turbine).data();
        const Real* angles = d_face_angles.data();

        ParallelFor(np, [=] AMREX_GPU_DEVICE (int ip) noexcept {
            const int it = turb[ip];
            Real nx, ny, e1x, e1y;
            rotor_basis(angles[it], nx, ny, e1x, e1y);
            const Real fn = fx[ip]*nx + fy[ip]*ny;
            const Real ftx = fx[ip] - fn*nx;
            const Real fty = fy[ip] - fn*ny;
            const Real ftz = fz[ip];
            // Use intrinsic rotor coordinates so this moment is independent
            // of AMReX's periodic wrapping of the stored particle position.
            const Real radial_in_plane = radius[ip]*std::cos(theta[ip]);
            const Real rx = radial_in_plane*e1x;
            const Real ry = radial_in_plane*e1y;
            const Real rz = radius[ip]*std::sin(theta[ip]);
            const Real cx = ry*fz[ip] - rz*fy[ip];
            const Real cy = rz*fx[ip] - rx*fz[ip];
            Gpu::Atomic::Add(&force_sum[3*it  ], fx[ip]);
            Gpu::Atomic::Add(&force_sum[3*it+1], fy[ip]);
            Gpu::Atomic::Add(&force_sum[3*it+2], fz[ip]);
            Gpu::Atomic::Add(&force_abs_sum[3*it  ], std::abs(fx[ip]));
            Gpu::Atomic::Add(&force_abs_sum[3*it+1], std::abs(fy[ip]));
            Gpu::Atomic::Add(&force_abs_sum[3*it+2], std::abs(fz[ip]));
            Gpu::Atomic::Add(&axial_sum[it], fn);
            Gpu::Atomic::Add(&torque_sum[it], cx*nx + cy*ny);
            Gpu::Atomic::Add(&tangent_abs[it], std::sqrt(ftx*ftx+fty*fty+ftz*ftz));
        });
    }
    Gpu::streamSynchronize();

    Vector<Real> force_sum_h(3*nturb), force_abs_sum_h(3*nturb), axial_sum_h(nturb),
                 torque_sum_h(nturb), tangent_abs_h(nturb);
    Gpu::copy(Gpu::deviceToHost, d_force_sum.begin(), d_force_sum.end(), force_sum_h.begin());
    Gpu::copy(Gpu::deviceToHost, d_force_abs_sum.begin(), d_force_abs_sum.end(),
              force_abs_sum_h.begin());
    Gpu::copy(Gpu::deviceToHost, d_axial_sum.begin(), d_axial_sum.end(), axial_sum_h.begin());
    Gpu::copy(Gpu::deviceToHost, d_torque_sum.begin(), d_torque_sum.end(), torque_sum_h.begin());
    Gpu::copy(Gpu::deviceToHost, d_tangent_abs.begin(), d_tangent_abs.end(), tangent_abs_h.begin());
    ParallelAllReduce::Sum(force_sum_h.data(), force_sum_h.size(), ParallelContext::CommunicatorAll());
    ParallelAllReduce::Sum(force_abs_sum_h.data(), force_abs_sum_h.size(),
                           ParallelContext::CommunicatorAll());
    ParallelAllReduce::Sum(axial_sum_h.data(), axial_sum_h.size(), ParallelContext::CommunicatorAll());
    ParallelAllReduce::Sum(torque_sum_h.data(), torque_sum_h.size(), ParallelContext::CommunicatorAll());
    ParallelAllReduce::Sum(tangent_abs_h.data(), tangent_abs_h.size(), ParallelContext::CommunicatorAll());
    for (int it = 0; it < nturb; ++it) {
        if (m_owner_levels[it] != lev) { continue; }
        const Real thrust_scale = amrex::max(std::abs(thrust_applied[it]), 1.0);
        const Real torque_ref = amrex::max(std::abs(torque_applied[it]), 1.0);
        if (std::abs(axial_sum_h[it]+thrust_applied[it]) > 1.0e-10*thrust_scale) {
            Abort("ClassicAD finite axial elements do not recover the applied thrust");
        }
        if (std::abs(torque_sum_h[it]+torque_applied[it]) > 1.0e-10*torque_ref) {
            Abort("ClassicAD finite tangential elements do not recover the applied torque");
        }
        Real nx, ny, e1x, e1y;
        rotor_basis(face_angles_h[it], nx, ny, e1x, e1y);
        const Real net_tx = force_sum_h[3*it  ] - axial_sum_h[it]*nx;
        const Real net_ty = force_sum_h[3*it+1] - axial_sum_h[it]*ny;
        const Real net_tz = force_sum_h[3*it+2];
        // The Cartesian and axial totals above are independent GPU reductions.
        // When tangential loading is disabled their exact difference is zero,
        // so retain a relative tolerance by scaling roundoff against the total
        // actuator load rather than an absolute unit force.
        const Real force_scale = std::sqrt(
            force_abs_sum_h[3*it  ]*force_abs_sum_h[3*it  ] +
            force_abs_sum_h[3*it+1]*force_abs_sum_h[3*it+1] +
            force_abs_sum_h[3*it+2]*force_abs_sum_h[3*it+2]);
        const Real tangent_scale = amrex::max(amrex::max(tangent_abs_h[it], force_scale), 1.0);
        if (std::sqrt(net_tx*net_tx+net_ty*net_ty+net_tz*net_tz) > 1.0e-10*tangent_scale) {
            Abort("ClassicAD symmetric tangential loading has a nonzero net force");
        }
    }

    // Scatter each finite element force independently to the three MAC grids.
    Gpu::DeviceVector<Real> d_deposited_force(3, 0.0);
    Gpu::DeviceVector<Real> d_les_torque(nturb, 0.0);
    Gpu::DeviceVector<Real> d_xhub(nturb);
    Gpu::DeviceVector<Real> d_yhub(nturb);
    const auto& xhub_h = m_model.x_locations();
    const auto& yhub_h = m_model.y_locations();
    Gpu::copyAsync(Gpu::hostToDevice, xhub_h.begin(), xhub_h.end(), d_xhub.begin());
    Gpu::copyAsync(Gpu::hostToDevice, yhub_h.begin(), yhub_h.end(), d_yhub.begin());
    Real* deposited_force = d_deposited_force.data();
    Real* les_torque = d_les_torque.data();
    const Real* xhub = d_xhub.data();
    const Real* yhub = d_yhub.data();
    const Real zhub_les = Geom(0).ProbLo(2) + m_model.hub_height();
    const auto phi = geom.ProbHiArray();
    const GpuArray<Real,AMREX_SPACEDIM> prob_length{{
        AMREX_D_DECL(phi[0]-plo[0], phi[1]-plo[1], phi[2]-plo[2])}};
    const auto is_periodic = geom.isPeriodicArray();
    for (ParIterType pti(*this, lev); pti.isValid(); ++pti) {
        const int grid = pti.index();
        auto& tile = ParticlesAt(lev, pti);
        auto& aos = tile.GetArrayOfStructs();
        auto& soa = tile.GetStructOfArrays();
        const int np = aos.numParticles();
        const auto* particles = aos().data();
        const auto* fx = soa.GetRealData(ClassicADRealIdx::force_x).data();
        const auto* fy = soa.GetRealData(ClassicADRealIdx::force_y).data();
        const auto* fz = soa.GetRealData(ClassicADRealIdx::force_z).data();
        const auto* turb = soa.GetIntData(ClassicADIntIdx::turbine).data();
        const Real* angles = d_face_angles.data();

        auto sx = xmom_src[grid].array();
        auto sy = ymom_src[grid].array();
        auto sz = zmom_src[grid].array();
        const Box bx = xmom_src[grid].box();
        const Box by = ymom_src[grid].box();
        const Box bz = zmom_src[grid].box();

        ParallelFor(np, [=] AMREX_GPU_DEVICE (int ip) noexcept {
            Real nx, ny, e1x, e1y;
            rotor_basis(angles[turb[ip]], nx, ny, e1x, e1y);
            const Real epsn = std::sqrt((nx*dx[0])*(nx*dx[0]) +
                                        (ny*dx[1])*(ny*dx[1]));
            const Real eps1 = std::sqrt((e1x*dx[0])*(e1x*dx[0]) +
                                        (e1y*dx[1])*(e1y*dx[1]));
            const Real eps2 = dx[2];
            const Real hx = 3.0*(std::abs(nx)*epsn + std::abs(e1x)*eps1);
            const Real hy = 3.0*(std::abs(ny)*epsn + std::abs(e1y)*eps1);
            const Real hz = 3.0*eps2;
            const Real px = particles[ip].pos(0);
            const Real py = particles[ip].pos(1);
            const Real pz = particles[ip].pos(2);

            // Each tuple is the face offset in index coordinates.
            const Real offsets[3][3] = {{0.0,0.5,0.5}, {0.5,0.0,0.5}, {0.5,0.5,0.0}};
            const Box boxes[3] = {bx, by, bz};
            const Real forces[3] = {fx[ip], fy[ip], fz[ip]};

            for (int comp = 0; comp < 3; ++comp) {
                const Box& fb = boxes[comp];
                int ilo = static_cast<int>(std::ceil((px-hx-plo[0])*dxi[0]-offsets[comp][0]));
                int ihi = static_cast<int>(std::floor((px+hx-plo[0])*dxi[0]-offsets[comp][0]));
                int jlo = static_cast<int>(std::ceil((py-hy-plo[1])*dxi[1]-offsets[comp][1]));
                int jhi = static_cast<int>(std::floor((py+hy-plo[1])*dxi[1]-offsets[comp][1]));
                int klo = static_cast<int>(std::ceil((pz-hz-plo[2])*dxi[2]-offsets[comp][2]));
                int khi = static_cast<int>(std::floor((pz+hz-plo[2])*dxi[2]-offsets[comp][2]));
                ilo = amrex::max(ilo, fb.smallEnd(0)); ihi = amrex::min(ihi, fb.bigEnd(0));
                jlo = amrex::max(jlo, fb.smallEnd(1)); jhi = amrex::min(jhi, fb.bigEnd(1));
                klo = amrex::max(klo, fb.smallEnd(2)); khi = amrex::min(khi, fb.bigEnd(2));

                Real weight_volume = 0.0;
                for (int k = klo; k <= khi; ++k) {
                    for (int j = jlo; j <= jhi; ++j) {
                        for (int i = ilo; i <= ihi; ++i) {
                            const Real x = plo[0] + (static_cast<Real>(i)+offsets[comp][0])*dx[0];
                            const Real y = plo[1] + (static_cast<Real>(j)+offsets[comp][1])*dx[1];
                            const Real z = plo[2] + (static_cast<Real>(k)+offsets[comp][2])*dx[2];
                            weight_volume += gaussian_kernel(x,y,z,px,py,pz,nx,ny,e1x,e1y,
                                                             epsn,eps1,eps2)*cell_volume;
                        }
                    }
                }
                if (weight_volume <= 0.0) { continue; }
                Real integrated_force = 0.0;
                Real integrated_torque = 0.0;
                for (int k = klo; k <= khi; ++k) {
                    for (int j = jlo; j <= jhi; ++j) {
                        for (int i = ilo; i <= ihi; ++i) {
                            const Real x = plo[0] + (static_cast<Real>(i)+offsets[comp][0])*dx[0];
                            const Real y = plo[1] + (static_cast<Real>(j)+offsets[comp][1])*dx[1];
                            const Real z = plo[2] + (static_cast<Real>(k)+offsets[comp][2])*dx[2];
                            const Real value = forces[comp]*
                                gaussian_kernel(x,y,z,px,py,pz,nx,ny,e1x,e1y,epsn,eps1,eps2)/
                                weight_volume;
                            if (comp == 0) { Gpu::Atomic::Add(&sx(i,j,k), value); }
                            if (comp == 1) { Gpu::Atomic::Add(&sy(i,j,k), value); }
                            if (comp == 2) { Gpu::Atomic::Add(&sz(i,j,k), value); }
                            integrated_force += value*cell_volume;

                            Real rx = x-xhub[turb[ip]];
                            Real ry = y-yhub[turb[ip]];
                            Real rz = z-zhub_les;
                            if (is_periodic[0]) {
                                if (rx >  0.5*prob_length[0]) { rx -= prob_length[0]; }
                                if (rx < -0.5*prob_length[0]) { rx += prob_length[0]; }
                            }
                            if (is_periodic[1]) {
                                if (ry >  0.5*prob_length[1]) { ry -= prob_length[1]; }
                                if (ry < -0.5*prob_length[1]) { ry += prob_length[1]; }
                            }
                            if (is_periodic[2]) {
                                if (rz >  0.5*prob_length[2]) { rz -= prob_length[2]; }
                                if (rz < -0.5*prob_length[2]) { rz += prob_length[2]; }
                            }
                            if (comp == 0) { integrated_torque += ny*rz*value*cell_volume; }
                            if (comp == 1) { integrated_torque -= nx*rz*value*cell_volume; }
                            if (comp == 2) {
                                integrated_torque += (nx*ry-ny*rx)*value*cell_volume;
                            }
                        }
                    }
                }
                Gpu::Atomic::Add(&deposited_force[comp], integrated_force);
                Gpu::Atomic::Add(&les_torque[turb[ip]], integrated_torque);
            }
        });
    }
    Gpu::streamSynchronize();
    xmom_src.SumBoundary(geom.periodicity());
    ymom_src.SumBoundary(geom.periodicity());
    zmom_src.SumBoundary(geom.periodicity());

    Vector<Real> deposited_force_h(3, 0.0);
    Gpu::copy(Gpu::deviceToHost, d_deposited_force.begin(), d_deposited_force.end(),
              deposited_force_h.begin());
    ParallelAllReduce::Sum(deposited_force_h.data(), deposited_force_h.size(),
                           ParallelContext::CommunicatorAll());
    Vector<Real> les_torque_h(nturb, 0.0);
    Gpu::copy(Gpu::deviceToHost, d_les_torque.begin(), d_les_torque.end(),
              les_torque_h.begin());
    ParallelAllReduce::Sum(les_torque_h.data(), les_torque_h.size(),
                           ParallelContext::CommunicatorAll());
    Real expected_force[3] = {0.0, 0.0, 0.0};
    Real expected_force_scale[3] = {0.0, 0.0, 0.0};
    for (int it = 0; it < nturb; ++it) {
        expected_force[0] += force_sum_h[3*it];
        expected_force[1] += force_sum_h[3*it+1];
        expected_force[2] += force_sum_h[3*it+2];
        expected_force_scale[0] += force_abs_sum_h[3*it];
        expected_force_scale[1] += force_abs_sum_h[3*it+1];
        expected_force_scale[2] += force_abs_sum_h[3*it+2];
    }
    for (int d = 0; d < 3; ++d) {
        // A rotating array can have a near-zero net transverse force even
        // though each disk carries a large tangential load.  Scale the
        // reduction comparison by the component's summed absolute loading so
        // cancellation does not turn this relative check into an unrealistically
        // tight absolute check on two independently ordered GPU reductions.
        const Real scale = amrex::max(expected_force_scale[d], 1.0);
        if (std::abs(deposited_force_h[d]-expected_force[d]) > 1.0e-10*scale) {
            Abort("ClassicAD Gaussian deposition failed discrete force conservation");
        }
    }
    for (int it = 0; it < nturb; ++it) {
        if (m_owner_levels[it] == lev) {
            state[it].les_torque = les_torque_h[it];
        }
    }

    m_model.write_diagnostics(time);
}

#endif
