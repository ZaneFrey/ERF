#include <ERF_ClassicADPC.H>

#if defined(ERF_USE_PARTICLES) && defined(ERF_USE_WINDFARM)

#include <ERF_IndexDefines.H>

#include <AMReX_GpuAtomic.H>
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
    const Real psi = (face_angle_deg - Real(90.0)) * classic_pi / Real(180.0);
    nx = std::cos(psi);
    ny = std::sin(psi);
    e1x = -ny;
    e1y = nx;
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

    const Real rotor_radius = m_model.rotor_radius();
    const int nlev = finestLevel() + 1;

    for (int lev = 0; lev < nlev; ++lev) {
        Gpu::HostVector<ParticleType> host_particles;
        std::array<Gpu::HostVector<ParticleReal>, ClassicADRealIdx::ncomps> host_real;
        std::array<Gpu::HostVector<int>, ClassicADIntIdx::ncomps> host_int;

        if (ParallelDescriptor::IOProcessor()) {
            const Real target_spacing = m_model.spacing_was_supplied()
                ? m_model.actuator_spacing()
                : Geom(lev).CellSize(2);
            const int nr = amrex::max(
                1, static_cast<int>(std::llround(rotor_radius / target_spacing)));
            const Real dr = rotor_radius / static_cast<Real>(nr);

            for (int turbine_id = 0; turbine_id < nturb; ++turbine_id) {
                if (owner_levels[turbine_id] != lev) {
                    continue;
                }

                Real nx;
                Real ny;
                Real e1x;
                Real e1y;
                rotor_basis(face_angles[turbine_id], nx, ny, e1x, e1y);
                const Real zhub = ground[turbine_id] + m_model.hub_height();
                int element_id = 0;

                for (int ring_id = 0; ring_id < nr; ++ring_id) {
                    const Real radius = (static_cast<Real>(ring_id) + Real(0.5)) * dr;
                    const int ntheta = amrex::max(
                        1, static_cast<int>(std::llround(
                            Real(2.0) * classic_pi * radius / target_spacing)));
                    const Real area = classic_pi *
                        static_cast<Real>((ring_id+1)*(ring_id+1) - ring_id*ring_id) *
                        dr*dr / static_cast<Real>(ntheta);

                    for (int azimuth_id = 0; azimuth_id < ntheta;
                         ++azimuth_id, ++element_id) {
                        const Real theta = Real(2.0) * classic_pi *
                            (static_cast<Real>(azimuth_id) + Real(0.5)) /
                            static_cast<Real>(ntheta);
                        const Real radial_in_plane = radius * std::cos(theta);
                        const Real radial_vertical = radius * std::sin(theta);

                        ParticleType particle;
                        particle.id() = ParticleType::NextID();
                        particle.cpu() = ParallelDescriptor::MyProc();
                        particle.pos(0) = xloc[turbine_id] + radial_in_plane*e1x;
                        particle.pos(1) = yloc[turbine_id] + radial_in_plane*e1y;
                        particle.pos(2) = zhub + radial_vertical;
                        host_particles.push_back(particle);

                        host_real[ClassicADRealIdx::radius].push_back(radius);
                        host_real[ClassicADRealIdx::theta].push_back(theta);
                        host_real[ClassicADRealIdx::area].push_back(area);
                        host_real[ClassicADRealIdx::force_x].push_back(Real(0.0));
                        host_real[ClassicADRealIdx::force_y].push_back(Real(0.0));
                        host_real[ClassicADRealIdx::force_z].push_back(Real(0.0));
                        host_int[ClassicADIntIdx::turbine].push_back(turbine_id);
                        host_int[ClassicADIntIdx::ring].push_back(ring_id);
                        host_int[ClassicADIntIdx::element].push_back(element_id);
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
            for (int comp = 0; comp < ClassicADRealIdx::ncomps; ++comp) {
                Gpu::copyAsync(Gpu::hostToDevice,
                               host_real[comp].begin(), host_real[comp].end(),
                               soa.GetRealData(comp).begin());
            }
            for (int comp = 0; comp < ClassicADIntIdx::ncomps; ++comp) {
                Gpu::copyAsync(Gpu::hostToDevice,
                               host_int[comp].begin(), host_int[comp].end(),
                               soa.GetIntData(comp).begin());
            }
            Gpu::streamSynchronize();
        }
        AddParticlesAtLevel(incoming, lev);
    }
}

void
ClassicADPC::sample_disk_state (int lev,
                                Real dt,
                                const MultiFab& cons,
                                const MultiFab& u,
                                const MultiFab& v,
                                const MultiFab& w)
{
    AMREX_ALWAYS_ASSERT(lev >= 0 && lev <= finestLevel());
    AMREX_ALWAYS_ASSERT(dt >= Real(0.0));

    const int nturb = static_cast<int>(m_model.x_locations().size());
    if (nturb == 0) {
        return;
    }

    AMREX_ALWAYS_ASSERT(static_cast<int>(m_owner_levels.size()) == nturb);
    auto& state = m_model.turbine_state();
    AMREX_ALWAYS_ASSERT(static_cast<int>(state.size()) == nturb);

    const Geometry& geom = Geom(lev);
    const auto plo = geom.ProbLoArray();
    const auto dxi = geom.InvCellSizeArray();

    Gpu::DeviceVector<Real> d_velocity_sum(nturb, Real(0.0));
    Gpu::DeviceVector<Real> d_density_sum(nturb, Real(0.0));
    Gpu::DeviceVector<Real> d_area_sum(nturb, Real(0.0));
    Real* velocity_sum_device = d_velocity_sum.data();
    Real* density_sum_device = d_density_sum.data();
    Real* area_sum_device = d_area_sum.data();

    const auto& face_angles = m_model.disk_face_angles_deg();
    AMREX_ALWAYS_ASSERT(static_cast<int>(face_angles.size()) == nturb);
    Gpu::DeviceVector<Real> d_face_angles(face_angles.size());
    Gpu::copyAsync(Gpu::hostToDevice,
                   face_angles.begin(), face_angles.end(),
                   d_face_angles.begin());
    const Real* face_angles_device = d_face_angles.data();

    for (ParIterType pti(*this, lev); pti.isValid(); ++pti) {
        const int grid = pti.index();
        auto& tile = ParticlesAt(lev, pti);
        auto& aos = tile.GetArrayOfStructs();
        auto& soa = tile.GetStructOfArrays();
        const int np = aos.numParticles();
        const auto* particles = aos().data();
        const auto* area = soa.GetRealData(ClassicADRealIdx::area).data();
        const auto* turbine = soa.GetIntData(ClassicADIntIdx::turbine).data();

        const GpuArray<Array4<const Real>, AMREX_SPACEDIM> velocity_arrays{{
            u[grid].const_array(),
            v[grid].const_array(),
            w[grid].const_array()}};
        const auto density_array = cons[grid].const_array(Rho_comp);

        ParallelFor(np, [=] AMREX_GPU_DEVICE (int ip) noexcept {
            ParticleReal velocity[AMREX_SPACEDIM];
            ParticleReal density[1];
            mac_interpolate(particles[ip], plo, dxi,
                            velocity_arrays, velocity);
            cic_interpolate(particles[ip], plo, dxi,
                            density_array, density, 1);

            Real nx;
            Real ny;
            Real e1x;
            Real e1y;
            rotor_basis(face_angles_device[turbine[ip]],
                        nx, ny, e1x, e1y);
            const Real disk_normal_velocity =
                velocity[0]*nx + velocity[1]*ny;

            Gpu::Atomic::Add(&velocity_sum_device[turbine[ip]],
                             disk_normal_velocity*area[ip]);
            Gpu::Atomic::Add(&density_sum_device[turbine[ip]],
                             density[0]*area[ip]);
            Gpu::Atomic::Add(&area_sum_device[turbine[ip]], area[ip]);
        });
    }
    Gpu::streamSynchronize();

    Vector<Real> velocity_sum(nturb);
    Vector<Real> density_sum(nturb);
    Vector<Real> area_sum(nturb);
    Gpu::copy(Gpu::deviceToHost,
              d_velocity_sum.begin(), d_velocity_sum.end(),
              velocity_sum.begin());
    Gpu::copy(Gpu::deviceToHost,
              d_density_sum.begin(), d_density_sum.end(),
              density_sum.begin());
    Gpu::copy(Gpu::deviceToHost,
              d_area_sum.begin(), d_area_sum.end(),
              area_sum.begin());

    ParallelAllReduce::Sum(velocity_sum.data(), velocity_sum.size(),
                           ParallelContext::CommunicatorAll());
    ParallelAllReduce::Sum(density_sum.data(), density_sum.size(),
                           ParallelContext::CommunicatorAll());
    ParallelAllReduce::Sum(area_sum.data(), area_sum.size(),
                           ParallelContext::CommunicatorAll());

    const Real rotor_area = classic_pi *
        m_model.rotor_radius() * m_model.rotor_radius();
    const Real minimum_area =
        std::numeric_limits<Real>::epsilon() * rotor_area;

    for (int turbine_id = 0; turbine_id < nturb; ++turbine_id) {
        if (m_owner_levels[turbine_id] != lev) {
            continue;
        }
        if (area_sum[turbine_id] <= minimum_area) {
            Abort("ClassicAD turbine has no locally/globally owned actuator area");
        }
        const Real relative_area_error =
            std::abs(area_sum[turbine_id] - rotor_area) / rotor_area;
        if (relative_area_error > Real(1.0e-11)) {
            Abort("ClassicAD actuator-element areas do not sum to the rotor area");
        }

        auto& turbine_state = state[turbine_id];
        turbine_state.actuator_area = area_sum[turbine_id];
        turbine_state.disk_velocity_raw =
            velocity_sum[turbine_id] / area_sum[turbine_id];
        turbine_state.disk_density =
            density_sum[turbine_id] / area_sum[turbine_id];

        if (!turbine_state.memory_initialized ||
            m_model.memory_time() == Real(0.0)) {
            turbine_state.disk_velocity_filtered =
                turbine_state.disk_velocity_raw;
            turbine_state.memory_initialized = true;
        } else {
            const Real alpha =
                Real(1.0) - std::exp(-dt / m_model.memory_time());
            turbine_state.disk_velocity_filtered += alpha *
                (turbine_state.disk_velocity_raw -
                 turbine_state.disk_velocity_filtered);
        }
    }
}

#endif
