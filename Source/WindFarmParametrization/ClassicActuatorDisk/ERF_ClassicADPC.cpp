#include <ERF_ClassicADPC.H>

#if defined(ERF_USE_PARTICLES) && defined(ERF_USE_WINDFARM)

#include <AMReX_ParallelDescriptor.H>

#include <array>
#include <cmath>

using namespace amrex;

namespace {

constexpr Real classic_pi = 3.141592653589793238462643383279502884;

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void rotor_basis (Real face_angle_deg, Real& e1x, Real& e1y) noexcept
{
    const Real psi = (face_angle_deg - Real(90.0)) * classic_pi / Real(180.0);
    const Real nx = std::cos(psi);
    const Real ny = std::sin(psi);
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

                Real e1x;
                Real e1y;
                rotor_basis(face_angles[turbine_id], e1x, e1y);
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

#endif
