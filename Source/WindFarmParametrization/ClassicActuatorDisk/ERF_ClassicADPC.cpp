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

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real gaussian_kernel (Real x, Real y, Real z,
                      Real particle_x, Real particle_y, Real particle_z,
                      Real nx, Real ny, Real e1x, Real e1y,
                      Real epsilon_normal,
                      Real epsilon_in_plane,
                      Real epsilon_vertical) noexcept
{
    const Real dx = x - particle_x;
    const Real dy = y - particle_y;
    const Real dz = z - particle_z;
    const Real normal_distance = dx*nx + dy*ny;
    const Real in_plane_distance = dx*e1x + dy*e1y;

    if (std::abs(normal_distance) > Real(3.0)*epsilon_normal ||
        std::abs(in_plane_distance) > Real(3.0)*epsilon_in_plane ||
        std::abs(dz) > Real(3.0)*epsilon_vertical) {
        return Real(0.0);
    }

    return std::exp(-Real(0.5) *
        (normal_distance*normal_distance /
             (epsilon_normal*epsilon_normal) +
         in_plane_distance*in_plane_distance /
             (epsilon_in_plane*epsilon_in_plane) +
         dz*dz / (epsilon_vertical*epsilon_vertical)));
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
ClassicADPC::update_positions ()
{
    const int nturb = static_cast<int>(m_model.x_locations().size());
    AMREX_ALWAYS_ASSERT(
        static_cast<int>(m_model.y_locations().size()) == nturb);
    AMREX_ALWAYS_ASSERT(
        static_cast<int>(m_model.ground_elevations().size()) == nturb);
    AMREX_ALWAYS_ASSERT(
        static_cast<int>(m_model.disk_face_angles_deg().size()) == nturb);

    Gpu::DeviceVector<Real> d_xloc(nturb);
    Gpu::DeviceVector<Real> d_yloc(nturb);
    Gpu::DeviceVector<Real> d_ground(nturb);
    Gpu::DeviceVector<Real> d_face_angles(nturb);
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
                   m_model.disk_face_angles_deg().begin(),
                   m_model.disk_face_angles_deg().end(),
                   d_face_angles.begin());
    const Real* xloc = d_xloc.data();
    const Real* yloc = d_yloc.data();
    const Real* ground = d_ground.data();
    const Real* face_angles = d_face_angles.data();
    const Real hub_height = m_model.hub_height();

    for (int lev = 0; lev <= finestLevel(); ++lev) {
        for (ParIterType pti(*this, lev); pti.isValid(); ++pti) {
            auto& tile = ParticlesAt(lev, pti);
            auto& aos = tile.GetArrayOfStructs();
            auto& soa = tile.GetStructOfArrays();
            const int np = aos.numParticles();
            auto* particles = aos().data();
            const auto* radius =
                soa.GetRealData(ClassicADRealIdx::radius).data();
            const auto* theta =
                soa.GetRealData(ClassicADRealIdx::theta).data();
            const auto* turbine =
                soa.GetIntData(ClassicADIntIdx::turbine).data();

            ParallelFor(np, [=] AMREX_GPU_DEVICE (int ip) noexcept {
                const int turbine_id = turbine[ip];
                Real nx;
                Real ny;
                Real e1x;
                Real e1y;
                rotor_basis(face_angles[turbine_id],
                            nx, ny, e1x, e1y);
                const Real radial_in_plane =
                    radius[ip]*std::cos(theta[ip]);
                particles[ip].pos(0) =
                    xloc[turbine_id] + radial_in_plane*e1x;
                particles[ip].pos(1) =
                    yloc[turbine_id] + radial_in_plane*e1y;
                particles[ip].pos(2) = ground[turbine_id] + hub_height +
                    radius[ip]*std::sin(theta[ip]);
            });
        }
    }
    Gpu::streamSynchronize();
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

void
ClassicADPC::update_axial_forces (int lev, Real load_factor)
{
    AMREX_ALWAYS_ASSERT(lev >= 0 && lev <= finestLevel());
    AMREX_ALWAYS_ASSERT(load_factor >= Real(0.0) &&
                        load_factor <= Real(1.0));

    const int nturb = static_cast<int>(m_model.x_locations().size());
    if (nturb == 0) {
        return;
    }

    AMREX_ALWAYS_ASSERT(static_cast<int>(m_owner_levels.size()) == nturb);
    auto& state = m_model.turbine_state();
    AMREX_ALWAYS_ASSERT(static_cast<int>(state.size()) == nturb);

    const Real rotor_area = classic_pi *
        m_model.rotor_radius() * m_model.rotor_radius();
    const Real minimum_area =
        std::numeric_limits<Real>::epsilon() * rotor_area;
    Vector<Real> thrust_applied(nturb, Real(0.0));
    Vector<Real> area_normalization(nturb, rotor_area);

    for (int turbine_id = 0; turbine_id < nturb; ++turbine_id) {
        if (m_owner_levels[turbine_id] != lev) {
            continue;
        }

        auto& turbine_state = state[turbine_id];
        AMREX_ALWAYS_ASSERT(turbine_state.memory_initialized);
        AMREX_ALWAYS_ASSERT(turbine_state.actuator_area > minimum_area);

        const Real disk_velocity =
            turbine_state.disk_velocity_filtered;
        turbine_state.thrust_target = Real(0.5) *
            turbine_state.disk_density * rotor_area *
            m_model.ctprime() * disk_velocity * disk_velocity;
        turbine_state.thrust_applied =
            load_factor * turbine_state.thrust_target;
        turbine_state.axial_power =
            turbine_state.thrust_target * disk_velocity;
        turbine_state.rotor_power = Real(0.0);
        turbine_state.omega = Real(0.0);
        turbine_state.torque_target = Real(0.0);
        turbine_state.torque_applied = Real(0.0);

        if (m_model.wake_rotation_enabled() &&
            std::abs(disk_velocity) > Real(1.0e-12)) {
            turbine_state.omega =
                m_model.tsr() * disk_velocity / m_model.rotor_radius();
            turbine_state.rotor_power = Real(0.5) *
                turbine_state.disk_density * rotor_area *
                m_model.cpprime() * disk_velocity * disk_velocity *
                disk_velocity;
            turbine_state.torque_target = Real(0.5) *
                turbine_state.disk_density * rotor_area *
                m_model.cpprime() * disk_velocity * disk_velocity *
                m_model.rotor_radius() / m_model.tsr();
            turbine_state.torque_applied =
                load_factor * turbine_state.torque_target;
        }

        thrust_applied[turbine_id] = turbine_state.thrust_applied;
        area_normalization[turbine_id] = turbine_state.actuator_area;
    }

    const auto& face_angles = m_model.disk_face_angles_deg();
    AMREX_ALWAYS_ASSERT(static_cast<int>(face_angles.size()) == nturb);
    Gpu::DeviceVector<Real> d_thrust_applied(nturb);
    Gpu::DeviceVector<Real> d_area_normalization(nturb);
    Gpu::DeviceVector<Real> d_face_angles(nturb);
    Gpu::copyAsync(Gpu::hostToDevice,
                   thrust_applied.begin(), thrust_applied.end(),
                   d_thrust_applied.begin());
    Gpu::copyAsync(Gpu::hostToDevice,
                   area_normalization.begin(), area_normalization.end(),
                   d_area_normalization.begin());
    Gpu::copyAsync(Gpu::hostToDevice,
                   face_angles.begin(), face_angles.end(),
                   d_face_angles.begin());

    const Real* thrust_applied_device = d_thrust_applied.data();
    const Real* area_normalization_device = d_area_normalization.data();
    const Real* face_angles_device = d_face_angles.data();

    for (ParIterType pti(*this, lev); pti.isValid(); ++pti) {
        auto& tile = ParticlesAt(lev, pti);
        auto& aos = tile.GetArrayOfStructs();
        auto& soa = tile.GetStructOfArrays();
        const int np = aos.numParticles();
        const auto* area = soa.GetRealData(ClassicADRealIdx::area).data();
        auto* force_x = soa.GetRealData(ClassicADRealIdx::force_x).data();
        auto* force_y = soa.GetRealData(ClassicADRealIdx::force_y).data();
        auto* force_z = soa.GetRealData(ClassicADRealIdx::force_z).data();
        const auto* turbine =
            soa.GetIntData(ClassicADIntIdx::turbine).data();

        ParallelFor(np, [=] AMREX_GPU_DEVICE (int ip) noexcept {
            const int turbine_id = turbine[ip];
            Real nx;
            Real ny;
            Real e1x;
            Real e1y;
            rotor_basis(face_angles_device[turbine_id],
                        nx, ny, e1x, e1y);

            const Real axial_force =
                -thrust_applied_device[turbine_id] * area[ip] /
                area_normalization_device[turbine_id];
            force_x[ip] = axial_force * nx;
            force_y[ip] = axial_force * ny;
            force_z[ip] = Real(0.0);
        });
    }
    Gpu::streamSynchronize();
}

void
ClassicADPC::deposit_forces (int lev,
                             MultiFab& xmom_src,
                             MultiFab& ymom_src,
                             MultiFab& zmom_src)
{
    AMREX_ALWAYS_ASSERT(lev >= 0 && lev <= finestLevel());
    AMREX_ALWAYS_ASSERT(xmom_src.ixType() == IndexType(IntVect(1,0,0)));
    AMREX_ALWAYS_ASSERT(ymom_src.ixType() == IndexType(IntVect(0,1,0)));
    AMREX_ALWAYS_ASSERT(zmom_src.ixType() == IndexType(IntVect(0,0,1)));

    xmom_src.setVal(Real(0.0));
    ymom_src.setVal(Real(0.0));
    zmom_src.setVal(Real(0.0));

    const int nturb = static_cast<int>(m_model.x_locations().size());
    if (nturb == 0) {
        return;
    }

    const Geometry& geom = Geom(lev);
    const auto prob_lo = geom.ProbLoArray();
    const auto inverse_cell_size = geom.InvCellSizeArray();
    const auto cell_size = geom.CellSizeArray();
    const Real cell_volume =
        cell_size[0] * cell_size[1] * cell_size[2];

    const auto& face_angles = m_model.disk_face_angles_deg();
    AMREX_ALWAYS_ASSERT(static_cast<int>(face_angles.size()) == nturb);
    Gpu::DeviceVector<Real> d_face_angles(nturb);
    Gpu::copyAsync(Gpu::hostToDevice,
                   face_angles.begin(), face_angles.end(),
                   d_face_angles.begin());
    const Real* face_angles_device = d_face_angles.data();

    Gpu::DeviceVector<Real> d_expected_force(3, Real(0.0));
    Gpu::DeviceVector<Real> d_expected_force_abs(3, Real(0.0));
    Gpu::DeviceVector<Real> d_deposited_force(3, Real(0.0));
    Real* expected_force_device = d_expected_force.data();
    Real* expected_force_abs_device = d_expected_force_abs.data();
    Real* deposited_force_device = d_deposited_force.data();

    for (ParIterType pti(*this, lev); pti.isValid(); ++pti) {
        const int grid = pti.index();
        auto& tile = ParticlesAt(lev, pti);
        auto& aos = tile.GetArrayOfStructs();
        auto& soa = tile.GetStructOfArrays();
        const int np = aos.numParticles();
        const auto* particles = aos().data();
        const auto* force_x =
            soa.GetRealData(ClassicADRealIdx::force_x).data();
        const auto* force_y =
            soa.GetRealData(ClassicADRealIdx::force_y).data();
        const auto* force_z =
            soa.GetRealData(ClassicADRealIdx::force_z).data();
        const auto* turbine =
            soa.GetIntData(ClassicADIntIdx::turbine).data();

        auto x_source = xmom_src[grid].array();
        auto y_source = ymom_src[grid].array();
        auto z_source = zmom_src[grid].array();
        const Box x_box = xmom_src[grid].box();
        const Box y_box = ymom_src[grid].box();
        const Box z_box = zmom_src[grid].box();

        ParallelFor(np, [=] AMREX_GPU_DEVICE (int ip) noexcept {
            Real nx;
            Real ny;
            Real e1x;
            Real e1y;
            rotor_basis(face_angles_device[turbine[ip]],
                        nx, ny, e1x, e1y);

            const Real epsilon_normal = std::sqrt(
                (nx*cell_size[0])*(nx*cell_size[0]) +
                (ny*cell_size[1])*(ny*cell_size[1]));
            const Real epsilon_in_plane = std::sqrt(
                (e1x*cell_size[0])*(e1x*cell_size[0]) +
                (e1y*cell_size[1])*(e1y*cell_size[1]));
            const Real epsilon_vertical = cell_size[2];
            const Real support_x = Real(3.0) *
                (std::abs(nx)*epsilon_normal +
                 std::abs(e1x)*epsilon_in_plane);
            const Real support_y = Real(3.0) *
                (std::abs(ny)*epsilon_normal +
                 std::abs(e1y)*epsilon_in_plane);
            const Real support_z = Real(3.0) * epsilon_vertical;
            const Real particle_x = particles[ip].pos(0);
            const Real particle_y = particles[ip].pos(1);
            const Real particle_z = particles[ip].pos(2);

            const Real offsets[3][3] = {
                {Real(0.0), Real(0.5), Real(0.5)},
                {Real(0.5), Real(0.0), Real(0.5)},
                {Real(0.5), Real(0.5), Real(0.0)}};
            const Box source_boxes[3] = {x_box, y_box, z_box};
            const Real forces[3] = {
                force_x[ip], force_y[ip], force_z[ip]};

            for (int component = 0; component < 3; ++component) {
                Gpu::Atomic::Add(&expected_force_device[component],
                                 forces[component]);
                Gpu::Atomic::Add(&expected_force_abs_device[component],
                                 std::abs(forces[component]));
                if (forces[component] == Real(0.0)) {
                    continue;
                }

                const Box& source_box = source_boxes[component];
                int ilo = static_cast<int>(std::ceil(
                    (particle_x-support_x-prob_lo[0])*
                    inverse_cell_size[0] - offsets[component][0]));
                int ihi = static_cast<int>(std::floor(
                    (particle_x+support_x-prob_lo[0])*
                    inverse_cell_size[0] - offsets[component][0]));
                int jlo = static_cast<int>(std::ceil(
                    (particle_y-support_y-prob_lo[1])*
                    inverse_cell_size[1] - offsets[component][1]));
                int jhi = static_cast<int>(std::floor(
                    (particle_y+support_y-prob_lo[1])*
                    inverse_cell_size[1] - offsets[component][1]));
                int klo = static_cast<int>(std::ceil(
                    (particle_z-support_z-prob_lo[2])*
                    inverse_cell_size[2] - offsets[component][2]));
                int khi = static_cast<int>(std::floor(
                    (particle_z+support_z-prob_lo[2])*
                    inverse_cell_size[2] - offsets[component][2]));
                ilo = amrex::max(ilo, source_box.smallEnd(0));
                ihi = amrex::min(ihi, source_box.bigEnd(0));
                jlo = amrex::max(jlo, source_box.smallEnd(1));
                jhi = amrex::min(jhi, source_box.bigEnd(1));
                klo = amrex::max(klo, source_box.smallEnd(2));
                khi = amrex::min(khi, source_box.bigEnd(2));

                Real weight_volume = Real(0.0);
                for (int k = klo; k <= khi; ++k) {
                    for (int j = jlo; j <= jhi; ++j) {
                        for (int i = ilo; i <= ihi; ++i) {
                            const Real x = prob_lo[0] +
                                (static_cast<Real>(i) +
                                 offsets[component][0]) * cell_size[0];
                            const Real y = prob_lo[1] +
                                (static_cast<Real>(j) +
                                 offsets[component][1]) * cell_size[1];
                            const Real z = prob_lo[2] +
                                (static_cast<Real>(k) +
                                 offsets[component][2]) * cell_size[2];
                            weight_volume += gaussian_kernel(
                                x, y, z,
                                particle_x, particle_y, particle_z,
                                nx, ny, e1x, e1y,
                                epsilon_normal,
                                epsilon_in_plane,
                                epsilon_vertical) * cell_volume;
                        }
                    }
                }
                if (weight_volume <= Real(0.0)) {
                    continue;
                }

                Real integrated_force = Real(0.0);
                for (int k = klo; k <= khi; ++k) {
                    for (int j = jlo; j <= jhi; ++j) {
                        for (int i = ilo; i <= ihi; ++i) {
                            const Real x = prob_lo[0] +
                                (static_cast<Real>(i) +
                                 offsets[component][0]) * cell_size[0];
                            const Real y = prob_lo[1] +
                                (static_cast<Real>(j) +
                                 offsets[component][1]) * cell_size[1];
                            const Real z = prob_lo[2] +
                                (static_cast<Real>(k) +
                                 offsets[component][2]) * cell_size[2];
                            const Real value = forces[component] *
                                gaussian_kernel(
                                    x, y, z,
                                    particle_x, particle_y, particle_z,
                                    nx, ny, e1x, e1y,
                                    epsilon_normal,
                                    epsilon_in_plane,
                                    epsilon_vertical) / weight_volume;

                            if (component == 0) {
                                Gpu::Atomic::Add(&x_source(i,j,k), value);
                            } else if (component == 1) {
                                Gpu::Atomic::Add(&y_source(i,j,k), value);
                            } else {
                                Gpu::Atomic::Add(&z_source(i,j,k), value);
                            }
                            integrated_force += value * cell_volume;
                        }
                    }
                }
                Gpu::Atomic::Add(&deposited_force_device[component],
                                 integrated_force);
            }
        });
    }
    Gpu::streamSynchronize();

    xmom_src.SumBoundary(geom.periodicity());
    ymom_src.SumBoundary(geom.periodicity());
    zmom_src.SumBoundary(geom.periodicity());

    Vector<Real> expected_force(3);
    Vector<Real> expected_force_abs(3);
    Vector<Real> deposited_force(3);
    Gpu::copy(Gpu::deviceToHost,
              d_expected_force.begin(), d_expected_force.end(),
              expected_force.begin());
    Gpu::copy(Gpu::deviceToHost,
              d_expected_force_abs.begin(), d_expected_force_abs.end(),
              expected_force_abs.begin());
    Gpu::copy(Gpu::deviceToHost,
              d_deposited_force.begin(), d_deposited_force.end(),
              deposited_force.begin());
    ParallelAllReduce::Sum(expected_force.data(), expected_force.size(),
                           ParallelContext::CommunicatorAll());
    ParallelAllReduce::Sum(expected_force_abs.data(), expected_force_abs.size(),
                           ParallelContext::CommunicatorAll());
    ParallelAllReduce::Sum(deposited_force.data(), deposited_force.size(),
                           ParallelContext::CommunicatorAll());

    for (int component = 0; component < 3; ++component) {
        const Real force_scale =
            amrex::max(expected_force_abs[component], Real(1.0));
        if (std::abs(deposited_force[component] -
                     expected_force[component]) >
            Real(1.0e-10) * force_scale) {
            Abort("ClassicAD Gaussian deposition failed discrete force conservation");
        }
    }
}

#endif
