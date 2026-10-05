#include <ERF_ClassicAD.H>

#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Utility.H>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <utility>

using namespace amrex;

namespace {

constexpr Real classic_pi = 3.141592653589793238462643383279502884;

Real wrap_pi (Real angle)
{
    while (angle <= -classic_pi) {
        angle += Real(2.0)*classic_pi;
    }
    while (angle > classic_pi) {
        angle -= Real(2.0)*classic_pi;
    }
    return angle;
}

Real wrap_360 (Real angle)
{
    angle = std::fmod(angle, Real(360.0));
    if (angle < Real(0.0)) {
        angle += Real(360.0);
    }
    return angle;
}

} // namespace

void
ClassicAD::configure (Real ctprime, Real cpprime, Real tsr,
                      bool wake_rotation, Real memory_time,
                      Real diameter, Real hub_height, Real spacing,
                      bool spacing_was_supplied)
{
    m_ctprime = ctprime;
    m_cpprime = cpprime;
    m_tsr = tsr;
    m_wake_rotation = wake_rotation;
    m_memory_time = memory_time;
    m_diameter = diameter;
    m_hub_height_classic = hub_height;
    m_actuator_spacing = spacing;
    m_spacing_was_supplied = spacing_was_supplied;

    set_turb_spec(0.5 * diameter, hub_height, 0.0, {}, {}, {});
}

void
ClassicAD::initialize_turbine_state (int nturb)
{
    if (static_cast<int>(m_turbine_state.size()) == nturb) { return; }

    Vector<ClassicADTurbineState> resized(nturb);
    const int ncopy = amrex::min(nturb, static_cast<int>(m_turbine_state.size()));
    for (int it = 0; it < ncopy; ++it) {
        resized[it] = m_turbine_state[it];
    }
    m_turbine_state = std::move(resized);
}

void
ClassicAD::initialize_dynamic_yaw (Real disk_face_angle_deg,
                                   Real sensor_distance_by_D,
                                   Real sensor_memory_time,
                                   Real windfarm_start_time)
{
    const int nturb = static_cast<int>(m_xloc.size());
    m_dynamic_yaw_enabled = true;
    m_yaw_sensor_distance_by_D = sensor_distance_by_D;
    m_yaw_sensor_memory_time = sensor_memory_time;
    m_next_yaw_output_time = windfarm_start_time;
    m_yaw_active_step_count = 0;
    m_yaw_state.assign(nturb, {});

    const Real initial_angle = wrap_pi(
        (disk_face_angle_deg - Real(90.0)) *
        classic_pi / Real(180.0));
    for (auto& state : m_yaw_state) {
        state.psi_ref = initial_angle;
        state.psi_rotor = initial_angle;
        state.phi_cmd = initial_angle;
    }
    synchronize_rotor_angles();
}

void
ClassicAD::synchronize_rotor_angles ()
{
    m_turb_disk_angles.resize(m_yaw_state.size());
    for (int turbine_id = 0;
         turbine_id < static_cast<int>(m_yaw_state.size());
        ++turbine_id) {
        auto& state = m_yaw_state[turbine_id];
        state.psi_rotor = wrap_pi(state.psi_ref);
        m_turb_disk_angles[turbine_id] = wrap_360(
            Real(90.0) +
            state.psi_rotor * Real(180.0) / classic_pi);
    }
}

void
ClassicAD::update_dynamic_yaw (const Vector<Real>& sensor_u_raw,
                               const Vector<Real>& sensor_v_raw,
                               Real dt)
{
    if (!m_dynamic_yaw_enabled) {
        return;
    }
    AMREX_ALWAYS_ASSERT(dt >= Real(0.0));

    const int nturb = static_cast<int>(m_yaw_state.size());
    if (static_cast<int>(sensor_u_raw.size()) != nturb ||
        static_cast<int>(sensor_v_raw.size()) != nturb) {
        Abort("ClassicAD dynamic-yaw sensor sample count does not match turbine count");
    }

    const Real alpha = (m_yaw_sensor_memory_time == Real(0.0))
        ? Real(1.0)
        : Real(1.0) - std::exp(-dt / m_yaw_sensor_memory_time);

    for (int turbine_id = 0; turbine_id < nturb; ++turbine_id) {
        auto& state = m_yaw_state[turbine_id];
        state.sensor_u_raw = sensor_u_raw[turbine_id];
        state.sensor_v_raw = sensor_v_raw[turbine_id];

        if (!state.sensor_initialized ||
            m_yaw_sensor_memory_time == Real(0.0)) {
            state.sensor_u_filtered = state.sensor_u_raw;
            state.sensor_v_filtered = state.sensor_v_raw;
            state.sensor_initialized = true;
        } else {
            state.sensor_u_filtered += alpha *
                (state.sensor_u_raw - state.sensor_u_filtered);
            state.sensor_v_filtered += alpha *
                (state.sensor_v_raw - state.sensor_v_filtered);
        }

        state.phi_cmd = std::atan2(
            state.sensor_v_filtered, state.sensor_u_filtered);
        const Real yaw_error = wrap_pi(
            state.phi_cmd - state.psi_ref);
        state.psi_ref = wrap_pi(state.psi_ref + alpha*yaw_error);
    }
    ++m_yaw_active_step_count;
    synchronize_rotor_angles();
}

void
ClassicAD::write_memory_state (const std::string& checkpointname) const
{
    if (!ParallelDescriptor::IOProcessor()) { return; }

    std::ofstream out(checkpointname + "/ClassicADMemoryState",
                      std::ios::out | std::ios::trunc);
    if (!out.good()) {
        Abort("Failed to open ClassicADMemoryState for checkpoint write");
    }

    out << std::setprecision(17);
    out << "ClassicADMemoryState_v1\n" << m_turbine_state.size() << '\n';
    for (const auto& state : m_turbine_state) {
        out << static_cast<int>(state.memory_initialized) << ' '
            << state.disk_velocity_filtered << '\n';
    }
}

bool
ClassicAD::read_memory_state (const std::string& restart_chkfile)
{
    const std::string filename = restart_chkfile + "/ClassicADMemoryState";
    if (!FileExists(filename)) {
        for (auto& state : m_turbine_state) {
            state.memory_initialized = false;
            state.disk_velocity_filtered = 0.0;
        }
        return false;
    }

    Vector<char> file_chars;
    ParallelDescriptor::ReadAndBcastFile(filename, file_chars);
    std::istringstream input(std::string(file_chars.dataPtr()), std::istringstream::in);

    std::string tag;
    int nturb = 0;
    input >> tag >> nturb;
    if (tag != "ClassicADMemoryState_v1") {
        Abort("Unknown ClassicADMemoryState format");
    }
    if (nturb != static_cast<int>(m_turbine_state.size())) {
        Abort("ClassicADMemoryState turbine count does not match windfarm_loc_table");
    }

    for (auto& state : m_turbine_state) {
        int initialized = 0;
        input >> initialized >> state.disk_velocity_filtered;
        state.memory_initialized = (initialized != 0);
    }
    if (!input.good() && !input.eof()) {
        Abort("Failed while reading ClassicADMemoryState");
    }
    return true;
}

void
ClassicAD::write_yaw_state (const std::string& checkpointname) const
{
    if (!m_dynamic_yaw_enabled || !ParallelDescriptor::IOProcessor()) {
        return;
    }

    std::ofstream out(checkpointname + "/ClassicADYawState",
                      std::ios::out | std::ios::trunc);
    if (!out.good()) {
        Abort("Failed to open ClassicADYawState for checkpoint write");
    }

    out << std::setprecision(17);
    out << "ClassicADYawState_v1\n" << m_yaw_state.size() << '\n';
    out << m_next_yaw_output_time << ' '
        << m_yaw_active_step_count << '\n';
    for (const auto& state : m_yaw_state) {
        out << state.psi_ref << ' '
            << state.psi_rotor << ' '
            << state.phi_cmd << ' '
            << state.sensor_u_raw << ' '
            << state.sensor_v_raw << ' '
            << state.sensor_u_filtered << ' '
            << state.sensor_v_filtered << ' '
            << static_cast<int>(state.sensor_initialized) << '\n';
    }
}

bool
ClassicAD::read_yaw_state (const std::string& restart_chkfile)
{
    if (!m_dynamic_yaw_enabled) {
        return false;
    }

    const std::string filename = restart_chkfile + "/ClassicADYawState";
    if (!FileExists(filename)) {
        return false;
    }

    Vector<char> file_chars;
    ParallelDescriptor::ReadAndBcastFile(filename, file_chars);
    std::istringstream input(
        std::string(file_chars.dataPtr()), std::istringstream::in);

    std::string tag;
    int nturb = 0;
    input >> tag >> nturb;
    if (tag != "ClassicADYawState_v1") {
        Abort("Unknown ClassicADYawState format");
    }
    if (nturb != static_cast<int>(m_yaw_state.size())) {
        Abort("ClassicADYawState turbine count does not match windfarm_loc_table");
    }

    input >> m_next_yaw_output_time >> m_yaw_active_step_count;
    for (auto& state : m_yaw_state) {
        int initialized = 0;
        input >> state.psi_ref
              >> state.psi_rotor
              >> state.phi_cmd
              >> state.sensor_u_raw
              >> state.sensor_v_raw
              >> state.sensor_u_filtered
              >> state.sensor_v_filtered
              >> initialized;
        state.sensor_initialized = (initialized != 0);
    }
    if (!input.good() && !input.eof()) {
        Abort("Failed while reading ClassicADYawState");
    }

    synchronize_rotor_angles();
    return true;
}
