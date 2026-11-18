#include <ERF_ClassicAD.H>

#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Utility.H>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
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

void advance_forcing_segment (Real& response, Real command,
                              Real duration, Real timescale,
                              Real* integral = nullptr)
{
    if (duration <= Real(0.0)) {
        return;
    }

    if (timescale == Real(0.0)) {
        response = command;
        if (integral != nullptr) {
            *integral += command * duration;
        }
        return;
    }

    const Real one_minus_decay = -std::expm1(-duration / timescale);
    if (integral != nullptr) {
        *integral += command * duration
            + (response - command) * timescale * one_minus_decay;
    }
    response = command
        + (response - command) * (Real(1.0) - one_minus_decay);
}

std::string schedule_error_prefix (const std::string& source_name,
                                   int source_line)
{
    std::ostringstream message;
    message << "ClassicAD forcing schedule " << source_name;
    if (source_line > 0) {
        message << ':' << source_line;
    }
    message << ": ";
    return message.str();
}

bool time_at_or_before (Real lhs, Real rhs)
{
    const Real scale = amrex::max(
        Real(1.0), amrex::max(std::abs(lhs), std::abs(rhs)));
    const Real tolerance = Real(32.0)
        * std::numeric_limits<Real>::epsilon() * scale;
    return lhs <= rhs + tolerance;
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
ClassicAD::configure_forcing_schedule (const std::string& filename,
                                       Real derate_timescale,
                                       Real windfarm_start_time)
{
    Vector<ClassicADForcingInterval> intervals;
    if (!filename.empty()) {
        Vector<char> file_chars;
        ParallelDescriptor::ReadAndBcastFile(filename, file_chars);
        std::istringstream input(
            std::string(file_chars.dataPtr()), std::istringstream::in);
        std::string line;
        int line_number = 0;
        while (std::getline(input, line)) {
            ++line_number;
            const auto comment = line.find('#');
            if (comment != std::string::npos) {
                line.erase(comment);
            }

            std::istringstream row(line);
            row >> std::ws;
            if (row.eof()) {
                continue;
            }

            ClassicADForcingInterval interval;
            interval.source_line = line_number;
            std::string extra;
            if (!(row >> interval.turbine_id
                      >> interval.start_time
                      >> interval.end_time
                      >> interval.forcing_fraction) ||
                (row >> extra)) {
                Abort(schedule_error_prefix(filename, line_number)
                      + "expected exactly four columns: turbine_id, start_time, "
                        "end_time, forcing_fraction");
            }
            intervals.push_back(interval);
        }
    }

    configure_forcing_schedule(intervals, derate_timescale,
                               windfarm_start_time,
                               filename.empty() ? "<none>" : filename);
}

void
ClassicAD::configure_forcing_schedule (
    const Vector<ClassicADForcingInterval>& intervals,
    Real derate_timescale,
    Real windfarm_start_time,
    const std::string& source_name)
{
    if (!std::isfinite(derate_timescale) || derate_timescale < Real(0.0)) {
        Abort("ClassicAD forcing derate timescale must be finite and >= 0");
    }

    const int nturb = static_cast<int>(m_turbine_state.size());
    m_derate_timescale = derate_timescale;
    m_windfarm_start_time = windfarm_start_time;
    m_forcing_schedule.assign(nturb, {});

    for (const auto& interval : intervals) {
        const auto prefix = schedule_error_prefix(source_name,
                                                  interval.source_line);
        if (interval.turbine_id < 0 || interval.turbine_id >= nturb) {
            Abort(prefix + "turbine_id " + std::to_string(interval.turbine_id)
                  + " is outside [0, " + std::to_string(nturb) + ")");
        }
        if (!std::isfinite(interval.start_time) ||
            !std::isfinite(interval.end_time) ||
            !std::isfinite(interval.forcing_fraction)) {
            Abort(prefix + "all schedule values must be finite");
        }
        if (interval.start_time < windfarm_start_time) {
            Abort(prefix + "start_time must be >= erf.windfarm_start_time");
        }
        if (interval.start_time >= interval.end_time) {
            Abort(prefix + "start_time must be less than end_time");
        }
        if (interval.forcing_fraction < Real(0.0) ||
            interval.forcing_fraction > Real(1.0)) {
            Abort(prefix + "forcing_fraction must be in [0, 1]");
        }
        m_forcing_schedule[interval.turbine_id].push_back(interval);
    }

    for (int turbine_id = 0; turbine_id < nturb; ++turbine_id) {
        auto& turbine_schedule = m_forcing_schedule[turbine_id];
        std::sort(turbine_schedule.begin(), turbine_schedule.end(),
                  [] (const auto& lhs, const auto& rhs) {
                      if (lhs.start_time != rhs.start_time) {
                          return lhs.start_time < rhs.start_time;
                      }
                      return lhs.end_time < rhs.end_time;
                  });
        for (int row = 1; row < static_cast<int>(turbine_schedule.size()); ++row) {
            const auto& previous = turbine_schedule[row-1];
            const auto& current = turbine_schedule[row];
            if (current.start_time < previous.end_time) {
                std::ostringstream message;
                message << "ClassicAD forcing schedule " << source_name
                        << ": overlapping periods for turbine " << turbine_id
                        << " on lines " << previous.source_line
                        << " and " << current.source_line;
                Abort(message.str());
            }
        }
    }
}

Real
ClassicAD::forcing_command_at (int turbine_id, Real time) const
{
    Real command = Real(1.0);
    for (const auto& interval : m_forcing_schedule[turbine_id]) {
        if (!time_at_or_before(interval.end_time, time)) {
            break;
        }
        command = interval.forcing_fraction;
    }
    return command;
}

Real
ClassicAD::advance_forcing_control (int turbine_id, Real time, Real dt,
                                    Real startup_factor)
{
    AMREX_ALWAYS_ASSERT(turbine_id >= 0 &&
                        turbine_id < static_cast<int>(m_turbine_state.size()));
    AMREX_ALWAYS_ASSERT(dt >= Real(0.0));
    AMREX_ALWAYS_ASSERT(startup_factor >= Real(0.0) &&
                        startup_factor <= Real(1.0));

    auto& state = m_turbine_state[turbine_id];
    if (!state.forcing_initialized) {
        state.forcing_response = Real(1.0);
        state.forcing_initialized = true;
    }

    Real response = state.forcing_response;
    Real command = forcing_command_at(turbine_id, time);
    Real segment_start = time;
    Real response_integral = Real(0.0);
    const Real step_end = time + dt;

    for (const auto& interval : m_forcing_schedule[turbine_id]) {
        const Real event_time = interval.end_time;
        if (time_at_or_before(event_time, time)) {
            continue;
        }
        if (!time_at_or_before(event_time, step_end)) {
            break;
        }
        const Real bounded_event_time = amrex::min(event_time, step_end);
        advance_forcing_segment(response, command,
                                bounded_event_time - segment_start,
                                m_derate_timescale, &response_integral);
        segment_start = bounded_event_time;
        command = interval.forcing_fraction;
        if (m_derate_timescale == Real(0.0)) {
            response = command;
        }
    }
    advance_forcing_segment(response, command, step_end - segment_start,
                            m_derate_timescale, &response_integral);

    state.forcing_command = command;
    state.forcing_response = amrex::max(
        Real(0.0), amrex::min(Real(1.0), response));
    state.forcing_response_step_average = (dt > Real(0.0))
        ? response_integral / dt : state.forcing_response;
    state.forcing_response_step_average = amrex::max(
        Real(0.0), amrex::min(Real(1.0),
                              state.forcing_response_step_average));
    state.forcing_factor = startup_factor
        * state.forcing_response_step_average;
    state.forcing_enabled = (state.forcing_command > Real(0.0));
    return state.forcing_factor;
}

Real
ClassicAD::reconstruct_forcing_response (int turbine_id, Real time) const
{
    Real response = Real(1.0);
    Real command = Real(1.0);
    Real segment_start = m_windfarm_start_time;
    if (time <= segment_start) {
        return response;
    }

    for (const auto& interval : m_forcing_schedule[turbine_id]) {
        if (!time_at_or_before(interval.end_time, time)) {
            break;
        }
        advance_forcing_segment(response, command,
                                interval.end_time - segment_start,
                                m_derate_timescale);
        segment_start = interval.end_time;
        command = interval.forcing_fraction;
        if (m_derate_timescale == Real(0.0)) {
            response = command;
        }
    }
    advance_forcing_segment(response, command, time - segment_start,
                            m_derate_timescale);
    return response;
}

void
ClassicAD::restore_uninitialized_forcing_state (Real time)
{
    for (int turbine_id = 0;
         turbine_id < static_cast<int>(m_turbine_state.size());
         ++turbine_id) {
        auto& state = m_turbine_state[turbine_id];
        if (state.forcing_initialized) {
            continue;
        }
        state.forcing_response = reconstruct_forcing_response(turbine_id, time);
        state.forcing_response_step_average = state.forcing_response;
        state.forcing_command = forcing_command_at(turbine_id, time);
        state.forcing_factor = state.forcing_response;
        state.forcing_enabled = (state.forcing_command > Real(0.0));
        state.forcing_initialized = true;
    }
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
ClassicAD::write_diagnostics (Real time) const
{
    if (!ParallelDescriptor::IOProcessor()) {
        return;
    }

    const std::string filename = "power_output_ClassicAD.txt";
    const bool write_header = !FileExists(filename);
    std::ofstream output(filename, std::ios::out | std::ios::app);
    if (!output.good()) {
        Abort("Failed to open ClassicAD diagnostics output");
    }
    output << std::setprecision(17);

    if (write_header) {
        output << "# time";
        for (int turbine_id = 0;
             turbine_id < static_cast<int>(m_turbine_state.size());
             ++turbine_id) {
            output << " Ud_raw_" << turbine_id
                   << " Ud_f_" << turbine_id
                   << " rho_d_" << turbine_id
                   << " forcing_factor_" << turbine_id
                   << " forcing_enabled_" << turbine_id
                   << " T_target_" << turbine_id
                   << " T_applied_" << turbine_id
                   << " P_axial_" << turbine_id;
            if (m_wake_rotation) {
                output << " P_rotor_" << turbine_id
                       << " Omega_" << turbine_id
                       << " Q_target_" << turbine_id
                       << " Q_applied_" << turbine_id
                       << " Q_LES_" << turbine_id;
            }
        }
        output << '\n';
    }

    output << time;
    for (const auto& state : m_turbine_state) {
        output << ' ' << state.disk_velocity_raw
               << ' ' << state.disk_velocity_filtered
               << ' ' << state.disk_density
               << ' ' << state.forcing_factor
               << ' ' << static_cast<int>(state.forcing_enabled)
               << ' ' << state.thrust_target
               << ' ' << state.thrust_applied
               << ' ' << state.axial_power;
        if (m_wake_rotation) {
            output << ' ' << state.rotor_power
                   << ' ' << state.omega
                   << ' ' << state.torque_target
                   << ' ' << state.torque_applied
                   << ' ' << state.les_torque;
        }
    }
    output << '\n';
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
    out << "ClassicADMemoryState_v2\n" << m_turbine_state.size() << '\n';
    for (const auto& state : m_turbine_state) {
        out << static_cast<int>(state.memory_initialized) << ' '
            << state.disk_velocity_filtered << ' '
            << static_cast<int>(state.forcing_initialized) << ' '
            << state.forcing_response << '\n';
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
            state.forcing_initialized = false;
            state.forcing_response = 1.0;
        }
        return false;
    }

    Vector<char> file_chars;
    ParallelDescriptor::ReadAndBcastFile(filename, file_chars);
    std::istringstream input(std::string(file_chars.dataPtr()), std::istringstream::in);

    std::string tag;
    int nturb = 0;
    input >> tag >> nturb;
    const bool version_1 = (tag == "ClassicADMemoryState_v1");
    const bool version_2 = (tag == "ClassicADMemoryState_v2");
    if (!version_1 && !version_2) {
        Abort("Unknown ClassicADMemoryState format");
    }
    if (nturb != static_cast<int>(m_turbine_state.size())) {
        Abort("ClassicADMemoryState turbine count does not match windfarm_loc_table");
    }

    for (auto& state : m_turbine_state) {
        int initialized = 0;
        if (!(input >> initialized >> state.disk_velocity_filtered)) {
            Abort("Failed while reading ClassicADMemoryState");
        }
        state.memory_initialized = (initialized != 0);
        if (version_2) {
            int forcing_initialized = 0;
            if (!(input >> forcing_initialized >> state.forcing_response)) {
                Abort("Failed while reading ClassicADMemoryState");
            }
            if (!std::isfinite(state.forcing_response) ||
                state.forcing_response < Real(0.0) ||
                state.forcing_response > Real(1.0)) {
                Abort("ClassicADMemoryState forcing response must be in [0, 1]");
            }
            state.forcing_initialized = (forcing_initialized != 0);
            state.forcing_response_step_average = state.forcing_response;
        } else {
            state.forcing_initialized = false;
            state.forcing_response = Real(1.0);
            state.forcing_response_step_average = Real(1.0);
        }
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
