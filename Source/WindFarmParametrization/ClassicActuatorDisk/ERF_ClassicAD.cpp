#include <ERF_ClassicAD.H>

#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Utility.H>

#include <fstream>
#include <iomanip>
#include <sstream>
#include <cmath>

using namespace amrex;

namespace {

constexpr Real classic_pi = 3.141592653589793238462643383279502884;

Real wrap_pi (Real angle)
{
    while (angle <= -classic_pi) { angle += 2.0*classic_pi; }
    while (angle > classic_pi) { angle -= 2.0*classic_pi; }
    return angle;
}

Real wrap_360 (Real angle)
{
    angle = std::fmod(angle, 360.0);
    if (angle < 0.0) { angle += 360.0; }
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
    m_tsr_classic = tsr;
    m_wake_rotation_classic = wake_rotation;
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
    for (int it = 0; it < ncopy; ++it) { resized[it] = m_turbine_state[it]; }
    m_turbine_state = std::move(resized);
}

void
ClassicAD::write_diagnostics (Real time) const
{
    if (!ParallelDescriptor::IOProcessor()) { return; }

    const std::string filename = "power_output_ClassicAD.txt";
    const bool need_header = !FileExists(filename);
    std::ofstream out(filename, std::ios::out | std::ios::app);
    if (!out.good()) { Abort("Failed to open ClassicAD diagnostics output"); }
    out << std::setprecision(17);

    if (need_header) {
        out << "# time";
        for (int it = 0; it < static_cast<int>(m_turbine_state.size()); ++it) {
            out << " Ud_raw_" << it << " Ud_f_" << it << " rho_d_" << it
                << " T_target_" << it << " T_applied_" << it << " P_axial_" << it;
            if (m_wake_rotation_classic) {
                out << " P_rotor_" << it << " Omega_" << it
                    << " Q_target_" << it << " Q_applied_" << it
                    << " Q_LES_" << it;
            }
        }
        out << '\n';
    }

    out << time;
    for (const auto& s : m_turbine_state) {
        out << ' ' << s.disk_velocity_raw << ' ' << s.disk_velocity_filtered
            << ' ' << s.disk_density << ' ' << s.thrust_target
            << ' ' << s.thrust_applied << ' ' << s.axial_power;
        if (m_wake_rotation_classic) {
            out << ' ' << s.rotor_power << ' ' << s.omega
                << ' ' << s.torque_target << ' ' << s.torque_applied
                << ' ' << s.les_torque;
        }
    }
    out << '\n';
}

void
ClassicAD::write_memory_state (const std::string& checkpointname) const
{
    if (!ParallelDescriptor::IOProcessor()) { return; }

    std::ofstream out(checkpointname + "/ClassicADMemoryState",
                      std::ios::out | std::ios::trunc);
    if (!out.good()) { Abort("Failed to open ClassicADMemoryState for checkpoint write"); }
    out << std::setprecision(17);
    out << "ClassicADMemoryState_v1\n" << m_turbine_state.size() << '\n';
    for (const auto& s : m_turbine_state) {
        out << static_cast<int>(s.memory_initialized) << ' '
            << s.disk_velocity_filtered << '\n';
    }
}

bool
ClassicAD::read_memory_state (const std::string& restart_chkfile)
{
    const std::string fname = restart_chkfile + "/ClassicADMemoryState";
    if (!FileExists(fname)) {
        for (auto& s : m_turbine_state) {
            s.memory_initialized = false;
            s.disk_velocity_filtered = 0.0;
        }
        return false;
    }

    Vector<char> chars;
    ParallelDescriptor::ReadAndBcastFile(fname, chars);
    std::istringstream in(std::string(chars.dataPtr()), std::istringstream::in);
    std::string tag;
    int nturb = 0;
    in >> tag >> nturb;
    if (tag != "ClassicADMemoryState_v1") {
        Abort("Unknown ClassicADMemoryState format");
    }
    if (nturb != static_cast<int>(m_turbine_state.size())) {
        Abort("ClassicADMemoryState turbine count does not match windfarm_loc_table");
    }
    for (auto& s : m_turbine_state) {
        int initialized = 0;
        in >> initialized >> s.disk_velocity_filtered;
        s.memory_initialized = (initialized != 0);
    }
    if (!in.good() && !in.eof()) { Abort("Failed while reading ClassicADMemoryState"); }
    return true;
}

void
ClassicAD::initialize_dynamic_yaw (Real disk_face_angle_deg,
                                   const Vector<Real>& yaw_offsets_deg,
                                   Real sensor_distance_by_D,
                                   Real sensor_memory_time,
                                   Real yaw_out_per,
                                   int yaw_out_int,
                                   bool use_yaw_out_int,
                                   Real windfarm_start_time)
{
    const int nturb = static_cast<int>(m_xloc.size());
    if (static_cast<int>(yaw_offsets_deg.size()) != nturb) {
        Abort("ClassicAD dynamic yaw offset count does not match turbine count");
    }

    m_dynamic_yaw_enabled = true;
    m_yaw_sensor_distance_by_D = sensor_distance_by_D;
    m_yaw_sensor_memory_time = sensor_memory_time;
    m_yaw_out_per = yaw_out_per;
    m_yaw_out_int = yaw_out_int;
    m_use_yaw_out_int = use_yaw_out_int;
    m_next_yaw_output_time = windfarm_start_time;
    m_yaw_active_step_count = 0;
    m_yaw_offsets_rad.resize(nturb);
    m_yaw_state.assign(nturb, {});

    const Real psi0 = wrap_pi((disk_face_angle_deg-90.0)*classic_pi/180.0);
    for (int it = 0; it < nturb; ++it) {
        m_yaw_offsets_rad[it] = yaw_offsets_deg[it]*classic_pi/180.0;
        m_yaw_state[it].psi_ref = psi0;
        m_yaw_state[it].psi_rotor = wrap_pi(psi0+m_yaw_offsets_rad[it]);
        m_yaw_state[it].phi_cmd = psi0;
    }
    synchronize_rotor_angles();
}

void
ClassicAD::synchronize_rotor_angles ()
{
    m_turb_disk_angles.resize(m_yaw_state.size());
    for (int it = 0; it < static_cast<int>(m_yaw_state.size()); ++it) {
        m_yaw_state[it].psi_rotor =
            wrap_pi(m_yaw_state[it].psi_ref+m_yaw_offsets_rad[it]);
        m_turb_disk_angles[it] =
            wrap_360(90.0+m_yaw_state[it].psi_rotor*180.0/classic_pi);
    }
}

void
ClassicAD::update_dynamic_yaw (const Vector<Real>& sensor_u_raw,
                               const Vector<Real>& sensor_v_raw,
                               Real dt)
{
    if (!m_dynamic_yaw_enabled) { return; }
    const int nturb = static_cast<int>(m_yaw_state.size());
    if (static_cast<int>(sensor_u_raw.size()) != nturb ||
        static_cast<int>(sensor_v_raw.size()) != nturb) {
        Abort("ClassicAD dynamic-yaw sensor sample count does not match turbine count");
    }

    const Real alpha = (m_yaw_sensor_memory_time == 0.0)
                     ? 1.0
                     : 1.0-std::exp(-dt/m_yaw_sensor_memory_time);
    for (int it = 0; it < nturb; ++it) {
        auto& s = m_yaw_state[it];
        s.sensor_u_raw = sensor_u_raw[it];
        s.sensor_v_raw = sensor_v_raw[it];
        if (!s.sensor_initialized || m_yaw_sensor_memory_time == 0.0) {
            s.sensor_u_filtered = s.sensor_u_raw;
            s.sensor_v_filtered = s.sensor_v_raw;
            s.sensor_initialized = true;
        } else {
            s.sensor_u_filtered += alpha*(s.sensor_u_raw-s.sensor_u_filtered);
            s.sensor_v_filtered += alpha*(s.sensor_v_raw-s.sensor_v_filtered);
        }
        s.phi_cmd = std::atan2(s.sensor_v_filtered, s.sensor_u_filtered);
        const Real yaw_error = wrap_pi(s.phi_cmd-s.psi_ref);
        s.psi_ref = wrap_pi(s.psi_ref+alpha*yaw_error);
    }
    ++m_yaw_active_step_count;
    synchronize_rotor_angles();
}

bool
ClassicAD::yaw_output_due (Real step_end_time)
{
    if (!m_dynamic_yaw_enabled || m_yaw_active_step_count == 0) { return false; }
    if (m_use_yaw_out_int) {
        return m_yaw_active_step_count == 1 ||
               (m_yaw_active_step_count % m_yaw_out_int) == 0;
    }

    const Real eps = 1.0e-12*amrex::max(1.0, std::abs(step_end_time));
    if (step_end_time+eps < m_next_yaw_output_time) { return false; }
    do {
        m_next_yaw_output_time += m_yaw_out_per;
    } while (m_next_yaw_output_time <= step_end_time+eps);
    return true;
}

void
ClassicAD::write_yaw_diagnostics (Real time) const
{
    if (!ParallelDescriptor::IOProcessor()) { return; }
    const std::string filename = "yaw_angles_ClassicAD.txt";
    const bool need_header = !FileExists(filename);
    std::ofstream out(filename, std::ios::out | std::ios::app);
    if (!out.good()) { Abort("Failed to open ClassicAD yaw diagnostics output"); }
    out << std::setprecision(17);
    if (need_header) {
        out << "# time";
        for (int it = 0; it < static_cast<int>(m_yaw_state.size()); ++it) {
            out << " psi_ref_deg_" << it << " psi_rotor_deg_" << it
                << " phi_cmd_deg_" << it << " Us_raw_" << it
                << " Vs_raw_" << it << " Us_f_" << it << " Vs_f_" << it
                << " yaw_offset_deg_" << it;
        }
        out << '\n';
    }
    out << time;
    for (int it = 0; it < static_cast<int>(m_yaw_state.size()); ++it) {
        const auto& s = m_yaw_state[it];
        out << ' ' << s.psi_ref*180.0/classic_pi
            << ' ' << s.psi_rotor*180.0/classic_pi
            << ' ' << s.phi_cmd*180.0/classic_pi
            << ' ' << s.sensor_u_raw << ' ' << s.sensor_v_raw
            << ' ' << s.sensor_u_filtered << ' ' << s.sensor_v_filtered
            << ' ' << m_yaw_offsets_rad[it]*180.0/classic_pi;
    }
    out << '\n';
}

void
ClassicAD::write_yaw_state (const std::string& checkpointname) const
{
    if (!m_dynamic_yaw_enabled || !ParallelDescriptor::IOProcessor()) { return; }
    std::ofstream out(checkpointname + "/ClassicADYawState",
                      std::ios::out | std::ios::trunc);
    if (!out.good()) { Abort("Failed to open ClassicADYawState for checkpoint write"); }
    out << std::setprecision(17);
    out << "ClassicADYawState_v1\n" << m_yaw_state.size() << '\n';
    out << m_next_yaw_output_time << ' ' << m_yaw_active_step_count << '\n';
    for (const auto& s : m_yaw_state) {
        out << s.psi_ref << ' ' << s.psi_rotor << ' ' << s.phi_cmd << ' '
            << s.sensor_u_raw << ' ' << s.sensor_v_raw << ' '
            << s.sensor_u_filtered << ' ' << s.sensor_v_filtered << ' '
            << static_cast<int>(s.sensor_initialized) << '\n';
    }
}

bool
ClassicAD::read_yaw_state (const std::string& restart_chkfile)
{
    if (!m_dynamic_yaw_enabled) { return false; }
    const std::string fname = restart_chkfile + "/ClassicADYawState";
    if (!FileExists(fname)) { return false; }

    Vector<char> chars;
    ParallelDescriptor::ReadAndBcastFile(fname, chars);
    std::istringstream in(std::string(chars.dataPtr()), std::istringstream::in);
    std::string tag;
    int nturb = 0;
    in >> tag >> nturb;
    if (tag != "ClassicADYawState_v1") { Abort("Unknown ClassicADYawState format"); }
    if (nturb != static_cast<int>(m_yaw_state.size())) {
        Abort("ClassicADYawState turbine count does not match windfarm_loc_table");
    }
    in >> m_next_yaw_output_time >> m_yaw_active_step_count;
    for (auto& s : m_yaw_state) {
        int initialized = 0;
        in >> s.psi_ref >> s.psi_rotor >> s.phi_cmd
           >> s.sensor_u_raw >> s.sensor_v_raw
           >> s.sensor_u_filtered >> s.sensor_v_filtered >> initialized;
        s.sensor_initialized = (initialized != 0);
    }
    if (!in.good() && !in.eof()) { Abort("Failed while reading ClassicADYawState"); }
    synchronize_rotor_angles();
    return true;
}
