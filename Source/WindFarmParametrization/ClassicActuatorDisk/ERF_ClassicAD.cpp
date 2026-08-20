#include <ERF_ClassicAD.H>

#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Utility.H>

#include <fstream>
#include <iomanip>
#include <sstream>

using namespace amrex;

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
