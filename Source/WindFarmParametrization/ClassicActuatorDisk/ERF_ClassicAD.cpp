#include <ERF_ClassicAD.H>

#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Utility.H>

#include <fstream>
#include <iomanip>
#include <sstream>
#include <utility>

using namespace amrex;

void
ClassicAD::configure (Real ctprime, Real memory_time,
                      Real diameter, Real hub_height, Real spacing,
                      bool spacing_was_supplied)
{
    m_ctprime = ctprime;
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
