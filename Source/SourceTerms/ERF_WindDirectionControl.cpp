#include "ERF_WindDirectionControl.H"

#include <AMReX_ParmParse.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>
#include <AMReX_Utility.H>

#include "ERF_Constants.H"
#include "ERF_PlaneAverage.H"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>

using namespace amrex;

namespace erf_wind_direction
{

Real
wrap_angle (Real angle) noexcept
{
    Real wrapped = std::remainder(angle, Real(2.0) * PI);
    if (wrapped >= PI) wrapped -= Real(2.0) * PI;
    return wrapped;
}

bool
update_correction_demand (bool current, Real abs_error,
                          Real tolerance_on, Real tolerance_off) noexcept
{
    if (abs_error >= tolerance_on) return true;
    if (abs_error <= tolerance_off) return false;
    return current;
}

Real
update_geostrophic_angle (Real angle, Real error, Real gain,
                           Real control_interval) noexcept
{
    return wrap_angle(angle - gain * error * control_interval);
}

void
latch_event (EventState& state, Real error) noexcept
{
    if (!state.active) {
        state.remaining_angle = -error;
        state.active = (state.remaining_angle != Real(0.0));
    }
}

Real
event_angle_rate (const EventState& state, Real rate_magnitude, Real dt) noexcept
{
    if (!state.active || !(dt > Real(0.0))) return Real(0.0);
    const Real uncapped = std::copysign(rate_magnitude, state.remaining_angle);
    if (std::abs(state.remaining_angle) <= rate_magnitude * dt) {
        return state.remaining_angle / dt;
    }
    return uncapped;
}

void
advance_event (EventState& state, Real angle_rate, Real dt) noexcept
{
    if (!state.active) return;
    state.remaining_angle -= angle_rate * dt;
    const Real scale = std::max(Real(1.0), std::abs(angle_rate * dt));
    if (std::abs(state.remaining_angle) <=
        Real(32.0) * std::numeric_limits<Real>::epsilon() * scale) {
        state.remaining_angle = Real(0.0);
        state.active = false;
    }
}

Real
interpolate_profile (const std::vector<Real>& heights,
                     const Gpu::HostVector<Real>& values,
                     Real target_height)
{
    AMREX_ALWAYS_ASSERT(heights.size() == values.size());
    AMREX_ALWAYS_ASSERT(!heights.empty());
    AMREX_ALWAYS_ASSERT(target_height >= heights.front() &&
                        target_height <= heights.back());

    if (heights.size() == 1) return values.front();

    const auto upper = std::lower_bound(heights.begin(), heights.end(), target_height);
    if (upper == heights.begin()) return values.front();
    if (upper == heights.end()) return values.back();
    const std::size_t hi = static_cast<std::size_t>(upper - heights.begin());
    if (*upper == target_height) return values[hi];
    const std::size_t lo = hi - 1;
    const Real fraction = (target_height - heights[lo]) / (heights[hi] - heights[lo]);
    return values[lo] + fraction * (values[hi] - values[lo]);
}

} // namespace erf_wind_direction

bool
WindDirectionControl::enabled_in_inputs ()
{
    ParmParse pp("erf.wind_direction_control");
    bool enabled = false;
    pp.query("enabled", enabled);
    return enabled;
}

WindDirectionControl::WindDirectionControl (const SolverChoice& choice, int max_level,
                                            const Geometry& geom,
                                            const Vector<Real>& zlevels_stag)
{
    ParmParse pp("erf.wind_direction_control");
    Real target_direction_degrees = Real(0.0);
    Real tolerance_on_degrees = Real(2.0);
    Real tolerance_off_degrees = Real(1.0);
    Real rotation_rate_degrees = Real(1.0 / 60.0);

    if (!pp.query("target_direction", target_direction_degrees)) {
        Error("erf.wind_direction_control.target_direction is required when the controller is enabled");
    }
    if (!pp.query("target_height", m_target_height)) {
        Error("erf.wind_direction_control.target_height is required when the controller is enabled");
    }
    pp.query("start_time", m_start_time);
    pp.query("check_interval", m_check_interval);
    pp.query("tolerance_on", tolerance_on_degrees);
    pp.query("tolerance_off", tolerance_off_degrees);
    pp.query("synthetic_rotation_rate", rotation_rate_degrees);
    m_geostrophic_gain = Real(1.0 / 600.0);
    pp.query("geostrophic_gain", m_geostrophic_gain);
    pp.query("diagnostics_file", m_diagnostics_file);

    if (!std::isfinite(target_direction_degrees) ||
        !std::isfinite(m_target_height)) {
        Error("wind-direction target direction and height must be finite");
    }
    if (!(m_start_time >= Real(0.0)) || !std::isfinite(m_start_time)) {
        Error("erf.wind_direction_control.start_time must be finite and nonnegative");
    }
    if (m_diagnostics_file.empty()) {
        Error("erf.wind_direction_control.diagnostics_file must not be empty");
    }

    constexpr Real degrees_to_radians = PI / Real(180.0);
    m_target_direction = erf_wind_direction::wrap_angle(
        target_direction_degrees * degrees_to_radians);
    m_tolerance_on = tolerance_on_degrees * degrees_to_radians;
    m_tolerance_off = tolerance_off_degrees * degrees_to_radians;
    m_rotation_rate = rotation_rate_degrees * degrees_to_radians;
    m_next_check_time = m_start_time;

    if (!(m_check_interval > Real(0.0)) || !std::isfinite(m_check_interval)) {
        Error("erf.wind_direction_control.check_interval must be positive");
    }
    if (!(m_rotation_rate > Real(0.0)) || !std::isfinite(m_rotation_rate)) {
        Error("erf.wind_direction_control.synthetic_rotation_rate must be positive");
    }
    if (!(m_geostrophic_gain > Real(0.0)) || !std::isfinite(m_geostrophic_gain)) {
        Error("erf.wind_direction_control.geostrophic_gain must be positive");
    }
    if (!(m_tolerance_off >= Real(0.0) && m_tolerance_off < m_tolerance_on &&
          m_tolerance_on < PI)) {
        Error("wind-direction tolerances must satisfy 0 <= tolerance_off < tolerance_on < 180 degrees");
    }
    if (max_level != 0) {
        Error("the wind-direction controller currently requires amr.max_level = 0");
    }
    const bool supported_constant = choice.mesh_type == MeshType::ConstantDz &&
                                    choice.terrain_type == TerrainType::None;
    const bool supported_stretched = choice.mesh_type == MeshType::StretchedDz &&
                                     choice.terrain_type == TerrainType::StaticFittedMesh &&
                                     choice.flat_terrain;
    if (!(supported_constant || supported_stretched)) {
        Error("the wind-direction controller supports ConstantDz without terrain or a flat "
              "StretchedDz StaticFittedMesh only");
    }
    if (!choice.use_coriolis || choice.coriolis_3d || choice.variable_coriolis ||
        choice.init_type == InitType::HindCast) {
        Error("the wind-direction controller requires constant-latitude 2-D Coriolis forcing");
    }
    if (choice.abl_driver_type != ABLDriverType::GeostrophicWind) {
        Error("the wind-direction controller requires erf.abl_driver_type = GeostrophicWind");
    }
    if (choice.have_geo_wind_profile) {
        Error("the wind-direction controller does not support height-dependent geostrophic wind");
    }
    if (choice.const_massflux_u != Real(0.0) || choice.const_massflux_v != Real(0.0)) {
        Error("the wind-direction controller cannot be combined with constant-mass-flux forcing");
    }
    if (choice.large_scale_forcing ||
        (choice.nudging_from_input_sounding && choice.nudging_u)) {
        Error("the wind-direction controller cannot be combined with large-scale forcing or momentum nudging");
    }
    if (choice.nudging_from_observations) {
        ParmParse pp_obs("erf.obs_nudging");
        bool nudge_wind = true;
        bool nudge_w = false;
        pp_obs.query("nudge_wind", nudge_wind);
        pp_obs.query("nudge_w", nudge_w);
        if (nudge_wind || nudge_w) {
            Error("the wind-direction controller cannot be combined with observational momentum nudging");
        }
    }

    m_planetary_f = choice.coriolis_factor * choice.sinphi;
    if (m_planetary_f == Real(0.0) || !std::isfinite(m_planetary_f)) {
        Error("the wind-direction controller requires a nonzero constant planetary Coriolis parameter");
    }
    const Real ug = choice.abl_geo_wind[0];
    const Real vg = choice.abl_geo_wind[1];
    m_geostrophic_magnitude = std::hypot(ug, vg);
    if (!(m_geostrophic_magnitude > Real(0.0)) ||
        !std::isfinite(m_geostrophic_magnitude)) {
        Error("the wind-direction controller requires a nonzero horizontal erf.abl_geo_wind");
    }
    m_geostrophic_angle = std::atan2(vg, ug);
    regenerate_geostrophic_forcing();

    update_vertical_grid(geom, zlevels_stag);

    Print() << "Wind-direction controller: target=" << target_direction_degrees
            << " deg, height=" << m_target_height << " m, check interval="
            << m_check_interval << " s\n";
}

void
WindDirectionControl::update_vertical_grid (const Geometry& geom,
                                            const Vector<Real>& zlevels_stag)
{
    const int nz = geom.Domain().length(2);
    if (static_cast<int>(zlevels_stag.size()) != nz + 1) {
        Error("wind-direction controller received an inconsistent zlevels_stag array");
    }
    m_mass_heights.resize(static_cast<std::size_t>(nz));
    const Real lower_boundary = zlevels_stag.front();
    for (int k = 0; k < nz; ++k) {
        m_mass_heights[static_cast<std::size_t>(k)] =
            Real(0.5) * (zlevels_stag[k] + zlevels_stag[k+1]) - lower_boundary;
    }
    if (m_target_height < m_mass_heights.front() ||
        m_target_height > m_mass_heights.back()) {
        Error("erf.wind_direction_control.target_height must lie between the lowest and "
              "highest horizontal-velocity mass-level centers");
    }
}

bool
WindDirectionControl::control_due (Real time) const noexcept
{
    const Real scale = std::max({Real(1.0), std::abs(time), std::abs(m_next_check_time)});
    return time + Real(32.0) * std::numeric_limits<Real>::epsilon() * scale >=
           m_next_check_time;
}

void
WindDirectionControl::advance_check_time (Real time) noexcept
{
    if (m_next_check_time > time) {
        m_next_check_time += m_check_interval;
        return;
    }
    const Real elapsed = time - m_next_check_time;
    const auto intervals = static_cast<long long>(std::floor(elapsed / m_check_interval)) + 1;
    m_next_check_time += static_cast<Real>(intervals) * m_check_interval;
    if (m_next_check_time <= time) m_next_check_time = time + m_check_interval;
}

WindDirectionControl::Measurement
WindDirectionControl::measure (const MultiFab& xvel, const MultiFab& yvel,
                               const Geometry& geom) const
{
    PlaneAverage u_average(&xvel, geom, 2, IntVect(0));
    PlaneAverage v_average(&yvel, geom, 2, IntVect(0));
    u_average.compute_averages(ZDir(), u_average.field());
    v_average.compute_averages(ZDir(), v_average.field());

    Gpu::HostVector<Real> u_profile(static_cast<std::size_t>(u_average.ncell_line()));
    Gpu::HostVector<Real> v_profile(static_cast<std::size_t>(v_average.ncell_line()));
    u_average.line_average(0, u_profile);
    v_average.line_average(0, v_profile);

    Measurement result;
    result.u = erf_wind_direction::interpolate_profile(m_mass_heights, u_profile,
                                                        m_target_height);
    result.v = erf_wind_direction::interpolate_profile(m_mass_heights, v_profile,
                                                        m_target_height);
    result.speed = std::hypot(result.u, result.v);
    result.valid = result.speed >= Real(1.0e-6);
    if (result.valid) {
        result.angle = std::atan2(result.v, result.u);
        result.error = erf_wind_direction::wrap_angle(result.angle - m_target_direction);
    }
    return result;
}

void
WindDirectionControl::begin_step (Real time, Real dt, const MultiFab& xvel,
                                  const MultiFab& yvel, const Geometry& geom)
{
    Measurement measurement;
    const bool checked = control_due(time);
    if (checked) {
        measurement = measure(xvel, yvel, geom);
        if (measurement.valid) {
            m_correction_demand = erf_wind_direction::update_correction_demand(
                m_correction_demand, std::abs(measurement.error),
                m_tolerance_on, m_tolerance_off);
            if (m_correction_demand) {
                m_geostrophic_angle = erf_wind_direction::update_geostrophic_angle(
                    m_geostrophic_angle, measurement.error,
                    m_geostrophic_gain, m_check_interval);
                regenerate_geostrophic_forcing();
            }
            if (!m_event.active && std::abs(measurement.error) >= m_tolerance_on) {
                erf_wind_direction::latch_event(m_event, measurement.error);
            }
        }
        advance_check_time(time);
    }

    m_step_angle_rate = erf_wind_direction::event_angle_rate(m_event, m_rotation_rate, dt);
    m_synthetic_coriolis = -m_step_angle_rate;

    if (checked) {
        write_diagnostic(time, measurement);
        constexpr Real radians_to_degrees = Real(180.0) / PI;
        Print() << "Wind-direction control t=" << time
                << " valid=" << static_cast<int>(measurement.valid)
                << " angle=" << measurement.angle * radians_to_degrees
                << " deg error=" << measurement.error * radians_to_degrees
                << " deg active=" << static_cast<int>(m_event.active) << "\n";
    }
}

void
WindDirectionControl::end_step (Real dt) noexcept
{
    erf_wind_direction::advance_event(m_event, m_step_angle_rate, dt);
    m_step_angle_rate = Real(0.0);
    m_synthetic_coriolis = Real(0.0);
}

void
WindDirectionControl::regenerate_geostrophic_forcing () noexcept
{
    const Real ug = m_geostrophic_magnitude * std::cos(m_geostrophic_angle);
    const Real vg = m_geostrophic_magnitude * std::sin(m_geostrophic_angle);
    m_geo_forcing = {-m_planetary_f * vg, m_planetary_f * ug, Real(0.0)};
}

void
WindDirectionControl::write_diagnostic (Real time, const Measurement& m) const
{
    if (!ParallelDescriptor::IOProcessor()) return;
    bool write_header = !FileExists(m_diagnostics_file);
    if (!write_header) {
        std::ifstream existing(m_diagnostics_file, std::ios::binary | std::ios::ate);
        write_header = existing.good() && existing.tellg() == std::streampos(0);
    }
    std::ofstream output(m_diagnostics_file, std::ios::out | std::ios::app);
    if (!output.good()) FileOpenFailed(m_diagnostics_file);
    if (write_header) {
        output << "time,u_bar,v_bar,speed,measured_angle_deg,target_angle_deg,error_deg,"
                  "geostrophic_magnitude,geostrophic_angle_deg,geo_forcing_x,geo_forcing_y,"
                  "synthetic_coriolis,remaining_angle_deg,active,correction_demand,valid\n";
    }
    constexpr Real radians_to_degrees = Real(180.0) / PI;
    output << std::setprecision(17)
           << time << ',' << m.u << ',' << m.v << ',' << m.speed << ','
           << m.angle * radians_to_degrees << ','
           << m_target_direction * radians_to_degrees << ','
           << m.error * radians_to_degrees << ','
           << m_geostrophic_magnitude << ','
           << m_geostrophic_angle * radians_to_degrees << ','
           << m_geo_forcing[0] << ',' << m_geo_forcing[1] << ','
           << m_synthetic_coriolis << ','
           << m_event.remaining_angle * radians_to_degrees << ','
           << static_cast<int>(m_event.active) << ','
           << static_cast<int>(m_correction_demand) << ','
           << static_cast<int>(m.valid) << '\n';
}

void
WindDirectionControl::write_checkpoint (const std::string& checkpoint_name) const
{
    if (!ParallelDescriptor::IOProcessor()) return;
    const std::string filename = checkpoint_name + "/WindDirectionControl";
    std::ofstream output(filename, std::ios::out | std::ios::trunc);
    if (!output.good()) FileOpenFailed(filename);
    output << "ERF wind direction control state\n1\n" << std::setprecision(17)
           << m_geostrophic_magnitude << ' ' << m_geostrophic_angle << '\n'
           << static_cast<int>(m_event.active) << ' ' << m_event.remaining_angle << '\n'
           << m_next_check_time << '\n'
           << static_cast<int>(m_correction_demand) << '\n';
}

void
WindDirectionControl::read_checkpoint (const std::string& checkpoint_name,
                                       Real restart_time)
{
    const std::string filename = checkpoint_name + "/WindDirectionControl";
    if (!FileExists(filename)) {
        Warning("Restart checkpoint has no WindDirectionControl state; initializing the "
                "enabled controller at the restart time");
        m_next_check_time = std::max(m_start_time, restart_time);
        return;
    }

    Vector<char> chars;
    ParallelDescriptor::ReadAndBcastFile(filename, chars);
    std::istringstream input(std::string(chars.dataPtr(), chars.size()));
    std::string title;
    int version = 0;
    int active = 0;
    int demand = 0;
    std::getline(input, title);
    input >> version;
    if (title != "ERF wind direction control state" || version != 1) {
        Error("Unsupported or malformed WindDirectionControl checkpoint record in '" +
              filename + "'");
    }
    input >> m_geostrophic_magnitude >> m_geostrophic_angle;
    input >> active >> m_event.remaining_angle;
    input >> m_next_check_time;
    input >> demand;
    if (!input || !(m_geostrophic_magnitude > Real(0.0)) ||
        !std::isfinite(m_geostrophic_magnitude) ||
        (active != 0 && active != 1) || (demand != 0 && demand != 1) ||
        !std::isfinite(m_geostrophic_angle) ||
        !std::isfinite(m_event.remaining_angle) ||
        !std::isfinite(m_next_check_time) ||
        (active != 0 && m_event.remaining_angle == Real(0.0))) {
        Error("Malformed WindDirectionControl checkpoint record in '" + filename + "'");
    }
    m_event.active = active != 0;
    m_correction_demand = demand != 0;
    if (!m_event.active) m_event.remaining_angle = Real(0.0);
    regenerate_geostrophic_forcing();
}
