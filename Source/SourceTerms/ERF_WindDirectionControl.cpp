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

namespace
{
constexpr const char* diagnostic_header =
    "time,phase,u_bar,v_bar,speed,measured_angle_deg,target_angle_deg,error_deg,"
    "latched_error_deg,event_start_time,geostrophic_magnitude,"
    "geostrophic_angle_deg,midpoint_geostrophic_angle_deg,geo_forcing_x,"
    "geo_forcing_y,rotation_omega3,remaining_flow_angle_deg,"
    "remaining_geo_angle_deg,settling_time_remaining,valid,event_initiated";
}

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
event_requested (Real error, Real threshold) noexcept
{
    return std::abs(error) >= threshold;
}

void
latch_episode (EpisodeState& state, Real error, Real proportional_gain,
               Real time) noexcept
{
    if (state.phase != Phase::Idle) return;
    state.phase = Phase::Rotating;
    state.latched_error = error;
    state.event_start_time = time;
    state.remaining_flow_angle = -error;
    state.remaining_geo_angle = -proportional_gain * error;
    state.settling_until = Real(0.0);
}

Real
episode_flow_angle_rate (const EpisodeState& state, Real rate_magnitude,
                         Real dt) noexcept
{
    if (state.phase != Phase::Rotating || !(dt > Real(0.0))) return Real(0.0);
    if (std::abs(state.remaining_flow_angle) <= rate_magnitude * dt) {
        return state.remaining_flow_angle / dt;
    }
    return std::copysign(rate_magnitude, state.remaining_flow_angle);
}

Real
episode_geo_increment (const EpisodeState& state, Real flow_increment) noexcept
{
    if (state.phase != Phase::Rotating || state.remaining_flow_angle == Real(0.0)) {
        return Real(0.0);
    }
    const Real fraction = flow_increment / state.remaining_flow_angle;
    return fraction * state.remaining_geo_angle;
}

void
advance_episode (EpisodeState& state, Real flow_increment, Real geo_increment,
                 Real completion_time, Real settling_time) noexcept
{
    if (state.phase != Phase::Rotating) return;
    state.remaining_flow_angle -= flow_increment;
    state.remaining_geo_angle -= geo_increment;

    const Real flow_scale = std::max(Real(1.0), std::abs(flow_increment));
    if (std::abs(state.remaining_flow_angle) <=
        Real(32.0) * std::numeric_limits<Real>::epsilon() * flow_scale) {
        state.remaining_flow_angle = Real(0.0);
        state.remaining_geo_angle = Real(0.0);
        state.phase = Phase::Settling;
        state.settling_until = completion_time + settling_time;
    }
}

const char*
phase_name (Phase phase) noexcept
{
    switch (phase) {
    case Phase::Waiting:  return "waiting";
    case Phase::Idle:     return "idle";
    case Phase::Rotating: return "rotating";
    case Phase::Settling: return "settling";
    }
    return "invalid";
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
    Real rotation_rate_degrees = Real(0.0005);

    if (!pp.query("target_direction", target_direction_degrees)) {
        Error("erf.wind_direction_control.target_direction is required when the controller is enabled");
    }
    if (!pp.query("target_height", m_target_height)) {
        Error("erf.wind_direction_control.target_height is required when the controller is enabled");
    }
    pp.query("start_time", m_start_time);
    pp.query("check_interval", m_check_interval);
    pp.query("tolerance_on", tolerance_on_degrees);
    pp.query("synthetic_rotation_rate", rotation_rate_degrees);
    pp.query("geostrophic_proportional_gain", m_geostrophic_proportional_gain);
    pp.query("post_event_settling_time", m_post_event_settling_time);
    pp.query("diagnostics_file", m_diagnostics_file);

    if (pp.contains("geostrophic_gain") || pp.contains("tolerance_off") ||
        pp.contains("synthetic_rotation_gain")) {
        Error("erf.wind_direction_control.geostrophic_gain, tolerance_off, and "
              "synthetic_rotation_gain are obsolete; use the dimensionless "
              "geostrophic_proportional_gain with the coupled SM14 controller");
    }

    if (!std::isfinite(target_direction_degrees) || !std::isfinite(m_target_height)) {
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
    m_rotation_rate = rotation_rate_degrees * degrees_to_radians;
    m_next_check_time = m_start_time;

    if (!(m_check_interval > Real(0.0)) || !std::isfinite(m_check_interval)) {
        Error("erf.wind_direction_control.check_interval must be positive");
    }
    if (!(m_rotation_rate > Real(0.0)) || !std::isfinite(m_rotation_rate)) {
        Error("erf.wind_direction_control.synthetic_rotation_rate must be positive");
    }
    if (!(m_geostrophic_proportional_gain >= Real(0.0) &&
          m_geostrophic_proportional_gain <= Real(1.0)) ||
        !std::isfinite(m_geostrophic_proportional_gain)) {
        Error("erf.wind_direction_control.geostrophic_proportional_gain must be between 0 and 1");
    }
    if (m_geostrophic_proportional_gain == Real(0.0)) {
        Warning("Wind-direction control geostrophic_proportional_gain is zero; "
                "the flow will rotate while the geostrophic direction remains fixed");
    }
    if (!(m_post_event_settling_time >= Real(0.0)) ||
        !std::isfinite(m_post_event_settling_time)) {
        Error("erf.wind_direction_control.post_event_settling_time must be finite and nonnegative");
    }
    if (!(m_tolerance_on > Real(0.0) && m_tolerance_on < PI)) {
        Error("erf.wind_direction_control.tolerance_on must be between 0 and 180 degrees");
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
    m_step_midpoint_geo_angle = m_geostrophic_angle;
    regenerate_geostrophic_forcing(m_geostrophic_angle);

    update_vertical_grid(geom, zlevels_stag);

    Print() << "Wind-direction controller: target=" << target_direction_degrees
            << " deg, height=" << m_target_height << " m, check interval="
            << m_check_interval << " s, rotation rate=" << rotation_rate_degrees
            << " deg/s, proportional gain=" << m_geostrophic_proportional_gain
            << ", settling time=" << m_post_event_settling_time << " s\n";
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
    bool event_initiated = false;
    const bool checked = control_due(time);
    if (checked) {
        measurement = measure(xvel, yvel, geom);

        if (m_episode.phase == erf_wind_direction::Phase::Waiting && time >= m_start_time) {
            m_episode.phase = erf_wind_direction::Phase::Idle;
        }
        if (m_episode.phase == erf_wind_direction::Phase::Settling &&
            time >= m_episode.settling_until) {
            m_episode.phase = erf_wind_direction::Phase::Idle;
        }
        if (m_episode.phase == erf_wind_direction::Phase::Idle && measurement.valid &&
            erf_wind_direction::event_requested(measurement.error, m_tolerance_on)) {
            erf_wind_direction::latch_episode(
                m_episode, measurement.error, m_geostrophic_proportional_gain, time);
            event_initiated = true;
        }
        advance_check_time(time);
    }

    const Real step_flow_rate = erf_wind_direction::episode_flow_angle_rate(
        m_episode, m_rotation_rate, dt);
    m_step_flow_increment = step_flow_rate * dt;
    m_step_geo_increment = erf_wind_direction::episode_geo_increment(
        m_episode, m_step_flow_increment);
    m_rotation_omega3 = -step_flow_rate;
    m_step_midpoint_geo_angle = erf_wind_direction::wrap_angle(
        m_geostrophic_angle + Real(0.5) * m_step_geo_increment);
    regenerate_geostrophic_forcing(m_step_midpoint_geo_angle);

    if (checked) {
        write_diagnostic(time, measurement, event_initiated);
        constexpr Real radians_to_degrees = Real(180.0) / PI;
        Print() << "Wind-direction control t=" << time
                << " phase=" << erf_wind_direction::phase_name(m_episode.phase)
                << " valid=" << static_cast<int>(measurement.valid)
                << " angle=" << measurement.angle * radians_to_degrees
                << " deg error=" << measurement.error * radians_to_degrees
                << " deg omega3=" << m_rotation_omega3 << " 1/s"
                << " remaining flow="
                << m_episode.remaining_flow_angle * radians_to_degrees << " deg\n";
    }
}

void
WindDirectionControl::end_step (Real time, Real dt) noexcept
{
    m_geostrophic_angle = erf_wind_direction::wrap_angle(
        m_geostrophic_angle + m_step_geo_increment);
    erf_wind_direction::advance_episode(
        m_episode, m_step_flow_increment, m_step_geo_increment,
        time + dt, m_post_event_settling_time);

    m_step_flow_increment = Real(0.0);
    m_step_geo_increment = Real(0.0);
    m_rotation_omega3 = Real(0.0);
    m_step_midpoint_geo_angle = m_geostrophic_angle;
    regenerate_geostrophic_forcing(m_geostrophic_angle);
}

void
WindDirectionControl::regenerate_geostrophic_forcing (Real angle) noexcept
{
    const Real ug = m_geostrophic_magnitude * std::cos(angle);
    const Real vg = m_geostrophic_magnitude * std::sin(angle);
    m_geo_forcing = {-m_planetary_f * vg, m_planetary_f * ug, Real(0.0)};
}

void
WindDirectionControl::write_diagnostic (Real time, const Measurement& m,
                                        bool event_initiated) const
{
    if (!ParallelDescriptor::IOProcessor()) return;
    bool write_header = !FileExists(m_diagnostics_file);
    if (!write_header) {
        std::ifstream existing(m_diagnostics_file);
        std::string header;
        if (!std::getline(existing, header)) {
            write_header = true;
        } else if (header != diagnostic_header) {
            Error("Existing wind-direction diagnostics file has an incompatible header; "
                  "choose a new erf.wind_direction_control.diagnostics_file");
        }
    }
    std::ofstream output(m_diagnostics_file, std::ios::out | std::ios::app);
    if (!output.good()) FileOpenFailed(m_diagnostics_file);
    if (write_header) output << diagnostic_header << '\n';

    constexpr Real radians_to_degrees = Real(180.0) / PI;
    const Real settling_remaining =
        m_episode.phase == erf_wind_direction::Phase::Settling
            ? std::max(Real(0.0), m_episode.settling_until - time)
            : Real(0.0);
    output << std::setprecision(17)
           << time << ',' << erf_wind_direction::phase_name(m_episode.phase) << ','
           << m.u << ',' << m.v << ',' << m.speed << ','
           << m.angle * radians_to_degrees << ','
           << m_target_direction * radians_to_degrees << ','
           << m.error * radians_to_degrees << ','
           << m_episode.latched_error * radians_to_degrees << ','
           << m_episode.event_start_time << ','
           << m_geostrophic_magnitude << ','
           << m_geostrophic_angle * radians_to_degrees << ','
           << m_step_midpoint_geo_angle * radians_to_degrees << ','
           << m_geo_forcing[0] << ',' << m_geo_forcing[1] << ','
           << m_rotation_omega3 << ','
           << m_episode.remaining_flow_angle * radians_to_degrees << ','
           << m_episode.remaining_geo_angle * radians_to_degrees << ','
           << settling_remaining << ','
           << static_cast<int>(m.valid) << ','
           << static_cast<int>(event_initiated) << '\n';
}

void
WindDirectionControl::write_checkpoint (const std::string& checkpoint_name) const
{
    if (!ParallelDescriptor::IOProcessor()) return;
    const std::string filename = checkpoint_name + "/WindDirectionControl";
    std::ofstream output(filename, std::ios::out | std::ios::trunc);
    if (!output.good()) FileOpenFailed(filename);
    output << "ERF wind direction control state\n4\n" << std::setprecision(17)
           << static_cast<int>(m_episode.phase) << '\n'
           << m_geostrophic_magnitude << ' ' << m_geostrophic_angle << '\n'
           << m_episode.latched_error << ' ' << m_episode.event_start_time << '\n'
           << m_episode.remaining_flow_angle << ' '
           << m_episode.remaining_geo_angle << '\n'
           << m_next_check_time << ' ' << m_episode.settling_until << '\n';
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
        m_episode.phase = restart_time >= m_start_time
                              ? erf_wind_direction::Phase::Idle
                              : erf_wind_direction::Phase::Waiting;
        return;
    }

    Vector<char> chars;
    ParallelDescriptor::ReadAndBcastFile(filename, chars);
    std::istringstream input(std::string(chars.dataPtr(), chars.size()));
    std::string title;
    int version = 0;
    int phase_value = 0;
    std::getline(input, title);
    input >> version;
    if (title != "ERF wind direction control state") {
        Error("Unsupported or malformed WindDirectionControl checkpoint record in '" +
              filename + "'");
    }
    if (version >= 1 && version <= 3) {
        Error("WindDirectionControl checkpoint versions 1-3 contain an incompatible "
              "controller state; remove the controller record to initialize a fresh "
              "coupled SM14 controller");
    }
    if (version != 4) {
        Error("Unsupported WindDirectionControl checkpoint version in '" + filename + "'");
    }

    input >> phase_value;
    input >> m_geostrophic_magnitude >> m_geostrophic_angle;
    input >> m_episode.latched_error >> m_episode.event_start_time;
    input >> m_episode.remaining_flow_angle >> m_episode.remaining_geo_angle;
    input >> m_next_check_time >> m_episode.settling_until;

    const bool phase_valid =
        phase_value >= static_cast<int>(erf_wind_direction::Phase::Waiting) &&
        phase_value <= static_cast<int>(erf_wind_direction::Phase::Settling);
    if (phase_valid) {
        m_episode.phase = static_cast<erf_wind_direction::Phase>(phase_value);
    }
    const bool rotating = phase_valid &&
        m_episode.phase == erf_wind_direction::Phase::Rotating;
    const bool inactive_angles_zero =
        m_episode.remaining_flow_angle == Real(0.0) &&
        m_episode.remaining_geo_angle == Real(0.0);
    if (!input || !phase_valid || !(m_geostrophic_magnitude > Real(0.0)) ||
        !std::isfinite(m_geostrophic_magnitude) ||
        !std::isfinite(m_geostrophic_angle) ||
        !std::isfinite(m_episode.latched_error) ||
        !std::isfinite(m_episode.event_start_time) ||
        !std::isfinite(m_episode.remaining_flow_angle) ||
        !std::isfinite(m_episode.remaining_geo_angle) ||
        !std::isfinite(m_next_check_time) ||
        !std::isfinite(m_episode.settling_until) ||
        (rotating && m_episode.remaining_flow_angle == Real(0.0)) ||
        (!rotating && !inactive_angles_zero)) {
        Error("Malformed WindDirectionControl checkpoint record in '" + filename + "'");
    }

    m_rotation_omega3 = Real(0.0);
    m_step_flow_increment = Real(0.0);
    m_step_geo_increment = Real(0.0);
    m_step_midpoint_geo_angle = m_geostrophic_angle;
    regenerate_geostrophic_forcing(m_geostrophic_angle);
}
