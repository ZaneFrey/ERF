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
    "time,target_u_bar,target_v_bar,target_speed,target_angle_deg,"
    "desired_angle_deg,error_deg,geostrophic_u_bar,geostrophic_v_bar,"
    "geostrophic_speed,geostrophic_angle_deg,relaxation_gain,rotation_omega3,"
    "geostrophic_magnitude,effective_ug,effective_vg,geo_forcing_x,geo_forcing_y";
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

Real
continuous_rotation_rate (Real error, Real relaxation_gain) noexcept
{
    return relaxation_gain * error;
}

GpuArray<Real, AMREX_SPACEDIM>
geostrophic_forcing (Real planetary_f, Real magnitude, Real angle) noexcept
{
    const Real ug = magnitude * std::cos(angle);
    const Real vg = magnitude * std::sin(angle);
    return {-planetary_f * vg, planetary_f * ug, Real(0.0)};
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

    if (!pp.query("target_direction", target_direction_degrees)) {
        Error("erf.wind_direction_control.target_direction is required when the controller is enabled");
    }
    if (!pp.query("target_height", m_target_height)) {
        Error("erf.wind_direction_control.target_height is required when the controller is enabled");
    }
    if (!pp.query("geostrophic_height", m_geostrophic_height)) {
        Error("erf.wind_direction_control.geostrophic_height is required when the controller is enabled");
    }
    pp.query("relaxation_gain", m_relaxation_gain);
    pp.query("diagnostics_interval", m_diagnostics_interval);
    pp.query("diagnostics_file", m_diagnostics_file);

    if (pp.contains("start_time") || pp.contains("check_interval") ||
        pp.contains("tolerance_on") || pp.contains("tolerance_off") ||
        pp.contains("synthetic_rotation_rate") ||
        pp.contains("synthetic_rotation_gain") ||
        pp.contains("geostrophic_gain") ||
        pp.contains("geostrophic_proportional_gain") ||
        pp.contains("post_event_settling_time") ||
        pp.contains("ramp_enabled") || pp.contains("ramp_duration")) {
        Error("event-based erf.wind_direction_control inputs are obsolete; use "
              "geostrophic_height and relaxation_gain with the continuous controller");
    }

    if (!std::isfinite(target_direction_degrees) ||
        !std::isfinite(m_target_height) ||
        !std::isfinite(m_geostrophic_height)) {
        Error("wind-direction target direction and sampling heights must be finite");
    }
    if (!(m_relaxation_gain > Real(0.0)) ||
        !std::isfinite(m_relaxation_gain)) {
        Error("erf.wind_direction_control.relaxation_gain must be positive and finite");
    }
    if (!(m_diagnostics_interval > Real(0.0)) ||
        !std::isfinite(m_diagnostics_interval)) {
        Error("erf.wind_direction_control.diagnostics_interval must be positive and finite");
    }
    if (m_diagnostics_file.empty()) {
        Error("erf.wind_direction_control.diagnostics_file must not be empty");
    }

    constexpr Real degrees_to_radians = PI / Real(180.0);
    m_target_direction = erf_wind_direction::wrap_angle(
        target_direction_degrees * degrees_to_radians);

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
    regenerate_geostrophic_forcing(m_geostrophic_angle);

    update_vertical_grid(geom, zlevels_stag);

    Print() << "Continuous wind-direction controller: target="
            << target_direction_degrees << " deg at " << m_target_height
            << " m, geostrophic sampling height=" << m_geostrophic_height
            << " m, relaxation gain=" << m_relaxation_gain
            << " 1/s, diagnostics interval=" << m_diagnostics_interval << " s\n";
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
    if (m_geostrophic_height < m_mass_heights.front() ||
        m_geostrophic_height > m_mass_heights.back()) {
        Error("erf.wind_direction_control.geostrophic_height must lie between the lowest and "
              "highest horizontal-velocity mass-level centers");
    }
}

bool
WindDirectionControl::diagnostic_due (Real time) const noexcept
{
    const Real scale = std::max({Real(1.0), std::abs(time),
                                 std::abs(m_next_diagnostic_time)});
    return time + Real(32.0) * std::numeric_limits<Real>::epsilon() * scale >=
           m_next_diagnostic_time;
}

void
WindDirectionControl::advance_diagnostic_time (Real time) noexcept
{
    const Real elapsed = time - m_next_diagnostic_time;
    const auto intervals =
        static_cast<long long>(std::floor(elapsed / m_diagnostics_interval)) + 1;
    m_next_diagnostic_time += static_cast<Real>(intervals) * m_diagnostics_interval;
    if (m_next_diagnostic_time <= time) {
        m_next_diagnostic_time = time + m_diagnostics_interval;
    }
}

WindDirectionControl::Measurements
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

    Measurements result;
    result.target.u = erf_wind_direction::interpolate_profile(
        m_mass_heights, u_profile, m_target_height);
    result.target.v = erf_wind_direction::interpolate_profile(
        m_mass_heights, v_profile, m_target_height);
    result.target.speed = std::hypot(result.target.u, result.target.v);
    result.target.angle = std::atan2(result.target.v, result.target.u);
    result.error = erf_wind_direction::wrap_angle(
        result.target.angle - m_target_direction);

    result.geostrophic.u = erf_wind_direction::interpolate_profile(
        m_mass_heights, u_profile, m_geostrophic_height);
    result.geostrophic.v = erf_wind_direction::interpolate_profile(
        m_mass_heights, v_profile, m_geostrophic_height);
    result.geostrophic.speed = std::hypot(result.geostrophic.u,
                                          result.geostrophic.v);
    result.geostrophic.angle = std::atan2(result.geostrophic.v,
                                          result.geostrophic.u);
    return result;
}

void
WindDirectionControl::update (Real time, const MultiFab& xvel,
                              const MultiFab& yvel, const Geometry& geom)
{
    const Measurements measurements = measure(xvel, yvel, geom);
    m_rotation_omega3 = erf_wind_direction::continuous_rotation_rate(
        measurements.error, m_relaxation_gain);
    m_geostrophic_angle = measurements.geostrophic.angle;
    regenerate_geostrophic_forcing(m_geostrophic_angle);

    if (diagnostic_due(time)) {
        write_diagnostic(time, measurements);
        constexpr Real radians_to_degrees = Real(180.0) / PI;
        Print() << "Wind-direction control t=" << time
                << " target angle=" << measurements.target.angle * radians_to_degrees
                << " deg error=" << measurements.error * radians_to_degrees
                << " deg geostrophic angle="
                << measurements.geostrophic.angle * radians_to_degrees
                << " deg omega3=" << m_rotation_omega3 << " 1/s\n";
        advance_diagnostic_time(time);
    }
}

void
WindDirectionControl::regenerate_geostrophic_forcing (Real angle) noexcept
{
    m_effective_ug = m_geostrophic_magnitude * std::cos(angle);
    m_effective_vg = m_geostrophic_magnitude * std::sin(angle);
    m_geo_forcing = erf_wind_direction::geostrophic_forcing(
        m_planetary_f, m_geostrophic_magnitude, angle);
}

void
WindDirectionControl::write_diagnostic (Real time,
                                        const Measurements& measurements) const
{
    if (!ParallelDescriptor::IOProcessor()) return;

    bool write_header = !FileExists(m_diagnostics_file);
    std::string last_data_line;
    if (!write_header) {
        std::ifstream existing(m_diagnostics_file);
        std::string header;
        if (!std::getline(existing, header)) {
            write_header = true;
        } else if (header != diagnostic_header) {
            Error("Existing wind-direction diagnostics file has an incompatible header; "
                  "choose a new erf.wind_direction_control.diagnostics_file");
        } else {
            std::string line;
            while (std::getline(existing, line)) {
                if (!line.empty()) last_data_line = line;
            }
        }
    }

    if (!last_data_line.empty()) {
        std::istringstream last(last_data_line);
        Real last_time = Real(0.0);
        last >> last_time;
        const Real scale = std::max({Real(1.0), std::abs(time), std::abs(last_time)});
        if (last && std::abs(last_time - time) <=
                        Real(32.0) * std::numeric_limits<Real>::epsilon() * scale) {
            return;
        }
    }

    std::ofstream output(m_diagnostics_file, std::ios::out | std::ios::app);
    if (!output.good()) FileOpenFailed(m_diagnostics_file);
    if (write_header) output << diagnostic_header << '\n';

    constexpr Real radians_to_degrees = Real(180.0) / PI;
    output << std::setprecision(17)
           << time << ','
           << measurements.target.u << ',' << measurements.target.v << ','
           << measurements.target.speed << ','
           << measurements.target.angle * radians_to_degrees << ','
           << m_target_direction * radians_to_degrees << ','
           << measurements.error * radians_to_degrees << ','
           << measurements.geostrophic.u << ',' << measurements.geostrophic.v << ','
           << measurements.geostrophic.speed << ','
           << measurements.geostrophic.angle * radians_to_degrees << ','
           << m_relaxation_gain << ',' << m_rotation_omega3 << ','
           << m_geostrophic_magnitude << ',' << m_effective_ug << ','
           << m_effective_vg << ',' << m_geo_forcing[0] << ','
           << m_geo_forcing[1] << '\n';
}
