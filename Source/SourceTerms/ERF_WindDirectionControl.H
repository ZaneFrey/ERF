#ifndef ERF_WIND_DIRECTION_CONTROL_H_
#define ERF_WIND_DIRECTION_CONTROL_H_

#include <AMReX_GpuContainers.H>
#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>

#include <string>
#include <vector>

#include "ERF_DataStruct.H"

namespace erf_wind_direction
{

enum class Phase : int {
    Waiting = 0,
    Idle,
    Rotating,
    Settling
};

struct EpisodeState {
    Phase phase{Phase::Waiting};
    amrex::Real latched_error{0.0};
    amrex::Real event_start_time{0.0};
    amrex::Real remaining_flow_angle{0.0};
    amrex::Real remaining_geo_angle{0.0};
    amrex::Real settling_until{0.0};
};

amrex::Real wrap_angle (amrex::Real angle) noexcept;
bool event_requested (amrex::Real error, amrex::Real threshold) noexcept;
void latch_episode (EpisodeState& state, amrex::Real error,
                    amrex::Real proportional_gain, amrex::Real time) noexcept;
amrex::Real episode_flow_angle_rate (const EpisodeState& state,
                                     amrex::Real rate_magnitude,
                                     amrex::Real dt) noexcept;
amrex::Real episode_geo_increment (const EpisodeState& state,
                                   amrex::Real flow_increment) noexcept;
void advance_episode (EpisodeState& state, amrex::Real flow_increment,
                      amrex::Real geo_increment, amrex::Real completion_time,
                      amrex::Real settling_time) noexcept;
const char* phase_name (Phase phase) noexcept;
amrex::Real interpolate_profile (const std::vector<amrex::Real>& heights,
                                 const amrex::Gpu::HostVector<amrex::Real>& values,
                                 amrex::Real target_height);

} // namespace erf_wind_direction

class WindDirectionControl
{
public:
    static bool enabled_in_inputs ();

    WindDirectionControl (const SolverChoice& choice, int max_level,
                          const amrex::Geometry& geom,
                          const amrex::Vector<amrex::Real>& zlevels_stag);

    void begin_step (amrex::Real time, amrex::Real dt,
                     const amrex::MultiFab& xvel,
                     const amrex::MultiFab& yvel,
                     const amrex::Geometry& geom);
    void end_step (amrex::Real time, amrex::Real dt) noexcept;

    [[nodiscard]] const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>&
    geostrophic_forcing () const noexcept { return m_geo_forcing; }
    [[nodiscard]] amrex::Real rotation_omega3 () const noexcept
    { return m_rotation_omega3; }

    void write_checkpoint (const std::string& checkpoint_name) const;
    void read_checkpoint (const std::string& checkpoint_name,
                          amrex::Real restart_time);
    void update_vertical_grid (const amrex::Geometry& geom,
                               const amrex::Vector<amrex::Real>& zlevels_stag);

private:
    struct Measurement {
        amrex::Real u{0.0};
        amrex::Real v{0.0};
        amrex::Real speed{0.0};
        amrex::Real angle{0.0};
        amrex::Real error{0.0};
        bool valid{false};
    };

    [[nodiscard]] bool control_due (amrex::Real time) const noexcept;
    void advance_check_time (amrex::Real time) noexcept;
    Measurement measure (const amrex::MultiFab& xvel,
                         const amrex::MultiFab& yvel,
                         const amrex::Geometry& geom) const;
    void regenerate_geostrophic_forcing (amrex::Real angle) noexcept;
    void write_diagnostic (amrex::Real time, const Measurement& measurement,
                           bool event_initiated) const;

    amrex::Real m_target_direction{0.0};
    amrex::Real m_target_height{0.0};
    amrex::Real m_start_time{0.0};
    amrex::Real m_check_interval{60.0};
    amrex::Real m_tolerance_on{0.0};
    amrex::Real m_rotation_rate{0.0};
    amrex::Real m_geostrophic_proportional_gain{1.0};
    amrex::Real m_post_event_settling_time{600.0};
    amrex::Real m_planetary_f{0.0};
    std::string m_diagnostics_file{"wind_direction_control.csv"};
    std::vector<amrex::Real> m_mass_heights;

    amrex::Real m_geostrophic_magnitude{0.0};
    amrex::Real m_geostrophic_angle{0.0};
    erf_wind_direction::EpisodeState m_episode;
    amrex::Real m_next_check_time{0.0};

    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> m_geo_forcing{0.0, 0.0, 0.0};
    amrex::Real m_rotation_omega3{0.0};
    amrex::Real m_step_flow_increment{0.0};
    amrex::Real m_step_geo_increment{0.0};
    amrex::Real m_step_midpoint_geo_angle{0.0};
};

#endif
