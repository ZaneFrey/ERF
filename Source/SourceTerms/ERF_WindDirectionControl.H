#ifndef ERF_WIND_DIRECTION_CONTROL_H_
#define ERF_WIND_DIRECTION_CONTROL_H_

#include <AMReX_GpuContainers.H>
#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>

#include <iosfwd>
#include <string>
#include <vector>

#include "ERF_DataStruct.H"

namespace erf_wind_direction
{

struct EventState {
    bool active{false};
    amrex::Real remaining_angle{0.0};
};

amrex::Real wrap_angle (amrex::Real angle) noexcept;
bool update_correction_demand (bool current, amrex::Real abs_error,
                               amrex::Real tolerance_on,
                               amrex::Real tolerance_off) noexcept;
amrex::Real update_geostrophic_angle (amrex::Real angle, amrex::Real error,
                                      amrex::Real gain,
                                      amrex::Real control_interval) noexcept;
void latch_event (EventState& state, amrex::Real error) noexcept;
amrex::Real event_angle_rate (const EventState& state, amrex::Real rate_magnitude,
                              amrex::Real dt) noexcept;
void advance_event (EventState& state, amrex::Real angle_rate, amrex::Real dt) noexcept;
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
    void end_step (amrex::Real dt) noexcept;

    [[nodiscard]] const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>&
    geostrophic_forcing () const noexcept { return m_geo_forcing; }
    [[nodiscard]] amrex::Real synthetic_coriolis () const noexcept
    { return m_synthetic_coriolis; }

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
    void regenerate_geostrophic_forcing () noexcept;
    void write_diagnostic (amrex::Real time, const Measurement& measurement) const;

    amrex::Real m_target_direction{0.0};
    amrex::Real m_target_height{0.0};
    amrex::Real m_start_time{0.0};
    amrex::Real m_check_interval{60.0};
    amrex::Real m_tolerance_on{0.0};
    amrex::Real m_tolerance_off{0.0};
    amrex::Real m_rotation_rate{0.0};
    amrex::Real m_geostrophic_gain{0.0};
    amrex::Real m_planetary_f{0.0};
    std::string m_diagnostics_file{"wind_direction_control.csv"};
    std::vector<amrex::Real> m_mass_heights;

    amrex::Real m_geostrophic_magnitude{0.0};
    amrex::Real m_geostrophic_angle{0.0};
    erf_wind_direction::EventState m_event;
    amrex::Real m_next_check_time{0.0};
    bool m_correction_demand{false};

    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> m_geo_forcing{0.0, 0.0, 0.0};
    amrex::Real m_synthetic_coriolis{0.0};
    amrex::Real m_step_angle_rate{0.0};
};

#endif
