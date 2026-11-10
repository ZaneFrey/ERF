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

amrex::Real wrap_angle (amrex::Real angle) noexcept;
amrex::Real continuous_rotation_rate (amrex::Real error,
                                      amrex::Real relaxation_gain) noexcept;
amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>
geostrophic_forcing (amrex::Real planetary_f, amrex::Real magnitude,
                     amrex::Real angle) noexcept;
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

    void update (amrex::Real time, const amrex::MultiFab& xvel,
                 const amrex::MultiFab& yvel,
                 const amrex::Geometry& geom);

    [[nodiscard]] const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>&
    geostrophic_forcing () const noexcept { return m_geo_forcing; }
    [[nodiscard]] amrex::Real rotation_omega3 () const noexcept
    { return m_rotation_omega3; }

    void update_vertical_grid (const amrex::Geometry& geom,
                               const amrex::Vector<amrex::Real>& zlevels_stag);

private:
    struct Measurement {
        amrex::Real u{0.0};
        amrex::Real v{0.0};
        amrex::Real speed{0.0};
        amrex::Real angle{0.0};
    };

    struct Measurements {
        Measurement target;
        Measurement geostrophic;
        amrex::Real error{0.0};
    };

    [[nodiscard]] bool diagnostic_due (amrex::Real time) const noexcept;
    void advance_diagnostic_time (amrex::Real time) noexcept;
    Measurements measure (const amrex::MultiFab& xvel,
                          const amrex::MultiFab& yvel,
                          const amrex::Geometry& geom) const;
    void regenerate_geostrophic_forcing (amrex::Real angle) noexcept;
    void write_diagnostic (amrex::Real time,
                           const Measurements& measurements) const;

    amrex::Real m_target_direction{0.0};
    amrex::Real m_target_height{0.0};
    amrex::Real m_geostrophic_height{0.0};
    amrex::Real m_relaxation_gain{1.0e-4};
    amrex::Real m_diagnostics_interval{60.0};
    amrex::Real m_next_diagnostic_time{0.0};
    amrex::Real m_planetary_f{0.0};
    std::string m_diagnostics_file{"wind_direction_control.csv"};
    std::vector<amrex::Real> m_mass_heights;

    amrex::Real m_geostrophic_magnitude{0.0};
    amrex::Real m_geostrophic_angle{0.0};
    amrex::Real m_effective_ug{0.0};
    amrex::Real m_effective_vg{0.0};
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> m_geo_forcing{0.0, 0.0, 0.0};
    amrex::Real m_rotation_omega3{0.0};
};

#endif
