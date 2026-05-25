#ifndef ERF_SIMPLEAD_H
#define ERF_SIMPLEAD_H

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include "ERF_NullWindFarm.H"

class SimpleAD : public NullWindFarm {

public:

    SimpleAD () {}

    virtual ~SimpleAD () = default;

    void advance (const amrex::Geometry& geom,
                  const amrex::Real& dt_advance,
                  amrex::MultiFab& cons_in,
                  amrex::MultiFab& mf_vars_windfarm,
                  amrex::MultiFab& U_old,
                  amrex::MultiFab& V_old,
                  amrex::MultiFab& W_old,
                  const amrex::MultiFab& mf_Nturb,
                  const amrex::MultiFab& mf_RMask,
                  const amrex::MultiFab& mf_SMark,
                  const amrex::Real& time) override;

    void compute_freestream_velocity (const amrex::MultiFab& cons_in,
                                      const amrex::MultiFab& U_old,
                                      const amrex::MultiFab& V_old,
                                      const amrex::MultiFab& mf_SMark);

    void source_terms_cellcentered (const amrex::Geometry& geom,
                                    const amrex::Real& dt_advance,
                                    const amrex::MultiFab& cons_in,
                                    const amrex::MultiFab& mf_Smark,
                                    const amrex::MultiFab& mf_RMask,
                                    amrex::MultiFab& mf_vars_simpleAD);

    void update (const amrex::Real& dt_advance,
                 amrex::MultiFab& cons_in,
                 amrex::MultiFab& U_old,
                 amrex::MultiFab& V_old,
                 amrex::MultiFab& W_old,
                 const amrex::MultiFab& mf_vars);

    void compute_power_output (const amrex::Real& time);
    void write_memory_state (const std::string& checkpointname) const override;
    bool read_memory_state (const std::string& restart_chkfile) override;

protected:
    amrex::Vector<amrex::Real> xloc, yloc, zloc;
    amrex::Real turb_disk_angle;
    amrex::Real hub_height, rotor_rad, thrust_coeff_standing, nominal_power;
    amrex::Vector<amrex::Real> wind_speed, thrust_coeff, power;
    amrex::Vector<amrex::Real> freestream_velocity, freestream_phi, disk_cell_count;
    amrex::Vector<amrex::Real> filtered_disk_velocity_nhat;
    bool filtered_disk_velocity_initialized{false};
    bool wake_rotation{false};
    amrex::Real tsr{9.0};
    amrex::Real C_P_prime{0.9};
};

#endif
