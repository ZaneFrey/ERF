#ifndef ERF_WINDFARM_H
#define ERF_WINDFARM_H

#include <ERF_DataStruct.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

#include "ERF_NullWindFarm.H"
#include "ERF_Fitch.H"
#include "ERF_EWP.H"
#include "ERF_SimpleAD.H"
#include "ERF_GeneralAD.H"

class WindFarm : public NullWindFarm {

public:

    WindFarm (){}

    virtual ~WindFarm () = default;

    WindFarm (int nlev,
              const WindFarmType& a_windfarm_type)
    {
        m_windfarm_model.resize(nlev);
        if (a_windfarm_type == WindFarmType::Fitch) {
            SetModel<Fitch>();
            amrex::Print() << "Fitch windfarm model!\n";
        }
        else if (a_windfarm_type == WindFarmType::EWP) {
            SetModel<EWP>();
            amrex::Print() << "EWP windfarm model!\n";
        }
        else if (a_windfarm_type == WindFarmType::SimpleAD) {
            SetModel<SimpleAD>();
            amrex::Print() << "Simplified actuator disk windfarm model!\n";
        }
        else if (a_windfarm_type == WindFarmType::GeneralAD) {
            SetModel<GeneralAD>();
            amrex::Print() << "Generalized actuator disk windfarm model!\n";
        }
         else {
            amrex::Abort("WindFarm: Dont know this windfarm_type!") ;
        }
    }

    void read_tables (std::string windfarm_loc_table,
                      std::string windfarm_spec_table,
                      bool x_y, bool lat_lon,
                      const amrex::Real windfarm_x_shift = 0.0,
                      const amrex::Real windfarm_y_shift = 0.0);

    void init_windfarm_lat_lon (const std::string windfarm_loc_table,
                                const amrex::Real windfarm_x_shift,
                                const amrex::Real windfarm_y_shift);

    void init_windfarm_x_y (const std::string windfarm_loc_table);

    void read_windfarm_locations_table (const std::string windfarm_loc_table,
                                        bool x_y, bool lat_lon,
                                        const amrex::Real windfarm_x_shift = 0.0,
                                        const amrex::Real windfarm_y_shift = 0.0);

    void read_windfarm_spec_table (const std::string windfarm_spec_table);

    void read_windfarm_blade_table (const std::string windfarm_blade_table);

    void read_windfarm_airfoil_tables (const std::string windfarm_airfoil_tables,
                                       const std::string windfarm_blade_table);

    void read_windfarm_spec_table_extra (const std::string windfarm_spec_table_extra);

    void fill_Nturb_multifab (const amrex::Geometry& geom,
                              amrex::MultiFab& mf_Nturb,
                              std::unique_ptr<amrex::MultiFab>& z_phys_nd);

    void fill_SMark_multifab (const amrex::Geometry& geom,
                              amrex::MultiFab& mf_SMark,
                              amrex::MultiFab& mf_RMask,
                              const amrex::Real& sampling_distance_by_D,
                              const amrex::Real& turb_disk_angle,
                              std::unique_ptr<amrex::MultiFab>& z_phys_cc);

    void fill_SMark_multifab_mesoscale_models (const amrex::Geometry& geom,
                                          amrex::MultiFab& mf_SMark,
                                          const amrex::MultiFab& mf_Nturb,
                                          std::unique_ptr<amrex::MultiFab>& z_phys_cc);

    void write_turbine_locations_vtk ();

    void write_actuator_disks_vtk (const amrex::Geometry& geom,
                                   const amrex::Real& sampling_distance_by_D);

    // Dynamic per-turbine yaw for SimpleAD
    void init_dynamic_yaw (const amrex::Real disk_angle0_deg,
                           const amrex::Real yaw_period,
                           const amrex::Real windfarm_start_time);

    bool dynamic_yaw_enabled () const { return m_dynamic_yaw_enabled; }

    void set_disk_angle0_deg (const amrex::Real disk_angle0_deg) { m_disk_angle0_deg = disk_angle0_deg; }

    void read_windfarm_yaw_file (const std::string& yaw_file);

    void sample_upstream_uv (const amrex::MultiFab& U_old,
                             const amrex::MultiFab& V_old,
                             const amrex::MultiFab& mf_SMark,
                             amrex::Vector<amrex::Real>& u_sum,
                             amrex::Vector<amrex::Real>& v_sum,
                             amrex::Vector<amrex::Real>& counts) const;

    void accumulate_yaw_samples (const amrex::Vector<amrex::Real>& u_mean,
                                 const amrex::Vector<amrex::Real>& v_mean,
                                 const amrex::Vector<amrex::Real>& counts,
                                 const amrex::Real dt);

    bool update_yaw_controller (const amrex::Real time,
                                const amrex::Real dt);

    bool should_rebuild_SMark (const amrex::Real dx_eff) const;
    void commit_yaw_geometry ();

    void get_disk_face_angles_deg (amrex::Vector<amrex::Real>& disk_face_angles_deg) const;

    void fill_SMark_multifab_dynamic (const amrex::Geometry& geom,
                                      amrex::MultiFab& mf_SMark,
                                      amrex::MultiFab& mf_RMask,
                                      const amrex::Real& sampling_distance_by_D,
                                      const amrex::Vector<amrex::Real>& disk_face_angles_deg,
                                      std::unique_ptr<amrex::MultiFab>& z_phys_cc);

    void write_dynamic_vtk_series (const amrex::Geometry& geom,
                                   const amrex::Real& sampling_distance_by_D,
                                   const amrex::Vector<amrex::Real>& disk_face_angles_deg,
                                   const amrex::Real time);

    void write_yaw_angles_time_series (const amrex::Real time) const;

    void write_dynamic_yaw_state (const std::string& checkpointname) const;
    bool read_dynamic_yaw_state (const std::string& restart_chkfile);

    void advance (const amrex::Geometry& a_geom,
                  const amrex::Real& dt_advance,
                  amrex::MultiFab& cons_in,
                  amrex::MultiFab& mf_vars_windfarm,
                  amrex::MultiFab& U_old,
                  amrex::MultiFab& V_old,
                  amrex::MultiFab& W_old,
                  const amrex::MultiFab& mf_Nturb,
                  const amrex::MultiFab& mf_SMark,
                  const amrex::Real& time) override
    {
        m_windfarm_model[0]->advance(a_geom, dt_advance, cons_in, mf_vars_windfarm,
                                     U_old, V_old, W_old, mf_Nturb, mf_SMark, time);
    }

    void set_turb_spec (const amrex::Real& a_rotor_rad, const amrex::Real& a_hub_height,
                        const amrex::Real& a_thrust_coeff_standing, const amrex::Vector<amrex::Real>& a_wind_speed,
                        const amrex::Vector<amrex::Real>& a_thrust_coeff,
                        const amrex::Vector<amrex::Real>& a_power) override
    {
        m_windfarm_model[0]->set_turb_spec(a_rotor_rad, a_hub_height, a_thrust_coeff_standing,
                  a_wind_speed, a_thrust_coeff, a_power);
    }

    void set_turb_loc (const amrex::Vector<amrex::Real>& a_xloc,
                       const amrex::Vector<amrex::Real>& a_yloc) override
    {
        m_windfarm_model[0]->set_turb_loc(a_xloc, a_yloc);
    }

    void set_turb_disk_angle (const amrex::Real& a_turb_disk_angle) override
    {
        m_windfarm_model[0]->set_turb_disk_angle(a_turb_disk_angle);
    }

    void set_turb_disk_angles (const amrex::Vector<amrex::Real>& a_turb_disk_angles) override
    {
        m_windfarm_model[0]->set_turb_disk_angles(a_turb_disk_angles);
    }

    void set_blade_spec (const amrex::Vector<amrex::Real>& a_bld_rad_loc,
                         const amrex::Vector<amrex::Real>& a_bld_twist,
                         const amrex::Vector<amrex::Real>& a_bld_chord) override
    {
        m_windfarm_model[0]->set_blade_spec(a_bld_rad_loc, a_bld_twist, a_bld_chord);
    }

    void set_blade_airfoil_spec (const amrex::Vector<amrex::Vector<amrex::Real>>& a_bld_airfoil_aoa,
                                 const amrex::Vector<amrex::Vector<amrex::Real>>& a_bld_airfoil_Cl,
                                 const amrex::Vector<amrex::Vector<amrex::Real>>& a_bld_airfoil_Cd) override
    {
        m_windfarm_model[0]->set_blade_airfoil_spec(a_bld_airfoil_aoa, a_bld_airfoil_Cl, a_bld_airfoil_Cd);
    }

    void set_turb_spec_extra (const amrex::Vector<amrex::Real>& a_velocity,
                              const amrex::Vector<amrex::Real>& a_C_P,
                              const amrex::Vector<amrex::Real>& a_C_T,
                              const amrex::Vector<amrex::Real>& a_rotor_RPM,
                              const amrex::Vector<amrex::Real>& a_blade_pitch) override
    {
        m_windfarm_model[0]->set_turb_spec_extra(a_velocity, a_C_P, a_C_T, a_rotor_RPM, a_blade_pitch);
    }

protected:

    amrex::Vector<amrex::Real> xloc, yloc, zloc;
    amrex::Real my_turb_disk_angle;
    amrex::Real hub_height, rotor_rad, thrust_coeff_standing, nominal_power;
    amrex::Vector<amrex::Real> wind_speed, thrust_coeff, power;
    amrex::Vector<amrex::Real> bld_rad_loc, bld_twist, bld_chord;
    amrex::Vector<amrex::Vector<amrex::Real>> bld_airfoil_aoa, bld_airfoil_Cl, bld_airfoil_Cd;
    int n_bld_sections;
    amrex::Vector<amrex::Real> velocity, C_P, C_T, rotor_RPM, blade_pitch;

    // Dynamic yaw state (SimpleAD)
    bool m_dynamic_yaw_enabled{false};
    amrex::Real m_disk_angle0_deg{0.0};
    amrex::Real m_yaw_period{0.0};
    amrex::Real m_tau_yaw{0.0};
    amrex::Real m_yaw_alpha{0.5};
    int m_yaw_update_idx{0};

    amrex::Vector<amrex::Real> m_yaw_angle_deg;
    amrex::Vector<amrex::Real> m_yaw_cmd_deg;
    amrex::Vector<amrex::Real> m_next_update_time;
    amrex::Vector<amrex::Real> m_sum_u, m_sum_v, m_sum_t;
    amrex::Vector<amrex::Real> m_yaw_angle_geom_deg;
    amrex::Vector<amrex::Real> m_yaw_offset_deg;

   /*! \brief Create and set the specified windfarm model */
    template<class NewWindFarmModel>
    void SetModel ()
    {
        for (int lev(0); lev<m_windfarm_model.size(); ++lev) {
            m_windfarm_model[lev] = std::make_unique<NewWindFarmModel>();
        }
    }

private:
    amrex::Vector<std::unique_ptr<NullWindFarm>> m_windfarm_model; /*!< windfarm model */
};

#endif
