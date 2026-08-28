#ifndef ERF_CLASSICAD_H
#define ERF_CLASSICAD_H

#include <ERF_NullWindFarm.H>

struct ClassicADTurbineState
{
    amrex::Real disk_velocity_raw{0.0};
    amrex::Real disk_velocity_filtered{0.0};
    amrex::Real disk_density{0.0};
    amrex::Real thrust_target{0.0};
    amrex::Real thrust_applied{0.0};
    amrex::Real axial_power{0.0};
    amrex::Real rotor_power{0.0};
    amrex::Real omega{0.0};
    amrex::Real torque_target{0.0};
    amrex::Real torque_applied{0.0};
    amrex::Real les_torque{0.0};
    amrex::Real actuator_area{0.0};
    bool memory_initialized{false};
};

struct ClassicADYawState
{
    amrex::Real psi_ref{0.0};
    amrex::Real psi_rotor{0.0};
    amrex::Real phi_cmd{0.0};
    amrex::Real sensor_u_raw{0.0};
    amrex::Real sensor_v_raw{0.0};
    amrex::Real sensor_u_filtered{0.0};
    amrex::Real sensor_v_filtered{0.0};
    bool sensor_initialized{false};
};

class ClassicAD : public NullWindFarm
{
public:
    ClassicAD () = default;

    void advance (const amrex::Geometry&,
                  const amrex::Real&,
                  amrex::MultiFab&,
                  amrex::MultiFab&,
                  amrex::MultiFab&,
                  amrex::MultiFab&,
                  amrex::MultiFab&,
                  const amrex::MultiFab&,
                  const amrex::MultiFab&,
                  const amrex::MultiFab&,
                  const amrex::Real&) override
    {
        // ClassicAD is coupled through persistent momentum-source fields, never
        // through the legacy actuator-disk velocity-kick interface.
    }

    void configure (amrex::Real ctprime,
                    amrex::Real cpprime,
                    amrex::Real tsr,
                    bool wake_rotation,
                    amrex::Real memory_time,
                    amrex::Real diameter,
                    amrex::Real hub_height,
                    amrex::Real spacing,
                    bool spacing_was_supplied);

    void initialize_turbine_state (int nturb);
    void write_diagnostics (amrex::Real time) const;
    void write_memory_state (const std::string& checkpointname) const override;
    bool read_memory_state (const std::string& restart_chkfile) override;

    void initialize_dynamic_yaw (amrex::Real disk_face_angle_deg,
                                 const amrex::Vector<amrex::Real>& yaw_offsets_deg,
                                 amrex::Real sensor_distance_by_D,
                                 amrex::Real sensor_memory_time,
                                 amrex::Real yaw_out_per,
                                 int yaw_out_int,
                                 bool use_yaw_out_int,
                                 amrex::Real windfarm_start_time);
    void update_dynamic_yaw (const amrex::Vector<amrex::Real>& sensor_u_raw,
                             const amrex::Vector<amrex::Real>& sensor_v_raw,
                             amrex::Real dt);
    bool yaw_output_due (amrex::Real step_end_time);
    void write_yaw_diagnostics (amrex::Real time) const;
    void write_yaw_state (const std::string& checkpointname) const;
    bool read_yaw_state (const std::string& restart_chkfile);

    amrex::Real ctprime () const { return m_ctprime; }
    amrex::Real cpprime () const { return m_cpprime; }
    amrex::Real tsr () const { return m_tsr_classic; }
    amrex::Real memory_time () const { return m_memory_time; }
    amrex::Real rotor_radius () const { return 0.5 * m_diameter; }
    amrex::Real hub_height () const { return m_hub_height_classic; }
    amrex::Real actuator_spacing () const { return m_actuator_spacing; }
    bool spacing_was_supplied () const { return m_spacing_was_supplied; }
    bool wake_rotation_enabled () const { return m_wake_rotation_classic; }
    bool dynamic_yaw_enabled () const { return m_dynamic_yaw_enabled; }
    amrex::Real yaw_sensor_distance_by_D () const { return m_yaw_sensor_distance_by_D; }

    const amrex::Vector<amrex::Real>& x_locations () const { return m_xloc; }
    const amrex::Vector<amrex::Real>& y_locations () const { return m_yloc; }
    const amrex::Vector<amrex::Real>& ground_elevations () const { return m_zloc; }
    const amrex::Vector<amrex::Real>& disk_face_angles_deg () const { return m_turb_disk_angles; }
    const amrex::Vector<amrex::Real>& yaw_offsets_rad () const { return m_yaw_offsets_rad; }
    const amrex::Vector<ClassicADYawState>& yaw_state () const { return m_yaw_state; }

    amrex::Vector<ClassicADTurbineState>& turbine_state () { return m_turbine_state; }
    const amrex::Vector<ClassicADTurbineState>& turbine_state () const { return m_turbine_state; }

private:
    void synchronize_rotor_angles ();

    amrex::Real m_ctprime{4.0/3.0};
    amrex::Real m_cpprime{0.8};
    amrex::Real m_tsr_classic{8.0};
    amrex::Real m_memory_time{600.0};
    amrex::Real m_diameter{-1.0};
    amrex::Real m_hub_height_classic{-1.0};
    amrex::Real m_actuator_spacing{-1.0};
    bool m_spacing_was_supplied{false};
    bool m_wake_rotation_classic{false};
    amrex::Vector<ClassicADTurbineState> m_turbine_state;
    bool m_dynamic_yaw_enabled{false};
    amrex::Real m_yaw_sensor_distance_by_D{1.0};
    amrex::Real m_yaw_sensor_memory_time{60.0};
    amrex::Real m_yaw_out_per{60.0};
    int m_yaw_out_int{-1};
    bool m_use_yaw_out_int{false};
    amrex::Real m_next_yaw_output_time{0.0};
    long m_yaw_active_step_count{0};
    amrex::Vector<amrex::Real> m_yaw_offsets_rad;
    amrex::Vector<ClassicADYawState> m_yaw_state;
};

#endif
