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

    amrex::Real ctprime () const { return m_ctprime; }
    amrex::Real cpprime () const { return m_cpprime; }
    amrex::Real tsr () const { return m_tsr_classic; }
    amrex::Real memory_time () const { return m_memory_time; }
    amrex::Real rotor_radius () const { return 0.5 * m_diameter; }
    amrex::Real hub_height () const { return m_hub_height_classic; }
    amrex::Real actuator_spacing () const { return m_actuator_spacing; }
    bool spacing_was_supplied () const { return m_spacing_was_supplied; }
    bool wake_rotation_enabled () const { return m_wake_rotation_classic; }

    const amrex::Vector<amrex::Real>& x_locations () const { return m_xloc; }
    const amrex::Vector<amrex::Real>& y_locations () const { return m_yloc; }
    const amrex::Vector<amrex::Real>& ground_elevations () const { return m_zloc; }
    const amrex::Vector<amrex::Real>& disk_face_angles_deg () const { return m_turb_disk_angles; }

    amrex::Vector<ClassicADTurbineState>& turbine_state () { return m_turbine_state; }
    const amrex::Vector<ClassicADTurbineState>& turbine_state () const { return m_turbine_state; }

private:
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
};

#endif
