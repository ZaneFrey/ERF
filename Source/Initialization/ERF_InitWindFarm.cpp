/**
 * \file ERF_InitWindFarm.cpp
 */
#include <ERF.H>

using namespace amrex;

/**
 * Read in the turbine locations in latitude-longitude from windturbines.txt
 * and convert it into x and y coordinates in metres
 *
 * @param lev Integer specifying the current level
 */

// Explicit instantiation

void
ERF::init_windfarm (int lev)
{
    const bool is_ad_model = (solverChoice.windfarm_type == WindFarmType::SimpleAD ||
                              solverChoice.windfarm_type == WindFarmType::GeneralAD);
    const bool defer_ad_outputs = is_ad_model && (solverChoice.windfarm_start_time > 0.0);

    windfarm->set_wake_rotation_params(solverChoice.wake_rotation, 9.0, 0.9);

    if(solverChoice.windfarm_loc_type == WindFarmLocType::lat_lon) {
        windfarm->read_tables(solverChoice.windfarm_loc_table,
                              solverChoice.windfarm_spec_table,
                              false, true,
                              solverChoice.windfarm_x_shift,
                              solverChoice.windfarm_y_shift);
    } else if(solverChoice.windfarm_loc_type == WindFarmLocType::x_y) {
        windfarm->read_tables(solverChoice.windfarm_loc_table,
                             solverChoice.windfarm_spec_table,
                             true, false);
    }

    if (solverChoice.windfarm_type == WindFarmType::SimpleAD ||
        solverChoice.windfarm_type == WindFarmType::GeneralAD) {
        windfarm->set_disk_angle0_deg(solverChoice.turb_disk_angle);
        windfarm->read_windfarm_yaw_file(solverChoice.yaw_file);
    }

    windfarm->fill_Nturb_multifab(geom[lev], Nturb[lev], z_phys_nd[lev]);

    if (!(solverChoice.dynamic_yaw && solverChoice.windfarm_type == WindFarmType::SimpleAD) &&
        !defer_ad_outputs) {
        windfarm->write_turbine_locations_vtk();
    }


    if(solverChoice.windfarm_type == WindFarmType::Fitch or
       solverChoice.windfarm_type == WindFarmType::EWP) {
        windfarm->fill_SMark_multifab_mesoscale_models(geom[lev],
                                                       SMark[lev],
                                                       Nturb[lev],
                                                       z_phys_nd[lev]);
    }

    if(solverChoice.windfarm_type == WindFarmType::SimpleAD or
       solverChoice.windfarm_type == WindFarmType::GeneralAD) {
        if ((solverChoice.dynamic_yaw && solverChoice.windfarm_type == WindFarmType::SimpleAD) ||
            !solverChoice.yaw_file.empty()) {
            amrex::Vector<amrex::Real> disk_face_angles_deg;
            windfarm->get_disk_face_angles_deg(disk_face_angles_deg);
            windfarm->fill_SMark_multifab_dynamic(geom[lev], SMark[lev], RMask[lev],
                                                  solverChoice.sampling_distance_by_D,
                                                  disk_face_angles_deg,
                                                  z_phys_cc[lev]);
        } else {
            windfarm->fill_SMark_multifab(geom[lev], SMark[lev], RMask[lev],
                                          solverChoice.sampling_distance_by_D,
                                          solverChoice.turb_disk_angle,
                                          z_phys_cc[lev]);
        }
        if (!(solverChoice.dynamic_yaw && solverChoice.windfarm_type == WindFarmType::SimpleAD) &&
            !defer_ad_outputs) {
            windfarm->write_actuator_disks_vtk(geom[lev],
                                               solverChoice.sampling_distance_by_D);
        }
    }

    if (lev == 0 && solverChoice.dynamic_yaw && solverChoice.windfarm_type == WindFarmType::SimpleAD) {
        windfarm->init_dynamic_yaw(solverChoice.turb_disk_angle,
                                   solverChoice.yaw_period,
                                   solverChoice.windfarm_start_time);
    }

    if(solverChoice.windfarm_type == WindFarmType::GeneralAD) {
        windfarm->read_windfarm_blade_table(solverChoice.windfarm_blade_table);
        windfarm->read_windfarm_airfoil_tables(solverChoice.windfarm_airfoil_tables,
                                               solverChoice.windfarm_blade_table);
        windfarm->read_windfarm_spec_table_extra(solverChoice.windfarm_spec_table_extra);
    }
}

void
ERF::advance_windfarm (const Geometry& a_geom,
                       const Real& dt_advance,
                       MultiFab& cons_in,
                       MultiFab& U_old,
                       MultiFab& V_old,
                       MultiFab& W_old,
                       MultiFab& mf_vars_windfarm,
                       const MultiFab& mf_Nturb,
                       const MultiFab& mf_RMask,
                       const MultiFab& mf_SMark,
                       const Real& time)
{
        windfarm->advance(a_geom, dt_advance, cons_in, mf_vars_windfarm,
                          U_old, V_old, W_old, mf_Nturb, mf_RMask, mf_SMark, time);
}
