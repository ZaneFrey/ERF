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
ERF::initialize_windfarm_catalog ()
{
    if (m_windfarm_catalog_initialized) {
        return;
    }

    const bool is_legacy_ad = (solverChoice.windfarm_type == WindFarmType::SimpleAD ||
                               solverChoice.windfarm_type == WindFarmType::GeneralAD);
    const bool is_classic_ad = (solverChoice.windfarm_type == WindFarmType::ClassicAD);

    windfarm->set_wake_rotation_params(solverChoice.wake_rotation, 9.0, 0.9);
    windfarm->set_turb_mem_time(solverChoice.turb_mem_time);
    windfarm->set_force_spreading(solverChoice.windfarm_force_spreading,
                                  solverChoice.windfarm_spreading_nsigma);

#ifdef ERF_USE_PARTICLES
    if (is_classic_ad) {
        windfarm->configure_classic_ad(solverChoice.classic_ad_ctprime,
                                       solverChoice.classic_ad_cpprime,
                                       solverChoice.classic_ad_tsr,
                                       solverChoice.classic_ad_wake_rotation,
                                       solverChoice.classic_ad_turb_mem_time,
                                       solverChoice.classic_ad_diameter,
                                       solverChoice.classic_ad_hub_height,
                                       solverChoice.classic_ad_actuator_spacing,
                                       solverChoice.classic_ad_spacing_was_supplied);
    }
#endif

    if (solverChoice.windfarm_loc_type == WindFarmLocType::lat_lon) {
        if (is_classic_ad) {
            windfarm->read_windfarm_locations_table(solverChoice.windfarm_loc_table,
                                                    false, true,
                                                    solverChoice.windfarm_x_shift,
                                                    solverChoice.windfarm_y_shift);
        } else {
            windfarm->read_tables(solverChoice.windfarm_loc_table,
                                  solverChoice.windfarm_spec_table,
                                  false, true,
                                  solverChoice.windfarm_x_shift,
                                  solverChoice.windfarm_y_shift);
        }
    } else if (solverChoice.windfarm_loc_type == WindFarmLocType::x_y) {
        const Real xshift = (solverChoice.windfarm_x_shift == -1.0) ? 0.0 : solverChoice.windfarm_x_shift;
        const Real yshift = (solverChoice.windfarm_y_shift == -1.0) ? 0.0 : solverChoice.windfarm_y_shift;
        if (is_classic_ad) {
            windfarm->read_windfarm_locations_table(solverChoice.windfarm_loc_table,
                                                    true, false, xshift, yshift);
        } else {
            windfarm->read_tables(solverChoice.windfarm_loc_table,
                                  solverChoice.windfarm_spec_table,
                                  true, false);
        }
    }

    if (is_legacy_ad || is_classic_ad) {
        windfarm->set_disk_angle0_deg(solverChoice.turb_disk_angle);
        windfarm->read_windfarm_yaw_file(solverChoice.yaw_file);
    }

    if (solverChoice.dynamic_yaw && solverChoice.windfarm_type == WindFarmType::SimpleAD) {
        windfarm->init_dynamic_yaw(solverChoice.turb_disk_angle,
                                   solverChoice.yaw_period,
                                   solverChoice.windfarm_start_time);
    }
#ifdef ERF_USE_PARTICLES
    if (solverChoice.dynamic_yaw && is_classic_ad) {
        windfarm->init_classic_ad_dynamic_yaw(
            solverChoice.turb_disk_angle,
            solverChoice.classic_ad_yaw_sensor_distance_by_D,
            solverChoice.classic_ad_yaw_sensor_mem_time,
            solverChoice.classic_ad_yaw_out_per,
            solverChoice.classic_ad_yaw_out_int,
            solverChoice.classic_ad_yaw_out_int_was_supplied,
            solverChoice.windfarm_start_time);
    }
#endif

    if (solverChoice.windfarm_type == WindFarmType::GeneralAD) {
        windfarm->read_windfarm_blade_table(solverChoice.windfarm_blade_table);
        windfarm->read_windfarm_airfoil_tables(solverChoice.windfarm_airfoil_tables,
                                               solverChoice.windfarm_blade_table);
        windfarm->read_windfarm_spec_table_extra(solverChoice.windfarm_spec_table_extra);
    }

    m_windfarm_catalog_initialized = true;
}

void
ERF::rebuild_windfarm_hierarchy ()
{
    initialize_windfarm_catalog();
    m_windfarm_hierarchy_initialized = false;

    const bool is_legacy_ad = (solverChoice.windfarm_type == WindFarmType::SimpleAD ||
                               solverChoice.windfarm_type == WindFarmType::GeneralAD);
    const bool is_classic_ad = (solverChoice.windfarm_type == WindFarmType::ClassicAD);
    const bool use_per_turbine_angles =
        ((solverChoice.dynamic_yaw && solverChoice.windfarm_type == WindFarmType::SimpleAD) ||
         !solverChoice.yaw_file.empty());
    const bool defer_ad_outputs = is_legacy_ad && (solverChoice.windfarm_start_time > 0.0);

    amrex::Vector<int> all_turbines(windfarm->num_turbines());
    for (int it = 0; it < windfarm->num_turbines(); ++it) {
        all_turbines[it] = it;
    }

    if (is_classic_ad) {
        windfarm->define_classic_ad_owner_levels(finest_level, geom, grids, ref_ratio);
#ifdef ERF_USE_PARTICLES
        if (!classic_ad_pc) {
            classic_ad_pc = std::make_unique<ClassicADPC>(
                static_cast<ParGDBBase*>(GetParGDB()), windfarm->classic_ad_model());
        }
        Vector<int> owners(windfarm->num_turbines());
        for (int it = 0; it < windfarm->num_turbines(); ++it) {
            owners[it] = windfarm->owner_level(it);
        }
        classic_ad_pc->rebuild(owners);
        if (solverChoice.dynamic_yaw) {
            if (!classic_ad_sensor_pc) {
                classic_ad_sensor_pc = std::make_unique<ClassicADSensorPC>(
                    static_cast<ParGDBBase*>(GetParGDB()), windfarm->classic_ad_model());
            }
            classic_ad_sensor_pc->rebuild();
        }
#endif
    } else if (is_legacy_ad) {
        windfarm->define_owner_levels(geom, grids, dmap, ref_ratio, z_phys_nd);
    }

    amrex::Vector<amrex::Real> disk_face_angles_deg;
    if (use_per_turbine_angles) {
        windfarm->get_disk_face_angles_deg(disk_face_angles_deg);
    }

    for (int lev = 0; lev <= finest_level; ++lev) {
        const amrex::Vector<int>& turbine_ids =
            (is_legacy_ad || is_classic_ad) ? windfarm->turbines_on_level(lev) : all_turbines;

        if (!is_classic_ad) {
            windfarm->fill_Nturb_multifab(geom[lev], Nturb[lev], z_phys_nd[lev], turbine_ids);
        }

        if (solverChoice.windfarm_type == WindFarmType::Fitch ||
            solverChoice.windfarm_type == WindFarmType::EWP) {
            windfarm->fill_SMark_multifab_mesoscale_models(geom[lev],
                                                           SMark[lev],
                                                           Nturb[lev],
                                                           z_phys_nd[lev]);
        } else if (is_legacy_ad) {
            if (use_per_turbine_angles) {
                windfarm->fill_SMark_multifab_dynamic(geom[lev], SMark[lev], RMask[lev],
                                                      solverChoice.sampling_distance_by_D,
                                                      disk_face_angles_deg,
                                                      z_phys_cc[lev],
                                                      turbine_ids);
            } else {
                windfarm->fill_SMark_multifab(geom[lev], SMark[lev], RMask[lev],
                                              solverChoice.sampling_distance_by_D,
                                              solverChoice.turb_disk_angle,
                                              z_phys_cc[lev],
                                              turbine_ids);
            }
        }
    }

    if (solverChoice.dynamic_yaw && solverChoice.windfarm_type == WindFarmType::SimpleAD) {
        for (int lev = 0; lev <= finest_level; ++lev) {
            windfarm->commit_yaw_geometry(windfarm->turbines_on_level(lev));
        }
    }

    if (is_legacy_ad && !solverChoice.dynamic_yaw &&
        !defer_ad_outputs && !m_windfarm_outputs_written) {
        windfarm->write_turbine_locations_vtk();
        windfarm->write_actuator_disks_vtk(Geom(0), solverChoice.sampling_distance_by_D);
        m_windfarm_outputs_written = true;
    }

    m_windfarm_hierarchy_initialized = true;
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
