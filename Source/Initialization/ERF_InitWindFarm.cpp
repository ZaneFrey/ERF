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

// Wind farm initialization entry point.

/**
 * @brief Load immutable wind-farm tables and initialize per-turbine state once.
 */
void
ERF::initialize_windfarm_catalog ()
{
    if (m_windfarm_catalog_initialized) {
        return;
    }

    const bool is_classic_ad = (solverChoice.windfarm_type == WindFarmType::ClassicAD);

    if(solverChoice.windfarm_loc_type == WindFarmLocType::lat_lon) {
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
    } else if(solverChoice.windfarm_loc_type == WindFarmLocType::x_y) {
        if (is_classic_ad) {
            const Real xshift = (solverChoice.windfarm_x_shift == -one)
                ? zero : solverChoice.windfarm_x_shift;
            const Real yshift = (solverChoice.windfarm_y_shift == -one)
                ? zero : solverChoice.windfarm_y_shift;
            windfarm->read_windfarm_locations_table(solverChoice.windfarm_loc_table,
                                                    true, false, xshift, yshift);
        } else {
            windfarm->read_tables(solverChoice.windfarm_loc_table,
                                  solverChoice.windfarm_spec_table,
                                  true, false);
        }
    }

    if (is_classic_ad) {
#ifdef ERF_USE_PARTICLES
        if (solverChoice.dynamic_yaw) {
            windfarm->classic_ad_model().initialize_dynamic_yaw(
                solverChoice.turb_disk_angle,
                solverChoice.classic_ad_yaw_sensor_distance_by_D,
                solverChoice.classic_ad_yaw_sensor_mem_time,
                solverChoice.windfarm_start_time);
        }
        windfarm->classic_ad_model().initialize_turbine_state(windfarm->num_turbines());
#endif
    }

    if(solverChoice.windfarm_type == WindFarmType::GeneralAD) {
        windfarm->read_windfarm_blade_table(solverChoice.windfarm_blade_table);
        windfarm->read_windfarm_airfoil_tables(solverChoice.windfarm_airfoil_tables,
                                               solverChoice.windfarm_blade_table);
        windfarm->read_windfarm_spec_table_extra(solverChoice.windfarm_spec_table_extra);
    }

    m_windfarm_catalog_initialized = true;
}

/**
 * @brief Recompute wind-turbine ownership for the active AMR hierarchy.
 */
void
ERF::rebuild_windfarm_hierarchy ()
{
    initialize_windfarm_catalog();
    m_windfarm_hierarchy_initialized = false;

    if (solverChoice.windfarm_type == WindFarmType::ClassicAD) {
        Vector<IntVect> active_ref_ratio;
        active_ref_ratio.reserve(finest_level);
        for (int lev = 0; lev < finest_level; ++lev) {
            active_ref_ratio.push_back(refRatio(lev));
        }

        windfarm->define_classic_ad_owner_levels(
            finest_level, geom, grids, active_ref_ratio,
            solverChoice.turb_disk_angle);

#ifdef ERF_USE_PARTICLES
        if (!classic_ad_pc) {
            classic_ad_pc = std::make_unique<ClassicADPC>(
                static_cast<ParGDBBase*>(GetParGDB()),
                windfarm->classic_ad_model());
        }

        Vector<int> owner_levels(windfarm->num_turbines());
        for (int turbine_id = 0; turbine_id < windfarm->num_turbines(); ++turbine_id) {
            owner_levels[turbine_id] = windfarm->owner_level(turbine_id);
        }
        classic_ad_pc->rebuild(owner_levels);

        if (solverChoice.dynamic_yaw) {
            if (!classic_ad_sensor_pc) {
                classic_ad_sensor_pc = std::make_unique<ClassicADSensorPC>(
                    static_cast<ParGDBBase*>(GetParGDB()),
                    windfarm->classic_ad_model());
            }
            classic_ad_sensor_pc->rebuild();
        }
#endif
    }

    m_windfarm_hierarchy_initialized = true;
}

/**
 * @brief Initialize level-dependent wind farm fields.
 * @param lev Level to initialize.
 */
void
ERF::init_windfarm (int lev)
{
    initialize_windfarm_catalog();

    if (solverChoice.windfarm_type == WindFarmType::ClassicAD) {
        return;
    }

    windfarm->fill_Nturb_multifab(geom[lev], Nturb[lev], z_phys_nd[lev]);

    windfarm->write_turbine_locations_vtk();


    if(solverChoice.windfarm_type == WindFarmType::Fitch or
       solverChoice.windfarm_type == WindFarmType::EWP) {
        windfarm->fill_SMark_multifab_mesoscale_models(geom[lev],
                                                       SMark[lev],
                                                       Nturb[lev],
                                                       z_phys_nd[lev]);
    }

    if(solverChoice.windfarm_type == WindFarmType::SimpleAD or
       solverChoice.windfarm_type == WindFarmType::GeneralAD) {
        windfarm->fill_SMark_multifab(geom[lev], SMark[lev],
                                      solverChoice.sampling_distance_by_D,
                                      solverChoice.turb_disk_angle,
                                      z_phys_cc[lev]);
        windfarm->write_actuator_disks_vtk(geom[lev],
                                           solverChoice.sampling_distance_by_D);
    }

}

/**
 * Advance the wind farm model and apply its source terms.
 *
 * @param a_geom Geometry for the current level
 * @param dt_advance Timestep over which to advance the wind farm model
 * @param cons_in Conserved state receiving wind farm tendencies
 * @param U_old x-velocity state used by the wind farm model
 * @param V_old y-velocity state used by the wind farm model
 * @param W_old z-velocity state used by the wind farm model
 * @param mf_vars_windfarm Wind farm work and diagnostic variables
 * @param mf_Nturb MultiFab storing turbine counts
 * @param mf_SMark MultiFab storing source-marker data
 * @param time Current simulation time
 */
void
ERF::advance_windfarm (const Geometry& a_geom,
                       const double& dt_advance,
                       MultiFab& cons_in,
                       MultiFab& U_old,
                       MultiFab& V_old,
                       MultiFab& W_old,
                       MultiFab& mf_vars_windfarm,
                       const MultiFab& mf_Nturb,
                       const MultiFab& mf_SMark,
                       const double& time)
{
        windfarm->advance(a_geom, dt_advance, cons_in, mf_vars_windfarm,
                          U_old, V_old, W_old, mf_Nturb, mf_SMark, time);
}
