#include <ERF.H>
#include <ERF_Derive.H>
#include <AMReX_BoxList.H>
#include <AMReX_Gpu.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

using namespace amrex;

#ifdef ERF_USE_NETCDF
Box read_subdomain_from_wrfinput (int lev, const std::string& fname, int& ratio);
Real read_start_time_from_wrfinput (int lev, const std::string& fname);
Box read_subdomain_from_metgrid (int lev, const std::string& fname, int& ratio, int& klo, int& khi);
#endif

void
tag_on_distance_from_eye(const Geometry& cgeom, TagBoxArray* tags,
                         const Real eye_x, const Real eye_y, const Real rad_tag);

namespace {

bool
using_pbl_model (const SolverChoice& solver_choice, int lev)
{
    return (solver_choice.turbChoice[lev].pbl_type == PBLType::MYJ      ||
            solver_choice.turbChoice[lev].pbl_type == PBLType::MYNN25   ||
            solver_choice.turbChoice[lev].pbl_type == PBLType::MYNNEDMF ||
            solver_choice.turbChoice[lev].pbl_type == PBLType::YSU      ||
            solver_choice.turbChoice[lev].pbl_type == PBLType::MRF);
}

bool
any_pbl_models_active (const SolverChoice& solver_choice, int finest_level)
{
    for (int lev = 0; lev <= finest_level; ++lev) {
        if (using_pbl_model(solver_choice, lev)) {
            return true;
        }
    }
    return false;
}

amrex::BoxArray
build_uncovered_valid_boxes (const amrex::Box& domain,
                             const amrex::BoxArray& level_grids,
                             const amrex::BoxArray* finer_grids,
                             const amrex::IntVect* ref_ratio)
{
    amrex::BoxList uncovered(level_grids);

    if (finer_grids != nullptr && ref_ratio != nullptr) {
        amrex::BoxArray covered_by_fine(*finer_grids);
        covered_by_fine.coarsen(*ref_ratio);

        amrex::BoxList complement;
        complement.complementIn(domain, covered_by_fine);
        uncovered.intersect(complement);
        uncovered.removeEmpty();
        uncovered.simplify();
    }

    return amrex::BoxArray(std::move(uncovered));
}

AMREX_FORCE_INLINE
amrex::Real missing_z_value ()
{
    return -std::numeric_limits<amrex::Real>::max() / 4.0;
}

AMREX_FORCE_INLINE
bool is_missing_z_value (amrex::Real value)
{
    return value <= missing_z_value() / 2.0;
}

amrex::Vector<amrex::Vector<amrex::Real>>
gather_z_columns_to_host (const amrex::Geometry& geom,
                          const amrex::MultiFab* z_phys_nd,
                          const amrex::Vector<int>& i_cols,
                          const amrex::Vector<int>& j_cols)
{
    const int ncol = static_cast<int>(i_cols.size());
    const int klo = geom.Domain().smallEnd(2);
    const int khi = geom.Domain().bigEnd(2) + 1;
    const int nnode = khi - klo + 1;
    const amrex::Real missing = missing_z_value();

    amrex::Vector<amrex::Vector<amrex::Real>> z_columns(ncol,
        amrex::Vector<amrex::Real>(nnode, missing));

    if (ncol == 0) {
        return z_columns;
    }

    if (z_phys_nd == nullptr) {
        const auto prob_lo = geom.ProbLoArray();
        const auto dx = geom.CellSizeArray();
        for (int idx = 0; idx < ncol; ++idx) {
            for (int k = klo; k <= khi; ++k) {
                z_columns[idx][k-klo] = prob_lo[2] + (k-klo) * dx[2];
            }
        }
        return z_columns;
    }

    for (amrex::MFIter mfi(*z_phys_nd, false); mfi.isValid(); ++mfi) {
        const amrex::Box& vb = mfi.validbox();
        const amrex::FArrayBox& fab = (*z_phys_nd)[mfi];
        amrex::Array4<const amrex::Real> z_arr = fab.const_array();

#ifdef AMREX_USE_GPU
        std::unique_ptr<amrex::FArrayBox> hostfab;
        if (fab.arena()->isManaged() || fab.arena()->isDevice()) {
            hostfab = std::make_unique<amrex::FArrayBox>(fab.box(), fab.nComp(), amrex::The_Pinned_Arena());
            amrex::Gpu::dtoh_memcpy_async(hostfab->dataPtr(), fab.dataPtr(),
                                          fab.size()*sizeof(amrex::Real));
            amrex::Gpu::streamSynchronize();
            z_arr = hostfab->const_array();
        }
#endif

        const int kbeg = std::max(vb.smallEnd(2), klo);
        const int kend = std::min(vb.bigEnd(2), khi);
        for (int idx = 0; idx < ncol; ++idx) {
            const int i = i_cols[idx];
            const int j = j_cols[idx];
            if (i < vb.smallEnd(0) || i > vb.bigEnd(0) ||
                j < vb.smallEnd(1) || j > vb.bigEnd(1)) {
                continue;
            }

            for (int k = kbeg; k <= kend; ++k) {
                z_columns[idx][k-klo] = z_arr(i, j, k);
            }
        }
    }

    return z_columns;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool intervals_overlap_inclusive (amrex::Real alo, amrex::Real ahi,
                                  amrex::Real blo, amrex::Real bhi)
{
    return !(ahi < blo || bhi < alo);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool oriented_rect_overlaps_cell_2d (amrex::Real cx, amrex::Real cy,
                                     amrex::Real nx, amrex::Real ny,
                                     amrex::Real tx, amrex::Real ty,
                                     amrex::Real half_streamwise,
                                     amrex::Real half_spanwise,
                                     amrex::Real xlo, amrex::Real xhi,
                                     amrex::Real ylo, amrex::Real yhi)
{
    const amrex::Real cell_cx = amrex::Real(0.5) * (xlo + xhi);
    const amrex::Real cell_cy = amrex::Real(0.5) * (ylo + yhi);
    const amrex::Real hx = amrex::Real(0.5) * (xhi - xlo);
    const amrex::Real hy = amrex::Real(0.5) * (yhi - ylo);

    const amrex::Real rect_rx = amrex::Math::abs(nx) * half_streamwise +
                                amrex::Math::abs(tx) * half_spanwise;
    const amrex::Real rect_ry = amrex::Math::abs(ny) * half_streamwise +
                                amrex::Math::abs(ty) * half_spanwise;

    if (!intervals_overlap_inclusive(cx - rect_rx, cx + rect_rx, xlo, xhi)) {
        return false;
    }

    if (!intervals_overlap_inclusive(cy - rect_ry, cy + rect_ry, ylo, yhi)) {
        return false;
    }

    const amrex::Real rect_cn = cx * nx + cy * ny;
    const amrex::Real cell_cn = cell_cx * nx + cell_cy * ny;
    const amrex::Real cell_rn = amrex::Math::abs(nx) * hx + amrex::Math::abs(ny) * hy;
    if (!intervals_overlap_inclusive(rect_cn - half_streamwise, rect_cn + half_streamwise,
                                     cell_cn - cell_rn, cell_cn + cell_rn)) {
        return false;
    }

    const amrex::Real rect_ct = cx * tx + cy * ty;
    const amrex::Real cell_ct = cell_cx * tx + cell_cy * ty;
    const amrex::Real cell_rt = amrex::Math::abs(tx) * hx + amrex::Math::abs(ty) * hy;
    return intervals_overlap_inclusive(rect_ct - half_spanwise, rect_ct + half_spanwise,
                                       cell_ct - cell_rt, cell_ct + cell_rt);
}

#ifdef ERF_USE_WINDFARM
bool
validate_turb_refine_region (const amrex::Geometry& geom,
                             const amrex::BoxArray& grids,
                             const amrex::BoxArray* finer_grids,
                             const amrex::IntVect* ref_ratio,
                             const amrex::MultiFab* z_phys_nd,
                             WindFarm* windfarm,
                             const amrex::Vector<int>& turbine_ids,
                             amrex::Real pad_streamwise_by_D,
                             amrex::Real pad_spanwise_by_D,
                             amrex::Real box_z_lo,
                             amrex::Real box_z_hi)
{
    if (turbine_ids.empty()) {
        return true;
    }

    const amrex::Box& domain = geom.Domain();
    const auto dx = geom.CellSizeArray();
    const auto prob_lo = geom.ProbLoArray();
    const int k_domain_lo = domain.smallEnd(2);
    const int k_domain_hi = domain.bigEnd(2);

    amrex::BoxArray uncovered_valid_boxes =
        build_uncovered_valid_boxes(domain, grids, finer_grids, ref_ratio);

    amrex::Vector<int> i_cols;
    amrex::Vector<int> j_cols;
    i_cols.reserve(turbine_ids.size());
    j_cols.reserve(turbine_ids.size());

    for (int turbine_id : turbine_ids) {
        amrex::Real x, y, zhub, diameter, nx, ny, tx, ty;
        windfarm->get_turbine_refinement_geometry(turbine_id, x, y, zhub, diameter, nx, ny, tx, ty);
        amrex::ignore_unused(zhub, diameter, nx, ny, tx, ty);

        const int i_center = static_cast<int>(std::floor((x - prob_lo[0]) / dx[0]));
        const int j_center = static_cast<int>(std::floor((y - prob_lo[1]) / dx[1]));
        if (i_center < domain.smallEnd(0) || i_center > domain.bigEnd(0) ||
            j_center < domain.smallEnd(1) || j_center > domain.bigEnd(1)) {
            return false;
        }

        i_cols.push_back(i_center);
        j_cols.push_back(j_center);
    }

    const auto z_columns = gather_z_columns_to_host(geom, z_phys_nd, i_cols, j_cols);

    for (int idx = 0; idx < static_cast<int>(turbine_ids.size()); ++idx) {
        const int turbine_id = turbine_ids[idx];
        amrex::Real x, y, zhub, diameter, nx, ny, tx, ty;
        windfarm->get_turbine_refinement_geometry(turbine_id, x, y, zhub, diameter, nx, ny, tx, ty);
        if (diameter <= 0.0) {
            return false;
        }

        const amrex::Real half_streamwise = pad_streamwise_by_D * diameter;
        const amrex::Real half_spanwise   = pad_spanwise_by_D  * diameter;
        amrex::ignore_unused(zhub);

        std::array<amrex::Real,4> xcorners{
            x + half_streamwise * nx + half_spanwise * tx,
            x + half_streamwise * nx - half_spanwise * tx,
            x - half_streamwise * nx + half_spanwise * tx,
            x - half_streamwise * nx - half_spanwise * tx
        };
        std::array<amrex::Real,4> ycorners{
            y + half_streamwise * ny + half_spanwise * ty,
            y + half_streamwise * ny - half_spanwise * ty,
            y - half_streamwise * ny + half_spanwise * ty,
            y - half_streamwise * ny - half_spanwise * ty
        };

        const amrex::Real x_min = *std::min_element(xcorners.begin(), xcorners.end());
        const amrex::Real x_max = *std::max_element(xcorners.begin(), xcorners.end());
        const amrex::Real y_min = *std::min_element(ycorners.begin(), ycorners.end());
        const amrex::Real y_max = *std::max_element(ycorners.begin(), ycorners.end());

        const amrex::Real x_min_in = std::nextafter(x_min, std::numeric_limits<amrex::Real>::infinity());
        const amrex::Real x_max_in = std::nextafter(x_max, -std::numeric_limits<amrex::Real>::infinity());
        const amrex::Real y_min_in = std::nextafter(y_min, std::numeric_limits<amrex::Real>::infinity());
        const amrex::Real y_max_in = std::nextafter(y_max, -std::numeric_limits<amrex::Real>::infinity());

        const int ilo = static_cast<int>(std::floor((x_min_in - prob_lo[0]) / dx[0]));
        const int ihi = static_cast<int>(std::floor((x_max_in - prob_lo[0]) / dx[0]));
        const int jlo = static_cast<int>(std::floor((y_min_in - prob_lo[1]) / dx[1]));
        const int jhi = static_cast<int>(std::floor((y_max_in - prob_lo[1]) / dx[1]));

        if (ilo < domain.smallEnd(0) || ihi > domain.bigEnd(0) ||
            jlo < domain.smallEnd(1) || jhi > domain.bigEnd(1)) {
            return false;
        }

        const amrex::Vector<amrex::Real>& z_nodes = z_columns[idx];
        int klo = -1;
        int khi = -1;
        bool has_gap_in_extent = false;

        for (int k = k_domain_lo; k <= k_domain_hi; ++k) {
            const int lo_idx = k - k_domain_lo;
            const int hi_idx = lo_idx + 1;
            if (lo_idx < 0 || hi_idx >= static_cast<int>(z_nodes.size())) {
                continue;
            }

            const amrex::Real cell_lo = z_nodes[lo_idx];
            const amrex::Real cell_hi = z_nodes[hi_idx];

            if (!is_missing_z_value(cell_lo) && klo >= 0 && cell_lo > box_z_hi) {
                break;
            }

            if (is_missing_z_value(cell_lo) || is_missing_z_value(cell_hi)) {
                if (klo >= 0 && khi >= 0) {
                    has_gap_in_extent = true;
                    break;
                }
                continue;
            }

            if (cell_hi >= box_z_lo && cell_lo <= box_z_hi) {
                if (klo < 0) { klo = k; }
                khi = k;
            }
        }

        if (klo < 0 || has_gap_in_extent) {
            return false;
        }

        amrex::Box footprint_box(amrex::IntVect(AMREX_D_DECL(ilo, jlo, klo)),
                                 amrex::IntVect(AMREX_D_DECL(ihi, jhi, khi)));
        if (!uncovered_valid_boxes.contains(footprint_box)) {
            return false;
        }
    }

    return true;
}
#endif

} // namespace


/**
 * Function to tag cells for refinement -- this overrides the pure virtual function in AmrCore
 *
 * @param[in ] levc level of refinement at which we tag cells (0 is coarsest level)
 * @param[out] tags array of tagged cells
 * @param[in ] time current time
 * @param[in ] ngrow number of ghost cells (not used here)
*/
void
ERF::ErrorEst (int levc, TagBoxArray& tags, Real time, int /*ngrow*/)
{
    const int clearval = TagBox::CLEAR;
    const int   tagval = TagBox::SET;

#ifdef ERF_USE_NETCDF
    if ((solverChoice.init_type == InitType::WRFInput) || (solverChoice.init_type == InitType::Metgrid)) {
        int ratio;
        Box subdomain;

        // This is the number of boxes that may have already been defined in the refinement_criteria_setup routine.
        // If nb == 0 then no boxes have been specified in the inputs file, and we will use the boxes given in wrfinput_d*
        // If nb >  0 then    boxes have been specified in the inputs file, and we will use the specified boxes as long
        //    as we can ensure that they are contained inside the boxes given in wrfinput_d*
        int nb_prespecified = num_boxes_at_level[levc+1];

        if (!nc_init_file[levc+1].empty())
        {
            Real levc_start_time = read_start_time_from_wrfinput(levc  , nc_init_file[levc  ][0]);
            if (solverChoice.init_type == InitType::WRFInput) {
                amrex::Print() << " WRFInput       time at level " << levc << " is " << levc_start_time << std::endl;
            } else if (solverChoice.init_type == InitType::Metgrid) {
                amrex::Print() << " met_em         time at level " << levc << " is " << levc_start_time << std::endl;
            }

            for (int isub = 0; isub < nc_init_file[levc+1].size(); isub++) {
                if (!have_read_nc_init_file[levc+1][isub])
                {
                    Real levf_start_time = read_start_time_from_wrfinput(levc+1, nc_init_file[levc+1][isub]);
                    if (solverChoice.init_type == InitType::WRFInput) {
                        amrex::Print() << " WRFInput start_time at level " << levc+1 << " is " << levf_start_time << std::endl;
                    } else if (solverChoice.init_type == InitType::Metgrid) {
                        amrex::Print() << " met_em   start time at level " << levc+1 << " is " << levf_start_time << std::endl;
                    }

                    // We assume there is only one subdomain at levc; otherwise we don't know
                    //     which one is the parent of the fine region we are trying to create
                    AMREX_ALWAYS_ASSERT(subdomains[levc].size() == 1);

                    if ((solverChoice.init_type == InitType::WRFInput) && ((ref_ratio[levc][2]) != 1)) {
                        amrex::Abort("The ref_ratio specified in the inputs file must have 1 in the z direction; please use ref_ratio_vect rather than ref_ratio");
                    }

                    if ( levf_start_time <= (levc_start_time + t_new[levc]) ) {
                        if (solverChoice.init_type == InitType::WRFInput) {
                            amrex::Print() << " WRFInput file to read: " << nc_init_file[levc+1][isub] << std::endl;
                            subdomain = read_subdomain_from_wrfinput(levc, nc_init_file[levc+1][isub], ratio);
                            amrex::Print() << " WRFInput subdomain " << isub << " at level " << levc+1 << " is " << subdomain << std::endl;
                        } else if (solverChoice.init_type == InitType::Metgrid) {
                            amrex::Print() << "met_em file to read: " << nc_init_file[levc+1][0] << std::endl;
                            const Box& domain = geom[levc].Domain();
                            int klo = domain.smallEnd(2);
                            int khi = domain.bigEnd(2);
                            subdomain = read_subdomain_from_metgrid(levc, nc_init_file[levc+1][0], ratio, klo, khi);
                            amrex::Print() << " met_em subdomain at level " << levc+1 << " is " << subdomain << std::endl;
                        }

                        if ( (ratio != ref_ratio[levc][0]) || (ratio != ref_ratio[levc][1]) ) {
                            amrex::Print() << "File " << nc_init_file[levc+1][0] << " has refinement ratio = " << ratio << std::endl;
                            amrex::Print() << "The inputs file has refinement ratio = " << ref_ratio[levc] << std::endl;
                            amrex::Abort("These must be the same -- please edit your inputs file and try again.");
                        }

                        subdomain.coarsen(ref_ratio[levc]);

                        // Recall we asserted that there is only one box at level levc
                        Box coarser_level(subdomains[levc][0].minimalBox());
                        subdomain.shift(coarser_level.smallEnd());

                        if (verbose > 0) {
                            amrex::Print() << " Crse version of subdomain available for tagging is" << subdomain << std::endl;
                        }

                        Box new_fine(subdomain);
                        if (solverChoice.init_type == InitType::WRFInput) {
                            new_fine.refine(IntVect(ratio,ratio,1));
                        } else if (solverChoice.init_type == InitType::Metgrid) {
                            new_fine.refine(ref_ratio[levc]);
                        }
                        if (nb_prespecified == 0) {
                            num_boxes_at_level[levc+1] += 1;
                            boxes_at_level[levc+1].push_back(new_fine);
                        } else {
                            if (!new_fine.contains(boxes_at_level[levc+1][isub])) {
                                amrex::Print() << "\n";
                                amrex::Print() << "Box available in wrfinputs file             " << new_fine << std::endl;
                                amrex::Print() << "Box requested for refinement in inputs file " << boxes_at_level[levc+1][isub] << std::endl;
                                amrex::Abort("Specified boxes must be contained within boxes specified in wrfinput at this level");
                            }
                        }

                        Box coarsened_bx(boxes_at_level[levc+1][isub]); coarsened_bx.coarsen(ref_ratio[levc]);

                        for (MFIter mfi(tags); mfi.isValid(); ++mfi)
                        {
                            auto tag_arr = tags.array(mfi);  // Get device-accessible array

                            Box bx = mfi.validbox() & coarsened_bx;

                            if (!bx.isEmpty()) {
                                ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                                    tag_arr(i,j,k) = TagBox::SET;
                                });
                            }
                        }
                    } // time is right
                } else {
                    // Re-tag this region
                    for (MFIter mfi(tags); mfi.isValid(); ++mfi)
                    {
                        auto tag_arr = tags.array(mfi);  // Get device-accessible array

                        Box existing_bx_coarsened(boxes_at_level[levc+1][isub]);
                        existing_bx_coarsened.coarsen(ref_ratio[levc]);

                        Box bx = mfi.validbox(); bx &= existing_bx_coarsened;

                        if (!bx.isEmpty()) {
                            ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                                tag_arr(i,j,k) = TagBox::SET;
                            });
                        }
                    }
                } // has file been read?
            } // isub
            return;
        } // file not empty
    }
#endif

    //
    // Make sure the ghost cells of the level we are tagging at are filled
    //    in case we take differences that require them
    // NOTE: We are Fillpatching only the cell-centered variables here
    //
    MultiFab& S_new = vars_new[levc][Vars::cons];
    MultiFab& U_new = vars_new[levc][Vars::xvel];
    MultiFab& V_new = vars_new[levc][Vars::yvel];
    MultiFab& W_new = vars_new[levc][Vars::zvel];
    //
    if (levc == 0) {
        FillPatchCrseLevel(levc, time, {&S_new, &U_new, &V_new, &W_new});
    } else {
        FillPatchFineLevel(levc, time, {&S_new, &U_new, &V_new, &W_new},
                           {&S_new, &rU_new[levc], &rV_new[levc], &rW_new[levc]},
                           base_state[levc], base_state[levc],
                           false, true);
    }

    for (int j=0; j < ref_tags.size(); ++j)
    {
        //
        // This mf must have ghost cells because we may take differences between adjacent values
        //
        std::unique_ptr<MultiFab> mf = std::make_unique<MultiFab>(grids[levc], dmap[levc], 1, 1);
        mf->setVal(0.0);

        // This allows dynamic refinement based on the value of the density
        if (ref_tags[j].Field() == "density")
        {
            MultiFab::Copy(*mf,vars_new[levc][Vars::cons],Rho_comp,0,1,1);

        // This allows dynamic refinement based on the value of qv
        } else if ( ref_tags[j].Field() == "qv" ) {
            MultiFab::Copy(  *mf, vars_new[levc][Vars::cons], RhoQ1_comp, 0, 1, 1);
            MultiFab::Divide(*mf, vars_new[levc][Vars::cons],   Rho_comp, 0, 1, 1);


        // This allows dynamic refinement based on the value of qc
        } else if (ref_tags[j].Field() == "qc" ) {
            MultiFab::Copy(  *mf, vars_new[levc][Vars::cons], RhoQ2_comp, 0, 1, 1);
            MultiFab::Divide(*mf, vars_new[levc][Vars::cons],   Rho_comp, 0, 1, 1);

        // This allows dynamic refinement based on the value of the z-component of vorticity
        } else if (ref_tags[j].Field() == "vorticity" ) {
            Vector<MultiFab> mf_cc_vel(1);
            mf_cc_vel[0].define(grids[levc], dmap[levc], AMREX_SPACEDIM, IntVect(1,1,1));
            average_face_to_cellcenter(mf_cc_vel[0],0,Array<const MultiFab*,3>{&U_new, &V_new, &W_new});

            // Impose bc's at domain boundaries at all levels
            FillBdyCCVels(mf_cc_vel,levc);

            mf->setVal(0.);

            for (MFIter mfi(*mf, TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                const Box& bx = mfi.tilebox();
                auto& dfab = (*mf)[mfi];
                auto& sfab = mf_cc_vel[0][mfi];
                derived::erf_dervortz(bx, dfab, 0, 1, sfab, Geom(levc), time, nullptr, levc);
            }

        // This allows dynamic refinement based on the value of the scalar/theta
        } else if ( (ref_tags[j].Field() == "scalar"  ) ||
                    (ref_tags[j].Field() == "theta"   ) )
        {
            for (MFIter mfi(*mf, TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                const Box& bx = mfi.growntilebox();
                auto& dfab = (*mf)[mfi];
                auto& sfab = vars_new[levc][Vars::cons][mfi];
                if (ref_tags[j].Field() == "scalar") {
                    derived::erf_derscalar(bx, dfab, 0, 1, sfab, Geom(levc), time, nullptr, levc);
                } else if (ref_tags[j].Field() == "theta") {
                    derived::erf_dertheta(bx, dfab, 0, 1, sfab, Geom(levc), time, nullptr, levc);
                }
            } // mfi
        // This allows dynamic refinement based on the value of the density
        } else if ( (SolverChoice::terrain_type == TerrainType::ImmersedForcing) &&
                    (ref_tags[j].Field() == "terrain_blanking") )
        {
            MultiFab::Copy(*mf,*terrain_blanking[levc],0,0,1,1);
        }
        else if (ref_tags[j].Field() == "velmag")
        {
            ParmParse pp(pp_prefix);
            Vector<std::string> refinement_indicators;
            pp.queryarr("refinement_indicators",refinement_indicators,0,pp.countval("refinement_indicators"));
            Real velmag_threshold;
            bool is_hurricane_tracker = false;
            for (int i=0; i<refinement_indicators.size(); ++i)
            {
                if (refinement_indicators[i]=="hurricane_tracker") {
                    is_hurricane_tracker = true;
                    std::string ref_prefix = pp_prefix + "." + refinement_indicators[i];
                    ParmParse ppr(ref_prefix);
                    ppr.get("value_greater", velmag_threshold);
                    break;
                }
            }

            Vector<MultiFab> mf_cc_vel(1);
            mf_cc_vel[0].define(grids[levc], dmap[levc], AMREX_SPACEDIM, IntVect(0,0,0));
            average_face_to_cellcenter(mf_cc_vel[0],0,Array<const MultiFab*,3>{&U_new, &V_new, &W_new});

            if (is_hurricane_tracker) {
                HurricaneTracker(levc, time, mf_cc_vel[0], velmag_threshold, &tags);
            } else {
                for (MFIter mfi(*mf, TilingIfNotGPU()); mfi.isValid(); ++mfi)
                {
                    const Box& bx = mfi.tilebox();
                    auto& dfab = (*mf)[mfi];
                    auto& sfab = mf_cc_vel[0][mfi];
                    derived::erf_dermagvel(bx, dfab, 0, 1, sfab, Geom(levc), time, nullptr, levc);
                }
            }

#ifdef ERF_USE_PARTICLES
        } else {
            //
            // This allows dynamic refinement based on the number of particles per cell
            //
            // Note that we must count all the particles in levels both at and above the current,
            //      since otherwise, e.g., if the particles are all at level 1, counting particles at
            //      level 0 will not trigger refinement when regridding so level 1 will disappear,
            //      then come back at the next regridding
            //
            const auto& particles_namelist( particleData.getNames() );
            mf->setVal(0.0);
            for (ParticlesNamesVector::size_type i = 0; i < particles_namelist.size(); i++)
            {
                std::string tmp_string(particles_namelist[i]+"_count");
                IntVect rr = IntVect::TheUnitVector();
                if (ref_tags[j].Field() == tmp_string) {
                    for (int lev = levc; lev <= finest_level; lev++)
                    {
                        MultiFab temp_dat(grids[lev], dmap[lev], 1, 0); temp_dat.setVal(0);
                        particleData[particles_namelist[i]]->IncrementWithTotal(temp_dat, lev);

                        MultiFab temp_dat_crse(grids[levc], dmap[levc], 1, 0); temp_dat_crse.setVal(0);

                        if (lev == levc) {
                            MultiFab::Copy(*mf, temp_dat, 0, 0, 1, 0);
                        } else {
                            for (int d = 0; d < AMREX_SPACEDIM; d++) {
                                rr[d] *= ref_ratio[levc][d];
                            }
                            average_down(temp_dat, temp_dat_crse, 0, 1, rr);
                            MultiFab::Add(*mf, temp_dat_crse, 0, 0, 1, 0);
                        }
                    }
                }
            }
#endif
        }

        ref_tags[j](tags,mf.get(),clearval,tagval,time,levc,geom[levc]);
    } // loop over j

    // ********************************************************************************************
    // Refinement based on 2d distance from the "eye" which is defined here as the (x,y) location of
    //    the integrated qv
    // ********************************************************************************************
    ParmParse pp(pp_prefix);
    Vector<std::string> refinement_indicators;
    pp.queryarr("refinement_indicators",refinement_indicators,0,pp.countval("refinement_indicators"));
    for (int i=0; i<refinement_indicators.size(); ++i)
    {
        if ( (refinement_indicators[i]=="storm_tracker") && (solverChoice.moisture_type != MoistureType::None) )
        {
            std::string ref_prefix = pp_prefix + "." + refinement_indicators[i];
            ParmParse ppr(ref_prefix);

            Real ref_start_time = -1.0;
            ppr.query("start_time",ref_start_time);

            if (time >= ref_start_time) {

                Real max_radius = -1.0;
                ppr.get("max_radius", max_radius);

                // Create the volume-weighted sum of (rho qv) in each column
                MultiFab mf_qv_int(ba2d[levc], dmap[levc], 1, 0); mf_qv_int.setVal(0.);

                // Define the 2D MultiFab holding the column-integrated (rho qv)
                volWgtColumnSum(levc, S_new, RhoQ1_comp, mf_qv_int, *detJ_cc[levc]);

                // Find the max value in the domain
                IntVect eye = mf_qv_int.maxIndex(0);

                const auto dx      = geom[levc].CellSizeArray();
                const auto prob_lo = geom[levc].ProbLoArray();

                Real eye_x = prob_lo[0] + (eye[0] + 0.5) * dx[0];
                Real eye_y = prob_lo[1] + (eye[1] + 0.5) * dx[1];

                tag_on_distance_from_eye(geom[levc], &tags, eye_x, eye_y, max_radius);
            }
        }
    }

#ifdef ERF_USE_WINDFARM
    if (turb_refine_info.enabled && windfarm &&
        time >= turb_refine_info.start_time &&
        time <= turb_refine_info.end_time)
    {
        if (!m_windfarm_hierarchy_initialized) {
            amrex::Abort("erf.refinement_indicators=turb_refine requires the wind-farm hierarchy to be initialized before tagging, but it was not ready when ErrorEst was called.");
        }

        int owner_level = -1;
        int owner_level_count = 0;
        for (int lev = 0; lev <= finest_level; ++lev) {
            if (!windfarm->turbines_on_level(lev).empty()) {
                owner_level = lev;
                ++owner_level_count;
            }
        }

        if (owner_level_count > 1) {
            amrex::Abort("erf.refinement_indicators=turb_refine requires all turbines to be contained within a single AMR level, but turbines were found on multiple levels.");
        }

        const int max_indicator_level = (turb_refine_info.max_level > 0)
            ? turb_refine_info.max_level
            : max_level;

        // Keep every parent in the refinement chain tagged after the turbine
        // moves to a finer owner level.  Restricting tags to levc ==
        // owner_level makes the parent stop requesting its child; the next
        // regrid then removes that child and ownership oscillates between
        // levels.
        if (owner_level >= 0 &&
            levc <= owner_level &&
            (levc + 1) <= max_indicator_level)
        {
            const auto& turbine_ids = windfarm->turbines_on_level(owner_level);
            const amrex::BoxArray* finer_grids = (levc < finest_level) ? &grids[levc+1] : nullptr;
            const amrex::IntVect* ratio = (levc < finest_level) ? &ref_ratio[levc] : nullptr;
            const amrex::MultiFab* z_nd_for_validation =
                (SolverChoice::mesh_type == MeshType::ConstantDz) ? nullptr : z_phys_nd[levc].get();

            // The single-level footprint guard applies on the current owner
            // level.  Ancestor levels are tagged only to preserve the parent
            // chain, and their footprint is intentionally covered by the
            // child that owns the turbine.
            if (levc == owner_level &&
                !validate_turb_refine_region(geom[levc], grids[levc], finer_grids, ratio,
                                             z_nd_for_validation, windfarm.get(),
                                             turbine_ids,
                                             turb_refine_info.pad_streamwise_by_D,
                                             turb_refine_info.pad_spanwise_by_D,
                                             turb_refine_info.box_z_lo,
                                             turb_refine_info.box_z_hi)) {
                amrex::Abort("erf.refinement_indicators=turb_refine requested a turbine refinement region that extends outside the immediate parent AMR level. Increase the parent-level refined region or reduce erf.turb_refine horizontal padding or z bounds.");
            }

            const auto dx = geom[levc].CellSizeArray();
            const auto prob_lo = geom[levc].ProbLoArray();
            const bool use_physical_z_bounds =
                (SolverChoice::mesh_type != MeshType::ConstantDz) && static_cast<bool>(z_phys_nd[levc]);

            for (int turbine_id : turbine_ids) {
                amrex::Real x, y, zhub, diameter, nx, ny, tx, ty;
                windfarm->get_turbine_refinement_geometry(turbine_id, x, y, zhub, diameter, nx, ny, tx, ty);
                if (diameter <= 0.0) {
                    amrex::Abort("erf.refinement_indicators=turb_refine could not obtain valid turbine geometry for refinement tagging.");
                }

                const amrex::Real half_streamwise = turb_refine_info.pad_streamwise_by_D * diameter;
                const amrex::Real half_spanwise   = turb_refine_info.pad_spanwise_by_D  * diameter;
                const amrex::Real z_lo = turb_refine_info.box_z_lo;
                const amrex::Real z_hi = turb_refine_info.box_z_hi;
                amrex::ignore_unused(zhub);

                for (MFIter mfi(tags, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                    auto tag_arr = tags.array(mfi);
                    const Box& bx = mfi.tilebox();
                    const Array4<const Real> z_nd_arr = use_physical_z_bounds
                        ? z_phys_nd[levc]->const_array(mfi)
                        : Array4<const Real>{};

                    ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                        const amrex::Real cell_xlo = prob_lo[0] + static_cast<amrex::Real>( i    ) * dx[0];
                        const amrex::Real cell_xhi = prob_lo[0] + static_cast<amrex::Real>( i + 1) * dx[0];
                        const amrex::Real cell_ylo = prob_lo[1] + static_cast<amrex::Real>( j    ) * dx[1];
                        const amrex::Real cell_yhi = prob_lo[1] + static_cast<amrex::Real>( j + 1) * dx[1];

                        if (!oriented_rect_overlaps_cell_2d(x, y, nx, ny, tx, ty,
                                                            half_streamwise, half_spanwise,
                                                            cell_xlo, cell_xhi,
                                                            cell_ylo, cell_yhi)) {
                            return;
                        }

                        amrex::Real cell_zlo;
                        amrex::Real cell_zhi;
                        if (use_physical_z_bounds) {
                            cell_zlo = z_nd_arr(i, j, k);
                            cell_zhi = cell_zlo;
                            for (int kk = 0; kk <= 1; ++kk) {
                                for (int jj = 0; jj <= 1; ++jj) {
                                    for (int ii = 0; ii <= 1; ++ii) {
                                        const amrex::Real z_node = z_nd_arr(i + ii, j + jj, k + kk);
                                        cell_zlo = amrex::min(cell_zlo, z_node);
                                        cell_zhi = amrex::max(cell_zhi, z_node);
                                    }
                                }
                            }
                        } else {
                            cell_zlo = prob_lo[2] + static_cast<amrex::Real>( k    ) * dx[2];
                            cell_zhi = prob_lo[2] + static_cast<amrex::Real>( k + 1) * dx[2];
                        }

                        if (intervals_overlap_inclusive(cell_zlo, cell_zhi, z_lo, z_hi)) {
                            tag_arr(i, j, k) = TagBox::SET;
                        }
                    });
                }
            }
        }
    }
#endif
}

/**
 * Function to define the refinement criteria based on user input
*/

void
ERF::refinement_criteria_setup ()
{
    if (max_level > 0)
    {
        ParmParse pp(pp_prefix);
        Vector<std::string> refinement_indicators;
        pp.queryarr("refinement_indicators",refinement_indicators,0,pp.countval("refinement_indicators"));
        turb_refine_info = TurbRefineInfo{};
        turb_refine_force_regrid.assign(max_level+1, 0);

        for (int i=0; i<refinement_indicators.size(); ++i)
        {
            std::string ref_prefix = pp_prefix + "." + refinement_indicators[i];

            ParmParse ppr(ref_prefix);
            RealBox realbox;
            int lev_for_box = -1;

            if (refinement_indicators[i] == "turb_refine") {
#ifdef ERF_USE_WINDFARM
                if (!(solverChoice.windfarm_type == WindFarmType::SimpleAD ||
                      solverChoice.windfarm_type == WindFarmType::GeneralAD ||
                      solverChoice.windfarm_type == WindFarmType::ClassicAD)) {
                    amrex::Abort("erf.refinement_indicators=turb_refine requires an actuator-disk windfarm model.");
                }

                if (any_pbl_models_active(solverChoice, max_level)) {
                    amrex::Abort("erf.refinement_indicators=turb_refine is incompatible with active PBL models because PBL refinement must span the full vertical domain.");
                }

                const bool has_streamwise = (ppr.countval("pad_streamwise_by_D") > 0);
                const bool has_spanwise = (ppr.countval("pad_spanwise_by_D") > 0);
                const bool has_box_z_lo = (ppr.countval("box_z_lo") > 0);
                const bool has_box_z_hi = (ppr.countval("box_z_hi") > 0);
                if (!(has_streamwise && has_spanwise && has_box_z_lo && has_box_z_hi)) {
                    amrex::Abort("erf.refinement_indicators=turb_refine requires erf.turb_refine.pad_streamwise_by_D, erf.turb_refine.pad_spanwise_by_D, erf.turb_refine.box_z_lo, and erf.turb_refine.box_z_hi.");
                }

                turb_refine_info.enabled = true;
                ppr.get("pad_streamwise_by_D", turb_refine_info.pad_streamwise_by_D);
                ppr.get("pad_spanwise_by_D", turb_refine_info.pad_spanwise_by_D);
                ppr.get("box_z_lo", turb_refine_info.box_z_lo);
                ppr.get("box_z_hi", turb_refine_info.box_z_hi);
                ppr.query("start_time", turb_refine_info.start_time);
                ppr.query("end_time", turb_refine_info.end_time);
                turb_refine_info.max_level = max_level;
                ppr.query("max_level", turb_refine_info.max_level);

                if (turb_refine_info.pad_streamwise_by_D < 0.0 ||
                    turb_refine_info.pad_spanwise_by_D < 0.0 ||
                    turb_refine_info.box_z_lo < 0.0 ||
                    turb_refine_info.box_z_hi < turb_refine_info.box_z_lo) {
                    amrex::Abort("erf.turb_refine.pad_streamwise_by_D and erf.turb_refine.pad_spanwise_by_D must be nonnegative, erf.turb_refine.box_z_lo must be nonnegative, and erf.turb_refine.box_z_hi must be greater than or equal to erf.turb_refine.box_z_lo.");
                }

                if (turb_refine_info.max_level < 1 || turb_refine_info.max_level > max_level) {
                    amrex::Abort("erf.turb_refine.max_level must be between 1 and amr.max_level.");
                }
#else
                amrex::Abort("erf.refinement_indicators=turb_refine requires ERF to be built with windfarm support.");
#endif
                continue;
            }

            int num_real_lo      = ppr.countval("in_box_lo");
            int num_indx_lo      = ppr.countval("in_box_lo_indices");
            int num_indx_lo_crse = ppr.countval("in_box_lo_indices_crse");

            int num_real_hi      = ppr.countval("in_box_hi");
            int num_indx_hi      = ppr.countval("in_box_hi_indices");
            int num_indx_hi_crse = ppr.countval("in_box_hi_indices_crse");

            AMREX_ALWAYS_ASSERT( (num_real_lo      == num_real_hi)      && (num_real_lo      == 0 || num_real_lo      >= 2) );
            AMREX_ALWAYS_ASSERT( (num_indx_lo      == num_indx_hi)      && (num_indx_lo      == 0 || num_indx_lo      >= 2) );
            AMREX_ALWAYS_ASSERT( (num_indx_lo_crse == num_indx_hi_crse) && (num_indx_lo_crse == 0 || num_indx_lo_crse >= 2) );

            // Problem low and high (in real not index space) are the same at all levels
            const Real* plo = geom[0].ProbLo();
            const Real* phi = geom[0].ProbHi();
            if ( !((num_real_lo >= AMREX_SPACEDIM-1 && num_indx_lo == 0 && num_indx_lo_crse == 0) ||
                   (num_indx_lo >= AMREX_SPACEDIM-1 && num_real_lo == 0 && num_indx_lo_crse == 0) ||
                   (num_indx_lo ==              0   && num_real_lo == 0 && num_indx_lo_crse == 0) ||
                   (num_indx_lo_crse >= AMREX_SPACEDIM-1 && num_real_lo == 0 && num_indx_lo == 0)
                ) )
            {
                amrex::Abort("Must only specify box for refinement using real OR index space with fine/coarse grid indices");
            }

            if (num_real_lo > 0) {
                std::vector<Real> rbox_lo(3), rbox_hi(3);
                lev_for_box = max_level;
                ppr.query("max_level",lev_for_box);
                if (lev_for_box > 0 && lev_for_box <= max_level)
                {
                    if (n_error_buf[0] != IntVect::TheZeroVector()) {
                        amrex::Abort("Don't use n_error_buf > 0 when setting the box explicitly");
                    }

                    ppr.getarr("in_box_lo",rbox_lo,0,num_real_lo);
                    ppr.getarr("in_box_hi",rbox_hi,0,num_real_hi);

                    if (rbox_lo[0] < plo[0]) rbox_lo[0] = plo[0];
                    if (rbox_lo[1] < plo[1]) rbox_lo[1] = plo[1];
                    if (rbox_hi[0] > phi[0]) rbox_hi[0] = phi[0];
                    if (rbox_hi[1] > phi[1]) rbox_hi[1] = phi[1];
                    if (num_real_lo < AMREX_SPACEDIM) {
                        rbox_lo[2] = plo[2];
                        rbox_hi[2] = phi[2];
                    }

                    const Box& domain = geom[lev_for_box].Domain();

                    realbox = RealBox(&(rbox_lo[0]),&(rbox_hi[0]));

                    Print() << "Realbox read in and intersected laterally with domain is " << realbox << std::endl;

                    num_boxes_at_level[lev_for_box] += 1;

                    int ilo, jlo, klo;
                    int ihi, jhi, khi;
                    const auto* dx  = geom[lev_for_box].CellSize();
                    ilo = static_cast<int>((rbox_lo[0] - plo[0])/dx[0]);
                    jlo = static_cast<int>((rbox_lo[1] - plo[1])/dx[1]);
                    ihi = static_cast<int>((rbox_hi[0] - plo[0])/dx[0]-1);
                    jhi = static_cast<int>((rbox_hi[1] - plo[1])/dx[1]-1);
                    if (SolverChoice::mesh_type != MeshType::ConstantDz) {
                        // Search for k indices corresponding to nominal grid
                        // AGL heights
                        klo = domain.smallEnd(2) - 1;
                        khi = domain.smallEnd(2) - 1;

                        if (rbox_lo[2] <= zlevels_stag[lev_for_box][domain.smallEnd(2)])
                        {
                            klo = domain.smallEnd(2);
                        }
                        else
                        {
                            for (int k=domain.smallEnd(2); k<=domain.bigEnd(2)+1; ++k) {
                                if (zlevels_stag[lev_for_box][k] > rbox_lo[2]) {
                                    klo = k-1;
                                    break;
                                }
                            }
                        }
                        AMREX_ASSERT(klo >= domain.smallEnd(2));

                        if (rbox_hi[2] >= zlevels_stag[lev_for_box][domain.bigEnd(2)+1])
                        {
                            khi = domain.bigEnd(2);
                        }
                        else
                        {
                            for (int k=klo+1; k<=domain.bigEnd(2)+1; ++k) {
                                if (zlevels_stag[lev_for_box][k] > rbox_hi[2]) {
                                    khi = k-1;
                                    break;
                                }
                            }
                        }
                        AMREX_ASSERT((khi <= domain.bigEnd(2)) && (khi > klo));

                        // Need to update realbox because tagging is based on
                        // the initial _un_deformed grid
                        realbox = RealBox(plo[0]+ ilo   *dx[0], plo[1]+ jlo   *dx[1], plo[2]+ klo   *dx[2],
                                          plo[0]+(ihi+1)*dx[0], plo[1]+(jhi+1)*dx[1], plo[2]+(khi+1)*dx[2]);
                    } else {
                        klo = static_cast<int>((rbox_lo[2] - plo[2])/dx[2]);
                        khi = static_cast<int>((rbox_hi[2] - plo[2])/dx[2]-1);
                    }

                    Box bx(IntVect(ilo,jlo,klo),IntVect(ihi,jhi,khi));
                    // Error check for each index
                    if(ilo%ref_ratio[lev_for_box-1][0] != 0){
                        amrex::Print()<< "Requested in_box_lo in x direction = " << rbox_lo[0] << " corresponds to ilo = " << ilo << std::endl;
                        amrex::Print() << "ilo = " << ilo << " is not divisible by ref_ratio in x direction = " << ref_ratio[lev_for_box-1][0] << std::endl;
                        amrex::Error("Adjust in_box_lo in x-direction to be divisible by ref_ratio and try again");
                    }
                    if((ihi+1)%ref_ratio[lev_for_box-1][0] != 0){
                        amrex::Print()<< "Requested in_box_hi in x direction = " << rbox_hi[0] << " corresponds to ihi+1 = " << ihi+1 << std::endl;
                        amrex::Print() << "ihi+1 = " << ihi+1 << " is not divisible by ref_ratio in x direction = " << ref_ratio[lev_for_box-1][0] << std::endl;
                        amrex::Error("Adjust in_box_hi in x-direction to be divisible by ref_ratio and try again");
                    }
                     if(jlo%ref_ratio[lev_for_box-1][1] != 0){
                        amrex::Print()<< "Requested in_box_lo in y direction = " << rbox_lo[1] << " corresponds to jlo = " << jlo << std::endl;
                        amrex::Print() << "jlo = " << jlo << " is not divisible by ref_ratio in y direction = " << ref_ratio[lev_for_box-1][1] << std::endl;
                        amrex::Error("Adjust in_box_lo in y-direction to be divisible by ref_ratio and try again");
                    }
                    if((jhi+1)%ref_ratio[lev_for_box-1][1] != 0){
                        amrex::Print()<< "Requested in_box_hi in y direction = " << rbox_hi[1] << " corresponds to jhi+1 = " << jhi+1 << std::endl;
                        amrex::Print() << "jhi+1 = " << jhi+1 << " is not divisible by ref_ratio in y direction = " << ref_ratio[lev_for_box-1][1] << std::endl;
                        amrex::Error("Adjust in_box_hi in y-direction to be divisible by ref_ratio and try again");
                    }
                    if(klo%ref_ratio[lev_for_box-1][2] != 0){
                        amrex::Print()<< "Requested in_box_lo in z direction = " << rbox_lo[2] << " corresponds to klo = " << klo << std::endl;
                        amrex::Print() << "klo = " << klo << " is not divisible by ref_ratio in z direction = " << ref_ratio[lev_for_box-1][2] << std::endl;
                        amrex::Error("Adjust in_box_lo in z-direction to be divisible by ref_ratio and try again");
                    }
                    if((khi+1)%ref_ratio[lev_for_box-1][2] != 0){
                        amrex::Print()<< "Requested in_box_hi in z direction = " << rbox_hi[2] << " corresponds to khi+1 = " << khi+1 << std::endl;
                        amrex::Print() << "khi+1 = " << khi+1 << " is not divisible by ref_ratio in z direction = " << ref_ratio[lev_for_box-1][2] << std::endl;
                        amrex::Error("Adjust in_box_hi in z-direction to be divisible by ref_ratio and try again");
                    }

                    bool using_pbl = (solverChoice.turbChoice[lev_for_box].pbl_type == PBLType::MYJ      ||
                                      solverChoice.turbChoice[lev_for_box].pbl_type == PBLType::MYNN25   ||
                                      solverChoice.turbChoice[lev_for_box].pbl_type == PBLType::MYNNEDMF ||
                                      solverChoice.turbChoice[lev_for_box].pbl_type == PBLType::YSU      ||
                                      solverChoice.turbChoice[lev_for_box].pbl_type == PBLType::MRF);

                    if ( using_pbl && ( (rbox_lo[2] > plo[2]) || (rbox_hi[2] < phi[2]) ) ) {
                        amrex::Print() << "PBL models need refinement boxes that go from the bottom to the top of the domain for calculation of PBLH" << std::endl;
                        amrex::Print() << "Please set in_box_lo to geometry.prob_lo in z and in_box_hi to geometry.prob_hi in z and try again" << std::endl;
                        amrex::Abort();
                    }

                    boxes_at_level[lev_for_box].push_back(bx);
                    Print() << "Saving in 'boxes at level' as " << bx << std::endl;
                } // lev

                if (solverChoice.init_type == InitType::WRFInput) {
                    if ( (num_files_at_level[lev_for_box] > 0) &&
                         (num_boxes_at_level[lev_for_box] != num_files_at_level[lev_for_box]) ) {
                        amrex::Error("Number of boxes doesn't match number of input files");

                    }
                }

            } else if (num_indx_lo > 0) {

                std::vector<int> box_lo(3), box_hi(3);
                ppr.get("max_level",lev_for_box);
                if (lev_for_box > 0 && lev_for_box <= max_level)
                {
                    if (n_error_buf[0] != IntVect::TheZeroVector()) {
                        amrex::Abort("Don't use n_error_buf > 0 when setting the box explicitly");
                    }

                    ppr.getarr("in_box_lo_indices",box_lo,0,num_indx_lo);
                    ppr.getarr("in_box_hi_indices",box_hi,0,num_indx_hi);

                    if (num_indx_lo < AMREX_SPACEDIM) {
                        box_lo[2] = geom[lev_for_box].Domain().smallEnd(2);
                        box_hi[2] = geom[lev_for_box].Domain().bigEnd(2);
                    }

                    Box bx(IntVect(box_lo[0],box_lo[1],box_lo[2]),IntVect(box_hi[0],box_hi[1],box_hi[2]));
                    const Box& domain = geom[lev_for_box].Domain();

                    if (!domain.contains(bx)) {
                        amrex::Print() << "\n";
                        amrex::Print() << "Box specified       is " << bx << std::endl;
                        amrex::Print() << "But domain at level is " << domain << std::endl;
                        amrex::Error("Specified box doesn't fit in the domain");
                    }

                    const auto* dx  = geom[lev_for_box].CellSize();
                    realbox = RealBox(plo[0]+ box_lo[0]   *dx[0], plo[1]+ box_lo[1]   *dx[1], plo[2]+ box_lo[2]   *dx[2],
                                      plo[0]+(box_hi[0]+1)*dx[0], plo[1]+(box_hi[1]+1)*dx[1], plo[2]+(box_hi[2]+1)*dx[2]);

                    Print() << "Reading " << bx << " at level " << lev_for_box << std::endl;
                    num_boxes_at_level[lev_for_box] += 1;

                    if(box_lo[0]%ref_ratio[lev_for_box-1][0] != 0){
                        amrex::Print()<< "Requested ilo in x-direction : " << box_lo[0] << std::endl;
                        amrex::Print() << "ilo = " << box_lo[0] << " is not divisible by ref_ratio in x direction = " <<
                                          ref_ratio[lev_for_box-1][0] << std::endl;
                        amrex::Error("Adjust in_box_lo_indices in x-direction to be divisible by ref_ratio and try again");
                    }
                    if((box_hi[0]+1)%ref_ratio[lev_for_box-1][0] != 0){
                        amrex::Print()<< "Requested ihi in x-direction : " << box_hi[0] << std::endl;
                        amrex::Print() << "ihi+1 = " << box_hi[0]+1 << " is not divisible by ref_ratio in x direction = " <<
                                          ref_ratio[lev_for_box-1][0] << std::endl;
                        amrex::Error("Adjust in_box_hi_indices in x-direction to be divisible by ref_ratio and try again");
                    }
                     if(box_lo[1]%ref_ratio[lev_for_box-1][1] != 0){
                        amrex::Print()<< "Requested jlo in y-direction : " << box_lo[1] << std::endl;
                        amrex::Print() << "jlo = " << box_lo[1] << " is not divisible by ref_ratio in y direction = " <<
                                          ref_ratio[lev_for_box-1][1] << std::endl;
                        amrex::Error("Adjust in_box_lo_indices in y-direction to be divisible by ref_ratio and try again");
                    }
                    if((box_hi[1]+1)%ref_ratio[lev_for_box-1][1] != 0){
                        amrex::Print()<< "Requested jhi in y-direction : " << box_hi[1] << std::endl;
                        amrex::Print() << "jhi+1 = " << box_hi[1]+1 << " is not divisible by ref_ratio in y direction = " <<
                                          ref_ratio[lev_for_box-1][1] << std::endl;
                        amrex::Error("Adjust in_box_hi_indices in y-direction to be divisible by ref_ratio and try again");
                    }
                    if(box_lo[2]%ref_ratio[lev_for_box-1][2] != 0){
                        amrex::Print()<< "Requested klo in z-direction : " << box_lo[2] << std::endl;
                        amrex::Print() << "klo = " << box_lo[2] << " is not   divisible by ref_ratio in z direction = " <<
                                          ref_ratio[lev_for_box-1][2] << std::endl;
                        amrex::Error("Adjust in_box_lo_indices in z-direction to be divisible by ref_ratio and try again");
                    }
                    if((box_hi[2]+1)%ref_ratio[lev_for_box-1][2] != 0){
                        amrex::Print()<< "Requested khi in z-direction : " << box_hi[2] << std::endl;
                        amrex::Print() << "khi+1 = " << box_hi[2]+1 << " is not divisible by ref_ratio in z direction = " <<
                                          ref_ratio[lev_for_box-1][2] << std::endl;
                        amrex::Error("Adjust in_box_hi_indices in z-direction to be divisible by ref_ratio and try again");
                    }

                    bool using_pbl = (solverChoice.turbChoice[lev_for_box].pbl_type == PBLType::MYJ      ||
                                      solverChoice.turbChoice[lev_for_box].pbl_type == PBLType::MYNN25   ||
                                      solverChoice.turbChoice[lev_for_box].pbl_type == PBLType::MYNNEDMF ||
                                      solverChoice.turbChoice[lev_for_box].pbl_type == PBLType::YSU      ||
                                      solverChoice.turbChoice[lev_for_box].pbl_type == PBLType::MRF);

                    if ( using_pbl && ( (box_lo[2] > 0) || (box_hi[2] < domain.bigEnd(2)) ) ) {
                        amrex::Print() << "PBL models need refinement boxes that go from the bottom to the top of the domain for calculation of PBLH" << std::endl;
                        amrex::Print() << "Please set in_box_lo_indices to 0 in z and in_box_hi_indices to amr.n_cell-1 in z and try again" << std::endl;
                        amrex::Abort();
                    }

                    boxes_at_level[lev_for_box].push_back(bx);
                    Print() << "Saving in 'boxes at level' as " << bx << std::endl;
                } // lev

                if (solverChoice.init_type == InitType::WRFInput) {
                    if ( (num_files_at_level[lev_for_box] > 0) &&
                         (num_boxes_at_level[lev_for_box] != num_files_at_level[lev_for_box]) ) {
                        amrex::Error("Number of boxes doesn't match number of input files");

                    }
                }
            }
            else if (num_indx_lo_crse > 0) {

                std::vector<int> box_lo(3), box_hi(3);
                ppr.get("max_level",lev_for_box);
                if (lev_for_box > 0 && lev_for_box <= max_level)
                {
                    if (n_error_buf[0] != IntVect::TheZeroVector()) {
                        amrex::Abort("Don't use n_error_buf > 0 when setting the box explicitly");
                    }

                    ppr.getarr("in_box_lo_indices_crse",box_lo,0,num_indx_lo_crse);
                    ppr.getarr("in_box_hi_indices_crse",box_hi,0,num_indx_hi_crse);

                    if (num_indx_lo_crse < AMREX_SPACEDIM) {
                        box_lo[2] = geom[lev_for_box-1].Domain().smallEnd(2);
                        box_hi[2] = geom[lev_for_box-1].Domain().bigEnd(2);
                    }

                    Box bx(IntVect(box_lo[0],box_lo[1],box_lo[2]),IntVect(box_hi[0],box_hi[1],box_hi[2]));

                    if (!geom[lev_for_box-1].Domain().contains(bx)) {
                        amrex::Print() << "\n";
                        amrex::Print() << "(Coarse) Box specified       is " << bx << std::endl;
                        amrex::Print() << "But (coarse) domain at level is " << geom[lev_for_box-1].Domain() << std::endl;
                        amrex::Error("Specified box doesn't fit in the domain");
                    }

                    bx.refine(ref_ratio[lev_for_box-1]);

                    const auto* dx  = geom[lev_for_box-1].CellSize();

                    realbox = RealBox(plo[0]+ box_lo[0]   *dx[0], plo[1]+ box_lo[1]   *dx[1], plo[2]+ box_lo[2]   *dx[2],
                                      plo[0]+(box_hi[0]+1)*dx[0], plo[1]+(box_hi[1]+1)*dx[1], plo[2]+(box_hi[2]+1)*dx[2]);

                    Print() << "Reading " << bx << " at level " << lev_for_box << std::endl;
                    num_boxes_at_level[lev_for_box] += 1;
                    bool using_pbl = (solverChoice.turbChoice[lev_for_box].pbl_type == PBLType::MYJ      ||
                                      solverChoice.turbChoice[lev_for_box].pbl_type == PBLType::MYNN25   ||
                                      solverChoice.turbChoice[lev_for_box].pbl_type == PBLType::MYNNEDMF ||
                                      solverChoice.turbChoice[lev_for_box].pbl_type == PBLType::YSU      ||
                                      solverChoice.turbChoice[lev_for_box].pbl_type == PBLType::MRF);

                    const Box& domain = geom[lev_for_box].Domain();
                    if ( using_pbl && ( (box_lo[2] > 0) || (box_hi[2] < domain.bigEnd(2)) ) ) {
                        amrex::Print() << "PBL models need refinement boxes that go from the bottom to the top of the domain for calculation of PBLH" << std::endl;
                        amrex::Print() << "Please set in_box_lo_indices_crse to 0 in z and in_box_hi_indices_crse  to amr.n_cell-1 in z and try again" << std::endl;
                        amrex::Abort();
                    }

                    boxes_at_level[lev_for_box].push_back(bx);
                    Print() << "Saving in 'boxes at level' as " << bx << std::endl;
                } // lev

                if (solverChoice.init_type == InitType::WRFInput) {
                    if ( (num_files_at_level[lev_for_box] > 0) &&
                         (num_boxes_at_level[lev_for_box] != num_files_at_level[lev_for_box]) ) {
                        amrex::Error("Number of boxes doesn't match number of input files");

                    }
                }
            }
            AMRErrorTagInfo info;

            if (realbox.ok()) {
                info.SetRealBox(realbox);
            }

            if (ppr.countval("start_time") > 0) {
                Real ref_min_time; ppr.get("start_time",ref_min_time);
                info.SetMinTime(ref_min_time);
            }

            if (ppr.countval("end_time") > 0) {
                Real ref_max_time; ppr.get("end_time",ref_max_time);
                info.SetMaxTime(ref_max_time);
            }

            if (ppr.countval("max_level") > 0) {
                int ref_max_level; ppr.get("max_level",ref_max_level);
                info.SetMaxLevel(ref_max_level);
            }

            if (ppr.countval("value_greater")) {
                int num_val = ppr.countval("value_greater");
                Vector<Real> value(num_val);
                ppr.getarr("value_greater",value,0,num_val);
                std::string field; ppr.get("field_name",field);
                ref_tags.push_back(AMRErrorTag(value,AMRErrorTag::GREATER,field,info));
            }
            else if (ppr.countval("value_less"))
            {
                int num_val = ppr.countval("value_less");
                Vector<Real> value(num_val);
                ppr.getarr("value_less",value,0,num_val);
                std::string field; ppr.get("field_name",field);
                ref_tags.push_back(AMRErrorTag(value,AMRErrorTag::LESS,field,info));
            }
            else if (ppr.countval("adjacent_difference_greater"))
            {
                int num_val = ppr.countval("adjacent_difference_greater");
                Vector<Real> value(num_val);
                ppr.getarr("adjacent_difference_greater",value,0,num_val);
                std::string field; ppr.get("field_name",field);
                ref_tags.push_back(AMRErrorTag(value,AMRErrorTag::GRAD,field,info));
            }
            else if (realbox.ok())
            {
                ref_tags.push_back(AMRErrorTag(info));
            }
            else if ( (lev_for_box > 0) && (refinement_indicators[i] != "storm_tracker") )
            {
                Abort(std::string("Unrecognized refinement indicator for " + refinement_indicators[i]).c_str());
            }
        } // loop over criteria
    } // if max_level > 0
}

bool
ERF::FindInitialEye(int levc,
                    const MultiFab& mf_cc_vel,
                    const Real velmag_threshold,
                    Real& eye_x, Real& eye_y)
{
    const auto dx = geom[levc].CellSizeArray();
    const auto prob_lo = geom[levc].ProbLoArray();

    Gpu::DeviceVector<Real> d_coords(2, 0.0);
    Gpu::DeviceVector<int>  d_found(1,0);

    Real* d_coords_ptr = d_coords.data();
    int*   d_found_ptr = d_found.data();

    for (MFIter mfi(mf_cc_vel); mfi.isValid(); ++mfi)
    {
        const Box& box = mfi.validbox();
        const Array4<const Real>& vel_arr = mf_cc_vel.const_array(mfi);

        ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            Real magnitude = std::sqrt(vel_arr(i,j,k,0) * vel_arr(i,j,k,0) +
                                       vel_arr(i,j,k,1) * vel_arr(i,j,k,1) +
                                       vel_arr(i,j,k,2) * vel_arr(i,j,k,2));

            magnitude *= 3.6;

            Real z = prob_lo[2] + (k + 0.5) * dx[2];

            // Check if magnitude exceeds threshold
            if (z < 2000. && magnitude > velmag_threshold) {
                // Use atomic operations to set found flag and store coordinates
                Gpu::Atomic::Add(&d_found_ptr[0], 1); // Mark as found

                Real x = prob_lo[0] + (i + 0.5) * dx[0];
                Real y = prob_lo[1] + (j + 0.5) * dx[1];

                // Store coordinates
                Gpu::Atomic::Add(&d_coords_ptr[0],x); // Store x index
                Gpu::Atomic::Add(&d_coords_ptr[1],y); // Store x index
            }
        });
    }

    // Synchronize to ensure all threads complete their execution
    amrex::Gpu::streamSynchronize(); // Wait for all GPU threads to finish

    Vector<int> h_found(1,0);
    Gpu::copy(Gpu::deviceToHost, d_found.begin(), d_found.end(), h_found.begin());
    ParallelAllReduce::Sum(h_found.data(), h_found.size(), ParallelContext::CommunicatorAll());

    // Broadcast coordinates if found
    if (h_found[0] > 0) {
        Vector<Real> h_coords(2,-1e10);
        Gpu::copy(Gpu::deviceToHost, d_coords.begin(), d_coords.end(), h_coords.begin());

        ParallelAllReduce::Sum(h_coords.data(), h_coords.size(), ParallelContext::CommunicatorAll());

        eye_x = h_coords[0]/h_found[0];
        eye_y = h_coords[1]/h_found[0];

    } else {
        // Random large negative numbers so we don't trigger refinement in this case
        eye_x = -1.e20;
        eye_y = -1.e20;
    }

    return (h_found[0] > 0);
}

void
tag_on_distance_from_eye(const Geometry& cgeom, TagBoxArray* tags,
                         const Real eye_x, const Real eye_y, const Real rad_tag)
{
    const auto dx      = cgeom.CellSizeArray();
    const auto prob_lo = cgeom.ProbLoArray();

    for (MFIter mfi(*tags); mfi.isValid(); ++mfi) {
        TagBox& tag = (*tags)[mfi];
        auto tag_arr = tag.array();  // Get device-accessible array

        const Box& tile_box = mfi.tilebox(); // The box for this tile

        ParallelFor(tile_box, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            // Compute cell center coordinates
            Real x = prob_lo[0] + (i + 0.5) * dx[0];
            Real y = prob_lo[1] + (j + 0.5) * dx[1];

            Real dist = std::sqrt((x - eye_x)*(x - eye_x) + (y - eye_y)*(y - eye_y));

            if (dist < rad_tag) {
                tag_arr(i,j,k) = TagBox::SET;
            } else {
                tag_arr(i,j,k) = TagBox::CLEAR;
            }
        });
    }
}

void
ERF::HurricaneTracker(int levc,
                      Real time,
                      const MultiFab& mf_cc_vel,
                      const Real velmag_threshold,
                      TagBoxArray* tags)
{
    bool is_found;

    Real eye_x, eye_y;

    if (time==0.0) {
        is_found = FindInitialEye(levc, mf_cc_vel, velmag_threshold, eye_x, eye_y);
    } else {
        is_found = true;
        const auto& last = hurricane_eye_track_xy.back();
        eye_x = last[0];
        eye_y = last[1];
    }

    if (is_found) {
        Real rad_tag = 4.e5 * std::pow(2, max_level-1-levc);
        tag_on_distance_from_eye(geom[levc], tags, eye_x, eye_y, rad_tag);
    }
}
