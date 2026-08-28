/**
 * \file ERF_InitWindFarm.cpp
 */

#include <ERF_WindFarm.H>
#include <filesystem>
#include <dirent.h>   // For POSIX directory handling
#include <algorithm> // For std::sort
#include <limits>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <cmath>

#include <AMReX_BoxList.H>
#include <AMReX_Gpu.H>
#include <AMReX_iMultiFab.H>

using namespace amrex;

namespace {
AMREX_FORCE_INLINE
amrex::Real wrap_360 (amrex::Real deg)
{
    deg = std::fmod(deg, 360.0);
    if (deg < 0.0) { deg += 360.0; }
    return deg;
}

AMREX_FORCE_INLINE
amrex::Real wrap_180 (amrex::Real deg)
{
    deg = wrap_360(deg);
    if (deg >= 180.0) { deg -= 360.0; }
    return deg;
}

AMREX_FORCE_INLINE
std::string vtk_suffix (int idx)
{
    std::ostringstream oss;
    oss << "_" << std::setw(6) << std::setfill('0') << idx;
    return oss.str();
}

void append_to_series (const std::string& series_name,
                       const std::string& vtk_file,
                       const amrex::Real time)
{
    std::string header =
        "{\n"
        "  \"file-series-version\" : \"1.0\",\n"
        "  \"files\" : [\n";
    std::string footer =
        "  ]\n"
        "}\n";

    std::ostringstream dataset;
    dataset << "    { \"name\" : \"" << vtk_file
            << "\", \"time\" : " << std::setprecision(17) << time << " }\n";

    std::ifstream in(series_name);
    if (!in.good()) {
        std::ofstream out(series_name, std::ios::out | std::ios::trunc);
        out << header << dataset.str() << footer;
        return;
    }

    std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (content.find("\"name\" : \"" + vtk_file + "\"") != std::string::npos) {
        return;
    }

    auto pos = content.rfind("  ]");
    if (pos == std::string::npos) {
        std::ofstream out(series_name, std::ios::out | std::ios::trunc);
        out << header << dataset.str() << footer;
        return;
    }

    const bool has_entries = (content.find("\"name\"") != std::string::npos);
    content.insert(pos, (has_entries ? ",\n" : "") + dataset.str());

    std::ofstream out(series_name, std::ios::out | std::ios::trunc);
    out << content;
}

amrex::Vector<int>
all_turbine_ids (int nturb)
{
    amrex::Vector<int> ids(nturb);
    for (int it = 0; it < nturb; ++it) {
        ids[it] = it;
    }
    return ids;
}

AMREX_FORCE_INLINE
amrex::Real projected_gaussian_sigma (const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& dx,
                                      const amrex::Real nx,
                                      const amrex::Real ny)
{
    return std::abs(dx[0] * nx) + std::abs(dx[1] * ny);
}

AMREX_FORCE_INLINE
void disk_frame_from_face_angle_deg (const amrex::Real disk_face_angle_deg,
                                     amrex::Real& nx,
                                     amrex::Real& ny,
                                     amrex::Real& tx,
                                     amrex::Real& ty)
{
    const amrex::Real theta = disk_face_angle_deg * M_PI / 180.0 - 0.5 * M_PI;
    nx = -std::cos(theta);
    ny = -std::sin(theta);
    tx = -ny;
    ty =  nx;
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

    // Host code must never dereference MultiFab arrays directly in GPU builds.
    // Use a pinned host mirror for the fab data before walking requested columns.
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
                z_columns[idx][k-klo] = z_arr(i,j,k);
            }
        }
    }

    for (int idx = 0; idx < ncol; ++idx) {
        amrex::ParallelAllReduce::Max(z_columns[idx].data(),
                                      z_columns[idx].size(),
                                      amrex::ParallelContext::CommunicatorAll());
    }

    return z_columns;
}
} // namespace

/**
 * Read in the turbine locations in latitude-longitude from windturbines.txt
 * and convert it into x and y coordinates in metres
 *
 * @param lev Integer specifying the current level
 */
void
WindFarm::read_tables (std::string windfarm_loc_table,
                       std::string windfarm_spec_table,
                       bool x_y, bool lat_lon,
                       const Real windfarm_x_shift,
                       const Real windfarm_y_shift)
{
    amrex::Print() << "Reading wind turbine locations table" << "\n";
    read_windfarm_locations_table(windfarm_loc_table,
                                  x_y, lat_lon,
                                  windfarm_x_shift, windfarm_y_shift);

    amrex::Print() << "Reading wind turbine specifications table" << "\n";
    read_windfarm_spec_table(windfarm_spec_table);
}

void
WindFarm::read_windfarm_locations_table (const std::string windfarm_loc_table,
                                         bool x_y, bool lat_lon,
                                         const Real windfarm_x_shift,
                                         const Real windfarm_y_shift)
{
    xloc.clear();
    yloc.clear();
    zloc.clear();
    ground_z.clear();
    m_owner_level.clear();
    m_turbines_on_level.clear();

    if(x_y) {
        init_windfarm_x_y(windfarm_loc_table);
        for (int it = 0; it < static_cast<int>(xloc.size()); ++it) {
            xloc[it] += windfarm_x_shift;
            yloc[it] += windfarm_y_shift;
        }
    }
    else if(lat_lon) {
        init_windfarm_lat_lon(windfarm_loc_table, windfarm_x_shift, windfarm_y_shift);
    }
    else {
        amrex::Abort("Are you using windfarms? For windfarm simulations, the inputs need to have an"
                     " entry erf.windfarm_loc_type which should be either lat_lon or x_y. \n");
    }

    set_turb_loc(xloc, yloc);
}

void
WindFarm::define_classic_ad_owner_levels (int finest_level,
                                          const Vector<Geometry>& geom,
                                          const Vector<BoxArray>& grids,
                                          const Vector<IntVect>& ref_ratio)
{
    const int nturb = static_cast<int>(xloc.size());
    const int nlev = finest_level + 1;
    m_owner_level.assign(nturb, -1);
    m_turbines_on_level.assign(nlev, {});
    ground_z.assign(nturb, geom[0].ProbLo(2));
    zloc = ground_z;

    Vector<Real> face_angles;
    get_disk_face_angles_deg(face_angles);

    Vector<BoxArray> uncovered(nlev);
    for (int lev = 0; lev < nlev; ++lev) {
        const BoxArray* finer = (lev < finest_level) ? &grids[lev+1] : nullptr;
        const IntVect* ratio = (lev < finest_level) ? &ref_ratio[lev] : nullptr;
        uncovered[lev] = build_uncovered_valid_boxes(geom[lev].Domain(), grids[lev], finer, ratio);
    }

    for (int it = 0; it < nturb; ++it) {
        bool assigned = false;
        for (int lev = finest_level; lev >= 0; --lev) {
            const auto dx = geom[lev].CellSizeArray();
            const auto plo = geom[lev].ProbLoArray();
            const Box& domain = geom[lev].Domain();
            const Real psi = (face_angles[it]-90.0)*M_PI/180.0;
            const Real nx = std::cos(psi);
            const Real ny = std::sin(psi);
            const Real e1x = -ny;
            const Real e1y = nx;
            const Real epsn = std::sqrt((nx*dx[0])*(nx*dx[0]) +
                                        (ny*dx[1])*(ny*dx[1]));
            const Real eps1 = std::sqrt((e1x*dx[0])*(e1x*dx[0]) +
                                        (e1y*dx[1])*(e1y*dx[1]));
            const Real eps2 = dx[2];
            const Real hx = 3.0*(std::abs(nx)*epsn + std::abs(e1x)*eps1);
            const Real hy = 3.0*(std::abs(ny)*epsn + std::abs(e1y)*eps1);
            const Real hz = 3.0*eps2;
            const Real xext = std::abs(e1x)*rotor_rad + hx;
            const Real yext = std::abs(e1y)*rotor_rad + hy;
            const Real zhub = geom[0].ProbLo(2) + hub_height;
            const Real zext = rotor_rad + hz;

            IntVect lo(AMREX_D_DECL(
                static_cast<int>(std::floor((xloc[it]-xext-plo[0])/dx[0])),
                static_cast<int>(std::floor((yloc[it]-yext-plo[1])/dx[1])),
                static_cast<int>(std::floor((zhub-zext-plo[2])/dx[2]))));
            IntVect hi(AMREX_D_DECL(
                static_cast<int>(std::floor((xloc[it]+xext-plo[0])/dx[0])),
                static_cast<int>(std::floor((yloc[it]+yext-plo[1])/dx[1])),
                static_cast<int>(std::floor((zhub+zext-plo[2])/dx[2]))));

            bool outside_nonperiodic = false;
            for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                if (!geom[lev].isPeriodic(d) &&
                    (lo[d] < domain.smallEnd(d) || hi[d] > domain.bigEnd(d))) {
                    outside_nonperiodic = true;
                }
            }
            if (outside_nonperiodic) { continue; }

            const Box footprint(lo, hi);
            bool support_is_uncovered = true;
            bool found_periodic_piece = false;
            const Box central_piece = footprint & domain;
            if (central_piece.ok()) {
                support_is_uncovered = uncovered[lev].contains(central_piece);
                found_periodic_piece = true;
            }
            Vector<IntVect> periodic_shifts;
            geom[lev].periodicShift(domain, footprint, periodic_shifts);
            for (const IntVect& shift : periodic_shifts) {
                Box periodic_piece = footprint;
                periodic_piece.shift(shift);
                periodic_piece &= domain;
                if (periodic_piece.ok()) {
                    support_is_uncovered = support_is_uncovered &&
                                           uncovered[lev].contains(periodic_piece);
                    found_periodic_piece = true;
                }
            }

            if (found_periodic_piece && support_is_uncovered) {
                m_owner_level[it] = lev;
                m_turbines_on_level[lev].push_back(it);
                assigned = true;
                break;
            }
        }
        if (!assigned) {
            std::ostringstream msg;
            msg << "ClassicAD turbine " << it
                << " and its complete 3-sigma force support do not lie within one AMR level";
            Abort(msg.str());
        }
    }

    set_turb_zloc(zloc);
    set_turb_disk_angles(face_angles);
}

void
WindFarm::read_windfarm_yaw_file (const std::string& yaw_file)
{
    const int nturb = static_cast<int>(xloc.size());
    m_yaw_offset_deg.assign(nturb, 0.0);

    if (yaw_file.empty() || nturb == 0) {
        return;
    }

    std::ifstream file(yaw_file);
    if (!file.is_open()) {
        amrex::Error("Wind turbine yaw offsets file not found. The file specified by erf.yaw_file = " +
                     yaw_file + " is missing.");
    }

    amrex::Vector<amrex::Real> offsets;
    offsets.reserve(nturb);
    amrex::Real value;
    while (file >> value) {
        offsets.push_back(value);
    }
    file.close();

    if (static_cast<int>(offsets.size()) != nturb) {
        amrex::Error("Yaw offsets file " + yaw_file + " has " + std::to_string(offsets.size()) +
                     " entries but windfarm_loc_table has " + std::to_string(nturb) +
                     " turbines. The yaw file must have exactly one entry per turbine.");
    }

    m_yaw_offset_deg = std::move(offsets);
}

void
WindFarm::define_owner_levels (const amrex::Vector<amrex::Geometry>& geom,
                               const amrex::Vector<amrex::BoxArray>& grids,
                               const amrex::Vector<amrex::DistributionMapping>& dmap,
                               const amrex::Vector<amrex::IntVect>& ref_ratio,
                               const amrex::Vector<std::unique_ptr<amrex::MultiFab>>& z_phys_nd)
{
    const int nturb = static_cast<int>(xloc.size());
    const int nlev = static_cast<int>(geom.size());
    amrex::ignore_unused(dmap);

    m_owner_level.assign(nturb, -1);
    m_turbines_on_level.resize(nlev);
    for (int lev = 0; lev < nlev; ++lev) {
        m_turbines_on_level[lev].clear();
    }
    ground_z.assign(nturb, 0.0);
    zloc.assign(nturb, 0.0);

    if (nturb == 0) {
        set_turb_zloc(zloc);
        return;
    }

    amrex::Vector<amrex::Real> disk_face_angles_deg;
    get_disk_face_angles_deg(disk_face_angles_deg);
    const bool use_gaussian_spreading =
        (m_force_spreading_type == WindFarmSpreadingType::Gaussian);

    amrex::Vector<amrex::BoxArray> uncovered_valid_boxes(nlev);
    amrex::Vector<amrex::Vector<amrex::Vector<amrex::Real>>> z_columns_by_level(nlev);
    amrex::Vector<amrex::Vector<int>> has_column_by_level(nlev);
    amrex::Vector<int> has_ground_column(nturb, 0);

    for (int lev = 0; lev < nlev; ++lev) {
        const amrex::BoxArray* finer_grids = (lev < nlev - 1) ? &grids[lev+1] : nullptr;
        const amrex::IntVect* ratio = (lev < nlev - 1) ? &ref_ratio[lev] : nullptr;
        uncovered_valid_boxes[lev] =
            build_uncovered_valid_boxes(geom[lev].Domain(), grids[lev], finer_grids, ratio);

        const auto dx = geom[lev].CellSizeArray();
        const auto prob_lo = geom[lev].ProbLoArray();
        const amrex::Box& domain = geom[lev].Domain();
        amrex::Vector<int> active_turbines;
        amrex::Vector<int> i_cols;
        amrex::Vector<int> j_cols;

        for (int it = 0; it < nturb; ++it) {
            const int i_center = static_cast<int>(std::floor((xloc[it] - prob_lo[0]) / dx[0]));
            const int j_center = static_cast<int>(std::floor((yloc[it] - prob_lo[1]) / dx[1]));
            if (i_center < domain.smallEnd(0) || i_center > domain.bigEnd(0) ||
                j_center < domain.smallEnd(1) || j_center > domain.bigEnd(1)) {
                continue;
            }

            active_turbines.push_back(it);
            i_cols.push_back(i_center);
            j_cols.push_back(j_center);
        }

        has_column_by_level[lev].assign(nturb, 0);
        z_columns_by_level[lev].resize(nturb);
        const auto gathered_columns =
            gather_z_columns_to_host(geom[lev], z_phys_nd[lev].get(), i_cols, j_cols);

        for (int idx = 0; idx < static_cast<int>(active_turbines.size()); ++idx) {
            const int it = active_turbines[idx];
            has_column_by_level[lev][it] = 1;
            z_columns_by_level[lev][it] = gathered_columns[idx];

            if (lev == 0 && !gathered_columns[idx].empty() &&
                !is_missing_z_value(gathered_columns[idx].front())) {
                has_ground_column[it] = 1;
                ground_z[it] = gathered_columns[idx].front();
                zloc[it] = ground_z[it];
            }
        }
    }

    for (int it = 0; it < nturb; ++it) {
        if (!has_ground_column[it]) {
            std::ostringstream oss;
            oss << "Failed to extract the ground elevation for wind turbine " << it
                << " from the level-0 z_phys_nd column. Turbine center is at (x,y)=("
                << xloc[it] << ", " << yloc[it] << ").";
            amrex::Abort(oss.str());
        }

        bool assigned = false;
        const amrex::Real z_lo = ground_z[it] + hub_height - rotor_rad;
        const amrex::Real z_hi = ground_z[it] + hub_height + rotor_rad;

        for (int lev = nlev - 1; lev >= 0; --lev) {
            const auto dx = geom[lev].CellSizeArray();
            const auto prob_lo = geom[lev].ProbLoArray();
            const amrex::Box& domain = geom[lev].Domain();

            const int i_center = static_cast<int>(std::floor((xloc[it] - prob_lo[0]) / dx[0]));
            const int j_center = static_cast<int>(std::floor((yloc[it] - prob_lo[1]) / dx[1]));
            if (i_center < domain.smallEnd(0) || i_center > domain.bigEnd(0) ||
                j_center < domain.smallEnd(1) || j_center > domain.bigEnd(1)) {
                amrex::Print() << "WindFarm owner skip: turbine " << it
                               << " level " << lev
                               << " center index (" << i_center << ", " << j_center
                               << ") lies outside domain " << domain
                               << "\n";
                continue;
            }

            if (!has_column_by_level[lev][it]) {
                amrex::Print() << "WindFarm owner skip: turbine " << it
                               << " level " << lev
                               << " has no gathered z column at center index ("
                               << i_center << ", " << j_center << ")"
                               << "\n";
                continue;
            }

            const amrex::Vector<amrex::Real>& z_nodes = z_columns_by_level[lev][it];

            amrex::Real x_extent = rotor_rad;
            amrex::Real y_extent = rotor_rad;
            if (use_gaussian_spreading) {
                amrex::Real nx = 0.0, ny = 0.0, tx = 0.0, ty = 0.0;
                disk_frame_from_face_angle_deg(disk_face_angles_deg[it], nx, ny, tx, ty);
                const amrex::Real sigma = projected_gaussian_sigma(dx, nx, ny);
                const amrex::Real support = m_force_spreading_nsigma * sigma;
                x_extent = std::abs(tx) * rotor_rad + std::abs(nx) * support;
                y_extent = std::abs(ty) * rotor_rad + std::abs(ny) * support;
            }

            const int ilo = static_cast<int>(std::floor((xloc[it] - x_extent - prob_lo[0]) / dx[0]));
            const int ihi = static_cast<int>(std::floor((xloc[it] + x_extent - prob_lo[0]) / dx[0]));
            const int jlo = static_cast<int>(std::floor((yloc[it] - y_extent - prob_lo[1]) / dx[1]));
            const int jhi = static_cast<int>(std::floor((yloc[it] + y_extent - prob_lo[1]) / dx[1]));

            if (ilo < domain.smallEnd(0) || ihi > domain.bigEnd(0) ||
                jlo < domain.smallEnd(1) || jhi > domain.bigEnd(1)) {
                amrex::Print() << "WindFarm owner skip: turbine " << it
                               << " level " << lev
                               << " horizontal rotor box [(" << ilo << ", " << jlo
                               << "), (" << ihi << ", " << jhi
                               << ")] exceeds domain " << domain
                               << "\n";
                continue;
            }

            int first_valid_node = -1;
            int last_valid_node = -1;
            for (int n = 0; n < static_cast<int>(z_nodes.size()); ++n) {
                if (!is_missing_z_value(z_nodes[n])) {
                    if (first_valid_node < 0) { first_valid_node = n; }
                    last_valid_node = n;
                }
            }

            if (first_valid_node < 0 || last_valid_node <= first_valid_node) {
                amrex::Print() << "WindFarm owner skip: turbine " << it
                               << " level " << lev
                               << " gathered z column has insufficient valid nodes"
                               << " first_valid_node=" << first_valid_node
                               << " last_valid_node=" << last_valid_node
                               << "\n";
                continue;
            }

            const amrex::Real z_avail_lo = z_nodes[first_valid_node];
            const amrex::Real z_avail_hi = z_nodes[last_valid_node];
            if (z_lo < z_avail_lo || z_hi > z_avail_hi) {
                amrex::Print() << "WindFarm owner skip: turbine " << it
                               << " level " << lev
                               << " rotor vertical range [" << z_lo << ", " << z_hi
                               << "] is not contained in available z column range ["
                               << z_avail_lo << ", " << z_avail_hi << "]"
                               << "\n";
                continue;
            }

            int klo = -1;
            int khi = -1;
            bool has_gap_in_rotor_extent = false;
            for (int k = domain.smallEnd(2); k <= domain.bigEnd(2); ++k) {
                const int lo_idx = k - domain.smallEnd(2);
                const int hi_idx = lo_idx + 1;
                if (lo_idx < 0 || hi_idx >= static_cast<int>(z_nodes.size())) {
                    continue;
                }

                const amrex::Real cell_lo = z_nodes[lo_idx];
                const amrex::Real cell_hi = z_nodes[hi_idx];

                // Once we have already found the rotor interval and the next valid cell
                // starts above rotor top, there is nothing else to learn from higher k.
                if (!is_missing_z_value(cell_lo) && klo >= 0 && cell_lo > z_hi) {
                    break;
                }

                if (is_missing_z_value(cell_lo) || is_missing_z_value(cell_hi)) {
                    // A missing node matters only if it interrupts the rotor interval
                    // before we have passed the rotor top on this level.
                    if (klo >= 0 && khi >= 0) {
                        has_gap_in_rotor_extent = true;
                        break;
                    }
                    continue;
                }

                if (cell_hi >= z_lo && cell_lo <= z_hi) {
                    if (klo < 0) { klo = k; }
                    khi = k;
                }
            }

            if (klo < 0 || has_gap_in_rotor_extent) {
                amrex::Print() << "WindFarm owner skip: turbine " << it
                               << " level " << lev
                               << " could not form a contiguous rotor k-range"
                               << " klo=" << klo
                               << " khi=" << khi
                               << " has_gap_in_rotor_extent=" << has_gap_in_rotor_extent
                               << "\n";
                continue;
            }

            const amrex::Box footprint_box(amrex::IntVect(AMREX_D_DECL(ilo, jlo, klo)),
                                           amrex::IntVect(AMREX_D_DECL(ihi, jhi, khi)));
            amrex::Print() << "WindFarm owner check: turbine " << it
                           << " level " << lev
                           << " footprint_box=" << footprint_box
                           << " uncovered_valid_boxes=" << uncovered_valid_boxes[lev]
                           << " z_avail=[" << z_avail_lo << ", " << z_avail_hi << "]"
                           << " rotor_z=[" << z_lo << ", " << z_hi << "]"
                           << "\n";
            if (uncovered_valid_boxes[lev].contains(footprint_box)) {
                m_owner_level[it] = lev;
                m_turbines_on_level[lev].push_back(it);
                assigned = true;
                break;
            }
        }

        if (!assigned) {
            std::ostringstream oss;
            oss << "Wind turbine " << it
                << " could not be assigned to an AMR owner level. "
	                << "Rotor footprint bounds are x=[" << xloc[it] - rotor_rad << ", " << xloc[it] + rotor_rad
	                << "], y=[" << yloc[it] - rotor_rad << ", " << yloc[it] + rotor_rad
	                << "], z=[ground_z + " << hub_height - rotor_rad
	                << ", ground_z + " << hub_height + rotor_rad
	                << "]. The rotor"
	                << (use_gaussian_spreading ? " plus Gaussian force-support" : "")
	                << " extent is not fully contained within any uncovered valid AMR region."
	                << " windfarm_force_spreading="
	                << (use_gaussian_spreading ? "Gaussian" : "None")
	                << " windfarm_spreading_nsigma=" << m_force_spreading_nsigma;
	            amrex::Abort(oss.str());
	        }
    }

    set_turb_zloc(zloc);
}

void
WindFarm::get_turbine_refinement_geometry (int turbine_id,
                                           amrex::Real& x,
                                           amrex::Real& y,
                                           amrex::Real& zhub,
                                           amrex::Real& diameter,
                                           amrex::Real& nx,
                                           amrex::Real& ny,
                                           amrex::Real& tx,
                                           amrex::Real& ty)
{
    AMREX_ALWAYS_ASSERT(turbine_id >= 0);
    AMREX_ALWAYS_ASSERT(turbine_id < static_cast<int>(xloc.size()));

    amrex::Vector<amrex::Real> disk_face_angles_deg;
    get_disk_face_angles_deg(disk_face_angles_deg);

    AMREX_ALWAYS_ASSERT(turbine_id < static_cast<int>(disk_face_angles_deg.size()));
    AMREX_ALWAYS_ASSERT(turbine_id < static_cast<int>(ground_z.size()));

    const amrex::Real theta_rad = disk_face_angles_deg[turbine_id] * M_PI / 180.0 - 0.5 * M_PI;

    x = xloc[turbine_id];
    y = yloc[turbine_id];
    zhub = ground_z[turbine_id] + hub_height;
    diameter = 2.0 * rotor_rad;

    nx = -std::cos(theta_rad);
    ny = -std::sin(theta_rad);

    tx = -ny;
    ty =  nx;
}

void
WindFarm::init_windfarm_lat_lon (const std::string windfarm_loc_table,
                                 const Real windfarm_x_shift,
                                 const Real windfarm_y_shift)
{

    // Read turbine locations from windturbines.txt
    std::ifstream file(windfarm_loc_table);
    if (!file.is_open()) {
        amrex::Error("Wind turbines location table not found. Either the inputs is missing the"
                     " erf.windfarm_loc_table entry or the file specified in the entry " + windfarm_loc_table + " is missing.");
    }
    // Vector of vectors to store the matrix
    Vector<Real> lat, lon;
    Real value1, value2, value3;

    while (file >> value1 >> value2 >> value3) {

        if(std::fabs(value1) > 90.0) {
            amrex::Error("The value of latitude for entry " + std::to_string(lat.size() + 1) +
                         " in " + windfarm_loc_table + " should be within -90 and 90");
        }

        if(std::fabs(value2) > 180.0) {
            amrex::Error("The value of longitude for entry " + std::to_string(lat.size() + 1) +
                         " in " + windfarm_loc_table + " should be within -180 and 180");
        }
        lat.push_back(value1);
        lon.push_back(value2);
    }
    file.close();

    Real rad_earth = 6371.0e3; // Radius of the earth
    Real m_per_deg_lat = rad_earth*2.0*M_PI/(2.0*180.0);

    // Find the coordinates of average of min and max of the farm
    // Rotate about that point
    ParmParse pp("erf");
    std::string fname_usgs;
    auto valid_fname_USGS = pp.query("terrain_file_name_USGS",fname_usgs);
    Real lon_ref, lat_ref;

    if (valid_fname_USGS) {
        std::ifstream file_usgs(fname_usgs);
        file_usgs >> lon_ref >> lat_ref;
        file_usgs.close();
        lon_ref = lon_ref*M_PI/180.0;
        lat_ref = lat_ref*M_PI/180.0;
    } else {
        Real lat_min = *std::min_element(lat.begin(), lat.end());
        Real lon_min = *std::min_element(lon.begin(), lon.end());

        lon_ref = lon_min*M_PI/180.0;
        lat_ref = lat_min*M_PI/180.0;
    }


    for(int it=0;it<lat.size();it++){
        lat[it] = lat[it]*M_PI/180.0;
        lon[it] = lon[it]*M_PI/180.0;
        Real delta_lat = (lat[it] - lat_ref);
        Real delta_lon = (lon[it] - lon_ref);

        Real term1 = std::pow(sin(delta_lat/2.0),2);
        Real term2 = cos(lat[it])*cos(lat_ref)*std::pow(sin(delta_lon/2.0),2);
        Real dist =  2.0*rad_earth*std::asin(std::sqrt(term1 + term2));
        Real dy_turb = delta_lat * m_per_deg_lat * 180.0/M_PI ;

        if(dist<dy_turb){
            if(std::fabs(dist-dy_turb)<1e-8){
                dist=dy_turb;
            }
            else{
                Abort("The value of dist is less than dy_turb "+ std::to_string(dist) + " " + std::to_string(dy_turb));
            }
        }
        Real tmp = std::pow(dist,2) - std::pow(dy_turb,2);

        if(std::fabs(tmp)<1e-8){
            tmp = 0.0;
        }
        Real dx_turb = std::sqrt(tmp);


        if(delta_lon >= 0.0) {
            xloc.push_back(dx_turb);
        }
        else {
            xloc.push_back(-dx_turb);
        }
        yloc.push_back(dy_turb);
    }

    for(int it = 0;it<xloc.size(); it++){
        xloc[it] = xloc[it] + windfarm_x_shift;
        yloc[it] = yloc[it] + windfarm_y_shift;
    }
}

void
WindFarm::init_windfarm_x_y (const std::string windfarm_loc_table)
{
    // Read turbine locations from windturbines.txt
    std::ifstream file(windfarm_loc_table);
    if (!file.is_open()) {
        amrex::Error("Wind turbines location table not found. Either the inputs is missing the"
                     " erf.windfarm_loc_table entry or the file specified in the entry " + windfarm_loc_table + " is missing.");
    }
    // Vector of vectors to store the matrix
    Real value1, value2;

    while (file >> value1 >> value2) {
        value1 = value1 + 1e-3;
        value2 = value2 + 1e-3;
        xloc.push_back(value1);
        yloc.push_back(value2);
    }
    file.close();
}


void
WindFarm::read_windfarm_spec_table (const std::string windfarm_spec_table)
{
    //The first line is the number of pairs entries for the power curve and thrust coefficient.
    //The second line gives first the height in meters of the turbine hub, second, the diameter in
    //meters of the rotor, third the standing thrust coefficient, and fourth the nominal power of
    //the turbine in MW.
    //The remaining lines contain the three values of: wind speed, thrust coefficient, and power production in kW.

     // Read turbine data from wind-turbine-1.tbl
    std::ifstream file_turb_table(windfarm_spec_table);
    if (!file_turb_table.is_open()) {
        Error("Wind farm specifications table not found. Either the inputs is missing the "
                      "erf.windfarm_spec_table entry or the file specified in the entry - " + windfarm_spec_table + " is missing.");
    }
    else {
        Print() << "Reading in wind farm specifications table: " << windfarm_spec_table << "\n";
    }

    int nlines;
    file_turb_table >> nlines;
    wind_speed.resize(nlines);
    thrust_coeff.resize(nlines);
    power.resize(nlines);

    bool wake_rotation = false;
    Real tsr = 9.0;
    Real C_P_prime = 0.9;
    get_wake_rotation_params(wake_rotation, tsr, C_P_prime);

    std::string spec_line;
    std::getline(file_turb_table, spec_line);
    while (std::getline(file_turb_table, spec_line)) {
        if (spec_line.find_first_not_of(" \t\r") != std::string::npos) {
            break;
        }
    }

    if (spec_line.find_first_not_of(" \t\r") == std::string::npos) {
        Abort("Could not read the second line in " + windfarm_spec_table + ". Aborting.....");
    }

    std::istringstream spec_stream(spec_line);
    amrex::Vector<Real> spec_entries;
    Real spec_entry;
    while (spec_stream >> spec_entry) {
        spec_entries.push_back(spec_entry);
    }

    if (spec_entries.size() != 4 && spec_entries.size() != 6) {
        Abort("The second line in " + windfarm_spec_table +
              " must contain either 4 values (hub_height diameter standing_thrust_coeff rated_power) "
              "or 6 values with optional tsr and C_P_prime appended. Aborting.....");
    }

    hub_height = spec_entries[0];
    Real rotor_dia = spec_entries[1];
    thrust_coeff_standing = spec_entries[2];
    nominal_power = spec_entries[3];
    if (spec_entries.size() == 6) {
        tsr = spec_entries[4];
        C_P_prime = spec_entries[5];
    } else if (wake_rotation) {
        tsr = 9.0;
        C_P_prime = 0.9;
    }

    rotor_rad = rotor_dia*0.5;
    if(rotor_rad > hub_height) {
        Abort("The blade length is more than the hub height. Check the second line in wind-turbine-1.tbl. Aborting.....");
    }
    if(thrust_coeff_standing > 1.0) {
        Abort("The standing thrust coefficient is greater than 1. Check the second line in wind-turbine-1.tbl. Aborting.....");
    }

    for(int iline=0;iline<nlines;iline++){
        file_turb_table >> wind_speed[iline] >> thrust_coeff[iline] >> power[iline];
        if(thrust_coeff[iline] > 1.0) {
            Abort("The thrust coefficient is greater than 1. Check wind-turbine-1.tbl. Aborting.....");
        }
    }
    file_turb_table.close();

    set_turb_spec(rotor_rad, hub_height, thrust_coeff_standing,
                  wind_speed, thrust_coeff, power);
    set_wake_rotation_params(wake_rotation, tsr, C_P_prime);

}

void
WindFarm::read_windfarm_blade_table (const std::string windfarm_blade_table)
{
    bld_rad_loc.clear();
    bld_twist.clear();
    bld_chord.clear();

    std::ifstream filename(windfarm_blade_table);
    std::string line;
    Real temp, var1, var2, var3;
    if (!filename.is_open()) {
        Error("You are using a generalized actuator disk model based on blade element theory. This needs info of blades."
                      " An entry erf.windfarm_blade_table is needed. Either the entry is missing or the file specified"
                      " in the entry - " + windfarm_blade_table + " is missing.");
    }
    else {
        Print() << "Reading in wind farm blade table: " << windfarm_blade_table << "\n";

        // First 6 lines are comments

        for (int i = 0; i < 6; ++i) {
            if (std::getline(filename, line)) {  // Read one line into the array
            }
        }

        while(filename >> var1 >> temp >> temp >> temp >> var2 >> var3 >> temp) {
            bld_rad_loc.push_back(var1);
            bld_twist.push_back(var2);
            bld_chord.push_back(var3);
            //int idx = bld_rad_loc.size()-1;
            //printf("Values are = %0.15g %0.15g %0.15g\n", bld_rad_loc[idx], bld_twist[idx], bld_chord[idx]);
        }
        set_blade_spec(bld_rad_loc, bld_twist, bld_chord);
        n_bld_sections = bld_rad_loc.size();
    }
}

void
WindFarm::read_windfarm_spec_table_extra (const std::string windfarm_spec_table_extra)
{
    velocity.clear();
    C_P.clear();
    C_T.clear();
    rotor_RPM.clear();
    blade_pitch.clear();

    // Open the file
    std::ifstream file(windfarm_spec_table_extra);

    // Check if file opened successfully
    if (!file.is_open()) {
        Abort("Error: You are using generalized wind farms option. This requires an input file erf.windfarm_spec_table_extra."
              " Either this entry is missing in the inputs or the file specified -" + windfarm_spec_table_extra + " does"
              " not exist. Exiting...");
    } else {
        printf("Reading in windfarm_spec_table_extra %s", windfarm_spec_table_extra.c_str());
    }

    // Ignore the first line (header)
    std::string header;
    std::getline(file, header);

    // Variables to hold each row's values
    double V, Cp, Ct, rpm, pitch, temp;

    // Read the file row by row
    while (file >> V) {
        char comma;  // To ignore the commas
        file >> comma >> Cp >> comma >> Ct >> comma >> temp >> comma >> temp >> comma
             >> temp >> comma >> rpm >> comma >> pitch >> comma >> temp;

        velocity.push_back(V);
        C_P.push_back(Cp);
        C_T.push_back(Ct);
        rotor_RPM.push_back(rpm);
        blade_pitch.push_back(pitch);
    }

    set_turb_spec_extra(velocity, C_P, C_T, rotor_RPM, blade_pitch);
}


void
WindFarm::read_windfarm_airfoil_tables (const std::string windfarm_airfoil_tables,
                                        const std::string windfarm_blade_table)
{
    bld_airfoil_aoa.clear();
    bld_airfoil_Cl.clear();
    bld_airfoil_Cd.clear();

    DIR* dir;
    struct dirent* entry;
    std::vector<std::string> files;

    // Check if directory exists
    if ((dir = opendir(windfarm_airfoil_tables.c_str())) == nullptr) {
       Abort("You are using a generalized actuator disk model based on blade element theory. This needs info of airfoil"
             " cross sections over the span of the blade. There needs to be an entry erf.airfoil_tables which is the directory that"
             " contains the angle of attack, Cl, Cd data for each airfoil cross-section. Either the entry is missing or the directory specified"
             " in the entry - " + windfarm_airfoil_tables + " is missing. Exiting...");
    }

    // Loop through directory entries and collect filenames
    while ((entry = readdir(dir)) != nullptr) {
        // Skip special directory entries "." and ".."
        if (std::string(entry->d_name) == "." || std::string(entry->d_name) == "..") {
            continue;
        }
        files.emplace_back(windfarm_airfoil_tables + "/" + entry->d_name);  // Add file path to vector
    }

    // Close the directory
    closedir(dir);

    if (files.empty()) {
        Abort("It seems the directory containing the info of airfoil cross sections of the blades - " + windfarm_airfoil_tables +
              " is empty. Exiting...");
    }

    if(files.size() != static_cast<long double>(n_bld_sections)) {
        printf("There are %d airfoil sections in the last column of %s. But the number"
               " of files in %s is only %ld.\n", n_bld_sections, windfarm_blade_table.c_str(),
                windfarm_airfoil_tables.c_str(), files.size());
        Abort("The number of blade sections from " + windfarm_blade_table + " should match the number of"
              " files in " + windfarm_airfoil_tables + ". Exiting...");
    }

    // Sort filenames in lexicographical (alphabetical) order
    std::sort(files.begin(), files.end());

    // Process each file
    int count = 0;
    bld_airfoil_aoa.resize(n_bld_sections);
    bld_airfoil_Cl.resize(n_bld_sections);
    bld_airfoil_Cd.resize(n_bld_sections);
    for (const auto& filePath : files) {
        std::ifstream filename(filePath.c_str());

        if (!filename.is_open()) {
            std::cerr << "Failed to open file: " << filePath << std::endl;
            continue;  // Move on to the next file
        }

           std::cout << "Reading file: " << filePath << std::endl;

        std::string line;
        for (int i = 0; i < 54; ++i) {
            if (std::getline(filename, line)) {  // Read one line into the array
            }
        }

        Real var1, var2, var3, temp;

        while(filename >> var1 >> var2 >> var3 >> temp) {
            bld_airfoil_aoa[count].push_back(var1);
            bld_airfoil_Cl[count].push_back(var2);
            bld_airfoil_Cd[count].push_back(var3);
            //int idx = bld_airfoil_aoa.size()-1;
            //printf("Values are = %0.15g %0.15g %0.15g\n", bld_airfoil_aoa[idx], bld_airfoil_Cl[idx], bld_airfoil_Cd[idx]);
        }
        count++;
    }

    set_blade_airfoil_spec(bld_airfoil_aoa, bld_airfoil_Cl, bld_airfoil_Cd);
}

void
WindFarm::fill_Nturb_multifab (const Geometry& geom,
                               MultiFab& mf_Nturb,
                               std::unique_ptr<MultiFab>& /*z_phys_nd*/,
                               const amrex::Vector<int>& turbine_ids)
{
    amrex::Gpu::DeviceVector<Real> d_xloc(xloc.size());
    amrex::Gpu::DeviceVector<Real> d_yloc(yloc.size());
    amrex::Gpu::DeviceVector<int> d_turbine_ids(turbine_ids.size());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, xloc.begin(), xloc.end(), d_xloc.begin());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, yloc.begin(), yloc.end(), d_yloc.begin());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, turbine_ids.begin(), turbine_ids.end(), d_turbine_ids.begin());

    Real* d_xloc_ptr       = d_xloc.data();
    Real* d_yloc_ptr       = d_yloc.data();
    int* d_turbine_ids_ptr = d_turbine_ids.data();

    mf_Nturb.setVal(0);
    if (turbine_ids.empty()) {
        return;
    }

    int i_lo = geom.Domain().smallEnd(0); int i_hi = geom.Domain().bigEnd(0);
    int j_lo = geom.Domain().smallEnd(1); int j_hi = geom.Domain().bigEnd(1);
    auto dx = geom.CellSizeArray();
    if(dx[0]<= 1e-3 or dx[1]<=1e-3 or dx[2]<= 1e-3) {
        Abort("The value of mesh spacing for wind farm parametrization cannot be less than 1e-3 m. "
              "It should be usually of order 1 m");
    }
    auto ProbLoArr = geom.ProbLoArray();
    int num_turb = turbine_ids.size();

     // Initialize wind farm
    for ( MFIter mfi(mf_Nturb,TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx     = mfi.tilebox();
        auto  Nturb_array = mf_Nturb.array(mfi);
        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            int li = amrex::min(amrex::max(i, i_lo), i_hi);
            int lj = amrex::min(amrex::max(j, j_lo), j_hi);

            Real x1 = ProbLoArr[0] + li*dx[0];
            Real x2 = ProbLoArr[0] + (li+1)*dx[0];
            Real y1 = ProbLoArr[1] + lj*dx[1];
            Real y2 = ProbLoArr[1] + (lj+1)*dx[1];

            for(int idx = 0; idx < num_turb; ++idx){
                const int it = d_turbine_ids_ptr[idx];
                if( d_xloc_ptr[it]+1e-3 > x1 and d_xloc_ptr[it]+1e-3 < x2 and
                    d_yloc_ptr[it]+1e-3 > y1 and d_yloc_ptr[it]+1e-3 < y2){
                    Nturb_array(i,j,k,0) = Nturb_array(i,j,k,0) + 1;
                }
            }
        });
    }
}

void
WindFarm::fill_SMark_multifab_mesoscale_models (const Geometry& geom,
                                                MultiFab& mf_SMark,
                                                const MultiFab& mf_Nturb,
                                                std::unique_ptr<MultiFab>& z_phys_nd)
{
    mf_SMark.setVal(-1.0);

    Real d_hub_height = hub_height;

    amrex::Gpu::DeviceVector<Real> d_xloc(xloc.size());
    amrex::Gpu::DeviceVector<Real> d_yloc(yloc.size());
    amrex::Gpu::DeviceVector<Real> d_zloc(xloc.size());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, xloc.begin(), xloc.end(), d_xloc.begin());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, yloc.begin(), yloc.end(), d_yloc.begin());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, zloc.begin(), zloc.end(), d_zloc.begin());

    int i_lo = geom.Domain().smallEnd(0); int i_hi = geom.Domain().bigEnd(0);
    int j_lo = geom.Domain().smallEnd(1); int j_hi = geom.Domain().bigEnd(1);
    int k_lo = geom.Domain().smallEnd(2); int k_hi = geom.Domain().bigEnd(2);

    auto dx = geom.CellSizeArray();
    auto ProbLoArr = geom.ProbLoArray();

     // Initialize wind farm
    for ( MFIter mfi(mf_SMark,TilingIfNotGPU()); mfi.isValid(); ++mfi) {

        const Box& gbx    = mfi.growntilebox(1);
        auto  SMark_array = mf_SMark.array(mfi);
        auto  Nturb_array = mf_Nturb.array(mfi);
        const Array4<const Real>& z_nd_arr = (z_phys_nd) ? z_phys_nd->const_array(mfi) : Array4<Real>{};
        int k0 = gbx.smallEnd()[2];

        ParallelFor(gbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            if(Nturb_array(i,j,k,0) > 0) {
                int li = amrex::min(amrex::max(i, i_lo), i_hi);
                int lj = amrex::min(amrex::max(j, j_lo), j_hi);
                int lk = amrex::min(amrex::max(k, k_lo), k_hi);

                Real z1 = (z_nd_arr) ? z_nd_arr(li,lj,lk) : ProbLoArr[2] + lk * dx[2];
                Real z2 = (z_nd_arr) ? z_nd_arr(li,lj,lk+1) : ProbLoArr[2] + (lk+1) * dx[2];

                Real zturb;
                if(z_nd_arr) {
                    zturb = z_nd_arr(li,lj,k0) + d_hub_height;
                } else {
                    zturb = d_hub_height;
                }
                if(zturb+1e-3 > z1 and zturb+1e-3 < z2) {
                    SMark_array(i,j,k,0) = 1.0;
                }
            }
        });
    }
}

void
WindFarm::fill_SMark_multifab (const Geometry& geom,
                               MultiFab& mf_SMark,
                               MultiFab& mf_RMask,
                               const Real& sampling_distance_by_D,
                               const Real& turb_disk_angle,
                               std::unique_ptr<MultiFab>& z_phys_cc,
                               const amrex::Vector<int>& turbine_ids)
{
    amrex::Gpu::DeviceVector<Real> d_xloc(xloc.size());
    amrex::Gpu::DeviceVector<Real> d_yloc(yloc.size());
    amrex::Gpu::DeviceVector<Real> d_zloc(xloc.size());
    amrex::Gpu::DeviceVector<int> d_turbine_ids(turbine_ids.size());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, xloc.begin(), xloc.end(), d_xloc.begin());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, yloc.begin(), yloc.end(), d_yloc.begin());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, zloc.begin(), zloc.end(), d_zloc.begin());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, turbine_ids.begin(), turbine_ids.end(), d_turbine_ids.begin());

    Real d_rotor_rad = rotor_rad;
    Real d_hub_height = hub_height;
    Real d_sampling_distance = sampling_distance_by_D*2.0*rotor_rad;

    Real* d_xloc_ptr     = d_xloc.data();
    Real* d_yloc_ptr     = d_yloc.data();
    Real* d_zloc_ptr     = d_zloc.data();
    int* d_turbine_ids_ptr = d_turbine_ids.data();

    mf_SMark.setVal(-1.0);
    mf_RMask.setVal(-1.0);
    if (turbine_ids.empty()) {
        return;
    }

    int i_lo = geom.Domain().smallEnd(0); int i_hi = geom.Domain().bigEnd(0);
    int j_lo = geom.Domain().smallEnd(1); int j_hi = geom.Domain().bigEnd(1);
    int k_lo = geom.Domain().smallEnd(2); int k_hi = geom.Domain().bigEnd(2);
    auto dx = geom.CellSizeArray();
    auto ProbLoArr = geom.ProbLoArray();
    int num_turb = turbine_ids.size();

    Real theta = turb_disk_angle*M_PI/180.0-0.5*M_PI;

    set_turb_disk_angle(theta);
    my_turb_disk_angle = theta;
    {
        amrex::Vector<amrex::Real> theta_rad(xloc.size(), theta);
        set_turb_disk_angles(theta_rad);
    }

    Real nx = -std::cos(theta);
    Real ny = -std::sin(theta);
    Real tx = -ny;
    Real ty =  nx;
    const bool use_gaussian_spreading =
        (m_force_spreading_type == WindFarmSpreadingType::Gaussian);
    const Real spread_sigma = projected_gaussian_sigma(dx, nx, ny);
    const Real spread_support = m_force_spreading_nsigma * spread_sigma;

     // Initialize wind farm
    for ( MFIter mfi(mf_SMark,TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& gbx      = mfi.growntilebox(1);
        auto  SMark_array = mf_SMark.array(mfi);
        auto  RMask_array = mf_RMask.array(mfi);

        const Array4<const Real>& z_cc_arr = (z_phys_cc) ? z_phys_cc->const_array(mfi) : Array4<Real>{};

        ParallelFor(gbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            int ii = amrex::min(amrex::max(i, i_lo), i_hi);
            int jj = amrex::min(amrex::max(j, j_lo), j_hi);
            int kk = amrex::min(amrex::max(k, k_lo), k_hi);

            // The x and y extents of the current mesh cell

            Real x1 = ProbLoArr[0] + ii*dx[0];
            Real x2 = ProbLoArr[0] + (ii+1)*dx[0];
            Real y1 = ProbLoArr[1] + jj*dx[1];
            Real y2 = ProbLoArr[1] + (jj+1)*dx[1];

            // The mesh cell centered z value

            Real z = (z_cc_arr) ? z_cc_arr(ii,jj,kk) : ProbLoArr[2] + (kk+0.5) * dx[2];

            int turb_indices_overlap[2];
            int check_int = 0;
            for(int idx = 0; idx < num_turb; ++idx){
                const int it = d_turbine_ids_ptr[idx];
                Real x0 = d_xloc_ptr[it] + d_sampling_distance*nx;
                Real y0 = d_yloc_ptr[it] + d_sampling_distance*ny;

                Real z0 = 0.0;
                if(z_cc_arr) {
                    z0 = d_zloc_ptr[it];
                }

                bool is_cell_marked = find_if_marked(x1, x2, y1, y2, x0, y0,
                                                     nx, ny, d_hub_height+z0, d_rotor_rad, z);
                if(is_cell_marked) {
                    SMark_array(i,j,k,0) = it;
                }
                x0 = d_xloc_ptr[it];
                y0 = d_yloc_ptr[it];

                is_cell_marked = find_if_marked(x1, x2, y1, y2, x0, y0,
                                                nx, ny, d_hub_height+z0, d_rotor_rad, z);
                Real normal_dist = 0.0;
                Real radial_dist = -1.0;
                if (use_gaussian_spreading) {
                    Real xc = ProbLoArr[0] + (ii+0.5_rt)*dx[0];
                    Real yc = ProbLoArr[1] + (jj+0.5_rt)*dx[1];
                    Real zhub = d_hub_height + z0;
                    Real dxp = xc - x0;
                    Real dyp = yc - y0;
                    Real dzp = z - zhub;
                    normal_dist = dxp*nx + dyp*ny;
                    Real tangential_dist = dxp*tx + dyp*ty;
                    radial_dist = std::sqrt(tangential_dist*tangential_dist + dzp*dzp);
                    is_cell_marked = (radial_dist <= d_rotor_rad &&
                                      std::abs(normal_dist) <= spread_support);
                }
                if(is_cell_marked) {
                    SMark_array(i,j,k,1) = it;
                    if (!use_gaussian_spreading) {
                        Real xc = ProbLoArr[0] + (ii+0.5_rt)*dx[0];
                        Real yc = ProbLoArr[1] + (jj+0.5_rt)*dx[1];
                        Real zhub = d_hub_height + z0;
                        Real dxp = xc - x0;
                        Real dyp = yc - y0;
                        Real dzp = z - zhub;
                        normal_dist = dxp*nx + dyp*ny;
                        Real rx = dxp - normal_dist*nx;
                        Real ry = dyp - normal_dist*ny;
                        radial_dist = std::sqrt(rx*rx + ry*ry + dzp*dzp);
                    }
                    RMask_array(i,j,k,0) = radial_dist;
                    RMask_array(i,j,k,1) = normal_dist;
                    turb_indices_overlap[check_int] = it;
                    check_int++;
                    if(check_int > 1){
                        printf("Actuator disks with indices %d and %d are overlapping\n",
                               turb_indices_overlap[0],turb_indices_overlap[1]);
                        amrex::Error("Actuator disks are overlapping. Visualize actuator_disks.vtk "
                        " and check the windturbine locations input file. Exiting..");
                    }
                }
            }
        });
    }
}

void
WindFarm::write_turbine_locations_vtk ()
{
    if (ParallelDescriptor::IOProcessor()){
        FILE* file_turbloc_vtk;
        file_turbloc_vtk = fopen("turbine_locations.vtk","w");
        fprintf(file_turbloc_vtk, "%s\n","# vtk DataFile Version 3.0");
        fprintf(file_turbloc_vtk, "%s\n","Wind turbine locations");
        fprintf(file_turbloc_vtk, "%s\n","ASCII");
        fprintf(file_turbloc_vtk, "%s\n","DATASET POLYDATA");
        fprintf(file_turbloc_vtk, "%s %ld %s\n", "POINTS", xloc.size(), "float");
        for(int it=0; it<xloc.size(); it++){
            fprintf(file_turbloc_vtk, "%0.15g %0.15g %0.15g\n", xloc[it], yloc[it], hub_height);
        }
        fclose(file_turbloc_vtk);
    }
}


void
WindFarm::write_actuator_disks_vtk (const Geometry& geom,
                                    const Real& sampling_distance_by_D)
{

    Real sampling_distance = sampling_distance_by_D*2.0*rotor_rad;

    if (ParallelDescriptor::IOProcessor()){
        amrex::Vector<amrex::Real> turb_disk_angles;
        get_turb_disk_angles(turb_disk_angles);
        const bool have_per_turb_angles =
            (!turb_disk_angles.empty() && turb_disk_angles.size() == xloc.size());

        FILE *file_actuator_disks_all, *file_actuator_disks_in_dom, *file_averaging_disks_in_dom;
        file_actuator_disks_all = fopen("actuator_disks_all.vtk","w");
        fprintf(file_actuator_disks_all, "%s\n","# vtk DataFile Version 3.0");
        fprintf(file_actuator_disks_all, "%s\n","Actuator Disks");
        fprintf(file_actuator_disks_all, "%s\n","ASCII");
        fprintf(file_actuator_disks_all, "%s\n","DATASET POLYDATA");

        file_actuator_disks_in_dom = fopen("actuator_disks_in_dom.vtk","w");
        fprintf(file_actuator_disks_in_dom, "%s\n","# vtk DataFile Version 3.0");
        fprintf(file_actuator_disks_in_dom, "%s\n","Actuator Disks");
        fprintf(file_actuator_disks_in_dom, "%s\n","ASCII");
        fprintf(file_actuator_disks_in_dom, "%s\n","DATASET POLYDATA");

        file_averaging_disks_in_dom = fopen("averaging_disks_in_dom.vtk","w");
        fprintf(file_averaging_disks_in_dom, "%s\n","# vtk DataFile Version 3.0");
        fprintf(file_averaging_disks_in_dom, "%s\n","Actuator Disks");
        fprintf(file_averaging_disks_in_dom, "%s\n","ASCII");
        fprintf(file_averaging_disks_in_dom, "%s\n","DATASET POLYDATA");


        int npts = 100;
        fprintf(file_actuator_disks_all, "%s %ld %s\n", "POINTS", xloc.size()*npts, "float");
        auto ProbLoArr = geom.ProbLoArray();
        auto ProbHiArr = geom.ProbHiArray();
        int num_turb_in_dom = 0;

        // Find the number of turbines inside the specified computational domain

        for(int it=0; it<xloc.size(); it++){
            Real x = xloc[it];
            Real y = yloc[it];
            if(x > ProbLoArr[0] and x < ProbHiArr[0] and y > ProbLoArr[1] and y < ProbHiArr[1]) {
                num_turb_in_dom++;
            }
        }
        fprintf(file_actuator_disks_in_dom, "%s %ld %s\n", "POINTS", static_cast<long int>(num_turb_in_dom*npts), "float");
        fprintf(file_averaging_disks_in_dom, "%s %ld %s\n", "POINTS", static_cast<long int>(num_turb_in_dom*npts), "float");

        for(int it=0; it<xloc.size(); it++){
            const Real theta_face = have_per_turb_angles ? turb_disk_angles[it] : my_turb_disk_angle;
            const Real nx  = std::cos(theta_face + 0.5*M_PI);
            const Real ny  = std::sin(theta_face + 0.5*M_PI);
            const Real nx1 = -std::cos(theta_face);
            const Real ny1 = -std::sin(theta_face);

            for(int pt=0;pt<100;pt++){
                Real x, y, z, xavg, yavg;
                Real theta = 2.0*M_PI/npts*pt;
                x = xloc[it] + rotor_rad*cos(theta)*nx;
                y = yloc[it] + rotor_rad*cos(theta)*ny;
                z = hub_height + zloc[it] + rotor_rad*sin(theta);

                xavg = xloc[it] + sampling_distance*nx1 + rotor_rad*cos(theta)*nx;
                yavg = yloc[it] + sampling_distance*ny1 + rotor_rad*cos(theta)*ny;

                fprintf(file_actuator_disks_all, "%0.15g %0.15g %0.15g\n", x, y, z);
                if(xloc[it] > ProbLoArr[0] and xloc[it] < ProbHiArr[0] and yloc[it] > ProbLoArr[1] and yloc[it] < ProbHiArr[1]) {
                    fprintf(file_actuator_disks_in_dom, "%0.15g %0.15g %0.15g\n", x, y, z);
                    fprintf(file_averaging_disks_in_dom, "%0.15g %0.15g %0.15g\n", xavg, yavg, z);
                }
            }
        }
        fprintf(file_actuator_disks_all, "%s %ld %ld\n", "LINES", xloc.size()*(npts-1), static_cast<long int>(xloc.size()*(npts-1)*3));
        fprintf(file_actuator_disks_in_dom, "%s %ld %ld\n", "LINES", static_cast<long int>(num_turb_in_dom*(npts-1)), static_cast<long int>(num_turb_in_dom*(npts-1)*3));
        fprintf(file_averaging_disks_in_dom, "%s %ld %ld\n", "LINES", static_cast<long int>(num_turb_in_dom*(npts-1)), static_cast<long int>(num_turb_in_dom*(npts-1)*3));
        for(int it=0; it<xloc.size(); it++){
            for(int pt=0;pt<99;pt++){
                fprintf(file_actuator_disks_all, "%ld %ld %ld\n",
                                             static_cast<long int>(2),
                                             static_cast<long int>(it*npts+pt),
                                             static_cast<long int>(it*npts+pt+1));
            }
        }
         for(int it=0; it<num_turb_in_dom; it++){
            for(int pt=0;pt<99;pt++){
                fprintf(file_actuator_disks_in_dom, "%ld %ld %ld\n",
                                             static_cast<long int>(2),
                                             static_cast<long int>(it*npts+pt),
                                             static_cast<long int>(it*npts+pt+1));
            }
        }

         for(int it=0; it<num_turb_in_dom; it++){
            for(int pt=0;pt<99;pt++){
                fprintf(file_averaging_disks_in_dom, "%ld %ld %ld\n",
                                             static_cast<long int>(2),
                                             static_cast<long int>(it*npts+pt),
                                             static_cast<long int>(it*npts+pt+1));
            }
        }

        fclose(file_actuator_disks_all);
        fclose(file_actuator_disks_in_dom);
        fclose(file_averaging_disks_in_dom);
    }
}

void
WindFarm::init_dynamic_yaw (const amrex::Real disk_angle0_deg,
                            const amrex::Real yaw_period,
                            const amrex::Real windfarm_start_time)
{
    if (xloc.empty()) {
        return;
    }

    m_dynamic_yaw_enabled = true;
    m_disk_angle0_deg = disk_angle0_deg;
    m_yaw_period = yaw_period;
    m_tau_yaw = yaw_period/3.0;
    m_yaw_update_idx = 0;

    const int nturb = static_cast<int>(xloc.size());
    m_yaw_angle_deg.assign(nturb, 0.0);
    m_yaw_cmd_deg.assign(nturb, 0.0);
    m_next_update_time.assign(nturb, windfarm_start_time + yaw_period);
    m_sum_u.assign(nturb, 0.0);
    m_sum_v.assign(nturb, 0.0);
    m_sum_t.assign(nturb, 0.0);
    m_yaw_angle_geom_deg.assign(nturb, 0.0);

    // Initialize per-turbine disk angles in the actuator model.
    amrex::Vector<amrex::Real> disk_face_angles_deg;
    get_disk_face_angles_deg(disk_face_angles_deg);

    amrex::Vector<amrex::Real> theta_rad(nturb, 0.0);
    for (int it = 0; it < nturb; ++it) {
        theta_rad[it] = disk_face_angles_deg[it] * M_PI/180.0 - 0.5*M_PI;
    }
    set_turb_disk_angles(theta_rad);
}

void
WindFarm::sample_upstream_uv (const amrex::MultiFab& U_old,
                              const amrex::MultiFab& V_old,
                              const amrex::MultiFab& mf_SMark,
                              amrex::Vector<amrex::Real>& u_sum,
                              amrex::Vector<amrex::Real>& v_sum,
                              amrex::Vector<amrex::Real>& counts) const
{
    const int nturb = static_cast<int>(xloc.size());
    u_sum.assign(nturb, 0.0);
    v_sum.assign(nturb, 0.0);
    counts.assign(nturb, 0.0);

    if (nturb == 0) {
        return;
    }

    Gpu::DeviceVector<Real> d_u_sum(nturb, 0.0);
    Gpu::DeviceVector<Real> d_v_sum(nturb, 0.0);
    Gpu::DeviceVector<Real> d_counts(nturb, 0.0);

    Real* d_u_sum_ptr = d_u_sum.data();
    Real* d_v_sum_ptr = d_v_sum.data();
    Real* d_counts_ptr = d_counts.data();

    for (MFIter mfi(mf_SMark, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        auto SMark_array = mf_SMark.array(mfi);
        auto u_vel = U_old.array(mfi);
        auto v_vel = V_old.array(mfi);

        Box tbx = mfi.nodaltilebox(0);
        ParallelFor(tbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            if (SMark_array(i,j,k,0) != -1.0) {
                int turb_index = static_cast<int>(SMark_array(i,j,k,0));
                Gpu::Atomic::Add(&d_u_sum_ptr[turb_index], u_vel(i,j,k));
                Gpu::Atomic::Add(&d_v_sum_ptr[turb_index], v_vel(i,j,k));
                Gpu::Atomic::Add(&d_counts_ptr[turb_index], 1.0);
            }
        });
    }

    Gpu::copy(Gpu::deviceToHost, d_u_sum.begin(), d_u_sum.end(), u_sum.begin());
    Gpu::copy(Gpu::deviceToHost, d_v_sum.begin(), d_v_sum.end(), v_sum.begin());
    Gpu::copy(Gpu::deviceToHost, d_counts.begin(), d_counts.end(), counts.begin());

    amrex::ParallelAllReduce::Sum(u_sum.data(), u_sum.size(),
                                  amrex::ParallelContext::CommunicatorAll());
    amrex::ParallelAllReduce::Sum(v_sum.data(), v_sum.size(),
                                  amrex::ParallelContext::CommunicatorAll());
    amrex::ParallelAllReduce::Sum(counts.data(), counts.size(),
                                  amrex::ParallelContext::CommunicatorAll());
}

void
WindFarm::accumulate_yaw_samples (const amrex::Vector<amrex::Real>& u_mean,
                                  const amrex::Vector<amrex::Real>& v_mean,
                                  const amrex::Vector<amrex::Real>& counts,
                                  const amrex::Real dt)
{
    const int nturb = static_cast<int>(xloc.size());
    for (int it = 0; it < nturb; ++it) {
        if (counts[it] > 0.0) {
            m_sum_u[it] += u_mean[it] * dt;
            m_sum_v[it] += v_mean[it] * dt;
            m_sum_t[it] += dt;
        }
    }
}

bool
WindFarm::update_yaw_controller (const amrex::Real time,
                                 const amrex::Real dt)
{
    return update_yaw_controller(all_turbine_ids(static_cast<int>(xloc.size())), time, dt);
}

bool
WindFarm::update_yaw_controller (const amrex::Vector<int>& turbine_ids,
                                 const amrex::Real time,
                                 const amrex::Real dt)
{
    if (!m_dynamic_yaw_enabled) {
        return false;
    }

    const amrex::Real step_end = time + dt;
    const amrex::Real eps = 1e-12;
    bool did_cmd_update = false;

    // Command update (boxcar over yaw_period)
    for (int idx = 0; idx < turbine_ids.size(); ++idx) {
        const int it = turbine_ids[idx];
        if (step_end + eps >= m_next_update_time[it]) {
            if (m_sum_t[it] > 0.0) {
                const amrex::Real u_bar = m_sum_u[it] / m_sum_t[it];
                const amrex::Real v_bar = m_sum_v[it] / m_sum_t[it];
                const amrex::Real phi_deg = std::atan2(v_bar, u_bar) * 180.0 / M_PI;
                const amrex::Real disk_face_cmd_deg = wrap_360(phi_deg + 90.0);
                m_yaw_cmd_deg[it] = wrap_180(disk_face_cmd_deg - m_disk_angle0_deg);
            }
            m_sum_u[it] = 0.0;
            m_sum_v[it] = 0.0;
            m_sum_t[it] = 0.0;

            while (m_next_update_time[it] <= step_end + eps) {
                m_next_update_time[it] += m_yaw_period;
            }

            did_cmd_update = true;
        }
    }

    // Smooth yaw actuator (first-order) every timestep
    for (int idx = 0; idx < turbine_ids.size(); ++idx) {
        const int it = turbine_ids[idx];
        const amrex::Real e = wrap_180(m_yaw_cmd_deg[it] - m_yaw_angle_deg[it]);
        m_yaw_angle_deg[it] = wrap_180(m_yaw_angle_deg[it] + (dt / m_tau_yaw) * e);
    }

    // Push per-turbine disk angles down into the actuator model each timestep.
    amrex::Vector<amrex::Real> disk_face_angles_deg;
    get_disk_face_angles_deg(disk_face_angles_deg);
    const int nturb = static_cast<int>(xloc.size());
    amrex::Vector<amrex::Real> theta_rad(nturb, 0.0);
    for (int it = 0; it < nturb; ++it) {
        theta_rad[it] = disk_face_angles_deg[it] * M_PI/180.0 - 0.5*M_PI;
    }
    set_turb_disk_angles(theta_rad);

    if (did_cmd_update) {
        ++m_yaw_update_idx;
        write_yaw_angles_time_series(step_end);
    }

    return did_cmd_update;
}

bool
WindFarm::should_rebuild_SMark (const amrex::Real dx_eff) const
{
    return should_rebuild_SMark(all_turbine_ids(static_cast<int>(xloc.size())), dx_eff);
}

bool
WindFarm::should_rebuild_SMark (const amrex::Vector<int>& turbine_ids,
                                const amrex::Real dx_eff) const
{
    if (!m_dynamic_yaw_enabled) {
        return false;
    }
    const amrex::Real alpha_dx = m_yaw_alpha * dx_eff;
    for (int idx = 0; idx < turbine_ids.size(); ++idx) {
        const int it = turbine_ids[idx];
        const amrex::Real dpsi_deg = wrap_180(m_yaw_angle_deg[it] - m_yaw_angle_geom_deg[it]);
        const amrex::Real rim_disp = rotor_rad * std::abs(dpsi_deg) * M_PI/180.0;
        if (rim_disp > alpha_dx) {
            return true;
        }
    }
    return false;
}

void
WindFarm::commit_yaw_geometry ()
{
    commit_yaw_geometry(all_turbine_ids(static_cast<int>(xloc.size())));
}

void
WindFarm::commit_yaw_geometry (const amrex::Vector<int>& turbine_ids)
{
    for (int idx = 0; idx < turbine_ids.size(); ++idx) {
        const int it = turbine_ids[idx];
        m_yaw_angle_geom_deg[it] = m_yaw_angle_deg[it];
    }
}

void
WindFarm::get_disk_face_angles_deg (amrex::Vector<amrex::Real>& disk_face_angles_deg) const
{
#ifdef ERF_USE_PARTICLES
    const auto* classic = dynamic_cast<const ClassicAD*>(m_windfarm_model[0].get());
    if (classic != nullptr && classic->dynamic_yaw_enabled()) {
        disk_face_angles_deg = classic->disk_face_angles_deg();
        return;
    }
#endif
    const int nturb = static_cast<int>(xloc.size());
    disk_face_angles_deg.resize(nturb);
    for (int it = 0; it < nturb; ++it) {
        const amrex::Real dyn =
            (m_dynamic_yaw_enabled && it < static_cast<int>(m_yaw_angle_deg.size())) ? m_yaw_angle_deg[it] : 0.0;
        const amrex::Real off =
            (it < static_cast<int>(m_yaw_offset_deg.size())) ? m_yaw_offset_deg[it] : 0.0;
        disk_face_angles_deg[it] = wrap_360(m_disk_angle0_deg + dyn + off);
    }
}

void
WindFarm::fill_SMark_multifab_dynamic (const amrex::Geometry& geom,
                                       amrex::MultiFab& mf_SMark,
                                       amrex::MultiFab& mf_RMask,
                                       const amrex::Real& sampling_distance_by_D,
                                       const amrex::Vector<amrex::Real>& disk_face_angles_deg,
                                       std::unique_ptr<amrex::MultiFab>& z_phys_cc,
                                       const amrex::Vector<int>& turbine_ids)
{
    amrex::Gpu::DeviceVector<Real> d_xloc(xloc.size());
    amrex::Gpu::DeviceVector<Real> d_yloc(yloc.size());
    amrex::Gpu::DeviceVector<Real> d_zloc(yloc.size());
    amrex::Gpu::DeviceVector<int> d_turbine_ids(turbine_ids.size());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, xloc.begin(), xloc.end(), d_xloc.begin());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, yloc.begin(), yloc.end(), d_yloc.begin());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, zloc.begin(), zloc.end(), d_zloc.begin());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, turbine_ids.begin(), turbine_ids.end(), d_turbine_ids.begin());

    const int num_turb = static_cast<int>(turbine_ids.size());
    const int nturb_global = static_cast<int>(xloc.size());

    amrex::Vector<amrex::Real> theta_rad(nturb_global, 0.0);
    amrex::Vector<amrex::Real> nx_h(nturb_global, 0.0);
    amrex::Vector<amrex::Real> ny_h(nturb_global, 0.0);
    amrex::Vector<amrex::Real> tx_h(nturb_global, 0.0);
    amrex::Vector<amrex::Real> ty_h(nturb_global, 0.0);
    for (int it = 0; it < nturb_global; ++it) {
        theta_rad[it] = disk_face_angles_deg[it] * M_PI/180.0 - 0.5*M_PI;
        nx_h[it] = -std::cos(theta_rad[it]);
        ny_h[it] = -std::sin(theta_rad[it]);
        tx_h[it] = -ny_h[it];
        ty_h[it] =  nx_h[it];
    }
    set_turb_disk_angles(theta_rad);

    amrex::Gpu::DeviceVector<Real> d_nx(nturb_global);
    amrex::Gpu::DeviceVector<Real> d_ny(nturb_global);
    amrex::Gpu::DeviceVector<Real> d_tx(nturb_global);
    amrex::Gpu::DeviceVector<Real> d_ty(nturb_global);
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, nx_h.begin(), nx_h.end(), d_nx.begin());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, ny_h.begin(), ny_h.end(), d_ny.begin());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, tx_h.begin(), tx_h.end(), d_tx.begin());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, ty_h.begin(), ty_h.end(), d_ty.begin());

    Real* d_xloc_ptr     = d_xloc.data();
    Real* d_yloc_ptr     = d_yloc.data();
    Real* d_zloc_ptr     = d_zloc.data();
    Real* d_nx_ptr       = d_nx.data();
    Real* d_ny_ptr       = d_ny.data();
    Real* d_tx_ptr       = d_tx.data();
    Real* d_ty_ptr       = d_ty.data();
    int* d_turbine_ids_ptr = d_turbine_ids.data();

    Real d_rotor_rad = rotor_rad;
    Real d_hub_height = hub_height;
    Real d_sampling_distance = sampling_distance_by_D*2.0*rotor_rad;

    mf_SMark.setVal(-1.0);
    mf_RMask.setVal(-1.0);
    if (turbine_ids.empty()) {
        return;
    }

    int i_lo = geom.Domain().smallEnd(0); int i_hi = geom.Domain().bigEnd(0);
    int j_lo = geom.Domain().smallEnd(1); int j_hi = geom.Domain().bigEnd(1);
    int k_lo = geom.Domain().smallEnd(2); int k_hi = geom.Domain().bigEnd(2);
    auto dx = geom.CellSizeArray();
    auto ProbLoArr = geom.ProbLoArray();
    const bool use_gaussian_spreading =
        (m_force_spreading_type == WindFarmSpreadingType::Gaussian);
    const Real spreading_nsigma = m_force_spreading_nsigma;

    for (MFIter mfi(mf_SMark, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& gbx = mfi.growntilebox(1);
        auto SMark_array = mf_SMark.array(mfi);
        auto RMask_array = mf_RMask.array(mfi);
        const Array4<const Real>& z_cc_arr = (z_phys_cc) ? z_phys_cc->const_array(mfi) : Array4<Real>{};

        ParallelFor(gbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            int ii = amrex::min(amrex::max(i, i_lo), i_hi);
            int jj = amrex::min(amrex::max(j, j_lo), j_hi);
            int kk = amrex::min(amrex::max(k, k_lo), k_hi);

            Real x1 = ProbLoArr[0] + ii*dx[0];
            Real x2 = ProbLoArr[0] + (ii+1)*dx[0];
            Real y1 = ProbLoArr[1] + jj*dx[1];
            Real y2 = ProbLoArr[1] + (jj+1)*dx[1];

            Real z = (z_cc_arr) ? z_cc_arr(ii,jj,kk) : ProbLoArr[2] + (kk+0.5) * dx[2];

            int turb_indices_overlap[2];
            int check_int = 0;

            for (int idx = 0; idx < num_turb; ++idx) {
                const int it = d_turbine_ids_ptr[idx];
	                Real nx = d_nx_ptr[it];
	                Real ny = d_ny_ptr[it];
	                Real tx = d_tx_ptr[it];
	                Real ty = d_ty_ptr[it];

                Real x0 = d_xloc_ptr[it] + d_sampling_distance*nx;
                Real y0 = d_yloc_ptr[it] + d_sampling_distance*ny;

                Real z0 = 0.0;
                if (z_cc_arr) {
                    z0 = d_zloc_ptr[it];
                }

                bool is_cell_marked = find_if_marked(x1, x2, y1, y2, x0, y0,
                                                     nx, ny, d_hub_height+z0, d_rotor_rad, z);
                if (is_cell_marked) {
                    SMark_array(i,j,k,0) = it;
                }

                x0 = d_xloc_ptr[it];
                y0 = d_yloc_ptr[it];

	                is_cell_marked = find_if_marked(x1, x2, y1, y2, x0, y0,
	                                                nx, ny, d_hub_height+z0, d_rotor_rad, z);
	                Real normal_dist = 0.0;
	                Real radial_dist = -1.0;
	                if (use_gaussian_spreading) {
	                    Real xc = ProbLoArr[0] + (ii+0.5_rt)*dx[0];
	                    Real yc = ProbLoArr[1] + (jj+0.5_rt)*dx[1];
	                    Real zhub = d_hub_height + z0;
	                    Real dxp = xc - x0;
	                    Real dyp = yc - y0;
	                    Real dzp = z - zhub;
	                    normal_dist = dxp*nx + dyp*ny;
	                    Real tangential_dist = dxp*tx + dyp*ty;
	                    radial_dist = std::sqrt(tangential_dist*tangential_dist + dzp*dzp);
	                    Real sigma = std::abs(dx[0]*nx) + std::abs(dx[1]*ny);
	                    is_cell_marked = (radial_dist <= d_rotor_rad &&
	                                      std::abs(normal_dist) <= spreading_nsigma*sigma);
	                }
	                if (is_cell_marked) {
	                    SMark_array(i,j,k,1) = it;
	                    if (!use_gaussian_spreading) {
	                        Real xc = ProbLoArr[0] + (ii+0.5_rt)*dx[0];
	                        Real yc = ProbLoArr[1] + (jj+0.5_rt)*dx[1];
	                        Real zhub = d_hub_height + z0;
	                        Real dxp = xc - x0;
	                        Real dyp = yc - y0;
	                        Real dzp = z - zhub;
	                        normal_dist = dxp*nx + dyp*ny;
	                        Real rx = dxp - normal_dist*nx;
	                        Real ry = dyp - normal_dist*ny;
	                        radial_dist = std::sqrt(rx*rx + ry*ry + dzp*dzp);
	                    }
	                    RMask_array(i,j,k,0) = radial_dist;
	                    RMask_array(i,j,k,1) = normal_dist;
	                    turb_indices_overlap[check_int] = it;
                    check_int++;
                    if (check_int > 1) {
                        printf("Actuator disks with indices %d and %d are overlapping\n",
                               turb_indices_overlap[0], turb_indices_overlap[1]);
                        amrex::Error("Actuator disks are overlapping. Visualize actuator_disks.vtk "
                                     " and check the windturbine locations input file. Exiting..");
                    }
                }
            }
        });
    }
}

void
WindFarm::write_dynamic_vtk_series (const amrex::Geometry& geom,
                                    const amrex::Real& sampling_distance_by_D,
                                    const amrex::Vector<amrex::Real>& disk_face_angles_deg,
                                    const amrex::Real time)
{
    if (!ParallelDescriptor::IOProcessor()) {
        return;
    }

    const std::string suf = vtk_suffix(m_yaw_update_idx);

    const std::string f_turb = "turbine_locations" + suf + ".vtk";
    const std::string f_all  = "actuator_disks_all" + suf + ".vtk";
    const std::string f_dom  = "actuator_disks_in_dom" + suf + ".vtk";
    const std::string f_avg  = "averaging_disks_in_dom" + suf + ".vtk";

    // turbine locations
    {
        FILE* fp = fopen(f_turb.c_str(), "w");
        fprintf(fp, "%s\n","# vtk DataFile Version 3.0");
        fprintf(fp, "%s\n","Wind turbine locations");
        fprintf(fp, "%s\n","ASCII");
        fprintf(fp, "%s\n","DATASET POLYDATA");
        fprintf(fp, "%s %ld %s\n", "POINTS", xloc.size(), "float");
        for (int it = 0; it < xloc.size(); ++it) {
            fprintf(fp, "%0.15g %0.15g %0.15g\n", xloc[it], yloc[it], hub_height + zloc[it]);
        }
        fclose(fp);
    }

    Real sampling_distance = sampling_distance_by_D*2.0*rotor_rad;
    int npts = 100;

    auto ProbLoArr = geom.ProbLoArray();
    auto ProbHiArr = geom.ProbHiArray();
    int num_turb_in_dom = 0;
    for (int it = 0; it < xloc.size(); ++it) {
        Real x = xloc[it];
        Real y = yloc[it];
        if (x > ProbLoArr[0] and x < ProbHiArr[0] and y > ProbLoArr[1] and y < ProbHiArr[1]) {
            num_turb_in_dom++;
        }
    }

    FILE *fp_all = fopen(f_all.c_str(), "w");
    FILE *fp_dom = fopen(f_dom.c_str(), "w");
    FILE *fp_avg = fopen(f_avg.c_str(), "w");

    fprintf(fp_all, "%s\n","# vtk DataFile Version 3.0");
    fprintf(fp_all, "%s\n","Actuator Disks");
    fprintf(fp_all, "%s\n","ASCII");
    fprintf(fp_all, "%s\n","DATASET POLYDATA");

    fprintf(fp_dom, "%s\n","# vtk DataFile Version 3.0");
    fprintf(fp_dom, "%s\n","Actuator Disks");
    fprintf(fp_dom, "%s\n","ASCII");
    fprintf(fp_dom, "%s\n","DATASET POLYDATA");

    fprintf(fp_avg, "%s\n","# vtk DataFile Version 3.0");
    fprintf(fp_avg, "%s\n","Actuator Disks");
    fprintf(fp_avg, "%s\n","ASCII");
    fprintf(fp_avg, "%s\n","DATASET POLYDATA");

    fprintf(fp_all, "%s %ld %s\n", "POINTS", xloc.size()*npts, "float");
    fprintf(fp_dom, "%s %ld %s\n", "POINTS", static_cast<long int>(num_turb_in_dom*npts), "float");
    fprintf(fp_avg, "%s %ld %s\n", "POINTS", static_cast<long int>(num_turb_in_dom*npts), "float");

    for (int it = 0; it < xloc.size(); ++it) {
        const Real disk_face_deg = disk_face_angles_deg[it];
        const Real theta = disk_face_deg*M_PI/180.0 - 0.5*M_PI;

        Real ehx = std::cos(theta + 0.5*M_PI);
        Real ehy = std::sin(theta + 0.5*M_PI);
        Real nx  = -std::cos(theta);
        Real ny  = -std::sin(theta);

        bool in_dom = (xloc[it] > ProbLoArr[0] and xloc[it] < ProbHiArr[0] and
                       yloc[it] > ProbLoArr[1] and yloc[it] < ProbHiArr[1]);
        for (int pt = 0; pt < npts; ++pt) {
            Real ang = 2.0*M_PI/npts*pt;
            Real x = xloc[it] + rotor_rad*std::cos(ang)*ehx;
            Real y = yloc[it] + rotor_rad*std::cos(ang)*ehy;
            Real z = hub_height + zloc[it] + rotor_rad*std::sin(ang);

            Real xavg = xloc[it] + sampling_distance*nx + rotor_rad*std::cos(ang)*ehx;
            Real yavg = yloc[it] + sampling_distance*ny + rotor_rad*std::cos(ang)*ehy;

            fprintf(fp_all, "%0.15g %0.15g %0.15g\n", x, y, z);
            if (in_dom) {
                fprintf(fp_dom, "%0.15g %0.15g %0.15g\n", x, y, z);
                fprintf(fp_avg, "%0.15g %0.15g %0.15g\n", xavg, yavg, z);
            }
        }
    }

    fprintf(fp_all, "%s %ld %ld\n", "LINES", xloc.size()*(npts-1), static_cast<long int>(xloc.size()*(npts-1)*3));
    fprintf(fp_dom, "%s %ld %ld\n", "LINES", static_cast<long int>(num_turb_in_dom*(npts-1)), static_cast<long int>(num_turb_in_dom*(npts-1)*3));
    fprintf(fp_avg, "%s %ld %ld\n", "LINES", static_cast<long int>(num_turb_in_dom*(npts-1)), static_cast<long int>(num_turb_in_dom*(npts-1)*3));

    for (int it = 0; it < xloc.size(); ++it) {
        for (int pt = 0; pt < npts-1; ++pt) {
            fprintf(fp_all, "%ld %ld %ld\n",
                    static_cast<long int>(2),
                    static_cast<long int>(it*npts+pt),
                    static_cast<long int>(it*npts+pt+1));
        }
    }

    for (int it = 0; it < num_turb_in_dom; ++it) {
        for (int pt = 0; pt < npts-1; ++pt) {
            fprintf(fp_dom, "%ld %ld %ld\n",
                    static_cast<long int>(2),
                    static_cast<long int>(it*npts+pt),
                    static_cast<long int>(it*npts+pt+1));
            fprintf(fp_avg, "%ld %ld %ld\n",
                    static_cast<long int>(2),
                    static_cast<long int>(it*npts+pt),
                    static_cast<long int>(it*npts+pt+1));
        }
    }

    fclose(fp_all);
    fclose(fp_dom);
    fclose(fp_avg);

    append_to_series("turbine_locations.vtk.series", f_turb, time);
    append_to_series("actuator_disks_all.vtk.series", f_all, time);
    append_to_series("actuator_disks_in_dom.vtk.series", f_dom, time);
    append_to_series("averaging_disks_in_dom.vtk.series", f_avg, time);
}

void
WindFarm::write_yaw_angles_time_series (const amrex::Real time) const
{
    if (!ParallelDescriptor::IOProcessor()) {
        return;
    }

    static std::ofstream file("yaw_angles_SimpleAD.txt", std::ios::app);
    static bool wrote_header = false;
    if (!file.is_open()) {
        amrex::Abort("Could not open file to write yaw angles (yaw_angles_SimpleAD.txt)");
    }

    if (!wrote_header) {
        file << "# time";
        for (int it = 0; it < m_yaw_angle_deg.size(); ++it) {
            file << " yaw_turb" << it;
        }
        file << "\n";
        wrote_header = true;
    }

    file << std::setprecision(17) << time;
    for (int it = 0; it < m_yaw_angle_deg.size(); ++it) {
        file << " " << m_yaw_angle_deg[it];
    }
    file << "\n";
    file.flush();
}

void
WindFarm::write_dynamic_yaw_state (const std::string& checkpointname) const
{
    if (!ParallelDescriptor::IOProcessor()) {
        return;
    }
    if (!m_dynamic_yaw_enabled) {
        return;
    }

    std::ofstream out(checkpointname + "/DynamicYawState", std::ios::out | std::ios::trunc);
    if (!out.good()) {
        amrex::Abort("Failed to open DynamicYawState for checkpoint write");
    }

    out << std::setprecision(17);
    out << "DynamicYawState_v1\n";
    out << xloc.size() << "\n";
    out << m_disk_angle0_deg << " " << m_yaw_period << " " << m_tau_yaw << " " << m_yaw_alpha << " " << m_yaw_update_idx << "\n";
    for (int it = 0; it < xloc.size(); ++it) {
        out << m_yaw_angle_deg[it] << " " << m_yaw_cmd_deg[it] << " " << m_next_update_time[it] << " "
            << m_sum_u[it] << " " << m_sum_v[it] << " " << m_sum_t[it] << " " << m_yaw_angle_geom_deg[it] << "\n";
    }
}

bool
WindFarm::read_dynamic_yaw_state (const std::string& restart_chkfile)
{
    std::string fname = restart_chkfile + "/DynamicYawState";
    if (!amrex::FileExists(fname)) {
        return false;
    }

    Vector<char> fileCharPtr;
    ParallelDescriptor::ReadAndBcastFile(fname, fileCharPtr);
    std::string content(fileCharPtr.dataPtr());
    std::istringstream is(content, std::istringstream::in);

    std::string tag;
    is >> tag;
    if (tag != "DynamicYawState_v1") {
        amrex::Abort("Unknown DynamicYawState format");
    }

    int nturb_file = 0;
    is >> nturb_file;
    if (nturb_file != xloc.size()) {
        amrex::Abort("DynamicYawState: number of turbines does not match current configuration");
    }

    is >> m_disk_angle0_deg >> m_yaw_period >> m_tau_yaw >> m_yaw_alpha >> m_yaw_update_idx;
    m_dynamic_yaw_enabled = true;

    const int nturb = nturb_file;
    m_yaw_angle_deg.resize(nturb);
    m_yaw_cmd_deg.resize(nturb);
    m_next_update_time.resize(nturb);
    m_sum_u.resize(nturb);
    m_sum_v.resize(nturb);
    m_sum_t.resize(nturb);
    m_yaw_angle_geom_deg.resize(nturb);

    for (int it = 0; it < nturb; ++it) {
        is >> m_yaw_angle_deg[it] >> m_yaw_cmd_deg[it] >> m_next_update_time[it]
           >> m_sum_u[it] >> m_sum_v[it] >> m_sum_t[it] >> m_yaw_angle_geom_deg[it];
    }

    // Push restored per-turbine angles down into the actuator model.
    amrex::Vector<amrex::Real> disk_face_angles_deg;
    get_disk_face_angles_deg(disk_face_angles_deg);
    amrex::Vector<amrex::Real> theta_rad(nturb, 0.0);
    for (int it = 0; it < nturb; ++it) {
        theta_rad[it] = disk_face_angles_deg[it] * M_PI/180.0 - 0.5*M_PI;
    }
    set_turb_disk_angles(theta_rad);
    return true;
}
