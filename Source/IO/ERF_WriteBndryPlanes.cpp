#include "AMReX_Gpu.H"
#include "AMReX_ParmParse.H"
#include "AMReX_PlotFileUtil.H"
#include "AMReX_MultiFabUtil.H"
#include "AMReX_Utility.H"
#include "AMReX_VisMF.H"
#include "ERF_WriteBndryPlanes.H"
#include "ERF_IndexDefines.H"
#include "ERF_Derive.H"
#include <algorithm>
#include <fstream>
#include <iomanip>

using namespace amrex;

namespace {

std::string lateral_face_name (Orientation ori)
{
    if (ori == Orientation(Direction::x, Orientation::low )) return "xlo";
    if (ori == Orientation(Direction::x, Orientation::high)) return "xhi";
    if (ori == Orientation(Direction::y, Orientation::low )) return "ylo";
    if (ori == Orientation(Direction::y, Orientation::high)) return "yhi";
    return "invalid";
}

int lateral_face_index (const std::string& name)
{
    if (name == "xlo") return int(Orientation(Direction::x, Orientation::low));
    if (name == "xhi") return int(Orientation(Direction::x, Orientation::high));
    if (name == "ylo") return int(Orientation(Direction::y, Orientation::low));
    if (name == "yhi") return int(Orientation(Direction::y, Orientation::high));
    return -1;
}

MultiFab make_normalized_z_face (const MultiFab& z_phys_nd,
                                 const Box& cell_box,
                                 Orientation ori)
{
    Box source_box = surroundingNodes(cell_box);
    const int normal = ori.coordDir();
    const int face_index = ori.isLow() ? source_box.smallEnd(normal)
                                       : source_box.bigEnd(normal);
    source_box.setRange(normal, face_index, 1);

    BoxArray source_ba(source_box);
    DistributionMapping source_dm(source_ba);
    MultiFab source(source_ba, source_dm, 1, 0);
    source.ParallelCopy(z_phys_nd, 0, 0, 1);

    Box normalized_box(source_box);
    normalized_box.shift(-normalized_box.smallEnd());
    BoxArray normalized_ba(normalized_box);
    DistributionMapping normalized_dm(normalized_ba);
    MultiFab normalized(normalized_ba, normalized_dm, 1, 0);

    for (MFIter mfi(normalized); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        const auto dst = normalized.array(mfi);
        const auto src = source.const_array(mfi);
        const IntVect offset = source_box.smallEnd();
        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            dst(i,j,k) = src(i+offset[0],j+offset[1],k+offset[2]);
        });
    }
    return normalized;
}

}

/**
 * Copies the contents of one boundary register to another in either x or y dimensions.
 *
 * @param oit Orientation iterator used for determining which dimension we are to copy
 * @param b1 Boundary register containing the source data
 * @param b2 Boundary register containing the destination data
 */
void br_shift (OrientationIter oit, const BndryRegister& b1, BndryRegister& b2)
{
    auto ori = oit();
    int ncomp = b1[ori].nComp();
    if (ori.coordDir() < 2) {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (FabSetIter bfsi(b1[ori]); bfsi.isValid(); ++bfsi) {
            int idx = bfsi.index();
            const Box& bx1 = b1[ori].boxArray()[idx];
            const Box& bx2 = b2[ori].boxArray()[idx];

            // Copy onto a boundary register based on a box starting at (0,0,0)
            Array4<Real>       dest_arr = b2[ori][idx].array();
            Array4<Real const>  src_arr = b1[ori][idx].const_array();
            int ioff = bx1.smallEnd(0) - bx2.smallEnd(0);
            int joff = bx1.smallEnd(1) - bx2.smallEnd(1);
            ParallelFor(bx2, ncomp,
            [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept
            {
                dest_arr(i,j,k,n) = src_arr(i+ioff,j+joff,k,n);
            });
        }
    }
}

// Default to level 0
int WriteBndryPlanes::bndry_lev = 0;

/**
 * Constructor for the WriteBndryPlanes class for writing the contents of boundary planes to an output file
 *
 * @param grids Vector of BoxArrays containing the grids at each level in the AMR
 * @param geom Vector of Geometry containing the geometry at each level in the AMR
 */
WriteBndryPlanes::WriteBndryPlanes (Vector<BoxArray>& grids,
                                    Vector<Geometry>& geom,
                                    const Vector<std::unique_ptr<MultiFab>>& z_phys_nd)
    : m_geom(geom)
{
    ParmParse pp("erf");
    bndry_lev = 0;

    // Get the radius inside the domain
    pp.query("in_rad",m_in_rad);

    // User-specified region is given in physical coordinates, not index space
    std::vector<Real> box_lo(3), box_hi(3);
    pp.getarr("bndry_output_box_lo",box_lo,0,2);
    pp.getarr("bndry_output_box_hi",box_hi,0,2);

    // If the target area is contained at a finer level, use the finest data possible
    for (int ilev = 0; ilev < grids.size(); ilev++) {

        const Real* xLo   = m_geom[ilev].ProbLo();
        auto const dxi    = geom[ilev].InvCellSizeArray();
        const Box& domain = m_geom[ilev].Domain();

        // We create the smallest box that contains all of the cell centers
        // in the physical region specified
        int ilo = static_cast<int>(Math::floor((box_lo[0] - xLo[0]) * dxi[0])+.5);
        int jlo = static_cast<int>(Math::floor((box_lo[1] - xLo[1]) * dxi[1])+.5);
        int ihi = static_cast<int>(Math::floor((box_hi[0] - xLo[0]) * dxi[0])+.5)-1;
        int jhi = static_cast<int>(Math::floor((box_hi[1] - xLo[1]) * dxi[1])+.5)-1;

        // Map this to index space -- for now we do no interpolation
        Box candidate_box(IntVect(ilo,jlo,0), IntVect(ihi,jhi,domain.bigEnd(2)));
        if (ilev == 0) target_box = candidate_box;

        // Test if the target box at this level fits in the grids at this level
        Box gbx = candidate_box; gbx.grow(IntVect(1,1,0));

        // Ensure that the box is no larger than can fit in the (periodically grown) domain
        // at level 0
        if (ilev == 0) {
            Box per_grown_domain(domain);
            int growx = (geom[0].isPeriodic(0)) ? 1 : 0;
            int growy = (geom[0].isPeriodic(1)) ? 1 : 0;
            per_grown_domain.grow(IntVect(growx,growy,0));
            /*
            if (!per_grown_domain.contains(gbx))
                Error("WriteBndryPlanes: Requested box is too large to fill");
            */
        }

        if (grids[ilev].contains(gbx)) {
            bndry_lev = ilev;
            target_box = candidate_box;
        }
    }

    // The folder "m_filename" will contain the time series of data and the time.dat file
    pp.get("bndry_output_planes_file", m_filename);

    m_time_file = m_filename + "/time.dat";

    if (pp.contains("bndry_output_var_names"))
    {
        int num_vars = pp.countval("bndry_output_var_names");
        m_var_names.resize(num_vars);
        pp.queryarr("bndry_output_var_names",m_var_names,0,num_vars);
    }

    if (std::find(m_var_names.begin(), m_var_names.end(), "density") == m_var_names.end()) {
        Error("erf.bndry_output_var_names must include density");
    }
    const Vector<std::string> valid_vars{
        "density", "temperature", "theta", "scalar", "ke", "qv", "qc", "velocity"};
    for (int i = 0; i < m_var_names.size(); ++i) {
        if (std::find(valid_vars.begin(), valid_vars.end(), m_var_names[i]) == valid_vars.end()) {
            Error("Invalid erf.bndry_output_var_names entry '" + m_var_names[i] + "'");
        }
        if (std::find(m_var_names.begin(), m_var_names.begin()+i, m_var_names[i]) !=
            m_var_names.begin()+i) {
            Error("Duplicate erf.bndry_output_var_names entry '" + m_var_names[i] + "'");
        }
    }

    m_output_faces.fill(false);
    if (pp.contains("bndry_output_faces")) {
        const int nfaces = pp.countval("bndry_output_faces");
        if (nfaces == 0) Error("erf.bndry_output_faces must contain at least one face");
        Vector<std::string> face_names(nfaces);
        pp.getarr("bndry_output_faces", face_names, 0, nfaces);
        for (const auto& face_name : face_names) {
            const int iface = lateral_face_index(amrex::toLower(face_name));
            if (iface < 0) {
                Error("Invalid erf.bndry_output_faces entry '" + face_name +
                      "'; valid values are xlo xhi ylo yhi");
            }
            if (m_output_faces[iface]) {
                Error("Duplicate erf.bndry_output_faces entry '" + face_name + "'");
            }
            m_output_faces[iface] = true;
        }
    } else {
        for (OrientationIter oit; oit != nullptr; ++oit) {
            if (oit().coordDir() < 2) m_output_faces[int(oit())] = true;
        }
    }

    write_header_and_coordinates(*z_phys_nd[bndry_lev]);
}

void
WriteBndryPlanes::write_header_and_coordinates (const MultiFab& z_phys_nd)
{
    if (ParallelDescriptor::IOProcessor()) {
        if (!UtilCreateDirectory(m_filename, 0755)) {
            CreateDirectoryFailed(m_filename);
        }
    }
    ParallelDescriptor::Barrier();

    for (OrientationIter oit; oit != nullptr; ++oit) {
        const Orientation ori = oit();
        if (ori.coordDir() >= 2 || !m_output_faces[int(ori)]) continue;
        MultiFab z_face = make_normalized_z_face(z_phys_nd, target_box, ori);
        VisMF::Write(z_face, m_filename + "/z_phys_nd_" + lateral_face_name(ori));
    }

    if (ParallelDescriptor::IOProcessor()) {
        const std::string header_name = m_filename + "/Header";
        std::ofstream header(header_name, std::ofstream::out | std::ofstream::trunc);
        if (!header.good()) FileOpenFailed(header_name);
        header << std::setprecision(17);
        header << "ERF_BOUNDARY_PLANES 2\n";
        header << "level " << bndry_lev << "\n";
        header << "cells " << target_box.length(0) << ' '
               << target_box.length(1) << ' ' << target_box.length(2) << "\n";
        const auto dx = m_geom[bndry_lev].CellSizeArray();
        header << "cell_size " << dx[0] << ' ' << dx[1] << "\n";
        int nfaces = 0;
        for (bool enabled : m_output_faces) nfaces += enabled ? 1 : 0;
        header << "faces " << nfaces;
        for (OrientationIter oit; oit != nullptr; ++oit) {
            const Orientation ori = oit();
            if (ori.coordDir() < 2 && m_output_faces[int(ori)]) {
                header << ' ' << lateral_face_name(ori);
            }
        }
        header << '\n';
        for (OrientationIter oit; oit != nullptr; ++oit) {
            const Orientation ori = oit();
            if (ori.coordDir() >= 2 || !m_output_faces[int(ori)]) continue;
            const int transverse = ori.coordDir() == 0 ? 1 : 0;
            header << "face " << lateral_face_name(ori) << ' '
                   << target_box.length(transverse) << ' '
                   << target_box.length(2) << ' '
                   << dx[transverse] << ' '
                   << "z_phys_nd_" << lateral_face_name(ori) << '\n';
        }
        header << "variables " << m_var_names.size();
        for (const auto& var : m_var_names) header << ' ' << var;
        header << '\n';
    }
    ParallelDescriptor::Barrier();
}

/**
 * Function to write the specified grid data to an output file
 *
 * @param t_step Timestep number
 * @param time Current time
 * @param vars_new Grid data for all variables across the AMR hierarchy
 */
void WriteBndryPlanes::write_planes (const int t_step, const Real time,
                                     Vector<Vector<MultiFab>>& vars_new,
                                     bool is_moist)
{
    BL_PROFILE("ERF::WriteBndryPlanes::write_planes");

    MultiFab& S    = vars_new[bndry_lev][Vars::cons];
    MultiFab& xvel = vars_new[bndry_lev][Vars::xvel];
    MultiFab& yvel = vars_new[bndry_lev][Vars::yvel];
    MultiFab& zvel = vars_new[bndry_lev][Vars::zvel];

    const std::string chkname =
        m_filename + Concatenate("/bndry_output", t_step);

    //Print() << "Writing boundary planes at time " << time << std::endl;

    const std::string level_prefix = "Level_";
    PreBuildDirectorHierarchy(chkname, level_prefix, 1, true);

    // note: by using the entire domain box we end up using 1 processor
    // to hold all boundaries
    BoxArray ba(target_box);
    DistributionMapping dm{ba};

    IntVect new_hi = target_box.bigEnd() - target_box.smallEnd();
    Box target_box_shifted(IntVect(0,0,0),new_hi);
    BoxArray ba_shifted(target_box_shifted);

    for (int i = 0; i < m_var_names.size(); i++)
    {
        std::string var_name = m_var_names[i];
        std::string filename = MultiFabFileFullPrefix(bndry_lev, chkname, level_prefix, var_name);

        int ncomp;
        if (var_name == "velocity") {
            ncomp = AMREX_SPACEDIM;
        } else {
            ncomp = 1;
        }

        BndryRegister bndry        (ba        , dm, m_in_rad, m_out_rad, m_extent_rad, ncomp);
        BndryRegister bndry_shifted(ba_shifted, dm, m_in_rad, m_out_rad, m_extent_rad, ncomp);

        int nghost = 0;
        if (var_name == "density")
        {
            bndry.copyFrom(S, nghost, Rho_comp, 0, ncomp, m_geom[bndry_lev].periodicity());

        } else if (var_name == "temperature") {
            MultiFab Temp(S.boxArray(),S.DistributionMap(),ncomp,0);
            for (MFIter mfi(Temp, TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                const Box& bx = mfi.tilebox();
                if (is_moist) {
                    derived::erf_dermoisttemp(bx, Temp[mfi], 0, 1, S[mfi], m_geom[bndry_lev], time, nullptr, bndry_lev);
                } else {
                    derived::erf_dertemp(bx, Temp[mfi], 0, 1, S[mfi], m_geom[bndry_lev], time, nullptr, bndry_lev);
                }
            }
            bndry.copyFrom(Temp, nghost, 0, 0, ncomp, m_geom[bndry_lev].periodicity());
        } else if (var_name == "theta") {
            MultiFab Temp(S.boxArray(),S.DistributionMap(),ncomp,0);
            for (MFIter mfi(Temp, TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                const Box& bx = mfi.tilebox();
                derived::erf_derrhodivide(bx, Temp[mfi], S[mfi], RhoTheta_comp);
            }
            bndry.copyFrom(Temp, nghost, 0, 0, ncomp, m_geom[bndry_lev].periodicity());
        } else if (var_name == "scalar") {
            MultiFab Temp(S.boxArray(),S.DistributionMap(),ncomp,0);
            for (MFIter mfi(Temp, TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                const Box& bx = mfi.tilebox();
                derived::erf_derrhodivide(bx, Temp[mfi], S[mfi], RhoScalar_comp);
            }
            bndry.copyFrom(Temp, nghost, 0, 0, ncomp, m_geom[bndry_lev].periodicity());
        } else if (var_name == "ke") {
            MultiFab Temp(S.boxArray(),S.DistributionMap(),ncomp,0);
            for (MFIter mfi(Temp, TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                const Box& bx = mfi.tilebox();
                derived::erf_derrhodivide(bx, Temp[mfi], S[mfi], RhoKE_comp);
            }
            bndry.copyFrom(Temp, nghost, 0, 0, ncomp, m_geom[bndry_lev].periodicity());
        } else if (var_name == "qv") {
            MultiFab Temp(S.boxArray(),S.DistributionMap(),ncomp,0);
            if (S.nComp() > RhoQ2_comp) {
                for (MFIter mfi(Temp, TilingIfNotGPU()); mfi.isValid(); ++mfi)
                {
                    const Box& bx = mfi.tilebox();
                    derived::erf_derrhodivide(bx, Temp[mfi], S[mfi], RhoQ1_comp);
                }
            } else {
                Temp.setVal(0.);
            }
            bndry.copyFrom(Temp, nghost, 0, 0, ncomp, m_geom[bndry_lev].periodicity());
        } else if (var_name == "qc") {
            MultiFab Temp(S.boxArray(),S.DistributionMap(),ncomp,0);
            if (S.nComp() > RhoQ2_comp) {
                for (MFIter mfi(Temp, TilingIfNotGPU()); mfi.isValid(); ++mfi)
                {
                    const Box& bx = mfi.tilebox();
                    derived::erf_derrhodivide(bx, Temp[mfi], S[mfi], RhoQ2_comp);
                }
            } else {
                Temp.setVal(0.);
            }
            bndry.copyFrom(Temp, nghost, 0, 0, ncomp, m_geom[bndry_lev].periodicity());
        } else if (var_name == "velocity") {
            MultiFab Vel(S.boxArray(), S.DistributionMap(), 3, m_out_rad);
            average_face_to_cellcenter(Vel,0,Array<const MultiFab*,3>{&xvel,&yvel,&zvel});
            bndry.copyFrom(Vel, nghost, 0, 0, ncomp, m_geom[bndry_lev].periodicity());
        } else {
            //Print() << "Trying to write planar output for " << var_name << std::endl;
            Error("Don't know how to output this variable");
        }

        for (OrientationIter oit; oit != nullptr; ++oit) {
            auto ori = oit();
            if (ori.coordDir() < 2 && m_output_faces[int(ori)]) {
                std::string facename = Concatenate(filename + '_', ori, 1);
                br_shift(oit, bndry, bndry_shifted);
                bndry_shifted[ori].write(facename);
            }
        }

    } // loop over num_vars

    // Writing time.dat
    if (ParallelDescriptor::IOProcessor()) {
        std::ofstream oftime(m_time_file, std::ios::out | std::ios::app);
        oftime << std::setprecision(17) << t_step << ' ' << time << '\n';
        oftime.close();
    }
}
