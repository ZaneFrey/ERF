#include "AMReX_Gpu.H"
#include "AMReX_ParmParse.H"
#include <AMReX_PlotFileUtil.H>
#include "ERF_ReadBndryPlanes.H"
#include "ERF_IndexDefines.H"
#include "AMReX_MultiFabUtil.H"
#include "AMReX_Utility.H"
#include "AMReX_VisMF.H"
#include "ERF_EOS.H"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>

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
    const IntVect offset = source_box.smallEnd();
    normalized_box.shift(-offset);
    BoxArray normalized_ba(normalized_box);
    DistributionMapping normalized_dm(normalized_ba);
    MultiFab normalized(normalized_ba, normalized_dm, 1, 0);

    for (MFIter mfi(normalized); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        const auto dst = normalized.array(mfi);
        const auto src = source.const_array(mfi);
        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            dst(i,j,k) = src(i+offset[0],j+offset[1],k+offset[2]);
        });
    }
    return normalized;
}

void read_and_align_face (MultiFab& mf, const std::string& filename,
                          const Box& target_box, int expected_ncomp,
                          const std::string& face_name)
{
    if (!FileExists(filename + "_H")) {
        Error("Missing boundary-plane file for active face " + face_name + ": " + filename);
    }
    VisMF::Read(mf, filename);
    if (mf.nComp() != expected_ncomp) {
        Error("Boundary-plane component mismatch for face " + face_name + " in " + filename);
    }
    if (mf.boxArray().size() != 1) {
        Error("Boundary-plane face file must contain exactly one box: " + filename);
    }
    const Box source_box = mf.boxArray().minimalBox();
    if (source_box.ixType() != target_box.ixType()) {
        Error("Boundary-plane index-type mismatch for active face " + face_name +
              " in " + filename);
    }
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        if (source_box.length(dir) != target_box.length(dir)) {
            Error("Boundary-plane shape mismatch for active face " + face_name +
                  " in " + filename);
        }
    }
    mf.shift(target_box.smallEnd() - source_box.smallEnd());
}

}

/**
 * Return closest index (from lower) of value in vector
 */
AMREX_FORCE_INLINE
int closest_index (const Vector<Real>& vec, const Real value)
{
    auto const it = std::upper_bound(vec.begin(), vec.end(), value);
    AMREX_ALWAYS_ASSERT(it != vec.end());

    const int idx = std::distance(vec.begin(), it);
    return std::max(idx - 1, 0);
}

/**
 * Return offset vector
 */
AMREX_FORCE_INLINE
IntVect offset (const int face_dir, const int normal)
{
    IntVect offset(IntVect::TheDimensionVector(normal));
    if (face_dir == 1) {
        for (auto& o : offset) {
            o *= -1;
        }
    }
    return offset;
}

/**
 * Function in ReadBndryPlanes class for allocating space
 * for the boundary plane data ERF will need.
 */
void ReadBndryPlanes::define_level_data (int /*lev*/)
{
    Print() << "ReadBndryPlanes::define_level_data" << std::endl;
    // *********************************************************
    // Allocate space for all of the boundary planes we may need
    // *********************************************************
    int ncomp = BCVars::NumTypes;
    const Box& domain = m_geom.Domain();
    for (OrientationIter oit; oit != nullptr; ++oit) {
        auto ori = oit();
        if (ori.coordDir() < 2 && m_active_faces[int(ori)]) {

            m_data_n[ori]      = std::make_unique<PlaneVector>();
            m_data_np1[ori]    = std::make_unique<PlaneVector>();
            m_data_np2[ori]    = std::make_unique<PlaneVector>();
            m_data_interp[ori] = std::make_unique<PlaneVector>();

            const auto& lo = domain.loVect();
            const auto& hi = domain.hiVect();

            IntVect plo(lo);
            IntVect phi(hi);
            const int normal = ori.coordDir();
            plo[normal] = ori.isHigh() ? hi[normal] - (m_in_rad - 1) : -m_out_rad;
            phi[normal] = ori.isHigh() ? hi[normal] + (m_out_rad   ) : (m_in_rad - 1);
            const Box pbx(plo, phi);
            m_data_n[ori]->push_back(FArrayBox(pbx, ncomp));
            m_data_np1[ori]->push_back(FArrayBox(pbx, ncomp));
            m_data_np2[ori]->push_back(FArrayBox(pbx, ncomp));
            m_data_interp[ori]->push_back(FArrayBox(pbx, ncomp));
        }
    }
}

/**
 * Function in ReadBndryPlanes class for interpolating boundary
 * data in time.
 *
 * @param time Constant specifying the time for interpolation
 */
Vector<std::unique_ptr<PlaneVector>>&
ReadBndryPlanes::interp_in_time (const Real& time)
{
    AMREX_ALWAYS_ASSERT(m_tn <= time && time <= m_tnp2);

    //Print() << "interp_in_time at time " << time << " given " << m_tn << " " << m_tnp1 << " " << m_tnp2 << std::endl;
    //Print() << "m_tinterp " << m_tinterp << std::endl;

    if (time == m_tinterp) {
        // We have already interpolated to this time
        return m_data_interp;

    } else {

        // We must now interpolate to a new time
        m_tinterp = time;

        if (time < m_tnp1) {
            for (OrientationIter oit; oit != nullptr; ++oit) {
                auto ori = oit();
                if (ori.coordDir() < 2 && m_active_faces[int(ori)]) {
                    const int nlevels = m_data_n[ori]->size();
                    for (int lev = 0; lev < nlevels; ++lev) {
                        const auto& datn   = (*m_data_n[ori])[lev];
                        const auto& datnp1 = (*m_data_np1[ori])[lev];
                        auto& dati = (*m_data_interp[ori])[lev];
                        dati.linInterp<RunOn::Device>(
                            datn, 0, datnp1, 0, m_tn, m_tnp1, m_tinterp, datn.box(), 0, dati.nComp());
                    }
                }
            }
        } else {
            for (OrientationIter oit; oit != nullptr; ++oit) {
                auto ori = oit();
                if (ori.coordDir() < 2 && m_active_faces[int(ori)]) {
                    const int nlevels = m_data_n[ori]->size();
                    for (int lev = 0; lev < nlevels; ++lev) {
                        const auto& datnp1 = (*m_data_np1[ori])[lev];
                        const auto& datnp2 = (*m_data_np2[ori])[lev];
                        auto& dati = (*m_data_interp[ori])[lev];
                        dati.linInterp<RunOn::Device>(
                            datnp1, 0, datnp2, 0, m_tnp1, m_tnp2, m_tinterp, datnp1.box(), 0,
                            dati.nComp());
                    }
                }
            }
        }
    }
    return m_data_interp;
}

const FArrayBox*
ReadBndryPlanes::interpolated_face (Orientation ori, int lev) const
{
    if (!face_is_active(ori) || !m_data_interp[int(ori)]) return nullptr;
    return &(*m_data_interp[int(ori)])[lev];
}

/**
 * ReadBndryPlanes class constructor. Handles initialization from inputs file parameters.
 *
 * @param geom Geometry for the domain
 * @param rdOcp_in Real constant for the Rhydberg constant ($R_d$) divided by the specific heat at constant pressure ($c_p$)
 */
ReadBndryPlanes::ReadBndryPlanes (const Geometry& geom, const Real& rdOcp_in)
:
    m_geom(geom),
    m_rdOcp(rdOcp_in)
{
    ParmParse pp("erf");

    // Get the radius inside the domain
    pp.query("in_rad",m_in_rad);

    // Are we using real bcs?
    pp.query("use_real_bcs", m_use_real_bcs);

    last_file_read = -1;

    m_tinterp = -1.;

    // What folder will the time series of planes be read from
    pp.get("bndry_file", m_filename);

    is_velocity_read     = 0;
    is_density_read      = 0;
    is_temperature_read  = 0;
    is_theta_read        = 0;
    is_scalar_read       = 0;
    is_q1_read           = 0;
    is_q2_read           = 0;
    is_KE_read           = 0;

    if (pp.contains("bndry_input_var_names"))
    {
        int num_vars = pp.countval("bndry_input_var_names");
        m_var_names.resize(num_vars);
        pp.queryarr("bndry_input_var_names",m_var_names,0,num_vars);
        const Vector<std::string> valid_vars{
            "density", "temperature", "theta", "scalar", "ke", "qv", "qc", "velocity"};
        for (int i = 0; i < m_var_names.size(); i++) {
            if (std::find(valid_vars.begin(), valid_vars.end(), m_var_names[i]) == valid_vars.end()) {
                Error("Invalid erf.bndry_input_var_names entry '" + m_var_names[i] + "'");
            }
            if (std::find(m_var_names.begin(), m_var_names.begin()+i, m_var_names[i]) !=
                m_var_names.begin()+i) {
                Error("Duplicate erf.bndry_input_var_names entry '" + m_var_names[i] + "'");
            }
            if (m_var_names[i] == "velocity")     is_velocity_read = 1;
            if (m_var_names[i] == "density")      is_density_read = 1;
            if (m_var_names[i] == "temperature")  is_temperature_read = 1;
            if (m_var_names[i] == "theta")        is_theta_read = 1;
            if (m_var_names[i] == "scalar")       is_scalar_read = 1;
            if (m_var_names[i] == "qv")           is_q1_read = 1;
            if (m_var_names[i] == "qc")           is_q2_read = 1;
            if (m_var_names[i] == "ke")           is_KE_read = 1;
        }
    }

    // time.dat will be in the same folder as the time series of data
    m_time_file = m_filename + "/time.dat";

    // each pointer (at at given time) has 6 components, one for each orientation
    // TODO: we really only need 4 not 6
    int size = 2*AMREX_SPACEDIM;
    m_data_n.resize(size);
    m_data_np1.resize(size);
    m_data_np2.resize(size);
    m_data_interp.resize(size);
}

void
ReadBndryPlanes::initialize_active_faces (const Vector<BCRec>& bcs,
                                          const MultiFab& z_phys_nd)
{
    m_active_faces.fill(false);
    int num_active = 0;
    for (OrientationIter oit; oit != nullptr; ++oit) {
        const Orientation ori = oit();
        if (ori.coordDir() >= 2) continue;
        const int dir = ori.coordDir();
        for (const auto& bc : bcs) {
            const int bc_type = ori.isLow() ? bc.lo(dir) : bc.hi(dir);
            if (bc_type == ERFBCType::ext_dir_ingested) {
                m_active_faces[int(ori)] = true;
                break;
            }
        }
        if (m_active_faces[int(ori)]) ++num_active;
    }
    if (num_active == 0) {
        Error("erf.input_bndry_planes is enabled, but no lateral boundary uses ingested data");
    }

    read_header_and_validate(z_phys_nd);
    define_level_data(0);
}

void
ReadBndryPlanes::read_header_and_validate (const MultiFab& z_phys_nd)
{
    const std::string header_name = m_filename + "/Header";
    if (!FileExists(header_name)) {
        static bool warned_legacy = false;
        if (!warned_legacy && ParallelDescriptor::IOProcessor()) {
            Warning("Boundary-plane directory has no versioned Header; only active-face array shapes "
                    "can be checked. Physical spacing and vertical coordinates are unverified.");
        }
        warned_legacy = true;
        m_has_header = false;
        return;
    }

    Vector<char> header_chars;
    ParallelDescriptor::ReadAndBcastFile(header_name, header_chars);
    std::istringstream header(std::string(header_chars.dataPtr()));

    std::string magic;
    int version = 0;
    header >> magic >> version;
    if (magic != "ERF_BOUNDARY_PLANES" || version != 2) {
        Error("Unsupported boundary-plane Header in " + header_name);
    }

    std::string label;
    int file_level = -1;
    int nx = 0, ny = 0, nz = 0;
    Real dx = 0.0, dy = 0.0;
    int nfaces = 0, nvars = 0;
    header >> label >> file_level;
    if (label != "level" || file_level != 0) {
        Error("Boundary-plane input currently supports only level 0");
    }
    header >> label >> nx >> ny >> nz;
    if (label != "cells") Error("Malformed cells record in " + header_name);
    header >> label >> dx >> dy;
    if (label != "cell_size") Error("Malformed cell_size record in " + header_name);
    header >> label >> nfaces;
    if (label != "faces" || nfaces < 1 || nfaces > 4) {
        Error("Malformed faces record in " + header_name);
    }
    m_file_faces.fill(false);
    for (int n = 0; n < nfaces; ++n) {
        std::string name;
        header >> name;
        const int iface = lateral_face_index(name);
        if (iface < 0 || m_file_faces[iface]) Error("Invalid or duplicate face in " + header_name);
        m_file_faces[iface] = true;
    }
    std::array<int,2*AMREX_SPACEDIM> face_ntrans{};
    std::array<int,2*AMREX_SPACEDIM> face_nz{};
    std::array<Real,2*AMREX_SPACEDIM> face_spacing{};
    std::array<std::string,2*AMREX_SPACEDIM> face_z_file{};
    for (int n = 0; n < nfaces; ++n) {
        std::string face_name;
        header >> label >> face_name;
        const int iface = lateral_face_index(face_name);
        if (label != "face" || iface < 0 || !m_file_faces[iface] ||
            !face_z_file[iface].empty()) {
            Error("Malformed or duplicate face metadata in " + header_name);
        }
        header >> face_ntrans[iface] >> face_nz[iface]
               >> face_spacing[iface] >> face_z_file[iface];
        if (!header || face_ntrans[iface] <= 0 || face_nz[iface] <= 0 ||
            face_spacing[iface] <= 0.0 || face_z_file[iface].empty()) {
            Error("Invalid face metadata for " + face_name + " in " + header_name);
        }
    }
    header >> label >> nvars;
    if (label != "variables" || nvars < 1) Error("Malformed variables record in " + header_name);
    m_file_var_names.resize(nvars);
    for (auto& name : m_file_var_names) header >> name;
    if (!header) Error("Malformed boundary-plane Header " + header_name);

    const Vector<std::string> valid_vars{
        "density", "temperature", "theta", "scalar", "ke", "qv", "qc", "velocity"};
    for (int i = 0; i < m_file_var_names.size(); ++i) {
        const auto& name = m_file_var_names[i];
        if (std::find(valid_vars.begin(), valid_vars.end(), name) == valid_vars.end() ||
            std::find(m_file_var_names.begin(), m_file_var_names.begin()+i, name) !=
            m_file_var_names.begin()+i) {
            Error("Invalid or duplicate variable '" + name + "' in " + header_name);
        }
    }

    if (std::find(m_file_var_names.begin(), m_file_var_names.end(), "density") == m_file_var_names.end()) {
        Error("Boundary-plane Header does not advertise required density data");
    }
    for (const auto& var : m_var_names) {
        if (std::find(m_file_var_names.begin(), m_file_var_names.end(), var) == m_file_var_names.end()) {
            Error("Requested boundary variable '" + var + "' is not present in " + header_name);
        }
    }

    const Box& domain = m_geom.Domain();
    const auto target_dx = m_geom.CellSizeArray();
    for (OrientationIter oit; oit != nullptr; ++oit) {
        const Orientation ori = oit();
        if (ori.coordDir() >= 2 || !m_active_faces[int(ori)]) continue;
        const std::string face = lateral_face_name(ori);
        if (!m_file_faces[int(ori)]) {
            Error("Boundary-plane Header does not contain active face " + face);
        }
        if (face_z_file[int(ori)] != "z_phys_nd_" + face) {
            Error("Boundary-plane coordinate-file reference disagrees with active face " + face);
        }
        if (face_nz[int(ori)] != nz || face_nz[int(ori)] != domain.length(2)) {
            Error("Boundary-plane nz mismatch on active face " + face);
        }
        if (ori.coordDir() == 0) {
            const Real spacing_tol = 128.0 * std::numeric_limits<Real>::epsilon() *
                std::max(Real(1.0), std::max(std::abs(dy), std::abs(target_dx[1])));
            if (face_ntrans[int(ori)] != ny || ny != domain.length(1) ||
                std::abs(face_spacing[int(ori)]-dy) > spacing_tol ||
                std::abs(dy-target_dx[1]) > spacing_tol) {
                Error("Boundary-plane y discretization mismatch on active face " + face);
            }
        } else {
            const Real spacing_tol = 128.0 * std::numeric_limits<Real>::epsilon() *
                std::max(Real(1.0), std::max(std::abs(dx), std::abs(target_dx[0])));
            if (face_ntrans[int(ori)] != nx || nx != domain.length(0) ||
                std::abs(face_spacing[int(ori)]-dx) > spacing_tol ||
                std::abs(dx-target_dx[0]) > spacing_tol) {
                Error("Boundary-plane x discretization mismatch on active face " + face);
            }
        }

        MultiFab source_z;
        const std::string z_name = m_filename + "/" + face_z_file[int(ori)];
        if (!FileExists(z_name + "_H")) {
            Error("Missing vertical-coordinate file for active face " + face + ": " + z_name);
        }
        VisMF::Read(source_z, z_name);
        MultiFab target_z = make_normalized_z_face(z_phys_nd, domain, ori);
        if (source_z.nComp() != 1 || !amrex::match(source_z.boxArray(), target_z.boxArray())) {
            Error("Vertical-coordinate shape mismatch on active face " + face);
        }
        MultiFab diff(target_z.boxArray(), target_z.DistributionMap(), 1, 0);
        MultiFab::Copy(diff, target_z, 0, 0, 1, 0);
        MultiFab::Subtract(diff, source_z, 0, 0, 1, 0);
        diff.abs(0, 1, 0);
        const Real max_error = diff.norm0();
        const Real vertical_extent = std::abs(m_geom.ProbHi(2)-m_geom.ProbLo(2));
        const Real z_tol = 128.0 * std::numeric_limits<Real>::epsilon() *
                           std::max(Real(1.0), vertical_extent);
        if (max_error > z_tol) {
            const IntVect max_index = diff.maxIndex(0, 0);
            const Box point_box(max_index, max_index, target_z.boxArray().ixType());
            const Real source_value = source_z.sum(point_box, 0);
            const Real target_value = target_z.sum(point_box, 0);
            std::ostringstream message;
            message << "Vertical-coordinate mismatch on active face " << face
                    << " at index " << max_index
                    << "; source=" << source_value
                    << ", target=" << target_value
                    << ", absolute difference=" << max_error;
            Error(message.str());
        }
    }
    m_has_header = true;
}

/**
 * Function in ReadBndryPlanes class for reading the external file
 * specifying time data and broadcasting this data across MPI ranks.
 */
void ReadBndryPlanes::read_time_file ()
{
    BL_PROFILE("ERF::ReadBndryPlanes::read_time_file");

    // *********************************************************
    // Read the time.data file and store the timesteps and times
    // *********************************************************
    int time_file_length = 0;

    if (ParallelDescriptor::IOProcessor()) {

        std::string line;
        std::ifstream time_file(m_time_file);
        if (!time_file.good()) {
            Abort("Cannot find time file: " + m_time_file);
        }
        while (std::getline(time_file, line)) {
            ++time_file_length;
        }

        time_file.close();
    }

    ParallelDescriptor::Bcast(
        &time_file_length, 1,
        ParallelDescriptor::IOProcessorNumber(),
        ParallelDescriptor::Communicator());

    m_in_times.resize(time_file_length);
    m_in_timesteps.resize(time_file_length);

    if (ParallelDescriptor::IOProcessor()) {
        std::ifstream time_file(m_time_file);
        for (int i = 0; i < time_file_length; ++i) {
            time_file >> m_in_timesteps[i] >> m_in_times[i];
        }
        // Sanity check that there are no duplicates or mis-orderings
        for (int i = 1; i < time_file_length; ++i) {
            if (m_in_timesteps[i] <= m_in_timesteps[i-1])
                Error("Bad timestep in time.dat file");
            if (m_in_times[i] <= m_in_times[i-1])
                Error("Bad time in time.dat file");
        }
        time_file.close();
    }

    ParallelDescriptor::Bcast(
        m_in_timesteps.data(), time_file_length,
        ParallelDescriptor::IOProcessorNumber(),
        ParallelDescriptor::Communicator());

    ParallelDescriptor::Bcast(
        m_in_times.data(), time_file_length,
        ParallelDescriptor::IOProcessorNumber(),
        ParallelDescriptor::Communicator());

    if (time_file_length < 3) {
        Error("Boundary-plane input requires at least three time records");
    }
    Print() << "Successfully read boundary-plane time file" << std::endl;
}

/**
 * Function in ReadBndryPlanes for reading boundary data
 * at a specific time and at the next timestep from input files.
 *
 * @param time Current time
 * @param dt Current timestep
 * @param m_bc_extdir_vals Container storing the external dirichlet boundary conditions we are reading from the input files
 */
void ReadBndryPlanes::read_input_files (Real time,
                                        Real dt,
                                        Array<Array<Real, AMREX_SPACEDIM*2>,AMREX_SPACEDIM+NBCVAR_max> m_bc_extdir_vals)
{
    BL_PROFILE("ERF::ReadBndryPlanes::read_input_files");

    // Assert that both the current time and the next time are within the bounds
    // of the data that we can read
    AMREX_ALWAYS_ASSERT((m_in_times[0] <= time) && (time <= m_in_times.back()));
    AMREX_ALWAYS_ASSERT((m_in_times[0] <= time+dt) && (time+dt <= m_in_times.back()));

    // The first time we enter this routine we read the first three files
    if (last_file_read == -1)
    {
        int idx_init = 0;
        read_file(idx_init,m_data_n,m_bc_extdir_vals);
        read_file(idx_init,m_data_interp,m_bc_extdir_vals); // We want to start with this filled
        m_tn = m_in_times[idx_init];

        idx_init = 1;
        read_file(idx_init,m_data_np1,m_bc_extdir_vals);
        m_tnp1 = m_in_times[idx_init];

        idx_init = 2;
        read_file(idx_init,m_data_np2,m_bc_extdir_vals);
        m_tnp2 = m_in_times[idx_init];

        last_file_read = idx_init;
    }

    // Compute the index such that time falls between times[idx] and times[idx+1]
    const int idx = closest_index(m_in_times, time);

    // Advance the read window until it spans the requested time.
    while (idx >= last_file_read-1 && last_file_read != m_in_times.size()-1) {
        int new_read = last_file_read+1;

        // We need to change which data the pointers point to before we read in the new data
        // This doesn't actually move the data, just swaps the pointers
        for (OrientationIter oit; oit != nullptr; ++oit) {
            auto ori = oit();
            if (ori.coordDir() < 2 && m_active_faces[int(ori)]) {
                std::swap(m_data_n[ori]  ,m_data_np1[ori]);
                std::swap(m_data_np1[ori],m_data_np2[ori]);
            }
        }

        // Set the times corresponding to the post-swap pointers
        m_tn   = m_tnp1;
        m_tnp1 = m_tnp2;
        m_tnp2 = m_in_times[new_read];

        read_file(new_read,m_data_np2,m_bc_extdir_vals);
        last_file_read = new_read;
    }

    AMREX_ASSERT(time    >= m_tn && time    <= m_tnp2);
    AMREX_ASSERT(time+dt >= m_tn && time+dt <= m_tnp2);
}

/**
 * Function in ReadBndryPlanes to read boundary data for each face and variable
 * from files.
 *
 * @param idx Specifies the index corresponding to the timestep we want
 * @param data_to_fill Container for face data on boundaries
 * @param m_bc_extdir_vals Container storing the external dirichlet boundary conditions we are reading from the input files
 */
void ReadBndryPlanes::read_file (const int idx,
                                 Vector<std::unique_ptr<PlaneVector>>& data_to_fill,
                                 Array<Array<Real, AMREX_SPACEDIM*2>,AMREX_SPACEDIM+NBCVAR_max> /*m_bc_extdir_vals*/)
{
    const int t_step = m_in_timesteps[idx];
    const std::string chkname1 = m_filename + Concatenate("/bndry_output", t_step);

    const std::string level_prefix = "Level_";
    const int lev = 0;

    // We need to initialize all the components because we may not fill all of them from files,
    //    but the loop in the interpolate routine goes over all the components anyway
    int ncomp_for_bc = BCVars::NumTypes;
    for (OrientationIter oit; oit != nullptr; ++oit) {
        auto ori = oit();
        if (ori.coordDir() < 2 && m_active_faces[int(ori)]) {
            FArrayBox& d = (*data_to_fill[ori])[lev];
            const auto& bx = d.box();
            Array4<Real> d_arr = d.array();
            ParallelFor(
                bx, ncomp_for_bc, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
                d_arr(i,j,k,n) = 0.;
            });
        }
    }

    // Read density for primitive to conserved conversions
    std::string filenamer = MultiFabFileFullPrefix(lev, chkname1, level_prefix, "density");
    std::array<std::unique_ptr<MultiFab>,2*AMREX_SPACEDIM> density_faces;
    for (OrientationIter oit; oit != nullptr; ++oit) {
        const Orientation ori = oit();
        if (ori.coordDir() < 2 && m_active_faces[int(ori)]) {
            const std::string facenamer = Concatenate(filenamer + '_', ori, 1);
            density_faces[int(ori)] = std::make_unique<MultiFab>();
            read_and_align_face(*density_faces[int(ori)], facenamer,
                                (*data_to_fill[ori])[lev].box(), 1,
                                lateral_face_name(ori));
        }
    }

    // Expose for GPU
    bool real_bcs = m_use_real_bcs;

    for (int ivar = 0; ivar < m_var_names.size(); ivar++)
    {
        std::string var_name = m_var_names[ivar];

        std::string filename1 = MultiFabFileFullPrefix(lev, chkname1, level_prefix, var_name);

        int ncomp;
        if (var_name == "velocity") {
            ncomp = AMREX_SPACEDIM;
        } else {
            ncomp = 1;
        }

        int n_offset;
        if (var_name == "density")     n_offset = BCVars::Rho_bc_comp;
        if (var_name == "theta")       n_offset = BCVars::RhoTheta_bc_comp;
        if (var_name == "temperature") n_offset = BCVars::RhoTheta_bc_comp;
        if (var_name == "ke")          n_offset = BCVars::RhoKE_bc_comp;
        if (var_name == "scalar")      n_offset = BCVars::RhoScalar_bc_comp;
        if (var_name == "qv")          n_offset = BCVars::RhoQ1_bc_comp;
        if (var_name == "qc")          n_offset = BCVars::RhoQ2_bc_comp;
        if (var_name == "velocity")    n_offset = BCVars::xvel_bc;

        // Print() << "Reading " << chkname1 << " for variable " << var_name << " with n_offset == " << n_offset << std::endl;

        // *********************************************************
        // Read in the BndryReg for all non-z faces
        // *********************************************************
        for (OrientationIter oit; oit != nullptr; ++oit) {
          auto ori = oit();
          if (ori.coordDir() < 2 && m_active_faces[int(ori)]) {

            std::string facename1 = Concatenate(filename1 + '_', ori, 1);
            MultiFab bndry_read;
            read_and_align_face(bndry_read, facename1,
                                (*data_to_fill[ori])[lev].box(), ncomp,
                                lateral_face_name(ori));

            int normal = ori.coordDir();
            IntVect v_offset = offset(ori.faceDir(), normal);
            if (real_bcs) { v_offset = IntVect(0); }

            const auto& bbx = (*data_to_fill[ori])[lev].box();
            const int read_anchor = ori.isLow() ? bbx.smallEnd(normal) : bbx.bigEnd(normal);

            // *********************************************************
            // Copy from the BndryReg into a MultiFab then use copyTo
            //     to write from the MultiFab to a single FAB for each face
            // *********************************************************
            MultiFab bndryMF(
                bndry_read.boxArray(), bndry_read.DistributionMap(),
                ncomp, 0, MFInfo());

            for (MFIter mfi(bndryMF); mfi.isValid(); ++mfi) {

                const auto& vbx = mfi.validbox();
                const auto& bndry_read_arr   = bndry_read.const_array(mfi);
                const auto& bndry_read_r_arr = density_faces[int(ori)]->const_array(mfi);
                const auto& bndry_mf_arr     = bndryMF.array(mfi);

                const auto& bx = bbx & vbx;
                if (bx.isEmpty()) {
                    continue;
                }

                // We average the two cell-centered data points in the normal direction
                //    to define a Dirichlet value on the face itself.

                // This is the scalars -- they all get multiplied by rho, and in the case of
                //   reading in temperature, we must convert to theta first
                Real rdOcp = m_rdOcp;
                {
                  if (var_name == "temperature") {
                    ParallelFor(
                        bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                             int ir = i;
                             int jr = j;
                             if (normal == 0) ir = read_anchor;
                             if (normal == 1) jr = read_anchor;
                             Real R1 =  bndry_read_r_arr(ir, jr, k, 0);
                             Real R2 =  bndry_read_r_arr(ir+v_offset[0],jr+v_offset[1],k+v_offset[2],0);
                             Real T1 =  bndry_read_arr(ir, jr, k, 0);
                             Real T2 =  bndry_read_arr(ir+v_offset[0],jr+v_offset[1],k+v_offset[2],0);
                             Real Th1 = getThgivenRandT(R1,T1,rdOcp);
                             Real Th2 = getThgivenRandT(R2,T2,rdOcp);
                             bndry_mf_arr(i, j, k, 0) = (real_bcs) ? bndry_read_arr(ir, jr, k, 0) :
                                                                     0.5 * (R1*Th1 + R2*Th2);
                        });
                  } else if (var_name == "theta" || var_name == "ke" || var_name == "scalar" ||
                             var_name == "qv"    || var_name == "qc") {
                    ParallelFor(
                        bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                             int ir = i;
                             int jr = j;
                             if (normal == 0) ir = read_anchor;
                             if (normal == 1) jr = read_anchor;
                             Real R1 =  bndry_read_r_arr(ir, jr, k, 0);
                             Real R2 =  bndry_read_r_arr(ir+v_offset[0],jr+v_offset[1],k+v_offset[2],0);
                             bndry_mf_arr(i, j, k, 0) = (real_bcs) ? bndry_read_arr(ir, jr, k, 0) :
                                 0.5 * ( R1 * bndry_read_arr(ir, jr, k, 0) +
                                         R2 * bndry_read_arr(ir+v_offset[0],jr+v_offset[1],k+v_offset[2], 0));
                        });
                   } else if (var_name == "density") {
                    ParallelFor(
                        bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                                int ir = i;
                                int jr = j;
                                if (normal == 0) ir = read_anchor;
                                if (normal == 1) jr = read_anchor;
                                bndry_mf_arr(i, j, k, 0) = (real_bcs) ? bndry_read_arr(ir, jr, k, 0) :
                                    0.5 * ( bndry_read_arr(ir, jr, k, 0) +
                                            bndry_read_arr(ir+v_offset[0],jr+v_offset[1],k+v_offset[2], 0));
                        });
                   }
                }

                // This is velocity
                if (var_name == "velocity") {
                    ParallelFor(
                        bx, ncomp, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
                                int ir = i;
                                int jr = j;
                                if (normal == 0) ir = read_anchor;
                                if (normal == 1) jr = read_anchor;
                                bndry_mf_arr(i, j, k, n) = (real_bcs) ? bndry_read_arr(ir, jr, k, n) :
                                  0.5 * (bndry_read_arr(ir, jr, k, n) +
                                         bndry_read_arr(ir+v_offset[0],jr+v_offset[1],k+v_offset[2], n));
                        });
                }

            } // mfi
            bndryMF.copyTo((*data_to_fill[ori])[lev], 0, n_offset, ncomp);
          } // coordDir < 2
        } // ori
    } // var_name
}
