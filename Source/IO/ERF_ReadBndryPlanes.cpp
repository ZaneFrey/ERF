/**
 * \file ERF_ReadBndryPlanes.cpp
 */
#include "AMReX_Gpu.H"
#include "ERF_Constants.H"
#include "AMReX_ParmParse.H"
#include <AMReX_PlotFileUtil.H>
#include "ERF_ReadBndryPlanes.H"
#include "ERF_IndexDefines.H"
#include "AMReX_MultiFabUtil.H"
#include "AMReX_Utility.H"
#include "AMReX_VisMF.H"
#include "ERF_EOS.H"
#include <cmath>
#include <limits>

using namespace amrex;

namespace {

std::string lateral_face_name (Orientation ori)
{
    if (ori == Orientation(Direction::x, Orientation::low )) {
        return "xlo";
    }
    if (ori == Orientation(Direction::x, Orientation::high)) {
        return "xhi";
    }
    if (ori == Orientation(Direction::y, Orientation::low )) {
        return "ylo";
    }
    if (ori == Orientation(Direction::y, Orientation::high)) {
        return "yhi";
    }
    return "invalid";
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
    MultiFab normalized(normalized_ba, source.DistributionMap(), 1, 0);

    for (MFIter mfi(normalized); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        const auto dst = normalized.array(mfi);
        const auto src = source.const_array(mfi);
        const IntVect source_offset = source_box.smallEnd();
        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            dst(i,j,k) = src(i+source_offset[0],j+source_offset[1],k+source_offset[2]);
        });
    }

    return normalized;
}

void read_and_align_face (MultiFab& mf,
                          const std::string& filename,
                          const Box& target_box,
                          int expected_ncomp,
                          const std::string& face_name)
{
    if (!FileExists(filename + "_H")) {
        Error("Missing boundary-plane file for active face " + face_name + ": " + filename);
    }

    VisMF::Read(mf, filename);
    if (mf.nComp() != expected_ncomp) {
        Error("Boundary-plane component mismatch for active face " + face_name +
              " in " + filename);
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
int closest_index (const Vector<double>& vec, const double value)
{
    auto const it = std::upper_bound(vec.begin(), vec.end(), value);
    AMREX_ALWAYS_ASSERT(it != vec.end());

    const int idx = static_cast<int>(std::distance(vec.begin(), it));
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

            m_data_n[ori]        = std::make_unique<PlaneVector>();
            m_data_np1[ori]      = std::make_unique<PlaneVector>();
            m_data_np2[ori]      = std::make_unique<PlaneVector>();
            m_data_interp[ori]   = std::make_unique<PlaneVector>();
            m_data_tendency[ori] = std::make_unique<PlaneVector>();

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
            m_data_tendency[ori]->push_back(FArrayBox(pbx, ncomp));
        }
    }
}

/**
 * Function in ReadBndryPlanes class for interpolating boundary
 * data in time.
 *
 * @param time_in Constant specifying the time for interpolation
 */
Vector<std::unique_ptr<PlaneVector>>&
ReadBndryPlanes::interp_in_time (const double& time_in)
{
    // A restart that lands exactly on a boundary-plane time can request a time a few
    // ULP outside [m_tn, m_tnp2] because per-level t_new drifts under subcycling.
    // Tolerate that drift, then clamp into the valid window before interpolating.
    const double eps  = 1.0e-8 * (m_tnp2 - m_tn);
    AMREX_ALWAYS_ASSERT(m_tn - eps <= time_in && time_in <= m_tnp2 + eps);
    const double time = std::min(std::max(time_in, m_tn), m_tnp2);

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
                    const int nlevels = static_cast<int>(m_data_n[ori]->size());
                    for (int lev = 0; lev < nlevels; ++lev) {
                        const auto& datn   = (*m_data_n[ori])[lev];
                        const auto& datnp1 = (*m_data_np1[ori])[lev];
                        auto& dati = (*m_data_interp[ori])[lev];
                        dati.linInterp<RunOn::Device>(datn, 0, datnp1, 0,
                                                      static_cast<Real>(m_tn), static_cast<Real>(m_tnp1),
                                                      static_cast<Real>(m_tinterp),
                                                      datn.box(), 0, dati.nComp());
                    }
                }
            }
        } else {
            for (OrientationIter oit; oit != nullptr; ++oit) {
                auto ori = oit();
                if (ori.coordDir() < 2 && m_active_faces[int(ori)]) {
                    const int nlevels = static_cast<int>(m_data_n[ori]->size());
                    for (int lev = 0; lev < nlevels; ++lev) {
                        const auto& datnp1 = (*m_data_np1[ori])[lev];
                        const auto& datnp2 = (*m_data_np2[ori])[lev];
                        auto& dati = (*m_data_interp[ori])[lev];
                        dati.linInterp<RunOn::Device>(datnp1, 0, datnp2, 0,
                                                      static_cast<Real>(m_tnp1), static_cast<Real>(m_tnp2),
                                                      static_cast<Real>(m_tinterp),
                                                      datnp1.box(), 0, dati.nComp());
                    }
                }
            }
        }
    }
    return m_data_interp;
}

/**
 * Function in ReadBndryPlanes class for interpolating boundary
 * data in time.
 *
 * @param time_in Constant specifying the time for interpolation
 */
Vector<std::unique_ptr<PlaneVector>>&
ReadBndryPlanes::get_tendency (const double& time_in)
{
    // A restart that lands exactly on a boundary-plane time can request a time a few
    // ULP outside [m_tn, m_tnp2] because per-level t_new drifts under subcycling.
    // Tolerate that drift, then clamp into the valid window before interpolating.
    const double eps  = 1.0e-8 * (m_tnp2 - m_tn);
    AMREX_ALWAYS_ASSERT(m_tn - eps <= time_in && time_in <= m_tnp2 + eps);
    const double time = std::min(std::max(time_in, m_tn), m_tnp2);

    if (time < m_tnp1) {
        Real idt = static_cast<Real>(1.0 / (m_tnp1 - m_tn));
        for (OrientationIter oit; oit != nullptr; ++oit) {
            auto ori = oit();
            if (ori.coordDir() < 2 && m_active_faces[int(ori)]) {
                const int nlevels = static_cast<int>(m_data_n[ori]->size());
                for (int lev = 0; lev < nlevels; ++lev) {
                    auto& fabt = (*m_data_tendency[ori])[lev];
                    Box bx     = fabt.box();
                    int ncomp  = fabt.nComp();

                    const auto& datt   = fabt.array();
                    const auto& datn   = (*m_data_n[ori])[lev].array();
                    const auto& datnp1 = (*m_data_np1[ori])[lev].array();
                    ParallelFor(bx, ncomp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
                    {
                        datt(i,j,k,n) = (datnp1(i,j,k,n) - datn(i,j,k,n)) * idt;
                    });
                }
            }
        }
    } else {
        Real idt = static_cast<Real>(1.0 / (m_tnp2 - m_tnp1));
        for (OrientationIter oit; oit != nullptr; ++oit) {
            auto ori = oit();
            if (ori.coordDir() < 2 && m_active_faces[int(ori)]) {
                const int nlevels = static_cast<int>(m_data_n[ori]->size());
                for (int lev = 0; lev < nlevels; ++lev) {
                    auto& fabt = (*m_data_tendency[ori])[lev];
                    Box bx     = fabt.box();
                    int ncomp  = fabt.nComp();

                    const auto& datt   = fabt.array();
                    const auto& datnp1 = (*m_data_np1[ori])[lev].array();
                    const auto& datnp2 = (*m_data_np2[ori])[lev].array();
                    ParallelFor(bx, ncomp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
                    {
                        datt(i,j,k,n) = (datnp2(i,j,k,n) - datnp1(i,j,k,n)) * idt;
                    });
                }
            }
        }
    }

    return m_data_tendency;
}

const FArrayBox*
ReadBndryPlanes::interpolated_face (Orientation ori, int lev) const
{
    const int iface = int(ori);
    if (!face_is_active(ori) || !m_data_interp[iface] ||
        lev < 0 || lev >= static_cast<int>(m_data_interp[iface]->size())) {
        return nullptr;
    }
    return &(*m_data_interp[iface])[lev];
}

const FArrayBox*
ReadBndryPlanes::tendency_face (Orientation ori, int lev) const
{
    const int iface = int(ori);
    if (!face_is_active(ori) || !m_data_tendency[iface] ||
        lev < 0 || lev >= static_cast<int>(m_data_tendency[iface]->size())) {
        return nullptr;
    }
    return &(*m_data_tendency[iface])[lev];
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
    pp.queryAdd("in_rad",m_in_rad);

    // Are we using real bcs?
    pp.queryAdd("use_real_bcs", m_use_real_bcs);

    last_file_read = -1;

    m_tinterp = -1.0;

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
        for (int i = 0; i < m_var_names.size(); i++) {
            if (m_var_names[i] == "velocity")     is_velocity_read = 1;
            if (m_var_names[i] == "density")      is_density_read = 1;
            if (m_var_names[i] == "temperature")  is_temperature_read = 1;
            if (m_var_names[i] == "theta")        is_theta_read = 1;
            if (m_var_names[i] == "scalar")       is_scalar_read = 1;
            if (m_var_names[i] == "qv")           is_q1_read = 1;
            if (m_var_names[i] == "qc")           is_q2_read = 1;
            if (m_var_names[i] == "ke")           is_KE_read = 1;
            const bool known = m_var_names[i] == "velocity" || m_var_names[i] == "density"     ||
                               m_var_names[i] == "temperature" || m_var_names[i] == "theta" ||
                               m_var_names[i] == "scalar"   || m_var_names[i] == "qv"          ||
                               m_var_names[i] == "qc"       || m_var_names[i] == "ke";
            if (!known) {
                Abort("erf.bndry_input_var_names: unknown variable \"" + m_var_names[i] +
                      "\"; the names read are velocity, density, temperature, theta, scalar, qv, qc and ke");
            }
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
    m_data_tendency.resize(size);
}

void
ReadBndryPlanes::initialize_active_faces (const Vector<BCRec>& bcs,
                                          const MultiFab& z_phys_nd)
{
    m_active_faces.fill(false);
    int num_active = 0;

    for (OrientationIter oit; oit != nullptr; ++oit) {
        const Orientation ori = oit();
        if (ori.coordDir() >= 2) {
            continue;
        }

        const int dir = ori.coordDir();
        for (const auto& bc : bcs) {
            const int bc_type = ori.isLow() ? bc.lo(dir) : bc.hi(dir);
            if (bc_type == ERFBCType::ext_dir_ingested) {
                m_active_faces[int(ori)] = true;
                ++num_active;
                break;
            }
        }
    }

    if (num_active == 0) {
        Error("erf.input_bndry_planes is enabled, but no lateral boundary uses ingested data");
    }

    validate_terrain_coordinates(z_phys_nd);
}

void
ReadBndryPlanes::validate_terrain_coordinates (const MultiFab& z_phys_nd)
{
    bool any_coordinate_file = false;
    for (OrientationIter oit; oit != nullptr; ++oit) {
        const Orientation ori = oit();
        if (ori.coordDir() < 2) {
            const std::string z_name = m_filename + "/z_phys_nd_" + lateral_face_name(ori);
            any_coordinate_file = any_coordinate_file || FileExists(z_name + "_H");
        }
    }

    if (!any_coordinate_file) {
        if (ParallelDescriptor::IOProcessor()) {
            Warning("Boundary-plane directory has no terrain-coordinate files; active-face "
                    "array shapes will be checked, but vertical coordinates are unverified.");
        }
        return;
    }

    const Box& domain = m_geom.Domain();
    for (OrientationIter oit; oit != nullptr; ++oit) {
        const Orientation ori = oit();
        if (ori.coordDir() >= 2 || !m_active_faces[int(ori)]) {
            continue;
        }

        const std::string face = lateral_face_name(ori);
        const std::string z_name = m_filename + "/z_phys_nd_" + face;
        if (!FileExists(z_name + "_H")) {
            Error("Missing terrain-coordinate file for active face " + face + ": " + z_name);
        }

        MultiFab source_z;
        VisMF::Read(source_z, z_name);
        MultiFab target_z = make_normalized_z_face(z_phys_nd, domain, ori);
        if (source_z.nComp() != 1 || source_z.boxArray().size() != 1 ||
            !amrex::match(source_z.boxArray(), target_z.boxArray())) {
            Error("Terrain-coordinate shape mismatch on active face " + face);
        }

        MultiFab source_on_target(target_z.boxArray(), target_z.DistributionMap(), 1, 0);
        source_on_target.ParallelCopy(source_z, 0, 0, 1);
        MultiFab diff(target_z.boxArray(), target_z.DistributionMap(), 1, 0);
        MultiFab::Copy(diff, target_z, 0, 0, 1, 0);
        MultiFab::Subtract(diff, source_on_target, 0, 0, 1, 0);
        diff.abs(0, 1, 0);

        const Real max_error = diff.norm0();
        const Real vertical_extent = std::abs(m_geom.ProbHi(2) - m_geom.ProbLo(2));
        const Real z_tol = 128.0 * std::numeric_limits<Real>::epsilon() *
                           std::max(Real(1.0), vertical_extent);
        if (max_error > z_tol) {
            Error("Terrain-coordinate mismatch on active face " + face +
                  "; maximum absolute difference is " + std::to_string(max_error));
        }
    }
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

    // Allocate data we will need -- for now just at one level
    int lev = 0;
    define_level_data(lev);
    Print() << "Successfully read time file and allocated data" << std::endl;
}

/**
 * Function in ReadBndryPlanes for reading boundary data
 * at a specific time and at the next timestep from input files.
 *
 * @param time Current time
 * @param dt Current timestep
 * @param m_bc_extdir_vals Container storing the external dirichlet boundary conditions we are reading from the input files
 */
void ReadBndryPlanes::read_input_files (double time,
                                        double dt,
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
        read_file(idx_init,m_data_n     ,m_bc_extdir_vals);
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
                                 Array<Array<Real, AMREX_SPACEDIM*2>,AMREX_SPACEDIM+NBCVAR_max> m_bc_extdir_vals)
{
    if (idx >= m_in_timesteps.size()) {
        Print() << "Asking for index " << idx << " but m_in_timesteps only has size " << m_in_timesteps.size() << std::endl;
        Abort();
    }
    const int t_step = m_in_timesteps[idx];
    const std::string chkname1 = m_filename + Concatenate("/bndry_output", t_step);

    const std::string level_prefix = "Level_";
    const int lev = 0;

    const Box& domain = m_geom.Domain();

    GpuArray<GpuArray<Real, AMREX_SPACEDIM*2>, AMREX_SPACEDIM+NBCVAR_max> l_bc_extdir_vals_d;

    for (int i = 0; i < BCVars::NumTypes; i++)
    {
        for (OrientationIter oit; oit != nullptr; ++oit) {
            auto ori = oit();
            l_bc_extdir_vals_d[i][ori] = m_bc_extdir_vals[i][ori];
        }
    }

    int n_for_density = -1;
    for (int i = 0; i < m_var_names.size(); i++)
    {
       if (m_var_names[i] == "density") n_for_density = i;
    }

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
                d_arr(i,j,k,n) = zero;
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

        // Every name was checked against this list in the constructor; the -1 only
        // keeps the compiler from seeing an unset value.
        int n_offset = -1;
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

                // Split the 2-cell-thick working box into ghost and interior
                // slots so the (i+v_offset) neighbor access stays in-bounds.
                // Both slots are filled with the same face-averaged Dirichlet
                // value; the interior slot just needs the opposite neighbor.
                Box bx_ghost = bx;
                Box bx_int   = bx;
                if (ori.isLow()) {
                    bx_ghost.setBig  (normal, domain.smallEnd(normal) - 1);
                    bx_int  .setSmall(normal, domain.smallEnd(normal));
                } else {
                    bx_ghost.setSmall(normal, domain.bigEnd(normal) + 1);
                    bx_int  .setBig  (normal, domain.bigEnd(normal));
                }
                const IntVect v_offset_int = -v_offset;

                // We average the two cell-centered data points in the normal direction
                //    to define a Dirichlet value on the face itself.

                // This is the scalars -- they all get multiplied by rho, and in the case of
                //   reading in temperature, we must convert to theta first
                Real rdOcp = m_rdOcp;
                if (n_for_density >= 0) {
                  if (var_name == "temperature") {
                    ParallelFor(
                        bx_ghost, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                             Real R1 =  bndry_read_r_arr(i, j, k, 0);
                             Real R2 =  bndry_read_r_arr(i+v_offset[0],j+v_offset[1],k+v_offset[2],0);
                             Real T1 =  bndry_read_arr(i, j, k, 0);
                             Real T2 =  bndry_read_arr(i+v_offset[0],j+v_offset[1],k+v_offset[2],0);
                             Real Th1 = getThgivenRandT(R1,T1,rdOcp);
                             Real Th2 = getThgivenRandT(R2,T2,rdOcp);
                             bndry_mf_arr(i, j, k, 0) = (real_bcs) ? bndry_read_arr(i, j, k, 0) :
                                                                     myhalf * (R1*Th1 + R2*Th2);
                        });
                  } else if (var_name == "theta" || var_name == "ke" || var_name == "scalar" ||
                             var_name == "qv"    || var_name == "qc") {
                    ParallelFor(
                        bx_ghost, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                             Real R1 =  bndry_read_r_arr(i, j, k, 0);
                             Real R2 =  bndry_read_r_arr(i+v_offset[0],j+v_offset[1],k+v_offset[2],0);
                             bndry_mf_arr(i, j, k, 0) = (real_bcs) ? bndry_read_arr(i, j, k, 0) :
                                 myhalf * ( R1 * bndry_read_arr(i, j, k, 0) +
                                         R2 * bndry_read_arr(i+v_offset[0],j+v_offset[1],k+v_offset[2], 0));
                        });
                   } else if (var_name == "density") {
                    ParallelFor(
                        bx_ghost, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                                bndry_mf_arr(i, j, k, 0) = (real_bcs) ? bndry_read_arr(i, j, k, 0) :
                                    myhalf * ( bndry_read_arr(i, j, k, 0) +
                                            bndry_read_arr(i+v_offset[0],j+v_offset[1],k+v_offset[2], 0));
                        });
                   }
                } else if (!ingested_density()) {
                  if (var_name == "temperature") {
                    ParallelFor(
                        bx_ghost, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                             Real R1  = l_bc_extdir_vals_d[BCVars::Rho_bc_comp][ori];
                             Real R2  = l_bc_extdir_vals_d[BCVars::Rho_bc_comp][ori];
                             Real T1  = bndry_read_arr(i, j, k, 0);
                             Real T2  = bndry_read_arr(i+v_offset[0],j+v_offset[1],k+v_offset[2], 0);
                             Real Th1 = getThgivenRandT(R1,T1,rdOcp);
                             Real Th2 = getThgivenRandT(R2,T2,rdOcp);
                             bndry_mf_arr(i, j, k, 0) = (real_bcs) ? bndry_read_arr(i, j, k, 0) :
                                                                     myhalf * (R1*Th1 + R2*Th2);
                        });
                  } else if (var_name == "theta" || var_name == "ke" || var_name == "scalar" ||
                             var_name == "qv"    || var_name == "qc") {
                      ParallelFor(
                        bx_ghost, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                             Real R1  = l_bc_extdir_vals_d[BCVars::Rho_bc_comp][ori];
                             Real R2  = l_bc_extdir_vals_d[BCVars::Rho_bc_comp][ori];
                             bndry_mf_arr(i, j, k, 0) = (real_bcs) ? bndry_read_arr(i, j, k, 0) :
                                myhalf * (R1 * bndry_read_arr(i, j, k, 0) +
                                       R2 * bndry_read_arr(i+v_offset[0],j+v_offset[1],k+v_offset[2], 0));
                        });
                  }
                }

                // This is velocity
                if (var_name == "velocity") {
                    ParallelFor(
                        bx_ghost, ncomp, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
                                bndry_mf_arr(i, j, k, n) = (real_bcs) ? bndry_read_arr(i, j, k, n) :
                                  myhalf * (bndry_read_arr(i, j, k, n) +
                                         bndry_read_arr(i+v_offset[0],j+v_offset[1],k+v_offset[2], n));
                        });
                }

                // --- interior-slot fill (same face-averaged Dirichlet value;
                //     neighbor offset is flipped because the "other cell"
                //     is now on the opposite side of the boundary face) ---
                if (n_for_density >= 0) {
                    if (var_name == "temperature") {
                        ParallelFor(
                            bx_int, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                                 Real R1 =  bndry_read_r_arr(i, j, k, 0);
                                 Real R2 =  bndry_read_r_arr(i+v_offset_int[0],j+v_offset_int[1],k+v_offset_int[2],0);
                                 Real T1 =  bndry_read_arr(i, j, k, 0);
                                 Real T2 =  bndry_read_arr(i+v_offset_int[0],j+v_offset_int[1],k+v_offset_int[2],0);
                                 Real Th1 = getThgivenRandT(R1,T1,rdOcp);
                                 Real Th2 = getThgivenRandT(R2,T2,rdOcp);
                                 bndry_mf_arr(i, j, k, 0) = (real_bcs) ? bndry_read_arr(i, j, k, 0) :
                                                                         myhalf * (R1*Th1 + R2*Th2);
                            });
                    } else if (var_name == "theta" || var_name == "ke" || var_name == "scalar" ||
                               var_name == "qv"    || var_name == "qc") {
                        ParallelFor(
                            bx_int, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                                 Real R1 =  bndry_read_r_arr(i, j, k, 0);
                                 Real R2 =  bndry_read_r_arr(i+v_offset_int[0],j+v_offset_int[1],k+v_offset_int[2],0);
                                 bndry_mf_arr(i, j, k, 0) = (real_bcs) ? bndry_read_arr(i, j, k, 0) :
                                     myhalf * ( R1 * bndry_read_arr(i, j, k, 0) +
                                             R2 * bndry_read_arr(i+v_offset_int[0],j+v_offset_int[1],k+v_offset_int[2], 0));
                            });
                    } else if (var_name == "density") {
                        ParallelFor(
                            bx_int, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                                    bndry_mf_arr(i, j, k, 0) = (real_bcs) ? bndry_read_arr(i, j, k, 0) :
                                        myhalf * ( bndry_read_arr(i, j, k, 0) +
                                                bndry_read_arr(i+v_offset_int[0],j+v_offset_int[1],k+v_offset_int[2], 0));
                            });
                    }
                } else if (!ingested_density()) {
                    if (var_name == "temperature") {
                        ParallelFor(
                            bx_int, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                                 Real R1  = l_bc_extdir_vals_d[BCVars::Rho_bc_comp][ori];
                                 Real R2  = l_bc_extdir_vals_d[BCVars::Rho_bc_comp][ori];
                                 Real T1  = bndry_read_arr(i, j, k, 0);
                                 Real T2  = bndry_read_arr(i+v_offset_int[0],j+v_offset_int[1],k+v_offset_int[2], 0);
                                 Real Th1 = getThgivenRandT(R1,T1,rdOcp);
                                 Real Th2 = getThgivenRandT(R2,T2,rdOcp);
                                 bndry_mf_arr(i, j, k, 0) = (real_bcs) ? bndry_read_arr(i, j, k, 0) :
                                                                         myhalf * (R1*Th1 + R2*Th2);
                            });
                    } else if (var_name == "theta" || var_name == "ke" || var_name == "scalar" ||
                               var_name == "qv"    || var_name == "qc") {
                          ParallelFor(
                            bx_int, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                                 Real R1  = l_bc_extdir_vals_d[BCVars::Rho_bc_comp][ori];
                                 Real R2  = l_bc_extdir_vals_d[BCVars::Rho_bc_comp][ori];
                                 bndry_mf_arr(i, j, k, 0) = (real_bcs) ? bndry_read_arr(i, j, k, 0) :
                                    myhalf * (R1 * bndry_read_arr(i, j, k, 0) +
                                           R2 * bndry_read_arr(i+v_offset_int[0],j+v_offset_int[1],k+v_offset_int[2], 0));
                            });
                    }
                }

                if (var_name == "velocity") {
                    ParallelFor(
                        bx_int, ncomp, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
                                bndry_mf_arr(i, j, k, n) = (real_bcs) ? bndry_read_arr(i, j, k, n) :
                                  myhalf * (bndry_read_arr(i, j, k, n) +
                                         bndry_read_arr(i+v_offset_int[0],j+v_offset_int[1],k+v_offset_int[2], n));
                        });
                }

            } // mfi
            bndryMF.copyTo((*data_to_fill[ori])[lev], 0, n_offset, ncomp);
          } // coordDir < 2
        } // ori
    } // var_name
}
