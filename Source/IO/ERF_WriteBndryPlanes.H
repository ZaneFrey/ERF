#ifndef ERF_WRITEBNDRYPLANES_H
#define ERF_WRITEBNDRYPLANES_H

#include "AMReX_Gpu.H"
#include "AMReX_AmrCore.H"
#include <AMReX_BndryRegister.H>
#include <array>


/** Interface for writing boundary planes
 *
 *  This class performs the necessary file operations to write boundary planes
 *
 */
class WriteBndryPlanes
{
public:
    explicit WriteBndryPlanes (amrex::Vector<amrex::BoxArray>& grids,
                               amrex::Vector<amrex::Geometry>& geom,
                               const amrex::Vector<std::unique_ptr<amrex::MultiFab>>& z_phys_nd);

    void write_planes (int t_step, amrex::Real time,
                       amrex::Vector<amrex::Vector<amrex::MultiFab>>& vars_new,
                       bool is_moist);

private:

    //! IO output box region
    amrex::Box target_box;

    //! Geometry objects for all levels
    amrex::Vector<amrex::Geometry>& m_geom;

    //! File name for IO
    std::string m_filename{""};

    //! File name for Native time file
    std::string m_time_file{""};

    //! Variables for IO
    amrex::Vector<std::string> m_var_names;

    //! Lateral faces selected for output (z faces are never written)
    std::array<bool,2*AMREX_SPACEDIM> m_output_faces{};

    //! Timestep and times to be stored in time.dat
    amrex::Vector<amrex::Real> m_in_times;
    amrex::Vector<int> m_in_timesteps;

    // This defaults to level 0, but if we can fit the whole target_box
    // inside a higher level, we will use the finest data possible
    static int bndry_lev;

    //! controls extents on native bndry output
    int m_in_rad        = 1;
    const int m_out_rad = 1;
    const int m_extent_rad = 0;

    void write_header_and_coordinates (const amrex::MultiFab& z_phys_nd);
};

#endif /* ERF_BOUNDARYPLANE_H */
