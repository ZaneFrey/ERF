/**
 * \file ERF_InitWindFarm.cpp
 */

#include <ERF_WindFarm.H>
#include <filesystem>
#include <dirent.h>   // For POSIX directory handling
#include <algorithm> // For std::sort
#include <fstream>
#include <iomanip>
#include <sstream>
#include <cmath>

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

void append_to_pvd (const std::string& pvd_name,
                    const std::string& vtk_file,
                    const amrex::Real time)
{
    std::string header =
        "<?xml version=\"1.0\"?>\n"
        "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
        "  <Collection>\n";
    std::string footer =
        "  </Collection>\n"
        "</VTKFile>\n";

    std::ostringstream dataset;
    dataset << "    <DataSet timestep=\"" << std::setprecision(17) << time
            << "\" group=\"\" part=\"0\" file=\"" << vtk_file << "\"/>\n";

    std::ifstream in(pvd_name);
    if (!in.good()) {
        std::ofstream out(pvd_name, std::ios::out | std::ios::trunc);
        out << header << dataset.str() << footer;
        return;
    }

    std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    auto pos = content.rfind("  </Collection>");
    if (pos == std::string::npos) {
        std::ofstream out(pvd_name, std::ios::out | std::ios::trunc);
        out << header << dataset.str() << footer;
        return;
    }

    content.insert(pos, dataset.str());
    std::ofstream out(pvd_name, std::ios::out | std::ios::trunc);
    out << content;
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
    if(x_y) {
        init_windfarm_x_y(windfarm_loc_table);
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
                               std::unique_ptr<MultiFab>& z_phys_nd)
{

    zloc.resize(xloc.size(),0.0);
    Vector<int> is_counted;
    is_counted.resize(xloc.size(),0);

    amrex::Gpu::DeviceVector<Real> d_xloc(xloc.size());
    amrex::Gpu::DeviceVector<Real> d_yloc(yloc.size());
    amrex::Gpu::DeviceVector<Real> d_zloc(xloc.size());
    amrex::Gpu::DeviceVector<int> d_is_counted(xloc.size());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, xloc.begin(), xloc.end(), d_xloc.begin());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, yloc.begin(), yloc.end(), d_yloc.begin());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, zloc.begin(), zloc.end(), d_zloc.begin());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, is_counted.begin(), is_counted.end(), d_is_counted.begin());

    Real* d_xloc_ptr       = d_xloc.data();
    Real* d_yloc_ptr       = d_yloc.data();
    Real* d_zloc_ptr       = d_zloc.data();
    int* d_is_counted_ptr = d_is_counted.data();

    mf_Nturb.setVal(0);

    int i_lo = geom.Domain().smallEnd(0); int i_hi = geom.Domain().bigEnd(0);
    int j_lo = geom.Domain().smallEnd(1); int j_hi = geom.Domain().bigEnd(1);
    auto dx = geom.CellSizeArray();
    if(dx[0]<= 1e-3 or dx[1]<=1e-3 or dx[2]<= 1e-3) {
        Abort("The value of mesh spacing for wind farm parametrization cannot be less than 1e-3 m. "
              "It should be usually of order 1 m");
    }
    auto ProbLoArr = geom.ProbLoArray();
    auto ProbHiArr = geom.ProbHiArray();
    int num_turb = xloc.size();

    bool is_terrain = z_phys_nd ? true: false;

     // Initialize wind farm
    for ( MFIter mfi(mf_Nturb,TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx     = mfi.tilebox();
        auto  Nturb_array = mf_Nturb.array(mfi);
        const Array4<const Real>& z_nd_arr = (z_phys_nd) ? z_phys_nd->const_array(mfi) : Array4<Real>{};
        int k0 = bx.smallEnd()[2];
        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            int li = amrex::min(amrex::max(i, i_lo), i_hi);
            int lj = amrex::min(amrex::max(j, j_lo), j_hi);

            Real x1 = ProbLoArr[0] + li*dx[0];
            Real x2 = ProbLoArr[0] + (li+1)*dx[0];
            Real y1 = ProbLoArr[1] + lj*dx[1];
            Real y2 = ProbLoArr[1] + (lj+1)*dx[1];

            for(int it=0; it<num_turb; it++){
                if( d_xloc_ptr[it]+1e-3 > x1 and d_xloc_ptr[it]+1e-3 < x2 and
                    d_yloc_ptr[it]+1e-3 > y1 and d_yloc_ptr[it]+1e-3 < y2){
                    Nturb_array(i,j,k,0) = Nturb_array(i,j,k,0) + 1;
                    // Perform atomic operations to ensure "increment only once"
                    if (is_terrain) {
                        int expected = 0;
                        int desired = 1;
                        // Atomic Compare-And-Swap: Increment only if d_is_counted_ptr[it] was 0
                        if (Gpu::Atomic::CAS(&d_is_counted_ptr[it], expected, desired) == expected) {
                            // The current thread successfully set d_is_counted_ptr[it] from 0 to 1
                            Gpu::Atomic::Add(&d_zloc_ptr[it], z_nd_arr(i, j, k0));
                        }
                    }
                }
            }
        });
    }

    Gpu::copy(Gpu::deviceToHost, d_zloc.begin(), d_zloc.end(), zloc.begin());
    Gpu::copy(Gpu::deviceToHost, d_is_counted.begin(), d_is_counted.end(), is_counted.begin());

    amrex::ParallelAllReduce::Sum(zloc.data(),
                                  zloc.size(),
                                  amrex::ParallelContext::CommunicatorAll());

    amrex::ParallelAllReduce::Sum(is_counted.data(),
                                  is_counted.size(),
                                  amrex::ParallelContext::CommunicatorAll());

    for(int it=0;it<num_turb;it++) {
        if(is_terrain and
           xloc[it] > ProbLoArr[0] and
           xloc[it] < ProbHiArr[0] and
           yloc[it] > ProbLoArr[1] and
           yloc[it] < ProbHiArr[1]    ) {
            if(is_counted[it] != 1) {
                Abort("Wind turbine " + std::to_string(it) + "has been counted " + std::to_string(is_counted[it]) + " times" +
                      " It should have been counted only once. Aborting....");
            }
        }
    }

    // Debugging
    /*int my_rank = amrex::ParallelDescriptor::MyProc();

    for(int it=0;it<num_turb;it++) {
        std::cout << "The value of zloc is " << my_rank << " " << zloc[it] << " " << is_counted[it] << "\n";
    }*/

    set_turb_zloc(zloc);
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
                               std::unique_ptr<MultiFab>& z_phys_cc)
{
    amrex::Gpu::DeviceVector<Real> d_xloc(xloc.size());
    amrex::Gpu::DeviceVector<Real> d_yloc(yloc.size());
    amrex::Gpu::DeviceVector<Real> d_zloc(xloc.size());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, xloc.begin(), xloc.end(), d_xloc.begin());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, yloc.begin(), yloc.end(), d_yloc.begin());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, zloc.begin(), zloc.end(), d_zloc.begin());

    Real d_rotor_rad = rotor_rad;
    Real d_hub_height = hub_height;
    Real d_sampling_distance = sampling_distance_by_D*2.0*rotor_rad;

    Real* d_xloc_ptr     = d_xloc.data();
    Real* d_yloc_ptr     = d_yloc.data();
    Real* d_zloc_ptr     = d_zloc.data();

    mf_SMark.setVal(-1.0);
    mf_RMask.setVal(-1.0);

    int i_lo = geom.Domain().smallEnd(0); int i_hi = geom.Domain().bigEnd(0);
    int j_lo = geom.Domain().smallEnd(1); int j_hi = geom.Domain().bigEnd(1);
    int k_lo = geom.Domain().smallEnd(2); int k_hi = geom.Domain().bigEnd(2);
    auto dx = geom.CellSizeArray();
    auto ProbLoArr = geom.ProbLoArray();
    int num_turb = xloc.size();

    Real theta = turb_disk_angle*M_PI/180.0-0.5*M_PI;

    set_turb_disk_angle(theta);
    my_turb_disk_angle = theta;
    {
        amrex::Vector<amrex::Real> theta_rad(xloc.size(), theta);
        set_turb_disk_angles(theta_rad);
    }

    Real nx = -std::cos(theta);
    Real ny = -std::sin(theta);

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
            for(int it=0; it<num_turb; it++){
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
                if(is_cell_marked) {
                    SMark_array(i,j,k,1) = it;
                    Real xc = ProbLoArr[0] + (ii+0.5_rt)*dx[0];
                    Real yc = ProbLoArr[1] + (jj+0.5_rt)*dx[1];
                    Real zhub = d_hub_height + z0;
                    Real dxp = xc - x0;
                    Real dyp = yc - y0;
                    Real dzp = z - zhub;
                    Real a = dxp*nx + dyp*ny;
                    Real rx = dxp - a*nx;
                    Real ry = dyp - a*ny;
                    RMask_array(i,j,k,0) = std::sqrt(rx*rx + ry*ry + dzp*dzp);
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
    if (!m_dynamic_yaw_enabled) {
        return false;
    }

    const amrex::Real step_end = time + dt;
    const amrex::Real eps = 1e-12;
    const int nturb = static_cast<int>(xloc.size());

    bool did_cmd_update = false;

    // Command update (boxcar over yaw_period)
    for (int it = 0; it < nturb; ++it) {
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
    for (int it = 0; it < nturb; ++it) {
        const amrex::Real e = wrap_180(m_yaw_cmd_deg[it] - m_yaw_angle_deg[it]);
        m_yaw_angle_deg[it] = wrap_180(m_yaw_angle_deg[it] + (dt / m_tau_yaw) * e);
    }

    // Push per-turbine disk angles down into the actuator model each timestep.
    amrex::Vector<amrex::Real> disk_face_angles_deg;
    get_disk_face_angles_deg(disk_face_angles_deg);
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
    if (!m_dynamic_yaw_enabled) {
        return false;
    }
    const int nturb = static_cast<int>(xloc.size());
    const amrex::Real alpha_dx = m_yaw_alpha * dx_eff;
    for (int it = 0; it < nturb; ++it) {
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
    m_yaw_angle_geom_deg = m_yaw_angle_deg;
}

void
WindFarm::get_disk_face_angles_deg (amrex::Vector<amrex::Real>& disk_face_angles_deg) const
{
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
                                       std::unique_ptr<amrex::MultiFab>& z_phys_cc)
{
    amrex::Gpu::DeviceVector<Real> d_xloc(xloc.size());
    amrex::Gpu::DeviceVector<Real> d_yloc(yloc.size());
    amrex::Gpu::DeviceVector<Real> d_zloc(yloc.size());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, xloc.begin(), xloc.end(), d_xloc.begin());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, yloc.begin(), yloc.end(), d_yloc.begin());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, zloc.begin(), zloc.end(), d_zloc.begin());

    const int num_turb = static_cast<int>(xloc.size());

    amrex::Vector<amrex::Real> theta_rad(num_turb, 0.0);
    amrex::Vector<amrex::Real> nx_h(num_turb, 0.0);
    amrex::Vector<amrex::Real> ny_h(num_turb, 0.0);
    for (int it = 0; it < num_turb; ++it) {
        theta_rad[it] = disk_face_angles_deg[it] * M_PI/180.0 - 0.5*M_PI;
        nx_h[it] = -std::cos(theta_rad[it]);
        ny_h[it] = -std::sin(theta_rad[it]);
    }
    set_turb_disk_angles(theta_rad);

    amrex::Gpu::DeviceVector<Real> d_nx(num_turb);
    amrex::Gpu::DeviceVector<Real> d_ny(num_turb);
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, nx_h.begin(), nx_h.end(), d_nx.begin());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, ny_h.begin(), ny_h.end(), d_ny.begin());

    Real* d_xloc_ptr     = d_xloc.data();
    Real* d_yloc_ptr     = d_yloc.data();
    Real* d_zloc_ptr     = d_zloc.data();
    Real* d_nx_ptr       = d_nx.data();
    Real* d_ny_ptr       = d_ny.data();

    Real d_rotor_rad = rotor_rad;
    Real d_hub_height = hub_height;
    Real d_sampling_distance = sampling_distance_by_D*2.0*rotor_rad;

    mf_SMark.setVal(-1.0);
    mf_RMask.setVal(-1.0);

    int i_lo = geom.Domain().smallEnd(0); int i_hi = geom.Domain().bigEnd(0);
    int j_lo = geom.Domain().smallEnd(1); int j_hi = geom.Domain().bigEnd(1);
    int k_lo = geom.Domain().smallEnd(2); int k_hi = geom.Domain().bigEnd(2);
    auto dx = geom.CellSizeArray();
    auto ProbLoArr = geom.ProbLoArray();

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

            for (int it = 0; it < num_turb; ++it) {
                Real nx = d_nx_ptr[it];
                Real ny = d_ny_ptr[it];

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
                if (is_cell_marked) {
                    SMark_array(i,j,k,1) = it;
                    Real xc = ProbLoArr[0] + (ii+0.5_rt)*dx[0];
                    Real yc = ProbLoArr[1] + (jj+0.5_rt)*dx[1];
                    Real zhub = d_hub_height + z0;
                    Real dxp = xc - x0;
                    Real dyp = yc - y0;
                    Real dzp = z - zhub;
                    Real a = dxp*nx + dyp*ny;
                    Real rx = dxp - a*nx;
                    Real ry = dyp - a*ny;
                    RMask_array(i,j,k,0) = std::sqrt(rx*rx + ry*ry + dzp*dzp);
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

    append_to_pvd("turbine_locations.pvd", f_turb, time);
    append_to_pvd("actuator_disks_all.pvd", f_all, time);
    append_to_pvd("actuator_disks_in_dom.pvd", f_dom, time);
    append_to_pvd("averaging_disks_in_dom.pvd", f_avg, time);
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
