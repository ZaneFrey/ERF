#include <ERF_SimpleAD.H>
#include <ERF_IndexDefines.H>
#include <ERF_Interpolation_1D.H>
#include <ERF_Constants.H>
#include <fstream>
#include <iomanip>
#include <sstream>

using namespace amrex;

void
SimpleAD::advance (const Geometry& geom,
                   const Real& dt_advance,
                   MultiFab& cons_in,
                   MultiFab& mf_vars_simpleAD,
                   MultiFab& U_old,
                   MultiFab& V_old,
                   MultiFab& W_old,
                   const MultiFab& mf_Nturb,
                   const MultiFab& mf_RMask,
                   const MultiFab& mf_SMark,
                   const Real& time)
{
    AMREX_ALWAYS_ASSERT(W_old.nComp() > 0);
    AMREX_ALWAYS_ASSERT(mf_Nturb.nComp() > 0);
    AMREX_ALWAYS_ASSERT(mf_RMask.nComp() > 0);
    AMREX_ALWAYS_ASSERT(mf_vars_simpleAD.nComp() > 2);
    compute_freestream_velocity(cons_in, U_old, V_old, mf_SMark);
    source_terms_cellcentered(geom, dt_advance, cons_in, mf_SMark, mf_RMask, mf_vars_simpleAD);
    update(dt_advance, cons_in, U_old, V_old, W_old, mf_vars_simpleAD);
    compute_power_output(time);
}

void
SimpleAD::compute_power_output (const Real& time)
{
	     get_turb_loc(xloc, yloc);
	     get_turb_spec(rotor_rad, hub_height, thrust_coeff_standing,
	                  wind_speed, thrust_coeff, power);

	     const int n_spec_table = wind_speed.size();
	  // Compute power based on the look-up table

	    if (ParallelDescriptor::IOProcessor()){
	        static std::ofstream file("power_output_SimpleAD.txt", std::ios::app);
	        static bool wrote_header = false;
	        // Check if the file opened successfully
	        if (!file.is_open()) {
	            std::cerr << "Error opening file!" << std::endl;
	            Abort("Could not open file to write power output in ERF_AdvanceSimpleAD.cpp");
	        }

	        if (!wrote_header) {
	            file << "# time";
	            for (int it = 0; it < xloc.size(); ++it) {
	                file << " P_turb" << it;
	            }
	            file << "\n";
	            wrote_header = true;
	        }

	        file << time;
	        for (int it = 0; it < xloc.size(); ++it) {
	            Real avg_vel = freestream_velocity[it]/(disk_cell_count[it] + 1e-10);
	            Real turb_power = interpolate_1d(wind_speed.data(), power.data(), avg_vel, n_spec_table);
	            file << " " << turb_power;
	            //printf("avg vel and power is %d %0.15g, %0.15g\n", it, avg_vel, turb_power);
	        }
	        file << "\n";
	        file.flush();
	    }
}

void
SimpleAD::write_memory_state (const std::string& checkpointname) const
{
    if (!ParallelDescriptor::IOProcessor()) {
        return;
    }

    std::ofstream out(checkpointname + "/SimpleADMemoryState", std::ios::out | std::ios::trunc);
    if (!out.good()) {
        amrex::Abort("Failed to open SimpleADMemoryState for checkpoint write");
    }

    out << std::setprecision(17);
    out << "SimpleADMemoryState_v1\n";
    out << m_xloc.size() << " " << static_cast<int>(filtered_disk_velocity_initialized) << "\n";
    for (int it = 0; it < static_cast<int>(m_xloc.size()); ++it) {
        const Real stored_value = (it < static_cast<int>(filtered_disk_velocity_nhat.size()))
                                    ? filtered_disk_velocity_nhat[it]
                                    : 0.0;
        out << stored_value << "\n";
    }
}

bool
SimpleAD::read_memory_state (const std::string& restart_chkfile)
{
    std::string fname = restart_chkfile + "/SimpleADMemoryState";
    if (!amrex::FileExists(fname)) {
        filtered_disk_velocity_nhat.clear();
        filtered_disk_velocity_initialized = false;
        return false;
    }

    Vector<char> fileCharPtr;
    ParallelDescriptor::ReadAndBcastFile(fname, fileCharPtr);
    std::string content(fileCharPtr.dataPtr());
    std::istringstream is(content, std::istringstream::in);

    std::string tag;
    is >> tag;
    if (tag != "SimpleADMemoryState_v1") {
        amrex::Abort("Unknown SimpleADMemoryState format");
    }

    int nturb_file = 0;
    int initialized_flag = 0;
    is >> nturb_file >> initialized_flag;
    if (nturb_file != static_cast<int>(m_xloc.size())) {
        amrex::Abort("SimpleADMemoryState: number of turbines does not match current configuration");
    }

    filtered_disk_velocity_nhat.resize(nturb_file, 0.0);
    for (int it = 0; it < nturb_file; ++it) {
        is >> filtered_disk_velocity_nhat[it];
    }
    filtered_disk_velocity_initialized = (initialized_flag != 0);
    return true;
}

void
SimpleAD::update (const Real& dt_advance,
                  MultiFab& cons_in,
                  MultiFab& U_old,
                  MultiFab& V_old,
                  MultiFab& W_old,
                  const MultiFab& mf_vars_simpleAD)
{

    for ( MFIter mfi(cons_in,TilingIfNotGPU()); mfi.isValid(); ++mfi) {

        Box tbx = mfi.nodaltilebox(0);
        Box tby = mfi.nodaltilebox(1);
        Box tbz = mfi.nodaltilebox(2);

        auto simpleAD_array = mf_vars_simpleAD.array(mfi);
        auto u_vel       = U_old.array(mfi);
        auto v_vel       = V_old.array(mfi);
        auto w_vel       = W_old.array(mfi);

        ParallelFor(tbx, tby, tbz,
        [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
        {
            u_vel(i,j,k) = u_vel(i,j,k) + (simpleAD_array(i-1,j,k,0) + simpleAD_array(i,j,k,0))/2.0*dt_advance;
        },
        [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
        {
            v_vel(i,j,k) = v_vel(i,j,k) + (simpleAD_array(i,j-1,k,1) + simpleAD_array(i,j,k,1))/2.0*dt_advance;
        },
        [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
        {
            w_vel(i,j,k) = w_vel(i,j,k) + (simpleAD_array(i,j,k-1,2) + simpleAD_array(i,j,k,2))/2.0*dt_advance;
        });
    }
}

void SimpleAD::compute_freestream_velocity (const MultiFab& cons_in,
                                            const MultiFab& U_old,
                                            const MultiFab& V_old,
                                            const MultiFab& mf_SMark)
{
     get_turb_loc(xloc, yloc);
     freestream_velocity.clear();
     freestream_phi.clear();
     disk_cell_count.clear();
     freestream_velocity.resize(xloc.size(),0.0);
     freestream_phi.resize(xloc.size(),0.0);
     disk_cell_count.resize(xloc.size(),0.0);

     Gpu::DeviceVector<Real> d_freestream_velocity(xloc.size());
     Gpu::DeviceVector<Real> d_freestream_phi(yloc.size());
     Gpu::DeviceVector<Real> d_disk_cell_count(yloc.size());
     Gpu::copy(Gpu::hostToDevice, freestream_velocity.begin(), freestream_velocity.end(), d_freestream_velocity.begin());
     Gpu::copy(Gpu::hostToDevice, freestream_phi.begin(), freestream_phi.end(), d_freestream_phi.begin());
     Gpu::copy(Gpu::hostToDevice, disk_cell_count.begin(), disk_cell_count.end(), d_disk_cell_count.begin());

     Real* d_freestream_velocity_ptr = d_freestream_velocity.data();
     Real* d_freestream_phi_ptr = d_freestream_phi.data();
     Real* d_disk_cell_count_ptr     = d_disk_cell_count.data();


     for ( MFIter mfi(cons_in,TilingIfNotGPU()); mfi.isValid(); ++mfi) {

        auto SMark_array    = mf_SMark.array(mfi);
        auto u_vel          = U_old.array(mfi);
        auto v_vel          = V_old.array(mfi);
        Box tbx = mfi.nodaltilebox(0);

        ParallelFor(tbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {

            if(SMark_array(i,j,k,0) != -1.0) {
                int turb_index = static_cast<int>(SMark_array(i,j,k,0));
                Real phi = std::atan2(v_vel(i,j,k),u_vel(i,j,k)); // Wind direction w.r.t the x-direction
                Gpu::Atomic::Add(&d_freestream_velocity_ptr[turb_index],std::pow(u_vel(i,j,k)*u_vel(i,j,k) + v_vel(i,j,k)*v_vel(i,j,k),0.5));
                Gpu::Atomic::Add(&d_disk_cell_count_ptr[turb_index],1.0);
                Gpu::Atomic::Add(&d_freestream_phi_ptr[turb_index],phi);
            }
        });
    }

    // Copy back to host
    Gpu::copy(Gpu::deviceToHost, d_freestream_velocity.begin(), d_freestream_velocity.end(), freestream_velocity.begin());
    Gpu::copy(Gpu::deviceToHost, d_freestream_phi.begin(), d_freestream_phi.end(), freestream_phi.begin());
    Gpu::copy(Gpu::deviceToHost, d_disk_cell_count.begin(), d_disk_cell_count.end(), disk_cell_count.begin());

    // Reduce the data on every processor
    amrex::ParallelAllReduce::Sum(freestream_velocity.data(),
                                  freestream_velocity.size(),
                                  amrex::ParallelContext::CommunicatorAll());

    amrex::ParallelAllReduce::Sum(freestream_phi.data(),
                                  freestream_phi.size(),
                                  amrex::ParallelContext::CommunicatorAll());


   amrex::ParallelAllReduce::Sum(disk_cell_count.data(),
                                 disk_cell_count.size(),
                                 amrex::ParallelContext::CommunicatorAll());

    get_turb_loc(xloc, yloc);
    /*if (ParallelDescriptor::IOProcessor()){
        for(int it=0; it<xloc.size(); it++){

            std::cout << "turbine index, freestream velocity is " << it << " " << freestream_velocity[it] << " " <<
                                                               disk_cell_count[it]  <<  " " <<
                                                                freestream_velocity[it]/(disk_cell_count[it] + 1e-10) << " " <<
                                                                freestream_phi[it]/(disk_cell_count[it] + 1e-10) << "\n";
        }
    }*/
}

void
SimpleAD::source_terms_cellcentered (const Geometry& geom,
                                     const Real& dt_advance,
                                     const MultiFab& cons_in,
                                     const MultiFab& mf_SMark,
                                     const MultiFab& mf_RMask,
                                     MultiFab& mf_vars_simpleAD)
{

    get_turb_loc(xloc, yloc);
    get_turb_zloc(zloc);
    get_turb_spec(rotor_rad, hub_height, thrust_coeff_standing,
                  wind_speed, thrust_coeff, power);
    get_wake_rotation_params(wake_rotation, tsr, C_P_prime);

    Gpu::DeviceVector<Real> d_xloc(xloc.size());
    Gpu::DeviceVector<Real> d_yloc(yloc.size());
    Gpu::DeviceVector<Real> d_zloc(zloc.size());
    Gpu::copy(Gpu::hostToDevice, xloc.begin(), xloc.end(), d_xloc.begin());
    Gpu::copy(Gpu::hostToDevice, yloc.begin(), yloc.end(), d_yloc.begin());
    Gpu::copy(Gpu::hostToDevice, zloc.begin(), zloc.end(), d_zloc.begin());

      auto dx = geom.CellSizeArray();
      auto ProbLoArr = geom.ProbLoArray();

  // Domain valid box
      const amrex::Box& domain = geom.Domain();
      int domlo_x = domain.smallEnd(0);
      int domhi_x = domain.bigEnd(0) + 1;
      int domlo_y = domain.smallEnd(1);
      int domhi_y = domain.bigEnd(1) + 1;
      int domlo_z = domain.smallEnd(2);
      int domhi_z = domain.bigEnd(2) + 1;

      // The order of variables are - Vabs dVabsdt, dudt, dvdt, dTKEdt
      mf_vars_simpleAD.setVal(0.0);

      long unsigned int nturbs = xloc.size();

     Real* d_xloc_ptr = d_xloc.data();
     Real* d_yloc_ptr = d_yloc.data();
     Real* d_zloc_ptr = d_zloc.data();
     const Real d_hub_height = hub_height;
     const Real d_rotor_rad = rotor_rad;
     const bool d_wake_rotation = wake_rotation;
     const Real d_tsr = tsr;
     const Real d_C_P_prime = C_P_prime;
     const Real eps = 1.0e-12;

     Gpu::DeviceVector<Real> d_freestream_velocity(nturbs);
     Gpu::DeviceVector<Real> d_disk_cell_count(nturbs);
     Gpu::copy(Gpu::hostToDevice, freestream_velocity.begin(), freestream_velocity.end(), d_freestream_velocity.begin());
     Gpu::copy(Gpu::hostToDevice, disk_cell_count.begin(), disk_cell_count.end(), d_disk_cell_count.begin());

     Real* d_freestream_velocity_ptr = d_freestream_velocity.data();
     Real* d_disk_cell_count_ptr     = d_disk_cell_count.data();

    amrex::Vector<amrex::Real> turb_disk_angles;
    get_turb_disk_angles(turb_disk_angles);
    if (turb_disk_angles.empty()) {
        get_turb_disk_angle(turb_disk_angle);
        turb_disk_angles.assign(nturbs, turb_disk_angle);
    }

    amrex::Vector<amrex::Real> nx_h(nturbs, 0.0);
    amrex::Vector<amrex::Real> ny_h(nturbs, 0.0);
    amrex::Vector<amrex::Real> cos_theta_h(nturbs, 0.0);
    amrex::Vector<amrex::Real> filtered_disk_velocity_nhat_h(nturbs, 0.0);
    if (filtered_disk_velocity_nhat.size() != nturbs) {
        filtered_disk_velocity_nhat.assign(nturbs, 0.0);
        filtered_disk_velocity_initialized = false;
    }

    const Real weight = (m_turb_mem_time > 0.0)
                          ? dt_advance / (m_turb_mem_time + dt_advance)
                          : 1.0;
    for (int it = 0; it < static_cast<int>(nturbs); ++it) {
        nx_h[it] = -std::cos(turb_disk_angles[it]);
        ny_h[it] = -std::sin(turb_disk_angles[it]);
        // Keep the projected disk area positive so turbines facing +x do not
        // inject momentum in the same direction as a negative-x inflow.
        cos_theta_h[it] = std::abs(std::cos(turb_disk_angles[it]));

        Real avg_vel = freestream_velocity[it] / (disk_cell_count[it] + 1e-10);
        Real phi = freestream_phi[it] / (disk_cell_count[it] + 1e-10);
        Real raw_disk_velocity_nhat = avg_vel * (std::cos(phi) * nx_h[it] + std::sin(phi) * ny_h[it]);

        if (!filtered_disk_velocity_initialized) {
            filtered_disk_velocity_nhat[it] = raw_disk_velocity_nhat;
        } else {
            filtered_disk_velocity_nhat[it] =
                (1.0 - weight) * filtered_disk_velocity_nhat[it] + weight * raw_disk_velocity_nhat;
        }

        filtered_disk_velocity_nhat_h[it] = filtered_disk_velocity_nhat[it];
    }
    filtered_disk_velocity_initialized = true;

	    Gpu::DeviceVector<Real> d_nx(nturbs);
	    Gpu::DeviceVector<Real> d_ny(nturbs);
	    Gpu::DeviceVector<Real> d_cos_theta(nturbs);
	    Gpu::DeviceVector<Real> d_filtered_disk_velocity_nhat(nturbs);
    Gpu::copy(Gpu::hostToDevice, nx_h.begin(), nx_h.end(), d_nx.begin());
    Gpu::copy(Gpu::hostToDevice, ny_h.begin(), ny_h.end(), d_ny.begin());
    Gpu::copy(Gpu::hostToDevice, cos_theta_h.begin(), cos_theta_h.end(), d_cos_theta.begin());
    Gpu::copy(Gpu::hostToDevice, filtered_disk_velocity_nhat_h.begin(), filtered_disk_velocity_nhat_h.end(),
              d_filtered_disk_velocity_nhat.begin());

	    Real* d_nx_ptr = d_nx.data();
	    Real* d_ny_ptr = d_ny.data();
	    Real* d_cos_theta_ptr = d_cos_theta.data();
	    Real* d_filtered_disk_velocity_nhat_ptr = d_filtered_disk_velocity_nhat.data();

	    WindFarmSpreadingType spreading_type;
	    Real spreading_nsigma;
	    get_force_spreading(spreading_type, spreading_nsigma);
	    const bool use_gaussian_spreading = (spreading_type == WindFarmSpreadingType::Gaussian);

	    Gpu::DeviceVector<Real> d_spread_weight_sum(nturbs, 0.0);
	    Real* d_spread_weight_sum_ptr = d_spread_weight_sum.data();

	    if (use_gaussian_spreading) {
	        for ( MFIter mfi(cons_in,TilingIfNotGPU()); mfi.isValid(); ++mfi) {
	            const Box& gbx      = mfi.growntilebox(1);
	            auto SMark_array    = mf_SMark.const_array(mfi);
	            auto RMask_array    = mf_RMask.const_array(mfi);

	            ParallelFor(gbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
	                int ii = amrex::min(amrex::max(i, domlo_x), domhi_x);
	                int jj = amrex::min(amrex::max(j, domlo_y), domhi_y);
	                int kk = amrex::min(amrex::max(k, domlo_z), domhi_z);
	                int it = static_cast<int>(SMark_array(ii,jj,kk,1));
	                if (it != -1) {
	                    Real nx_it = d_nx_ptr[it];
	                    Real ny_it = d_ny_ptr[it];
	                    Real sigma = std::abs(dx[0]*nx_it) + std::abs(dx[1]*ny_it);
	                    Real normal_dist = RMask_array(ii,jj,kk,1);
	                    if (sigma > eps && std::abs(normal_dist) <= spreading_nsigma*sigma) {
	                        Real kernel = std::exp(-0.5*normal_dist*normal_dist/(sigma*sigma)) /
	                                      (std::sqrt(2.0*PI)*sigma);
	                        Gpu::Atomic::Add(&d_spread_weight_sum_ptr[it], kernel*dx[0]*dx[1]*dx[2]);
	                    }
	                }
	            });
	        }
	        amrex::Vector<Real> spread_weight_sum(nturbs, 0.0);
	        Gpu::copy(Gpu::deviceToHost, d_spread_weight_sum.begin(), d_spread_weight_sum.end(),
	                  spread_weight_sum.begin());
	        amrex::ParallelAllReduce::Sum(spread_weight_sum.data(),
	                                      spread_weight_sum.size(),
	                                      amrex::ParallelContext::CommunicatorAll());
	        for (int it = 0; it < static_cast<int>(nturbs); ++it) {
	            if (disk_cell_count[it] > 0.0 && spread_weight_sum[it] <= eps) {
	                amrex::Abort("SimpleAD Gaussian force spreading produced zero normalization weight for turbine " +
	                             std::to_string(it));
	            }
	        }
	        Gpu::copy(Gpu::hostToDevice, spread_weight_sum.begin(), spread_weight_sum.end(),
	                  d_spread_weight_sum.begin());
	    }

	    Gpu::DeviceVector<Real> d_wind_speed(wind_speed.size());
    Gpu::DeviceVector<Real> d_thrust_coeff(thrust_coeff.size());

    // Copy data from host vectors to device vectors
    Gpu::copy(Gpu::hostToDevice, wind_speed.begin(), wind_speed.end(), d_wind_speed.begin());
    Gpu::copy(Gpu::hostToDevice, thrust_coeff.begin(), thrust_coeff.end(), d_thrust_coeff.begin());

    const Real* wind_speed_d     = d_wind_speed.dataPtr();
    const Real* thrust_coeff_d   = d_thrust_coeff.dataPtr();
    const int n_spec_table = d_wind_speed.size();

    for ( MFIter mfi(cons_in,TilingIfNotGPU()); mfi.isValid(); ++mfi) {

        const Box& gbx      = mfi.growntilebox(1);
        auto SMark_array    = mf_SMark.array(mfi);
        auto RMask_array    = mf_RMask.const_array(mfi);
        auto simpleAD_array = mf_vars_simpleAD.array(mfi);

        ParallelFor(gbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            int ii = amrex::min(amrex::max(i, domlo_x), domhi_x);
            int jj = amrex::min(amrex::max(j, domlo_y), domhi_y);
            int kk = amrex::min(amrex::max(k, domlo_z), domhi_z);


            Real source_x = 0.0;
            Real source_y = 0.0;
            Real source_z = 0.0;

            int it = static_cast<int>(SMark_array(ii,jj,kk,1));

              if(it != -1) {
                Real avg_vel  = d_freestream_velocity_ptr[it]/(d_disk_cell_count_ptr[it] + 1e-10);
                Real C_T = interpolate_1d(wind_speed_d, thrust_coeff_d, avg_vel, n_spec_table);
                Real a;
                if(C_T <= 1) {
                    a = 0.5 - 0.5*std::pow(1.0-C_T,0.5);
                }
                Real nx_it = d_nx_ptr[it];
                Real ny_it = d_ny_ptr[it];
                Real cos_theta_it = d_cos_theta_ptr[it];
                Real Uinfty_dot_nhat = d_filtered_disk_velocity_nhat_ptr[it];
	                    Real thrust_coeff_source = 0.0;
	                    if(C_T <= 1) {
	                        thrust_coeff_source = 2.0*std::pow(Uinfty_dot_nhat, 2.0)*a*(1.0-a);
	                    }
	                    else {
	                        thrust_coeff_source = 0.5*C_T*std::pow(Uinfty_dot_nhat, 2.0);
	                    }

	                    Real source_factor = 0.0;
	                    if (use_gaussian_spreading) {
	                        Real nx_it_tmp = d_nx_ptr[it];
	                        Real ny_it_tmp = d_ny_ptr[it];
	                        Real sigma = std::abs(dx[0]*nx_it_tmp) + std::abs(dx[1]*ny_it_tmp);
	                        Real normal_dist = RMask_array(ii,jj,kk,1);
	                        if (sigma > eps && d_spread_weight_sum_ptr[it] > eps) {
	                            Real kernel = std::exp(-0.5*normal_dist*normal_dist/(sigma*sigma)) /
	                                          (std::sqrt(2.0*PI)*sigma);
	                            source_factor = PI*d_rotor_rad*d_rotor_rad*kernel /
	                                            d_spread_weight_sum_ptr[it];
	                        }
	                    } else {
	                        source_factor = dx[1]*dx[2]*cos_theta_it/(dx[0]*dx[1]*dx[2]);
	                    }

	                    source_x = thrust_coeff_source*source_factor*nx_it;
	                    source_y = thrust_coeff_source*source_factor*ny_it;

	                    if (d_wake_rotation) {
	                        Real r = RMask_array(ii,jj,kk,0);
                        if (r > eps) {
                            Real xc = ProbLoArr[0] + (ii+0.5_rt)*dx[0];
                            Real yc = ProbLoArr[1] + (jj+0.5_rt)*dx[1];
                            Real z = ProbLoArr[2] + (kk+0.5_rt)*dx[2];
                            Real x0 = d_xloc_ptr[it];
                            Real y0 = d_yloc_ptr[it];
                            Real zhub = d_hub_height + d_zloc_ptr[it];
                            Real dxp = xc - x0;
                            Real dyp = yc - y0;
                            Real dzp = z - zhub;
                            Real a = dxp*nx_it + dyp*ny_it;
                            Real rx = dxp - a*nx_it;
                            Real ry = dyp - a*ny_it;
                            Real omega = d_tsr * Uinfty_dot_nhat / d_rotor_rad;
                            Real omega_r = omega * r;
                            if (std::abs(omega_r) > eps) {
	                                Real P = 0.5*d_C_P_prime*Uinfty_dot_nhat*Uinfty_dot_nhat*
	                                         (Uinfty_dot_nhat/(omega_r))*source_factor;
	                                Real t_x = ny_it*dzp/r;
                                Real t_y = -nx_it*dzp/r;
                                Real t_z = (nx_it*ry - ny_it*rx)/r;
                                source_x += P*t_x;
                                source_y += P*t_y;
                                source_z += P*t_z;
                            }
                        }
                    }
             }

            simpleAD_array(i,j,k,0) = source_x;
            simpleAD_array(i,j,k,1) = source_y;
            simpleAD_array(i,j,k,2) = source_z;
         });
    }
}
