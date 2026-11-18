#include <ERF_Utils.H>
#include <ERF_IndexDefines.H>
#include <ERF_IntervalMeans.H>

using namespace amrex;

/*
 * Accumulate time averaged velocity fields
 */
void
Time_Avg_Vel_atCC (double dt_d,
                   double& t_avg_cnt,
                   MultiFab* vel_t_avg,
                   MultiFab& xvel,
                   MultiFab& yvel,
                   MultiFab& zvel)
{
    // Augment the counter
    t_avg_cnt += dt_d;

    Real dt = static_cast<Real>(dt_d);

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(*(vel_t_avg),TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        // CC tilebox
        Box tbx = mfi.tilebox();

        // Velocity on faces
        const Array4<Real>& velx = xvel.array(mfi);
        const Array4<Real>& vely = yvel.array(mfi);
        const Array4<Real>& velz = zvel.array(mfi);

        // Time average at CC
        Array4<Real> vel_t_avg_arr = vel_t_avg->array(mfi);

        ParallelFor(tbx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            Real u_cc = myhalf * ( velx(i,j,k) + velx(i+1,j  ,k  ) );
            Real v_cc = myhalf * ( vely(i,j,k) + vely(i  ,j+1,k  ) );
            Real w_cc = myhalf * ( velz(i,j,k) + velz(i  ,j  ,k+1) );
            Real umag_cc = std::sqrt(u_cc*u_cc + v_cc*v_cc + w_cc*w_cc);
            vel_t_avg_arr(i,j,k,0) += u_cc * dt;
            vel_t_avg_arr(i,j,k,1) += v_cc * dt;
            vel_t_avg_arr(i,j,k,2) += w_cc * dt;
            vel_t_avg_arr(i,j,k,3) += umag_cc * dt;
        });
    }
}

void
Accumulate_Interval_Means (double dt_d,
                           double& t_mean_cnt,
                           MultiFab* interval_means,
                           MultiFab& xvel,
                           MultiFab& yvel,
                           MultiFab& zvel,
                           MultiFab& cons)
{
    AMREX_ALWAYS_ASSERT(interval_means != nullptr);

    t_mean_cnt += dt_d;
    const Real dt = static_cast<Real>(dt_d);

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(*interval_means, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.tilebox();
        const Array4<const Real>& u = xvel.const_array(mfi);
        const Array4<const Real>& v = yvel.const_array(mfi);
        const Array4<const Real>& w = zvel.const_array(mfi);
        const Array4<const Real>& state = cons.const_array(mfi);
        const Array4<Real>& mean = interval_means->array(mfi);

        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            const Real u_cc = Real(0.5) * (u(i,j,k) + u(i+1,j,k));
            const Real v_cc = Real(0.5) * (v(i,j,k) + v(i,j+1,k));
            const Real w_cc = Real(0.5) * (w(i,j,k) + w(i,j,k+1));
            const Real theta = state(i,j,k,RhoTheta_comp) / state(i,j,k,Rho_comp);

            mean(i,j,k,erf_interval_means::U) += u_cc * dt;
            mean(i,j,k,erf_interval_means::V) += v_cc * dt;
            mean(i,j,k,erf_interval_means::W) += w_cc * dt;
            mean(i,j,k,erf_interval_means::Theta) += theta * dt;
            mean(i,j,k,erf_interval_means::UU) += u_cc * u_cc * dt;
            mean(i,j,k,erf_interval_means::VV) += v_cc * v_cc * dt;
            mean(i,j,k,erf_interval_means::WW) += w_cc * w_cc * dt;
            mean(i,j,k,erf_interval_means::UW) += u_cc * w_cc * dt;
            mean(i,j,k,erf_interval_means::VW) += v_cc * w_cc * dt;
            mean(i,j,k,erf_interval_means::WTheta) += w_cc * theta * dt;
            mean(i,j,k,erf_interval_means::UV) += u_cc * v_cc * dt;
        });
    }
}

void
Accumulate_Interval_PertPres (
    double dt_d,
    MultiFab* pert_pres_moment,
    Array<MultiFab*, AMREX_SPACEDIM> pert_pres_face_moments,
    const MultiFab& pert_pres,
    const Array<const MultiFab*, AMREX_SPACEDIM>& velocity,
    const Geometry& geom,
    const EBFArrayBoxFactory* eb_factory)
{
    AMREX_ALWAYS_ASSERT(pert_pres_moment != nullptr);
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        AMREX_ALWAYS_ASSERT(pert_pres_face_moments[dir] != nullptr);
        AMREX_ALWAYS_ASSERT(velocity[dir] != nullptr);
    }

    const Real dt = static_cast<Real>(dt_d);
    const Box domain = geom.Domain();
    const auto domlo = lbound(domain);
    const auto domhi = ubound(domain);
    const auto is_per = geom.isPeriodicArray();

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(*pert_pres_moment, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        const auto p = pert_pres.const_array(mfi);
        const auto sum = pert_pres_moment->array(mfi);
        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            sum(i,j,k,erf_interval_means::PertPres) += p(i,j,k) * dt;
        });
    }

    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        MultiFab& face_sum = *pert_pres_face_moments[dir];
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(face_sum, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const Box& fbx = mfi.tilebox();
            const auto p = pert_pres.const_array(mfi);
            const auto vel = velocity[dir]->const_array(mfi);
            const auto sum = face_sum.array(mfi);
            Array4<const Real> afrac;
            if (eb_factory != nullptr) {
                afrac = eb_factory->getAreaFrac()[dir]->const_array(mfi);
            }

            ParallelFor(fbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                if (afrac && afrac(i,j,k) <= Real(0.0)) {
                    sum(i,j,k,erf_interval_means::FaceVelocity) = Real(0.0);
                    sum(i,j,k,erf_interval_means::PertPresVelocity) = Real(0.0);
                    return;
                }

                int ilo = i, jlo = j, klo = k;
                if (dir == 0) --ilo;
                if (dir == 1) --jlo;
                if (dir == 2) --klo;

                const int face_index = dir == 0 ? i : (dir == 1 ? j : k);
                const int domain_lo = dir == 0 ? domlo.x : (dir == 1 ? domlo.y : domlo.z);
                const int domain_hi = dir == 0 ? domhi.x : (dir == 1 ? domhi.y : domhi.z);
                Real pface;
                if (!is_per[dir] && face_index == domain_lo) {
                    pface = p(i,j,k);
                } else if (!is_per[dir] && face_index == domain_hi + 1) {
                    pface = p(ilo,jlo,klo);
                } else {
                    pface = Real(0.5) * (p(ilo,jlo,klo) + p(i,j,k));
                }

                const Real uface = vel(i,j,k);
                sum(i,j,k,erf_interval_means::FaceVelocity) += uface * dt;
                sum(i,j,k,erf_interval_means::PertPresVelocity) += pface * uface * dt;
            });
        }
    }
}
