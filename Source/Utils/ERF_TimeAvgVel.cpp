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

void
Accumulate_Interval_GradP (
    double dt_d,
    const Array<MultiFab*, AMREX_SPACEDIM>& gradp_face_moments,
    const Array<const MultiFab*, AMREX_SPACEDIM>& gradp,
    const EBFArrayBoxFactory* eb_factory)
{
    const Real dt = static_cast<Real>(dt_d);
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        AMREX_ALWAYS_ASSERT(gradp_face_moments[dir] != nullptr);
        AMREX_ALWAYS_ASSERT(gradp[dir] != nullptr);
        MultiFab& face_sum = *gradp_face_moments[dir];
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(face_sum, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const Box& bx = mfi.tilebox();
            const auto source = gradp[dir]->const_array(mfi);
            const auto sum = face_sum.array(mfi);
            Array4<const Real> afrac;
            if (eb_factory != nullptr) {
                afrac = eb_factory->getAreaFrac()[dir]->const_array(mfi);
            }
            ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                if (afrac && afrac(i,j,k) <= Real(0.0)) {
                    sum(i,j,k,erf_interval_means::GradP) = Real(0.0);
                } else {
                    sum(i,j,k,erf_interval_means::GradP) += source(i,j,k) * dt;
                }
            });
        }
    }
}

void
Accumulate_Interval_Tau (
    double dt_d,
    MultiFab* tau_moments,
    const Vector<std::unique_ptr<MultiFab>>& tau)
{
    AMREX_ALWAYS_ASSERT(tau_moments != nullptr);
    AMREX_ALWAYS_ASSERT(tau.size() > TauType::tau33);
    AMREX_ALWAYS_ASSERT(tau[TauType::tau11] != nullptr);
    AMREX_ALWAYS_ASSERT(tau[TauType::tau12] != nullptr);
    AMREX_ALWAYS_ASSERT(tau[TauType::tau13] != nullptr);
    AMREX_ALWAYS_ASSERT(tau[TauType::tau22] != nullptr);
    AMREX_ALWAYS_ASSERT(tau[TauType::tau23] != nullptr);
    AMREX_ALWAYS_ASSERT(tau[TauType::tau33] != nullptr);

    const Real dt = static_cast<Real>(dt_d);
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(*tau_moments, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        const auto tau11 = tau[TauType::tau11]->const_array(mfi);
        const auto tau12 = tau[TauType::tau12]->const_array(mfi);
        const auto tau13 = tau[TauType::tau13]->const_array(mfi);
        const auto tau22 = tau[TauType::tau22]->const_array(mfi);
        const auto tau23 = tau[TauType::tau23]->const_array(mfi);
        const auto tau33 = tau[TauType::tau33]->const_array(mfi);
        const auto sum = tau_moments->array(mfi);

        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            const Real tau12_cc = Real(0.25) *
                (tau12(i,j,k) + tau12(i,j+1,k) +
                 tau12(i+1,j+1,k) + tau12(i+1,j,k));
            const Real tau13_cc = Real(0.25) *
                (tau13(i,j,k) + tau13(i,j,k+1) +
                 tau13(i+1,j,k+1) + tau13(i+1,j,k));
            const Real tau23_cc = Real(0.25) *
                (tau23(i,j,k) + tau23(i,j,k+1) +
                 tau23(i,j+1,k+1) + tau23(i,j+1,k));

            sum(i,j,k,erf_interval_means::Tau11) += tau11(i,j,k) * dt;
            sum(i,j,k,erf_interval_means::Tau12) += tau12_cc * dt;
            sum(i,j,k,erf_interval_means::Tau13) += tau13_cc * dt;
            sum(i,j,k,erf_interval_means::Tau22) += tau22(i,j,k) * dt;
            sum(i,j,k,erf_interval_means::Tau23) += tau23_cc * dt;
            sum(i,j,k,erf_interval_means::Tau33) += tau33(i,j,k) * dt;
        });
    }
}

void
Accumulate_Interval_SGSEnergyFlux (
    double dt_d,
    const Array<MultiFab*, AMREX_SPACEDIM>& flux_moments,
    const Vector<std::unique_ptr<MultiFab>>& tau,
    const Array<const MultiFab*, AMREX_SPACEDIM>& velocity,
    const EBFArrayBoxFactory* eb_factory)
{
    AMREX_ALWAYS_ASSERT(tau.size() > TauType::tau32);
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        AMREX_ALWAYS_ASSERT(flux_moments[dir] != nullptr);
        AMREX_ALWAYS_ASSERT(velocity[dir] != nullptr);
    }
    AMREX_ALWAYS_ASSERT(tau[TauType::tau11] != nullptr);
    AMREX_ALWAYS_ASSERT(tau[TauType::tau12] != nullptr);
    AMREX_ALWAYS_ASSERT(tau[TauType::tau13] != nullptr);
    AMREX_ALWAYS_ASSERT(tau[TauType::tau22] != nullptr);
    AMREX_ALWAYS_ASSERT(tau[TauType::tau23] != nullptr);
    AMREX_ALWAYS_ASSERT(tau[TauType::tau33] != nullptr);

    const Real dt = static_cast<Real>(dt_d);
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        MultiFab& flux_sum = *flux_moments[dir];
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(flux_sum, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const Box& bx = mfi.tilebox();
            const auto u = velocity[0]->const_array(mfi);
            const auto v = velocity[1]->const_array(mfi);
            const auto w = velocity[2]->const_array(mfi);
            const auto tau11 = tau[TauType::tau11]->const_array(mfi);
            const auto tau12 = tau[TauType::tau12]->const_array(mfi);
            const auto tau13 = tau[TauType::tau13]->const_array(mfi);
            const auto tau21 = tau[TauType::tau21]
                ? tau[TauType::tau21]->const_array(mfi) : tau12;
            const auto tau22 = tau[TauType::tau22]->const_array(mfi);
            const auto tau23 = tau[TauType::tau23]->const_array(mfi);
            const auto tau31 = tau[TauType::tau31]
                ? tau[TauType::tau31]->const_array(mfi) : tau13;
            const auto tau32 = tau[TauType::tau32]
                ? tau[TauType::tau32]->const_array(mfi) : tau23;
            const auto tau33 = tau[TauType::tau33]->const_array(mfi);
            const auto sum = flux_sum.array(mfi);
            Array4<const Real> afrac;
            if (eb_factory != nullptr) {
                afrac = eb_factory->getAreaFrac()[dir]->const_array(mfi);
            }

            if (dir == 0) {
                ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                    if (afrac && afrac(i,j,k) <= Real(0.0)) {
                        sum(i,j,k) = Real(0.0);
                        return;
                    }
                    const Real u_tau11 = u(i,j,k) * Real(0.5) *
                        (tau11(i-1,j,k) + tau11(i,j,k));
                    const Real v_tau21 = Real(0.25) *
                        ((v(i-1,j,k) + v(i,j,k)) * tau21(i,j,k) +
                         (v(i-1,j+1,k) + v(i,j+1,k)) * tau21(i,j+1,k));
                    const Real w_tau31 = Real(0.25) *
                        ((w(i-1,j,k) + w(i,j,k)) * tau31(i,j,k) +
                         (w(i-1,j,k+1) + w(i,j,k+1)) * tau31(i,j,k+1));
                    sum(i,j,k) += (u_tau11 + v_tau21 + w_tau31) * dt;
                });
            } else if (dir == 1) {
                ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                    if (afrac && afrac(i,j,k) <= Real(0.0)) {
                        sum(i,j,k) = Real(0.0);
                        return;
                    }
                    const Real u_tau12 = Real(0.25) *
                        ((u(i,j-1,k) + u(i,j,k)) * tau12(i,j,k) +
                         (u(i+1,j-1,k) + u(i+1,j,k)) * tau12(i+1,j,k));
                    const Real v_tau22 = v(i,j,k) * Real(0.5) *
                        (tau22(i,j-1,k) + tau22(i,j,k));
                    const Real w_tau32 = Real(0.25) *
                        ((w(i,j-1,k) + w(i,j,k)) * tau32(i,j,k) +
                         (w(i,j-1,k+1) + w(i,j,k+1)) * tau32(i,j,k+1));
                    sum(i,j,k) += (u_tau12 + v_tau22 + w_tau32) * dt;
                });
            } else {
                ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                    if (afrac && afrac(i,j,k) <= Real(0.0)) {
                        sum(i,j,k) = Real(0.0);
                        return;
                    }
                    const Real u_tau13 = Real(0.25) *
                        ((u(i,j,k-1) + u(i,j,k)) * tau13(i,j,k) +
                         (u(i+1,j,k-1) + u(i+1,j,k)) * tau13(i+1,j,k));
                    const Real v_tau23 = Real(0.25) *
                        ((v(i,j,k-1) + v(i,j,k)) * tau23(i,j,k) +
                         (v(i,j+1,k-1) + v(i,j+1,k)) * tau23(i,j+1,k));
                    const Real w_tau33 = w(i,j,k) * Real(0.5) *
                        (tau33(i,j,k-1) + tau33(i,j,k));
                    sum(i,j,k) += (u_tau13 + v_tau23 + w_tau33) * dt;
                });
            }
        }
    }
}

void
Accumulate_Interval_TripleVelocity (
    double dt_d,
    MultiFab* triple_velocity_moments,
    const MultiFab& xvel,
    const MultiFab& yvel,
    const MultiFab& zvel)
{
    AMREX_ALWAYS_ASSERT(triple_velocity_moments != nullptr);
    AMREX_ALWAYS_ASSERT(
        triple_velocity_moments->nComp() ==
        erf_interval_means::NumTripleVelocityMoments);
    const Real dt = static_cast<Real>(dt_d);

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(*triple_velocity_moments, TilingIfNotGPU());
         mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        const auto u = xvel.const_array(mfi);
        const auto v = yvel.const_array(mfi);
        const auto w = zvel.const_array(mfi);
        const auto sum = triple_velocity_moments->array(mfi);
        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            const Real u_cc = Real(0.5) * (u(i,j,k) + u(i+1,j,k));
            const Real v_cc = Real(0.5) * (v(i,j,k) + v(i,j+1,k));
            const Real w_cc = Real(0.5) * (w(i,j,k) + w(i,j,k+1));
            const Real speed_sq =
                u_cc*u_cc + v_cc*v_cc + w_cc*w_cc;
            sum(i,j,k,erf_interval_means::UiUiU) += speed_sq * u_cc * dt;
            sum(i,j,k,erf_interval_means::UiUiV) += speed_sq * v_cc * dt;
            sum(i,j,k,erf_interval_means::UiUiW) += speed_sq * w_cc * dt;
        });
    }
}

void
Accumulate_Interval_StressStrain (
    double dt_d,
    MultiFab* stress_strain_moments,
    const MultiFab* stress_strain_instantaneous)
{
    AMREX_ALWAYS_ASSERT(stress_strain_moments != nullptr);
    AMREX_ALWAYS_ASSERT(stress_strain_instantaneous != nullptr);
    AMREX_ALWAYS_ASSERT(
        stress_strain_moments->nComp() ==
        erf_interval_means::NumStressStrainMoments);
    MultiFab::Saxpy(*stress_strain_moments, static_cast<Real>(dt_d),
                    *stress_strain_instantaneous, 0,
                    erf_interval_means::TotalStressStrain, 1, 0);
}
