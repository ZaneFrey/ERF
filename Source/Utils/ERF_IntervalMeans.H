#ifndef ERF_INTERVAL_MEANS_H_
#define ERF_INTERVAL_MEANS_H_

#include <AMReX_REAL.H>

namespace erf_interval_means {

enum MomentComponent : int
{
    U = 0,
    V,
    W,
    Theta,
    UU,
    VV,
    WW,
    UW,
    VW,
    WTheta,
    UV,
    NumIntervalMoments
};

inline constexpr int LegacyNumIntervalMoments = 10;

enum PertPresCellComponent : int
{
    PertPres = 0,
    NumPertPresCellMoments
};

enum PertPresFaceComponent : int
{
    FaceVelocity = 0,
    PertPresVelocity,
    NumPertPresFaceMoments
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real covariance_from_sums (amrex::Real product_sum,
                                  amrex::Real first_sum,
                                  amrex::Real second_sum,
                                  amrex::Real inv_normalizer) noexcept
{
    return product_sum * inv_normalizer
        - (first_sum * inv_normalizer) * (second_sum * inv_normalizer);
}

} // namespace erf_interval_means

#endif
