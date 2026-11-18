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

enum GradPFaceComponent : int
{
    GradP = 0,
    NumGradPFaceMoments
};

// Six independent components of the symmetric total stress tensor.  The
// interval representation is cell-centered, matching the existing Tau*
// plotfile diagnostics rather than the native edge staggering of the
// off-diagonal solver fields.
enum TauMeanComponent : int
{
    Tau11 = 0,
    Tau12,
    Tau13,
    Tau22,
    Tau23,
    Tau33,
    NumTauMeanMoments
};

enum SGSEnergyFluxFaceComponent : int
{
    SGSEnergyFlux = 0,
    NumSGSEnergyFluxFaceMoments
};

enum TripleVelocityComponent : int
{
    UiUiU = 0,
    UiUiV,
    UiUiW,
    NumTripleVelocityMoments
};

enum StressStrainComponent : int
{
    TotalStressStrain = 0,
    NumStressStrainMoments
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
