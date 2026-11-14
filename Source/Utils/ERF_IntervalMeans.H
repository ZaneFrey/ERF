#ifndef ERF_INTERVAL_MEANS_H_
#define ERF_INTERVAL_MEANS_H_

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

} // namespace erf_interval_means

#endif
