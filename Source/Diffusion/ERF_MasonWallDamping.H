#ifndef ERF_MASON_WALL_DAMPING_H_
#define ERF_MASON_WALL_DAMPING_H_

#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

#include <cmath>

namespace mason_wall_damping {

/**
 * Blend an interior Smagorinsky length with the rough-wall mixing length.
 *
 * Mason & Thomson (1992), Eq. (9):
 *
 *     l_eff^(-n) = l_grid^(-n) + [kappa (z + z0)]^(-n).
 *
 * This ratio form avoids raising dimensional quantities to negative powers.
 */
AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
amrex::Real
effective_length (amrex::Real grid_length,
                  amrex::Real wall_length,
                  amrex::Real exponent) noexcept
{
    if (grid_length <= amrex::Real(0.0) || wall_length <= amrex::Real(0.0)) {
        return amrex::Real(0.0);
    }

    const amrex::Real ratio = grid_length / wall_length;
    return grid_length * std::pow(amrex::Real(1.0) + std::pow(ratio, exponent),
                                  -amrex::Real(1.0) / exponent);
}

} // namespace mason_wall_damping

#endif
