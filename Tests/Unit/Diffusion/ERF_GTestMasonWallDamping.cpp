#include <ERF_MasonWallDamping.H>

#include <gtest/gtest.h>

#include <cmath>

using amrex::Real;

TEST(MasonWallDamping, MatchesMasonThomsonBlend)
{
    const Real grid_length = Real(3.0);
    const Real wall_length = Real(2.0);
    const Real exponent = Real(2.0);
    const Real expected = Real(1.0) /
        std::sqrt(Real(1.0) / (grid_length * grid_length) +
                  Real(1.0) / (wall_length * wall_length));

    EXPECT_DOUBLE_EQ(mason_wall_damping::effective_length(
                         grid_length, wall_length, exponent),
                     expected);
}

TEST(MasonWallDamping, ApproachesTheSmallerLimitingLength)
{
    const Real exponent = Real(2.0);
    const Real grid_limited = mason_wall_damping::effective_length(
        Real(2.0), Real(2.0e6), exponent);
    const Real wall_limited = mason_wall_damping::effective_length(
        Real(2.0e6), Real(2.0), exponent);

    EXPECT_NEAR(grid_limited, Real(2.0), Real(2.0e-12));
    EXPECT_NEAR(wall_limited, Real(2.0), Real(2.0e-12));
}

TEST(MasonWallDamping, IsMonotoneInWallDistance)
{
    const Real grid_length = Real(3.0);
    const Real exponent = Real(2.0);
    Real previous = Real(0.0);

    for (int i = 1; i <= 100; ++i) {
        const Real wall_length = Real(i) * Real(0.1);
        const Real effective = mason_wall_damping::effective_length(
            grid_length, wall_length, exponent);
        EXPECT_GT(effective, previous);
        EXPECT_LT(effective, grid_length);
        previous = effective;
    }
}

TEST(MasonWallDamping, NonpositiveLengthsProduceNoMixing)
{
    EXPECT_EQ(mason_wall_damping::effective_length(Real(0.0), Real(1.0), Real(2.0)),
              Real(0.0));
    EXPECT_EQ(mason_wall_damping::effective_length(Real(1.0), Real(0.0), Real(2.0)),
              Real(0.0));
}
