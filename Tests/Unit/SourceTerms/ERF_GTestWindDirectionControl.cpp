#include <gtest/gtest.h>

#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

#include "ERF_Constants.H"
#include "ERF_PlaneAverage.H"
#include "ERF_WindDirectionControl.H"

#include <cmath>
#include <vector>

using amrex::Real;

namespace
{
void
fill_linear_velocity_profiles (amrex::MultiFab& u, amrex::MultiFab& v)
{
    for (amrex::MFIter mfi(u); mfi.isValid(); ++mfi) {
        const auto array = u.array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE
                           (int i, int j, int k) noexcept {
            array(i,j,k) = Real(2.0) + Real(3.0) * static_cast<Real>(k);
        });
    }
    for (amrex::MFIter mfi(v); mfi.isValid(); ++mfi) {
        const auto array = v.array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE
                           (int i, int j, int k) noexcept {
            array(i,j,k) = Real(-1.0) + Real(4.0) * static_cast<Real>(k);
        });
    }
    amrex::Gpu::streamSynchronize();
}
}

TEST(WindDirectionControl, WrapsAnglesAcrossBranchCut)
{
    constexpr Real deg = PI / Real(180.0);
    EXPECT_NEAR(erf_wind_direction::wrap_angle(Real(181.0) * deg),
                Real(-179.0) * deg, Real(1.0e-14));
    EXPECT_NEAR(erf_wind_direction::wrap_angle(Real(-181.0) * deg),
                Real(179.0) * deg, Real(1.0e-14));
    EXPECT_NEAR(erf_wind_direction::wrap_angle(Real(360.0) * deg),
                Real(0.0), Real(1.0e-14));
}

TEST(WindDirectionControl, RotationRateIsContinuousAndProportional)
{
    constexpr Real deg = PI / Real(180.0);
    constexpr Real gain = Real(1.0e-4);

    EXPECT_NEAR(erf_wind_direction::continuous_rotation_rate(
                    Real(10.0) * deg, gain),
                Real(0.001) * deg, Real(1.0e-16));
    EXPECT_NEAR(erf_wind_direction::continuous_rotation_rate(
                    Real(2.0) * deg, gain),
                Real(0.0002) * deg, Real(1.0e-16));
    EXPECT_NEAR(erf_wind_direction::continuous_rotation_rate(
                    Real(-3.0) * deg, gain),
                Real(-0.0003) * deg, Real(1.0e-16));
    EXPECT_EQ(erf_wind_direction::continuous_rotation_rate(
                  Real(0.0), gain), Real(0.0));
}

TEST(WindDirectionControl, HasNoDeadband)
{
    constexpr Real deg = PI / Real(180.0);
    const Real rate = erf_wind_direction::continuous_rotation_rate(
        Real(1.0e-9) * deg, Real(1.0e-4));
    EXPECT_GT(rate, Real(0.0));
}

TEST(WindDirectionControl, PositiveErrorCommandsClockwiseRotation)
{
    constexpr Real deg = PI / Real(180.0);
    const Real omega3 = erf_wind_direction::continuous_rotation_rate(
        Real(5.0) * deg, Real(1.0e-4));
    const Real u = Real(10.0);
    const Real v = Real(0.0);
    const Real source_u = omega3 * v;
    const Real source_v = -omega3 * u;

    EXPECT_EQ(source_u, Real(0.0));
    EXPECT_LT(source_v, Real(0.0));
}

TEST(WindDirectionControl, GeostrophicForcingPreservesMagnitudeAndDirection)
{
    constexpr Real deg = PI / Real(180.0);
    const Real f = Real(1.0e-4);
    const Real magnitude = Real(12.0);
    const Real angle = Real(30.0) * deg;
    const auto forcing = erf_wind_direction::geostrophic_forcing(
        f, magnitude, angle);

    EXPECT_NEAR(forcing[0], -f * magnitude * std::sin(angle), Real(1.0e-16));
    EXPECT_NEAR(forcing[1],  f * magnitude * std::cos(angle), Real(1.0e-16));
    EXPECT_EQ(forcing[2], Real(0.0));
    EXPECT_NEAR(std::hypot(forcing[0], forcing[1]),
                f * magnitude, Real(1.0e-16));
}

TEST(WindDirectionControl, TargetAndGeostrophicHeightsInterpolateIndependently)
{
    const std::vector<Real> heights{Real(15.0), Real(45.0), Real(100.0)};
    amrex::Gpu::HostVector<Real> values{Real(2.0), Real(8.0), Real(19.0)};

    const Real target_value = erf_wind_direction::interpolate_profile(
        heights, values, Real(30.0));
    const Real geostrophic_value = erf_wind_direction::interpolate_profile(
        heights, values, Real(80.0));

    EXPECT_NEAR(target_value, Real(5.0), Real(1.0e-14));
    EXPECT_NEAR(geostrophic_value, Real(15.0), Real(1.0e-14));
    EXPECT_EQ(erf_wind_direction::interpolate_profile(
                  heights, values, Real(100.0)), Real(19.0));
}

TEST(WindDirectionControl, TotalVelocityRotationIsEnergyNeutral)
{
    const Real u = Real(7.0);
    const Real v = Real(-3.0);
    const Real density = Real(1.2);
    const Real omega3 = Real(2.5e-4);
    const Real source_u = density * omega3 * v;
    const Real source_v = -density * omega3 * u;

    EXPECT_NEAR(u * source_u + v * source_v, Real(0.0), Real(1.0e-15));
}

TEST(WindDirectionControl, FaceVelocityPlaneAveragesSupportLateralBoundaryTypes)
{
    const amrex::Box domain(amrex::IntVect(0, 0, 0), amrex::IntVect(3, 2, 2));
    const amrex::BoxArray cell_boxes(domain);
    const amrex::DistributionMapping dm(cell_boxes);
    const amrex::RealBox real_box({0.0, 0.0, 0.0}, {4.0, 3.0, 3.0});

    for (const int lateral_periodic : {0, 1}) {
        const amrex::Array<int, AMREX_SPACEDIM> periodic{
            lateral_periodic, lateral_periodic, 0};
        const amrex::Geometry geom(domain, &real_box, amrex::CoordSys::cartesian,
                                   periodic.data());
        amrex::MultiFab u(amrex::convert(cell_boxes, amrex::IntVect(1, 0, 0)),
                          dm, 1, 0);
        amrex::MultiFab v(amrex::convert(cell_boxes, amrex::IntVect(0, 1, 0)),
                          dm, 1, 0);
        fill_linear_velocity_profiles(u, v);

        PlaneAverage u_average(&u, geom, 2);
        PlaneAverage v_average(&v, geom, 2);
        u_average();
        v_average();
        amrex::Gpu::HostVector<Real> u_profile(3);
        amrex::Gpu::HostVector<Real> v_profile(3);
        u_average.line_average(0, u_profile);
        v_average.line_average(0, v_profile);
        for (int k = 0; k < 3; ++k) {
            EXPECT_NEAR(u_profile[k], Real(2.0) + Real(3.0) * k, Real(1.0e-14));
            EXPECT_NEAR(v_profile[k], Real(-1.0) + Real(4.0) * k, Real(1.0e-14));
        }
    }
}
