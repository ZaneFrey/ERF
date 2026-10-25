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

TEST(WindDirectionControl, HysteresisRetainsStateBetweenThresholds)
{
    constexpr Real deg = PI / Real(180.0);
    EXPECT_TRUE(erf_wind_direction::update_correction_demand(
        false, Real(2.0) * deg, Real(2.0) * deg, Real(1.0) * deg));
    EXPECT_TRUE(erf_wind_direction::update_correction_demand(
        true, Real(1.5) * deg, Real(2.0) * deg, Real(1.0) * deg));
    EXPECT_FALSE(erf_wind_direction::update_correction_demand(
        true, Real(1.0) * deg, Real(2.0) * deg, Real(1.0) * deg));
    EXPECT_FALSE(erf_wind_direction::update_correction_demand(
        false, Real(1.5) * deg, Real(2.0) * deg, Real(1.0) * deg));
}

TEST(WindDirectionControl, GeostrophicControllerUsesSM14Sign)
{
    constexpr Real deg = PI / Real(180.0);
    const Real updated = erf_wind_direction::update_geostrophic_angle(
        Real(30.0) * deg, Real(10.0) * deg, Real(1.0 / 600.0), Real(60.0));
    EXPECT_NEAR(updated, Real(29.0) * deg, Real(1.0e-14));
}

TEST(WindDirectionControl, EventIsLatchedCappedAndCompleted)
{
    constexpr Real deg = PI / Real(180.0);
    erf_wind_direction::EventState event;
    erf_wind_direction::latch_event(event, Real(3.0) * deg);
    ASSERT_TRUE(event.active);
    EXPECT_NEAR(event.remaining_angle, Real(-3.0) * deg, Real(1.0e-14));

    // A later error cannot replace an already active SM14 event.
    erf_wind_direction::latch_event(event, Real(-20.0) * deg);
    EXPECT_NEAR(event.remaining_angle, Real(-3.0) * deg, Real(1.0e-14));

    const Real first_rate = erf_wind_direction::event_angle_rate(
        event, Real(1.0) * deg, Real(2.0));
    EXPECT_NEAR(first_rate, Real(-1.0) * deg, Real(1.0e-14));
    erf_wind_direction::advance_event(event, first_rate, Real(2.0));
    ASSERT_TRUE(event.active);
    EXPECT_NEAR(event.remaining_angle, Real(-1.0) * deg, Real(1.0e-14));

    const Real final_rate = erf_wind_direction::event_angle_rate(
        event, Real(1.0) * deg, Real(2.0));
    EXPECT_NEAR(final_rate, Real(-0.5) * deg, Real(1.0e-14));
    erf_wind_direction::advance_event(event, final_rate, Real(2.0));
    EXPECT_FALSE(event.active);
    EXPECT_EQ(event.remaining_angle, Real(0.0));
}

TEST(WindDirectionControl, InterpolatesPrimitiveComponentsOnStretchedLevels)
{
    const std::vector<Real> heights{Real(15.0), Real(45.0), Real(100.0)};
    amrex::Gpu::HostVector<Real> values{Real(2.0), Real(8.0), Real(19.0)};
    EXPECT_NEAR(erf_wind_direction::interpolate_profile(
                    heights, values, Real(30.0)),
                Real(5.0), Real(1.0e-14));
    EXPECT_EQ(erf_wind_direction::interpolate_profile(
                  heights, values, Real(100.0)),
              Real(19.0));
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
