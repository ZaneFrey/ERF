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

TEST(WindDirectionControl, UsesSingleInclusiveEventThreshold)
{
    constexpr Real deg = PI / Real(180.0);
    EXPECT_FALSE(erf_wind_direction::event_requested(
        Real(1.999) * deg, Real(2.0) * deg));
    EXPECT_TRUE(erf_wind_direction::event_requested(
        Real(2.0) * deg, Real(2.0) * deg));
    EXPECT_TRUE(erf_wind_direction::event_requested(
        Real(-2.0) * deg, Real(2.0) * deg));
}

TEST(WindDirectionControl, CoupledEpisodeUsesSM14SignsAndGain)
{
    constexpr Real deg = PI / Real(180.0);
    erf_wind_direction::EpisodeState episode;
    episode.phase = erf_wind_direction::Phase::Idle;
    erf_wind_direction::latch_episode(
        episode, Real(10.0) * deg, Real(0.4), Real(120.0));

    EXPECT_EQ(episode.phase, erf_wind_direction::Phase::Rotating);
    EXPECT_NEAR(episode.latched_error, Real(10.0) * deg, Real(1.0e-14));
    EXPECT_NEAR(episode.remaining_flow_angle, Real(-10.0) * deg, Real(1.0e-14));
    EXPECT_NEAR(episode.remaining_geo_angle, Real(-4.0) * deg, Real(1.0e-14));
    EXPECT_EQ(episode.event_start_time, Real(120.0));
}

TEST(WindDirectionControl, EpisodeIsImmutableCappedAndSettles)
{
    constexpr Real deg = PI / Real(180.0);
    erf_wind_direction::EpisodeState episode;
    episode.phase = erf_wind_direction::Phase::Idle;
    erf_wind_direction::latch_episode(
        episode, Real(3.0) * deg, Real(1.0), Real(100.0));
    ASSERT_EQ(episode.phase, erf_wind_direction::Phase::Rotating);
    EXPECT_NEAR(episode.remaining_flow_angle, Real(-3.0) * deg, Real(1.0e-14));

    // A later measurement cannot replace an already active episode.
    erf_wind_direction::latch_episode(
        episode, Real(-20.0) * deg, Real(1.0), Real(110.0));
    EXPECT_NEAR(episode.remaining_flow_angle, Real(-3.0) * deg, Real(1.0e-14));

    const Real first_rate = erf_wind_direction::episode_flow_angle_rate(
        episode, Real(1.0) * deg, Real(2.0));
    EXPECT_NEAR(first_rate, Real(-1.0) * deg, Real(1.0e-14));
    EXPECT_GT(-first_rate, Real(0.0)); // positive omega_3 rotates clockwise
    const Real first_flow = first_rate * Real(2.0);
    const Real first_geo = erf_wind_direction::episode_geo_increment(
        episode, first_flow);
    erf_wind_direction::advance_episode(
        episode, first_flow, first_geo, Real(102.0), Real(600.0));
    ASSERT_EQ(episode.phase, erf_wind_direction::Phase::Rotating);
    EXPECT_NEAR(episode.remaining_flow_angle, Real(-1.0) * deg, Real(1.0e-14));
    EXPECT_NEAR(episode.remaining_geo_angle, Real(-1.0) * deg, Real(1.0e-14));

    const Real final_rate = erf_wind_direction::episode_flow_angle_rate(
        episode, Real(1.0) * deg, Real(2.0));
    EXPECT_NEAR(final_rate, Real(-0.5) * deg, Real(1.0e-14));
    EXPECT_NEAR(first_rate * Real(2.0) + final_rate * Real(2.0),
                Real(-3.0) * deg, Real(1.0e-14));
    const Real final_flow = final_rate * Real(2.0);
    const Real final_geo = erf_wind_direction::episode_geo_increment(
        episode, final_flow);
    erf_wind_direction::advance_episode(
        episode, final_flow, final_geo, Real(104.0), Real(600.0));
    EXPECT_EQ(episode.phase, erf_wind_direction::Phase::Settling);
    EXPECT_EQ(episode.remaining_flow_angle, Real(0.0));
    EXPECT_EQ(episode.remaining_geo_angle, Real(0.0));
    EXPECT_EQ(episode.settling_until, Real(704.0));
}

TEST(WindDirectionControl, ZeroGainKeepsGeostrophicDirectionFixed)
{
    constexpr Real deg = PI / Real(180.0);
    erf_wind_direction::EpisodeState episode;
    episode.phase = erf_wind_direction::Phase::Idle;
    erf_wind_direction::latch_episode(
        episode, Real(-5.0) * deg, Real(0.0), Real(0.0));
    const Real flow_rate = erf_wind_direction::episode_flow_angle_rate(
        episode, Real(1.0) * deg, Real(1.0));
    EXPECT_GT(flow_rate, Real(0.0));
    EXPECT_EQ(erf_wind_direction::episode_geo_increment(
                  episode, flow_rate), Real(0.0));
}

TEST(WindDirectionControl, UnitGainKeepsFlowAndGeostrophicCommandsSynchronized)
{
    constexpr Real deg = PI / Real(180.0);
    erf_wind_direction::EpisodeState episode;
    episode.phase = erf_wind_direction::Phase::Idle;
    erf_wind_direction::latch_episode(
        episode, Real(5.0) * deg, Real(1.0), Real(0.0));

    Real cumulative_flow = Real(0.0);
    Real cumulative_geo = Real(0.0);
    for (int step = 0; step < 3; ++step) {
        const Real rate = erf_wind_direction::episode_flow_angle_rate(
            episode, Real(2.0) * deg, Real(1.0));
        const Real flow_increment = rate;
        const Real geo_increment = erf_wind_direction::episode_geo_increment(
            episode, flow_increment);
        EXPECT_NEAR(flow_increment, geo_increment, Real(1.0e-14));
        cumulative_flow += flow_increment;
        cumulative_geo += geo_increment;
        erf_wind_direction::advance_episode(
            episode, flow_increment, geo_increment,
            static_cast<Real>(step + 1), Real(600.0));
    }

    EXPECT_EQ(episode.phase, erf_wind_direction::Phase::Settling);
    EXPECT_NEAR(cumulative_flow, Real(-5.0) * deg, Real(1.0e-14));
    EXPECT_NEAR(cumulative_geo, cumulative_flow, Real(1.0e-14));

    // Measurements cannot relatch a command during the settling hold.
    erf_wind_direction::latch_episode(
        episode, Real(-20.0) * deg, Real(1.0), Real(10.0));
    EXPECT_EQ(episode.phase, erf_wind_direction::Phase::Settling);
    EXPECT_EQ(episode.remaining_flow_angle, Real(0.0));
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
