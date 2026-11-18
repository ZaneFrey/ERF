#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>
#include <AMReX.H>

#include <gtest/gtest.h>

#include "ERF_ClassicAD.H"
#include "ERF_ClassicADGrid.H"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace {

struct TestParticle
{
    amrex::ParticleReal xyz[3];
    amrex::ParticleReal pos (int direction) const noexcept
    {
        return xyz[direction];
    }
};

amrex::Geometry make_geometry ()
{
    const amrex::Box domain(amrex::IntVect(0,0,0), amrex::IntVect(3,3,3));
    const amrex::RealBox bounds(
        {amrex::Real(0), amrex::Real(0), amrex::Real(0)},
        {amrex::Real(40), amrex::Real(40), amrex::Real(40)});
    const std::array<int,3> periodic{{0,0,0}};
    return amrex::Geometry(domain, &bounds, 0, periodic.data());
}

[[noreturn]] void throwing_error_handler (const char* message)
{
    throw std::runtime_error(message ? message : "amrex::Error");
}

class ErrorHandlerScope
{
public:
    ErrorHandlerScope () : m_old_handler(amrex::system::error_handler)
    {
        amrex::SetErrorHandler(throwing_error_handler);
    }

    ~ErrorHandlerScope () { amrex::SetErrorHandler(m_old_handler); }

private:
    amrex::ErrorHandler m_old_handler;
};

ClassicAD make_control_model (
    int nturb, amrex::Real timescale,
    const amrex::Vector<ClassicADForcingInterval>& intervals)
{
    ClassicAD model;
    model.initialize_turbine_state(nturb);
    model.configure_forcing_schedule(intervals, timescale, 0.0);
    return model;
}

} // namespace

TEST(ClassicADVerticalGrid, FaceBelongsToCellAboveAndZetaRoundTrips)
{
    const auto geometry = make_geometry();
    ClassicADVerticalGrid grid;
    grid.configure(geometry, {0.0, 10.0, 20.0, 30.0, 40.0}, 20.0, true);

    EXPECT_EQ(grid.find_cell(19.999), 1);
    EXPECT_EQ(grid.find_cell(20.0), 2);
    EXPECT_EQ(grid.find_cell(40.0), 3);
    EXPECT_EQ(grid.hub_cell(), 2);
    EXPECT_DOUBLE_EQ(grid.hub_dz(), 10.0);
    EXPECT_DOUBLE_EQ(grid.physical_to_zeta(25.0), 25.0);
    EXPECT_DOUBLE_EQ(grid.zeta_to_physical(25.0), 25.0);
    EXPECT_FALSE(grid.cell_center_stencil_in_domain(1.0));
    EXPECT_TRUE(grid.cell_center_stencil_in_domain(20.0));
    EXPECT_FALSE(grid.cell_center_stencil_in_domain(39.0));
}

TEST(ClassicADVerticalGrid, UsesPrescribedLayerFractionForZeta)
{
    const auto geometry = make_geometry();
    ClassicADVerticalGrid grid;
    grid.configure(geometry, {0.0, 5.0, 15.0, 25.0, 40.0}, 20.0, true);

    EXPECT_EQ(grid.find_cell(15.0), 2);
    const amrex::Real zeta = grid.physical_to_zeta(20.0);
    EXPECT_DOUBLE_EQ(zeta, 25.0);
    EXPECT_DOUBLE_EQ(grid.zeta_to_physical(zeta), 20.0);
}

TEST(ClassicADVerticalGrid, InterpolatesConstantCellAndFaceProfiles)
{
    const auto geometry = make_geometry();
    ClassicADVerticalGrid grid;
    grid.configure(geometry, {0.0, 5.0, 15.0, 25.0, 40.0}, 20.0, true);

    const amrex::Box box(amrex::IntVect(-1,-1,-1), amrex::IntVect(4,4,5));
    amrex::FArrayBox values(box, 1);
    values.setVal<amrex::RunOn::Host>(9.25);
    TestParticle particle{{17.0, 23.0, grid.physical_to_zeta(17.0)}};

    for (const amrex::IntVect nodal :
         {amrex::IntVect(0,0,0), amrex::IntVect(0,0,1)}) {
        const amrex::Real result = classic_ad_interpolate_stretched(
            particle, amrex::ParticleReal(17.0), geometry.ProbLoArray(),
            geometry.InvCellSizeArray(), values.const_array(), nodal,
            grid.faces().data(), grid.centers().data());
        EXPECT_DOUBLE_EQ(result, 9.25);
    }
}

TEST(ClassicADVerticalGrid, InterpolatesLinearCellCenteredProfile)
{
    const auto geometry = make_geometry();
    ClassicADVerticalGrid grid;
    grid.configure(geometry, {0.0, 5.0, 15.0, 25.0, 40.0}, 20.0, true);

    const amrex::Box box(amrex::IntVect(-1,-1,-1), amrex::IntVect(4,4,4));
    amrex::FArrayBox values(box, 1);
    const auto array = values.array();
    const amrex::Real extended_centers[] = {-2.5, 2.5, 10.0, 20.0, 32.5, 47.5};
    for (int k = -1; k <= 4; ++k) {
        for (int j = -1; j <= 4; ++j) {
            for (int i = -1; i <= 4; ++i) {
                array(i,j,k) = amrex::Real(3.0) * extended_centers[k+1]
                             + amrex::Real(7.0);
            }
        }
    }

    TestParticle particle{{20.0, 20.0, grid.physical_to_zeta(17.0)}};
    const amrex::IntVect cell_centered(0,0,0);
    const amrex::Real result = classic_ad_interpolate_stretched(
        particle, amrex::ParticleReal(17.0), geometry.ProbLoArray(),
        geometry.InvCellSizeArray(), values.const_array(), cell_centered,
        grid.faces().data(), grid.centers().data());
    EXPECT_NEAR(result, 58.0, 1.0e-12);
}

TEST(ClassicADVerticalGrid, InterpolatesLinearFaceCenteredProfile)
{
    const auto geometry = make_geometry();
    ClassicADVerticalGrid grid;
    grid.configure(geometry, {0.0, 5.0, 15.0, 25.0, 40.0}, 20.0, true);

    const amrex::Box box(amrex::IntVect(-1,-1,-1), amrex::IntVect(4,4,5));
    amrex::FArrayBox values(box, 1);
    const auto array = values.array();
    const amrex::Real extended_faces[] = {-5.0, 0.0, 5.0, 15.0, 25.0, 40.0, 55.0};
    for (int k = -1; k <= 5; ++k) {
        for (int j = -1; j <= 4; ++j) {
            for (int i = -1; i <= 4; ++i) {
                array(i,j,k) = amrex::Real(2.0) * extended_faces[k+1]
                             - amrex::Real(4.0);
            }
        }
    }

    TestParticle particle{{20.0, 20.0, grid.physical_to_zeta(17.0)}};
    const amrex::IntVect z_face_centered(0,0,1);
    const amrex::Real result = classic_ad_interpolate_stretched(
        particle, amrex::ParticleReal(17.0), geometry.ProbLoArray(),
        geometry.InvCellSizeArray(), values.const_array(), z_face_centered,
        grid.faces().data(), grid.centers().data());
    EXPECT_NEAR(result, 30.0, 1.0e-12);
}

TEST(ClassicADForcingSchedule, InstantaneousEventsUseExactStepAverage)
{
    auto model = make_control_model(2, 0.0, {
        {0, 0.0, 10.0, 0.0, 1},
        {0, 20.0, 30.0, 0.5, 2},
        {1, 5.0, 12.0, 0.25, 3}});

    EXPECT_NEAR(model.advance_forcing_control(0, 0.0, 15.0, 1.0),
                10.0/15.0, 1.0e-14);
    EXPECT_DOUBLE_EQ(model.turbine_state()[0].forcing_response, 0.0);
    EXPECT_FALSE(model.turbine_state()[0].forcing_enabled);

    EXPECT_DOUBLE_EQ(model.advance_forcing_control(0, 15.0, 10.0, 1.0), 0.0);
    EXPECT_NEAR(model.advance_forcing_control(0, 25.0, 10.0, 1.0),
                0.25, 1.0e-14);
    EXPECT_DOUBLE_EQ(model.turbine_state()[0].forcing_response, 0.5);
    EXPECT_TRUE(model.turbine_state()[0].forcing_enabled);

    EXPECT_NEAR(model.advance_forcing_control(1, 0.0, 15.0, 1.0),
                12.75/15.0, 1.0e-14);
    EXPECT_DOUBLE_EQ(model.turbine_state()[1].forcing_response, 0.25);
}

TEST(ClassicADForcingSchedule, ExponentialDecayUsesAnalyticStateAndAverage)
{
    auto model = make_control_model(1, 30.0, {
        {0, 0.0, 10.0, 0.0, 1}});

    const amrex::Real factor =
        model.advance_forcing_control(0, 10.0, 30.0, 1.0);
    EXPECT_NEAR(model.turbine_state()[0].forcing_response,
                std::exp(-1.0), 1.0e-14);
    EXPECT_NEAR(factor, 1.0-std::exp(-1.0), 1.0e-14);
    EXPECT_FALSE(model.turbine_state()[0].forcing_enabled);
}

TEST(ClassicADForcingSchedule, EventAtStepEndUpdatesCommandNotAverage)
{
    auto model = make_control_model(1, 0.0, {
        {0, 0.0, 10.0, 0.0, 1}});

    EXPECT_DOUBLE_EQ(model.advance_forcing_control(0, 0.0, 10.0, 1.0), 1.0);
    EXPECT_DOUBLE_EQ(model.turbine_state()[0].forcing_response, 0.0);
    EXPECT_FALSE(model.turbine_state()[0].forcing_enabled);
    EXPECT_DOUBLE_EQ(model.advance_forcing_control(0, 10.0, 1.0, 1.0), 0.0);
}

TEST(ClassicADForcingSchedule, IntegratesMultipleEventsWithinOneStep)
{
    auto model = make_control_model(1, 0.0, {
        {0, 0.0, 2.0, 0.0, 1},
        {0, 2.0, 4.0, 0.5, 2}});

    EXPECT_DOUBLE_EQ(model.advance_forcing_control(0, 0.0, 6.0, 1.0), 0.5);
    EXPECT_DOUBLE_EQ(model.turbine_state()[0].forcing_response, 0.5);
    EXPECT_TRUE(model.turbine_state()[0].forcing_enabled);
}

TEST(ClassicADForcingSchedule, ReconstructsLegacyCheckpointControlState)
{
    auto model = make_control_model(1, 30.0, {
        {0, 0.0, 10.0, 0.0, 1}});
    model.restore_uninitialized_forcing_state(40.0);

    EXPECT_NEAR(model.turbine_state()[0].forcing_response,
                std::exp(-1.0), 1.0e-14);
    EXPECT_FALSE(model.turbine_state()[0].forcing_enabled);
}

TEST(ClassicADForcingSchedule, RejectsInvalidIntervals)
{
    ErrorHandlerScope error_handler;
    const amrex::Real nan = std::numeric_limits<amrex::Real>::quiet_NaN();

    EXPECT_THROW(make_control_model(1, -1.0, {}), std::runtime_error);
    EXPECT_THROW(make_control_model(1, 0.0, {{1, 0.0, 1.0, 1.0, 1}}),
                 std::runtime_error);
    EXPECT_THROW(make_control_model(1, 0.0, {{0, nan, 1.0, 1.0, 1}}),
                 std::runtime_error);
    EXPECT_THROW(make_control_model(1, 0.0, {{0, 2.0, 1.0, 1.0, 1}}),
                 std::runtime_error);
    EXPECT_THROW(make_control_model(1, 0.0, {{0, 0.0, 1.0, 1.1, 1}}),
                 std::runtime_error);

    ClassicAD before_start;
    before_start.initialize_turbine_state(1);
    EXPECT_THROW(before_start.configure_forcing_schedule(
                     amrex::Vector<ClassicADForcingInterval>{
                         {0, 4.0, 6.0, 0.0, 1}}, 0.0, 5.0),
                 std::runtime_error);
    EXPECT_THROW(make_control_model(1, 0.0, {
                     {0, 0.0, 5.0, 0.0, 10},
                     {0, 4.0, 6.0, 1.0, 11}}),
                 std::runtime_error);
}
