#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>

#include <gtest/gtest.h>

#include "ERF_ClassicADGrid.H"

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
    values.setVal(9.25);
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
