#include <array>
#include <sstream>
#include <string>

#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

#include <gtest/gtest.h>

#include "ERF_IntervalMeansCheckpoint.H"
#include "ERF_IndexDefines.H"
#include "ERF_Utils.H"

namespace {

std::string
make_metadata (const amrex::Vector<amrex::BoxArray>& boxes,
               const amrex::Vector<double>& counts,
               const amrex::Vector<int>& reset_flags,
               int version = 2,
               int components = erf_interval_means::NumIntervalMoments)
{
    std::ostringstream output;
    output << "ERF interval means checkpoint v" << version << "\n";
    output << boxes.size() << " " << components << "\n";
    for (int lev = 0; lev < static_cast<int>(boxes.size()); ++lev) {
        output << lev << " " << counts[lev] << " " << reset_flags[lev] << "\n";
        boxes[lev].writeOn(output);
        output << "\n";
        output << boxes[lev].size();
        for (int ibox = 0; ibox < boxes[lev].size(); ++ibox) {
            output << " " << (ibox + lev);
        }
        output << "\n";
    }
    return output.str();
}

std::string
make_pert_pres_metadata (const amrex::Vector<amrex::BoxArray>& boxes,
                         const amrex::Vector<double>& counts)
{
    std::ostringstream output;
    output << "ERF interval pert_pres checkpoint v1\n";
    output << boxes.size() << " "
           << erf_interval_means::NumPertPresCellMoments << " "
           << erf_interval_means::NumPertPresFaceMoments << "\n";
    for (int lev = 0; lev < static_cast<int>(boxes.size()); ++lev) {
        output << lev << " " << counts[lev] << "\n";
        boxes[lev].writeOn(output);
        output << "\n";
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            amrex::convert(boxes[lev], amrex::IntVect::TheDimensionVector(dir)).writeOn(output);
            output << "\n";
        }
    }
    return output.str();
}

std::string
make_gradp_metadata (const amrex::Vector<amrex::BoxArray>& boxes,
                     const amrex::Vector<double>& counts)
{
    std::ostringstream output;
    output << "ERF interval gradp checkpoint v1\n";
    output << boxes.size() << " "
           << erf_interval_means::NumGradPFaceMoments << "\n";
    for (int lev = 0; lev < static_cast<int>(boxes.size()); ++lev) {
        output << lev << " " << counts[lev] << "\n";
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            amrex::convert(boxes[lev], amrex::IntVect::TheDimensionVector(dir)).writeOn(output);
            output << "\n";
        }
    }
    return output.str();
}

std::string
make_tau_metadata (const amrex::Vector<amrex::BoxArray>& boxes,
                   const amrex::Vector<double>& counts)
{
    std::ostringstream output;
    output << "ERF interval tau checkpoint v1\n";
    output << boxes.size() << " "
           << erf_interval_means::NumTauMeanMoments << "\n";
    for (int lev = 0; lev < static_cast<int>(boxes.size()); ++lev) {
        output << lev << " " << counts[lev] << "\n";
        boxes[lev].writeOn(output);
        output << "\n";
    }
    return output.str();
}

std::string
make_stress_strain_metadata (const amrex::Vector<amrex::BoxArray>& boxes,
                             const amrex::Vector<double>& counts)
{
    std::ostringstream output;
    output << "ERF interval stress strain checkpoint v1\n";
    output << boxes.size() << " "
           << erf_interval_means::NumStressStrainMoments << "\n";
    for (int lev = 0; lev < static_cast<int>(boxes.size()); ++lev) {
        output << lev << " " << counts[lev] << "\n";
        boxes[lev].writeOn(output);
        output << "\n";
    }
    return output.str();
}

std::string
make_sgs_energy_flux_metadata (const amrex::Vector<amrex::BoxArray>& boxes,
                               const amrex::Vector<double>& counts)
{
    std::ostringstream output;
    output << "ERF interval sgs energy flux checkpoint v1\n";
    output << boxes.size() << " "
           << erf_interval_means::NumSGSEnergyFluxFaceMoments << "\n";
    for (int lev = 0; lev < static_cast<int>(boxes.size()); ++lev) {
        output << lev << " " << counts[lev] << "\n";
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            amrex::convert(boxes[lev],
                           amrex::IntVect::TheDimensionVector(dir)).writeOn(output);
            output << "\n";
        }
    }
    return output.str();
}

std::string
make_triple_velocity_metadata (const amrex::Vector<amrex::BoxArray>& boxes,
                               const amrex::Vector<double>& counts)
{
    std::ostringstream output;
    output << "ERF interval triple velocity checkpoint v1\n";
    output << boxes.size() << " "
           << erf_interval_means::NumTripleVelocityMoments << "\n";
    for (int lev = 0; lev < static_cast<int>(boxes.size()); ++lev) {
        output << lev << " " << counts[lev] << "\n";
        boxes[lev].writeOn(output);
        output << "\n";
    }
    return output.str();
}

amrex::Vector<amrex::BoxArray>
make_boxes ()
{
    amrex::Vector<amrex::BoxArray> boxes;
    boxes.emplace_back(amrex::Box(amrex::IntVect(0, 0, 0), amrex::IntVect(3, 3, 1)));
    boxes.emplace_back(amrex::Box(amrex::IntVect(0, 0, 0), amrex::IntVect(7, 7, 3)));
    boxes[0].maxSize(2);
    boxes[1].maxSize(4);
    return boxes;
}

} // namespace

TEST(IntervalMeansRestart, ParsesCurrentVersionCountsBoxesAndProcessorMaps)
{
    const auto boxes = make_boxes();
    const std::string text = make_metadata(boxes, {3.5, 8.25}, {0, 1});
    erf_interval_means::Metadata metadata;
    std::string error;
    std::istringstream input(text);
    ASSERT_TRUE(erf_interval_means::parse_metadata(input, metadata, error)) << error;
    EXPECT_EQ(metadata.version, 2);
    EXPECT_EQ(metadata.levels, 2);
    EXPECT_EQ(metadata.components, erf_interval_means::NumIntervalMoments);
    EXPECT_DOUBLE_EQ(metadata.level[0].accumulation_count, 3.5);
    EXPECT_DOUBLE_EQ(metadata.level[1].accumulation_count, 8.25);
    EXPECT_EQ(metadata.level[0].processor_map[0], 0);
    EXPECT_EQ(metadata.level[1].processor_map[0], 1);
    EXPECT_EQ(erf_interval_means::global_reset_done(metadata), 1);
    EXPECT_TRUE(erf_interval_means::validate_metadata(
        metadata, 2, erf_interval_means::NumIntervalMoments, boxes).empty());
    EXPECT_FALSE(erf_interval_means::restart_resets_window(metadata));
}

TEST(IntervalMeansRestart, ValidatesOptionalPertPresStaggeredMetadata)
{
    const auto boxes = make_boxes();
    const amrex::Vector<double> counts{2.5, 2.5};
    std::istringstream input(make_pert_pres_metadata(boxes, counts));
    erf_interval_means::PertPresMetadata metadata;
    std::string error;
    ASSERT_TRUE(erf_interval_means::parse_pert_pres_metadata(input, metadata, error)) << error;
    EXPECT_TRUE(erf_interval_means::validate_pert_pres_metadata(
                    metadata, boxes, counts).empty());

    auto mismatched_counts = counts;
    mismatched_counts[1] = 3.0;
    EXPECT_FALSE(erf_interval_means::validate_pert_pres_metadata(
                     metadata, boxes, mismatched_counts).empty());

    metadata.face_components += 1;
    EXPECT_FALSE(erf_interval_means::validate_pert_pres_metadata(
                     metadata, boxes, counts).empty());
}

TEST(IntervalMeansRestart, ValidatesOptionalGradPStaggeredMetadata)
{
    const auto boxes = make_boxes();
    const amrex::Vector<double> counts{1.25, 1.25};
    std::istringstream input(make_gradp_metadata(boxes, counts));
    erf_interval_means::GradPMetadata metadata;
    std::string error;
    ASSERT_TRUE(erf_interval_means::parse_gradp_metadata(input, metadata, error)) << error;
    EXPECT_TRUE(erf_interval_means::validate_gradp_metadata(
                    metadata, boxes, counts).empty());

    metadata.face_components += 1;
    EXPECT_FALSE(erf_interval_means::validate_gradp_metadata(
                     metadata, boxes, counts).empty());
}

TEST(IntervalMeansRestart, ValidatesOptionalTauMetadata)
{
    const auto boxes = make_boxes();
    const amrex::Vector<double> counts{4.0, 4.0};
    std::istringstream input(make_tau_metadata(boxes, counts));
    erf_interval_means::TauMetadata metadata;
    std::string error;
    ASSERT_TRUE(erf_interval_means::parse_tau_metadata(input, metadata, error)) << error;
    EXPECT_TRUE(erf_interval_means::validate_tau_metadata(
                    metadata, boxes, counts).empty());

    auto mismatched_counts = counts;
    mismatched_counts[1] = 5.0;
    EXPECT_FALSE(erf_interval_means::validate_tau_metadata(
                     metadata, boxes, mismatched_counts).empty());

    metadata.components += 1;
    EXPECT_FALSE(erf_interval_means::validate_tau_metadata(
                     metadata, boxes, counts).empty());
}

TEST(IntervalMeansRestart, ValidatesOptionalStressStrainMetadata)
{
    const auto boxes = make_boxes();
    const amrex::Vector<double> counts{4.0, 4.0};
    std::istringstream input(make_stress_strain_metadata(boxes, counts));
    erf_interval_means::StressStrainMetadata metadata;
    std::string error;
    ASSERT_TRUE(erf_interval_means::parse_stress_strain_metadata(
        input, metadata, error)) << error;
    EXPECT_TRUE(erf_interval_means::validate_stress_strain_metadata(
                    metadata, boxes, counts).empty());

    auto mismatched_counts = counts;
    mismatched_counts[1] = 5.0;
    EXPECT_FALSE(erf_interval_means::validate_stress_strain_metadata(
                     metadata, boxes, mismatched_counts).empty());

    metadata.components += 1;
    EXPECT_FALSE(erf_interval_means::validate_stress_strain_metadata(
                     metadata, boxes, counts).empty());
}

TEST(IntervalMeansRestart, ValidatesOptionalSGSEnergyFluxMetadata)
{
    const auto boxes = make_boxes();
    const amrex::Vector<double> counts{4.5, 4.5};
    std::istringstream input(make_sgs_energy_flux_metadata(boxes, counts));
    erf_interval_means::SGSEnergyFluxMetadata metadata;
    std::string error;
    ASSERT_TRUE(erf_interval_means::parse_sgs_energy_flux_metadata(
        input, metadata, error)) << error;
    EXPECT_TRUE(erf_interval_means::validate_sgs_energy_flux_metadata(
                    metadata, boxes, counts).empty());

    auto mismatched_counts = counts;
    mismatched_counts[1] = 5.0;
    EXPECT_FALSE(erf_interval_means::validate_sgs_energy_flux_metadata(
                     metadata, boxes, mismatched_counts).empty());

    metadata.face_components += 1;
    EXPECT_FALSE(erf_interval_means::validate_sgs_energy_flux_metadata(
                     metadata, boxes, counts).empty());
}

TEST(IntervalMeansRestart, ValidatesOptionalTripleVelocityMetadata)
{
    const auto boxes = make_boxes();
    const amrex::Vector<double> counts{6.0, 6.0};
    std::istringstream input(make_triple_velocity_metadata(boxes, counts));
    erf_interval_means::TripleVelocityMetadata metadata;
    std::string error;
    ASSERT_TRUE(erf_interval_means::parse_triple_velocity_metadata(
        input, metadata, error)) << error;
    EXPECT_TRUE(erf_interval_means::validate_triple_velocity_metadata(
                    metadata, boxes, counts).empty());

    auto mismatched_counts = counts;
    mismatched_counts[1] = 7.0;
    EXPECT_FALSE(erf_interval_means::validate_triple_velocity_metadata(
                     metadata, boxes, mismatched_counts).empty());
    metadata.components += 1;
    EXPECT_FALSE(erf_interval_means::validate_triple_velocity_metadata(
                     metadata, boxes, counts).empty());
}

TEST(IntervalMeansRestart, BoxArrayMismatchIsRejectedButProcessorMapIsNotAConstraint)
{
    const auto boxes = make_boxes();
    erf_interval_means::Metadata metadata;
    std::string error;
    std::istringstream input(make_metadata(boxes, {1.0, 2.0}, {0, 0}));
    ASSERT_TRUE(erf_interval_means::parse_metadata(input, metadata, error)) << error;

    auto changed_boxes = boxes;
    changed_boxes[0] = amrex::BoxArray(
        amrex::Box(amrex::IntVect(0, 0, 0), amrex::IntVect(4, 3, 1)));
    const auto mismatch = erf_interval_means::validate_metadata(
        metadata, 2, erf_interval_means::NumIntervalMoments, changed_boxes);
    EXPECT_NE(mismatch.find("BoxArray"), std::string::npos);
    EXPECT_TRUE(erf_interval_means::validate_metadata(
        metadata, 2, erf_interval_means::NumIntervalMoments, boxes).empty());
}

TEST(IntervalMeansRestart, AcceptsV1MetadataAndRequiresAnEmptyMigratedWindow)
{
    const auto boxes = make_boxes();
    erf_interval_means::Metadata metadata;
    std::string error;
    std::istringstream input(make_metadata(
        boxes, {3.5, 8.25}, {0, 1}, 1,
        erf_interval_means::LegacyNumIntervalMoments));
    ASSERT_TRUE(erf_interval_means::parse_metadata(input, metadata, error)) << error;
    EXPECT_EQ(metadata.version, 1);
    EXPECT_EQ(metadata.components, erf_interval_means::LegacyNumIntervalMoments);
    EXPECT_TRUE(erf_interval_means::validate_metadata(
        metadata, 2, erf_interval_means::LegacyNumIntervalMoments, boxes).empty());
    EXPECT_TRUE(erf_interval_means::restart_resets_window(metadata));
    EXPECT_EQ(erf_interval_means::global_reset_done(metadata), 1);
}

TEST(IntervalMeansRestart, RejectsMalformedAndTruncatedHeaders)
{
    for (const std::string& text : {
             std::string("ERF interval means checkpoint v0\n1 10\n"),
             std::string("ERF interval means checkpoint v1\n1 11\n"),
             std::string("ERF interval means checkpoint v2\n1 10\n"),
             std::string("ERF interval means checkpoint v1\n1 10\n0 1.0 0\n"),
             std::string("ERF interval means checkpoint v1\n1 10\n0 -1.0 0\n")}) {
        erf_interval_means::Metadata metadata;
        std::string error;
        std::istringstream input(text);
        EXPECT_FALSE(erf_interval_means::parse_metadata(input, metadata, error));
        EXPECT_FALSE(error.empty());
    }
}

TEST(IntervalMeansAccumulation, TimeWeightsHorizontalVelocityCovariance)
{
    using namespace amrex;

    const Box cell_box(IntVect(0, 0, 0), IntVect(0, 0, 0));
    const BoxArray cells(cell_box);
    const DistributionMapping dm(cells);
    MultiFab xvel(convert(cells, IntVect(1, 0, 0)), dm, 1, 0);
    MultiFab yvel(convert(cells, IntVect(0, 1, 0)), dm, 1, 0);
    MultiFab zvel(convert(cells, IntVect(0, 0, 1)), dm, 1, 0);
    MultiFab cons(cells, dm, RhoTheta_comp + 1, 0);
    MultiFab moments(cells, dm, erf_interval_means::NumIntervalMoments, 0);

    cons.setVal(Real(0.0));
    cons.setVal(Real(2.0), Rho_comp, 1, 0);
    cons.setVal(Real(600.0), RhoTheta_comp, 1, 0);
    zvel.setVal(Real(0.0));
    moments.setVal(Real(0.0));

    double accumulated_time = 0.0;
    xvel.setVal(Real(2.0));
    yvel.setVal(Real(3.0));
    Accumulate_Interval_Means(
        1.0, accumulated_time, &moments, xvel, yvel, zvel, cons);

    xvel.setVal(Real(4.0));
    yvel.setVal(Real(1.0));
    Accumulate_Interval_Means(
        3.0, accumulated_time, &moments, xvel, yvel, zvel, cons);

    EXPECT_DOUBLE_EQ(accumulated_time, 4.0);
    const Real u_mean = moments.sum(erf_interval_means::U) / accumulated_time;
    const Real v_mean = moments.sum(erf_interval_means::V) / accumulated_time;
    const Real uv_mean = moments.sum(erf_interval_means::UV) / accumulated_time;
    EXPECT_DOUBLE_EQ(u_mean, Real(3.5));
    EXPECT_DOUBLE_EQ(v_mean, Real(1.5));
    EXPECT_DOUBLE_EQ(uv_mean, Real(4.5));
    EXPECT_DOUBLE_EQ(uv_mean - u_mean * v_mean, Real(-0.75));
}

TEST(IntervalMeansAccumulation, TimeWeightsCentralThirdVelocityMoments)
{
    using namespace amrex;

    const BoxArray cells(Box(IntVect(0,0,0), IntVect(0,0,0)));
    const DistributionMapping dm(cells);
    MultiFab xvel(convert(cells, IntVect(1,0,0)), dm, 1, 0);
    MultiFab yvel(convert(cells, IntVect(0,1,0)), dm, 1, 0);
    MultiFab zvel(convert(cells, IntVect(0,0,1)), dm, 1, 0);
    MultiFab cons(cells, dm, RhoTheta_comp + 1, 0);
    MultiFab lower(cells, dm, erf_interval_means::NumIntervalMoments, 0);
    MultiFab third(cells, dm, erf_interval_means::NumTripleVelocityMoments, 0);
    cons.setVal(Real(0.0));
    cons.setVal(Real(1.0), Rho_comp, 1, 0);
    cons.setVal(Real(300.0), RhoTheta_comp, 1, 0);

    const Real expected_central[] = {Real(-9.0), Real(-4.5), Real(4.5)};
    for (int shifted = 0; shifted < 2; ++shifted) {
        const Real du = shifted ? Real(10.0) : Real(0.0);
        const Real dv = shifted ? Real(-7.0) : Real(0.0);
        const Real dw = shifted ? Real(2.0) : Real(0.0);
        lower.setVal(Real(0.0));
        third.setVal(Real(0.0));
        double accumulated_time = 0.0;

        xvel.setVal(Real(1.0) + du);
        yvel.setVal(Real(2.0) + dv);
        zvel.setVal(Real(3.0) + dw);
        Accumulate_Interval_Means(
            1.0, accumulated_time, &lower, xvel, yvel, zvel, cons);
        Accumulate_Interval_TripleVelocity(
            1.0, &third, xvel, yvel, zvel);

        xvel.setVal(Real(5.0) + du);
        yvel.setVal(Real(4.0) + dv);
        zvel.setVal(Real(1.0) + dw);
        Accumulate_Interval_Means(
            3.0, accumulated_time, &lower, xvel, yvel, zvel, cons);
        Accumulate_Interval_TripleVelocity(
            3.0, &third, xvel, yvel, zvel);

        EXPECT_DOUBLE_EQ(accumulated_time, 4.0);
        const Real inv = Real(1.0) / static_cast<Real>(accumulated_time);
        Real lower_sum[erf_interval_means::NumIntervalMoments];
        for (int component = 0;
             component < erf_interval_means::NumIntervalMoments; ++component) {
            lower_sum[component] = lower.sum(component);
        }
        Real third_sum[erf_interval_means::NumTripleVelocityMoments];
        for (int component = 0;
             component < erf_interval_means::NumTripleVelocityMoments; ++component) {
            third_sum[component] = third.sum(component);
        }
        const int first[] = {
            erf_interval_means::U,
            erf_interval_means::V,
            erf_interval_means::W};
        const int cross[AMREX_SPACEDIM][AMREX_SPACEDIM] = {
            {erf_interval_means::UU, erf_interval_means::UV, erf_interval_means::UW},
            {erf_interval_means::UV, erf_interval_means::VV, erf_interval_means::VW},
            {erf_interval_means::UW, erf_interval_means::VW, erf_interval_means::WW}};
        for (int component = 0; component < AMREX_SPACEDIM; ++component) {
            const Real central =
                erf_interval_means::third_velocity_central_from_sums(
                    third_sum[component],
                    lower_sum[erf_interval_means::U],
                    lower_sum[erf_interval_means::V],
                    lower_sum[erf_interval_means::W],
                    lower_sum[erf_interval_means::UU],
                    lower_sum[erf_interval_means::VV],
                    lower_sum[erf_interval_means::WW],
                    lower_sum[first[component]],
                    lower_sum[cross[0][component]],
                    lower_sum[cross[1][component]],
                    lower_sum[cross[2][component]], inv);
            EXPECT_NEAR(central, expected_central[component], Real(1.0e-12));
        }
        if (!shifted) {
            EXPECT_DOUBLE_EQ(
                third_sum[erf_interval_means::UiUiU] * inv, Real(161.0));
            EXPECT_DOUBLE_EQ(
                third_sum[erf_interval_means::UiUiV] * inv, Real(133.0));
            EXPECT_DOUBLE_EQ(
                third_sum[erf_interval_means::UiUiW] * inv, Real(42.0));
        }
    }
}

TEST(IntervalMeansAccumulation, TimeWeightsFaceCenteredPertPresCovariances)
{
    using namespace amrex;

    const Box cell_box(IntVect(0, 0, 0), IntVect(0, 0, 0));
    const BoxArray cells(cell_box);
    const DistributionMapping dm(cells);
    const RealBox real_box({Real(0.0), Real(0.0), Real(0.0)},
                           {Real(1.0), Real(1.0), Real(1.0)});
    const int periodic[AMREX_SPACEDIM] = {0, 0, 0};
    const Geometry geom(cell_box, &real_box, CoordSys::cartesian, periodic);
    MultiFab pressure(cells, dm, 1, 1);
    MultiFab pressure_sum(cells, dm, erf_interval_means::NumPertPresCellMoments, 0);
    Array<MultiFab, AMREX_SPACEDIM> velocity;
    Array<MultiFab, AMREX_SPACEDIM> face_moments;
    Array<MultiFab*, AMREX_SPACEDIM> face_moment_ptrs;
    Array<const MultiFab*, AMREX_SPACEDIM> velocity_ptrs;
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        const BoxArray faces = convert(cells, IntVect::TheDimensionVector(dir));
        velocity[dir].define(faces, dm, 1, 0);
        face_moments[dir].define(
            faces, dm, erf_interval_means::NumPertPresFaceMoments, 0);
        face_moments[dir].setVal(Real(0.0));
        face_moment_ptrs[dir] = &face_moments[dir];
        velocity_ptrs[dir] = &velocity[dir];
    }
    pressure_sum.setVal(Real(0.0));

    pressure.setVal(Real(2.0));
    velocity[0].setVal(Real(1.0));
    velocity[1].setVal(Real(2.0));
    velocity[2].setVal(Real(3.0));
    Accumulate_Interval_PertPres(
        1.0, &pressure_sum, face_moment_ptrs, pressure, velocity_ptrs, geom);

    pressure.setVal(Real(6.0));
    velocity[0].setVal(Real(5.0));
    velocity[1].setVal(Real(4.0));
    velocity[2].setVal(Real(1.0));
    Accumulate_Interval_PertPres(
        3.0, &pressure_sum, face_moment_ptrs, pressure, velocity_ptrs, geom);

    constexpr Real inv_time = Real(0.25);
    EXPECT_DOUBLE_EQ(pressure_sum[0](IntVect(0,0,0), erf_interval_means::PertPres) * inv_time,
                     Real(5.0));
    const Real expected_product[] = {Real(23.0), Real(19.0), Real(6.0)};
    const Real expected_covariance[] = {Real(3.0), Real(1.5), Real(-1.5)};
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        const IntVect low_face(0,0,0);
        const Real velocity_sum = face_moments[dir][0](
            low_face, erf_interval_means::FaceVelocity);
        const Real product_sum = face_moments[dir][0](
            low_face, erf_interval_means::PertPresVelocity);
        EXPECT_DOUBLE_EQ(product_sum * inv_time, expected_product[dir]);
        EXPECT_DOUBLE_EQ(erf_interval_means::covariance_from_sums(
                             product_sum, Real(20.0), velocity_sum, inv_time),
                         expected_covariance[dir]);
    }
}

TEST(IntervalMeansAccumulation, InterpolatesPertPresToPeriodicAndPhysicalFaces)
{
    using namespace amrex;

    const Box cell_box(IntVect(0, 0, 0), IntVect(1, 0, 0));
    const BoxArray cells(cell_box);
    const DistributionMapping dm(cells);
    const RealBox real_box({Real(0.0), Real(0.0), Real(0.0)},
                           {Real(2.0), Real(1.0), Real(1.0)});

    auto accumulate_x_faces = [&] (int periodic_x) {
        const int periodic[AMREX_SPACEDIM] = {periodic_x, 0, 0};
        const Geometry geom(cell_box, &real_box, CoordSys::cartesian, periodic);
        MultiFab pressure(cells, dm, 1, 1);
        pressure.setVal(Real(0.0));
        pressure[0](IntVect(0,0,0), 0) = Real(2.0);
        pressure[0](IntVect(1,0,0), 0) = Real(6.0);
        pressure.FillBoundary(geom.periodicity());

        MultiFab pressure_sum(cells, dm, erf_interval_means::NumPertPresCellMoments, 0);
        pressure_sum.setVal(Real(0.0));
        Array<MultiFab, AMREX_SPACEDIM> velocity;
        Array<MultiFab, AMREX_SPACEDIM> face_moments;
        Array<MultiFab*, AMREX_SPACEDIM> face_moment_ptrs;
        Array<const MultiFab*, AMREX_SPACEDIM> velocity_ptrs;
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            const BoxArray faces = convert(cells, IntVect::TheDimensionVector(dir));
            velocity[dir].define(faces, dm, 1, 0);
            velocity[dir].setVal(Real(1.0));
            face_moments[dir].define(
                faces, dm, erf_interval_means::NumPertPresFaceMoments, 0);
            face_moments[dir].setVal(Real(0.0));
            face_moment_ptrs[dir] = &face_moments[dir];
            velocity_ptrs[dir] = &velocity[dir];
        }
        Accumulate_Interval_PertPres(
            1.0, &pressure_sum, face_moment_ptrs, pressure, velocity_ptrs, geom);

        return std::array<Real, 3>{
            face_moments[0][0](IntVect(0,0,0), erf_interval_means::PertPresVelocity),
            face_moments[0][0](IntVect(1,0,0), erf_interval_means::PertPresVelocity),
            face_moments[0][0](IntVect(2,0,0), erf_interval_means::PertPresVelocity)};
    };

    const auto physical = accumulate_x_faces(0);
    EXPECT_DOUBLE_EQ(physical[0], Real(2.0));
    EXPECT_DOUBLE_EQ(physical[1], Real(4.0));
    EXPECT_DOUBLE_EQ(physical[2], Real(6.0));

    const auto periodic = accumulate_x_faces(1);
    EXPECT_DOUBLE_EQ(periodic[0], Real(4.0));
    EXPECT_DOUBLE_EQ(periodic[1], Real(4.0));
    EXPECT_DOUBLE_EQ(periodic[2], Real(4.0));
}

TEST(IntervalMeansAccumulation, TimeWeightsNativeFacePressureGradients)
{
    using namespace amrex;

    const BoxArray cells(Box(IntVect(0,0,0), IntVect(0,0,0)));
    const DistributionMapping dm(cells);
    Array<MultiFab, AMREX_SPACEDIM> gradient;
    Array<MultiFab, AMREX_SPACEDIM> moments;
    Array<const MultiFab*, AMREX_SPACEDIM> gradient_ptrs;
    Array<MultiFab*, AMREX_SPACEDIM> moment_ptrs;
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        const BoxArray faces = convert(cells, IntVect::TheDimensionVector(dir));
        gradient[dir].define(faces, dm, 1, 0);
        moments[dir].define(faces, dm, erf_interval_means::NumGradPFaceMoments, 0);
        moments[dir].setVal(Real(0.0));
        gradient_ptrs[dir] = &gradient[dir];
        moment_ptrs[dir] = &moments[dir];
    }

    auto set_faces = [&] (Real low, Real high) {
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            IntVect lo(0,0,0);
            IntVect hi(0,0,0);
            hi[dir] = 1;
            gradient[dir][0](lo,0) = low;
            gradient[dir][0](hi,0) = high;
        }
    };

    set_faces(Real(2.0), Real(4.0));
    Accumulate_Interval_GradP(1.0, moment_ptrs, gradient_ptrs);
    set_faces(Real(6.0), Real(10.0));
    Accumulate_Interval_GradP(3.0, moment_ptrs, gradient_ptrs);

    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        IntVect lo(0,0,0);
        IntVect hi(0,0,0);
        hi[dir] = 1;
        const Real low_mean = moments[dir][0](lo,erf_interval_means::GradP) / Real(4.0);
        const Real high_mean = moments[dir][0](hi,erf_interval_means::GradP) / Real(4.0);
        EXPECT_DOUBLE_EQ(low_mean, Real(5.0));
        EXPECT_DOUBLE_EQ(high_mean, Real(8.5));
        EXPECT_DOUBLE_EQ(Real(0.5) * (low_mean + high_mean), Real(6.75));
    }
}

TEST(IntervalMeansAccumulation, TimeWeightsCellCenteredTotalStress)
{
    using namespace amrex;

    const BoxArray cells(Box(IntVect(0,0,0), IntVect(0,0,0)));
    const DistributionMapping dm(cells);
    Vector<std::unique_ptr<MultiFab>> tau(9);
    tau[TauType::tau11] = std::make_unique<MultiFab>(cells, dm, 1, 0);
    tau[TauType::tau22] = std::make_unique<MultiFab>(cells, dm, 1, 0);
    tau[TauType::tau33] = std::make_unique<MultiFab>(cells, dm, 1, 0);
    tau[TauType::tau12] = std::make_unique<MultiFab>(
        convert(cells, IntVect(1,1,0)), dm, 1, 0);
    tau[TauType::tau13] = std::make_unique<MultiFab>(
        convert(cells, IntVect(1,0,1)), dm, 1, 0);
    tau[TauType::tau23] = std::make_unique<MultiFab>(
        convert(cells, IntVect(0,1,1)), dm, 1, 0);
    MultiFab moments(cells, dm, erf_interval_means::NumTauMeanMoments, 0);
    moments.setVal(Real(0.0));

    auto set_four_edges = [] (MultiFab& field, int dir1, int dir2, Real average) {
        IntVect p0(0,0,0), p1(0,0,0), p2(0,0,0), p3(0,0,0);
        p1[dir1] = 1;
        p2[dir2] = 1;
        p3[dir1] = 1;
        p3[dir2] = 1;
        field[0](p0,0) = average - Real(3.0);
        field[0](p1,0) = average - Real(1.0);
        field[0](p2,0) = average + Real(1.0);
        field[0](p3,0) = average + Real(3.0);
    };
    auto set_stress = [&] (Real diagonal, Real off_diagonal) {
        tau[TauType::tau11]->setVal(diagonal);
        tau[TauType::tau22]->setVal(diagonal + Real(1.0));
        tau[TauType::tau33]->setVal(diagonal + Real(2.0));
        set_four_edges(*tau[TauType::tau12], 0, 1, off_diagonal);
        set_four_edges(*tau[TauType::tau13], 0, 2, off_diagonal + Real(1.0));
        set_four_edges(*tau[TauType::tau23], 1, 2, off_diagonal + Real(2.0));
    };

    set_stress(Real(2.0), Real(5.0));
    Accumulate_Interval_Tau(1.0, &moments, tau);
    set_stress(Real(6.0), Real(9.0));
    Accumulate_Interval_Tau(3.0, &moments, tau);

    const Real expected[] = {Real(5.0), Real(8.0), Real(9.0),
                             Real(6.0), Real(10.0), Real(7.0)};
    for (int comp = 0; comp < erf_interval_means::NumTauMeanMoments; ++comp) {
        EXPECT_DOUBLE_EQ(moments.sum(comp) / Real(4.0), expected[comp]);
    }
}

TEST(IntervalMeansAccumulation, TimeWeightsNativeFaceSGSEnergyFlux)
{
    using namespace amrex;

    const BoxArray cells(Box(IntVect(0,0,0), IntVect(0,0,0)));
    const DistributionMapping dm(cells);
    Vector<std::unique_ptr<MultiFab>> tau(9);
    tau[TauType::tau11] = std::make_unique<MultiFab>(cells, dm, 1, 1);
    tau[TauType::tau22] = std::make_unique<MultiFab>(cells, dm, 1, 1);
    tau[TauType::tau33] = std::make_unique<MultiFab>(cells, dm, 1, 1);
    tau[TauType::tau12] = std::make_unique<MultiFab>(
        convert(cells, IntVect(1,1,0)), dm, 1, 1);
    tau[TauType::tau13] = std::make_unique<MultiFab>(
        convert(cells, IntVect(1,0,1)), dm, 1, 1);
    tau[TauType::tau23] = std::make_unique<MultiFab>(
        convert(cells, IntVect(0,1,1)), dm, 1, 1);
    tau[TauType::tau11]->setVal(Real(1.0));
    tau[TauType::tau22]->setVal(Real(2.0));
    tau[TauType::tau33]->setVal(Real(3.0));
    tau[TauType::tau12]->setVal(Real(5.0));
    tau[TauType::tau13]->setVal(Real(6.0));
    tau[TauType::tau23]->setVal(Real(7.0));

    Array<MultiFab, AMREX_SPACEDIM> velocity;
    Array<MultiFab, AMREX_SPACEDIM> moments;
    Array<const MultiFab*, AMREX_SPACEDIM> velocity_ptrs;
    Array<MultiFab*, AMREX_SPACEDIM> moment_ptrs;
    const Real first_velocity[] = {Real(2.0), Real(3.0), Real(4.0)};
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        const BoxArray faces = convert(cells, IntVect::TheDimensionVector(dir));
        velocity[dir].define(faces, dm, 1, 1);
        moments[dir].define(
            faces, dm, erf_interval_means::NumSGSEnergyFluxFaceMoments, 0);
        velocity[dir].setVal(first_velocity[dir]);
        moments[dir].setVal(Real(0.0));
        velocity_ptrs[dir] = &velocity[dir];
        moment_ptrs[dir] = &moments[dir];
    }

    // Missing transpose storage must use the symmetric aliases 21=12,
    // 31=13, and 32=23.
    Accumulate_Interval_SGSEnergyFlux(
        1.0, moment_ptrs, tau, velocity_ptrs);
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        velocity[dir].setVal(Real(2.0) * first_velocity[dir]);
    }
    Accumulate_Interval_SGSEnergyFlux(
        3.0, moment_ptrs, tau, velocity_ptrs);

    const Real expected[] = {Real(71.75), Real(77.0), Real(78.75)};
    const IntVect sample(0,0,0);
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        EXPECT_DOUBLE_EQ(
            moments[dir][0](sample, erf_interval_means::SGSEnergyFlux) /
                Real(4.0),
            expected[dir]);
    }

    // When terrain/implicit-diffusion storage provides distinct transpose
    // stresses, the contraction must use those actual components.
    tau[TauType::tau21] = std::make_unique<MultiFab>(
        convert(cells, IntVect(1,1,0)), dm, 1, 1);
    tau[TauType::tau31] = std::make_unique<MultiFab>(
        convert(cells, IntVect(1,0,1)), dm, 1, 1);
    tau[TauType::tau32] = std::make_unique<MultiFab>(
        convert(cells, IntVect(0,1,1)), dm, 1, 1);
    tau[TauType::tau21]->setVal(Real(8.0));
    tau[TauType::tau31]->setVal(Real(9.0));
    tau[TauType::tau32]->setVal(Real(10.0));
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        velocity[dir].setVal(first_velocity[dir]);
        moments[dir].setVal(Real(0.0));
    }
    Accumulate_Interval_SGSEnergyFlux(
        1.0, moment_ptrs, tau, velocity_ptrs);
    const Real distinct_expected[] = {Real(62.0), Real(56.0), Real(45.0)};
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        EXPECT_DOUBLE_EQ(
            moments[dir][0](sample, erf_interval_means::SGSEnergyFlux),
            distinct_expected[dir]);
    }
}

TEST(IntervalMeansRestart, LegacyCheckpointFallbackPreservesTimeResetSemantics)
{
    EXPECT_EQ(erf_interval_means::legacy_reset_done(true, 10.0, 10.0), 1);
    EXPECT_EQ(erf_interval_means::legacy_reset_done(true, 9.0, 10.0), 0);
    EXPECT_EQ(erf_interval_means::legacy_reset_done(false, 20.0, 10.0), 0);
}

TEST(IntervalMeansRestart, StressStrainAccumulatorUsesSharedTimeWeighting)
{
    using namespace amrex;
    const BoxArray boxes(Box(IntVect(0,0,0), IntVect(1,1,1)));
    const DistributionMapping dm(boxes);
    MultiFab instantaneous(boxes, dm, 1, 0);
    MultiFab moments(
        boxes, dm, erf_interval_means::NumStressStrainMoments, 0);
    moments.setVal(Real(0.0));

    instantaneous.setVal(Real(-2.0));
    Accumulate_Interval_StressStrain(1.0, &moments, &instantaneous);
    instantaneous.setVal(Real(-6.0));
    Accumulate_Interval_StressStrain(3.0, &moments, &instantaneous);

    const IntVect sample(0,0,0);
    EXPECT_DOUBLE_EQ(
        moments[0](sample, erf_interval_means::TotalStressStrain) / Real(4.0),
        Real(-5.0));
}

TEST(IntervalMeansRestart, InitializationPreservesRestoredWindow)
{
    EXPECT_TRUE(erf_interval_means::initialization_accumulates_state(false));
    EXPECT_FALSE(erf_interval_means::initialization_accumulates_state(true));
    EXPECT_TRUE(erf_interval_means::initialization_plot_consumes_interval(false, true));
    EXPECT_FALSE(erf_interval_means::initialization_plot_consumes_interval(true, true));
    EXPECT_FALSE(erf_interval_means::initialization_plot_consumes_interval(false, false));
}
