#include <sstream>
#include <string>

#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
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

TEST(IntervalMeansRestart, LegacyCheckpointFallbackPreservesTimeResetSemantics)
{
    EXPECT_EQ(erf_interval_means::legacy_reset_done(true, 10.0, 10.0), 1);
    EXPECT_EQ(erf_interval_means::legacy_reset_done(true, 9.0, 10.0), 0);
    EXPECT_EQ(erf_interval_means::legacy_reset_done(false, 20.0, 10.0), 0);
}

TEST(IntervalMeansRestart, InitializationPreservesRestoredWindow)
{
    EXPECT_TRUE(erf_interval_means::initialization_accumulates_state(false));
    EXPECT_FALSE(erf_interval_means::initialization_accumulates_state(true));
    EXPECT_TRUE(erf_interval_means::initialization_plot_consumes_interval(false, true));
    EXPECT_FALSE(erf_interval_means::initialization_plot_consumes_interval(true, true));
    EXPECT_FALSE(erf_interval_means::initialization_plot_consumes_interval(false, false));
}
