#ifndef ERF_INTERVAL_MEANS_CHECKPOINT_H_
#define ERF_INTERVAL_MEANS_CHECKPOINT_H_

#include <cmath>
#include <istream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <AMReX_BoxArray.H>
#include <AMReX_IntVect.H>
#include <AMReX_Vector.H>

#include "ERF_IntervalMeans.H"

namespace erf_interval_means {

namespace detail {

inline bool
read_int_vector (std::istream& input, amrex::IntVect& value)
{
    char delimiter;
    if (!(input >> delimiter) || delimiter != '(') return false;
    for (int idir = 0; idir < AMREX_SPACEDIM; ++idir) {
        if (idir > 0 && (!(input >> delimiter) || delimiter != ',')) return false;
        if (!(input >> value[idir])) return false;
    }
    return static_cast<bool>(input >> delimiter) && delimiter == ')';
}

// BoxArray::readFrom calls amrex::Error on malformed input.  Checkpoint
// metadata is external input, so parse the format emitted by writeOn first
// and only construct a BoxArray after the complete record is validated.
inline bool
read_box_array (std::istream& input, amrex::BoxArray& result)
{
    char delimiter;
    long long box_count;
    unsigned long long hash;
    if (!(input >> delimiter) || delimiter != '(' ||
        !(input >> box_count >> hash) || box_count < 0 ||
        box_count > static_cast<long long>(std::numeric_limits<int>::max())) {
        return false;
    }

    std::vector<amrex::Box> boxes;
    boxes.reserve(static_cast<std::size_t>(box_count));
    for (long long ibox = 0; ibox < box_count; ++ibox) {
        if (!(input >> delimiter) || delimiter != '(') return false;
        amrex::IntVect small, big, type;
        if (!read_int_vector(input, small) || !read_int_vector(input, big) ||
            !read_int_vector(input, type) ||
            !(input >> delimiter) || delimiter != ')') {
            return false;
        }
        boxes.emplace_back(small, big, type);
    }
    if (!(input >> delimiter) || delimiter != ')') return false;

    result = boxes.empty()
        ? amrex::BoxArray{}
        : amrex::BoxArray(boxes.data(), static_cast<int>(boxes.size()));
    return true;
}

} // namespace detail

struct LevelMetadata
{
    int level{-1};
    double accumulation_count{0.0};
    int reset_done{0};
    amrex::BoxArray boxes;
    amrex::Vector<int> processor_map;
};

struct Metadata
{
    int version{0};
    int levels{0};
    int components{0};
    amrex::Vector<LevelMetadata> level;
};

struct PertPresLevelMetadata
{
    int level{-1};
    double accumulation_count{0.0};
    amrex::BoxArray cell_boxes;
    amrex::Array<amrex::BoxArray, AMREX_SPACEDIM> face_boxes;
};

struct PertPresMetadata
{
    int levels{0};
    int cell_components{0};
    int face_components{0};
    amrex::Vector<PertPresLevelMetadata> level;
};

inline bool
parse_pert_pres_metadata (std::istream& input, PertPresMetadata& result,
                          std::string& error)
{
    std::string title;
    if (!std::getline(input, title) || title != "ERF interval pert_pres checkpoint v1") {
        error = "unsupported interval pert_pres checkpoint title '" + title + "'";
        return false;
    }
    if (!(input >> result.levels >> result.cell_components >> result.face_components) ||
        result.levels < 0) {
        error = "missing or invalid interval pert_pres dimensions";
        return false;
    }
    result.level.resize(result.levels);
    for (int lev = 0; lev < result.levels; ++lev) {
        auto& record = result.level[lev];
        if (!(input >> record.level >> record.accumulation_count) ||
            record.level != lev || !std::isfinite(record.accumulation_count) ||
            record.accumulation_count < 0.0) {
            error = "invalid interval pert_pres level record " + std::to_string(lev);
            return false;
        }
        if (!detail::read_box_array(input, record.cell_boxes)) {
            error = "invalid interval pert_pres cell BoxArray at level " + std::to_string(lev);
            return false;
        }
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            if (!detail::read_box_array(input, record.face_boxes[dir])) {
                error = "invalid interval pert_pres face BoxArray at level " +
                        std::to_string(lev);
                return false;
            }
        }
    }
    return true;
}

inline std::string
validate_pert_pres_metadata (
    const PertPresMetadata& metadata,
    const amrex::Vector<amrex::BoxArray>& expected_cells,
    const amrex::Vector<double>& expected_counts)
{
    const int expected_levels = static_cast<int>(expected_cells.size());
    if (metadata.levels != expected_levels ||
        static_cast<int>(metadata.level.size()) != expected_levels ||
        static_cast<int>(expected_counts.size()) < expected_levels) {
        return "interval pert_pres metadata level count does not match the restart";
    }
    if (metadata.cell_components != NumPertPresCellMoments ||
        metadata.face_components != NumPertPresFaceMoments) {
        return "interval pert_pres metadata component counts are unsupported";
    }
    for (int lev = 0; lev < expected_levels; ++lev) {
        if (!(metadata.level[lev].cell_boxes == expected_cells[lev])) {
            return "interval pert_pres cell BoxArray does not match level " + std::to_string(lev);
        }
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            amrex::BoxArray expected = amrex::convert(
                expected_cells[lev], amrex::IntVect::TheDimensionVector(dir));
            if (!(metadata.level[lev].face_boxes[dir] == expected)) {
                return "interval pert_pres face BoxArray does not match level " +
                       std::to_string(lev);
            }
        }
        const double scale = std::max(1.0, std::abs(expected_counts[lev]));
        if (std::abs(metadata.level[lev].accumulation_count - expected_counts[lev]) >
            16.0 * std::numeric_limits<double>::epsilon() * scale) {
            return "interval pert_pres accumulated time does not match level " +
                   std::to_string(lev);
        }
    }
    return {};
}

inline bool
parse_metadata (std::istream& input, Metadata& result, std::string& error)
{
    std::string title;
    if (!std::getline(input, title)) {
        error = "missing interval-mean checkpoint title";
        return false;
    }
    if (title == "ERF interval means checkpoint v1") {
        result.version = 1;
    } else if (title == "ERF interval means checkpoint v2") {
        result.version = 2;
    } else {
        error = "unsupported interval-mean checkpoint title '" + title + "'";
        return false;
    }
    if (!(input >> result.levels >> result.components) || result.levels < 0 ||
        result.components < 0) {
        error = "missing or invalid interval-mean level/component counts";
        return false;
    }
    const int schema_components = result.version == 1
        ? LegacyNumIntervalMoments : NumIntervalMoments;
    if (result.components != schema_components) {
        error = "interval-mean checkpoint v" + std::to_string(result.version) +
                " requires " + std::to_string(schema_components) + " components";
        return false;
    }

    result.level.clear();
    result.level.resize(result.levels);
    for (int lev = 0; lev < result.levels; ++lev) {
        auto& record = result.level[lev];
        if (!(input >> record.level >> record.accumulation_count >> record.reset_done) ||
            record.level != lev || !std::isfinite(record.accumulation_count) ||
            record.accumulation_count < 0.0 ||
            (record.reset_done != 0 && record.reset_done != 1)) {
            error = "invalid interval-mean level record " + std::to_string(lev);
            return false;
        }

        if (!detail::read_box_array(input, record.boxes)) {
            error = "truncated interval-mean BoxArray record for level " +
                    std::to_string(lev);
            return false;
        }
        std::string box_remainder;
        if (!std::getline(input, box_remainder)) {
            error = "truncated interval-mean BoxArray line for level " +
                    std::to_string(lev);
            return false;
        }

        long long map_size = -1;
        if (!(input >> map_size) || map_size < 0 ||
            map_size != static_cast<long long>(record.boxes.size())) {
            error = "invalid interval-mean processor-map size for level " +
                    std::to_string(lev);
            return false;
        }
        record.processor_map.resize(static_cast<std::size_t>(map_size));
        for (long long ibox = 0; ibox < map_size; ++ibox) {
            if (!(input >> record.processor_map[static_cast<std::size_t>(ibox)])) {
                error = "truncated interval-mean processor map for level " +
                        std::to_string(lev);
                return false;
            }
        }
    }
    return true;
}

inline std::string
validate_metadata (const Metadata& metadata,
                   int expected_levels,
                   int expected_components,
                   const amrex::Vector<amrex::BoxArray>& expected_boxes)
{
    if (metadata.levels != expected_levels || metadata.components != expected_components) {
        return "expected " + std::to_string(expected_levels) + " levels and " +
               std::to_string(expected_components) + " components";
    }
    if (static_cast<int>(expected_boxes.size()) < expected_levels ||
        static_cast<int>(metadata.level.size()) != expected_levels) {
        return "interval-mean metadata level count does not match the restart";
    }
    for (int lev = 0; lev < expected_levels; ++lev) {
        if (!(metadata.level[lev].boxes == expected_boxes[lev])) {
            return "interval-mean checkpoint BoxArray does not match level " +
                   std::to_string(lev);
        }
    }
    return {};
}

inline int
global_reset_done (const Metadata& metadata)
{
    for (const auto& record : metadata.level) {
        if (record.reset_done != 0) return 1;
    }
    return 0;
}

inline bool
restart_resets_window (const Metadata& metadata) noexcept
{
    return metadata.version == 1;
}

inline int
legacy_reset_done (bool time_reset_mode, double checkpoint_time, double reset_time)
{
    return time_reset_mode && checkpoint_time >= reset_time ? 1 : 0;
}

inline bool
initialization_accumulates_state (bool restarting) noexcept
{
    return !restarting;
}

inline bool
initialization_plot_consumes_interval (bool restarting,
                                       bool interval_diagnostic_selected) noexcept
{
    return !restarting && interval_diagnostic_selected;
}

} // namespace erf_interval_means

#endif
