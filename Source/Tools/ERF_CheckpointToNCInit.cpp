/**
 * Convert a level-0 ERF checkpoint into an idealized NCFile initialization
 * dataset, periodically tiled in x and/or y.
 */

#include <AMReX.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Gpu.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Utility.H>
#include <AMReX_VisMF.H>

#include <ERF_IndexDefines.H>
#include <ERF_NCInterface.H>

#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

using namespace amrex;

namespace {

void
goto_next_line (std::istream& input)
{
    input.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
}

struct CheckpointMetadata {
    int finest_level = -1;
    int ncomp_cons = 0;
    Real time = 0.0;
    Vector<BoxArray> grids;
};

CheckpointMetadata
read_checkpoint_header (const std::string& checkpoint)
{
    const std::string header_name = checkpoint + "/Header";
    if (!FileExists(header_name)) {
        Abort("Checkpoint header does not exist: " + header_name);
    }

    Vector<char> chars;
    ParallelDescriptor::ReadAndBcastFile(header_name, chars);
    std::istringstream input(std::string(chars.dataPtr()), std::istringstream::in);

    CheckpointMetadata metadata;
    std::string line;
    std::getline(input, line); // title
    input >> metadata.finest_level;
    goto_next_line(input);
    input >> metadata.ncomp_cons;
    goto_next_line(input);

    int ncomp_velocity = 0;
    for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
        input >> ncomp_velocity;
        goto_next_line(input);
        if (ncomp_velocity != 1) {
            Abort("Checkpoint velocity data must have one component");
        }
    }

    std::getline(input, line); // istep
    std::getline(input, line); // dt
    std::getline(input, line); // t_new
    {
        std::istringstream times(line);
        times >> metadata.time;
    }

    metadata.grids.resize(metadata.finest_level + 1);
    for (int lev = 0; lev <= metadata.finest_level; ++lev) {
        metadata.grids[lev].readFrom(input);
        goto_next_line(input);
    }
    return metadata;
}

void
read_base_layout (const std::string& checkpoint, int& ncomp, IntVect& ngrow)
{
    ncomp = 3;
    ngrow = IntVect(1);
    const std::string name = checkpoint + "/num_base_state_comps";
    if (!FileExists(name)) {
        return;
    }

    Vector<char> chars;
    ParallelDescriptor::ReadAndBcastFile(name, chars);
    std::istringstream input(std::string(chars.dataPtr()), std::istringstream::in);
    input >> ncomp >> ngrow;
}

void
tile_multifab (const MultiFab& source, MultiFab& target,
               int tile_x, int tile_y, int source_nx, int source_ny)
{
    target.setVal(0.0);
    for (int jy = 0; jy < tile_y; ++jy) {
        for (int ix = 0; ix < tile_x; ++ix) {
            MultiFab shifted(source.boxArray(), source.DistributionMap(),
                             source.nComp(), 0);
            MultiFab::Copy(shifted, source, 0, 0, source.nComp(), 0);
            shifted.shift(IntVect(ix * source_nx, jy * source_ny, 0));
            target.ParallelCopy(shifted, 0, 0, source.nComp(), 0, 0);
        }
    }
}

Vector<Real>
gather_component (const MultiFab& source, const Box& domain, int component)
{
    BoxArray one_box(domain);
    Vector<int> processor_map(1, ParallelDescriptor::IOProcessorNumber());
    DistributionMapping io_mapping(processor_map);
    MultiFab gathered(one_box, io_mapping, 1, 0);
    gathered.ParallelCopy(source, component, 0, 1, 0, 0);

    Vector<Real> host_data;
    if (ParallelDescriptor::IOProcessor()) {
        const FArrayBox& device_fab = gathered[0];
        host_data.resize(domain.numPts());
        Gpu::copy(Gpu::deviceToHost, device_fab.dataPtr(),
                  device_fab.dataPtr() + domain.numPts(), host_data.begin());
    }
    return host_data;
}

int
main_main ()
{
    ParmParse pp;
    std::string checkpoint;
    std::string output;
    int tile_x = 1;
    int tile_y = 1;
    int max_grid_size = 64;
    pp.get("checkpoint", checkpoint);
    pp.get("output", output);
    pp.query("tile_x", tile_x);
    pp.query("tile_y", tile_y);
    pp.query("max_grid_size", max_grid_size);

    if (tile_x < 1 || tile_y < 1 || max_grid_size < 1) {
        Abort("tile_x, tile_y, and max_grid_size must be positive");
    }
    while (!checkpoint.empty() && checkpoint.back() == '/') {
        checkpoint.pop_back();
    }

    const CheckpointMetadata metadata = read_checkpoint_header(checkpoint);
    if (metadata.finest_level != 0) {
        Abort("erf_chk_to_nc_init currently requires a single-level checkpoint");
    }
    if (metadata.ncomp_cons <= RhoKE_comp) {
        Abort("Checkpoint does not contain the RHO_KE conserved component");
    }

    const Box source_domain = metadata.grids[0].minimalBox();
    if (source_domain.smallEnd() != IntVect(0)) {
        Abort("Checkpoint level-0 domain must start at index (0,0,0)");
    }
    const int source_nx = source_domain.length(0);
    const int source_ny = source_domain.length(1);
    const int source_nz = source_domain.length(2);

    Box target_domain = source_domain;
    target_domain.setBig(0, source_domain.smallEnd(0) + tile_x * source_nx - 1);
    target_domain.setBig(1, source_domain.smallEnd(1) + tile_y * source_ny - 1);

    BoxArray target_cell_ba(target_domain);
    target_cell_ba.maxSize(max_grid_size);
    DistributionMapping target_dm(target_cell_ba);
    DistributionMapping source_dm(metadata.grids[0]);

    MultiFab source_cons(metadata.grids[0], source_dm, metadata.ncomp_cons, 0);
    VisMF::Read(source_cons,
        MultiFabFileFullPrefix(0, checkpoint, "Level_", "Cell"));
    MultiFab target_cons(target_cell_ba, target_dm, metadata.ncomp_cons, 0);
    tile_multifab(source_cons, target_cons, tile_x, tile_y, source_nx, source_ny);

    Array<IntVect, AMREX_SPACEDIM> staggering{
        IntVect(1,0,0), IntVect(0,1,0), IntVect(0,0,1)};
    Array<std::string, AMREX_SPACEDIM> checkpoint_names{
        "XFace", "YFace", "ZFace"};
    Vector<std::unique_ptr<MultiFab>> target_velocity(AMREX_SPACEDIM);
    for (int direction = 0; direction < AMREX_SPACEDIM; ++direction) {
        BoxArray source_face_ba = convert(metadata.grids[0], staggering[direction]);
        DistributionMapping source_face_dm(source_face_ba);
        MultiFab source_velocity(source_face_ba, source_face_dm, 1, 0);
        VisMF::Read(source_velocity,
            MultiFabFileFullPrefix(0, checkpoint, "Level_", checkpoint_names[direction]));

        BoxArray target_face_ba = convert(target_cell_ba, staggering[direction]);
        target_velocity[direction] =
            std::make_unique<MultiFab>(target_face_ba, target_dm, 1, 0);
        tile_multifab(source_velocity, *target_velocity[direction],
                      tile_x, tile_y, source_nx, source_ny);
    }

    int ncomp_base = 0;
    IntVect ngrow_base;
    read_base_layout(checkpoint, ncomp_base, ngrow_base);
    if (ncomp_base <= BaseState::th0_comp) {
        Abort("Checkpoint BaseState does not contain theta_hse");
    }
    MultiFab source_base(metadata.grids[0], source_dm, ncomp_base, ngrow_base);
    VisMF::Read(source_base,
        MultiFabFileFullPrefix(0, checkpoint, "Level_", "BaseState"));
    MultiFab target_base(target_cell_ba, target_dm, ncomp_base, 0);
    tile_multifab(source_base, target_base, tile_x, tile_y, source_nx, source_ny);

    const Box u_domain = convert(target_domain, staggering[0]);
    const Box v_domain = convert(target_domain, staggering[1]);
    const Box w_domain = convert(target_domain, staggering[2]);

    Vector<Real> rho = gather_component(target_cons, target_domain, Rho_comp);
    Vector<Real> theta = gather_component(target_cons, target_domain, RhoTheta_comp);
    Vector<Real> rho_ke = gather_component(target_cons, target_domain, RhoKE_comp);
    Vector<Real> u = gather_component(*target_velocity[0], u_domain, 0);
    Vector<Real> v = gather_component(*target_velocity[1], v_domain, 0);
    Vector<Real> w = gather_component(*target_velocity[2], w_domain, 0);
    Vector<Real> rho_hse = gather_component(target_base, target_domain, BaseState::r0_comp);
    Vector<Real> p_hse = gather_component(target_base, target_domain, BaseState::p0_comp);
    Vector<Real> theta_hse = gather_component(target_base, target_domain, BaseState::th0_comp);

    if (ParallelDescriptor::IOProcessor()) {
        const Long num_cells = target_domain.numPts();
        for (Long n = 0; n < num_cells; ++n) {
            if (rho[n] <= 0.0) {
                Abort("Non-positive density encountered while converting checkpoint");
            }
            theta[n] /= rho[n];
        }

        auto file = ncutils::NCFile::create(output);
        file.enter_def_mode();
        file.put_attr("title", "ERF tiled checkpoint initialization");
        file.put_attr("source_checkpoint", checkpoint);
        file.put_attr("source_time", std::vector<double>{metadata.time});
        file.put_attr("tile_x", std::vector<int>{tile_x});
        file.put_attr("tile_y", std::vector<int>{tile_y});

        file.def_dim("Time", 1);
        file.def_dim("bottom_top", source_nz);
        file.def_dim("bottom_top_stag", source_nz + 1);
        file.def_dim("south_north", tile_y * source_ny);
        file.def_dim("south_north_stag", tile_y * source_ny + 1);
        file.def_dim("west_east", tile_x * source_nx);
        file.def_dim("west_east_stag", tile_x * source_nx + 1);

        const std::vector<std::string> cell_dims{
            "Time", "bottom_top", "south_north", "west_east"};
        file.def_var("RHO", ncutils::NCDType::Real, cell_dims);
        file.def_var("T", ncutils::NCDType::Real, cell_dims);
        file.def_var("RHO_KE", ncutils::NCDType::Real, cell_dims);
        file.def_var("RHO_HSE", ncutils::NCDType::Real, cell_dims);
        file.def_var("P_HSE", ncutils::NCDType::Real, cell_dims);
        file.def_var("T_HSE", ncutils::NCDType::Real, cell_dims);
        file.def_var("U", ncutils::NCDType::Real,
            {"Time", "bottom_top", "south_north", "west_east_stag"});
        file.def_var("V", ncutils::NCDType::Real,
            {"Time", "bottom_top", "south_north_stag", "west_east"});
        file.def_var("W", ncutils::NCDType::Real,
            {"Time", "bottom_top_stag", "south_north", "west_east"});
        file.exit_def_mode();

        file.var("RHO").put(rho.data());
        file.var("T").put(theta.data());
        file.var("RHO_KE").put(rho_ke.data());
        file.var("RHO_HSE").put(rho_hse.data());
        file.var("P_HSE").put(p_hse.data());
        file.var("T_HSE").put(theta_hse.data());
        file.var("U").put(u.data());
        file.var("V").put(v.data());
        file.var("W").put(w.data());
        file.close();

        Print() << "Wrote " << output << " from checkpoint " << checkpoint << "\n"
                << "  source cells: " << source_nx << " x " << source_ny
                << " x " << source_nz << "\n"
                << "  target cells: " << tile_x * source_nx << " x "
                << tile_y * source_ny << " x " << source_nz << "\n"
                << "  checkpoint time: " << metadata.time << std::endl;
    }

    return 0;
}

} // namespace

int
main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    const int status = main_main();
    amrex::Finalize();
    return status;
}
