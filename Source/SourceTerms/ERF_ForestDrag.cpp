#include <ERF_ForestDrag.H>

#include <AMReX_Print.H>
#include <AMReX_ParallelDescriptor.H>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <utility>
#include <vector>

using namespace amrex;

namespace {

std::string
forest_geometry_filename (const std::string& forestfile)
{
    std::string basename = forestfile;
    auto slash_pos = basename.find_last_of("/\\");
    if (slash_pos != std::string::npos) {
        basename = basename.substr(slash_pos + 1);
    }

    auto dot_pos = basename.find_last_of('.');
    if (dot_pos != std::string::npos && dot_pos > 0) {
        basename = basename.substr(0, dot_pos);
    }

    if (basename.empty()) {
        basename = "forest_geometry";
    }

    return basename + ".vtp";
}

}

/*
  Constructor to get the forest parameters:
  TreeType xc, yc, height, diameter, cd, lai, laimax
*/
ForestDrag::ForestDrag (std::string forestfile)
    : m_forestfile(std::move(forestfile))
{
    std::ifstream file(m_forestfile, std::ios::in);
    if (!file.good()) {
        Abort("Cannot find forest file: " + m_forestfile);
    }
    // TreeType xc yc height diameter cd lai laimax
    Real value1, value2, value3, value4, value5, value6, value7, value8;
    while (file >> value1 >> value2 >> value3 >> value4 >> value5 >> value6 >>
           value7 >> value8) {
        m_type_forest.push_back(value1);
        m_x_forest.push_back(value2);
        m_y_forest.push_back(value3);
        m_height_forest.push_back(value4);
        m_diameter_forest.push_back(value5);
        m_cd_forest.push_back(value6);
        m_lai_forest.push_back(value7);
        m_laimax_forest.push_back(value8);
    }
    file.close();
}

void
ForestDrag::define_drag_field (const BoxArray& ba,
                               const DistributionMapping& dm,
                               Geometry& geom,
                               MultiFab* z_phys_cc,
                               MultiFab* z_phys_nd)
{
    // Geometry params
    const auto& dx = geom.CellSizeArray();
    const auto& prob_lo = geom.ProbLoArray();

    bool all_boxes_touch_bottom = true;
    for (int i = 0; i < ba.size(); i++) {
        if (ba[i].smallEnd(2) != geom.ProbLo(2)) {
            all_boxes_touch_bottom = false;
        }
    }
    AMREX_ALWAYS_ASSERT(all_boxes_touch_bottom);

    // Allocate the forest drag MF
    // NOTE: 1 ghost cell for averaging to faces
    m_forest_drag.reset();
    m_forest_drag = std::make_unique<MultiFab>(ba,dm,1,1);
    m_forest_drag->setVal(0.);

    // Loop over forest types and pre-compute factors
    for (unsigned ii = 0; ii < m_x_forest.size(); ++ii)
    {
        // Expose CPU data for GPU capture
        Real af; // Depends upon the type of forest (tf)
        Real treeZm = 0.0; // Only for forest type 2
        int  tf = int(m_type_forest[ii]);
        Real hf = m_height_forest[ii];
        Real xf = m_x_forest[ii];
        Real yf = m_y_forest[ii];
        Real df = m_diameter_forest[ii];
        Real cdf  = m_cd_forest[ii];
        Real laif = m_lai_forest[ii];
        Real laimaxf = m_laimax_forest[ii];
        if (tf == 1) {
            // Constant factor
            af = laif / hf;
        } else {
            // Discretize integral with 100 points and pre-compute
            int nk      = 100;
            Real ztree  = 0;
            Real expFun = 0;
            Real ratio  = 0;
            const Real dz = hf / Real(nk);
            treeZm = laimaxf * hf;
            for (int k(0); k<nk; ++k) {
                ratio = (hf - treeZm) / (hf - ztree);
                if (ztree < treeZm) {
                    expFun += std::pow(ratio, 6.0) *
                              std::exp(6 * (1 - ratio));
                } else {
                    expFun += std::pow(ratio, 0.5) *
                              std::exp(0.5 * (1 - ratio));
                }
                ztree += dz;
            }
            af = laif / (expFun * dz);
        }

        // Set the forest drag data
        for (MFIter mfi(*m_forest_drag); mfi.isValid(); ++mfi) {
            Box gtbx = mfi.growntilebox();
            const Array4<Real>& levelDrag  = m_forest_drag->array(mfi);
            const Array4<const Real>& z_cc = z_phys_cc->const_array(mfi);
            const Array4<const Real>& z_nd = z_phys_nd->const_array(mfi);

            ParallelFor(gtbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
            {
                // Physical positions of cell-centers
                const Real x = prob_lo[0] + (i + 0.5) * dx[0];
                const Real y = prob_lo[1] + (j + 0.5) * dx[1];

                // "z" is measured as distance from cell center to ground
                const Real z_sfc = 0.25 * ( z_nd(i,j  ,0) + z_nd(i+1,j  ,0)
                                           +z_nd(i,j+1,0) + z_nd(i+1,j+1,0));
                const Real z = std::max((z_cc(i,j,k)-z_sfc),0.0);

                // Proximity to the forest
                const Real radius = std::sqrt((x - xf) * (x - xf) +
                                              (y - yf) * (y - yf));

                // Hit for canopy region
                Real factor = 1;
                if ((z <= hf) && (radius <= (0.5 * df))) {
                    if (tf == 2) {
                        Real ratio = (hf - treeZm) / (hf - z);
                        if (z < treeZm) {
                            factor = std::pow(ratio, 6.0) *
                                     std::exp(6.0 * (1.0 - ratio));
                        } else if (z <= hf) {
                            factor = std::pow(ratio, 0.5) *
                                     std::exp(0.5 * (1.0 - ratio));
                        }
                    }
                    levelDrag(i, j, k) = cdf * af * factor;
                }
            });
        } // mfi
    } // ii (forest type)

    // Fillboundary for periodic ghost cell copy
    m_forest_drag->FillBoundary(geom.periodicity());

} // init_drag_field

void
ForestDrag::write_geometry_vtp (const std::string& output_file) const
{
    if (!ParallelDescriptor::IOProcessor()) {
        return;
    }

    constexpr int nphi = 48;
    constexpr Real pi = Real(3.141592653589793238462643383279502884L);

    std::vector<Real> points;
    std::vector<int> connectivity;
    std::vector<int> offsets;
    std::vector<int> tree_id;
    std::vector<int> tree_type;

    points.reserve(m_x_forest.size() * (2 * nphi + 2) * 3);
    connectivity.reserve(m_x_forest.size() * 4 * nphi * 3);
    offsets.reserve(m_x_forest.size() * 4 * nphi);
    tree_id.reserve(m_x_forest.size() * 4 * nphi);
    tree_type.reserve(m_x_forest.size() * 4 * nphi);

    int next_offset = 0;

    for (int itree = 0; itree < static_cast<int>(m_x_forest.size()); ++itree) {
        const Real xc = m_x_forest[itree];
        const Real yc = m_y_forest[itree];
        const Real height = m_height_forest[itree];
        const Real radius = Real(0.5) * m_diameter_forest[itree];
        const int type = static_cast<int>(m_type_forest[itree]);

        const int point_base = static_cast<int>(points.size() / 3);
        for (int iphi = 0; iphi < nphi; ++iphi) {
            const Real angle = Real(2.0) * pi * Real(iphi) / Real(nphi);
            points.push_back(xc + radius * std::cos(angle));
            points.push_back(yc + radius * std::sin(angle));
            points.push_back(Real(0.0));
        }
        for (int iphi = 0; iphi < nphi; ++iphi) {
            const Real angle = Real(2.0) * pi * Real(iphi) / Real(nphi);
            points.push_back(xc + radius * std::cos(angle));
            points.push_back(yc + radius * std::sin(angle));
            points.push_back(height);
        }

        const int bottom_center = static_cast<int>(points.size() / 3);
        points.push_back(xc);
        points.push_back(yc);
        points.push_back(Real(0.0));

        const int top_center = static_cast<int>(points.size() / 3);
        points.push_back(xc);
        points.push_back(yc);
        points.push_back(height);

        for (int iphi = 0; iphi < nphi; ++iphi) {
            const int next = (iphi + 1) % nphi;

            const int b0 = point_base + iphi;
            const int b1 = point_base + next;
            const int t0 = point_base + nphi + iphi;
            const int t1 = point_base + nphi + next;

            const int tris[4][3] = {
                {b0, b1, t1},
                {b0, t1, t0},
                {bottom_center, b1, b0},
                {top_center, t0, t1}
            };

            for (const auto& tri : tris) {
                connectivity.insert(connectivity.end(), tri, tri + 3);
                next_offset += 3;
                offsets.push_back(next_offset);
                tree_id.push_back(itree);
                tree_type.push_back(type);
            }
        }
    }

    const std::string filename = output_file.empty()
        ? forest_geometry_filename(m_forestfile)
        : output_file;

    std::ofstream vtp(filename, std::ios::out | std::ios::trunc);
    if (!vtp.is_open()) {
        Abort("Cannot open forest geometry file for writing: " + filename);
    }

    vtp << std::setprecision(17);
    vtp << "<?xml version=\"1.0\"?>\n";
    vtp << "<VTKFile type=\"PolyData\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    vtp << "  <PolyData>\n";
    vtp << "    <Piece NumberOfPoints=\"" << points.size() / 3
        << "\" NumberOfVerts=\"0\" NumberOfLines=\"0\" NumberOfStrips=\"0\" NumberOfPolys=\""
        << offsets.size() << "\">\n";
    vtp << "      <Points>\n";
    vtp << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (std::size_t ip = 0; ip < points.size(); ip += 3) {
        vtp << "          " << points[ip] << ' ' << points[ip + 1] << ' ' << points[ip + 2] << '\n';
    }
    vtp << "        </DataArray>\n";
    vtp << "      </Points>\n";
    vtp << "      <Polys>\n";
    vtp << "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
    for (std::size_t ic = 0; ic < connectivity.size(); ic += 3) {
        vtp << "          " << connectivity[ic] << ' ' << connectivity[ic + 1] << ' ' << connectivity[ic + 2] << '\n';
    }
    vtp << "        </DataArray>\n";
    vtp << "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
    for (int offset : offsets) {
        vtp << "          " << offset << '\n';
    }
    vtp << "        </DataArray>\n";
    vtp << "      </Polys>\n";
    vtp << "      <CellData>\n";
    vtp << "        <DataArray type=\"Int32\" Name=\"tree_id\" format=\"ascii\">\n";
    for (int id : tree_id) {
        vtp << "          " << id << '\n';
    }
    vtp << "        </DataArray>\n";
    vtp << "        <DataArray type=\"Int32\" Name=\"tree_type\" format=\"ascii\">\n";
    for (int type : tree_type) {
        vtp << "          " << type << '\n';
    }
    vtp << "        </DataArray>\n";
    vtp << "      </CellData>\n";
    vtp << "    </Piece>\n";
    vtp << "  </PolyData>\n";
    vtp << "</VTKFile>\n";

    Print() << "Wrote forest geometry to " << filename << '\n';
}
