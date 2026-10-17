#ifndef ERF_CLASSICAD_GRID_H
#define ERF_CLASSICAD_GRID_H

#include <AMReX_Array.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_Math.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>

#include <algorithm>
#include <cmath>

/**
 * Physical vertical coordinates used by ClassicAD on a horizontally flat
 * stretched mesh.  Particle ownership continues to use the uniformly spaced
 * computational coordinate stored in Particle::pos(2).
 */
class ClassicADVerticalGrid
{
public:
    void configure (const amrex::Geometry& geom,
                    const amrex::Vector<amrex::Real>& z_faces,
                    amrex::Real hub_height,
                    bool stretched)
    {
        m_stretched = stretched;
        m_prob_lo_z = geom.ProbLo(2);
        m_dzeta = geom.CellSize(2);

        const int nz = geom.Domain().length(2);
        AMREX_ALWAYS_ASSERT(static_cast<int>(z_faces.size()) == nz + 1);
        m_faces = z_faces;
        m_centers.resize(nz);
        for (int k = 0; k < nz; ++k) {
            AMREX_ALWAYS_ASSERT(m_faces[k+1] > m_faces[k]);
            m_centers[k] = amrex::Real(0.5) * (m_faces[k] + m_faces[k+1]);
        }

        m_ground = m_faces.front();
        const amrex::Real zhub = m_ground + hub_height;
        m_hub_cell = find_cell(zhub);
        AMREX_ALWAYS_ASSERT(m_hub_cell >= 0 && m_hub_cell < nz);
        m_hub_dz = m_faces[m_hub_cell+1] - m_faces[m_hub_cell];

        m_faces_device.resize(m_faces.size());
        m_centers_device.resize(m_centers.size());
        amrex::Gpu::copy(amrex::Gpu::hostToDevice,
                         m_faces.begin(), m_faces.end(), m_faces_device.begin());
        amrex::Gpu::copy(amrex::Gpu::hostToDevice,
                         m_centers.begin(), m_centers.end(), m_centers_device.begin());
    }

    [[nodiscard]] bool stretched () const noexcept { return m_stretched; }
    [[nodiscard]] amrex::Real ground () const noexcept { return m_ground; }
    [[nodiscard]] amrex::Real hub_dz () const noexcept { return m_hub_dz; }
    [[nodiscard]] int hub_cell () const noexcept { return m_hub_cell; }
    [[nodiscard]] const amrex::Vector<amrex::Real>& faces () const noexcept { return m_faces; }
    [[nodiscard]] const amrex::Vector<amrex::Real>& centers () const noexcept { return m_centers; }
    [[nodiscard]] const amrex::Real* faces_device () const noexcept { return m_faces_device.data(); }
    [[nodiscard]] const amrex::Real* centers_device () const noexcept { return m_centers_device.data(); }

    /** A height on an interior face belongs to the cell above it. */
    [[nodiscard]] int find_cell (amrex::Real z) const noexcept
    {
        if (m_faces.size() < 2 || z < m_faces.front() || z > m_faces.back()) {
            return -1;
        }
        if (z == m_faces.back()) {
            return static_cast<int>(m_faces.size()) - 2;
        }
        return static_cast<int>(
            std::upper_bound(m_faces.begin(), m_faces.end(), z) - m_faces.begin()) - 1;
    }

    [[nodiscard]] bool cell_center_stencil_in_domain (amrex::Real z) const noexcept
    {
        const int containing_k = find_cell(z);
        if (containing_k < 0) {
            return false;
        }
        const int lower_k = z < m_centers[containing_k]
            ? containing_k-1 : containing_k;
        return lower_k >= 0 && lower_k+1 < static_cast<int>(m_centers.size());
    }

    [[nodiscard]] amrex::Real physical_to_zeta (amrex::Real z) const noexcept
    {
        if (!m_stretched) {
            return z;
        }
        const int k = find_cell(z);
        AMREX_ALWAYS_ASSERT(k >= 0);
        const amrex::Real fraction =
            (z - m_faces[k]) / (m_faces[k+1] - m_faces[k]);
        return m_prob_lo_z + (static_cast<amrex::Real>(k) + fraction) * m_dzeta;
    }

    [[nodiscard]] amrex::Real zeta_to_physical (amrex::Real zeta) const noexcept
    {
        if (!m_stretched) {
            return zeta;
        }
        const amrex::Real coordinate = (zeta-m_prob_lo_z) / m_dzeta;
        const int nz = static_cast<int>(m_faces.size()) - 1;
        AMREX_ALWAYS_ASSERT(coordinate >= amrex::Real(0.0) &&
                            coordinate <= static_cast<amrex::Real>(nz));
        if (coordinate == static_cast<amrex::Real>(nz)) {
            return m_faces.back();
        }
        const int k = static_cast<int>(std::floor(coordinate));
        const amrex::Real fraction = coordinate-static_cast<amrex::Real>(k);
        return m_faces[k] + fraction*(m_faces[k+1]-m_faces[k]);
    }

private:
    bool m_stretched{false};
    int m_hub_cell{-1};
    amrex::Real m_prob_lo_z{0.0};
    amrex::Real m_dzeta{0.0};
    amrex::Real m_ground{0.0};
    amrex::Real m_hub_dz{0.0};
    amrex::Vector<amrex::Real> m_faces;
    amrex::Vector<amrex::Real> m_centers;
    amrex::Gpu::DeviceVector<amrex::Real> m_faces_device;
    amrex::Gpu::DeviceVector<amrex::Real> m_centers_device;
};

/**
 * Interpolate one staggered mesh array at a ClassicAD particle on a flat
 * stretched mesh.  zeta determines the containing computational cell and
 * physical_z supplies the interpolation fraction in physical space.
 */
template <typename P>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::ParticleReal
classic_ad_interpolate_stretched (
    const P& p,
    amrex::ParticleReal physical_z,
    amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& plo,
    amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dxi,
    const amrex::Array4<amrex::Real const>& data,
    const amrex::IntVect& nodal,
    const amrex::Real* z_faces,
    const amrex::Real* z_centers) noexcept
{
    const amrex::Real lx = (amrex::Real(p.pos(0))-plo[0])*dxi[0]
                         - amrex::Real(!nodal[0])*amrex::Real(0.5);
    const amrex::Real ly = (amrex::Real(p.pos(1))-plo[1])*dxi[1]
                         - amrex::Real(!nodal[1])*amrex::Real(0.5);
    const int i0 = static_cast<int>(amrex::Math::floor(lx));
    const int j0 = static_cast<int>(amrex::Math::floor(ly));
    const amrex::Real fx = lx - static_cast<amrex::Real>(i0);
    const amrex::Real fy = ly - static_cast<amrex::Real>(j0);

    const amrex::Real lzeta =
        (amrex::Real(p.pos(2))-plo[2])*dxi[2];
    const int containing_k = static_cast<int>(amrex::Math::floor(lzeta));
    int k0;
    amrex::Real fz;
    if (nodal[2]) {
        k0 = containing_k;
        fz = (physical_z-z_faces[k0]) / (z_faces[k0+1]-z_faces[k0]);
    } else {
        k0 = (physical_z < z_centers[containing_k])
            ? containing_k-1 : containing_k;
        fz = (physical_z-z_centers[k0]) /
             (z_centers[k0+1]-z_centers[k0]);
    }

    const amrex::Real sx[2] = {amrex::Real(1.0)-fx, fx};
    const amrex::Real sy[2] = {amrex::Real(1.0)-fy, fy};
    const amrex::Real sz[2] = {amrex::Real(1.0)-fz, fz};
    amrex::ParticleReal value = amrex::ParticleReal(0.0);
    for (int kk = 0; kk <= 1; ++kk) {
        for (int jj = 0; jj <= 1; ++jj) {
            for (int ii = 0; ii <= 1; ++ii) {
                value += static_cast<amrex::ParticleReal>(
                    data(i0+ii,j0+jj,k0+kk) * sx[ii]*sy[jj]*sz[kk]);
            }
        }
    }
    return value;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int classic_ad_first_at_or_above (const amrex::Real* coordinates,
                                  int count,
                                  amrex::Real target) noexcept
{
    int lo = 0;
    int hi = count;
    while (lo < hi) {
        const int mid = lo + (hi-lo)/2;
        if (coordinates[mid] < target) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int classic_ad_last_at_or_below (const amrex::Real* coordinates,
                                 int count,
                                 amrex::Real target) noexcept
{
    int lo = 0;
    int hi = count;
    while (lo < hi) {
        const int mid = lo + (hi-lo)/2;
        if (coordinates[mid] <= target) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo - 1;
}

#endif
