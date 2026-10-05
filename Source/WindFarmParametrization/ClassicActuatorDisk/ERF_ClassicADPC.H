#ifndef ERF_CLASSICADPC_H
#define ERF_CLASSICADPC_H

#if defined(ERF_USE_PARTICLES) && defined(ERF_USE_WINDFARM)

#include <AMReX_Particles.H>

#include <ERF_ClassicAD.H>

struct ClassicADRealIdx
{
    enum { radius = 0, theta, area, force_x, force_y, force_z, ncomps };
};

struct ClassicADIntIdx
{
    enum { turbine = 0, ring, element, ncomps };
};

class ClassicADPC
    : public amrex::ParticleContainer<0, 0,
                                      ClassicADRealIdx::ncomps,
                                      ClassicADIntIdx::ncomps>
{
public:
    using Base = amrex::ParticleContainer<0, 0,
                                          ClassicADRealIdx::ncomps,
                                          ClassicADIntIdx::ncomps>;

    ClassicADPC (amrex::ParGDBBase* gdb, ClassicAD& model)
        : Base(gdb), m_model(model)
    {}

    void rebuild (const amrex::Vector<int>& owner_levels);

private:
    ClassicAD& m_model;
    amrex::Vector<int> m_owner_levels;
};

#endif
#endif
