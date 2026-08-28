#ifndef ERF_CLASSICADSENSORPC_H
#define ERF_CLASSICADSENSORPC_H

#if defined(ERF_USE_PARTICLES) && defined(ERF_USE_WINDFARM)

#include <AMReX_Particles.H>
#include <ERF_ClassicAD.H>

struct ClassicADSensorRealIdx
{
    enum { radius = 0, theta, area, ncomps };
};

struct ClassicADSensorIntIdx
{
    enum { turbine = 0, ring, element, ncomps };
};

class ClassicADSensorPC
    : public amrex::ParticleContainer<0, 0,
                                      ClassicADSensorRealIdx::ncomps,
                                      ClassicADSensorIntIdx::ncomps>
{
public:
    using Base = amrex::ParticleContainer<0, 0,
                                          ClassicADSensorRealIdx::ncomps,
                                          ClassicADSensorIntIdx::ncomps>;

    ClassicADSensorPC (amrex::ParGDBBase* gdb, ClassicAD& model)
        : Base(gdb), m_model(model) {}

    void rebuild ();
    void update_positions ();
    void sample_uv (int lev,
                    const amrex::MultiFab& u,
                    const amrex::MultiFab& v,
                    const amrex::MultiFab& w,
                    amrex::Vector<amrex::Real>& u_mean,
                    amrex::Vector<amrex::Real>& v_mean) const;

private:
    ClassicAD& m_model;
};

#endif
#endif
