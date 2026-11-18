# ClassicAD integration smoke test

This short, single-turbine case exercises the upstream-compatible ClassicAD
path without turbine refinement, forest geometry, or RMask. It covers
wind-farm catalog loading, owner-level assignment, actuator and sensor particle
construction, filtered disk velocity, startup/ramp behavior, dynamic yaw, wake
rotation, force deposition, and checkpoint/restart state.

Per-turbine physical forcing can optionally be scheduled with
`erf.classic_ad.forcing_schedule`.  Its four whitespace-delimited columns are
`turbine_id start_time end_time forcing_fraction`; comments begin with `#`.
The zero-based turbine starts commanded fully on, and each row changes its
persistent command at `end_time`.  Repeated, non-overlapping rows provide
multiple off/on or fractional-loading periods.  The row `start_time` must be
at least `erf.windfarm_start_time`.  `erf.classic_ad.derate_timescale` controls
the exponential transition to each new command and defaults to an
instantaneous transition.

Build ERF with wind-farm and particle support, then run from a fresh directory:

```bash
ERF_EXE=/path/to/erf_exec
CASE_DIR=/path/to/ERF/Exec/RegTests/ClassicAD

"${ERF_EXE}" "${CASE_DIR}/inputs" \
  erf.windfarm_loc_table="${CASE_DIR}/classic_ad_locations.txt"
python3 "${CASE_DIR}/validate_checkpoint.py" .
```

The same test on a horizontally flat prescribed-z mesh uses
``inputs_stretched``.  Its levels remain uniform through the rotor and the
three-sigma force support, then stretch above it:

```bash
"${ERF_EXE}" "${CASE_DIR}/inputs_stretched" \
  erf.windfarm_loc_table="${CASE_DIR}/classic_ad_locations.txt"
python3 "${CASE_DIR}/validate_checkpoint.py" .
```

The run should complete three time steps and create `chk00002` containing both
`ClassicADMemoryState` and `ClassicADYawState`. To exercise restoration of
both state files, continue that checkpoint for one additional step:

```bash
"${ERF_EXE}" "${CASE_DIR}/inputs" \
  erf.windfarm_loc_table="${CASE_DIR}/classic_ad_locations.txt" \
  amr.restart=chk00002 max_step=3 erf.check_int=-1
```

The validator checks the configured rotor-power, angular-velocity, and torque
equations, the startup-ramped applied torque, the per-turbine forcing
diagnostics, and the deposited LES torque. The
model's runtime checks also verify actuator/sensor area, rigid sensor-disk
geometry, finite-element thrust and torque, tangential-force symmetry, and
discrete Gaussian force conservation.
