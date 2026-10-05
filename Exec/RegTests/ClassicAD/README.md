# ClassicAD integration smoke test

This short, single-turbine case exercises the upstream-compatible ClassicAD
path without turbine refinement, forest geometry, RMask, or wake rotation. It
covers wind-farm catalog loading, owner-level assignment, actuator and sensor
particle construction, filtered disk velocity, startup/ramp behavior, dynamic
yaw, force deposition, and checkpoint/restart state.

Build ERF with wind-farm and particle support, then run from a fresh directory:

```bash
ERF_EXE=/path/to/erf_exec
CASE_DIR=/path/to/ERF/Exec/RegTests/ClassicAD

"${ERF_EXE}" "${CASE_DIR}/inputs" \
  erf.windfarm_loc_table="${CASE_DIR}/classic_ad_locations.txt"
python3 "${CASE_DIR}/validate_checkpoint.py" .
```

The run should complete two time steps and create `chk00002` containing both
`ClassicADMemoryState` and `ClassicADYawState`. To exercise restoration of
both state files, continue that checkpoint for one additional step:

```bash
"${ERF_EXE}" "${CASE_DIR}/inputs" \
  erf.windfarm_loc_table="${CASE_DIR}/classic_ad_locations.txt" \
  amr.restart=chk00002 max_step=3 erf.check_int=-1
```

The model's runtime checks also verify actuator/sensor area, rigid sensor-disk
geometry, and discrete Gaussian force conservation.
