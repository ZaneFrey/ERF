# ClassicAD smoke regression

This two-step, uniform-flow case exercises particle construction, staggered
velocity/density sampling, axial and rotational loading, Gaussian deposition,
and slow-RHS coupling. The 16-box layout also exercises deposition across
BoxArray boundaries. It intentionally omits `erf.windfarm_spec_table`.

Run it with an ERF executable configured with both wind farms and particles.
The `power_output_ClassicAD.txt` diagnostics should start with `Ud_raw = 10`,
`rho_d` near `1.1520116233`, target thrust near `217149.0752 N`, and target
torque near `488585.4191 N m`. The model's runtime conservation checks cover
actuator area, finite-element thrust and torque, transverse-force symmetry,
and volume-integrated staggered Gaussian force. `Q_LES` reports the first
moment after smoothing; for this grid it is about `-485392.6673 N m`.

## ClassicAD V1.1 dynamic yaw

`inputs_dynamic_aligned` is the no-op baseline. `inputs_dynamic_response`
starts the rotor normal along x in a uniform 30-degree wind and checks the
exact exponential mechanical response. `inputs_dynamic_boundary` places the
turbine near a periodic boundary so both disks cross AMReX boxes/ranks while
yawing to 45 degrees. `inputs_dynamic_turbulent` exercises sensor low-pass
filtering with a perturbed velocity field. `classic_ad_yaw_offset_15.txt` is a
rotor-only manual offset: the sensor remains at the reference angle and the
rotor is always 15 degrees beyond it.

The controller uses, once per large timestep,

```text
alpha = 1 - exp(-dt / yaw_sensor_mem_time)
U_f   = U_f + alpha (U_raw - U_f)
V_f   = V_f + alpha (V_raw - V_f)
phi   = atan2(V_f, U_f)
psi_ref   = wrap_pi(psi_ref + alpha wrap_pi(phi - psi_ref))
psi_rotor = wrap_pi(psi_ref + yaw_file_offset)
```

The first valid sensor sample initializes `U_f,V_f` directly. A zero memory
time sets `alpha=1`. Dynamic ClassicAD does not use or require
`erf.yaw_period`.

Run a one-rank CPU response test from a fresh directory with:

```bash
/home/zane/ERF/build_classic_validation_cpu/Exec_dev/WindFarmTests/erf_windfarmtest \
  /home/zane/ERF/Exec/RegTests/ClassicAD/inputs_dynamic_response \
  erf.windfarm_loc_table=/home/zane/ERF/Exec/RegTests/ClassicAD/classic_ad_locations.txt
python3 /home/zane/ERF/Exec/RegTests/ClassicAD/validate_dynamic_yaw.py . \
  --tau 1.0 --dt 0.02 --require-checkpoint
```

For CUDA, replace the executable with the one under
`build_classic_validation_cuda` and set `CUDA_VISIBLE_DEVICES=0`.
Validate the perturbed case with
`--tau 0.05 --dt 0.005 --require-filtering`.

The practical V1.1 regression matrix is:

1. aligned flow (no yaw drift);
2. angled flow (analytical mechanical response);
3. naturally evolving sensor samples (analytical sensor-filter recurrence);
4. +15-degree manual offset (sensor/reference and rotor separation);
5. 15-, 45-, and 90-degree rigid rotations via uniform-flow overrides;
6. periodic box/rank crossing with `inputs_dynamic_boundary`;
7. force/torque conservation during each of the above (runtime assertions);
8. perturbed inflow using `prob.U_0_Pert_Mag`/`prob.V_0_Pert_Mag` overrides;
9. one- versus four-rank comparison of the boundary case;
10. CPU/CUDA diagnostic comparison;
11. continuous versus checkpoint/restart comparison (the checkpoint must
    contain both ClassicAD state files);
12. guard tests: `amr.max_level=1`, non-flat terrain, invalid sensor settings,
    and simultaneous `yaw_out_per`/`yaw_out_int` must exit nonzero.
