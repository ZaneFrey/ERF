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
