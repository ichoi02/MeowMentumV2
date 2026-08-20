# Telemetry analysis

Run from the repository root:

```bash
python analysis/telemetry_analysis.py
```

The pipeline reads only `telemetry/raw/`, estimates impact from the acceleration
peak, evaluates each trajectory at an exact 2.5 m ballistic fall cutoff, writes
one row per usable drop, fits the statistical models, runs an overlap-stratified
permutation sensitivity analysis, and calculates minimum detectable effects and
future sample sizes.

The primary outcome is a coupled whole-cat pose score. The spherical midpoint of
the front and rear up-vectors defines whole-body tilt, scored linearly from 1
(upright) to 0 (inverted). The full relative front-to-rear rotation defines spine
alignment, also scored from 1 (aligned) to 0 (180 degrees apart). The product is
the primary score. Rear orientation is reconstructed from the front IMU and the
measured roll-pitch-roll kinematic chain. Centerline fused roll/pitch, component
scores, raw quaternion angles, and angular speed remain explanatory diagnostics.

Generated files go to `analysis/results/`. To check cutoff sensitivity without
overwriting the primary run, use a different output folder, for example:

```bash
python analysis/telemetry_analysis.py --distance 2.8 --output analysis/results_2p8m
python analysis/telemetry_analysis.py --distance 3.0 --output analysis/results_3p0m
```

`sweep_results.csv` contains two pooled tail contrasts requested for scientific
interpretation: the 180-degree-roll pitch sweep (0/15/30/45 degrees pitch) and
the no-pitch roll sweep (45/90/180 degrees roll). These are one test per sweep,
with nominal condition controlled inside each model; they are not per-condition
tests.

`time_effort_analysis.csv` contains the righting-time and actuator-effort
analyses. Righting time is the first sustained (0.10 s) crossing of a coupled
score of 0.80; trials that never cross are assigned the common observation
horizon. Thresholds 0.75 and 0.85 are included as sensitivity checks.

Electrical energy is not available from these recordings: `Cmd_*` stores joint
position targets, while applied PWM, current, and voltage were not logged. The
reported effort endpoints are explicitly model-based proxies: the time integral
of squared reconstructed normalized PD duty, once for the three actuators shared
by both morphologies and once for all four actuators including the tail.

No generated result should be described as causal unless tail/no-tail assignments
were randomized or interleaved within collection block. The present recordings
are partly confounded by date.
