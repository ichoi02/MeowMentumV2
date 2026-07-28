# MeowMentumV2

A bio-inspired **self-righting robot** — the "falling cat" reflex. A two-body robot
(front + rear) joined by an actuated spine, plus a tail, learns in simulation to
reorient itself during a short free fall (~0.74 s) so it lands upright, then that
policy is distilled and deployed to real hardware. (CMU 24-775, Bio-inspired Robot Design.)

The maneuver is a **zero-angular-momentum reorientation**: dropped at rest in an
arbitrary attitude, the robot must bend and counter-twist its spine (and swing the
tail) to rotate its body halves upright — exactly how a cat rights itself mid-air.

## Pipeline

```
  MuJoCo sim (Cat-v0)                     hardware
  ┌───────────────────┐   distill  ┌──────────────────────────────┐
  │ SAC teacher       │──DAgger──▶ │ student MLP (partial obs)    │
  │ (privileged obs)  │            │  → ONNX → Raspberry Pi       │
  └───────────────────┘            │  → serial → 2× Teensy (PD)   │
        train.py                   └──────────────────────────────┘
     cat_env/  model/                 onnx_conversion.py  hardware/
```

1. **Train** a privileged SAC *teacher* in sim on the full state (`train.py`).
2. **Distill** it (DAgger) into a *student* MLP that sees only what the real robot
   has — the front IMU + joint encoders (`distillation.py`).
3. **Export** the student to ONNX (`onnx_conversion.py`).
4. **Deploy**: a Raspberry Pi runs the ONNX policy at 50 Hz, streaming joint targets
   over serial to two Teensy 4.0 boards that close a 1 kHz PD loop (`hardware/`).

## Repository layout

| Path | Role |
|---|---|
| `cat_env/cat_env.py` | Gymnasium env `Cat-v0` (physics, obs, reward, domain randomization) |
| `cat_env/env_util.py` | PD controller, quaternion / projected-gravity / IMU-align helpers |
| `model/cat.xml` | MuJoCo model (front/rear bodies, spine joints, tail, actuators) |
| `train.py` | SAC teacher training + TensorBoard reward logging |
| `distillation.py` | DAgger distillation of teacher → student (partial obs, frame-stacked) |
| `test.py` | MuJoCo viewer for the teacher or student policy |
| `onnx_conversion.py` | Export the student `.pth` → `cat_controller.onnx` (opset 11, single file) |
| `hardware/controller.py` | Raspberry Pi control loop: read IMUs/encoders → ONNX → motor targets |
| `hardware/PD_control_{front,back}/*.ino` | Teensy 4.0 firmware: 1 kHz PD, BNO08x IMU, serial protocol |
| `hardware/e2e_test.py` | Hardware-in-the-loop test (ONNX/obs/action parity + closed-loop righting) |
| `hardware/reconstruct_viz.py` | **[Pi]** telemetry → student obs → kinematics, saved to `.npz` |
| `hardware/reconstruct_viz_view.py` | **[desktop]** replay the saved `.npz` in the MuJoCo viewer |
| `hardware/{discover_teensy,flash_pd_control_teensy}.*` | Teensy discovery + build/flash tooling |
| `hardware/{postmortem,plot_telemetry_motor_angles}.py` | Telemetry plotting |

## Simulation — `Cat-v0`

- **Model** (`model/cat.xml`): `front_body` (free root) → `rot1` (spine roll) → `pitch`
  (spine) → `rear_body` (`rot2`, roll) → `tail` (pitch). Contact disabled; 1 ms timestep,
  `frame_skip=20` → 50 Hz control; episode = 37 steps ≈ 0.74 s.
- **Observation (25-dim, yaw-invariant):** front & rear **projected gravity** (3+3),
  front & rear **gyro** (3+3), joint angles (4), joint velocities (4), applied torque
  (4), normalized step (1). Projected gravity (world −z in body frame) replaces the
  raw rotation matrix so the policy is invariant to heading.
- **Action (3-dim, in [−1,1]):** `[roll, pitch, tail]`, mapped to joint targets over
  `jnt_range`; the rear roll `rot2` is driven as `−roll` (counter-twist). A first-order
  action low-pass (`filter_alpha=0.3`) suppresses jitter. An inner **PD controller**
  (`env_util.PDController`, per-joint gains) tracks the targets each 1 ms substep — the
  sim model of the Teensy inner loop.
- **Reward** (dense, no time ramp — the discount drives "upright as soon as possible"):
  `r = w_pos·(½(up_f+up_r) + up_f·up_r) + w_bonus·1[both upright] − w_sm·‖Δa‖² − w_en·‖τ‖²`,
  with per-body uprightness `up = ½(cos(tilt)+1)`.
- **Domain randomization** (per reset): mass, damping, armature, friction, COM, inertia,
  action delay, and a uniformly random initial attitude (`init_ang_vel_max` sets an
  optional initial tumble). PD gains are **not** randomized (the sim-tuned gains are
  flashed to the hardware inner loop).

## Training & distillation

- **Teacher** (`train.py`): SAC, `MlpPolicy` `[256,256]`, 10 parallel envs, ~1M steps.
  Saves `cat_controller_<timestamp>.zip` (rename/stage as `cat_controller.zip`).
- **Student** (`distillation.py`): DAgger. The student sees only the real robot's
  sensors — **front projected gravity (3) + 4 joint angles**, **stacked over 4 timesteps
  (28-dim)** so it can infer the velocities the privileged teacher used. Trained with
  observation noise + random delay for robustness. Saves `student_policy_<timestamp>.pth`.

## Hardware

- **Boards:** two Teensy 4.0, each with a BNO08x IMU and two geared DC motors + encoders,
  running a 1 kHz PD loop. Front board → `rot1` (roll) + `pitch`; back board → `tail` +
  `rot2` (roll). Serial protocol: host sends `m1,m2` targets; boards stream
  `qr,qi,qj,qk,angle1,angle2,acc` at 50 Hz. Gear ratios: roll **9.68:1**, pitch/tail
  **34.014:1**. Teensy PD gains = sim gains × 1024 (normalized torque → PWM).
- **Pi loop** (`hardware/controller.py`): 50 Hz. Reads both IMUs/encoders → builds the
  28-dim student obs (front projected gravity + joints, 4-frame stack) → ONNX inference →
  action filter → joint-target mapping → serial. Detects free-fall (`acc < 3.5`), then
  drives for `CONTROL_DURATION = 0.74 s` and logs telemetry CSV.

## Validation tools

- `hardware/e2e_test.py` — drives MuJoCo as the "physical robot" through the **real**
  controller code + ONNX, checking dimension/ONNX/obs/action parity and closed-loop
  righting (seed-matched, the hardware path reproduces the student exactly).
- `hardware/reconstruct_viz.py` (**Pi**, MuJoCo-free) + `reconstruct_viz_view.py`
  (**desktop**) — reconstruct the robot pose from the telemetry the policy actually
  consumes and replay it in the viewer. The key signal is the **rear-IMU cross-check**
  (rear pose implied by front IMU + joints vs the independent rear IMU): ~0° = consistent;
  several degrees on hardware = a gear-ratio / IMU-alignment / spine-flex mismatch to fix
  before dropping. `--source {sim,sweep,serial}`, `--selftest` for headless validation.

## Quickstart

```bash
pip install -r requirements.txt          # sim/training deps

# --- simulation ---
python train.py                          # train the SAC teacher (-> cat_controller_*.zip)
python test.py                           # view teacher/student in MuJoCo
python distillation.py                   # DAgger distill teacher -> student_policy_*.pth
python onnx_conversion.py                # student_policy.pth -> cat_controller.onnx

# --- validation ---
python hardware/e2e_test.py                              # full pipeline parity + righting
python hardware/reconstruct_viz.py --selftest --source sweep

# --- hardware (Raspberry Pi + desktop) ---
bash hardware/flash_pd_control_teensy.sh                 # build + flash both Teensys
python hardware/controller.py                            # Pi: run the policy on the robot
python hardware/reconstruct_viz.py --source serial --out recon.npz --duration 10   # Pi: record
python hardware/reconstruct_viz_view.py recon.npz        # desktop: replay in MuJoCo
```

## notes

- **`env_util.reverse_align_imu_quaternions` is not the true inverse** of
  `controller.py::align_imu_quaternions` (one-sided vs two-sided transform); the
  reconstruction tools define the correct inverse locally.
