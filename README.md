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
| `onnx_conversion.py` | Export the student `.pth` → `policies/cat_controller.onnx` (opset 11, single file) |
| `policies/` | **All trained artifacts**: teacher `.zip`, student `.pth`, exported `.onnx`. Paths come from `variants.py` (`teacher_path` / `student_path` / `onnx_path`), so no script hardcodes a location |
| `hardware/controller.py` | Raspberry Pi control loop: read IMUs/encoders → ONNX → motor targets |
| `hardware/PD_control_{front,back}/*.ino` | Teensy 4.0 firmware: 1 kHz PD, BNO08x IMU, serial protocol |
| `hardware/e2e_test.py` | Hardware-in-the-loop test (ONNX/obs/action parity + closed-loop righting) |
| `hardware/reconstruct_viz.py` | **[Pi]** telemetry → student obs → kinematics, saved to `.npz` |
| `hardware/reconstruct_viz_view.py` | **[desktop]** replay the saved `.npz` in the MuJoCo viewer |
| `hardware/{discover_teensy,flash_pd_control_teensy}.*` | Teensy discovery + build/flash tooling |
| `hardware/{postmortem,plot_telemetry_motor_angles}.py` | Telemetry plotting |
| `docs/evaluate.py` | Closed-loop success rate + penalty-magnitude diagnostics (`--stats`) |
| `docs/sweep.py` | Parallel reward-weight sweep driver (train → evaluate → JSONL) |
| `docs/REWARD_TUNING.md` | How the five penalty weights were tuned, and the sweep data |

## Simulation — `Cat-v0`

- **Model** (`model/cat.xml`): `front_body` (free root) → `rot1` (spine roll) → `pitch`
  (spine) → `rear_body` (`rot2`, roll) → `tail` (pitch). Contact disabled; 1 ms timestep,
  `frame_skip=20` → 50 Hz control; episode = 37 steps ≈ 0.74 s.
- **Observation (73-dim, yaw-invariant):** front & rear **projected gravity** (3+3),
  front & rear **gyro** (3+3), joint angles (4), joint velocities (4), applied torque
  (4), normalized step (1), plus a **privileged DR block** (48) carrying this episode's
  actual randomization draw — per-body mass / COM / inertia, per-motor-group damping,
  armature, friction, torque limit and PD gains, and the action delay, each normalized
  to ~[−1,1] with nominal at 0. Projected gravity (world −z in body frame) replaces the
  raw rotation matrix so the policy is invariant to heading. The DR block is
  teacher-only and sits **last**, so the student slices are unaffected; construct the
  env with `privileged=False` to get the old 25-dim space back for pre-DR checkpoints.
- **Action (3-dim, in [−1,1]):** `[roll, pitch, tail]`, mapped to joint targets over
  `jnt_range`; the rear roll `rot2` is driven as `−roll` (counter-twist). An inner **PD controller**
  (`env_util.PDController`, per-joint gains) tracks the targets each 1 ms substep — the
  sim model of the Teensy inner loop.
- **Reward** (dense, no time ramp — the discount drives "upright as soon as possible"):
  `r = w_pos·(½(up_f+up_r) + up_f·up_r) + w_bonus·1[both upright] − Σ penalties`,
  with per-body uprightness `up = ½(cos(tilt)+1)`. Four penalties, all hardware-facing:

  | term | penalizes | why |
  |---|---|---|
  | `w_en·mean(τ²)` | applied torque | energy |
  | `w_av·mean(ω²)` | body angular velocity | landing upright but still tumbling |
  | `w_jv·mean(q̇²)` | joint velocity | motion that does not buy reorientation |
  | `w_time·Σ_{i<j}\|Δaᵢ\|\|Δaⱼ\|` | joints moving *simultaneously* | a step-by-step maneuver stays in states the sim models well |

  Each `info` dict also reports the **unweighted** magnitude `m_*` beside `r_*` — that,
  not the weighted term, is what reward tuning reads. See `docs/REWARD_TUNING.md`;
  weights are overridable as `CAT_W_{EN,AV,JV,TIME}` without a code edit.

  Both variants **share one weight vector** (`cat_env.py::PENALTY_WEIGHTS`) — `en` 0.65,
  `av` 0.0055, `jv` 0.0008, `time` 0.85 — because this is a *morphology* ablation: if the
  arms run different rewards, the tail-vs-no-tail gap mixes the tail's contribution with a
  reward-budget difference. Found by scaling the old per-variant tail vector by a single
  `k` and scoring no-tail (the binding arm) on fixed releases at roll 180/90/45/0, pitch 0
  — mean success rose monotonically 47.2% at `k`=0 to 58.6% at `k`=1
  (`docs/shared_k_runs.jsonl`). Two caveats: the vector costs the trained no-tail policy
  only **~12% of task reward**, not the ~43% the same numbers cost when they were tuned
  (moving action rate to the actor loss collapsed the `m_time` baseline 7x, so the old
  ~18–30% collapse ceiling was never reached anywhere in the sweep); and the mean hides a
  **monotone regression at roll 180**, 24.0% at `k`=0 down to 16.3% at `k`=1, while roll 90
  climbs 34.0% → 72.3%. Not yet validated on the tail arm. Historical per-variant tuning,
  including the budgets those weights came from, is in `docs/REWARD_TUNING.md`.
- **Action smoothness is an actor-loss term, not a reward penalty**
  (`smooth_sac.py::SmoothSAC`). The old `w_sm·mean(Δa²)` is gone; the actor now
  minimizes `smooth_coef·‖π_μ(s_{t+1}) − π_μ(s_t)‖²` over consecutive replay states,
  where `π_μ = tanh(μ)` is the **deterministic mean** — what `deterministic=True`, the
  ONNX export and the Teensy actually command — rather than a sample carrying SAC's
  exploration noise. Penalizing the sample would price jitter the deployed policy does
  not have and would let the actor satisfy the term by shrinking `log_std`, fighting the
  `ent_coef` auto-tuner. As a reward it also had to be routed through the critic and
  traded against task reward inside the same scalar `Q`, which is what made `w_sm` a
  collapse risk; here the gradient reaches the actor directly. Logged as
  `train/smooth_loss` at any coefficient (including 0), same convention as `m_*`.
- **Domain randomization** (per reset): mass, COM, inertia, action delay, initial joint
  angles (±0.2 rad), and a uniformly random initial attitude (`init_ang_vel_max` sets an
  optional initial tumble). Damping, armature, friction, `ctrlrange` and the PD gains are
  drawn **once per motor group** — `rot1`/`rot2` are the same 9.68:1 roll motor and
  `pitch`/`tail` the same 34.014:1 motor, so identical hardware gets identical
  multipliers rather than independent draws the real robot could never exhibit.

## Training & distillation

- **Teacher** (`train.py`): `SmoothSAC` (SAC + the actor smoothness term above),
  `MlpPolicy` `[256,256]`, 10 parallel envs, ~1M steps. `--smooth-coef` sets the
  coefficient (default 10.0, sized so the term carries the pressure the old `w_sm` did
  — derivation in `smooth_sac.py`; `0` gives plain SAC). Checkpoints stay loadable by
  plain `SAC.load`, since only the training objective changes, not the policy. Saves
  `policies/cat_controller_<timestamp>.zip` (rename/stage as
  `policies/cat_controller.zip`, which is what every script loads by default).
- **Student** (`distillation.py`): DAgger. The student sees only the real robot's
  sensors — **front projected gravity (3) + 4 joint angles**, **stacked over 2 timesteps
  (14-dim, `distillation.N_FRAMES`)** so it can infer the velocities the privileged
  teacher used; `--frames` changes it and tags the filename. Trained with
  observation noise + random delay for robustness. Saves
  `policies/student_policy_<timestamp>.pth` (stage as `policies/student_policy.pth`).

## Hardware

- **Boards:** two Teensy 4.0, each with a BNO08x IMU and two geared DC motors + encoders,
  running a 1 kHz PD loop. Front board → `rot1` (roll) + `pitch`; back board → `tail` +
  `rot2` (roll). Serial protocol: host sends `m1,m2` targets; boards stream
  `qr,qi,qj,qk,angle1,angle2,acc` at 50 Hz. Gear ratios: roll **9.68:1**, pitch/tail
  **34.014:1**. Teensy PD gains = sim gains × 1024 (normalized torque → PWM).
- **Pi loop** (`hardware/controller.py`): 50 Hz. Reads both IMUs/encoders → builds the
  14-dim student obs (front projected gravity + joints, 2-frame stack) → ONNX inference →
  joint-target mapping → serial. The policy output is commanded as-is: there is no action
  filter, so smoothness comes from the actor-loss term rather than the control path, and
  `e2e_test.py` checks the mapping against the sim's executed target. Detects free-fall
  (`acc < 3.5`), then drives for `CONTROL_DURATION = 0.74 s` and logs telemetry CSV.

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
python train.py                          # SAC teacher  -> policies/cat_controller_*.zip
python test.py                           # view teacher/student in MuJoCo
python distillation.py                   # DAgger        -> policies/student_policy_*.pth
python onnx_conversion.py                # student .pth  -> policies/cat_controller.onnx

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
