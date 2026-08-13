# MeowMentumV2

A bio-inspired **self-righting robot** — the "falling cat" reflex. A two-body robot
(front + rear) joined by an actuated spine, plus a tail, learns in simulation to
reorient itself during a **1.0 s** free fall so it lands upright; that policy is then
distilled and deployed to real hardware. (CMU 24-775, Bio-inspired Robot Design.)

The maneuver is a **zero-angular-momentum reorientation**: released in an arbitrary
attitude, the robot bends and counter-twists its spine (and swings the tail) to rotate
its body halves upright — how a cat rights itself mid-air. The study is a **morphology
ablation**: `tail` vs `notail`, scored as both-bodies-upright success rate.

## Pipeline

```
  MuJoCo sim (Cat-v0 / CatNoTail-v0)          hardware
  ┌────────────────────────┐  distill  ┌──────────────────────────────┐
  │ SmoothSAC teacher      │──DAgger─▶ │ student MLP (11-dim obs)     │
  │ (73-dim privileged obs)│           │  → ONNX → Raspberry Pi 50 Hz │
  └────────────────────────┘           │  → serial → 2× Teensy (1 kHz)│
      train.py  cat_env/  model/       └──────────────────────────────┘
                                        distillation.py  onnx_conversion.py  hardware/
```

## Layout

| Path | Role |
|---|---|
| `cat_env/cat_env.py` | Gymnasium env `Cat-v0` — physics, obs, reward, domain randomization |
| `cat_env/env_util.py` | PD controller (Teensy inner loop, step for step), projected-gravity helpers |
| `model/cat{,_notail}.xml` | MuJoCo models; the no-tail arm keeps the joint but starves the motor |
| `train.py` | SAC teacher training (`SmoothSAC`) + TensorBoard reward logging |
| `smooth_sac.py` | SAC with the action-smoothness term in the **actor loss** |
| `distillation.py` | DAgger teacher → student (partial obs); defines the student obs layout |
| `onnx_conversion.py` | Student `.pth` → `policies/cat_controller<suffix>.onnx` (single file, opset 11) |
| `test.py` | MuJoCo viewer for a teacher or student policy |
| `policies/` | **All trained artifacts**; paths come from `variants.py`, nothing hardcodes a location |
| `docs/evaluate.py` | Closed-loop success rate + penalty-magnitude diagnostics (`--stats`) |
| `docs/sweep.py`, `docs/sweep_jobs/` | Parallel reward-weight sweep driver → JSONL |
| `docs/{EVALUATION,REWARD_TUNING}.md` | Ablation results; how the weights were tuned, with the data |
| `tools/tune_pd_gains.py` | Step-response tuner for the inner-loop PD gains |
| `plots/plot_joint_tracking.py` | Commanded vs filtered vs achieved joint angles, per drop |
| `hardware/controller.py` | Pi control loop: IMUs/encoders → ONNX → joint targets → serial |
| `hardware/PD_control_{front,back}/*.ino` | Teensy 4.0 firmware: 1 kHz PD, BNO08x IMU, serial protocol |
| `hardware/e2e_test.py` | Hardware-in-the-loop parity test (no physical robot needed) |
| `hardware/reconstruct_viz{,_view}.py` | **[Pi]** telemetry → pose `.npz`; **[desktop]** replay it in MuJoCo |
| `hardware/{discover_teensy.py,flash_pd_control_teensy.sh}` | Teensy discovery + build/flash |
| `experiments/`, `telemetry/` | Logged hardware drops (CSV) and sim2real analysis |

## Simulation

- **Model**: `front_body` (free root) → `rot1` (spine roll) → `pitch` → `rear_body`
  (`rot2`, roll) → `tail` (pitch). Contact disabled; 1 ms timestep, `frame_skip=20`
  → 50 Hz control; episode = **50 steps = 1.0 s** (a 4.9 m drop).
- **Observation (73-dim, yaw-invariant)**: front/rear **projected gravity** (3+3),
  front/rear gyro (3+3), joint angles (4), joint velocities (4), applied torque (4),
  normalized step (1), plus a **privileged DR block (48)** carrying this episode's
  actual randomization draw. The block is teacher-only and sits **last**, so student
  slices never move; `privileged=False` gives the old 25-dim space back.
- **Action (3-dim, [−1,1])**: `[roll, pitch, tail]`. Passed through a per-channel
  first-order low-pass (`filter_alpha = [0.1, 0.3, 0.1]`, sized against measured
  hardware travel), then mapped to joint targets; `rot2` is driven as `−roll`. An
  inner PD loop tracks the targets every 1 ms, reproducing the Teensy's encoder
  quantization, velocity filter, deadband and minimum-PWM floor.
- **Reward** (dense, per-step, no time ramp — the discount drives "upright ASAP"):

  ```
  r = w_pos·(½(up_f + up_r) + up_f·up_r) − w_en·mean(τ²) − w_av·mean(ω²) − w_jv·mean(q̇²) − w_time·Σ_{i<j}|Δaᵢ||Δaⱼ|
  ```

  with per-body uprightness **linear** in tilt, `up = 1 − tilt/π`, and `w_pos = 1.5`.
  There is no success bonus (removed: it was morphology-biased). Penalties price
  torque, body angular velocity, joint velocity and simultaneous multi-joint motion —
  all hardware-facing. Both variants **share one weight vector**
  (`cat_env.py::PENALTY_WEIGHTS`, currently `en 0.03859 / av 0.002123 / jv 0.0003087 /
  time 0`) because a morphology ablation must not also vary the reward. `w_time` is 0:
  the action low-pass absorbs raw action rate, so the term priced a quantity the robot
  does not feel. Each `info` also reports the **unweighted** magnitude `m_*` beside
  `r_*` — that, not the weighted term, is what tuning reads. Overridable as
  `CAT_W_{POS,EN,AV,JV,TIME}` without a code edit; see `docs/REWARD_TUNING.md`.
- **Action smoothness is an actor-loss term, not a reward penalty**
  (`smooth_sac.py`): the actor minimizes `smooth_coef·‖π_μ(s_{t+1}) − π_μ(s_t)‖²` over
  the **deterministic mean** action — what `deterministic=True`, the ONNX export and
  the Teensy actually command. As a reward it traded against task reward inside the
  same scalar `Q` and made passivity look optimal; here the gradient reaches the actor
  directly. Logged as `train/smooth_loss` at any coefficient, including 0.
- **Release distribution**: full roll (±180°), pitch ±45°, yaw fixed at 0 (everything
  is yaw-invariant), initial tumble ±0.5 rad/s per axis, initial joint angles ±0.2 rad.
- **Domain randomization** (per reset): body mass, COM, inertia; action delay 0–2 steps;
  and per **motor group** — damping, armature, friction, `ctrlrange` and PD gains.
  `rot1`/`rot2` are the same 9.68:1 motor and `pitch`/`tail` the same 34.014:1, so
  identical hardware draws identical multipliers rather than a robot that cannot exist.

## Training & distillation

- **Teacher** (`train.py`): `SmoothSAC`, `MlpPolicy [256,256]`, 10 parallel envs, ~1M
  steps. `--smooth-coef` sets the smoothness weight (default 2.0; `0` = plain SAC).
  Checkpoints stay loadable by plain `SAC.load` — only the objective changes. Saves
  `policies/cat_controller<suffix><tag>_<timestamp>.zip`; stage it as
  `policies/cat_controller<suffix>.zip`, which is what every script loads by default.
- **Student** (`distillation.py`): DAgger, 50 iterations × 2000 steps. Sees only what
  the robot has — **11 dims**: front projected gravity (3), joint angles (4), the
  **previous action** (3) and episode progress (1). The previous action is required,
  not decorative: the action low-pass is persistent state, so without command history
  the student's problem is not Markov. Trained with per-episode sensor bias, per-frame
  noise and random observation delay. `--smooth-coef` adds the student-side smoothness
  term (default 0). Saves `policies/student_policy<suffix>_<timestamp>.pth`.
- **No-tail arm**: `cat_notail.xml` keeps the tail joint but caps its motor at ±1e-6
  torque, so `action[2]` gets no gradient. The tail joint reading **and** the tail
  action are forced to 0 in the sim frame, in `docs/evaluate.py` and in
  `hardware/controller.py` alike — pinning only one side would leave the two apart.

## Hardware

- **Boards**: two Teensy 4.0, each with a BNO08x IMU and two geared DC motors +
  encoders, running a 1 kHz PD loop. Front → `rot1` + `pitch`; back → `tail` + `rot2`.
  Serial: host sends `m1,m2` targets (4 dp, rad); boards stream
  `qr,qi,qj,qk,angle1,angle2,acc` at 50 Hz. Teensy PD gains = sim gains × 1024.
- **Pi loop** (`hardware/controller.py`): 50 Hz. Both IMUs/encoders → 11-dim student
  obs → ONNX → per-channel low-pass → joint targets → serial. The filter, the mapping
  and the previous-action bookkeeping mirror `cat_env.step` exactly, in the same order.
  Detects free fall (`acc < 3.5`), drives for `CONTROL_DURATION = 1.0 s`, logs CSV.
- **Validation**: `hardware/e2e_test.py` drives MuJoCo as the "physical robot" through
  the real controller code + ONNX (dimension/ONNX/obs/action parity, then closed-loop
  righting). `reconstruct_viz.py` rebuilds the pose from the telemetry the policy
  actually consumes; its key signal is the **rear-IMU cross-check** — rear pose implied
  by front IMU + joints vs the independent rear IMU. ~0° = consistent; several degrees
  on hardware = a gear-ratio / IMU-alignment / spine-flex mismatch to fix before a drop.

## Quickstart

```bash
pip install -r requirements.txt

# --- simulation ---  (add --variant notail for the ablation arm)
python train.py                          # SAC teacher  -> policies/cat_controller*.zip
python distillation.py                   # DAgger       -> policies/student_policy*.pth
python onnx_conversion.py                # student .pth -> policies/cat_controller*.onnx
python test.py --agent student --roll 180        # watch one policy in MuJoCo

# --- evaluation ---
python docs/evaluate.py --agent teacher --episodes 300 --stats
python docs/evaluate.py --roll 180 --episodes 300     # fixed release attitude
python plots/plot_joint_tracking.py --agent student --deg

# --- validation ---
python hardware/e2e_test.py
python hardware/reconstruct_viz.py --selftest --source sweep

# --- hardware (Raspberry Pi + desktop) ---
bash hardware/flash_pd_control_teensy.sh                 # build + flash both Teensys
python hardware/controller.py --variant tail             # Pi: fly the policy
python hardware/reconstruct_viz.py --source serial --out recon.npz --duration 10
python hardware/reconstruct_viz_view.py recon.npz        # desktop: replay
```

## Notes

- Success is scored as **both bodies within 30° of upright** at the end of the drop
  (`docs/evaluate.py::UPRIGHT_DEG`).
- `build_student_obs` is duplicated in `distillation.py` and `hardware/controller.py`
  by design (the Pi must not import torch/MuJoCo). They **must** stay identical;
  `e2e_test.py` is what catches a drift.
- `env_util.reverse_align_imu_quaternions` is **not** the true inverse of
  `controller.py::align_imu_quaternions` (one-sided vs two-sided); the reconstruction
  tools define the correct inverse locally.
- `visualize.py` (repo root) is a legacy telemetry replay with a hardcoded absolute
  path to an older checkout — use `hardware/reconstruct_viz{,_view}.py` instead.
