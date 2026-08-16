import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from stable_baselines3 import SAC
from scipy.spatial.transform import Rotation as R
import cat_env
import cat_env.env_util as util
from cat_env.cat_env import BASE_OBS_DIM
import os
import time
from variants import VARIANTS, policy_dir, teacher_path

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# Full obs layout (see cat_env/cat_env.py::_get_obs; 73-dim, yaw-invariant):
#   [0:3]    front_proj_grav   (front-body gravity direction; front IMU on real robot)
#   [3:6]    rear_proj_grav
#   [6:9]    front_gyro
#   [9:12]   rear_gyro
#   [12:16]  joint angles (rot1, pitch, rot2, tail)
#   [16:20]  joint velocities
#   [20:24]  ctrl
#   [24]     step
#   [25:73]  privileged DR block (this episode's mass/COM/inertia/motor draw)
# The real robot has only the FRONT IMU + joint encoders, so the student sees
# front_proj_grav (yaw-invariant, matching the teacher) + joint angles. The DR
# block is teacher-only and deliberately unreachable from the slices below: the
# student has to recover the plant from how it responded, which is the whole
# point of distilling a privileged teacher rather than deploying it.
FRONT_GRAV_SLICE = slice(0, 3)
JOINT_ANGLE_SLICE = slice(12, 16)
STEP_IDX = 24                           # normalized progress, steps / max_steps

GRAV_DIM = 3
JOINT_DIM = 4
ACT_DIM = 3
FRAME_DIM = GRAV_DIM + JOINT_DIM        # single-timestep kinematic features (7)

# Student observation: the CURRENT kinematic frame, the previous action, and how far
# through the episode we are.
#
#   [0:3]    front projected gravity
#   [3:7]    joint angles (rot1, pitch, rot2, tail)
#   [7:10]   previous action (normalized, pre-filter -- what the policy last commanded)
#   [10]     step / max_steps
#
# Replaces the old 2-frame difference stack. Two reasons the previous action has to be
# here: the action low-pass in cat_env.step is persistent state, so without the command
# history the filter's position is unobservable and the student's problem is not Markov;
# and the teacher has always seen `step` (cat_env::_get_obs), so including it closes a
# teacher/student asymmetry rather than inventing a new signal.
#
# The trade this makes: the frame differences were the only velocity cue the student
# had, and they were measured to be well inside the sim's distribution on hardware
# (d_grav 0.87-0.98x sim sd, 1-5% out of support -- telemetry/sim2real.ipynb). Dropping
# them is a deliberate choice to rely on the previous action instead; if the student
# regresses relative to its teacher, this is the first thing to revisit.
STUDENT_OBS_DIM = FRAME_DIM + ACT_DIM + 1   # 11

# Zero the tail channel for the no-tail variant. `cat_notail.xml` does not remove the
# tail joint, it starves the motor to ctrlrange +-1e-6, so `action[2]` is inert in
# training and the policy gets no gradient on it. On hardware the same channel is wired
# to a real +-110 deg motor, which drove `q_tail`/`dq_tail` 76%/81% outside the
# distribution the student was distilled on. Forcing the tail to zero in BOTH the sim
# frame and hardware/controller.py keeps the two identical -- pinning only the hardware
# would leave it at the mode of the sim's distribution rather than matching it.
TAIL_JOINT_IDX = 3


def zero_tail(frame_or_obs, variant, joint_offset=GRAV_DIM):
    """No-tail variant: force the tail joint reading to 0. Identity for the tail arm."""
    if variant == "notail":
        frame_or_obs = np.asarray(frame_or_obs, dtype=float).copy()
        frame_or_obs[joint_offset + TAIL_JOINT_IDX] = 0.0
    return frame_or_obs


def build_student_obs(frame, prev_action, step_frac):
    """[grav(3), joints(4)] + prev action(3) + step(1) -> the 11-dim student input.

    MUST match hardware/controller.py::build_student_obs exactly.
    """
    return np.concatenate([frame, np.asarray(prev_action, dtype=float).ravel(),
                           [float(step_frac)]])


# --- legacy, kept so telemetry/sim2real.ipynb can still parse the pre-redesign logs ---
N_FRAMES = 2                            # frames the OLD stacked observation used

def sample_sensor_bias(grav_bias_std=0.035, joint_bias_std=0.02):
    """Constant per-drop sensor offsets, drawn once per episode.

    Real sensor error is dominated by fixed offsets, not per-frame noise: the IMU
    is bolted on with a few degrees of misalignment, and the encoders are zeroed
    with the robot not exactly at its nominal pose. Both hold for a whole drop, so
    zero-mean per-frame noise does not cover them -- the network can average that
    away, but it cannot average away a bias that shifts every frame identically.
    """
    return {
        "grav": np.random.normal(0.0, grav_bias_std, GRAV_DIM),    # rotvec, rad
        "joint": np.random.normal(0.0, joint_bias_std, JOINT_DIM),
    }

def get_noisy_student_frame(full_obs, bias, grav_noise_std=0.02, joint_noise_std=0.02):
    """One timestep of what the student is allowed to see: [front_proj_grav(3), joint_angles(4)]."""
    front_grav = full_obs[FRONT_GRAV_SLICE].copy()
    joint_angles = full_obs[JOINT_ANGLE_SLICE].copy()

    # IMU error: a constant mounting misalignment plus per-frame tilt noise, both
    # applied as a rotation of the measured gravity direction.
    grav_rotvec = bias["grav"] + np.random.normal(0.0, grav_noise_std, GRAV_DIM)
    noisy_grav = R.from_rotvec(grav_rotvec).apply(front_grav)

    # Encoder error: a constant zeroing offset plus per-frame noise.
    noisy_joints = util.add_gaussian_noise(joint_angles + bias["joint"], joint_noise_std)

    return np.concatenate([noisy_grav, noisy_joints])

def stack_frames(frames):
    """LEGACY -- the pre-redesign stacked observation. Not used by the current student.

    Kept because telemetry/sim2real.ipynb rebuilds the observations of the drops that
    were flown with the 14-dim policy, and those logs are only interpretable through
    the layout that produced them.

    Newest frame plus successive backward DIFFERENCES, grouped by feature type.

    With the default 2 frames the layout is [current, current - previous] rather
    than [previous, current]: the same 14 numbers and the same information, but the
    velocity-like part is handed over directly instead of leaving the network to
    subtract two nearly-equal inputs itself. The two raw frames differ by ~1 part in
    100 over a 20 ms step, so their difference is a small signal riding on a large
    common-mode term -- exactly the thing a first layer resolves poorly and that
    per-frame sensor noise swamps.

    Layout for N=2: [grav(3), d_grav(3), joints(4), d_joints(4)].
    """
    seq = list(frames)                                    # oldest -> newest
    deltas = [seq[i] - seq[i - 1] for i in range(len(seq) - 1, 0, -1)]
    parts = [seq[-1]] + deltas
    gravs = [p[:GRAV_DIM] for p in parts]
    joints = [p[GRAV_DIM:] for p in parts]
    return np.concatenate(gravs + joints)

# ---- 1. Define the Student Policy ----
class StudentPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, act_dim),
            nn.Tanh() # Binds outputs to [-1, 1]
        )

    def forward(self, x):
        return self.net(x)

# ---- 2. Data Collection Logic ----
def collect_data(env, student_policy, expert_policy, num_steps, is_student_acting=False,
                 max_delay=2, variant="tail"):
    """
    Rolls out a policy. The expert uses the FULL CURRENT observation.
    The student uses the DELAYED NOISY kinematic frame plus its own previous action
    and the episode progress.
    """
    student_states = []
    expert_actions = []
    # True when student_states[i+1] is genuinely the next tick of the SAME episode.
    # False at the last sample of an episode and of the collection, where the pair
    # would straddle a reset and the "action change" would be meaningless.
    pair_valid = []

    full_obs, _ = env.reset()

    # Initialize random delay, sensor bias and observation buffer for the first episode
    obs_delay = np.random.randint(0, max_delay + 1)
    bias = sample_sensor_bias()
    obs_buffer = []
    # The action that actually drove the plant last tick, whoever produced it. Neutral
    # at release: the robot is dropped with no command history.
    prev_action = np.zeros(ACT_DIM)

    for _ in range(num_steps):
        # 1. Extract what the student is allowed to see at this timestep
        obs_buffer.append(zero_tail(get_noisy_student_frame(full_obs, bias), variant))

        # 2. Retrieve the delayed frame. Only the sensor frame is delayed -- the
        #    previous action and the step count are known to the controller exactly.
        frame = obs_buffer[max(len(obs_buffer) - 1 - obs_delay, 0)]
        student_obs = build_student_obs(frame, prev_action, full_obs[STEP_IDX])
        student_states.append(student_obs)

        # 3. The Privileged Expert gets the full, current observation to generate ground-truth labels
        exp_action, _ = expert_policy.predict(full_obs, deterministic=True)
        expert_actions.append(exp_action)

        # 4. Decide who drives the environment for this step
        if is_student_acting:
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(student_obs).unsqueeze(0)
                act_action = student_policy(obs_tensor).squeeze(0).numpy()
        else:
            act_action = exp_action

        # 5. Step the environment
        prev_action = np.asarray(act_action, dtype=float).ravel().copy()
        full_obs, _, terminated, truncated, _ = env.step(act_action)
        pair_valid.append(not (terminated or truncated))

        # Handle environment resets mid-collection
        if terminated or truncated:
            full_obs, _ = env.reset()
            obs_delay = np.random.randint(0, max_delay + 1)
            bias = sample_sensor_bias()
            obs_buffer = []
            prev_action = np.zeros(ACT_DIM)

    v = np.array(pair_valid, dtype=bool)
    v[-1] = False                      # nothing follows the final sample
    return np.array(student_states), np.array(expert_actions), v

# ---- 3. Main DAgger Loop ----
def run_dagger(variant="tail", teacher=None, tag="", smooth_coef=0.0):
    cfg = VARIANTS[variant]

    # The teacher is loaded BEFORE the env so the env can be built at whatever
    # width that checkpoint expects -- 73-dim with the privileged DR block, 25-dim
    # without it. The student's slices are identical either way (the block is
    # appended last), so this only affects what the expert is shown.
    print("Loading privileged expert policy...")
    expert = SAC.load(teacher or teacher_path(variant))
    privileged = expert.observation_space.shape[0] > BASE_OBS_DIM
    env = gym.make(cfg["env_id"], privileged=privileged)
    print(f"  teacher obs {expert.observation_space.shape[0]}-dim "
          f"(privileged={privileged})")

    student_obs_dim = STUDENT_OBS_DIM
    act_dim = env.action_space.shape[0]
    print(f"  student obs {student_obs_dim}-dim "
          f"[grav 3 | joints 4 | prev action {ACT_DIM} | step 1]"
          + ("  (tail channel zeroed)" if variant == "notail" else ""))

    # Initialize Student with restricted observation space
    student = StudentPolicy(student_obs_dim, act_dim)
    optimizer = optim.Adam(student.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    iterations = 50
    steps_per_iter = 2000
    batch_size = 32
    epochs_per_iter = 20

    print("Iteration 0: Collecting initial expert data...")
    # Expert drives, Expert labels
    D_states, D_actions, D_valid = collect_data(env, student, expert, steps_per_iter,
                                                is_student_acting=False, variant=variant)

    for i in range(1, iterations + 1):
        print(f"\n--- DAgger Iteration {i}/{iterations} ---")

        # Train student mapping: Partial Obs -> Expert Action
        # Consecutive-pair view: row i is (s_t, expert a_t, s_t+1, is_real_pair).
        # s_next for the final row is a copy of itself, masked off by pair_valid.
        S = torch.FloatTensor(D_states)
        S_next = torch.cat([S[1:], S[-1:]], dim=0)
        dataset = TensorDataset(S, torch.FloatTensor(D_actions), S_next,
                                torch.FloatTensor(D_valid.astype(np.float32)))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        student.train()
        for epoch in range(epochs_per_iter):
            total_loss = 0
            for batch_states, batch_actions, batch_next, batch_valid in dataloader:
                optimizer.zero_grad()
                pred_actions = student(batch_states)
                loss = criterion(pred_actions, batch_actions)
                # Action-smoothness term, the student-side counterpart of
                # smooth_sac.py::SmoothSAC. The teacher is smooth because its actor
                # loss says so; the student only imitates, and it maps an 11-dim NOISY
                # observation where the teacher had 73 clean ones, so per-frame sensor
                # noise passes straight into the command. Measured on the shipped pair,
                # the student's action rate was 35-45x its teacher's.
                if smooth_coef > 0.0:
                    d = student(batch_next) - pred_actions
                    n = batch_valid.sum()
                    if n > 0:
                        loss = loss + smooth_coef * (
                            batch_valid * (d ** 2).sum(dim=1)).sum() / n
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{epochs_per_iter}, Loss: {total_loss/len(dataloader):.4f}")

        # Student drives (based on partial obs), Expert corrects/labels (based on full obs)
        new_states, new_expert_actions, new_valid = collect_data(
            env, student, expert, steps_per_iter, is_student_acting=True, variant=variant)

        D_states = np.concatenate([D_states, new_states], axis=0)
        D_actions = np.concatenate([D_actions, new_expert_actions], axis=0)
        D_valid = np.concatenate([D_valid, new_valid], axis=0)
        D_valid[len(D_valid) - len(new_valid) - 1] = False   # the join is not a real pair

        max_buffer_size = 40000
        if len(D_states) > max_buffer_size:
            D_states = D_states[-max_buffer_size:]
            D_actions = D_actions[-max_buffer_size:]
            D_valid = D_valid[-max_buffer_size:]

    out = os.path.join(
        policy_dir(),
        f"student_policy{cfg['suffix']}{tag}_{time.strftime('%Y%m%d-%H%M%S')}.pth")
    torch.save(student.state_dict(), out)
    print(f"Student saved to {out}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=list(VARIANTS), default="tail",
                         help="ablation condition to distill (default: tail)")
    parser.add_argument("--teacher", default=None,
                         help="explicit teacher .zip "
                              "(default: policies/cat_controller<suffix>.zip)")
    parser.add_argument("--tag", default="", help="extra suffix for the output filename")
    parser.add_argument("--smooth-coef", type=float, default=0.0,
                        help="weight on ||pi(s_t+1) - pi(s_t)||^2 in the distillation "
                             "loss (default 0.0 = off, the historical behaviour). The "
                             "student is otherwise pure imitation and inherits none of "
                             "the teacher's actor-loss smoothness.")
    args = parser.parse_args()
    run_dagger(args.variant, args.teacher, args.tag, args.smooth_coef)
