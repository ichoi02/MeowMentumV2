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
import time
from variants import VARIANTS

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

GRAV_DIM = 3
JOINT_DIM = 4
FRAME_DIM = GRAV_DIM + JOINT_DIM        # single-timestep student features (7)
N_FRAMES = 2                            # number of stacked timesteps of history
STUDENT_OBS_DIM = N_FRAMES * FRAME_DIM  # stacked history (14)

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
    """Newest frame plus successive backward DIFFERENCES, grouped by feature type.

    With the default 2 frames the layout is [current, current - previous] rather
    than [previous, current]: the same 14 numbers and the same information, but the
    velocity-like part is handed over directly instead of leaving the network to
    subtract two nearly-equal inputs itself. The two raw frames differ by ~1 part in
    100 over a 20 ms step, so their difference is a small signal riding on a large
    common-mode term -- exactly the thing a first layer resolves poorly and that
    per-frame sensor noise swamps.

    Layout for N=2: [grav(3), d_grav(3), joints(4), d_joints(4)].
    MUST match hardware/controller.py::stack_frames exactly.
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
                 max_delay=2, n_frames=N_FRAMES):
    """
    Rolls out a policy. The expert uses the FULL CURRENT observation.
    The student uses the DELAYED NOISY observation (n_frames stacked timesteps).
    """
    student_states = []
    expert_actions = []

    full_obs, _ = env.reset()

    # Initialize random delay, sensor bias and observation buffer for the first episode
    obs_delay = np.random.randint(0, max_delay + 1)
    bias = sample_sensor_bias()
    obs_buffer = []

    for _ in range(num_steps):
        # 1. Extract what the student is allowed to see at this timestep
        obs_buffer.append(get_noisy_student_frame(full_obs, bias))

        # 2. Retrieve the delayed frame plus the (n_frames-1) preceding it (clamped at 0)
        curr_idx = max(len(obs_buffer) - 1 - obs_delay, 0)
        frame_idxs = [max(curr_idx - k, 0) for k in reversed(range(n_frames))]
        delayed_student_obs = stack_frames([obs_buffer[i] for i in frame_idxs])
        student_states.append(delayed_student_obs)

        # 3. The Privileged Expert gets the full, current observation to generate ground-truth labels
        exp_action, _ = expert_policy.predict(full_obs, deterministic=True)
        expert_actions.append(exp_action)

        # 4. Decide who drives the environment for this step
        if is_student_acting:
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(delayed_student_obs).unsqueeze(0)
                act_action = student_policy(obs_tensor).squeeze(0).numpy()
        else:
            act_action = exp_action

        # 5. Step the environment
        full_obs, _, terminated, truncated, _ = env.step(act_action)

        # Handle environment resets mid-collection
        if terminated or truncated:
            full_obs, _ = env.reset()
            obs_delay = np.random.randint(0, max_delay + 1)
            bias = sample_sensor_bias()
            obs_buffer = []

    return np.array(student_states), np.array(expert_actions)

# ---- 3. Main DAgger Loop ----
def run_dagger(variant="tail", n_frames=N_FRAMES, teacher=None, tag=""):
    cfg = VARIANTS[variant]

    # The teacher is loaded BEFORE the env so the env can be built at whatever
    # width that checkpoint expects -- 73-dim with the privileged DR block, 25-dim
    # without it. The student's slices are identical either way (the block is
    # appended last), so this only affects what the expert is shown.
    print("Loading privileged expert policy...")
    expert = SAC.load(teacher or f"cat_controller{cfg['suffix']}")
    privileged = expert.observation_space.shape[0] > BASE_OBS_DIM
    env = gym.make(cfg["env_id"], privileged=privileged)
    print(f"  teacher obs {expert.observation_space.shape[0]}-dim "
          f"(privileged={privileged})")

    # n_frames x (3 front projected gravity + 4 joint angles) total dimensions
    student_obs_dim = n_frames * FRAME_DIM
    act_dim = env.action_space.shape[0]

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
    D_states, D_actions = collect_data(env, student, expert, steps_per_iter,
                                       is_student_acting=False, n_frames=n_frames)

    for i in range(1, iterations + 1):
        print(f"\n--- DAgger Iteration {i}/{iterations} ---")

        # Train student mapping: Partial Obs -> Expert Action
        dataset = TensorDataset(torch.FloatTensor(D_states), torch.FloatTensor(D_actions))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        student.train()
        for epoch in range(epochs_per_iter):
            total_loss = 0
            for batch_states, batch_actions in dataloader:
                optimizer.zero_grad()
                pred_actions = student(batch_states)
                loss = criterion(pred_actions, batch_actions)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{epochs_per_iter}, Loss: {total_loss/len(dataloader):.4f}")

        # Student drives (based on partial obs), Expert corrects/labels (based on full obs)
        new_states, new_expert_actions = collect_data(env, student, expert, steps_per_iter,
                                                      is_student_acting=True, n_frames=n_frames)

        D_states = np.concatenate([D_states, new_states], axis=0)
        D_actions = np.concatenate([D_actions, new_expert_actions], axis=0)

        max_buffer_size = 40000
        if len(D_states) > max_buffer_size:
            D_states = D_states[-max_buffer_size:]
            D_actions = D_actions[-max_buffer_size:]

    # Frame count is tagged into the filename only when it differs from the
    # default, so the standard artifact names stay exactly as the pipeline expects.
    frame_tag = "" if n_frames == N_FRAMES else f"_f{n_frames}"
    out = f"student_policy{cfg['suffix']}{frame_tag}{tag}_{time.strftime('%Y%m%d-%H%M%S')}.pth"
    torch.save(student.state_dict(), out)
    print(f"Student saved to {out}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=list(VARIANTS), default="tail",
                         help="ablation condition to distill (default: tail)")
    parser.add_argument("--frames", type=int, default=N_FRAMES,
                         help=f"stacked student frames (default: {N_FRAMES})")
    parser.add_argument("--teacher", default=None,
                         help="explicit teacher .zip (default: cat_controller<suffix>)")
    parser.add_argument("--tag", default="", help="extra suffix for the output filename")
    args = parser.parse_args()
    run_dagger(args.variant, args.frames, args.teacher, args.tag)
