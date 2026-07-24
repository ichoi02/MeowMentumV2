import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from stable_baselines3 import SAC
import cat_env
import cat_env.env_util as util
import time

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# Full obs layout (see cat_env/cat_env.py::_get_obs):
#   [0:9]    front_body_rot
#   [9:18]   rear_body_rot
#   [18:21]  front_gyro
#   [21:24]  rear_gyro
#   [24:35]  qpos (7 free + 4 joints)  -> joint angles at [31:35]
# Only the front body carries an IMU on the real robot, so the student sees front_body_rot.
FRONT_ROT_SLICE = slice(0, 9)
JOINT_ANGLE_SLICE = slice(24 + 7, 24 + 7 + 4)

ROT_DIM = 9
JOINT_DIM = 4
FRAME_DIM = ROT_DIM + JOINT_DIM      # single-timestep student features
STUDENT_OBS_DIM = 2 * FRAME_DIM      # two stacked timesteps: t-1 and t

def get_noisy_student_frame(full_obs, rot_noise_std=0.01, joint_noise_std=0.02):
    """One timestep of what the student is allowed to see: [front_rot(9), joint_angles(4)]."""
    front_rot = full_obs[FRONT_ROT_SLICE].copy()
    joint_angles = full_obs[JOINT_ANGLE_SLICE].copy()

    noisy_rot = util.add_rotational_noise(front_rot, rot_noise_std)
    noisy_joints = util.add_gaussian_noise(joint_angles, joint_noise_std)

    return np.concatenate([noisy_rot, noisy_joints])

def stack_frames(prev_frame, curr_frame):
    """[front_rot_{t-1}, front_rot_t, joint_angles_{t-1}, joint_angles_t]"""
    return np.concatenate([
        prev_frame[:ROT_DIM],
        curr_frame[:ROT_DIM],
        prev_frame[ROT_DIM:],
        curr_frame[ROT_DIM:],
    ])

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
def collect_data(env, student_policy, expert_policy, num_steps, is_student_acting=False, max_delay=2):
    """
    Rolls out a policy. The expert uses the FULL CURRENT observation.
    The student uses the DELAYED NOISY observation (two stacked timesteps).
    """
    student_states = []
    expert_actions = []

    full_obs, _ = env.reset()

    # Initialize random delay and observation buffer for the first episode
    obs_delay = np.random.randint(0, max_delay + 1)
    obs_buffer = []

    for _ in range(num_steps):
        # 1. Extract what the student is allowed to see at this timestep
        obs_buffer.append(get_noisy_student_frame(full_obs))

        # 2. Retrieve the delayed frame plus the one before it
        curr_idx = max(len(obs_buffer) - 1 - obs_delay, 0)
        prev_idx = max(curr_idx - 1, 0)
        delayed_student_obs = stack_frames(obs_buffer[prev_idx], obs_buffer[curr_idx])
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
            obs_buffer = []

    return np.array(student_states), np.array(expert_actions)

# ---- 3. Main DAgger Loop ----
def run_dagger():
    env = gym.make("Cat-v0")

    # 2 x (9 front rotation matrix + 4 joint angles) = 26 total dimensions
    student_obs_dim = STUDENT_OBS_DIM
    act_dim = env.action_space.shape[0]

    print("Loading privileged expert policy...")
    expert = SAC.load("cat_controller")

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
    D_states, D_actions = collect_data(env, student, expert, steps_per_iter, is_student_acting=False)

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
        new_states, new_expert_actions = collect_data(env, student, expert, steps_per_iter, is_student_acting=True)

        D_states = np.concatenate([D_states, new_states], axis=0)
        D_actions = np.concatenate([D_actions, new_expert_actions], axis=0)

        max_buffer_size = 40000
        if len(D_states) > max_buffer_size:
            D_states = D_states[-max_buffer_size:]
            D_actions = D_actions[-max_buffer_size:]

    # Changed save filename to reflect new observation space
    torch.save(student.state_dict(), f"student_policy_{str(time.time())}.pth")

if __name__ == "__main__":
    run_dagger()
