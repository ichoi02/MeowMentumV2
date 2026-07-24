import gymnasium as gym
from stable_baselines3 import SAC
import torch
import torch.nn as nn
import numpy as np
import os
import time
import mujoco
import mujoco.viewer
import cat_env

class StudentPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, act_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

def get_student_obs(full_obs):
    quats = full_obs[0:9] #FIXME
    gyros = full_obs[18:21]
    joint_angles = full_obs[18+6+7:18+6+7+4]
    return np.concatenate([quats, gyros, joint_angles])

def visualize():
    env = gym.make("Cat-v0")

    agent = 'teacher'
    if agent == 'teacher':
        print("Loading teacher policy")
        teacher = SAC.load("cat_controller")
    elif agent =='student':
        print("Loading student policy")
        student_obs_dim = 16
        act_dim = env.action_space.shape[0]
        student = StudentPolicy(student_obs_dim, act_dim)
        student.load_state_dict(torch.load("student_policy.pth"))

    obs, _ = env.reset()
    
    mj_model = env.unwrapped.model
    mj_data = env.unwrapped.data

    print("Starting visualization")
    
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        # Cam tracking
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "spine_1")
        viewer.cam.trackbodyid = body_id
        viewer.cam.distance = 1.5
        viewer.cam.elevation = -20
        viewer.cam.azimuth = 90
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_WORLD

        slow = 0.5

        try:
            while viewer.is_running():
                step_start = time.time()
                
                if agent == 'teacher':
                    action, _ = teacher.predict(obs, deterministic=True)
                elif agent == 'student':
                    student_obs = get_student_obs(obs)
                    with torch.no_grad():
                        obs_tensor = torch.FloatTensor(student_obs).unsqueeze(0)
                        action = student(obs_tensor).squeeze(0).numpy()
                obs, reward, terminated, truncated, info = env.step(action)
                
                if terminated or truncated:
                    obs, _ = env.reset()
                
                viewer.sync()
                
                time_until_next_step = env.unwrapped.dt / slow - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
                    
        except KeyboardInterrupt:
            print("\nStopped.")
        finally:
            env.close()

if __name__ == "__main__":
    visualize()