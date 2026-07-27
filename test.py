import gymnasium as gym
from stable_baselines3 import SAC
import torch
import numpy as np
import time
import mujoco
import mujoco.viewer
from collections import deque
import cat_env
from distillation import (
    StudentPolicy, stack_frames, STUDENT_OBS_DIM, N_FRAMES,
    FRONT_GRAV_SLICE, JOINT_ANGLE_SLICE,
)

def student_frame(full_obs):
    """Clean (noise-free) single student frame: [front_proj_grav(3), joint_angles(4)]."""
    return np.concatenate([full_obs[FRONT_GRAV_SLICE], full_obs[JOINT_ANGLE_SLICE]])

def visualize():
    env = gym.make("Cat-v0")

    agent = 'student'
    if agent == 'teacher':
        print("Loading teacher policy")
        teacher = SAC.load("cat_controller")
    elif agent =='student':
        print("Loading student policy")
        student_obs_dim = STUDENT_OBS_DIM
        act_dim = env.action_space.shape[0]
        student = StudentPolicy(student_obs_dim, act_dim)
        student.load_state_dict(torch.load("student_policy.pth", map_location="cpu"))
        student.eval()

    obs, _ = env.reset()
    frame_hist = deque([student_frame(obs)] * N_FRAMES, maxlen=N_FRAMES)  # oldest -> newest

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

        slow = 1.0

        try:
            while viewer.is_running():
                step_start = time.time()
                
                if agent == 'teacher':
                    action, _ = teacher.predict(obs, deterministic=True)
                elif agent == 'student':
                    frame_hist.append(student_frame(obs))
                    student_obs = stack_frames(list(frame_hist))
                    with torch.no_grad():
                        obs_tensor = torch.FloatTensor(student_obs).unsqueeze(0)
                        action = student(obs_tensor).squeeze(0).numpy()
                obs, reward, terminated, truncated, info = env.step(action)

                if terminated or truncated:
                    obs, _ = env.reset()
                    frame_hist = deque([student_frame(obs)] * N_FRAMES, maxlen=N_FRAMES)
                
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