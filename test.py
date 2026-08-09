import gymnasium as gym
from stable_baselines3 import SAC
import torch
import numpy as np
import time
import mujoco
import mujoco.viewer
from collections import deque
from scipy.spatial.transform import Rotation as R
import cat_env
from cat_env.cat_env import BASE_OBS_DIM
from distillation import (
    StudentPolicy, build_student_obs, zero_tail, STUDENT_OBS_DIM, ACT_DIM,
    STEP_IDX, FRONT_GRAV_SLICE, JOINT_ANGLE_SLICE,
)
from variants import VARIANTS, teacher_path, student_path

def student_frame(full_obs):
    """Clean (noise-free) single student frame: [front_proj_grav(3), joint_angles(4)]."""
    return np.concatenate([full_obs[FRONT_GRAV_SLICE], full_obs[JOINT_ANGLE_SLICE]])

def force_release(env, roll_deg=None, still=False):
    """Re-release the robot at a fixed roll instead of the random attitude.

    reset_model() samples a uniform SO(3) attitude, so every drop looks different.
    This overwrites just the root orientation (and optionally the initial tumble)
    after the reset, leaving the domain-randomization draw and the randomized joint
    angles alone -- the point is to watch the SAME maneuver from a repeatable
    release, not to remove the rest of the randomization.

    Returns the recomputed observation, which the caller must use: the one reset()
    handed back describes the pre-override state.
    """
    u = env.unwrapped
    qpos, qvel = u.data.qpos.copy(), u.data.qvel.copy()
    if roll_deg is not None:
        # Roll about the world x-axis. MuJoCo stores the free joint as wxyz.
        x, y, z, w = R.from_euler("x", roll_deg, degrees=True).as_quat()
        qpos[3:7] = [w, x, y, z]
    if still:
        qvel[3:6] = 0.0   # drop with zero angular momentum, not already tumbling
    u.set_state(qpos, qvel)
    return u._get_obs()

def visualize(variant="tail", agent="student", roll_deg=None, still=False,
              policy_path=None):
    cfg = VARIANTS[variant]

    # Teacher loads BEFORE the env, so the env can be built at whatever width the
    # checkpoint expects -- same reason as docs/evaluate.py. Teachers trained
    # before the privileged DR block are 25-dim and would otherwise fail against
    # the 73-dim env. The student's slices are unaffected either way (the block is
    # appended last), so it always gets the full env.
    teacher = None
    if agent == 'teacher':
        print("Loading teacher policy")
        teacher = SAC.load(policy_path or teacher_path(variant))
        privileged = teacher.observation_space.shape[0] > BASE_OBS_DIM
    else:
        privileged = True

    env = gym.make(cfg["env_id"], privileged=privileged)

    if agent == 'student':
        print("Loading student policy")
        student_obs_dim = STUDENT_OBS_DIM
        act_dim = env.action_space.shape[0]
        student = StudentPolicy(student_obs_dim, act_dim)
        student.load_state_dict(torch.load(
            policy_path or student_path(variant), map_location="cpu"))
        student.eval()

    obs, _ = env.reset()
    if roll_deg is not None or still:
        obs = force_release(env, roll_deg, still)
    prev_action = np.zeros(ACT_DIM)   # no command history at release

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
                    frame = zero_tail(student_frame(obs), variant)
                    student_obs = build_student_obs(frame, prev_action, obs[STEP_IDX])
                    with torch.no_grad():
                        obs_tensor = torch.FloatTensor(student_obs).unsqueeze(0)
                        action = student(obs_tensor).squeeze(0).numpy()
                    if variant == "notail":
                        action[2] = 0.0    # untrained channel, as the controller does
                    prev_action = np.asarray(action, dtype=float).copy()
                obs, reward, terminated, truncated, info = env.step(action)

                if terminated or truncated:
                    obs, _ = env.reset()
                    if roll_deg is not None or still:
                        obs = force_release(env, roll_deg, still)
                    prev_action = np.zeros(ACT_DIM)
                
                viewer.sync()
                
                time_until_next_step = env.unwrapped.dt / slow - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
                    
        except KeyboardInterrupt:
            print("\nStopped.")
        finally:
            env.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=list(VARIANTS), default="tail",
                         help="ablation condition to view (default: tail)")
    parser.add_argument("--agent", choices=["student", "teacher"], default="student",
                         help="policy to view (default: student)")
    parser.add_argument("--roll", type=float, default=None, metavar="DEG",
                         help="release at a fixed roll about world x instead of a "
                              "random attitude, e.g. 180 (upside-down) or 90 (on its side)")
    parser.add_argument("--still", action="store_true",
                         help="release with zero angular velocity (no initial tumble)")
    parser.add_argument("--policy", default=None,
                         help="explicit policy file to load, instead of the staged "
                              "policies/cat_controller<suffix>.zip or "
                              "policies/student_policy<suffix>.pth "
                              "(same flag as docs/evaluate.py)")
    args = parser.parse_args()
    visualize(args.variant, args.agent, args.roll, args.still, args.policy)