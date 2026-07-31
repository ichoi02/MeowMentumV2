"""Closed-loop righting success rate in simulation.

Rolls a trained policy over many random drops and reports how often BOTH bodies
finish upright. This is the number the tail ablation actually compares, and the
one that says whether a given student frame-count survives realistic sensor
error -- so by default the student is evaluated through the same per-episode
bias + per-frame noise model it was distilled against.
"""
import argparse
import json
import os
import sys
from collections import deque

# Runnable as `python docs/evaluate.py` from the repo root, which puts docs/ on
# sys.path but not the root the imports below need.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import gymnasium as gym
import torch
from scipy.spatial.transform import Rotation as R
from stable_baselines3 import SAC

import cat_env  # noqa: F401  (registers Cat-v0 / CatNoTail-v0)
from distillation import (
    StudentPolicy, stack_frames, sample_sensor_bias, get_noisy_student_frame,
    FRAME_DIM, JOINT_DIM, GRAV_DIM, N_FRAMES, FRONT_GRAV_SLICE, JOINT_ANGLE_SLICE,
)
from variants import VARIANTS

UPRIGHT_DEG = 30.0  # per-body tilt below which a landing counts as upright

# Unweighted penalty magnitudes reported by the env (see cat_env::_get_reward).
# These are what reward-weight tuning reads: a weight is doing something when the
# magnitude it prices moves, which is visible whether or not the weight is zero.
MAGNITUDES = ("m_sm", "m_en", "m_av", "m_jv", "m_time")
# Task terms, for scale: a penalty weight is "large" relative to these.
TASK_TERMS = ("r_pos", "r_bonus")

def tilt_deg(env, body):
    """Angle between the body's +z and world +z, in degrees."""
    up = R.from_quat(env.data.xquat[env._body_idx[body]], scalar_first=True).apply([0, 0, 1])
    return np.degrees(np.arccos(np.clip(up[2], -1.0, 1.0)))

def clean_frame(full_obs):
    """Noise-free student frame: [front_proj_grav(3), joint_angles(4)]."""
    return np.concatenate([full_obs[FRONT_GRAV_SLICE], full_obs[JOINT_ANGLE_SLICE]])

def evaluate(variant="tail", agent="student", n_frames=N_FRAMES, episodes=200,
             noisy=True, policy_path=None, seed=0):
    cfg = VARIANTS[variant]
    env = gym.make(cfg["env_id"])
    u = env.unwrapped

    # The env's domain randomization draws from the global numpy stream, so
    # seeding it here makes two configs see the same sequence of drops.
    np.random.seed(seed)

    if agent == "teacher":
        policy = SAC.load(policy_path or f"cat_controller{cfg['suffix']}")
    else:
        path = policy_path or f"student_policy{cfg['suffix']}.pth"
        policy = StudentPolicy(n_frames * FRAME_DIM, env.action_space.shape[0])
        policy.load_state_dict(torch.load(path, map_location="cpu"))
        policy.eval()

    no_bias = {"grav": np.zeros(GRAV_DIM), "joint": np.zeros(JOINT_DIM)}
    ok = 0
    ftilts, rtilts = [], []
    mags = {k: [] for k in MAGNITUDES + TASK_TERMS}   # per-step reward terms
    abs_action, travel = [], []             # |a| per channel, joint travel per episode
    final_omega = []                        # |body omega| at the end of the drop

    for _ in range(episodes):
        obs, _ = env.reset()
        bias = sample_sensor_bias() if noisy else no_bias
        hist = None
        done = False
        ep_abs, prev_q = [], u.data.qpos[7:].copy()
        ep_travel = np.zeros(4)

        while not done:
            if agent == "teacher":
                action, _ = policy.predict(obs, deterministic=True)
            else:
                frame = get_noisy_student_frame(obs, bias) if noisy else clean_frame(obs)
                # Prefill the history with the first frame, matching controller.py.
                hist = deque([frame] * n_frames, maxlen=n_frames) if hist is None else hist
                hist.append(frame)
                with torch.no_grad():
                    x = torch.FloatTensor(stack_frames(list(hist))).unsqueeze(0)
                    action = policy(x).squeeze(0).numpy()

            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            for k in mags:
                mags[k].append(info[k])
            ep_abs.append(np.abs(action))
            q = u.data.qpos[7:].copy()
            ep_travel += np.abs(q - prev_q)
            prev_q = q

        f, r = tilt_deg(u, "front_body"), tilt_deg(u, "rear_body")
        ftilts.append(f)
        rtilts.append(r)
        ok += (f < UPRIGHT_DEG and r < UPRIGHT_DEG)
        abs_action.append(np.mean(ep_abs, axis=0))
        travel.append(ep_travel)
        final_omega.append(np.linalg.norm(
            np.concatenate([u._body_gyro("front_body"), u._body_gyro("rear_body")])))

    env.close()
    res = {
        "success_pct": 100.0 * ok / episodes,
        "median_front_tilt": float(np.median(ftilts)),
        "median_rear_tilt": float(np.median(rtilts)),
        "mean_tilt": float(np.mean(ftilts + rtilts)),
        "episodes": episodes,
        "abs_action": np.mean(abs_action, axis=0).tolist(),   # [roll, pitch, tail]
        "travel": np.mean(travel, axis=0).tolist(),           # [rot1, pitch, rot2, tail] rad
        "final_omega": float(np.mean(final_omega)),           # rad/s
    }
    res.update({k: float(np.mean(v)) for k, v in mags.items()})
    return res

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=list(VARIANTS), default="tail")
    parser.add_argument("--agent", choices=["student", "teacher"], default="student")
    parser.add_argument("--frames", type=int, default=N_FRAMES,
                        help=f"stacked student frames (default: {N_FRAMES})")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--policy", default=None, help="explicit policy file to load")
    parser.add_argument("--clean", action="store_true",
                        help="evaluate without sensor noise/bias (optimistic)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stats", action="store_true",
                        help="also print the unweighted penalty magnitudes + motion stats")
    parser.add_argument("--json", action="store_true", help="dump the full result as JSON")
    args = parser.parse_args()

    res = evaluate(args.variant, args.agent, args.frames, args.episodes,
                   noisy=not args.clean, policy_path=args.policy, seed=args.seed)

    if args.json:
        print(json.dumps(res))
        return

    label = f"{args.variant}/{args.agent}"
    if args.agent == "student":
        label += f" f{args.frames}"
    label += " (clean)" if args.clean else " (noisy)"
    print(f"{label}: {res['success_pct']:.1f}% both-upright over {res['episodes']} drops "
          f"| median tilt f/r {res['median_front_tilt']:.0f}/{res['median_rear_tilt']:.0f} deg")

    if args.stats:
        print("  task        " + "  ".join(f"{k}={res[k]:.4g}" for k in TASK_TERMS))
        print("  magnitudes  " + "  ".join(f"{k[2:]}={res[k]:.4g}" for k in MAGNITUDES))
        print(f"  |action| roll/pitch/tail  "
              + "/".join(f"{v:.2f}" for v in res["abs_action"]))
        print(f"  joint travel (rad)  "
              + "/".join(f"{v:.2f}" for v in res["travel"])
              + f"   final |omega| {res['final_omega']:.2f} rad/s")

if __name__ == "__main__":
    main()
