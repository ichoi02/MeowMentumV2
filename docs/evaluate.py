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
from cat_env.cat_env import BASE_OBS_DIM
from distillation import (
    StudentPolicy, stack_frames, sample_sensor_bias, get_noisy_student_frame,
    FRAME_DIM, JOINT_DIM, GRAV_DIM, N_FRAMES, FRONT_GRAV_SLICE, JOINT_ANGLE_SLICE,
)
from variants import VARIANTS, teacher_path, student_path

UPRIGHT_DEG = 30.0  # per-body tilt below which a landing counts as upright

# Unweighted penalty magnitudes reported by the env (see cat_env::_get_reward).
# These are what reward-weight tuning reads: a weight is doing something when the
# magnitude it prices moves, which is visible whether or not the weight is zero.
MAGNITUDES = ("m_en", "m_av", "m_jv", "m_time")
# Deployed action smoothness, ||a_t - a_{t-1}||^2 summed over the 3 channels.
# This is NOT read from `info`: the env stopped pricing action rate when the term
# moved into the actor loss (smooth_sac.py), so it is measured here instead, on
# the deterministic action the policy actually commands. It is the number to watch
# when tuning `--smooth-coef`, and it is the direct successor to the old `m_sm`
# -- but not comparable to it, since `m_sm` averaged over channels rather than
# summing, and was recorded on the training-time (noisy) action.
DET_SMOOTHNESS = "m_dsm"
# Task terms, for scale: a penalty weight is "large" relative to these.
TASK_TERMS = ("r_pos", "r_bonus")

def tilt_deg(env, body):
    """Angle between the body's +z and world +z, in degrees."""
    up = R.from_quat(env.data.xquat[env._body_idx[body]], scalar_first=True).apply([0, 0, 1])
    return np.degrees(np.arccos(np.clip(up[2], -1.0, 1.0)))

def clean_frame(full_obs):
    """Noise-free student frame: [front_proj_grav(3), joint_angles(4)]."""
    return np.concatenate([full_obs[FRONT_GRAV_SLICE], full_obs[JOINT_ANGLE_SLICE]])

def force_attitude(u, roll_deg, pitch_deg):
    """Re-release at a fixed roll/pitch instead of the uniform random attitude.

    reset_model() samples a uniform SO(3) attitude, so success over N drops mixes
    every release orientation together. This overwrites ONLY the root orientation,
    leaving the domain-randomization draw, the randomized initial joint angles and
    the initial tumble alone -- the point is to hold the release attitude fixed
    while everything else still varies, not to make the drop deterministic.

    Whichever of roll/pitch is not given is zero, so `--roll 180` is a pure roll
    release and `--pitch 90` a pure pitch one. Yaw is always 0 (the task and the
    observation are yaw-invariant, so it would not change anything).

    Returns the recomputed observation: the one reset() handed back describes the
    pre-override state.
    """
    qpos, qvel = u.data.qpos.copy(), u.data.qvel.copy()
    euler = [roll_deg or 0.0, pitch_deg or 0.0, 0.0]
    x, y, z, w = R.from_euler("xyz", euler, degrees=True).as_quat()
    qpos[3:7] = [w, x, y, z]   # MuJoCo stores the free joint as wxyz
    u.set_state(qpos, qvel)
    return u._get_obs()

def evaluate(variant="tail", agent="student", n_frames=N_FRAMES, episodes=200,
             noisy=True, policy_path=None, seed=0, roll_deg=None, pitch_deg=None):
    cfg = VARIANTS[variant]

    # The teacher is loaded BEFORE the env so the env can be built at whatever
    # width that checkpoint expects. Teachers trained before the privileged DR
    # block are 25-dim and would otherwise fail against the 73-dim env; the
    # student never sees the block either way, so it always gets the full env.
    policy = None
    if agent == "teacher":
        policy = SAC.load(policy_path or teacher_path(variant))
        privileged = policy.observation_space.shape[0] > BASE_OBS_DIM
    else:
        privileged = True

    env = gym.make(cfg["env_id"], privileged=privileged)
    u = env.unwrapped

    # The env's domain randomization draws from the global numpy stream, so
    # seeding it here makes two configs see the same sequence of drops.
    np.random.seed(seed)

    if agent != "teacher":
        path = policy_path or student_path(variant)
        policy = StudentPolicy(n_frames * FRAME_DIM, env.action_space.shape[0])
        policy.load_state_dict(torch.load(path, map_location="cpu"))
        policy.eval()

    no_bias = {"grav": np.zeros(GRAV_DIM), "joint": np.zeros(JOINT_DIM)}
    ok = 0
    ftilts, rtilts = [], []
    mags = {k: [] for k in MAGNITUDES + TASK_TERMS}   # per-step reward terms
    abs_action, travel = [], []             # |a| per channel, joint travel per episode
    final_omega = []                        # |body omega| at the end of the drop
    dsm = []                                # deployed action smoothness, per episode

    fixed_attitude = roll_deg is not None or pitch_deg is not None

    for _ in range(episodes):
        obs, _ = env.reset()
        if fixed_attitude:
            obs = force_attitude(u, roll_deg, pitch_deg)
        bias = sample_sensor_bias() if noisy else no_bias
        hist = None
        done = False
        ep_abs, prev_q = [], u.data.qpos[7:].copy()
        ep_travel = np.zeros(4)
        ep_dsm, prev_a = [], None

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

            # Measured on the action as commanded, before env.step consumes it,
            # so it matches what smooth_sac.py penalizes.
            a = np.asarray(action, dtype=np.float64)
            if prev_a is not None:
                ep_dsm.append(float(np.sum((a - prev_a) ** 2)))
            prev_a = a

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
        dsm.append(np.mean(ep_dsm))
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
        DET_SMOOTHNESS: float(np.mean(dsm)),
        "roll_deg": roll_deg,      # None = uniform random attitude
        "pitch_deg": pitch_deg,
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
    parser.add_argument("--roll", type=float, default=None, metavar="DEG",
                        help="release at a fixed roll about world x with pitch 0, "
                             "instead of a uniformly random attitude (e.g. 180 = "
                             "upside-down, 90 = on its side). DR, initial joint "
                             "angles and the initial tumble still vary per drop")
    parser.add_argument("--pitch", type=float, default=None, metavar="DEG",
                        help="release at a fixed pitch about world y with roll 0; "
                             "combine with --roll to set both")
    args = parser.parse_args()

    res = evaluate(args.variant, args.agent, args.frames, args.episodes,
                   noisy=not args.clean, policy_path=args.policy, seed=args.seed,
                   roll_deg=args.roll, pitch_deg=args.pitch)

    if args.json:
        print(json.dumps(res))
        return

    label = f"{args.variant}/{args.agent}"
    if args.agent == "student":
        label += f" f{args.frames}"
    label += " (clean)" if args.clean else " (noisy)"
    if res["roll_deg"] is not None or res["pitch_deg"] is not None:
        label += f" [roll {res['roll_deg'] or 0:+.0f} pitch {res['pitch_deg'] or 0:+.0f}]"
    print(f"{label}: {res['success_pct']:.1f}% both-upright over {res['episodes']} drops "
          f"| median tilt f/r {res['median_front_tilt']:.0f}/{res['median_rear_tilt']:.0f} deg")

    if args.stats:
        print("  task        " + "  ".join(f"{k}={res[k]:.4g}" for k in TASK_TERMS))
        print("  magnitudes  " + "  ".join(f"{k[2:]}={res[k]:.4g}"
                                           for k in MAGNITUDES + (DET_SMOOTHNESS,)))
        print(f"  |action| roll/pitch/tail  "
              + "/".join(f"{v:.2f}" for v in res["abs_action"]))
        print(f"  joint travel (rad)  "
              + "/".join(f"{v:.2f}" for v in res["travel"])
              + f"   final |omega| {res['final_omega']:.2f} rad/s")

if __name__ == "__main__":
    main()
