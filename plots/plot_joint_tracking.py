"""Plot commanded vs actual joint angles for one or more simulated drops.

Each drop produces one PNG: four small multiples (rot1, pitch, rot2, tail), each
showing three traces:

  command   the policy's RAW output, mapped to joint units (dotted, faint)
  target    that command after the per-channel low-pass (cat_env.filter_alpha) --
            this is what the inner PD loop is actually asked to reach (dashed)
  actual    the angle the PD loop reached (solid)

command -> target is what the filter absorbed; target -> actual is the inner-loop
tracking error, the thing that has to be small for a sim policy to survive on
hardware. Both gaps matter, and they have different fixes: the first is tuned with
`filter_alpha`, the second is a property of the actuator.

The plot title is the drop's initial attitude (roll, pitch), since that is what
distinguishes one random drop from another.

Examples
--------
    python plots/plot_joint_tracking.py                       # 4 drops, tail teacher
    python plots/plot_joint_tracking.py --n 20                # 20 drops
    python plots/plot_joint_tracking.py --agent student --deg # what ships on the robot
    python plots/plot_joint_tracking.py --variant notail --dark
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import gymnasium as gym
import torch
from stable_baselines3 import SAC

import cat_env  # noqa: F401  (registers Cat-v0 / CatNoTail-v0)
from cat_env.cat_env import BASE_OBS_DIM
from distillation import (
    StudentPolicy, build_student_obs, zero_tail, sample_sensor_bias,
    get_noisy_student_frame, GRAV_DIM, JOINT_DIM, ACT_DIM, STUDENT_OBS_DIM,
    STEP_IDX, FRONT_GRAV_SLICE, JOINT_ANGLE_SLICE,
)
from variants import VARIANTS, teacher_path, student_path

# Palette: categorical slots 1-2 (blue, orange), light and dark steps. Target vs
# actual is an identity pair, so it takes categorical hues; the dashed target adds
# a second, non-color channel so the pair reads without relying on hue.
THEME = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", ink3="#8a8880",
                  grid="#e4e3dd", actual="#2a78d6", target="#eb6834", raw="#f0a887"),
    "dark":  dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", ink3="#8a8880",
                  grid="#333330", actual="#3987e5", target="#d95926", raw="#8f4a28"),
}

# The four joints in XML/actuator order: (label, target column, sign, actual column,
# raw column) in the env's command log. The log row is
#   [filtered mapped target (3), joint qpos (4), raw mapped target (3)]
# so rot1 and rot2 share the roll channel -- rot2 is driven as its counter-twist,
# hence the -1, which applies to the raw command too.
JOINTS = [
    ("rot1  (spine roll, front)", 0, +1.0, 3, 7),
    ("pitch (spine)",             1, +1.0, 4, 8),
    ("rot2  (spine roll, rear)",  0, -1.0, 5, 7),
    ("tail  (pitch)",             2, +1.0, 6, 9),
]

def load_policy(variant, agent, policy_path):
    if agent == "teacher":
        return SAC.load(policy_path or teacher_path(variant))
    path = policy_path or student_path(variant)
    net = StudentPolicy(STUDENT_OBS_DIM, 3)
    net.load_state_dict(torch.load(path, map_location="cpu"))
    net.eval()
    return net

def roll_pitch_deg(quat_wxyz):
    """Initial attitude of the root body as intrinsic roll / pitch, in degrees."""
    roll, pitch, _ = R.from_quat(quat_wxyz, scalar_first=True).as_euler("xyz", degrees=True)
    return roll, pitch

def run_episode(env, u, policy, agent, variant, noisy):
    """One drop. Returns (log, initial roll/pitch, final per-body tilt)."""
    obs, _ = env.reset()
    roll0, pitch0 = roll_pitch_deg(u.data.qpos[3:7].copy())
    bias = sample_sensor_bias() if noisy else {"grav": np.zeros(GRAV_DIM),
                                              "joint": np.zeros(JOINT_DIM)}
    prev_action = np.zeros(ACT_DIM)   # no command history at release
    done = False
    while not done:
        if agent == "teacher":
            action, _ = policy.predict(obs, deterministic=True)
        else:
            frame = (get_noisy_student_frame(obs, bias) if noisy else
                     np.concatenate([obs[FRONT_GRAV_SLICE], obs[JOINT_ANGLE_SLICE]]))
            frame = zero_tail(frame, variant)
            # Current frame + last commanded action + episode progress, matching
            # distillation.py and hardware/controller.py.
            x = build_student_obs(frame, prev_action, obs[STEP_IDX])
            with torch.no_grad():
                action = policy(torch.FloatTensor(x).unsqueeze(0)).squeeze(0).numpy()
            if variant == "notail":
                action[2] = 0.0       # untrained channel, as the controller does
            prev_action = np.asarray(action, dtype=float).copy()
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    tilts = []
    for body in ("front_body", "rear_body"):
        up = R.from_quat(u.data.xquat[u._body_idx[body]], scalar_first=True).apply([0, 0, 1])
        tilts.append(np.degrees(np.arccos(np.clip(up[2], -1.0, 1.0))))
    return np.array(u.ctrls), (roll0, pitch0), tilts

def plot_episode(log, init, tilts, u, meta, out_path, c, degrees):
    """Four small multiples: commanded vs actual angle for each joint."""
    dt = u.model.opt.timestep * u.frame_skip
    t = np.arange(len(log)) * dt
    conv, unit = (np.degrees, "deg") if degrees else (lambda x: x, "rad")

    fig, axes = plt.subplots(2, 2, figsize=(11, 6.4), sharex=True)
    fig.patch.set_facecolor(c["surface"])

    has_raw = log.shape[1] >= 10        # pre-filter block is optional (older logs)
    for ax, (label, tgt_col, sign, act_col, raw_col) in zip(axes.flat, JOINTS):
        target = conv(sign * log[:, tgt_col])
        actual = conv(log[:, act_col])
        ax.set_facecolor(c["surface"])
        ax.axhline(0, color=c["grid"], lw=1, zorder=1)
        ax.grid(axis="y", color=c["grid"], lw=0.7, zorder=0)
        ax.set_axisbelow(True)
        # Raw command underneath, then the filtered target, then the achieved angle:
        # each layer is what the one below it was asked to follow.
        if has_raw:
            raw = conv(sign * log[:, raw_col])
            ax.plot(t, raw, color=c["raw"], lw=1.2, ls=(0, (1, 2)),
                    label="command (pre-filter)", zorder=2)
        ax.plot(t, target, color=c["target"], lw=1.6, ls=(0, (5, 2)),
                label="target (filtered)", zorder=3)
        ax.plot(t, actual, color=c["actual"], lw=1.9, label="actual", zorder=4)

        # RMS goes in the title row, not inside the axes -- the traces use the full
        # height and any in-axes annotation eventually lands on top of one. Two
        # numbers, because the two gaps have different causes: how much travel the
        # filter removed, and how far the PD loop then fell behind.
        rms = float(np.sqrt(np.mean((target - actual) ** 2)))
        ax.set_title(label, color=c["ink"], fontsize=10, loc="left", pad=6)
        if has_raw:
            raw_tr = float(np.abs(np.diff(conv(sign * log[:, raw_col]))).sum())
            tgt_tr = float(np.abs(np.diff(target)).sum())
            note = (f"filter {tgt_tr/raw_tr:.0%} of cmd travel   ·   "
                    f"PD RMS err {rms:.2f} {unit}") if raw_tr > 1e-9 else \
                   f"PD RMS err {rms:.2f} {unit}"
        else:
            note = f"RMS err {rms:.2f} {unit}"
        ax.set_title(note, color=c["ink3"], fontsize=8.5, loc="right", pad=7)
        ax.set_ylabel(f"angle ({unit})", color=c["ink2"], fontsize=9)
        ax.tick_params(colors=c["ink2"], labelsize=8.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(c["grid"])

    for ax in axes[1]:
        ax.set_xlabel("time (s)", color=c["ink2"], fontsize=9)

    roll0, pitch0 = init
    fig.suptitle(f"initial roll {roll0:+.1f}°,  pitch {pitch0:+.1f}°",
                 color=c["ink"], fontsize=14, x=0.055, ha="left", y=0.985)
    fig.text(0.055, 0.925,
             f"{meta}   ·   final tilt {tilts[0]:.0f}°/{tilts[1]:.0f}° "
             f"(front/rear)   ·   dotted = raw command, dashed = filtered target, "
             f"solid = achieved",
             color=c["ink2"], fontsize=9, ha="left")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.985, 0.995),
                     frameon=False, ncol=3, fontsize=9.5)
    for txt in leg.get_texts():
        txt.set_color(c["ink2"])

    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_path, dpi=150, facecolor=c["surface"])
    plt.close(fig)

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=4, help="how many drops to plot (default 4)")
    p.add_argument("--variant", choices=list(VARIANTS), default="tail")
    p.add_argument("--agent", choices=["teacher", "student"], default="teacher")
    p.add_argument("--policy", default=None, help="explicit policy file")
    p.add_argument("--clean", action="store_true",
                   help="student only: no sensor noise/bias")
    p.add_argument("--deg", action="store_true", help="plot degrees instead of radians")
    p.add_argument("--dark", action="store_true", help="dark theme")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=os.path.join(REPO, "plots", "out"),
                   help="output directory (default plots/out)")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    c = THEME["dark" if args.dark else "light"]

    # Policy first, so the env can be built at whatever width the checkpoint
    # expects -- same reason as docs/evaluate.py. Teachers trained before the
    # privileged DR block are 25-dim and would otherwise fail against the 73-dim
    # env, which is exactly the case --policy exists to reach. The student reads
    # slices that are unaffected either way, so it always gets the full env.
    policy = load_policy(args.variant, args.agent, args.policy)
    privileged = (args.agent != "teacher"
                  or policy.observation_space.shape[0] > BASE_OBS_DIM)
    env = gym.make(VARIANTS[args.variant]["env_id"], privileged=privileged)
    u = env.unwrapped
    meta = f"{args.variant} / {args.agent}"

    # Seeded AFTER the policy loads, not before: the env's attitude/DR draws come
    # from the global numpy stream, and SAC.load consumes a checkpoint-dependent
    # amount of it. Seeding first made --seed 0 produce *different* drops for two
    # different checkpoints, which defeats the main use of this script -- putting
    # two policies on the same drop. docs/evaluate.py seeds here for the same reason.
    np.random.seed(args.seed)

    for i in range(args.n):
        log, init, tilts = run_episode(env, u, policy, args.agent, args.variant,
                                       noisy=(args.agent == "student" and not args.clean))
        name = f"ep{i:02d}_roll{init[0]:+04.0f}_pitch{init[1]:+04.0f}.png"
        path = os.path.join(args.out, name)
        plot_episode(log, init, tilts, u, meta, path, c, args.deg)
        print(f"[{i+1}/{args.n}] {path}  "
              f"(initial roll {init[0]:+.1f} pitch {init[1]:+.1f}, "
              f"final tilt {tilts[0]:.0f}/{tilts[1]:.0f})")
    env.close()

if __name__ == "__main__":
    main()
