import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cat_env

# ctrls columns (see CatEnv.step): [target_roll, target_pitch, target_tail,
#                                   cur_rot1, cur_pitch, cur_rot2, cur_tail]
DT = 0.02  # frame_skip(20) * timestep(0.001)

def rollout():
    env = gym.make("Cat-v0")
    model = SAC.load("cat_controller")

    obs, _ = env.reset()
    u = env.unwrapped
    log = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if len(u.ctrls) > 0:            # cleared on the terminal step
            log.append(u.ctrls[-1].copy())
    env.close()
    return np.array(log)

def main():
    c = rollout()
    t = np.arange(len(c)) * DT

    # (real angle column, target series, title)
    panels = [
        (3, c[:, 0],  "rot1 (roll)"),
        (4, c[:, 1],  "pitch"),
        (5, -c[:, 0], "rot2 (-roll)"),
        (6, c[:, 2],  "tail"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    for ax, (real_col, target, title) in zip(axes.flat, panels):
        ax.plot(t, target, "--", color="tab:red", label="target angle")
        ax.plot(t, c[:, real_col], "-", color="tab:blue", label="real angle")
        ax.set_title(title)
        ax.set_ylabel("angle (rad)")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("time (s)")

    fig.suptitle("Teacher rollout: commanded target vs real joint angle", fontweight="bold")
    fig.tight_layout()
    out = "rollout_plot.png"
    fig.savefig(out, dpi=130)
    print(f"saved {out}  ({len(c)} steps, {t[-1]:.2f}s)")

if __name__ == "__main__":
    main()
