import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import cat_env
import os
import time
from smooth_sac import SmoothSAC
from variants import VARIANTS, policy_dir

class TensorboardRewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    # r_sm is gone: action smoothness is an actor-loss term now, logged by
    # SmoothSAC as train/smooth_loss rather than by the env. r_bonus is gone too:
    # the success bonus was removed from the reward (cat_env.py).
    #
    # record_mean, not record: `record` overwrites, so a dumped value was whichever
    # step happened to land last in the logging interval -- a single sample presented
    # as if it were an episode statistic, which made cross-run comparisons unreliable.
    TERMS = ("r_pos", "r_en", "r_av", "r_jv", "r_time", "up_mean")

    def _on_step(self) -> bool:
        # locals["infos"] is a list of info dictionaries from the vectorized environments
        for info in self.locals["infos"]:
            if "r_pos" in info:
                # Log each term under a "rewards/" group in TensorBoard
                for term in self.TERMS:
                    self.logger.record_mean(f"rewards/{term}", info[term])
        return True

def train(variant="tail", total_timesteps=1_000_000, tag="", n_envs=10, seed=None,
          out=None, gradient_steps=1, run_name=None, privileged=True,
          smooth_coef=2.0):
    cfg = VARIANTS[variant]

    env = make_vec_env(
        cfg["env_id"],
        n_envs=n_envs,
        vec_env_cls=SubprocVecEnv,
        env_kwargs={"privileged": privileged},
    )

    model = SmoothSAC(
        "MlpPolicy",
        env,
        smooth_coef=smooth_coef,  # L2 action-smoothness in the actor loss
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        tensorboard_log="./run_logs/",
        device="cpu",
        seed=seed,
        learning_rate=3e-4,
        buffer_size=300_000,
        learning_starts=10_000,   # collect some random transitions before learning
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,             # train after every vec-env step
        gradient_steps=gradient_steps,
        ent_coef="auto",          # auto-tune entropy (key for multimodal exploration)
    )

    reward_callback = TensorboardRewardCallback()

    model.learn(
        total_timesteps=total_timesteps,
        log_interval=20,
        progress_bar=True,
        callback=reward_callback,
        tb_log_name=run_name or "SAC",
    )

    # Explicit --out is honoured as given (sweeps write to their own dirs);
    # otherwise the timestamped checkpoint lands in policies/.
    model_path = out or os.path.join(
        policy_dir(),
        f"cat_controller{cfg['suffix']}{tag}_{time.strftime('%Y%m%d-%H%M%S')}.zip")
    model.save(model_path)
    print(f"Model saved to {model_path}")
    return model_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=list(VARIANTS), default="tail",
                         help="ablation condition to train (default: tail)")
    parser.add_argument("--steps", type=int, default=1_000_000,
                         help="total training timesteps (default: 1000000)")
    parser.add_argument("--tag", default="",
                         help="extra suffix for the saved model filename")
    parser.add_argument("--envs", type=int, default=10,
                         help="parallel envs (default: 10; lower to share the box)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", default=None, help="explicit output .zip path")
    parser.add_argument("--gradient-steps", type=int, default=1,
                         help="gradient steps per vec-env step (default: 1)")
    parser.add_argument("--run-name", default=None, help="TensorBoard run name")
    parser.add_argument("--no-privileged", action="store_true",
                        help="train on the 25-dim obs without the DR block "
                             "(control arm for whether the privileged block helps)")
    parser.add_argument("--smooth-coef", type=float, default=2.0,
                        help="weight on the L2 action-smoothness term in the actor "
                             "loss (default: 10.0; 0 disables it, and train/smooth_loss "
                             "is still logged so the ablation is readable)")
    args = parser.parse_args()
    train(args.variant, args.steps, args.tag, args.envs, args.seed, args.out,
          args.gradient_steps, args.run_name, privileged=not args.no_privileged,
          smooth_coef=args.smooth_coef)
