import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import cat_env
import time

class TensorboardRewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # locals["infos"] is a list of info dictionaries from the vectorized environments
        for info in self.locals["infos"]:
            if "r_pos" in info:
                # Log each term under a "rewards/" group in TensorBoard
                self.logger.record("rewards/r_pos", info["r_pos"])
                self.logger.record("rewards/r_bonus", info["r_bonus"])
                self.logger.record("rewards/r_sm", info["r_sm"])
                self.logger.record("rewards/r_en", info["r_en"])
                self.logger.record("rewards/up_mean", info["up_mean"])
        return True

def train():
    num_cpu = 10

    env = make_vec_env(
        "Cat-v0",
        n_envs=num_cpu,
        vec_env_cls=SubprocVecEnv,
        #env_kwargs={"render_mode": "rgb_array"}
    )

    model = SAC(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        tensorboard_log="./run_logs/",
        device="cpu",
        seed=None,
        learning_rate=3e-4,
        buffer_size=300_000,
        learning_starts=10_000,   # collect some random transitions before learning
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,             # one gradient step per env step
        gradient_steps=1,
        ent_coef="auto",          # auto-tune entropy (key for multimodal exploration)
    )

    reward_callback = TensorboardRewardCallback()

    model.learn(
        total_timesteps=1_000_000,
        log_interval=20,
        progress_bar=True,
        callback=reward_callback
    )

    model_path = f"cat_controller_{time.strftime('%Y%m%d-%H%M%S')}.zip"
    model.save(model_path)
    print(f"Model saved")

if __name__ == "__main__":
    train()
