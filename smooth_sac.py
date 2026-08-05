"""SAC with a temporal action-smoothness term in the ACTOR loss.

Replaces the old `w_sm * mean(da^2)` reward penalty (removed from
cat_env::_get_reward). Same goal -- commands the real geared motors can track --
but paid for in the policy objective instead of the return:

    L_smooth = || pi_mu(s_{t+1}) - pi_mu(s_t) ||^2

where `pi_mu` is the DETERMINISTIC mean action, tanh(mu), not a sample from the
squashed Gaussian. Two reasons this matters more than it looks:

  1. The sampled action carries SAC's exploration noise, which never reaches
     hardware. Penalizing it prices jitter the deployed policy does not have,
     and -- because the noise scale is `log_std`, itself a network output --
     hands the actor a second, degenerate way to satisfy the term: shrink the
     entropy. That fights the ent_coef auto-tuner directly.
  2. tanh(mu) is exactly what `predict(deterministic=True)`, the ONNX export and
     the Teensy run, so the quantity being smoothed is the commanded trajectory
     itself.

Why the actor loss rather than the reward: as a reward penalty, smoothness is
routed through the critic -- the actor only ever sees it after the TD backup has
learned it, and it trades against task reward inside the same scalar Q, which is
what made `w_sm` a collapse risk (a variant that cannot reliably right itself
scores better holding still; docs/REWARD_TUNING.md). Here the gradient reaches
the actor directly, on every gradient step, and cannot make passivity look
optimal to the critic.

`smooth_coef` does NOT carry over from `w_sm` -- it acts on the return, not on the
per-step reward. The default 10.0 is sized to reproduce the pressure `w_sm` used to
apply, measured on the shipped tail teacher (`cat_controller.zip`):

    ||pi_mu(s_t+1) - pi_mu(s_t)||^2  = 0.204 mean (0.54 p90, 3.74 max)
    |min_qf_pi|                      = 29.4 mean
    w_sm's swept grid position       = 6% of task reward per step

    coef = 0.06 * 29.4 / 0.204 ~= 9  ->  10

That equates loss *magnitudes*, not gradient magnitudes, so it was a starting point
rather than a tuned value -- but it holds up measured. Tail, 1M steps, seed 0, 500
drops (docs/REWARD_TUNING.md has the full table):

    arm                     success   m_dsm    m_en   m_jv
    shipped (old w_sm)       57.2%    0.197   0.351   82.0
    smooth_coef = 10         56.8%    0.0128  0.228   55.5
    smooth_coef = 0           44.0%   0.277   0.378   66.9

Deleting the penalty and NOT replacing it costs 13.2 pp; this term recovers all of
it (0.4 pp is inside the ~2.2 pp binomial se) while running 15x smoother than the
teacher it replaces. Still only one seed per arm, and neither 3 nor 30 was tried.

Note the quantity is not the old `m_sm`: `m_sm` averaged the SAMPLED action delta
over 3 dims and read ~0.25 at baseline, which is mostly exploration noise -- this
one sums the deterministic delta over 3 dims and is ~80x smaller at initialization
(0.003), growing only as the policy learns actual maneuvers. Tune against
`train/smooth_loss`, logged whether or not the coefficient is zero -- the same
convention as the env's unweighted `m_*` magnitudes -- and against `m_dsm` from
`docs/evaluate.py --stats`, which is the same quantity on the deployed policy.
"""
import numpy as np
import torch as th
from torch.nn import functional as F
from stable_baselines3 import SAC
from stable_baselines3.common.utils import polyak_update


class SmoothSAC(SAC):
    """SAC whose actor loss carries an L2 action-smoothness term.

    Drop-in for SAC. Checkpoints stay loadable by plain `SAC.load` (the policy is
    unchanged -- only the training objective differs), which is what
    distillation.py, docs/evaluate.py and onnx_conversion.py rely on.
    """

    def __init__(self, *args, smooth_coef: float = 10.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.smooth_coef = smooth_coef

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Body is stable-baselines3 2.7.1 SAC.train() with the smoothness term
        # added at the actor loss; kept line-for-line otherwise so a version bump
        # diffs cleanly against upstream.
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses, smooth_losses = [], [], []

        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            # For n-step replay, discount factor is gamma**n_steps (when no early termination)
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                ent_coef = th.exp(self.log_ent_coef.detach())
                assert isinstance(self.target_entropy, float)
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)
            critic_losses.append(critic_loss.item())

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # --- action smoothness, on the deterministic mean ---
            # `actor(obs, deterministic=True)` is tanh(mu) -- the mean of the
            # squashed Gaussian, with no reparameterized sample in it. Called
            # AFTER action_log_prob above: both share `actor.action_dist`, and
            # the call mutates its stored Normal. Harmless here (log_prob is
            # already a tensor and its graph does not read the object back), but
            # it is why this block cannot be hoisted above the sampling.
            #
            # (s_t, s_{t+1}) is exactly the consecutive pair the term wants: SB3
            # stores the true `terminal_observation` for truncated steps, and
            # this env only ever truncates (_is_terminated is always False), so
            # no sampled pair straddles an episode boundary. Add a `(1 - dones)`
            # mask here if that ever stops being true.
            pi_t = self.actor(replay_data.observations, deterministic=True)
            pi_next = self.actor(replay_data.next_observations, deterministic=True)
            # Squared L2 per transition (summed over the 3 action dims), averaged
            # over the batch.
            smooth_loss = ((pi_next - pi_t) ** 2).sum(dim=1).mean()
            smooth_losses.append(smooth_loss.item())

            # Compute actor loss
            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean() + self.smooth_coef * smooth_loss
            actor_losses.append(actor_loss.item())

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        # Unweighted, like the env's m_* magnitudes: what the coefficient prices,
        # readable at any coefficient including zero.
        self.logger.record("train/smooth_loss", np.mean(smooth_losses))
        self.logger.record("train/smooth_coef", self.smooth_coef)
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
