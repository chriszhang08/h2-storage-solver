import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class TensorboardCallback(BaseCallback):
    """
    Custom SB3 callback that logs domain-specific H2 storage metrics to TensorBoard.

    Metrics logged:
      - trading/*   : per-step H2 inventory, utilization, cumulative injection/withdrawal
      - actions/*   : per-step withdraw/inject fractions; rollout-level mean & std
      - episode/*   : per-episode total reward, length

    Usage:
        model = PPO("MlpPolicy", env, tensorboard_log="./logs/tensorboard/")
        model.learn(total_timesteps=50_000, callback=TensorboardCallback(verbose=1))

    Then launch TensorBoard with:
        tensorboard --logdir ./logs/tensorboard/
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._ep_reward: float = 0.0
        self._ep_reward_adj: float = 0.0
        self._ep_boil_off: float = 0.0
        self._ep_len: int = 0
        self._ep_rewards_history: list[float] = []
        self._rollout_actions: list[np.ndarray] = []

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def _on_training_start(self) -> None:
        self._ep_reward = 0.0
        self._ep_len = 0
        self._ep_rewards_history = []
        self._rollout_actions = []

    def _on_rollout_start(self) -> None:
        self._rollout_actions = []

    def _on_step(self) -> bool:
        """Called after every env.step(). Returns False to abort training early."""
        reward = float(self.locals["rewards"][0])
        done = bool(self.locals["dones"][0])
        info = self.locals["infos"][0]
        action = self.locals["actions"][0]

        # Accumulate episode totals
        self._ep_reward += reward

        # Adjusted reward: realized PnL + boil-off loss (excludes potential/speculative component)
        self._ep_reward_adj += info["realized_avoided_cost"] + info["boil_off_loss"]
        self._ep_boil_off += info["boil_off_loss"]
        self._ep_len += 1
        self._rollout_actions.append(action)

        # --- Per-step trading metrics ---
        self.logger.record("trading/h2_inventory", info["h2_inventory"])
        self.logger.record("trading/boil_off_loss", info["boil_off_loss"])

        # Injection and withdrawal PnL statistics
        self.logger.record("actions/inject_profit", info["curr_potential_avoided_cost"])
        self.logger.record("actions/withdraw_profit", info["realized_avoided_cost"])
        self.logger.record("actions/injection_cost_basis", info["dollar_cost_basis"])

        # --- Episode-level metrics (logged when episode ends) ---
        if done:
            self._ep_rewards_history.append(self._ep_reward)
            self.logger.record("episode/total_reward", self._ep_reward)
            self.logger.record("episode/adjusted_reward", self._ep_reward_adj)
            self.logger.record("episode/boil_off_loss", self._ep_boil_off)

            # Reset episode accumulators
            self._ep_reward = 0.0
            self._ep_reward_adj = 0.0
            self._ep_boil_off = 0.0
            self._ep_len = 0

        return True

    def _on_rollout_end(self) -> None:
        """Called before the PPO policy update. Log rollout-level action statistics."""
        if not self._rollout_actions:
            return

        actions = np.array(
            self._rollout_actions
        ).flatten()  # shape (n_steps,), values in {0, 1, 2}
        n = len(actions)
        self.logger.record("actions/pct_withdraw", float(np.sum(actions == 0) / n))
        self.logger.record("actions/pct_nothing", float(np.sum(actions == 1) / n))
        self.logger.record("actions/pct_inject", float(np.sum(actions == 2) / n))

        # Log mean episode reward over all completed episodes so far
        if self._ep_rewards_history:
            self.logger.record(
                "episode/mean_reward_last20",
                float(np.mean(self._ep_rewards_history[-20:])),
            )
