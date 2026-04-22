import json
import logging
import os
import gymnasium as gym
import numpy as np
from typing import Optional, Dict, Any

from numpy import ndarray

from constants import Z_SCORE_CLIP, STORAGE_CAPACITY, NUM_M
from trader.state import EnvState
from trader.utils.reward_calc_utils import compute_reward
from gymnasium import spaces


class TradingEnv(gym.Env):
    """
    Template Gymnasium environment for PPO-based TTF curve trading.

    API follows Gymnasium v0.29+:
      - reset(seed=None, options=None) -> (obs, info)
      - step(action) -> (obs, reward, terminated, truncated, info)
    """

    metadata = {
        "render_modes": ["human", "none"],
        "render_fps": 4,
    }

    def __init__(
        self,
        h2_np_matrix: np.ndarray,
        render_mode: Optional[str] = None,
        seed: int = 42,
        verbosity: int = 0,
        log_dir: Optional[str] = None,
    ):
        """
        Args:
            render_mode: "human" or None.
            verbosity: 0 = no logging, 1 = write per-step debug records to file.
            log_dir: Directory for debug log files. Defaults to ./logs/ when verbosity > 0.
        """
        super().__init__()

        # ----------------------------
        # 1. Store config / data
        # ----------------------------
        self.seed = seed
        self.verbosity = verbosity
        self._log_dir = log_dir or os.path.join(os.getcwd(), "logs")
        self._episode = 0
        self._step_logger: Optional[logging.Logger] = None

        self.h2_np_matrix = h2_np_matrix  # shape (T, NUM_M + 1) full h2 shape (T, M) forward curve with spot price in column 0

        self.render_mode = render_mode

        # ----------------------------
        # 2. Action and observation spaces
        # ----------------------------
        # ACTION SPACE:
        #  0 = withdraw at max withdrawal rate
        #  1 = do nothing
        #  2 = inject at max injection rate
        self.action_space = spaces.Discrete(3)

        # OBSERVATION SPACE: 14-element vector matching observe_state() output.
        #
        # Indices 0–9: encoded forward curve (2 years × 5 features each):
        #   [front_month, first_extreme, first_extreme_month_idx, second_extreme, second_extreme_month_idx]
        #   Price features are z-score normalised (unbounded); month indices are in [0, 11].
        # Indices 10–13: scalars
        #   [inventory_pct ∈ [0,1], h2_spot_price_norm (z-score), dollar_cost_basis (z-score) max_withdraw_norm ∈ [0,1], max_inject_norm ∈ [0,1]]
        low_vec = np.concatenate(
            (
                np.full(NUM_M, -Z_SCORE_CLIP, dtype=np.float32),  # price curve observation
                np.array([0, -Z_SCORE_CLIP, -Z_SCORE_CLIP, 0, 0], dtype=np.float32),  # scalar metadata observations
            )
        )
        high_vec = np.concatenate(
            (
                np.full(NUM_M, Z_SCORE_CLIP, dtype=np.float32),  # price curve observation
                np.array([1, Z_SCORE_CLIP, Z_SCORE_CLIP, 1, 1], dtype=np.float32),  # scalar metadata observations
            )
        )
        num_obs = NUM_M + 5
        self.observation_space = spaces.Box(
            low=low_vec, high=high_vec, shape=(num_obs,), dtype=np.float32
        )

        self.state: EnvState | None = None

    # ------------------------------------------------------------------ #
    # CORE GYM API
    # ------------------------------------------------------------------ #

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[Any, Any]]:
        """
        Reset environment to initial state.

        Returns:
            obs: Initial observation (np.ndarray)
            info: Dict with optional debug info
        """
        super().reset(seed=seed)

        # Reset portfolio / agent state
        self.state = EnvState(
            h2_spot_prices=self.h2_np_matrix[
                :, 0
            ],  # shape (T,) spot price is in column 0
            h2_fwd_curve=self.h2_np_matrix[
                :, 1 : self.h2_np_matrix.shape[1]
            ],  # shape (T, M) forward curve with spot price in column 0
        )

        self._episode += 1
        self._reset_step_logger()

        return self.state.observe_state(), {}

    def _to_serializable(self, obj):
        """Recursively convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64, np.integer)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: self._to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._to_serializable(v) for v in obj]
        else:
            return obj

    def step(
        self,
        act: int,
    ) -> tuple[ndarray, float, bool, bool, dict[str, Any]]:
        """
        Apply action, advance environment one step.

        Args:
            act: Action chosen by policy - shape (2,)

        Returns:
            obs: Next observation
            reward: Scalar reward
            terminated: True if terminal state reached (natural episode end)
            truncated: True if episode was cut short (time limit, etc.)
            info: Extra logging/debug info (not used for learning)
        """
        # apply action to update internal state
        action = self.state.update_state(act)

        reward, reward_debug = compute_reward(self.state, action)

        # snapshot of current state for observation and info
        obs = self.state.observe_state()
        info = self.state.get_debug_info(action)

        # Update action history and time step
        self.state.action_history.append(act)
        self.state.time_step += 1

        # terminated if bankrupt or time limit reached
        terminated, truncated = self.state.is_terminal()

        info.update(reward_debug)

        if terminated or truncated:
            info["episode_summary"] = self._build_episode_summary()

        if self.verbosity >= 2 and self._step_logger is not None:
            record = {
                "episode": self._episode,
                "step": self.state.time_step,
                "mkt_h2_spot": self.state.spot_h2_price_history[-1],
                **info,
            }
            # Convert all numpy types to native Python types
            serializable_record = self._to_serializable(record)
            self._step_logger.info(json.dumps(serializable_record))

        return obs, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """
        Return a boolean mask over the 3 discrete actions for MaskablePPO.

        - action 0 (withdraw): forbidden when inventory is empty (nothing to withdraw)
        - action 1 (nothing):  always allowed
        - action 2 (inject):   forbidden when storage is full (no capacity remaining)
        """
        mask = np.ones(3, dtype=bool)
        if self.state is not None:
            if self.state.h2_inventory <= 0:
                mask[0] = False
            if self.state.h2_inventory >= STORAGE_CAPACITY:
                mask[2] = False
        return mask

    def render(self) -> None:
        """
        Render the environment (for human debugging).

        You can print key metrics, plot curves, etc.
        """
        if self.render_mode == "human":
            print(
                f"Step {self.state.time_step} | "
                f"Inventory: {self.state.h2_inventory:,.0f}"
            )

    def close(self):
        """Clean up resources if needed (files, figures, etc.)."""
        self._close_step_logger()

    # ------------------------------------------------------------------ #
    # DEBUG LOGGING HELPERS
    # ------------------------------------------------------------------ #

    def _build_episode_summary(self) -> dict:
        """Compute per-episode metrics from state accumulators (no file I/O)."""
        s = self.state
        spots   = np.array(s.spot_h2_price_history)
        actions = np.array(s.action_history)

        inject_spots   = spots[actions == 2] if (actions == 2).any() else np.array([])
        withdraw_spots = spots[actions == 0] if (actions == 0).any() else np.array([])

        lcoi = float(np.mean(inject_spots))   if len(inject_spots)   > 0 else float("nan")
        lcow = float(np.mean(withdraw_spots)) if len(withdraw_spots) > 0 else float("nan")

        final_spot = float(s.spot_h2_price_history[-1]) if s.spot_h2_price_history else 0.0
        s.total_withdraw_dollars += s.h2_inventory * final_spot
        total_cashflow = (
            s.total_withdraw_dollars - s.total_inject_dollars
        )
        return {
            "levelized_cost_of_injection":   lcoi,
            "levelized_cost_of_withdrawal":  lcow,
            "total_cashflow":                total_cashflow,
            "withdraw_cashflow":             s.total_withdraw_dollars,
            "total_withdrawal_units":        s.total_withdraw_units + s.h2_inventory,
            "final_inventory":               s.h2_inventory,
            "final_spot":                    final_spot,
        }

    def _reset_step_logger(self) -> None:
        """Open a new JSON-lines log file for the current episode (verbosity >= 2 only)."""
        self._close_step_logger()
        if self.verbosity < 2:
            return

        os.makedirs(self._log_dir, exist_ok=True)
        log_path = os.path.join(self._log_dir, f"episode_{self._episode:04d}.jsonl")

        logger = logging.getLogger(f"trading_env.ep{self._episode}")
        logger.setLevel(logging.INFO)
        logger.propagate = False  # do not bubble up to the root logger

        handler = logging.FileHandler(log_path, mode="w")
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)

        self._step_logger = logger

    def _close_step_logger(self) -> None:
        """Flush and remove all handlers from the current step logger."""
        if self._step_logger is None:
            return
        for handler in list(self._step_logger.handlers):
            handler.flush()
            handler.close()
            self._step_logger.removeHandler(handler)
        self._step_logger = None
