from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any

import numpy as np

from trader.utils.analyst import encode_price_curve, normalize_spot_price, scale_spot_price
from constants import (
    STORAGE_CAPACITY,
    MAX_INJECTION_RATE,
    MAX_WITHDRAW_RATE,
)
from trader.action import InterpretedAction


@dataclass
class EnvState:
    """
    Snapshot of market and agent state at time t.
    Encapsulates all variables relevant to reward computation.
    """

    # === Market/Price State ===
    h2_spot_prices: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=float)
    )
    h2_fwd_curve: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=float)
    )  # shape (T,M)

    # === Agent Position/Inventory ===
    h2_inventory: float = 0  # H2 in storage (units)
    max_injection_rate: float = MAX_INJECTION_RATE
    max_withdrawal_rate: float = MAX_WITHDRAW_RATE
    dollar_cost_basis: float = 0.0  # Average cost basis of current inventory (EUR/unit)

    # === Time/Scenario Context ===
    time_step: int = 0  # Current step (0 to T)
    time_horizon: int = 0  # Total episode length (T)

    # === Historical/Statistics ===
    spot_h2_price_history: List[float] = field(default_factory=list)
    realized_avoided_cost: np.ndarray = field(default_factory=lambda: np.array([]))
    curr_potential_avoided_cost: np.ndarray = field(
        default_factory=lambda: np.array([])
    )
    action_history: List[int] = field(default_factory=list)

    # === Episode running totals (accumulated in update_state) ===
    total_inject_dollars:   float = 0.0
    total_withdraw_dollars: float = 0.0
    total_withdraw_units:   float = 0.0

    def __post_init__(self):
        self.time_horizon = len(self.h2_spot_prices)

    def update_state(self, act: int) -> InterpretedAction:
        """
        Update inventory and performance variables.

        :param act: (int): Discrete action — 0 = withdraw at max rate, 1 = do nothing, 2 = inject at max rate.
        :return: (InterpretedAction): The human-readable action object.
        """
        # Interpret discrete action into constrained injection/withdrawal decisions
        action: InterpretedAction = InterpretedAction.interpret(
            action=act,
            curr_h2_spot=self.observe_current_price_info()["h2_spot_price"],
            h2_inventory=self.h2_inventory,
            max_withdraw_rate=self.max_withdrawal_rate,
            max_inject_rate=self.max_injection_rate,
        )

        # boil off some of the inventory
        self.h2_inventory -= action.boil_off_units

        self.spot_h2_price_history.append(
            self.observe_current_price_info()["h2_spot_price"]
        )

        # update dollar cost basis IFF agent is injecting
        if act == 2:
            # start with previous total cost basis
            total_cost = self.dollar_cost_basis * self.h2_inventory
            total_cost += action.h2_inject_dollars
            self.dollar_cost_basis = total_cost / (self.h2_inventory + action.h2_inject_units)

        # update inventory
        self.h2_inventory += action.h2_inject_units - action.h2_withdraw_units

        self.h2_inventory = min(self.h2_inventory, STORAGE_CAPACITY)
        self.h2_inventory = max(self.h2_inventory, 0.0)

        # accumulate episode totals
        self.total_inject_dollars   += action.h2_inject_dollars
        self.total_withdraw_dollars += action.h2_withdraw_dollars
        self.total_withdraw_units   += action.h2_withdraw_units

        return action

    def observe_state(self) -> np.ndarray:
        """
        Build observation vector from internal state + current curve slice.
        Convert structured observation to normalized Box vector for RL policy.

        Returns:
            np.ndarray, dtype=np.float32
        """
        # Inventory observations
        inventory_pct = np.clip(self.h2_inventory / STORAGE_CAPACITY, 0.0, 1.0)

        max_withdraw_norm = self.max_withdrawal_rate / STORAGE_CAPACITY
        max_inject_norm = self.max_injection_rate / STORAGE_CAPACITY

        # put the h2 spot price in relative terms of the historical spot prices
        h2_spot_price_norm, sample_mean, sample_std = normalize_spot_price(self.spot_h2_price_history)

        # put the dollar cost norm in terms of the h2_spot_price_norm z-score
        dollar_cost_norm = scale_spot_price(self.dollar_cost_basis, sample_mean, sample_std)
        scalars = np.array(
            [inventory_pct, h2_spot_price_norm, dollar_cost_norm, max_withdraw_norm, max_inject_norm],
            dtype=np.float32,
        )

        # 1. Normalize current observed forward curves for agent input
        h2_fwd_curve_norm = encode_price_curve(
            self.observe_current_price_info()["h2_fwd_curve"],
            sample_mean,
            sample_std
        )  # shape (36,)

        return np.concatenate([h2_fwd_curve_norm, scalars]).astype(
            np.float32
        )  # shape (41,)

    def observe_current_price_info(self) -> Dict[str, Any]:
        """
        Return a dict of current price information for reward calculation and logging.
        This includes the current H2 spot price and the average cost basis of inventory.
        """
        return {
            "h2_spot_price": self.h2_spot_prices[self.time_step],
            "h2_fwd_curve": self.h2_fwd_curve[self.time_step],
        }

    def get_debug_info(self, action: InterpretedAction) -> Dict:
        """
        Return a dict of internal state variables for logging/debugging.
        This information is used in callbacks to log trading metrics and diagnostics at each step.
        """
        info = {
            "dollar_cost_basis": self.dollar_cost_basis,
        }
        if action is not None:
            info.update(action.to_dict())
        return info

    def is_terminal(self) -> Tuple[bool, bool]:
        """
        Check if the episode should terminate if simulation horizon has ended.

        Returns:
            terminated (bool): False for now.
            truncated (bool): True if time horizon is reached.
        """
        terminated = False
        truncated = self.time_step >= self.time_horizon - 1
        return terminated, truncated
