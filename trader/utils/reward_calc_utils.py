from typing import Tuple, Dict

import numpy as np
import pulp

from constants import STORAGE_CAPACITY, TRADING_DAYS_PER_MONTH, EPS, INJ_ALPHA, RISK_FREE_INT, MAX_INJECTION_RATE, \
    MAX_WITHDRAW_RATE, WITHDRAWAL_COST_PER_UNIT, INJECTION_COST_PER_UNIT, BOIL_OFF
from curve_factory.utils.curve_data_transformers import interpolate_forward_curve
from trader.action import InterpretedAction
from trader.state import EnvState


def storage_valuation(
    forward_curve: np.ndarray,
    inventory: float,
    end_inventory: float = 0.0,
    discount_rate: float = RISK_FREE_INT,
    max_withdraw_rate: float = MAX_WITHDRAW_RATE,
    max_inject_rate: float = MAX_INJECTION_RATE,
    granularity_of_action_per_month: int = 4,
    is_daily: bool = False,
    injection_allowed: bool = True,
) -> dict:
    if not is_daily:
        prices = interpolate_forward_curve(
            forward_curve,
            steps_per_period=granularity_of_action_per_month
        )
    else:
        prices = np.asarray(forward_curve, dtype=float)
        granularity_of_action_per_month = TRADING_DAYS_PER_MONTH

    prices = np.asarray(prices, dtype=float)
    T = len(prices)

    empty = {
        "npv": 0.0,
        "action_schedule": np.array([]),
        "inject_schedule": np.array([]),
        "withdraw_schedule": np.array([]),
        "inventory_path": np.array([]),
        "revenue_schedule": np.array([]),
        "purchase_cost_schedule": np.array([]),
        "net_cashflow_schedule": np.array([]),
        "discounted_cashflows": np.array([]),
        "discount_factors": np.array([]),
    }
    if T == 0:
        return empty

    dt_rate = discount_rate / (12 * granularity_of_action_per_month)
    discount_factors = np.array([1 / (1 + dt_rate) ** (t + 1) for t in range(T)])

    prob = pulp.LpProblem("storage_valuation", pulp.LpMaximize)

    inj = [
        pulp.LpVariable(f"inj_{t}", lowBound=0, upBound=max_inject_rate, cat=pulp.LpContinuous)
        for t in range(T)
    ]
    wd = [
        pulp.LpVariable(f"wd_{t}", lowBound=0, upBound=max_withdraw_rate, cat=pulp.LpContinuous)
        for t in range(T)
    ]

    if not injection_allowed:
        for t in range(T):
            prob += inj[t] == 0

    flow = [inj[t] - wd[t] for t in range(T)]

    prob += pulp.lpSum(
        discount_factors[t] * (prices[t] * wd[t] - prices[t] * inj[t]
                               - WITHDRAWAL_COST_PER_UNIT * wd[t]
                               - INJECTION_COST_PER_UNIT * inj[t])
        for t in range(T)
    )

    for t in range(T):
        inv_t = inventory + pulp.lpSum(flow[s] for s in range(t + 1))
        prob += inv_t >= 0
        prob += inv_t <= STORAGE_CAPACITY

    prob += inventory + pulp.lpSum(flow) == end_inventory

    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"LP solver failed: {pulp.LpStatus[status]}")

    inj_vals = np.array([pulp.value(v) or 0.0 for v in inj], dtype=float)
    wd_vals = np.array([pulp.value(v) or 0.0 for v in wd], dtype=float)
    flow_vals = inj_vals - wd_vals
    inv_vals = inventory + np.cumsum(flow_vals)

    revenue = prices * wd_vals - WITHDRAWAL_COST_PER_UNIT * wd_vals
    purchase_cost = prices * inj_vals + INJECTION_COST_PER_UNIT * inj_vals
    net_cashflow = revenue - purchase_cost
    discounted_cfs = discount_factors * net_cashflow

    return {
        "npv": float(discounted_cfs.sum()),
        "action_schedule": flow_vals,
        "inject_schedule": inj_vals,
        "withdraw_schedule": wd_vals,
        "inventory_path": inv_vals,
        "revenue_schedule": revenue,
        "purchase_cost_schedule": purchase_cost,
        "net_cashflow_schedule": net_cashflow,
        "discounted_cashflows": discounted_cfs,
        "discount_factors": discount_factors,
    }


# ========== CORE REWARD COMPUTATION ==========
def compute_reward(
    curr_state: EnvState, action: InterpretedAction
) -> Tuple[float, Dict]:
    """
    Compute total reward R(t) and break down by component.

    Args:
        curr_state: The current state.
        action (InterpretedAction): The action taken.

    Returns:
        reward_total: Scalar reward for RL agent.
        components: Dict of individual reward terms for diagnostics.
    """
    components = {}

    r_wdrw = _compute_withdrawal_reward(curr_state, action)
    components["realized_avoided_cost"] = r_wdrw

    r_pot = _compute_injection_reward(curr_state, action)
    components["curr_potential_avoided_cost"] = r_pot

    r_boil = _compute_boiloff_reward(curr_state, action)
    components["boil_off_loss"] = r_boil

    # Weighted sum
    reward_total = r_wdrw + INJ_ALPHA * r_pot + r_boil

    # update state's reward tracking data structures
    curr_state.realized_avoided_cost = np.append(
        curr_state.realized_avoided_cost, r_wdrw
    )
    curr_state.curr_potential_avoided_cost = np.append(
        curr_state.curr_potential_avoided_cost, r_pot
    )

    # Store for diagnostics
    components["reward_total"] = reward_total

    return reward_total, components


def compute_terminal_reward(final_state: EnvState) -> float:
    """
    Compute any final reward at end of episode (e.g., liquidation value of remaining inventory).

    DISABLED as it creates an unnecessary reward signal to the last action.
    """
    # value the remaining inventory at current spot price
    return final_state.h2_inventory * final_state.observe_current_price_info()["h2_spot_price"]


# ========== COMPONENT FUNCTIONS ==========
def _compute_withdrawal_reward(s: EnvState, a: InterpretedAction) -> float:
    """
    Immediate PnL as shown by spot basis.

    This equals a.h2_withdraw_dollars - avg_injection_dollars
    """
    if a.discrete_action != 0:
        return 0.0

    return (a.real_h2_spot - s.dollar_cost_basis) * a.h2_withdraw_units


def _compute_injection_reward(s: EnvState, a: InterpretedAction) -> float:
    """
    The injection reward is a signal to the agent that the current injection decision has the potential to avoid future costs,
    by filling storage at a lower cost basis than the current forward curve.
    """
    if a.discrete_action != 2:
        return 0.0

    injection_npv = storage_valuation(
        forward_curve=s.observe_current_price_info()["h2_fwd_curve"],
        inventory=a.h2_inject_units,
        end_inventory=0.0,
        granularity_of_action_per_month=1,  # evlauate on the monthly tenor average
        injection_allowed=False
    )
    injection_option_value_per_unit = injection_npv["npv"] / a.h2_inject_units

    # return the difference between the option value and the current cost of injection
    return (injection_option_value_per_unit - a.real_h2_spot) * a.h2_inject_units


def _compute_boiloff_reward(s: EnvState, a: InterpretedAction) -> float:
    """
    The penalty incurred from boil off, as a function of quantity boiled off.
    It reflects the cost of reinjection (the cost of injection) for the amount of units boiled off.
    Negative because the reward signal should reflect the cost incurred.

    Reward = action.boil_off_units * INJECTION_COST
    """
    return -INJECTION_COST_PER_UNIT * a.boil_off_units
