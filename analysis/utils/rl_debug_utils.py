"""
Post-training analysis for RL trading agents.

Compares an agent's action log against the LP-optimal storage schedule
and reports accuracy metrics.
"""

import json
import os
from dataclasses import dataclass
from typing import List

import numpy as np

from constants import MAX_INJECTION_RATE, MAX_WITHDRAW_RATE, EPS
from trader.utils.reward_calc_utils import storage_valuation


@dataclass
class AgentAccuracyReport:
    """Accuracy metrics comparing agent actions to the optimal schedule."""

    T: int
    hits: int
    misses: int
    double_misses: int
    score: int
    optimal_discrete: np.ndarray
    agent_discrete: np.ndarray

    @property
    def hit_rate(self) -> float:
        return self.hits / self.T if self.T > 0 else 0.0


def compute_optimal_bit_schedule(
    h2_spot_prices: np.ndarray,
    inventory: float = 0,
    end_inventory: float = 0,
    max_inject_rate: float = MAX_INJECTION_RATE,
    max_withdraw_rate: float = MAX_WITHDRAW_RATE,
    granularity_of_action_per_month: int = 1,
    is_daily: bool = True,
) -> np.ndarray:
    """
    Compute the LP-optimal storage action bit schedule from H2 spot prices.

    Args:
        h2_spot_prices: 1-D array of H2 spot prices (shape (T,)).
        inventory: Starting inventory level.
        end_inventory: Required terminal inventory.
        max_inject_rate: Maximum injection rate per step.
        max_withdraw_rate: Maximum withdrawal rate per step.
        granularity_of_action_per_month: Action granularity for the LP.

    Returns:
        1-D integer array in discrete action space {0, 1, 2} where
        0 = withdraw, 1 = hold, 2 = inject.
    """
    res = storage_valuation(
        h2_spot_prices,
        inventory=inventory,
        end_inventory=end_inventory,
        max_inject_rate=max_inject_rate,
        max_withdraw_rate=max_withdraw_rate,
        granularity_of_action_per_month=granularity_of_action_per_month,
        is_daily=is_daily
    )
    # get continuous schedule as numpy array
    cont_actions = np.asarray(res["action_schedule"], dtype=float)

    # threshold near-zero values to exactly zero
    cont_actions[np.abs(cont_actions) < EPS] = 0.0

    # Map sign to discrete: -1 -> 0 (withdraw), 0 -> 1 (hold), 1 -> 2 (inject)
    discrete = (np.sign(cont_actions).astype(int) + 1).astype(int)

    return discrete


def parse_agent_actions(log_dir: str) -> np.ndarray:
    """
    Parse discrete actions from the most recent complete JSONL episode log.

    Selects the second-to-last JSONL file (sorted alphabetically), which is
    typically the last fully completed episode.

    Args:
        log_dir: Path to the session's debug log directory.

    Returns:
        1-D integer array of discrete actions taken by the agent.
    """
    jsonl_files = sorted(f for f in os.listdir(log_dir) if f.endswith(".jsonl"))
    if len(jsonl_files) < 2:
        raise FileNotFoundError(
            f"Expected at least 2 JSONL files in {log_dir}, found {len(jsonl_files)}"
        )

    target_path = os.path.join(log_dir, jsonl_files[-2])
    print(f"Analysing: {target_path}")

    agent_actions: List[int] = []
    with open(target_path) as fh:
        for line in fh:
            record = json.loads(line)
            agent_actions.append(int(record["discrete_action"]))

    return np.array(agent_actions, dtype=int)


def evaluate_agent_against_lp(
    h2_spot_prices: np.ndarray,
    log_dir: str,
) -> AgentAccuracyReport:
    """
    Compare an RL agent's trading decisions against the LP-optimal schedule.

    Args:
        h2_spot_prices: 1-D array of H2 spot prices used to compute the
                        optimal bit schedule (typically h2_np_matrix[:, 0]).
        log_dir: Path to the session's debug log directory containing JSONL
                 episode logs.

    Returns:
        AgentAccuracyReport with hit/miss counts and aligned action arrays.
    """
    optimal_discrete = compute_optimal_bit_schedule(h2_spot_prices)
    agent_discrete = parse_agent_actions(log_dir)

    T = min(len(optimal_discrete), len(agent_discrete))
    opt = optimal_discrete[:T]
    agt = agent_discrete[:T]

    diff = np.abs(opt - agt)
    hits = int(np.sum(diff == 0))
    misses = int(np.sum(diff == 1))
    double_misses = int(np.sum(diff == 2))
    score = int(np.sum(diff))

    return AgentAccuracyReport(
        T=T,
        hits=hits,
        misses=misses,
        double_misses=double_misses,
        score=score,
        optimal_discrete=opt,
        agent_discrete=agt,
    )


def print_accuracy_report(report: AgentAccuracyReport) -> None:
    """Print a formatted accuracy report to stdout."""
    T = report.T
    print(f"\n=== Agent vs Optimal Bit Schedule ({T} steps) ===")
    print(
        f"  Hits         (|diff|=0): {report.hits:>5}  ({100 * report.hits / T:.1f}%)"
    )
    print(
        f"  Misses       (|diff|=1): {report.misses:>5}  ({100 * report.misses / T:.1f}%)"
    )
    print(
        f"  Double misses(|diff|=2): {report.double_misses:>5}  ({100 * report.double_misses / T:.1f}%)"
    )
    print(f"  Total score  (Σ|diff|) : {report.score:>5}  (lower = better)")
