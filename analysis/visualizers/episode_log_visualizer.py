"""
EpisodeLogVisualizer: H2 Storage Trading Episode Log Parser & Visualiser
================================================================================
Ingests the JSON-lines debug logs produced by TradingEnv and generates two
families of diagnostics:

  IntraEpisodeVisualizer  — step-level decisions within a single episode
  CrossEpisodeVisualizer  — episode-level summary metrics across a training run

Each log line has the schema:
    {
        "episode": int,
        "step": int,
        "mkt_h2_spot": float,
        "dollar_cost_basis": float,
        "discrete_action": int,
        "real_h2_spot": float,
        "h2_inventory": float,
        "max_inject_rate": float,
        "max_withdraw_rate": float,
        "h2_withdraw_units": float,
        "h2_withdraw_dollars": float,
        "h2_inject_units": float,
        "h2_inject_dollars": float,
        "realized_avoided_cost": float,
        "curr_potential_avoided_cost": float,
        "boil_off_loss": float,
        "reward_total": float
    }

USAGE:
------
    # Intra-episode: drill into one episode
    viz = IntraEpisodeVisualizer()
    viz.load("logs/run_abc/")
    viz.generate_diagnostic_suite(episode=56, output_dir="results/ep56/")

    # Cross-episode: track learning progress
    viz = CrossEpisodeVisualizer()
    viz.load("logs/run_abc/")
    viz.generate_diagnostic_suite(output_dir="results/run_abc/")
"""

import json
import os
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from analysis.utils import compute_optimal_bit_schedule, evaluate_agent_against_lp, print_accuracy_report


plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")

_REWARD_COMPONENTS = [
    "realized_avoided_cost",
    "curr_potential_avoided_cost",
    "boil_off_loss",
]
_REWARD_COLORS = {
    "realized_avoided_cost": "#2ecc71",
    "curr_potential_avoided_cost": "#3498db",
    "boil_off_loss": "#e67e22",
}
_ACTION_COLORS = {0: "#e74c3c", 1: "#95a5a6", 2: "#2ecc71"}
_ACTION_LABELS = {0: "Withdraw", 1: "Hold", 2: "Inject"}


# ============================================================================ #
# DATA LOADING / FLATTENING  (shared by both visualizers)
# ============================================================================ #


def _read_jsonl(path: Path) -> List[dict]:
    records = []
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _flatten(records: List[dict]) -> pd.DataFrame:
    """
    Flatten log records into a DataFrame.

    Handles two schemas:
    - Flat (current TradingEnv): reward fields are top-level keys.
    - Legacy nested: reward fields are in a "reward_components" sub-dict
      and are promoted to rc_* columns for backward compatibility.
    """
    rows = []
    for r in records:
        row = {k: v for k, v in r.items() if k != "reward_components"}
        for k, v in (r.get("reward_components") or {}).items():
            row.setdefault(k, v)  # flat name takes priority
        rows.append(row)
    df = pd.DataFrame(rows)
    df.sort_values(["episode", "step"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def _load_path(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.is_dir():
        files = sorted(p.glob("*.jsonl"))
        if not files:
            raise FileNotFoundError(f"No .jsonl files found in {path}")
        records = []
        for f in files:
            records.extend(_read_jsonl(f))
    elif p.is_file():
        records = _read_jsonl(p)
    else:
        raise FileNotFoundError(f"Path not found: {path}")
    df = _flatten(records)
    print(f"Loaded {len(df):,} steps | episodes: {sorted(df['episode'].unique())}")
    return df


def _save_or_show(
    fig: plt.Figure, save_path: Optional[str], dpi: int = 150, show: bool = True
):
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def _max_drawdown(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    peak = np.maximum.accumulate(values)
    dd = (values - peak) / (np.abs(peak) + 1e-8)
    return float(-dd.min())


# ============================================================================ #
# INTRA-EPISODE VISUALIZER
# ============================================================================ #


class IntraEpisodeVisualizer:
    """
    Step-level diagnostic plots for a single episode.

    Panels
    ------
    1. H2 spot price over episode
    2. Dollar cost basis over episode
    3. H2 inventory levels
    4. realized avoided cost over episode (are withdrawals making money)
    5. curr potential avoided cost over episode (are positive injections made)
    6. Agent action schedule overlaid with optimal action schedule

    Usage
    -----
        viz = IntraEpisodeVisualizer()
        viz.load("logs/run_abc/")
        viz.generate_diagnostic_suite(episode=56, output_dir="results/ep56/")
    """

    def __init__(self, figsize_width: int = 14, dpi: int = 150):
        self.figsize_width = figsize_width
        self.dpi = dpi
        self.df: Optional[pd.DataFrame] = None

    def load(self, path: str) -> "IntraEpisodeVisualizer":
        self.df = _load_path(path)
        return self

    def load_df(self, df: pd.DataFrame) -> "IntraEpisodeVisualizer":
        """Attach a pre-flattened DataFrame directly."""
        self.df = df.copy()
        return self

    # ------------------------------------------------------------------ #

    def generate_diagnostic_suite(
        self,
        episode: Optional[int] = None,
        output_dir: Optional[str] = None,
        show: bool = True,
    ) -> None:
        """
        Produce all intra-episode panels for *episode* (defaults to last episode).
        Files written:
            01_spot_and_inventory.png
            02_dollar_cost_basis.png
            03_realized_avoided_cost.png
            04_potential_avoided_cost.png
            05_action_vs_optimal.png
        """
        self._check_loaded()
        ep = episode if episode is not None else self.df["episode"].max()
        sub = self.df[self.df["episode"] == ep].copy()
        if sub.empty:
            raise ValueError(f"Episode {ep} not found in loaded data.")

        prefix = f"Ep {ep} | {len(sub)} steps"
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        def _path(name):
            return os.path.join(output_dir, name) if output_dir else None

        print(f"\n{'='*65}")
        print(f"  Intra-episode diagnostics  →  {prefix}")
        print(f"{'='*65}")

        _save_or_show(self.plot_spot_and_inventory(ep), _path("01_spot_and_inventory.png"), self.dpi, show)
        _save_or_show(self.plot_dollar_cost_basis(ep), _path("02_dollar_cost_basis.png"), self.dpi, show)
        _save_or_show(self.plot_realized_avoided_cost(ep), _path("03_realized_avoided_cost.png"), self.dpi, show)
        _save_or_show(self.plot_potential_avoided_cost(ep), _path("04_potential_avoided_cost.png"), self.dpi, show)
        _save_or_show(self.plot_action_vs_optimal(ep), _path("05_action_vs_optimal.png"), self.dpi, show)

    # ------------------------------------------------------------------ #
    # Individual panels
    # ------------------------------------------------------------------ #

    def plot_spot_and_inventory(self, episode: Optional[int] = None) -> plt.Figure:
        """H₂ spot price and inventory level overlaid on a shared step axis.
        Spot price on the left y-axis; inventory with capacity bounds on the right."""
        from constants import STORAGE_CAPACITY

        sub, ep = self._get_episode(episode)
        steps = sub["step"].values
        spot = sub["mkt_h2_spot"].values
        inv = sub["h2_inventory"].values
        cap = STORAGE_CAPACITY

        fig, ax1 = plt.subplots(figsize=(self.figsize_width, 4))
        ax2 = ax1.twinx()

        # Spot price — left axis
        ax1.plot(steps, spot, color="steelblue", linewidth=1.8, label="H2 spot price")
        ax1.fill_between(steps, spot.min(), spot, alpha=0.10, color="steelblue")
        ax1.set_ylabel("H2 Spot Price ($/kg)", color="steelblue")
        ax1.tick_params(axis="y", labelcolor="steelblue")
        ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.2f}"))

        # Inventory — right axis
        ax2.plot(steps, inv, color="darkorange", linewidth=1.6, label="H2 inventory", alpha=0.85)
        ax2.fill_between(steps, 0, inv, alpha=0.12, color="darkorange")
        ax2.set_ylabel("H2 Inventory (units)", color="darkorange")
        ax2.tick_params(axis="y", labelcolor="darkorange")
        ax2.set_ylim(-cap * 0.05, cap * 1.1)

        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(handles1 + handles2, labels1 + labels2, fontsize=9, loc="upper left")

        ax1.set_xlabel("Step")
        ax1.set_title(f"H2 Spot Price & Inventory", fontweight="bold")
        fig.tight_layout()
        return fig

    def plot_spot_price(self, episode: Optional[int] = None) -> plt.Figure:
        """H2 spot price over episode steps."""
        sub, ep = self._get_episode(episode)
        steps = sub["step"].values
        spot = sub["mkt_h2_spot"].values

        fig, ax = plt.subplots(figsize=(self.figsize_width, 4))
        ax.plot(steps, spot, color="steelblue", linewidth=1.8, label="H2 spot price")
        ax.fill_between(steps, spot.min(), spot, alpha=0.12, color="steelblue")

        ax.set_title(f"H2 Spot Price — Ep {ep}", fontweight="bold")
        ax.set_xlabel("Step")
        ax.set_ylabel("H2 Spot Price (€/kg)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.2f}"))
        ax.text(
            0.01, 0.97,
            f"min={spot.min():.2f}  max={spot.max():.2f}  mean={spot.mean():.2f}",
            transform=ax.transAxes, fontsize=9, va="top",
        )
        ax.legend(fontsize=9)
        fig.tight_layout()
        return fig

    def plot_dollar_cost_basis(self, episode: Optional[int] = None) -> plt.Figure:
        """Dollar cost basis vs H2 spot price over episode steps."""
        sub, ep = self._get_episode(episode)
        steps = sub["step"].values
        spot = sub["mkt_h2_spot"].values
        basis = sub["dollar_cost_basis"].values

        fig, ax = plt.subplots(figsize=(self.figsize_width, 4))
        ax.plot(steps, spot, color="steelblue", linewidth=1.5, alpha=0.7, label="H2 spot price")
        ax.plot(steps, basis, color="darkorange", linewidth=1.8, label="Dollar cost basis")

        # shade spread between basis and spot
        ax.fill_between(
            steps, basis, spot,
            where=(spot >= basis), alpha=0.15, color="#2ecc71", label="Spread > 0 (profitable)",
        )
        ax.fill_between(
            steps, basis, spot,
            where=(spot < basis), alpha=0.15, color="#e74c3c", label="Spread < 0 (loss)",
        )

        ax.set_title(f"Dollar Cost Basis vs Spot Price — Ep {ep}", fontweight="bold")
        ax.set_xlabel("Step")
        ax.set_ylabel("€/kg")
        ax.legend(fontsize=9)
        fig.tight_layout()
        return fig

    def plot_inventory_path(self, episode: Optional[int] = None) -> plt.Figure:
        """H2 inventory over steps with capacity bounds and rate annotations."""
        sub, ep = self._get_episode(episode)

        from constants import STORAGE_CAPACITY

        cap = STORAGE_CAPACITY
        steps = sub["step"].values
        inv = sub["h2_inventory"].values

        fig, ax = plt.subplots(figsize=(self.figsize_width, 4))
        ax.plot(steps, inv, color="darkorange", linewidth=1.8, label="H2 inventory")
        ax.fill_between(steps, 0, inv, alpha=0.15, color="darkorange")
        ax.axhline(cap, color="red", linestyle="--", linewidth=1.0, label=f"Capacity ({cap:,})")
        ax.axhline(0, color="navy", linestyle="--", linewidth=1.0, label="Floor (0)")

        max_inj = sub["max_inject_rate"].iloc[0] if "max_inject_rate" in sub.columns else None
        max_wdr = sub["max_withdraw_rate"].iloc[0] if "max_withdraw_rate" in sub.columns else None
        rate_txt = ""
        if max_inj is not None:
            rate_txt = f"  max_inject={max_inj:.0f}  max_withdraw={max_wdr:.0f}"

        ax.set_title(f"H2 Inventory Path — Ep {ep}{rate_txt}", fontweight="bold")
        ax.set_xlabel("Step")
        ax.set_ylabel("H2 Inventory (units)")
        ax.set_ylim(-cap * 0.05, cap * 1.1)
        ax.legend(fontsize=9)
        fig.tight_layout()
        return fig

    def plot_realized_avoided_cost(self, episode: Optional[int] = None) -> plt.Figure:
        """Realized avoided cost per step — profit from withdrawals."""
        sub, ep = self._get_episode(episode)
        steps = sub["step"].values
        vals = sub["realized_avoided_cost"].fillna(0).values

        fig, ax = plt.subplots(figsize=(self.figsize_width, 4))
        colors = ["#27ae60" if v >= 0 else "#e74c3c" for v in vals]
        ax.bar(steps, vals, color=colors, width=0.8, alpha=0.85, label="Realized avoided cost")
        ax.plot(
            steps, pd.Series(vals).cumsum().values,
            color="navy", linewidth=1.5, linestyle="--", label="Cumulative",
        )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(f"Realized Avoided Cost (Withdrawal PnL) — Ep {ep}", fontweight="bold")
        ax.set_xlabel("Step")
        ax.set_ylabel("Reward (kEUR)")
        ax.text(
            0.01, 0.97,
            f"sum={vals.sum():.4f}  mean={vals.mean():.4f}",
            transform=ax.transAxes, fontsize=9, va="top",
        )
        ax.legend(fontsize=9)
        fig.tight_layout()
        return fig

    def plot_potential_avoided_cost(self, episode: Optional[int] = None) -> plt.Figure:
        """Current potential avoided cost per step — option value of injected inventory."""
        sub, ep = self._get_episode(episode)
        steps = sub["step"].values
        vals = sub["curr_potential_avoided_cost"].fillna(0).values

        fig, ax = plt.subplots(figsize=(self.figsize_width, 4))
        colors = ["#3498db" if v >= 0 else "#e74c3c" for v in vals]
        ax.bar(steps, vals, color=colors, width=0.8, alpha=0.85, label="Potential avoided cost")
        ax.plot(
            steps, pd.Series(vals).cumsum().values,
            color="navy", linewidth=1.5, linestyle="--", label="Cumulative",
        )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(f"Potential Avoided Cost (Injection Option Value) — Ep {ep}", fontweight="bold")
        ax.set_xlabel("Step")
        ax.set_ylabel("Reward (kEUR)")
        ax.text(
            0.01, 0.97,
            f"sum={vals.sum():.4f}  mean={vals.mean():.4f}",
            transform=ax.transAxes, fontsize=9, va="top",
        )
        ax.legend(fontsize=9)
        fig.tight_layout()
        return fig

    def plot_action_vs_optimal(self, episode: Optional[int] = None) -> plt.Figure:
        """
        Agent discrete action schedule vs LP-optimal schedule.

        Computes the LP-optimal schedule from the episode's spot prices
        using compute_optimal_bit_schedule, then compares step-by-step.
        """
        sub, ep = self._get_episode(episode)
        steps = sub["step"].values
        agent_actions = sub["discrete_action"].values

        spot_prices = sub["mkt_h2_spot"].values
        optimal_actions = compute_optimal_bit_schedule(spot_prices)
        T = min(len(agent_actions), len(optimal_actions))
        agent_actions = agent_actions[:T]
        optimal_actions = optimal_actions[:T]
        steps = steps[:T]

        hit_rate = float(np.mean(agent_actions == optimal_actions))

        fig, axes = plt.subplots(3, 1, figsize=(self.figsize_width, 8), sharex=True)

        # Panel 1: Agent actions
        for action_val, color, label in [
            (2, "#2ecc71", "Inject"), (1, "#95a5a6", "Hold"), (0, "#e74c3c", "Withdraw")
        ]:
            mask = agent_actions == action_val
            axes[0].scatter(
                steps[mask], np.full(mask.sum(), action_val),
                c=color, s=8, label=label, zorder=3,
            )
        axes[0].set_yticks([0, 1, 2])
        axes[0].set_yticklabels(["Withdraw", "Hold", "Inject"])
        axes[0].set_title(f"Agent Actions — Ep {ep}", fontweight="bold")
        axes[0].legend(fontsize=8, loc="upper right")

        # Panel 2: Optimal actions
        for action_val, color, label in [
            (2, "#2ecc71", "Inject"), (1, "#95a5a6", "Hold"), (0, "#e74c3c", "Withdraw")
        ]:
            mask = optimal_actions == action_val
            axes[1].scatter(
                steps[mask], np.full(mask.sum(), action_val),
                c=color, s=8, label=label, zorder=3,
            )
        axes[1].set_yticks([0, 1, 2])
        axes[1].set_yticklabels(["Withdraw", "Hold", "Inject"])
        axes[1].set_title("LP-Optimal Actions", fontweight="bold")
        axes[1].legend(fontsize=8, loc="upper right")

        # Panel 3: Agreement / disagreement
        diff = np.abs(agent_actions.astype(int) - optimal_actions.astype(int))
        agree_mask = diff == 0
        axes[2].bar(steps[agree_mask], np.ones(agree_mask.sum()), color="#2ecc71", width=0.8, alpha=0.7, label="Match")
        axes[2].bar(steps[~agree_mask], diff[~agree_mask], color="#e74c3c", width=0.8, alpha=0.7, label="Mismatch (|diff|)")
        axes[2].axhline(0, color="black", linewidth=0.6)
        axes[2].set_title(
            f"Action Disagreement  (hit rate: {hit_rate:.1%})", fontweight="bold"
        )
        axes[2].set_xlabel("Step")
        axes[2].set_ylabel("|diff|")
        axes[2].legend(fontsize=8)
        axes[2].text(
            0.01, 0.97,
            f"hits={agree_mask.sum()}  misses={T - agree_mask.sum()}  score={diff.sum()}",
            transform=axes[2].transAxes, fontsize=9, va="top",
        )

        fig.suptitle(f"Agent vs LP-Optimal Action Schedule — Ep {ep}", fontsize=13, fontweight="bold")
        fig.tight_layout()
        return fig

    # Kept for backward compatibility / standalone use
    def plot_action_volumes(self, episode: Optional[int] = None) -> plt.Figure:
        """Per-step inject and withdraw volumes as a mirrored bar chart."""
        sub, ep = self._get_episode(episode)
        steps = sub["step"].values

        fig, ax = plt.subplots(figsize=(self.figsize_width, 4))
        inj = sub["h2_inject_units"].fillna(0).values
        wdr = sub["h2_withdraw_units"].fillna(0).values

        ax.bar(steps, inj, color="#2ecc71", width=0.8, label="Inject (units)")
        ax.bar(steps, -wdr, color="#e74c3c", width=0.8, label="Withdraw (units)")
        ax.axhline(0, color="black", linewidth=0.8)

        ax.set_title(f"Inject / Withdraw Volumes — Ep {ep}", fontweight="bold")
        ax.set_xlabel("Step")
        ax.set_ylabel("Volume (units)")
        ax.text(
            0.01, 0.97,
            f"Σ inject: {inj.sum():,.1f}   Σ withdraw: {wdr.sum():,.1f}",
            transform=ax.transAxes, fontsize=9, va="top",
        )
        ax.legend(fontsize=9)
        fig.tight_layout()
        return fig

    def plot_action_dollars(self, episode: Optional[int] = None) -> plt.Figure:
        """Per-step inject cost and withdraw revenue in dollar terms."""
        sub, ep = self._get_episode(episode)
        steps = sub["step"].values

        inj_d = sub["h2_inject_dollars"].fillna(0).values
        wdr_d = sub["h2_withdraw_dollars"].fillna(0).values
        net = wdr_d - inj_d

        fig, axes = plt.subplots(2, 1, figsize=(self.figsize_width, 6), sharex=True)

        axes[0].bar(steps, inj_d, color="#e67e22", width=0.8, alpha=0.8, label="Inject cost (€)")
        axes[0].bar(steps, wdr_d, color="#27ae60", width=0.8, alpha=0.8, label="Withdraw revenue (€)")
        axes[0].set_ylabel("€")
        axes[0].set_title(f"Inject Cost vs Withdraw Revenue — Ep {ep}", fontweight="bold")
        axes[0].legend(fontsize=9)

        colors = ["#27ae60" if v >= 0 else "#e74c3c" for v in net]
        axes[1].bar(steps, net, color=colors, width=0.8, label="Net cash-flow (€)")
        axes[1].axhline(0, color="black", linewidth=0.8)
        axes[1].set_ylabel("€")
        axes[1].set_xlabel("Step")
        axes[1].set_title("Net Cash-Flow per Step", fontweight="bold")
        axes[1].text(
            0.01, 0.97, f"Σ net: {net.sum():,.2f} €",
            transform=axes[1].transAxes, fontsize=9, va="top",
        )

        fig.suptitle(f"Dollar-Value Action Breakdown — Ep {ep}", fontweight="bold")
        fig.tight_layout()
        return fig

    def plot_reward_components(self, episode: Optional[int] = None) -> plt.Figure:
        """Reward components per step as bar charts with cumulative overlay."""
        sub, ep = self._get_episode(episode)
        steps = sub["step"].values

        available = [c for c in _REWARD_COMPONENTS if c in sub.columns]

        fig, axes = plt.subplots(
            len(available) + 1, 1,
            figsize=(self.figsize_width, 3 * (len(available) + 1)),
            sharex=True,
        )

        for i, comp in enumerate(available):
            vals = sub[comp].fillna(0).values
            color = _REWARD_COLORS.get(comp, "grey")
            axes[i].bar(steps, vals, color=color, width=0.8, alpha=0.8, label=comp)
            axes[i].axhline(0, color="black", linewidth=0.6)
            axes[i].set_ylabel("Reward")
            axes[i].set_title(comp.replace("_", " ").title(), fontweight="bold")
            axes[i].text(
                0.01, 0.97,
                f"mean={vals.mean():.4f}  sum={vals.sum():.4f}",
                transform=axes[i].transAxes, fontsize=8, va="top",
            )

        if "reward_total" in sub.columns:
            total = sub["reward_total"].fillna(0).values
            colors_t = ["#27ae60" if v >= 0 else "#e74c3c" for v in total]
            axes[-1].bar(steps, total, color=colors_t, width=0.8, alpha=0.9, label="reward_total")
            axes[-1].axhline(0, color="black", linewidth=0.8)
            axes[-1].set_title("Total Reward", fontweight="bold")
            axes[-1].set_ylabel("Reward")
            axes[-1].text(
                0.01, 0.97,
                f"mean={total.mean():.4f}  sum={total.sum():.4f}",
                transform=axes[-1].transAxes, fontsize=8, va="top",
            )

        axes[-1].set_xlabel("Step")
        fig.suptitle(f"Reward Components — Ep {ep}", fontsize=13, fontweight="bold")
        fig.tight_layout()
        return fig

    def plot_cumulative_reward(self, episode: Optional[int] = None) -> plt.Figure:
        """Cumulative total reward and per-component cumulative build-up."""
        sub, ep = self._get_episode(episode)
        steps = sub["step"].values

        fig, axes = plt.subplots(1, 2, figsize=(self.figsize_width, 4))

        if "reward_total" in sub.columns:
            cum = sub["reward_total"].fillna(0).cumsum().values
            axes[0].plot(steps, cum, color="purple", linewidth=2.0)
            axes[0].fill_between(steps, 0, cum, alpha=0.15, color="purple")
            axes[0].axhline(0, color="black", linewidth=0.8, linestyle="--")
            axes[0].set_title("Cumulative Total Reward", fontweight="bold")
            axes[0].set_xlabel("Step")
            axes[0].set_ylabel("Cumulative Reward")
            axes[0].text(
                0.01, 0.97, f"Final: {cum[-1]:.4f}",
                transform=axes[0].transAxes, fontsize=9, va="top",
            )

        available = [c for c in _REWARD_COMPONENTS if c in sub.columns]
        for comp in available:
            cum_c = sub[comp].fillna(0).cumsum().values
            axes[1].plot(
                steps, cum_c, linewidth=1.5,
                color=_REWARD_COLORS.get(comp, "grey"), label=comp,
            )
        axes[1].axhline(0, color="black", linewidth=0.6, linestyle="--")
        axes[1].set_title("Cumulative Reward by Component", fontweight="bold")
        axes[1].set_xlabel("Step")
        axes[1].set_ylabel("Cumulative Reward")
        axes[1].legend(fontsize=8)

        fig.suptitle(f"Cumulative Reward — Ep {ep}", fontsize=13, fontweight="bold")
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------ #

    def _check_loaded(self):
        if self.df is None:
            raise RuntimeError("Call .load(path) before plotting.")

    def _get_episode(self, episode: Optional[int]):
        self._check_loaded()
        ep = episode if episode is not None else self.df["episode"].max()
        sub = self.df[self.df["episode"] == ep]
        if sub.empty:
            raise ValueError(f"Episode {ep} not found.")
        return sub, ep


# ============================================================================ #
# CROSS-EPISODE VISUALIZER
# ============================================================================ #


class CrossEpisodeVisualizer:
    """
    Episode-level summary metrics across an entire training run.

    Panels
    ------
    1. Total realized_avoided_cost per episode
    2. Average dollar cost basis across episodes
    3. Total profit (sum withdrawals dollars - sum injections dollars) per episode
    4. Average Levelized Cost of Injection avg(curr_spot_h2 when action == 2)
    5. Average Levelized Cost of Withdrawal avg(curr_spot_h2 when action == 0)
    6. Mean reward component breakdown per episode
    7. Constraint violation penalty per episode
    8. Net cash-flow per episode

    Usage
    -----
        viz = CrossEpisodeVisualizer()
        viz.load("logs/run_abc/")
        viz.generate_diagnostic_suite(output_dir="results/run_abc/")
        summary = viz.episode_summary()
    """

    def __init__(self, figsize_width: int = 14, dpi: int = 150):
        self.figsize_width = figsize_width
        self.dpi = dpi
        self.df: Optional[pd.DataFrame] = None
        self._summary: Optional[pd.DataFrame] = None

    def load(self, path: str) -> "CrossEpisodeVisualizer":
        self.df = _load_path(path)
        self._summary = None
        return self

    def load_df(self, df: pd.DataFrame) -> "CrossEpisodeVisualizer":
        self.df = df.copy()
        self._summary = None
        return self

    # ------------------------------------------------------------------ #

    def episode_summary(self) -> pd.DataFrame:
        """
        Return (and cache) a per-episode summary DataFrame.

        Uses vectorized groupby operations (mirroring EnvState.action_history /
        spot_h2_price_history arrays) instead of row-by-row Python loops.
        """
        self._check_loaded()
        if self._summary is not None:
            return self._summary

        df = self.df
        grp = df.groupby("episode")

        # --- inventory ---
        inv_agg = grp["h2_inventory"].agg(
            final_h2_inventory="last",
            mean_h2_inventory="mean",
            max_h2_inventory="max",
        )

        # --- dollar flows (vectorized sum) ---
        flow_agg = grp.agg(
            n_steps=("step", "count"),
            total_inject_dollars=("h2_inject_dollars", "sum"),
            total_withdraw_dollars=("h2_withdraw_dollars", "sum"),
            total_injected=("h2_inject_units", "sum"),
            total_withdrawn=("h2_withdraw_units", "sum"),
            mean_dollar_cost_basis=("dollar_cost_basis", "mean"),
        )
        final_spot = grp["mkt_h2_spot"].last()
        final_inventory_valuation = inv_agg["final_h2_inventory"] * final_spot
        flow_agg["net_cashflow"] = (
            flow_agg["total_withdraw_dollars"]
            - flow_agg["total_inject_dollars"]
            + final_inventory_valuation
        )
        flow_agg["total_withdrawn"] = (
            flow_agg["total_withdrawn"] + inv_agg["final_h2_inventory"]
        )

        # --- levelized costs (filter by action, then groupby) ---
        # mirrors EnvState.action_history: action==2 → inject, action==0 → withdraw
        inject_mask = df["discrete_action"] == 2
        withdraw_mask = df["discrete_action"] == 0

        lcoi = (
            df[inject_mask].groupby("episode")["real_h2_spot"].mean()
            .rename("levelized_cost_of_injection")
        )
        lcow = (
            df[withdraw_mask].groupby("episode")["mkt_h2_spot"].mean()
            .rename("levelized_cost_of_withdrawal")
        )

        # --- reward components (mirrors realized_avoided_cost / curr_potential_avoided_cost arrays) ---
        reward_cols = [c for c in _REWARD_COMPONENTS if c in df.columns]
        if reward_cols:
            reward_agg_mean = grp[reward_cols].mean().add_prefix("mean_")
            reward_agg_sum = grp[reward_cols].sum().add_prefix("sum_")
        else:
            reward_agg_mean = pd.DataFrame(index=grp.groups.keys())
            reward_agg_sum = pd.DataFrame(index=grp.groups.keys())

        if "reward_total" in df.columns:
            total_r = grp["reward_total"].agg(
                total_reward="sum",
                mean_reward="mean",
            )
            # drawdown on cumulative reward — requires per-episode computation
            drawdown = grp["reward_total"].apply(
                lambda s: _max_drawdown(s.fillna(0).cumsum().values)
            ).rename("reward_drawdown")
            r_excl_agg = pd.DataFrame(index=grp.groups.keys())
        else:
            total_r = pd.DataFrame(index=grp.groups.keys())
            drawdown = pd.Series(dtype=float)
            r_excl_agg = pd.DataFrame(index=grp.groups.keys())

        summary = (
            flow_agg
            .join(inv_agg)
            .join(lcoi)
            .join(lcow)
            .join(reward_agg_mean)
            .join(reward_agg_sum)
            .join(total_r)
            .join(drawdown)
            .join(r_excl_agg)
        )

        self._summary = summary
        return self._summary

    def print_episode_summary(self) -> pd.DataFrame:
        summary = self.episode_summary()
        print(f"\n{'='*75}")
        print("  CROSS-EPISODE SUMMARY")
        print(f"{'='*75}")
        print(summary.to_string(float_format="{:.4f}".format))
        print(f"{'='*75}\n")
        return summary

    # ------------------------------------------------------------------ #

    def generate_diagnostic_suite(
        self,
        output_dir: Optional[str] = None,
        show: bool = True,
    ) -> None:
        """
        Produce all cross-episode panels.
        Files written:
            01_realized_avoided_cost.png
            02_dollar_cost_basis.png
            03_profit.png
            04_levelized_injection_cost.png
            05_levelized_withdrawal_cost.png
            06_reward_components.png
            07_net_cashflow.png
        """
        self._check_loaded()
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        def _path(name):
            return os.path.join(output_dir, name) if output_dir else None

        print(f"\n{'='*65}")
        print(f"  Cross-episode diagnostics  →  {len(self.episode_summary())} episodes")
        print(f"{'='*65}")
        self.print_episode_summary()

        _save_or_show(self.plot_realized_avoided_cost(), _path("01_realized_avoided_cost.png"), self.dpi, show)
        _save_or_show(self.plot_dollar_cost_basis(), _path("02_dollar_cost_basis.png"), self.dpi, show)
        _save_or_show(self.plot_profit(), _path("03_profit.png"), self.dpi, show)
        _save_or_show(self.plot_levelized_injection_cost(), _path("04_levelized_injection_cost.png"), self.dpi, show)
        _save_or_show(self.plot_levelized_withdrawal_cost(), _path("05_levelized_withdrawal_cost.png"), self.dpi, show)
        _save_or_show(self.plot_reward_components(), _path("06_reward_components.png"), self.dpi, show)
        _save_or_show(self.plot_net_cashflow(), _path("08_net_cashflow.png"), self.dpi, show)

    # ------------------------------------------------------------------ #
    # Individual panels
    # ------------------------------------------------------------------ #

    def plot_realized_avoided_cost(self) -> plt.Figure:
        """Total realized avoided cost (withdrawal PnL) per episode with rolling mean."""
        s = self.episode_summary()
        col = "sum_realized_avoided_cost"
        if col not in s.columns:
            raise ValueError("No realized_avoided_cost data available.")

        episodes = s.index.values
        vals = s[col].values

        fig, ax = plt.subplots(figsize=(self.figsize_width, 4))
        ax.bar(
            episodes, vals,
            color=["#27ae60" if v >= 0 else "#e74c3c" for v in vals],
            alpha=0.7, width=0.8, label="Total realized avoided cost",
        )
        window = min(10, len(vals))
        if len(vals) >= window:
            roll = pd.Series(vals).rolling(window, min_periods=1).mean().values
            ax.plot(episodes, roll, color="navy", linewidth=2.0, label=f"Rolling mean (w={window})")

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total Realized Avoided Cost (kEUR)")
        ax.set_title("Total Realized Avoided Cost per Episode", fontweight="bold")
        ax.legend(fontsize=9)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        fig.tight_layout()
        return fig

    def plot_dollar_cost_basis(self) -> plt.Figure:
        """Average dollar cost basis per episode — tracks injection price discipline."""
        s = self.episode_summary()
        if "mean_dollar_cost_basis" not in s.columns:
            raise ValueError("No dollar_cost_basis data available.")

        episodes = s.index.values
        basis = s["mean_dollar_cost_basis"].values

        fig, ax = plt.subplots(figsize=(self.figsize_width, 4))
        ax.plot(
            episodes, basis,
            color="darkorange", linewidth=1.8, marker="o", markersize=4,
            label="Mean dollar cost basis",
        )
        ax.fill_between(episodes, basis.min() * 0.98, basis, alpha=0.1, color="darkorange")

        window = min(10, len(basis))
        if len(basis) >= window:
            roll = pd.Series(basis).rolling(window, min_periods=1).mean().values
            ax.plot(episodes, roll, color="saddlebrown", linewidth=2.0,
                    linestyle="--", label=f"Rolling mean (w={window})")

        ax.set_xlabel("Episode")
        ax.set_ylabel("Mean Cost Basis (€/kg)")
        ax.set_title("Average Dollar Cost Basis per Episode", fontweight="bold")
        ax.legend(fontsize=9)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        fig.tight_layout()
        return fig

    def plot_profit(self) -> plt.Figure:
        """Total profit (withdraw revenue − inject cost) per episode."""
        s = self.episode_summary()
        if "net_cashflow" not in s.columns:
            raise ValueError("No cashflow data available.")

        episodes = s.index.values
        profit = s["net_cashflow"].values

        fig, ax = plt.subplots(figsize=(self.figsize_width, 4))
        ax.bar(
            episodes, profit,
            color=["#27ae60" if v >= 0 else "#e74c3c" for v in profit],
            alpha=0.8, width=0.8, label="Net profit (€)",
        )
        window = min(10, len(profit))
        if len(profit) >= window:
            roll = pd.Series(profit).rolling(window, min_periods=1).mean().values
            ax.plot(episodes, roll, color="purple", linewidth=2.0, label=f"Rolling mean (w={window})")

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Net Profit (€)")
        ax.set_title(
            "Total Profit per Episode  (Σ withdraw revenue − Σ inject cost)", fontweight="bold"
        )
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax.legend(fontsize=9)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        fig.tight_layout()
        return fig

    def plot_levelized_injection_cost(self) -> plt.Figure:
        """
        Average Levelized Cost of Injection (LCOI) per episode.

        LCOI = mean H2 spot price across all steps where the agent chose to inject.
        Lower LCOI is better — the agent should buy cheap.
        """
        s = self.episode_summary()
        if "levelized_cost_of_injection" not in s.columns:
            raise ValueError("No injection price data available (no inject actions logged).")

        episodes = s.index.values
        lcoi = s["levelized_cost_of_injection"].values

        fig, ax = plt.subplots(figsize=(self.figsize_width, 4))
        ax.plot(
            episodes, lcoi,
            color="#e67e22", linewidth=1.8, marker="o", markersize=4,
            label="LCOI (avg spot when injecting)",
        )
        window = min(10, int(np.sum(~np.isnan(lcoi))))
        if window >= 2:
            roll = pd.Series(lcoi).rolling(window, min_periods=1).mean().values
            ax.plot(episodes, roll, color="saddlebrown", linewidth=2.0,
                    linestyle="--", label=f"Rolling mean (w={window})")

        ax.set_xlabel("Episode")
        ax.set_ylabel("LCOI (€/kg)")
        ax.set_title(
            "Levelized Cost of Injection per Episode  (lower = better)", fontweight="bold"
        )
        ax.legend(fontsize=9)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        fig.tight_layout()
        return fig

    def plot_levelized_withdrawal_cost(self) -> plt.Figure:
        """
        Average Levelized Cost of Withdrawal (LCOW) per episode.

        LCOW = mean H2 spot price across all steps where the agent chose to withdraw.
        Higher LCOW is better — the agent should sell dear.
        """
        s = self.episode_summary()
        if "levelized_cost_of_withdrawal" not in s.columns:
            raise ValueError("No withdrawal price data available (no withdraw actions logged).")

        episodes = s.index.values
        lcow = s["levelized_cost_of_withdrawal"].values

        fig, ax = plt.subplots(figsize=(self.figsize_width, 4))
        ax.plot(
            episodes, lcow,
            color="#27ae60", linewidth=1.8, marker="o", markersize=4,
            label="LCOW (avg spot when withdrawing)",
        )
        window = min(10, int(np.sum(~np.isnan(lcow))))
        if window >= 2:
            roll = pd.Series(lcow).rolling(window, min_periods=1).mean().values
            ax.plot(episodes, roll, color="darkgreen", linewidth=2.0,
                    linestyle="--", label=f"Rolling mean (w={window})")

        ax.set_xlabel("Episode")
        ax.set_ylabel("LCOW (€/kg)")
        ax.set_title(
            "Levelized Cost of Withdrawal per Episode  (higher = better)", fontweight="bold"
        )
        ax.legend(fontsize=9)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        fig.tight_layout()
        return fig

    def plot_reward_components(self) -> plt.Figure:
        """Mean reward component breakdown across episodes (stacked bars + line sums)."""
        s = self.episode_summary()
        episodes = s.index.values

        available = [c for c in _REWARD_COMPONENTS if f"mean_{c}" in s.columns]
        if not available:
            raise ValueError("No reward component data available.")

        fig, axes = plt.subplots(1, 2, figsize=(self.figsize_width, 5))

        bottoms = np.zeros(len(episodes))
        for comp in available:
            vals = s[f"mean_{comp}"].values
            color = _REWARD_COLORS.get(comp, "grey")
            axes[0].bar(
                episodes, vals, bottom=bottoms,
                color=color, alpha=0.8, width=0.8, label=comp,
            )
            bottoms += vals
        axes[0].axhline(0, color="black", linewidth=0.8)
        axes[0].set_title("Mean Reward Component per Episode", fontweight="bold")
        axes[0].set_xlabel("Episode")
        axes[0].set_ylabel("Mean Reward")
        axes[0].legend(fontsize=8)
        axes[0].xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

        for comp in available:
            vals = s[f"sum_{comp}"].values
            color = _REWARD_COLORS.get(comp, "grey")
            axes[1].plot(
                episodes, vals,
                color=color, linewidth=1.5, marker="o", markersize=3, label=comp,
            )
        axes[1].axhline(0, color="black", linewidth=0.6, linestyle="--")
        axes[1].set_title("Total Reward Component per Episode", fontweight="bold")
        axes[1].set_xlabel("Episode")
        axes[1].set_ylabel("Sum Reward")
        axes[1].legend(fontsize=8)
        axes[1].xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

        fig.suptitle("Reward Component Breakdown", fontsize=13, fontweight="bold")
        fig.tight_layout()
        return fig

    def plot_net_cashflow(self) -> plt.Figure:
        """
        Two-panel cash-flow analysis per episode.

        Left  — Net cash-flow bars with total withdrawn volume overlaid on a twin axis.
        Right — Normalised cash-flow (net_cashflow / total_withdraw_dollars) showing
                return per € of withdrawal revenue.
        """
        s = self.episode_summary()
        if "net_cashflow" not in s.columns:
            raise ValueError("No cash-flow data available.")

        episodes = s.index.values
        net = s["net_cashflow"].values
        total_withdrawn = s["total_withdrawn"].values if "total_withdrawn" in s.columns else None

        fig, axes = plt.subplots(1, 2, figsize=(self.figsize_width, 4))

        # --- Left: net cashflow bars + total_withdrawn volume overlay ---
        colors = ["#27ae60" if v >= 0 else "#e74c3c" for v in net]
        axes[0].bar(episodes, net, color=colors, alpha=0.7, width=0.8, label="Net cash-flow (€)")
        axes[0].axhline(0, color="black", linewidth=0.8, linestyle="--")
        axes[0].set_title("Net Cash-Flow per Episode (€)", fontweight="bold")
        axes[0].set_xlabel("Episode")
        axes[0].set_ylabel("€")
        axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        axes[0].xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

        if total_withdrawn is not None:
            ax0_twin = axes[0].twinx()
            ax0_twin.plot(
                episodes, total_withdrawn,
                color="steelblue", linewidth=1.8, marker="o", markersize=4,
                label="Total withdrawn (units)", zorder=3,
            )
            ax0_twin.set_ylabel("Withdrawn Volume (units)", color="steelblue")
            ax0_twin.tick_params(axis="y", labelcolor="steelblue")
            # combined legend
            handles0, labels0 = axes[0].get_legend_handles_labels()
            handles1, labels1 = ax0_twin.get_legend_handles_labels()
            axes[0].legend(handles0 + handles1, labels0 + labels1, fontsize=8, loc="upper left")
        else:
            axes[0].legend(fontsize=9)

        # --- Right: normalised cashflow = net_cashflow / total_withdraw_dollars ---
        if "total_withdraw_dollars" in s.columns:
            denom = s["total_withdraw_dollars"].values.copy().astype(float)
            denom[np.abs(denom) < 1e-8] = np.nan  # avoid division by zero
            norm_cf = net / denom

            norm_colors = ["#27ae60" if v >= 0 else "#e74c3c" for v in np.nan_to_num(norm_cf)]
            axes[1].bar(episodes, norm_cf, color=norm_colors, alpha=0.7, width=0.8,
                        label="net_cashflow / withdraw_revenue")
            axes[1].axhline(0, color="black", linewidth=0.8, linestyle="--")
            axes[1].axhline(1, color="navy", linewidth=1.0, linestyle=":",
                            label="Break-even (ratio = 1)")

            window = min(10, int(np.sum(~np.isnan(norm_cf))))
            if window >= 2:
                roll = pd.Series(norm_cf).rolling(window, min_periods=1).mean().values
                axes[1].plot(episodes, roll, color="purple", linewidth=2.0,
                             label=f"Rolling mean (w={window})")

            axes[1].set_title(
                "Normalised Cash-Flow  (net / withdraw revenue)", fontweight="bold"
            )
            axes[1].set_xlabel("Episode")
            axes[1].set_ylabel("Ratio")
            axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.2f}"))
            axes[1].legend(fontsize=8)
            axes[1].xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        else:
            axes[1].set_visible(False)

        fig.suptitle("Cash-Flow Analysis", fontsize=13, fontweight="bold")
        fig.tight_layout()
        return fig

    # Kept for backward compatibility
    def plot_total_reward(self) -> plt.Figure:
        """Episode total reward with a smoothed rolling average."""
        s = self.episode_summary()
        if "total_reward" not in s.columns:
            raise ValueError("No reward data available.")

        episodes = s.index.values
        rewards = s["total_reward"].values

        fig, ax = plt.subplots(figsize=(self.figsize_width, 4))
        ax.bar(
            episodes, rewards,
            color=["#27ae60" if r >= 0 else "#e74c3c" for r in rewards],
            alpha=0.6, width=0.8, label="Episode reward",
        )
        window = min(10, len(rewards))
        if len(rewards) >= window:
            roll = pd.Series(rewards).rolling(window, min_periods=1).mean().values
            ax.plot(episodes, roll, color="purple", linewidth=2.0,
                    label=f"Rolling mean (w={window})")

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total Reward")
        ax.set_title("Total Reward per Episode", fontweight="bold")
        ax.legend(fontsize=9)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        fig.tight_layout()
        return fig

    def plot_final_inventory(self) -> plt.Figure:
        """Final H2 inventory level at end of each episode."""
        s = self.episode_summary()
        if "final_h2_inventory" not in s.columns:
            raise ValueError("No inventory data available.")

        from constants import STORAGE_CAPACITY

        episodes = s.index.values
        inv = s["final_h2_inventory"].values

        fig, ax = plt.subplots(figsize=(self.figsize_width, 4))
        ax.plot(episodes, inv, marker="o", color="darkorange", linewidth=1.5,
                markersize=4, label="Final inventory")
        ax.axhline(STORAGE_CAPACITY, color="red", linestyle="--", linewidth=1.0,
                   label=f"Capacity ({STORAGE_CAPACITY:,})")
        ax.axhline(0, color="navy", linestyle="--", linewidth=1.0, label="Floor (0)")
        ax.fill_between(episodes, 0, inv, alpha=0.1, color="darkorange")
        ax.set_xlabel("Episode")
        ax.set_ylabel("H2 Inventory (units)")
        ax.set_title("Final H2 Inventory per Episode", fontweight="bold")
        ax.set_ylim(-STORAGE_CAPACITY * 0.05, STORAGE_CAPACITY * 1.1)
        ax.legend(fontsize=9)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------ #

    def _check_loaded(self):
        if self.df is None:
            raise RuntimeError("Call .load(path) before plotting.")


def evaluate_complete_horizon(
    h2_np_matrix: np.ndarray,
    session_id: str,
    log_dir: str | None = None,
    output_dir: str | None = None,
):
    """Run cross-episode evaluation over the full training horizon.

    Parameters
    ----------
    h2_np_matrix : np.ndarray
        H2 forward-curve matrix used during training.
    session_id : str
        Training session identifier (used to locate logs when *log_dir*
        is not provided).
    log_dir : str, optional
        Path to the directory containing ``.jsonl`` debug logs.  Defaults
        to ``./logs/debug/ppo_{session_id}`` for local runs.
    output_dir : str, optional
        Directory for output plots.  Defaults to
        ``./results/ppo_{session_id}``.
    """
    if log_dir is None:
        log_dir = f"./logs/debug/ppo_{session_id}"
    if output_dir is None:
        output_dir = f"./results/ppo_{session_id}"

    report = evaluate_agent_against_lp(h2_spot_prices=h2_np_matrix[:, 0], log_dir=log_dir)
    print_accuracy_report(report)

    viz = CrossEpisodeVisualizer()
    viz.load(log_dir)
    viz.generate_diagnostic_suite(output_dir=output_dir)


def evaluate_specific_episode(
    session_id: str,
    episode_num: int,
    log_dir: str | None = None,
    output_dir: str | None = None,
):
    """Run intra-episode diagnostics for a single episode.

    Parameters
    ----------
    session_id : str
        Training session identifier.
    episode_num : int
        Episode index to visualise.
    log_dir : str, optional
        Path to the directory containing ``.jsonl`` debug logs.
    output_dir : str, optional
        Base directory for output plots (episode subfolder is appended).
    """
    if log_dir is None:
        log_dir = f"./logs/debug/ppo_{session_id}"
    if output_dir is None:
        output_dir = f"./results/ppo_{session_id}"

    viz = IntraEpisodeVisualizer()
    viz.load(log_dir)
    viz.generate_diagnostic_suite(
        episode=episode_num,
        output_dir=os.path.join(output_dir, f"episode_{episode_num}"),
    )
