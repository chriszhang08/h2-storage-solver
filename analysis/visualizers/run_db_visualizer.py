"""
RunDBVisualizer: RL Agent Performance Analysis Suite
================================================================================
Extracts and visualizes model performance trends from the simulation_runs table
in results/simulation.duckdb.

Each storage technology gets its own 9-panel diagnostic suite so that the RL
model's performance can be assessed independently, always benchmarked against:
  • the baseline policy (same storage tech, same scenario set)
  • the LP optimal solution (per-scenario upper bound stored in each row)

Core metric: capture_rate = total_cashflow / optimal_npv

USAGE:
------
    viz = RunDBVisualizer("results/simulation.duckdb")

    # Single-technology suite
    viz.plot_full_suite("lh2", output_path="results/diagnostic_lh2.png")

    # All technologies at once
    viz.plot_all_suites(output_dir="results/")
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import seaborn as sns
import duckdb

from typing import Optional, Tuple
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TECH_COLORS = {
    "lh2":      "#52AA5E",   # light green
    "lnh3":     "#2BD9FE",   # sky blue
    "lohc":     "#955E42",   # brown
    "meoh":     "#FF74D4",   # pink
    "baseline": "#5D737E",   # gray
    "lp":       "#0A122A",   # dark blue
}
DEFAULT_COLOR = "#34495e"

TEXT_COLOR  = "#2c3e50"
PARITY_COLOR = "#e74c3c"   # bright red parity line used throughout
GRID_ALPHA  = 0.3

VARIANCE_ORDER  = ["low", "medium", "high"]

# ---------------------------------------------------------------------------
# Master SQL
# ---------------------------------------------------------------------------
# Loads ALL model types (RL techs + baseline) in one pass.
# Python-side _tech_split() then partitions into (tech_df, baseline_df)
# so every visualisation compares a single RL tech against its baseline.
# The LP optimal columns (optimal_npv, optimal_withdrawal_cashflow, etc.)
# are per-scenario upper bounds that travel with every row regardless of model.

_MASTER_QUERY = """
SELECT
    run_id, seed, model_name, ppo_model, storage_type,
    carbon_price, h2_spot_mean, h2_spot_variance,
    COALESCE(dollar_var, SQRT(h2_spot_variance)) AS dollar_var,
    seasonality_alpha,
    optimal_npv,
    optimal_withdrawal_units,
    optimal_withdrawal_cashflow,
    episode_reward,
    total_cashflow,
    withdraw_cashflow,
    total_withdrawal_units,
    levelized_cost_of_injection,
    levelized_cost_of_withdrawal,
    storage_delta,
    final_inventory,
    final_spot,
    total_cashflow / optimal_npv                        AS capture_rate,
    total_withdrawal_units / optimal_withdrawal_units   AS withdrawal_efficiency
FROM simulation_clipped
WHERE optimal_withdrawal_units > 0
  AND optimal_npv               > 0
  AND h2_spot_mean              >= 0
"""

_ZEROS_QUERY = """
SELECT
    run_id, seed, model_name, ppo_model, storage_type,
    carbon_price, h2_spot_mean, h2_spot_variance,
    COALESCE(dollar_var, SQRT(h2_spot_variance)) AS dollar_var,
    seasonality_alpha,
    optimal_npv,
    optimal_withdrawal_units,
    optimal_withdrawal_cashflow,
    episode_reward,
    total_cashflow,
    withdraw_cashflow,
    total_withdrawal_units,
    levelized_cost_of_injection,
    levelized_cost_of_withdrawal,
    storage_delta,
    final_inventory,
    final_spot
FROM simulation_clipped
WHERE optimal_withdrawal_units = 0
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tech_color(tech: str) -> str:
    return TECH_COLORS.get(str(tech).lower(), DEFAULT_COLOR)


def _blank_ax(ax: plt.Axes) -> None:
    """Turn an axes into a silent inset container (no ticks/spines)."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("none")
    for spine in ax.spines.values():
        spine.set_visible(False)


def _dollar_fmt() -> mticker.FuncFormatter:
    return mticker.FuncFormatter(
        lambda x, _: f"${x/1e3:.0f}M" if abs(x) >= 1e3 else f"${x:.0f}"
    )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class RunDBVisualizer:
    """
    Per-technology 9-panel diagnostic suite for the RL H₂ storage trading agent.

    Call plot_full_suite(tech) once per technology to get a dedicated figure
    that benchmarks that technology's RL model against the baseline policy and
    the LP optimal solution.
    """

    def __init__(self, db_path: str = "results/simulation.duckdb"):
        self.db_path = db_path
        self._runs: Optional[pd.DataFrame] = None
        self._zeros_cache: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------ #
    # Data loading & splitting                                              #
    # ------------------------------------------------------------------ #

    def _load_runs(self) -> pd.DataFrame:
        if self._runs is not None:
            return self._runs

        print(f"Connecting to {self.db_path} ...")
        con = duckdb.connect(self.db_path, read_only=True)
        try:
            df = con.execute(_MASTER_QUERY).df()
        finally:
            con.close()

        df["capture_rate"] = df["capture_rate"].clip(lower=-0.5, upper=1.0)
        df["withdrawal_efficiency"] = df["withdrawal_efficiency"].clip(lower=-0.5, upper=1.0)

        # LP levelized costs: per-scenario upper-bound reference values
        df["lp_lcow"] = df["optimal_withdrawal_cashflow"] / df["optimal_withdrawal_units"]
        df["lp_lcoi"] = df["lp_lcow"] - df["optimal_npv"] / df["optimal_withdrawal_units"]
        df["lp_lc_delta"] = df["lp_lcow"] - df["lp_lcoi"]

        self._runs = df
        all_models = sorted(df["model_name"].dropna().unique())
        print(f"Loaded {len(df):,} runs  |  models: {all_models}")
        return df

    def _load_zeros(self) -> pd.DataFrame:
        if self._zeros_cache is not None:
            return self._zeros_cache
        con = duckdb.connect(self.db_path, read_only=True)
        try:
            df = con.execute(_ZEROS_QUERY).df()
        finally:
            con.close()
        self._zeros_cache = df
        return df

    def _tech_split(self, tech: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Partition the master dataset into (tech_df, baseline_df).

        tech_df    — rows where model_name == tech (the RL agent under analysis)
        baseline_df — rows where model_name == 'baseline'
        """
        df = self._load_runs()
        tech_df = df[df["model_name"] == tech].copy()
        baseline_df = df[df["model_name"] == "baseline"].copy()

        return tech_df, baseline_df

    # ------------------------------------------------------------------ #
    # Plot 2 — Hexbin heatmap: RL cashflow vs LP, coloured by volatility   #
    # ------------------------------------------------------------------ #

    def plot_2d_heatmap(
        self, ax: plt.Axes, tech_df: pd.DataFrame, baseline_df: pd.DataFrame,
        n_bins: int = 12, eval_dollar_var: float = 1.5,
    ) -> None:
        """
        Hexbin of RL total_cashflow (y) vs LP optimal_npv (x) for this technology,
        colour-mapped by mean dollar_var per hex bin.
        The parity y = x line is drawn in bright red.

        The hex nearest the median total_cashflow for scenarios where
        dollar_var is within ±0.5 of eval_dollar_var is circled and labelled.
        """
        tech = tech_df["model_name"].iat[0]

        sub = tech_df[["optimal_npv", "total_cashflow", "dollar_var"]].dropna()
        sub = sub[sub["dollar_var"] < 5]
        if len(sub) < 5:
            ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=9)
            return

        hb = ax.hexbin(
            sub["optimal_npv"], sub["total_cashflow"],
            C=sub["dollar_var"], reduce_C_function=np.mean,
            gridsize=n_bins, cmap="plasma", linewidths=0.2,
        )

        # Parity line — bright red
        lo = min(sub["optimal_npv"].min(), sub["total_cashflow"].min())
        hi = max(sub["optimal_npv"].max(), sub["total_cashflow"].max())
        ax.plot([lo, hi], [lo, hi], color=PARITY_COLOR, linewidth=1.6,
                linestyle="--", alpha=0.9, zorder=5, label="y = x (parity)")

        cbar = plt.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Mean Standard Deviation ($/kg)", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

        # Highlight median cashflow at eval_dollar_var (±0.5 window)
        tol = 0.5
        eval_sub = sub[
            (sub["dollar_var"] >= eval_dollar_var - tol) &
            (sub["dollar_var"] <= eval_dollar_var + tol)
        ]
        if len(eval_sub) >= 1:
            median_cf = eval_sub["total_cashflow"].median()
            nearest = eval_sub.iloc[(eval_sub["total_cashflow"] - median_cf).abs().argsort()[:1]]
            mx = nearest["optimal_npv"].values[0]
            my = nearest["total_cashflow"].values[0]
            ax.scatter([mx], [my], s=220, facecolors="none", edgecolors="white",
                       linewidths=2.0, zorder=7)
            ax.annotate(
                f"median (σ≈{eval_dollar_var}$/kg): {_dollar_fmt()(median_cf, None)}",
                xy=(mx, my), xytext=(8, 8), textcoords="offset points",
                fontsize=8, color="white", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.55, lw=0),
            )

        ax.xaxis.set_major_locator(mticker.MultipleLocator(100000))
        ax.yaxis.set_major_locator(mticker.MultipleLocator(100000))
        ax.xaxis.set_major_formatter(_dollar_fmt())
        ax.yaxis.set_major_formatter(_dollar_fmt())
        ax.set_xlabel("LP Total Cashflow ($)", fontsize=9, color=TEXT_COLOR)
        ax.set_ylabel("RL Total Cashflow ($)", fontsize=9, color=TEXT_COLOR)
        ax.set_title(f"{tech} — RL Cashflow vs LP Optimal (Coloured by Volatility)",
                     fontsize=10, color=TEXT_COLOR, pad=5)
        ax.legend(fontsize=8, framealpha=0.85)
        ax.tick_params(labelsize=8)

    # ------------------------------------------------------------------ #
    # Plot 6 — Levelized costs: LCOI and LCOW violins                      #
    # ------------------------------------------------------------------ #

    def plot_levelized_costs(
        self, ax: plt.Axes, tech_df: pd.DataFrame, baseline_df: pd.DataFrame
    ) -> None:
        """
        Box-and-whisker of storage_delta (LCOW − LCOI) for RL, baseline, and LP optimal.
        LP reference: lp_lc_delta = lp_lcow − lp_lcoi (per-scenario theoretical max spread).
        """
        tech = tech_df["model_name"].iat[0]

        frames = []
        for df_, label in [(tech_df, f"{tech} (RL)"), (baseline_df, "baseline")]:
            if len(df_) < 5:
                continue
            frames.append(
                df_[["storage_delta"]].rename(columns={"storage_delta": "delta"})
                .assign(model=label)
            )

        if len(tech_df) >= 5:
            frames.append(
                tech_df[["lp_lc_delta"]].rename(columns={"lp_lc_delta": "delta"})
                .assign(model=f"{tech} (LP)")
            )

        if not frames:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
            return

        melted = pd.concat(frames, ignore_index=True).dropna(subset=["delta"])
        order = [k for k in [f"{tech} (RL)", "baseline", f"{tech} (LP)"]
                 if k in melted["model"].values]
        palette = {
            f"{tech} (RL)": _tech_color(tech),
            "baseline": TECH_COLORS["baseline"],
            f"{tech} (LP)": TECH_COLORS["lp"],
        }

        sns.boxplot(
            data=melted, x="model", y="delta", ax=ax,
            palette=palette, order=order, linewidth=0.8,
            showfliers=False,
        )
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_ylim(top=3.5, bottom=-1.5)
        ax.set_xlabel("Model", fontsize=9, color=TEXT_COLOR)
        ax.set_ylabel("Storage Delta — LCOW − LCOI ($/kg)", fontsize=9, color=TEXT_COLOR)
        ax.set_title(f"{tech} — Storage Spread Captured vs LP Theoretical Max",
                     fontsize=10, color=TEXT_COLOR, pad=5)
        ax.tick_params(labelsize=8)

    # ------------------------------------------------------------------ #
    # Plot 7 — Quantity withdrawn by price volatility                      #
    # ------------------------------------------------------------------ #

    def plot_withdrawal_efficiency(
        self, ax: plt.Axes, tech_df: pd.DataFrame, baseline_df: pd.DataFrame
    ) -> None:
        """
        Line chart of mean total_withdrawal_units vs dollar_var (15 equal-width
        bins spanning the combined range of RL and baseline scenarios).
        Three lines: RL tech, baseline, and LP optimal (dotted black).
        """
        tech = tech_df["model_name"].iat[0]
        color = _tech_color(tech)

        all_dv = pd.concat([
            tech_df["dollar_var"].dropna(),
            baseline_df["dollar_var"].dropna(),
        ])
        if len(all_dv) < 2:
            ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes,
                    ha="center", va="center")
            return

        # Custom domain-informed bins; arithmetic midpoints avoid the sqrt(0) issue
        bin_edges = [0, 1, 2, 3, 4, 5, 6, 8, 10, 15, 20]
        bin_centers = [(a + b) / 2 for a, b in zip(bin_edges[:-1], bin_edges[1:])]
        # [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 7.0, 9.0, 12.5, 17.5]

        def _binned(df_: pd.DataFrame, col: str):
            d = df_.dropna(subset=["dollar_var", col]).copy()
            d = d[d["dollar_var"] >= 0]
            if d.empty:
                return [], []
            d["_bin"] = pd.cut(d["dollar_var"], bins=bin_edges, labels=False, include_lowest=True)
            agg = d.groupby("_bin")[col].mean()
            valid = [(bin_centers[int(b)], y) for b, y in zip(agg.index, agg.values)
                     if pd.notna(b) and pd.notna(y)]
            if not valid:
                return [], []
            xs, ys = zip(*valid)
            return list(xs), list(ys)

        xs, ys = _binned(tech_df, "total_withdrawal_units")
        if xs:
            ax.plot(xs, ys, color=color, linewidth=1.8,
                    label=f"{tech} (RL)", marker="o", markersize=3)

        xs_b, ys_b = _binned(baseline_df, "total_withdrawal_units")
        if xs_b:
            ax.plot(xs_b, ys_b, color=TECH_COLORS["baseline"], linewidth=1.6,
                    linestyle="--", label="baseline", marker="s", markersize=3)

        xs_lp, ys_lp = _binned(tech_df, "optimal_withdrawal_units")
        if xs_lp:
            ax.plot(xs_lp, ys_lp, color="black", linewidth=1.4,
                    linestyle=":", label="LP Optimal", marker="^", markersize=3)

        ax.set_xlabel("H2 Price Volatility ($/kg)", fontsize=9, color=TEXT_COLOR)
        ax.set_ylabel("Mean Withdrawal Units", fontsize=9, color=TEXT_COLOR)
        ax.set_title(f"{tech} — Quantity Withdrawn by Price Volatility",
                     fontsize=10, color=TEXT_COLOR, pad=5)
        ax.legend(fontsize=8, framealpha=0.85)
        ax.tick_params(labelsize=8)


    def plot_withdrawal_efficiency_all(self, ax: plt.Axes) -> None:
        """
        Multi-technology withdrawal efficiency chart on a single axes.

        For each RL technology:
          - Solid line in tech color       → RL agent (total_withdrawal_units)
          - Dotted line in same tech color → LP optimal (optimal_withdrawal_units)

        A single aggregated baseline line (dashed gray) is drawn across all
        storage types combined, since the baseline policy is technology-agnostic.

        LP optimal lines are derived from each technology's own tech_df rows so
        that the per-technology LP upper bound is correctly matched.
        """
        df = self._load_runs()

        bin_edges = [0, 1, 2, 3, 4, 5, 6, 8, 10, 15, 20]
        bin_centers = [(a + b) / 2 for a, b in zip(bin_edges[:-1], bin_edges[1:])]

        def _binned(df_: pd.DataFrame, col: str):
            d = df_.dropna(subset=["dollar_var", col]).copy()
            d = d[d["dollar_var"] >= 0]
            if d.empty:
                return [], []
            d["_bin"] = pd.cut(d["dollar_var"], bins=bin_edges, labels=False, include_lowest=True)
            agg = d.groupby("_bin")[col].mean()
            valid = [(bin_centers[int(b)], y) for b, y in zip(agg.index, agg.values)
                     if pd.notna(b) and pd.notna(y)]
            if not valid:
                return [], []
            xs, ys = zip(*valid)
            return list(xs), list(ys)

        techs = sorted(t for t in df["model_name"].dropna().unique() if t != "baseline")

        for tech in techs:
            tech_df = df[df["model_name"] == tech]
            color = _tech_color(tech)

            xs, ys = _binned(tech_df, "total_withdrawal_units")
            if xs:
                ax.plot(xs, ys, color=color, linewidth=1.8, linestyle="-",
                        marker="o", markersize=3, label=f"{tech} (RL)")

            # LP optimal — same tech color, dotted, to visually pair with the RL line
            xs_lp, ys_lp = _binned(tech_df, "optimal_withdrawal_units")
            if xs_lp:
                ax.plot(xs_lp, ys_lp, color=color, linewidth=1.4, linestyle=":",
                        marker="^", markersize=3, label=f"{tech} (LP)")

        # Baseline — aggregated across all storage types (technology-agnostic policy)
        baseline_df = df[df["model_name"] == "baseline"]
        xs_b, ys_b = _binned(baseline_df, "total_withdrawal_units")
        if xs_b:
            ax.plot(xs_b, ys_b, color=TECH_COLORS["baseline"], linewidth=1.6,
                    linestyle="--", marker="s", markersize=3, label="baseline")

        ax.set_xlabel("H2 Price Volatility ($/kg)", fontsize=9, color=TEXT_COLOR)
        ax.set_ylabel("Mean Withdrawal Units", fontsize=9, color=TEXT_COLOR)
        ax.set_title("Withdrawal Efficiency by Technology vs Price Volatility",
                     fontsize=10, color=TEXT_COLOR, pad=5)
        ax.legend(fontsize=7, framealpha=0.85, ncol=2)
        ax.tick_params(labelsize=8)

    def plot_zero_wdw_var(self, ax: plt.Axes) -> None:
        """
        KDE of dollar_var for all scenarios where LP optimal_withdrawal_units == 0,
        grouped by storage_type. All technology KDEs are overlaid on the same axes.
        These are the 'untradeabble' scenarios — useful for understanding what market
        conditions make storage unprofitable even under the LP benchmark.
        """
        df = self._load_zeros()
        if df.empty:
            ax.text(0.5, 0.5, "No zero-LP scenarios found",
                    transform=ax.transAxes, ha="center", va="center", fontsize=9)
            return

        techs = sorted(df["storage_type"].dropna().unique())
        for tech in techs:
            sub = df[df["storage_type"] == tech]["dollar_var"].dropna()
            sub = sub[sub > 0]
            if len(sub) < 5:
                continue
            sns.kdeplot(sub, ax=ax, color=_tech_color(tech), label=tech,
                        linewidth=1.8, fill=True, alpha=0.15)

        ax.set_xlim(right=0.5)
        ax.set_xlabel("H2 Price Volatility — std ($/kg)", fontsize=9, color=TEXT_COLOR)
        ax.set_ylabel("Density", fontsize=9, color=TEXT_COLOR)
        ax.set_title("Volatility Distribution — Scenarios with Zero LP Withdrawal",
                     fontsize=10, color=TEXT_COLOR, pad=5)
        ax.legend(fontsize=8, framealpha=0.85)
        ax.tick_params(labelsize=8)


    # ------------------------------------------------------------------ #
    # Single-plot save                                                      #
    # ------------------------------------------------------------------ #

    _PER_TECH_PLOTS = frozenset({
        "plot_2d_heatmap",
        "plot_levelized_costs",
        "plot_withdrawal_efficiency",
    })
    _GLOBAL_PLOTS = frozenset({
        "plot_zero_wdw_var",
        "plot_withdrawal_efficiency_all",
    })

    def save_plot(
        self,
        plot_name: str,
        output_path: str,
        tech: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 7),
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        dpi: int = 150,
        **plot_kwargs,
    ) -> plt.Figure:
        """
        Render a single named plot as a standalone figure and save it to disk.

        Parameters
        ----------
        plot_name : str
            One of: 'plot_2d_heatmap', 'plot_levelized_costs',
                    'plot_withdrawal_efficiency', 'plot_zero_wdw_var'.
        output_path : str
            Destination file path (PNG recommended).
        tech : str, optional
            Required for per-technology plots (all except 'plot_zero_wdw_var').
        figsize : (width, height), default (10, 7)
        title : str, optional
            Overrides the default plot title if provided.
        xlabel : str, optional
            Overrides the default x-axis label if provided.
        ylabel : str, optional
            Overrides the default y-axis label if provided.
        dpi : int, default 150
        **plot_kwargs
            Extra keyword arguments forwarded to the plot method.
            Example: eval_dollar_var=2.0 for plot_2d_heatmap.

        Example
        -------
        viz.save_plot(
            "plot_2d_heatmap", "out/heatmap_lh2.png",
            tech="lh2", figsize=(9, 7), eval_dollar_var=2.0,
        )
        """
        all_known = self._PER_TECH_PLOTS | self._GLOBAL_PLOTS
        if plot_name not in all_known:
            raise ValueError(
                f"Unknown plot_name '{plot_name}'. "
                f"Available: {sorted(all_known)}"
            )

        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=figsize)

        if plot_name in self._PER_TECH_PLOTS:
            if tech is None:
                raise ValueError(f"'{plot_name}' requires a tech= argument.")
            tech_df, baseline_df = self._tech_split(tech)
            if len(tech_df) == 0:
                raise ValueError(f"No simulation data for tech='{tech}'.")
            getattr(self, plot_name)(ax, tech_df, baseline_df, **plot_kwargs)
        else:
            getattr(self, plot_name)(ax, **plot_kwargs)

        if title is not None:
            ax.set_title(title, fontsize=10, color=TEXT_COLOR, pad=5)
        if xlabel is not None:
            ax.set_xlabel(xlabel, fontsize=9, color=TEXT_COLOR)
        if ylabel is not None:
            ax.set_ylabel(ylabel, fontsize=9, color=TEXT_COLOR)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.tight_layout()
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"[{plot_name}] Saved to {output_path}")
        return fig

    def save_grid(
        self,
        plots: list,
        output_path: str,
        ncols: int = 1,
        panel_size: Tuple[int, int] = (9, 7),
        dpi: int = 150,
        title: Optional[str] = None,
    ) -> plt.Figure:
        """
        Render multiple named plots into a single figure and save to disk.

        Parameters
        ----------
        plots : list of (plot_name, tech_or_None) or (plot_name, tech_or_None, kwargs)
            2-tuples use plot method defaults; 3-tuples pass the dict as **kwargs
            to the plot method, enabling per-panel parameter overrides.
            Use None as tech for global plots.
        output_path : str
            Destination file path.
        ncols : int, default 1
            Number of columns in the grid. Rows are computed automatically.
        panel_size : (width, height) per panel, default (9, 7)
        dpi : int, default 150
        title : str, optional
            Overall figure suptitle.

        Examples
        --------
        # Uniform kwargs across all panels
        viz.save_grid(
            [("plot_2d_heatmap", t, {"eval_dollar_var": 2.0}) for t in techs],
            "out/heatmaps.png", ncols=2,
        )

        # Mixed: per-panel kwargs
        viz.save_grid(
            [
                ("plot_2d_heatmap", "lh2",  {"eval_dollar_var": 1.0}),
                ("plot_2d_heatmap", "lnh3", {"eval_dollar_var": 3.0}),
                ("plot_zero_wdw_var", None),
            ],
            "out/mixed.png", ncols=1,
        )
        """
        n = len(plots)
        nrows = (n + ncols - 1) // ncols
        figsize = (panel_size[0] * ncols, panel_size[1] * nrows)

        plt.style.use("seaborn-v0_8-whitegrid")
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

        for idx, entry in enumerate(plots):
            plot_name, tech = entry[0], entry[1]
            kwargs = entry[2] if len(entry) == 3 else {}
            ax = axes[idx // ncols][idx % ncols]
            try:
                if plot_name in self._PER_TECH_PLOTS:
                    if tech is None:
                        raise ValueError(f"'{plot_name}' requires a tech.")
                    tech_df, baseline_df = self._tech_split(tech)
                    if len(tech_df) == 0:
                        raise ValueError(f"No data for tech='{tech}'.")
                    getattr(self, plot_name)(ax, tech_df, baseline_df, **kwargs)
                else:
                    getattr(self, plot_name)(ax, **kwargs)
            except Exception as e:
                ax.text(0.5, 0.5, f"Error:\n{e}", transform=ax.transAxes,
                        ha="center", va="center", fontsize=8, color="red")

        # Hide any unused axes
        for idx in range(n, nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)

        if title:
            fig.suptitle(title, fontsize=13, color=TEXT_COLOR, y=1.01)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.tight_layout()
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"[save_grid] Saved {n} panels to {output_path}")
        return fig

    # ------------------------------------------------------------------ #
    # Full suite — per technology                                           #
    # ------------------------------------------------------------------ #

    def plot_full_suite(
        self, tech: str, output_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Generate the complete 9-panel diagnostic suite for a single storage
        technology.  Panels 1-7 compare the RL model for `tech` against the
        baseline policy and the LP optimal solution.  Panels 8-9 are global
        zero-LP analyses shared across all technologies.

        Parameters
        ----------
        tech : str
            Storage technology identifier — one of {lh2, lnh3, lohc, meoh}.
        output_path : str, optional
            Save path for the figure PNG.  If None, the figure is shown
            interactively.

        Returns
        -------
        matplotlib.figure.Figure
        """
        plt.style.use("seaborn-v0_8-whitegrid")
        tech_df, baseline_df = self._tech_split(tech)

        if len(tech_df) == 0:
            raise ValueError(
                f"No simulation data found for technology '{tech}'. "
                f"Available model_names: {sorted(self._load_runs()['model_name'].dropna().unique())}"
            )

        fig, axes = plt.subplots(2, 2, figsize=(16, 11))
        fig.suptitle(
            f"RL H₂ Trading Agent — {tech.upper()} Performance Diagnostic Suite"
            f"  (n_rl={len(tech_df):,}  n_baseline={len(baseline_df):,})",
            fontsize=14, color=TEXT_COLOR, y=1.01,
        )

        all_panels = [
            ("plot_2d_heatmap",            lambda ax: self.plot_2d_heatmap(ax, tech_df, baseline_df)),
            ("plot_levelized_costs",       lambda ax: self.plot_levelized_costs(ax, tech_df, baseline_df)),
            ("plot_withdrawal_efficiency", lambda ax: self.plot_withdrawal_efficiency(ax, tech_df, baseline_df)),
            ("plot_zero_wdw_var",          self.plot_zero_wdw_var),
        ]

        for (name, fn), ax in zip(all_panels, axes.flat):
            try:
                fn(ax)
            except Exception as e:
                ax.text(0.5, 0.5, f"Error in {name}:\n{e}",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=8, color="red", wrap=True)
                ax.set_title(name, fontsize=9)

        fig.tight_layout(rect=[0, 0, 1, 0.98])

        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"[{tech}] Saved to {output_path}")
        else:
            plt.show()

        return fig

    def plot_all_suites(self, output_dir: str = "results") -> dict:
        """
        Generate and save one diagnostic figure per RL technology.

        Files are written to output_dir/diagnostic_{tech}.png.
        Returns a dict mapping tech → Figure.
        """
        df = self._load_runs()
        techs = sorted(t for t in df["model_name"].dropna().unique() if t != "baseline")

        os.makedirs(output_dir, exist_ok=True)
        figs = {}
        for tech in techs:
            path = os.path.join(output_dir, f"diagnostic_{tech}.png")
            figs[tech] = self.plot_full_suite(tech, output_path=path)
        return figs
