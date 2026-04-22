"""
SimulationVisualizer: H2 Spot Price Monte Carlo Distribution Diagnostic
================================================================================
Efficiently summarizes the distribution and variance of a large memmap of
simulated hydrogen spot price paths without loading the full array into RAM.

USAGE:
------
    viz = SimulationVisualizer("results/h2_simulations.npy")
    viz.plot(output_path="results/h2_distribution_diagnostic.png")

The input array is expected to have shape (n_sims, n_days) in float32.
All statistical passes are done via memmap slicing to keep peak RAM reasonable.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from scipy.stats import skew as scipy_skew
from typing import Optional
import warnings

warnings.filterwarnings("ignore")


class SimulationVisualizer:
    """
    4-panel diagnostic figure for a Monte Carlo ensemble of price paths.

    Panels
    ------
    1. Fan chart — percentile bands + sample paths
    2. Temporal density heatmap — 2D histogram (price × time)
    3. Violin plots — cross-sectional distribution at key timesteps
    4. Variance & skewness over time
    """

    # --- color palette (matches codebase conventions) ---
    MEDIAN_COLOR = "#2980b9"
    STD_COLOR = "#e74c3c"
    SKEW_COLOR = "#9b59b6"
    SAMPLE_PATH_COLOR = "#95a5a6"
    TEXT_COLOR = "#2c3e50"
    GRID_COLOR = "#ecf0f1"

    # Percentile bands (inner to outer)
    BAND_PAIRS = [
        (25, 75, 0.22),
        (10, 90, 0.14),
        (5,  95, 0.09),
        # (1,  99, 0.05),
    ]
    ALL_PCTS = [5, 10, 25, 50, 75, 90, 95]

    def __init__(
        self,
        sims_path: str,
        n_sample_paths: int = 100,
        violin_downsample: int = 50_000,
        heatmap_bins: int = 200,
        heatmap_col_batch: int = 30,
    ):
        self.sims_path = sims_path
        self.n_sample_paths = n_sample_paths
        self.violin_downsample = violin_downsample
        self.heatmap_bins = heatmap_bins
        self.heatmap_col_batch = heatmap_col_batch

        # Populated by _compute_stats()
        self._pcts: Optional[np.ndarray] = None   # (9, n_days)
        self._std: Optional[np.ndarray] = None    # (n_days,)
        self._skewness: Optional[np.ndarray] = None  # (n_days,)
        self._sample_paths: Optional[np.ndarray] = None  # (n_sample, n_days)
        self._mean: Optional[np.ndarray] = None     # (n_days,)
        self._density: Optional[np.ndarray] = None  # (n_bins, n_days)
        self._price_edges: Optional[np.ndarray] = None  # (n_bins+1,)
        self._n_sims: int = 0
        self._n_days: int = 0

    # ------------------------------------------------------------------ #
    # Internal computation helpers                                         #
    # ------------------------------------------------------------------ #

    def _load_memmap(self) -> np.ndarray:
        return np.load(self.sims_path, mmap_mode="r")

    def _compute_stats(self) -> None:
        print("Loading memmap...")
        sims = self._load_memmap()
        self._n_sims, self._n_days = sims.shape
        n_sims, n_days = self._n_sims, self._n_days

        # --- percentiles (one pass, reads full array) ---
        print(f"Computing percentiles over {n_sims:,} paths × {n_days} days...")
        self._pcts = np.percentile(sims, self.ALL_PCTS, axis=0).astype(np.float32)
        print(self._pcts)

        # --- global mean (one pass) ---
        print("Computing mean...")
        self._mean = np.mean(sims, axis=0).astype(np.float32)

        # --- std and skewness (one pass each) ---
        print("Computing std and skewness...")
        # self._std = np.std(sims, axis=0).astype(np.float32)
        # Manual skewness to avoid loading full array in scipy (same cost, explicit)
        # mean = np.mean(sims, axis=0)
        # diff = sims - mean  # broadcasts: (n_sims, n_days), reads full array
        # self._skewness = (np.mean(diff ** 3, axis=0) / (self._std ** 3 + 1e-12)).astype(np.float32)

        # --- sample paths (cheap stride subsample) ---
        stride = max(1, n_sims // self.n_sample_paths)
        self._sample_paths = np.array(sims[12::stride][: self.n_sample_paths], dtype=np.float32)

        # --- density heatmap (column-batch iteration) ---
        print("Building density heatmap...")
        global_min = float(self._pcts[0].min())   # 5st percentile floor
        global_max = float(self._pcts[-1].max())  # 95th percentile ceiling
        self._price_edges = np.linspace(global_min, global_max, self.heatmap_bins + 1)
        density = np.zeros((self.heatmap_bins, n_days), dtype=np.float32)

        for t_start in range(0, n_days, self.heatmap_col_batch):
            t_end = min(t_start + self.heatmap_col_batch, n_days)
            batch = sims[:, t_start:t_end]  # (n_sims, batch_size) — memmap read
            for i, t in enumerate(range(t_start, t_end)):
                counts, _ = np.histogram(batch[:, i], bins=self._price_edges)
                density[:, t] = counts

        self._density = density
        print("Stats computed.")

    # ------------------------------------------------------------------ #
    # Plot helpers                                                         #
    # ------------------------------------------------------------------ #

    def _plot_fan_chart(self, ax: plt.Axes) -> None:
        pcts = self._pcts
        n_days = self._n_days
        days = np.arange(n_days)

        # Sample paths — each a distinct shade cycling through bright hues
        n_paths = len(self._sample_paths)
        path_colors = [
            mcolors.hsv_to_rgb((i / n_paths, 0.85, 0.95))
            for i in range(n_paths)
        ]
        for path, color in zip(self._sample_paths, path_colors):
            ax.plot(days, path, color=color, alpha=0.5, linewidth=1, rasterized=True)

        # Shaded bands (outer to inner so darker band renders on top)
        for lo, hi, alpha in reversed(self.BAND_PAIRS):
            lo_idx = self.ALL_PCTS.index(lo)
            hi_idx = self.ALL_PCTS.index(hi)
            ax.fill_between(
                days,
                pcts[lo_idx],
                pcts[hi_idx],
                alpha=alpha,
                color=self.MEDIAN_COLOR,
                linewidth=0,
            )

        # Global mean line
        ax.plot(days, self._mean, color=self.MEDIAN_COLOR, linewidth=1.8, label="Mean", zorder=5)

        # 5th / 95th boundary lines
        ax.plot(days, pcts[self.ALL_PCTS.index(5)],  color=self.MEDIAN_COLOR, linewidth=0.7, linestyle="--", alpha=0.6)
        ax.plot(days, pcts[self.ALL_PCTS.index(95)], color=self.MEDIAN_COLOR, linewidth=0.7, linestyle="--", alpha=0.6)

        ax.set_xlabel("Trading Day", fontsize=9, color=self.TEXT_COLOR)
        ax.set_ylabel("H₂ Spot Price ($/kg)", fontsize=9, color=self.TEXT_COLOR)
        ax.set_title("Fan Chart — Percentile Bands & Sample Paths", fontsize=10, color=self.TEXT_COLOR, pad=6)

        legend_elements = [
            Line2D([0], [0], color=self.MEDIAN_COLOR, linewidth=1.8, label="Mean"),
            plt.Rectangle((0, 0), 1, 1, fc=self.MEDIAN_COLOR, alpha=0.22, label="25th–75th pct"),
            plt.Rectangle((0, 0), 1, 1, fc=self.MEDIAN_COLOR, alpha=0.14, label="10th–90th pct"),
            plt.Rectangle((0, 0), 1, 1, fc=self.MEDIAN_COLOR, alpha=0.09, label="5th–95th pct"),
            # plt.Rectangle((0, 0), 1, 1, fc=self.MEDIAN_COLOR, alpha=0.05, label="1st–99th pct"),
            Line2D([0], [0], color=mcolors.hsv_to_rgb((0.55, 0.75, 0.95)), linewidth=0.8, alpha=0.7, label=f"Sample paths (n={len(self._sample_paths)})"),
        ]
        ax.legend(handles=legend_elements, fontsize=8, loc="upper left", framealpha=0.85)
        ax.tick_params(labelsize=8)

    def _plot_density_heatmap(self, ax: plt.Axes) -> None:
        density = self._density.copy().astype(float)
        # Log scale: add 1 to avoid log(0)
        log_density = np.log1p(density)

        price_centers = 0.5 * (self._price_edges[:-1] + self._price_edges[1:])
        price_min, price_max = self._price_edges[0], self._price_edges[-1]

        im = ax.imshow(
            log_density,
            aspect="auto",
            origin="lower",
            extent=[0, self._n_days - 1, price_min, price_max],
            cmap="viridis",
            interpolation="nearest",
        )
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("log(1 + count)", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

        # Overlay median
        med_idx = self.ALL_PCTS.index(50)
        ax.plot(np.arange(self._n_days), self._pcts[med_idx], color="white", linewidth=1.0, alpha=0.7, label="Median")

        ax.set_xlabel("Trading Day", fontsize=9, color=self.TEXT_COLOR)
        ax.set_ylabel("H₂ Spot Price ($/kg)", fontsize=9, color=self.TEXT_COLOR)
        ax.set_title("Temporal Density Heatmap", fontsize=10, color=self.TEXT_COLOR, pad=6)
        ax.tick_params(labelsize=8)

    def _plot_violins(self, ax: plt.Axes) -> None:
        sims = self._load_memmap()
        n_days = self._n_days

        # Key timesteps: day 0, ~every 6 weeks, and final day
        key_days = [0, 42, 84, 126, 168, 210, n_days - 1]
        key_days = sorted(set(min(d, n_days - 1) for d in key_days))

        rng = np.random.default_rng(42)
        violin_data = []
        for t in key_days:
            col = np.array(sims[:, t], dtype=np.float32)
            if len(col) > self.violin_downsample:
                idx = rng.choice(len(col), self.violin_downsample, replace=False)
                col = col[idx]
            violin_data.append(col)

        parts = ax.violinplot(
            violin_data,
            positions=list(range(len(key_days))),
            widths=0.7,
            showmedians=True,
            showextrema=False,
        )
        for body in parts["bodies"]:
            body.set_facecolor(self.MEDIAN_COLOR)
            body.set_alpha(0.45)
            body.set_edgecolor(self.MEDIAN_COLOR)
        parts["cmedians"].set_color(self.TEXT_COLOR)
        parts["cmedians"].set_linewidth(1.5)

        ax.set_xticks(range(len(key_days)))
        ax.set_xticklabels([f"Day {d}" for d in key_days], fontsize=8, rotation=30, ha="right")
        ax.set_xlabel("Timestep", fontsize=9, color=self.TEXT_COLOR)
        ax.set_ylabel("H₂ Spot Price ($/kg)", fontsize=9, color=self.TEXT_COLOR)
        ax.set_title("Cross-Sectional Distribution at Key Timesteps", fontsize=10, color=self.TEXT_COLOR, pad=6)
        ax.tick_params(labelsize=8)

    def _plot_variance_skewness(self, ax: plt.Axes) -> None:
        days = np.arange(self._n_days)

        ax.plot(days, self._std, color=self.STD_COLOR, linewidth=1.5, label="Std Dev")
        ax.set_xlabel("Trading Day", fontsize=9, color=self.TEXT_COLOR)
        ax.set_ylabel("Std Dev ($/kg)", fontsize=9, color=self.STD_COLOR)
        ax.tick_params(axis="y", labelcolor=self.STD_COLOR, labelsize=8)
        ax.tick_params(axis="x", labelsize=8)

        ax2 = ax.twinx()
        ax2.plot(days, self._skewness, color=self.SKEW_COLOR, linewidth=1.2, linestyle="--", alpha=0.85, label="Skewness")
        ax2.axhline(0, color=self.SKEW_COLOR, linewidth=0.5, linestyle=":", alpha=0.5)
        ax2.set_ylabel("Skewness", fontsize=9, color=self.SKEW_COLOR)
        ax2.tick_params(axis="y", labelcolor=self.SKEW_COLOR, labelsize=8)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left", framealpha=0.85)
        ax.set_title("Variance & Skewness Over Time", fontsize=10, color=self.TEXT_COLOR, pad=6)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def plot(self, output_path: Optional[str] = None) -> plt.Figure:
        """
        Build and optionally save the 4-panel diagnostic figure.

        Parameters
        ----------
        output_path : str, optional
            File path to save the PNG. If None, the figure is displayed.

        Returns
        -------
        matplotlib Figure
        """
        plt.style.use("seaborn-v0_8-whitegrid")

        if self._pcts is None:
            self._compute_stats()

        fig = plt.figure(figsize=(32, 10))
        fig.suptitle(
            f"H₂ Spot Price Simulation Distribution  (N={self._n_sims:,} paths, T={self._n_days} days)",
            fontsize=13,
            color=self.TEXT_COLOR,
            y=0.98,
        )

        # Layout: top row = fan chart (full width), bottom row = 3 panels
        gs = fig.add_gridspec(1, 3, hspace=0.38, wspace=0.35,
                              top=0.93, bottom=0.08, left=0.07, right=0.97)

        ax_fan = fig.add_subplot(gs[0, :])       # top row, all 3 columns
        # ax_heat = fig.add_subplot(gs[1, 0])
        # ax_violin = fig.add_subplot(gs[1, 1])
        # ax_var = fig.add_subplot(gs[1, 2])

        self._plot_fan_chart(ax_fan)
        # self._plot_density_heatmap(ax_heat)
        # self._plot_violins(ax_violin)
        # self._plot_variance_skewness(ax_var)

        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"Saved to {output_path}")
        else:
            plt.show()

        return fig
