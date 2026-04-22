"""
StorageValuationVisualizer: LP Storage Valuation Diagnostic Suite
================================================================================
Ingests the dict output of storage_valuation() and generates a diagnostic
panel covering:
  - Forward curve used for optimisation
  - Action schedule (inject / withdraw per step)
  - Inventory path vs capacity bounds
  - Net cash-flow and discounted cash-flow schedules
  - Cumulative NPV build-up

USAGE:
------
    result = storage_valuation(fwd_curve, inventory=0.0)
    viz = StorageValuationVisualizer(result, fwd_curve=fwd_curve)
    viz.generate_diagnostic_suite(output_dir="analysis/lp_debug/", show=True)
"""
import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from constants import STORAGE_CAPACITY

plt.style.use("seaborn-v0_8-whitegrid")


class StorageValuationVisualizer:
    """
    Visual analysis of a single storage_valuation() result dict.

    Parameters
    ----------
    result      : dict returned by storage_valuation()
    fwd_curve   : original monthly forward curve passed to storage_valuation
                  (optional – enables a separate monthly-price panel)
    title_suffix: extra string appended to figure titles (e.g. scenario name)
    """

    def __init__(
        self,
        result: dict,
        fwd_curve: Optional[np.ndarray] = None,
        title_suffix: str = "",
    ):
        self.r = result
        self.fwd_curve = fwd_curve
        self.title_suffix = title_suffix
        self.T = len(result["action_schedule"])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_diagnostic_suite(
        self,
        output_dir: Optional[str] = None,
        show: bool = True,
        filename: str = "lp_diagnostic.png",
    ) -> None:
        """
        Produce and optionally save a multi-panel diagnostic figure.

        Panels
        ------
        1. Forward price curve (interpolated steps)
        2. Action schedule – bar chart (+inject / –withdraw)
        3. Inventory path vs [0, STORAGE_CAPACITY] bounds
        4. Undiscounted net cash-flow per step
        5. Discounted cash-flow per step + cumulative NPV
        6. Monthly forward curve (if provided)
        """
        n_panels = 6 if self.fwd_curve is not None else 5
        fig, axes = plt.subplots(n_panels, 1, figsize=(14, 4 * n_panels))
        fig.suptitle(
            f"Storage Valuation LP Diagnostic  {self.title_suffix}\n"
            f"Total NPV = {self.r['npv']:,.2f}   |   T = {self.T} steps",
            fontsize=13,
            y=1.005,
        )

        axes = list(axes)
        ax_idx = 0

        # Panel 1 – interpolated forward price
        self._plot_interpolated_price(axes[ax_idx])
        ax_idx += 1

        # Panel 2 – action schedule
        self._plot_action_schedule(axes[ax_idx])
        ax_idx += 1

        # Panel 3 – inventory path
        self._plot_inventory_path(axes[ax_idx])
        ax_idx += 1

        # Panel 4 – net cash-flow (undiscounted)
        self._plot_net_cashflow(axes[ax_idx])
        ax_idx += 1

        # Panel 5 – discounted cash-flow + cumulative NPV
        self._plot_discounted_cashflow(axes[ax_idx])
        ax_idx += 1

        # Panel 6 (optional) – original monthly forward curve
        if self.fwd_curve is not None:
            self._plot_monthly_forward_curve(axes[ax_idx])

        fig.tight_layout()

        if output_dir is not None:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            path = os.path.join(output_dir, filename)
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Saved diagnostic figure → {path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

    def plot_action_schedule(self, show: bool = True) -> plt.Figure:
        """Standalone action-schedule bar chart."""
        fig, ax = plt.subplots(figsize=(14, 4))
        self._plot_action_schedule(ax)
        fig.tight_layout()
        if show:
            plt.show()
        return fig

    def plot_inventory_path(self, show: bool = True) -> plt.Figure:
        """Standalone inventory-path chart."""
        fig, ax = plt.subplots(figsize=(14, 4))
        self._plot_inventory_path(ax)
        fig.tight_layout()
        if show:
            plt.show()
        return fig

    # ------------------------------------------------------------------
    # Private panel renderers
    # ------------------------------------------------------------------

    def _plot_interpolated_price(self, ax: plt.Axes) -> None:
        steps = np.arange(self.T)
        # reconstruct interpolated price from revenue/inject or withdraw signals
        # net_cashflow = revenue - purchase_cost; revenue = price * withdraw; cost = price * inject
        # recover price: wherever either activity is non-zero use that; else price unknown
        inject = self.r["inject_schedule"]
        withdraw = self.r["withdraw_schedule"]
        revenue = self.r["revenue_schedule"]
        cost = self.r["purchase_cost_schedule"]

        price = np.where(
            withdraw > 1e-9,
            revenue / np.where(withdraw > 1e-9, withdraw, 1.0),
            np.where(
                inject > 1e-9,
                cost / np.where(inject > 1e-9, inject, 1.0),
                np.nan,
            ),
        )

        ax.plot(
            steps, price, color="steelblue", linewidth=1.2, label="Interpolated price"
        )
        ax.set_ylabel("Price (€/unit)")
        ax.set_title("Interpolated Forward Price (LP time steps)")
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=20))
        ax.legend(fontsize=9)

    def _plot_action_schedule(self, ax: plt.Axes) -> None:
        steps = np.arange(self.T)
        actions = self.r["action_schedule"]
        colors = ["#2ecc71" if a >= 0 else "#e74c3c" for a in actions]
        ax.bar(
            steps,
            actions,
            color=colors,
            width=0.8,
            label="Action (+inject / −withdraw)",
        )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel("Volume (units)")
        ax.set_title("Action Schedule  (green = inject, red = withdraw)")
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=20))

        # annotate totals
        total_inj = self.r["inject_schedule"].sum()
        total_wdr = self.r["withdraw_schedule"].sum()
        ax.text(
            0.01,
            0.95,
            f"Total inject: {total_inj:,.1f}   Total withdraw: {total_wdr:,.1f}",
            transform=ax.transAxes,
            fontsize=9,
            va="top",
        )

    def _plot_inventory_path(self, ax: plt.Axes) -> None:
        steps = np.arange(self.T)
        inv = self.r["inventory_path"]
        ax.plot(steps, inv, color="darkorange", linewidth=1.5, label="Inventory")
        ax.axhline(
            STORAGE_CAPACITY,
            color="red",
            linestyle="--",
            linewidth=1.0,
            label=f"Capacity ({STORAGE_CAPACITY})",
        )
        ax.axhline(0, color="navy", linestyle="--", linewidth=1.0, label="Floor (0)")
        ax.fill_between(steps, 0, inv, alpha=0.15, color="darkorange")
        ax.set_ylabel("Inventory (units)")
        ax.set_title("Inventory Path vs Bounds")
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=20))
        ax.legend(fontsize=9)

        # annotate terminal inventory
        ax.annotate(
            f"Terminal: {inv[-1]:.1f}",
            xy=(steps[-1], inv[-1]),
            xytext=(-40, 10),
            textcoords="offset points",
            fontsize=8,
            arrowprops=dict(arrowstyle="->", lw=0.8),
        )

    def _plot_net_cashflow(self, ax: plt.Axes) -> None:
        steps = np.arange(self.T)
        ncf = self.r["net_cashflow_schedule"]
        colors = ["#27ae60" if v >= 0 else "#c0392b" for v in ncf]
        ax.bar(steps, ncf, color=colors, width=0.8, label="Net cash-flow")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel("Cash-flow (€)")
        ax.set_title("Undiscounted Net Cash-Flow per Step")
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=20))

        total_ncf = ncf.sum()
        ax.text(
            0.01,
            0.95,
            f"Total undiscounted: {total_ncf:,.2f}",
            transform=ax.transAxes,
            fontsize=9,
            va="top",
        )

    def _plot_discounted_cashflow(self, ax: plt.Axes) -> None:
        steps = np.arange(self.T)
        dcf = self.r["discounted_cashflows"]
        cumulative = np.cumsum(dcf)

        colors = ["#27ae60" if v >= 0 else "#c0392b" for v in dcf]
        ax.bar(steps, dcf, color=colors, width=0.8, alpha=0.7, label="Discounted CF")

        ax2 = ax.twinx()
        ax2.plot(
            steps, cumulative, color="purple", linewidth=1.5, label="Cumulative NPV"
        )
        ax2.set_ylabel("Cumulative NPV (€)", color="purple")
        ax2.tick_params(axis="y", labelcolor="purple")

        ax.set_ylabel("Discounted CF (€)")
        ax.set_title(
            f"Discounted Cash-Flow & Cumulative NPV   [NPV = {self.r['npv']:,.2f}]"
        )
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=20))

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper left")

    def _plot_monthly_forward_curve(self, ax: plt.Axes) -> None:
        months = np.arange(1, len(self.fwd_curve) + 1)
        ax.plot(
            months,
            self.fwd_curve,
            marker="o",
            color="teal",
            linewidth=1.5,
            markersize=4,
        )
        ax.set_xlabel("Month")
        ax.set_ylabel("Price (€/unit)")
        ax.set_title("Original Monthly Forward Curve")
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

        # shade contango / backwardation regions
        for i in range(len(self.fwd_curve) - 1):
            color = (
                "#d5f5e3" if self.fwd_curve[i + 1] > self.fwd_curve[i] else "#fadbd8"
            )
            ax.axvspan(months[i], months[i + 1], alpha=0.4, color=color)

        # legend proxy
        from matplotlib.patches import Patch

        ax.legend(
            handles=[
                Patch(facecolor="#d5f5e3", alpha=0.8, label="Contango"),
                Patch(facecolor="#fadbd8", alpha=0.8, label="Backwardation"),
            ],
            fontsize=9,
        )
