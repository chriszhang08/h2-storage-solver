"""
CurveVisualizer: Expert-Grade 3D Forward Curve Analysis
================================================================================
A production-ready Python class for analyzing commodity forward curves across
date, maturity, and price dimensions. Designed for initial data exploration
of term structures, with extensibility for multi-index comparison.

USAGE:
------
    # Single curve analysis
    viz = CurveVisualizer(commodity_name='Brent Crude', commodity_type='energy')
    viz.load_curve_data(
        df=melted,  # columns: [date, maturity, price]
        date_col='date',
        maturity_col='maturity',
        price_col='price'
    )
    viz.generate_diagnostic_suite(output_dir='./analysis/')

    # Multi-index comparison (future extension)
    viz.add_curve(df_ttf, label='TTF', color='steelblue')
    viz.add_curve(df_jkm, label='JKM', color='coral')
    viz.plot_multi_curve_comparison(save_path='comparison.png')

EXPERT FEATURES:
----------------
✓ Full 3D term structure analysis (date × maturity × price)
✓ Surface visualization showing curve evolution
✓ Contour plots for identification of structural shifts
✓ Curve shape analysis: slope, curvature, term premium
✓ Maturity-specific time series (spread analysis)
✓ Volatility surface & term structure of volatility
✓ Curve statistics: mean reversion, convexity, backwardation/contango
✓ Extensible for multi-index comparison
✓ Publication-quality matplotlib + interactive plotly
✓ Data quality diagnostics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3D projection
from matplotlib.patches import Rectangle
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from scipy.interpolate import CubicSpline
from scipy.stats import skew, kurtosis

from curve_factory.utils.curve_data_transformers import (
    long_to_matrix,
    matrix_to_long,
)
from curve_factory.utils.feature_generator_utils import compute_rolling_volatility

from typing import Dict, List, Optional, Tuple, Union
import os
from dataclasses import dataclass, field
import warnings

warnings.filterwarnings("ignore")


@dataclass
class CurveDataSet:
    """Container for a single curve dataset with metadata."""

    label: str
    df: pd.DataFrame
    dates: np.ndarray
    maturities: np.ndarray
    matrix: Optional[np.ndarray] = field(default=None)
    color: str = "#1f77b4"
    linestyle: str = "-"
    alpha: float = 0.8

    # Cached processed data
    pivot_surface: Optional[pd.DataFrame] = field(default=None, init=False)
    curve_stats: Optional[Dict] = field(default=None, init=False)


class CurveVisualizer:
    """
    Expert-grade 3D forward curve analyzer for commodities.
    Designed for term structure exploration and multi-index comparison.
    """

    def __init__(
        self,
        commodity_name: str = "Commodity",
        commodity_type: str = "energy",
        figsize: Tuple[float, float] = (14, 8),
        dpi: int = 300,
    ):
        """
        Initialize curve visualizer.

        Args:
            commodity_name: Display name (e.g., 'Brent Crude', 'TTF Gas')
            commodity_type: 'energy', 'agriculture', 'metals', 'soft'
            figsize: Default matplotlib figure size
            dpi: Resolution for saved figures
        """
        self.commodity_name = commodity_name
        self.commodity_type = commodity_type
        self.figsize = figsize
        self.dpi = dpi

        # Primary curve (required)
        self.primary_curve: Optional[CurveDataSet] = None

        # Comparison curves (optional, for future use)
        self.comparison_curves: Dict[str, CurveDataSet] = {}

        # Color palette
        self.colors = {
            "contango": "#2ecc71",  # Green: upward sloping
            "backwardation": "#e74c3c",  # Red: downward sloping
            "flat": "#95a5a6",  # Gray: flat
            "grid": "#ecf0f1",
            "text": "#2c3e50",
        }

        plt.style.use("seaborn-v0_8-whitegrid")
        sns.set_palette("husl")

    # ================================================================ #
    # DATA LOADING & PREPROCESSING
    # ================================================================ #

    def load_curve_data(
        self,
        matrix: np.ndarray,
        maturities: np.ndarray,
        dates: np.ndarray,
        label: str = "Primary",
    ) -> "CurveVisualizer":
        """
        Load primary curve dataset.

        Args:
            matrix: Price matrix of shape (n_dates, n_maturities).
                Rows are time-series observations; columns are maturities.
            maturities: 1-D array of maturity values for each column.
                Defaults to [1, 2, ..., n_maturities].
            dates: 1-D array of date labels for each row.
                Defaults to [0, 1, ..., n_dates-1].
            label: Display label

        Returns:
            self for method chaining
        """
        df = matrix_to_long(matrix, dates, maturities)
        self.primary_curve = CurveDataSet(
            label=label, df=df, color="#1f77b4", dates=dates, maturities=maturities, matrix=matrix
        )

        date_min, date_max = df["date"].min(), df["date"].max()
        date_min_str = date_min.date() if hasattr(date_min, "date") else date_min
        date_max_str = date_max.date() if hasattr(date_max, "date") else date_max
        print(f"✓ Loaded {label}: {len(df)} observations")
        print(f"  Date range: {date_min_str} to {date_max_str}")
        print(
            f"  Maturity range: {df['maturity'].min():.0f} to {df['maturity'].max():.0f}"
        )
        print(f"  Price range: {df['price'].min():.2f} to {df['price'].max():.2f}")

        return self

    def add_comparison_curve(
        self,
        matrix: np.ndarray,
        label: str,
        maturities: np.ndarray,
        dates: np.ndarray,
        color: str = None,
    ) -> "CurveVisualizer":
        """Add comparison curve for multi-index analysis."""
        if color is None:
            color = plt.cm.tab10(len(self.comparison_curves))

        df = matrix_to_long(matrix, dates, maturities)
        self.comparison_curves[label] = CurveDataSet(label=label, df=df, color=color)
        print(f"✓ Added {label} for comparison")
        return self

    # ================================================================ #
    # CORE ANALYSIS: SURFACE & CONTOUR
    # ================================================================ #

    def plot_surface_3d_interactive(self, save_path: Optional[str] = None):
        """
        Interactive 3D surface plot: date × maturity → price
        Best for identifying term structure evolution.
        """
        if self.primary_curve is None:
            raise ValueError("Load data first with load_curve_data()")

        df = self.primary_curve.df

        # Pivot to surface grid
        pivot = long_to_matrix(df).fillna(method="ffill").dropna(how="all")

        # Thin if too dense
        if len(pivot) > 200:
            pivot = pivot.iloc[:: max(1, len(pivot) // 200)]

        # Convert dates to numeric
        dates_numeric = np.arange(len(pivot))

        fig = go.Figure(
            data=[
                go.Surface(
                    z=pivot.values,
                    x=dates_numeric,
                    y=pivot.columns.values,
                    colorscale="viridis",
                    colorbar=dict(title=f"Price\n($/MWh)"),
                )
            ]
        )

        fig.update_layout(
            title=f"{self.commodity_name}: Forward Curve Surface<br>Date × Maturity → Price",
            scene=dict(
                xaxis_title="Date",
                yaxis_title=f"Maturity (months)",
                zaxis_title="Price",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
            ),
            width=1400,
            height=800,
            hovermode="closest",
        )

        if save_path:
            fig.write_html(save_path)
            print(f"✓ Saved interactive 3D: {save_path}")

        fig.show()
        return fig

    def plot_contour_heatmap(self, save_path: Optional[str] = None):
        """
        Contour/heatmap showing price landscape.
        Highlights structural shifts and regime changes.
        """
        if self.primary_curve is None:
            raise ValueError("Load data first")

        df = self.primary_curve.df

        pivot = long_to_matrix(df).fillna(method="ffill").dropna(how="all")

        fig = go.Figure(
            data=[
                go.Contour(
                    z=pivot.values.T,
                    x=pivot.index,
                    y=pivot.columns,
                    colorscale="RdYlGn",
                    contours=dict(showlabels=True, labelfont=dict(size=10)),
                )
            ]
        )

        fig.update_layout(
            title=f"{self.commodity_name}: Price Contours (Date × Maturity)",
            xaxis_title="Date",
            yaxis_title="Maturity (months)",
            width=1200,
            height=600,
        )

        if save_path:
            fig.write_html(save_path)

        fig.show()
        return fig

    # ================================================================ #
    # CURVE SHAPE ANALYSIS
    # ================================================================ #

    def plot_curve_statistics_time_series(self, save_path: Optional[str] = None):
        """
        Time series of curve statistics: slope, curvature, volatility.
        Reveals market regimes and structural changes.
        """
        df = self.primary_curve.df

        stats_list = []

        for date in sorted(df["date"].unique()):
            curve = df[df["date"] == date].sort_values("maturity")

            if len(curve) < 3:
                continue

            prices = curve["price"].values
            maturities = curve["maturity"].values

            # Slope: change from short to long end
            slope = (prices[-1] - prices[0]) / (maturities[-1] - maturities[0])

            # Curvature: 2nd derivative
            if len(prices) >= 3:
                cs = CubicSpline(maturities, prices)
                mid_idx = len(maturities) // 2
                curvature = cs(maturities[mid_idx], 2)
            else:
                curvature = 0

            # Volatility: std of price
            vol = prices.std()

            # Term premium: long-end minus short-end
            term_premium = prices[-1] - prices[0]

            stats_list.append(
                {
                    "date": date,
                    "slope": slope,
                    "curvature": curvature,
                    "volatility": vol,
                    "term_premium": term_premium,
                    "mean_price": prices.mean(),
                }
            )

        stats_df = pd.DataFrame(stats_list)

        # 4-panel plot — one metric per subplot, no fill_between
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))

        panels = [
            (
                axes[0, 0],
                "slope",
                "Slope ($/month)",
                "#2980b9",
                "Curve Slope: Contango ↔ Backwardation",
                True,
            ),
            (
                axes[0, 1],
                "curvature",
                "Curvature (2nd deriv)",
                "#9b59b6",
                "Curve Convexity",
                True,
            ),
            (
                axes[1, 0],
                "volatility",
                "Volatility (σ)",
                "#e74c3c",
                "Curve Volatility",
                False,
            ),
            (
                axes[1, 1],
                "term_premium",
                "Term Premium ($)",
                "#16a085",
                "Long-End Premium",
                True,
            ),
        ]

        for ax, col, ylabel, color, title, zero_line in panels:
            ax.plot(stats_df["date"], stats_df[col], linewidth=1.5, color=color)
            if zero_line:
                ax.axhline(0, color="black", linestyle="--", linewidth=0.6, alpha=0.6)
            ax.set_ylabel(ylabel, fontweight="bold")
            ax.set_title(title)
            ax.grid(True, alpha=0.3)

        for ax in axes.flatten():
            ax.xaxis.set_major_locator(mdates.YearLocator(1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            ax.tick_params(axis="x", rotation=45)

        plt.suptitle(
            f"{self.commodity_name}: Curve Statistics Time Series",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            print(f"✓ Saved: {save_path}")

        plt.show()
        return fig, axes, stats_df

    # ================================================================ #
    # MATURITY-SPECIFIC ANALYSIS
    # ================================================================ #

    def plot_maturity_specific_spreads(self, save_path: Optional[str] = None):
        """
        Time series of prices at specific maturities.
        Reveals mean reversion and carry dynamics.
        """
        df = self.primary_curve.df

        n_maturities = len(self.primary_curve.maturities)

        unique_maturities = sorted(df["maturity"].unique())

        fig, axes = plt.subplots(
            n_maturities, 1, figsize=(14, 3 * n_maturities), sharex=True
        )

        if n_maturities == 1:
            axes = [axes]

        for idx, mat in enumerate(self.primary_curve.maturities):
            ts = df[df["maturity"] == mat].sort_values("date")

            ax = axes[idx]
            ax.plot(
                ts["date"],
                ts["price"],
                linewidth=2,
                color="#2980b9",
                marker="o",
                markersize=3,
            )

            # Mean reversion line
            mean_price = ts["price"].mean()
            ax.axhline(
                mean_price,
                color="red",
                linestyle="--",
                linewidth=1.5,
                alpha=0.7,
                label=f"Mean: {mean_price:.2f}",
            )

            # ±1 std bands
            std_price = ts["price"].std()
            ax.fill_between(
                ts["date"],
                mean_price - std_price,
                mean_price + std_price,
                alpha=0.2,
                color="red",
                label="±1σ",
            )

            ax.set_ylabel(f"M{mat:.0f} Price", fontweight="bold")
            ax.set_title(
                f"Maturity {mat:.0f}M: Time Series & Mean Reversion", fontweight="bold"
            )
            ax.legend(loc="upper right", fontsize=9)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Date", fontweight="bold")
        plt.suptitle(
            f"{self.commodity_name}: Maturity-Specific Price Dynamics",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            print(f"✓ Saved: {save_path}")

        plt.show()
        return fig, axes

    # ================================================================ #
    # CURVE SAMPLE GRID
    # ================================================================ #

    def plot_curve_sample_grid(
        self,
        n_samples: int = 100,
        seed: Optional[int] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Sample n_samples random dates and plot each forward curve on a 10×10 grid.

        Args:
            n_samples: Number of curves to sample (capped at the number of unique dates).
            seed:      Optional random seed for reproducibility.
            save_path: Optional path to save the figure.

        Returns:
            matplotlib Figure.
        """
        df = self.primary_curve.df
        unique_dates = np.array(sorted(df["date"].unique()))

        rng = np.random.default_rng(seed)
        n = min(n_samples, len(unique_dates))
        sampled_dates = sorted(rng.choice(unique_dates, size=n, replace=False))

        n_cols = 10
        n_rows = int(np.ceil(n / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.2, n_rows * 2.0))
        axes_flat = axes.flatten()

        for i, date in enumerate(sampled_dates):
            ax = axes_flat[i]
            curve = df[df["date"] == date].sort_values("maturity")
            ax.plot(curve["maturity"], curve["price"], linewidth=1.0, color="#2980b9")
            date_str = (
                date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date)
            )
            ax.set_title(date_str, fontsize=5.5, pad=2)
            ax.tick_params(labelsize=4.5)
            ax.grid(True, alpha=0.2)

        for ax in axes_flat[n:]:
            ax.axis("off")

        plt.suptitle(
            f"{self.commodity_name}: {n} Random Forward Curve Samples",
            fontsize=13,
            fontweight="bold",
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            print(f"✓ Saved: {save_path}")

        plt.show()
        return fig

    # ================================================================ #
    # VOLATILITY & TERM STRUCTURE
    # ================================================================ #

    def plot_volatility_surface(
        self, window: int = 30, save_path: Optional[str] = None
    ):
        """
        Rolling volatility surface: identifies volatility term structure.
        High vol in specific maturity buckets indicates market stress/opportunity.
        """
        df = self.primary_curve.df

        # Reconstruct (n_dates, n_maturities) price matrix, sorted by date and maturity
        price_pivot = long_to_matrix(df)

        # Delegate rolling log-return vol computation to the shared utility
        _, rolling_std = compute_rolling_volatility(
            price_pivot.values,
            window_size=window,
            annualize=True,
            log_returns=True,
        )

        # Wrap back into a labelled DataFrame and drop the leading NaN window
        vol_pivot = pd.DataFrame(
            rolling_std, index=price_pivot.index, columns=price_pivot.columns
        ).dropna(how="all")

        # Heatmap
        fig, ax = plt.subplots(figsize=(14, 6))

        sns.heatmap(
            vol_pivot.T,
            cmap="YlOrRd",
            ax=ax,
            cbar_kws={"label": f"Annualised Log-Return Vol ({window}d rolling)"},
        )

        ax.set_ylabel("Maturity (months)", fontweight="bold")
        ax.set_xlabel("Date", fontweight="bold")
        ax.set_title(
            f"{self.commodity_name}: Volatility Term Structure ({window}-day rolling)",
            fontsize=14,
            fontweight="bold",
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            print(f"✓ Saved: {save_path}")

        plt.show()
        return fig, ax

    # ================================================================ #
    # STATIC SURFACE PLOT
    # ================================================================ #

    def plot_surface_3d_static(
        self,
        matrix: Optional[np.ndarray] = None,
        dates: Optional[np.ndarray] = None,
        maturities: Optional[np.ndarray] = None,
        title: Optional[str] = None,
        z_label: str = "Price",
        cmap: str = "viridis",
        stride: int = 1,
        figsize: Tuple[float, float] = (14, 8),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot a static matplotlib 3D surface from a raw price matrix.

        Args:
            matrix:     Price matrix of shape (n_dates, n_maturities).
                        Defaults to the matrix loaded via load_curve_data().
            dates:      1-D array of date labels for the row axis.
                        Defaults to dates from the loaded primary curve.
            maturities: 1-D array of maturity labels for the column axis.
                        Defaults to maturities from the loaded primary curve.
            title:      Figure title. Defaults to commodity name.
            z_label:    Label for the Z axis.
            cmap:       Matplotlib colormap name.
            stride:     Row/column stride for surface mesh density (increase to thin).
            figsize:    Figure size tuple.
            save_path:  Optional path to save the figure.

        Returns:
            matplotlib Figure.
        """
        if self.primary_curve is None and matrix is None:
            raise ValueError("Load data first with load_curve_data() or pass matrix explicitly.")

        if matrix is None:
            matrix = self.primary_curve.matrix
        if dates is None:
            dates = self.primary_curve.dates
        if maturities is None:
            maturities = self.primary_curve.maturities

        n_dates, n_mats = matrix.shape

        if maturities is None:
            maturities = np.arange(n_mats)
        if dates is None:
            dates = np.arange(n_dates)

        # Build a numeric date axis (ordinal if datetime, else identity)
        if hasattr(dates[0], "toordinal"):
            dates_numeric = np.array([d.toordinal() for d in dates], dtype=float)
        else:
            dates_numeric = np.asarray(dates, dtype=float)

        # Normalize date axis to [0, n_dates) so axis ticks are interpretable
        dates_norm = dates_numeric - dates_numeric[0]

        mat_grid, date_grid = np.meshgrid(maturities, dates_norm)

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

        surf = ax.plot_surface(
            date_grid,
            mat_grid,
            matrix,
            cmap=cmap,
            rstride=stride,
            cstride=stride,
            linewidth=0,
            antialiased=True,
            alpha=0.9,
        )

        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=12, label=z_label)

        # Label the date axis with actual date strings at a few tick positions
        n_ticks = min(6, n_dates)
        tick_pos = np.linspace(0, dates_norm[-1], n_ticks)
        tick_idx = np.linspace(0, n_dates - 1, n_ticks, dtype=int)
        tick_labels = [
            (
                dates[i].strftime("%Y-%m")
                if hasattr(dates[i], "strftime")
                else str(dates[i])
            )
            for i in tick_idx
        ]
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labels, fontsize=7, rotation=15)

        ax.set_xlabel("Date", labelpad=10)
        ax.set_ylabel("Maturity (months)", labelpad=8)
        ax.set_zlabel(z_label, labelpad=8)
        ax.set_title(
            title if title else f"{self.commodity_name}: Forward Curve Surface",
            fontsize=13,
            fontweight="bold",
            pad=15,
        )

        ax.view_init(elev=28, azim=-55)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            print(f"Saved: {save_path}")

        plt.show()
        return fig

    # Approximate MATLAB's parula colormap via key colour stops (blue → cyan → green → yellow)
    _PARULA = plt.matplotlib.colors.LinearSegmentedColormap.from_list(
        "parula",
        [
            (0.2081, 0.1663, 0.5292),  # deep indigo
            (0.0117, 0.5594, 0.8972),  # sky blue
            (0.0781, 0.6983, 0.8569),  # cyan
            (0.5441, 0.8472, 0.6804),  # seafoam
            (0.9769, 0.9839, 0.0805),  # yellow
        ],
    )

    def plot_surface_3d_wireframe(
        self,
        matrix: Optional[np.ndarray] = None,
        dates: Optional[np.ndarray] = None,
        maturities: Optional[np.ndarray] = None,
        title: Optional[str] = None,
        z_label: str = "Price",
        cmap=None,
        stride: int = 1,
        elev: float = 30,
        azim: float = -37.5,
        figsize: Tuple[float, float] = (14, 8),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot a MATLAB surf-styled 3D surface using mpl_toolkits Axes3D.

        Mimics MATLAB's default surf aesthetics:
          - Parula colormap (blue → cyan → green → yellow) mapped to Z values.
          - Flat-shaded faces with dark gray edges [0.1294, 0.1294, 0.1294],
            producing MATLAB's characteristic faceted grid appearance.
          - Light gray axis panes with white grid lines.
          - White figure background.

        Args:
            matrix:     Price matrix (n_dates, n_maturities). Defaults to loaded curve.
            dates:      Row-axis date labels. Defaults to loaded curve dates.
            maturities: Column-axis maturity labels. Defaults to loaded curve maturities.
            title:      Figure title. Defaults to commodity name.
            z_label:    Z-axis label.
            cmap:       Colormap. Defaults to the parula approximation.
            stride:     Row/column stride controlling mesh density.
            elev:       Camera elevation angle in degrees (MATLAB default ≈ 30).
            azim:       Camera azimuth angle in degrees (MATLAB default ≈ −37.5).
            figsize:    Figure size tuple.
            save_path:  Optional path to save the figure.

        Returns:
            matplotlib Figure.
        """
        if self.primary_curve is None and matrix is None:
            raise ValueError("Load data first with load_curve_data() or pass matrix explicitly.")

        if matrix is None:
            matrix = self.primary_curve.matrix
        if dates is None:
            dates = self.primary_curve.dates
        if maturities is None:
            maturities = self.primary_curve.maturities
        if cmap is None:
            cmap = self._PARULA

        n_dates, n_mats = matrix.shape

        if hasattr(dates[0], "toordinal"):
            dates_numeric = np.array([d.toordinal() for d in dates], dtype=float)
        else:
            dates_numeric = np.asarray(dates, dtype=float)

        dates_norm = dates_numeric - dates_numeric[0]
        mat_grid, date_grid = np.meshgrid(maturities, dates_norm)

        # ---- figure / axes ------------------------------------------------
        fig = plt.figure(figsize=figsize, facecolor="white")
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor("white")

        # ---- MATLAB surf: flat-shaded faces + dark gray mesh edges ----------
        _MATLAB_EDGE = (0.1294, 0.1294, 0.1294)
        surf = ax.plot_surface(
            date_grid,
            mat_grid,
            matrix,
            cmap=cmap,
            rstride=stride,
            cstride=stride,
            linewidth=0.4,
            edgecolor=_MATLAB_EDGE,
            antialiased=False,   # flat shading — no Gouraud smoothing
            shade=False,         # colour driven by Z/cmap only, no lighting
            alpha=1.0,
        )

        # ---- MATLAB-style axis panes (light gray fill, white grid) ----------
        _PANE = (0.925, 0.925, 0.925, 1.0)
        _GRID_WHITE = (1.0, 1.0, 1.0, 1.0)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.fill = True
            axis.pane.set_facecolor(_PANE)
            axis.pane.set_edgecolor((0.8, 0.8, 0.8, 1.0))
            axis._axinfo["grid"]["color"] = _GRID_WHITE
            axis._axinfo["grid"]["linewidth"] = 0.8

        # ---- date tick labels -----------------------------------------------
        n_ticks = min(6, n_dates)
        tick_pos = np.linspace(0, dates_norm[-1], n_ticks)
        tick_idx = np.linspace(0, n_dates - 1, n_ticks, dtype=int)
        tick_labels = [
            dates[i].strftime("%Y-%m") if hasattr(dates[i], "strftime") else str(dates[i])
            for i in tick_idx
        ]
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labels, fontsize=7, rotation=15)

        # ---- axis labels & title -------------------------------------------
        ax.set_xlabel("Date", labelpad=10, fontsize=9)
        ax.set_ylabel("Maturity (months)", labelpad=8, fontsize=9)
        ax.set_zlabel(z_label, labelpad=8, fontsize=9)
        ax.set_title(
            title if title else f"{self.commodity_name}: Forward Curve Surface",
            fontsize=13,
            fontweight="bold",
            pad=15,
        )

        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=12, label=z_label, pad=0.1)

        ax.view_init(elev=elev, azim=azim)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            print(f"Saved: {save_path}")

        plt.show()
        return fig

    # ================================================================ #
    # SPOT PRICE TIME SERIES
    # ================================================================ #

    def plot_spot_price(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot the front-month (maturity index 0) price over time.

        Args:
            save_path: Optional path to save the figure.

        Returns:
            matplotlib Figure.
        """
        if self.primary_curve is None:
            raise ValueError("Load data first with load_curve_data()")

        df = self.primary_curve.df
        spot_maturity = self.primary_curve.maturities[0]
        spot = df[df["maturity"] == spot_maturity].sort_values("date")

        fig, ax = plt.subplots(figsize=self.figsize)

        ax.plot(
            spot["date"],
            spot["price"],
            linewidth=1.5,
            color="#2980b9",
            label=f"M{spot_maturity:.0f} (spot)",
        )

        mean_price = spot["price"].mean()
        ax.axhline(
            mean_price,
            color="red",
            linestyle="--",
            linewidth=1.0,
            alpha=0.7,
            label=f"Mean: {mean_price:.2f}",
        )

        ax.set_xlabel("Date", fontweight="bold")
        ax.set_ylabel("Price", fontweight="bold")
        ax.set_title(
            f"{self.commodity_name}: Spot Price (M{spot_maturity:.0f}) Over Time",
            fontsize=14,
            fontweight="bold",
        )
        ax.xaxis.set_major_locator(mdates.YearLocator(1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(axis="x", rotation=45)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            print(f"✓ Saved: {save_path}")

        plt.show()
        return fig

    # ================================================================ #
    # DIAGNOSTIC SUMMARY
    # ================================================================ #

    def print_curve_diagnostics(self):
        """Print comprehensive curve quality & structure diagnostics."""
        if self.primary_curve is None:
            raise ValueError("Load data first")

        df = self.primary_curve.df

        print(f"\n{'='*75}")
        print(f"  {self.commodity_name.upper()} - FORWARD CURVE DIAGNOSTICS")
        print(f"{'='*75}\n")

        print(f"DATA COVERAGE:")
        print(f"  Total observations:  {len(df)}")
        print(
            f"  Date range:          {df['date'].min().date()} to {df['date'].max().date()}"
        )
        print(f"  Unique dates:        {df['date'].nunique()}")
        print(f"  Unique maturities:   {df['maturity'].nunique()}")
        print(
            f"  Price range:         {df['price'].min():.4f} to {df['price'].max():.4f}"
        )
        print(
            f"  Price avg/std:       {df['price'].mean():.4f} ± {df['price'].std():.4f}"
        )

        # Missing data pattern
        coverage = (df["date"].nunique() * df["maturity"].nunique()) / len(df)
        print(f"\nDATA QUALITY:")
        print(f"  Grid coverage:       {coverage:.1%} (full=100%)")
        print(f"  Avg points/date:     {len(df) / df['date'].nunique():.1f}")
        print(f"  Avg points/maturity: {len(df) / df['maturity'].nunique():.1f}")

        # Curve shape regimes
        unique_dates = sorted(df["date"].unique())
        contango_days = 0
        backwardation_days = 0

        for date in unique_dates:
            curve = df[df["date"] == date].sort_values("maturity")
            if len(curve) >= 2:
                slope = (curve["price"].iloc[-1] - curve["price"].iloc[0]) / len(curve)
                if slope > 0.01:
                    contango_days += 1
                elif slope < -0.01:
                    backwardation_days += 1

        print(f"\nMARKET REGIMES:")
        print(
            f"  Contango days:       {contango_days} ({100*contango_days/len(unique_dates):.1f}%)"
        )
        print(
            f"  Backwardation days:  {backwardation_days} ({100*backwardation_days/len(unique_dates):.1f}%)"
        )

        # Volatility
        returns = df["price"].pct_change().dropna() * 100
        print(f"\nVOLATILITY METRICS:")
        print(f"  Daily returns (μ):   {returns.mean():+.4f}%")
        print(f"  Daily volatility:    {returns.std():.4f}%")
        print(f"  Annualized:          {returns.std() * np.sqrt(252):.2f}%")
        print(f"  Skewness:            {skew(returns):.3f}")
        print(f"  Kurtosis:            {kurtosis(returns):.3f}")

        print(f"\n{'='*75}\n")

    # ================================================================ #
    # FULL PIPELINE
    # ================================================================ #

    def generate_diagnostic_suite(self, output_dir: str = "./curve_analysis/"):
        """Generate complete diagnostic suite for exploration."""
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n{'='*75}")
        print(f"Generating diagnostic suite to {output_dir}")
        print(f"{'='*75}\n")

        # 1. Print diagnostics
        self.print_curve_diagnostics()

        # 2. Spot price
        self.plot_spot_price(save_path=f"{output_dir}01_spot_price.png")

        # 3. Surface
        self.plot_surface_3d_interactive(save_path=f"{output_dir}02_surface_3d.html")

        # 4. Contour
        self.plot_contour_heatmap(save_path=f"{output_dir}03_contour.html")

        # 5. Statistics
        self.plot_curve_statistics_time_series(
            save_path=f"{output_dir}04_curve_stats.png"
        )

        # 6. Maturity spreads
        self.plot_maturity_specific_spreads(
            save_path=f"{output_dir}05_maturity_spreads.png"
        )

        # 7. Volatility surface
        self.plot_volatility_surface(
            window=30, save_path=f"{output_dir}06_volatility_surface.png"
        )

        # 8. Sample grid
        self.plot_curve_sample_grid(
            n_samples=100, seed=42, save_path=f"{output_dir}07_sample_grid.png"
        )

        print(f"\n✓ Complete diagnostic suite saved to {output_dir}\n")
        print(f"Files generated:")
        print(f"  01_spot_price.png               → Front-month spot price over time")
        print(f"  02_surface_3d.html              → Interactive 3D surface")
        print(f"  03_contour.html                 → Contour heatmap")
        print(
            f"  04_curve_stats.png              → Slope, curvature, volatility, term premium"
        )
        print(f"  05_maturity_spreads.png         → Time series per maturity")
        print(f"  06_volatility_surface.png       → Volatility term structure")
        print(f"  07_sample_grid.png              → 100 random forward curve samples")
