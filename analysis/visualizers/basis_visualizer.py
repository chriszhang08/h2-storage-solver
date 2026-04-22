import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime as dt
from sklearn.metrics import r2_score, mean_squared_error
from typing import Optional, Tuple, Union

from curve_factory import CurveRegressorFactory


class BasisVisualizer:
    """Visualize the fit quality of basis function models to curve data."""

    def __init__(self, curve_regressor: CurveRegressorFactory):
        """
        Initialize the BasisVisualizer.

        Parameters:
            curve_regressor: Fitted HistoricalCurveRegressor instance with:
                - curve_data: DataFrame ['date','maturity','price']
                - unique_dates: list/array of unique dates (aligned to coefficients)
                - maturities: sorted 1D array of maturities
                - basis_matrix: (n_maturities, n_basis)
                - coefficients: (n_dates, n_basis)
                - fit_r2: 1D array of R² per date
                - fit_rmse: 1D array of RMSE per date
                - fit_mse: 1D array of MSE per date (optional, recommended)
        """
        self.curve_regressor = curve_regressor
        self.save_path = "./analysis"  # use relative path by default

    # -------------------------------------------------------------------------
    # Fit diagnostics: now includes tail metrics and L^p penalization
    # -------------------------------------------------------------------------
    def plot_fit_diagnostics(
        self, p_penalty: int = 4, save_path: Optional[str] = None
    ) -> None:
        """
        Plot fit quality diagnostics, including:
        - R² distribution
        - RMSE distribution
        - Best / worst fit curves
        - Penalized RMSE and tail RMSE metrics

        Args:
            p_penalty: Power for L^p penalized RMSE aggregation (p>2 penalizes large errors more)
            save_path: Optional path (without filename) to save the figure
        """
        cr = self.curve_regressor
        unique_dates = cr.unique_dates
        maturities = cr.maturities

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. R² distribution
        axes[0, 0].hist(
            cr.fit_r2, bins=20, alpha=0.7, color="steelblue", edgecolor="black"
        )
        mean_r2 = np.mean(cr.fit_r2)
        axes[0, 0].axvline(mean_r2, color="red", lw=2, label=f"Mean: {mean_r2:.3f}")
        axes[0, 0].set_xlabel("R²")
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 0].set_title("R² Distribution")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. RMSE distribution + tail statistics
        axes[0, 1].hist(
            cr.fit_rmse, bins=20, alpha=0.7, color="coral", edgecolor="black"
        )
        mean_rmse = np.mean(cr.fit_rmse)
        rmse_p95 = np.percentile(cr.fit_rmse, 95)
        rmse_p99 = np.percentile(cr.fit_rmse, 99)

        penalized_rmse = (np.mean(cr.fit_rmse**p_penalty)) ** (1.0 / p_penalty)

        axes[0, 1].axvline(mean_rmse, color="red", lw=2, label=f"Mean: {mean_rmse:.3f}")
        axes[0, 1].axvline(
            rmse_p95,
            color="purple",
            lw=1.5,
            linestyle="--",
            label=f"95th pct: {rmse_p95:.3f}",
        )
        axes[0, 1].set_xlabel("RMSE")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].set_title(
            f"RMSE Distribution (L^{p_penalty} penalized={penalized_rmse:.3f})"
        )
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Best vs worst fit curves (by R²)
        best_idx = int(np.argmin(cr.penalized_rmse))
        worst_idx = int(np.argmax(cr.penalized_rmse))

        # Best fit
        ax_best = axes[1, 0]
        best_date = unique_dates[best_idx]
        best_prices = cr.curve_data[best_idx, :]
        best_fitted = cr.basis_matrix @ cr.coefficients[best_idx]
        ax_best.plot(maturities, best_prices, "o-", label="Observed", color="green")
        ax_best.plot(
            maturities,
            best_fitted,
            "--",
            label=f"Fitted (R²={cr.fit_r2[best_idx]:.3f}, RMSE={cr.fit_rmse[best_idx]:.3f})",
            color="darkgreen",
        )
        ax_best.set_title(f"Best Fit: {best_date}")
        ax_best.legend()
        ax_best.grid(True, alpha=0.3)

        # Worst fit
        ax_worst = axes[1, 1]
        worst_date = unique_dates[worst_idx]
        worst_prices = cr.curve_data[worst_idx, :]
        worst_fitted = cr.basis_matrix @ cr.coefficients[worst_idx]
        ax_worst.plot(maturities, worst_prices, "o-", label="Observed", color="red")
        ax_worst.plot(
            maturities,
            worst_fitted,
            "--",
            label=f"Fitted (R²={cr.fit_r2[worst_idx]:.3f}, RMSE={cr.fit_rmse[worst_idx]:.3f})",
            color="darkred",
        )
        ax_worst.set_title(f"Worst Fit: {worst_date}")
        ax_worst.legend()
        ax_worst.grid(True, alpha=0.3)

        plt.tight_layout()

        out_path = "basis_fit_diagnostics.png"
        if save_path:
            out_path = f"{save_path.rstrip('/')}/{out_path}"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.show()
        print(f"✓ Saved diagnostics plot: {out_path}")
        print(f"Mean RMSE       : {mean_rmse:.4f}")
        print(f"95th pct RMSE   : {rmse_p95:.4f}")
        print(f"99th pct RMSE   : {rmse_p99:.4f}")
        print(f"L^{p_penalty} penalized RMSE: {penalized_rmse:.4f}")

    def plot_worst_curves_by_rmse(
        self,
        percentile: float = 95.0,
        max_plots: int = 12,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Visualize all curves in the bottom (100 - percentile) tail of RMSE,
        i.e. curves with RMSE >= given percentile threshold.

        Args:
            percentile: Percentile threshold (e.g. 95 → worst 5% of curves).
            max_plots: Maximum number of curves to plot (for readability).
            save_path: Optional directory to save the figure(s).
        """
        cr = self.curve_regressor
        rmse = cr.fit_rmse
        dates = cr.unique_dates
        maturities = cr.maturities

        # Compute threshold and select worst indices
        thresh = np.percentile(rmse, percentile)
        worst_indices = np.where(rmse >= thresh)[0]

        if len(worst_indices) == 0:
            print(f"No curves found with RMSE >= {thresh:.4f}")
            return

        # Optionally cap number of plots
        if len(worst_indices) > max_plots:
            print(
                f"Found {len(worst_indices)} curves >= {percentile}th pct RMSE "
                f"({thresh:.4f}). Plotting first {max_plots}."
            )
            worst_indices = worst_indices[:max_plots]
        else:
            print(
                f"Plotting all {len(worst_indices)} curves with RMSE >= {thresh:.4f}."
            )

        n_worst = len(worst_indices)
        n_cols = 3
        n_rows = int(np.ceil(n_worst / n_cols))

        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False
        )

        for ax, idx in zip(axes.flatten(), worst_indices):
            date = dates[idx]
            rmse_i = rmse[idx]
            r2_i = cr.fit_r2[idx]

            # Observed prices for this date
            curve_prices = cr.curve_data[idx, :]

            # Fitted curve
            fitted_prices = cr.basis_matrix @ cr.coefficients[idx]

            ax.plot(
                maturities,
                curve_prices,
                "o-",
                label="Observed",
                color="steelblue",
                linewidth=2,
                markersize=5,
            )
            ax.plot(
                maturities,
                fitted_prices,
                "--",
                label="Fitted",
                color="darkorange",
                linewidth=2,
            )

            ax.fill_between(
                maturities, curve_prices, fitted_prices, alpha=0.2, color="red"
            )

            ax.set_title(f"{date} | RMSE={rmse_i:.3f}, R²={r2_i:.3f}", fontsize=9)
            ax.set_xlabel("Maturity")
            ax.set_ylabel("Price")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7)

        # Turn off unused axes
        for ax in axes.flatten()[n_worst:]:
            ax.axis("off")

        plt.tight_layout()

        if save_path:
            out_path = (
                f"{save_path.rstrip('/')}/worst_curves_rmse_{int(percentile)}pct.png"
            )
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            print(f"✓ Saved worst-curve RMSE plot to {out_path}")

        plt.show()

    # -------------------------------------------------------------------------
    # Single-date diagnostics:
    # -------------------------------------------------------------------------
    def plot_coefficients_vs_observed(
        self,
        target_date: Union[pd.Timestamp, dt.datetime],
        save_path: Optional[str] = None,
        figsize: Tuple[float, float] = (12, 8),
    ) -> plt.Figure:
        """
        Plot basis coefficients vs observed data for a specific date.

        Args:
            target_date: Date to plot (must have been fitted)
            save_path: Optional save path (full file path)
            figsize: Figure size tuple

        Returns:
            matplotlib Figure object
        """
        cr = self.curve_regressor
        result = cr.reconstruct_maturity_for_date(target_date)

        coefs = cr.coefficients[cr.date_to_idx[target_date]]
        r2_score = result["r2_score"]
        rmse_score = result["rmse_score"]
        curve_prices = result["observed_prices"]
        fitted_prices = result["fitted_prices"]

        # Plot: curve + residuals + coefficients
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[2, 1])

        # Main plot: Observed vs Fitted
        ax1.plot(
            cr.maturities,
            curve_prices,
            "o-",
            label="Observed",
            color="steelblue",
            linewidth=2,
            markersize=6,
        )
        ax1.plot(
            cr.maturities,
            fitted_prices,
            "--",
            label=f"Fitted (R²={r2_score:.4f}, RMSE={rmse_score:.4f})",
            color="darkorange",
            linewidth=2,
        )
        ax1.fill_between(
            cr.maturities,
            curve_prices,
            fitted_prices,
            alpha=0.2,
            color="red",
            label="Residuals",
        )
        ax1.set_xlabel("Maturity")
        ax1.set_ylabel("Price")
        ax1.set_title(
            f'Curve Fit: {target_date.strftime("%Y-%m-%d")} '
            f"(R²={r2_score:.4f}, RMSE={rmse_score:.4f})"
        )
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Residuals
        residuals = curve_prices - fitted_prices
        ax2.plot(cr.maturities, residuals, "o-", color="darkred", markersize=4)
        ax2.axhline(0, color="black", linestyle="--", alpha=0.5)
        ax2.set_xlabel("Maturity")
        ax2.set_ylabel("Residuals")
        ax2.set_title("Residuals (Observed - Fitted)")
        ax2.grid(True, alpha=0.3)

        # Coefficients inset
        ax_coef = ax1.inset_axes([0.65, 0.7, 0.3, 0.25])
        colors = plt.cm.viridis(np.linspace(0, 1, len(coefs)))
        ax_coef.bar(range(len(coefs)), coefs, color=colors, alpha=0.8)
        ax_coef.set_title("Basis Coefficients", fontsize=10)
        ax_coef.set_xlabel("Index")
        ax_coef.tick_params(labelsize=8)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"✓ Saved plot: {save_path}")

        plt.show()
        return fig

    def plot_coefficients_vs_observed_for_dates(self, dates):
        """
        For each date in the list, plot observed vs fitted curve. If the date is not found,
        increment by one day until a valid date is found and plot for that date instead.
        """
        import datetime as dt

        for date in dates:
            original_date = date
            while True:
                try:
                    self.plot_coefficients_vs_observed(date)
                    if date != original_date:
                        print(
                            f"Plotted for {date.strftime('%Y-%m-%d')} instead of {original_date.strftime('%Y-%m-%d')}"
                        )
                    break
                except ValueError as e:
                    err_msg = str(e)
                    if (
                        err_msg
                        == f"Date {date.strftime('%Y-%m-%d')} 00:00:00 not found in fitted data."
                    ):
                        date += dt.timedelta(days=1)
                        continue
                    else:
                        print(
                            f"Error for {original_date.strftime('%Y-%m-%d')}: {err_msg}"
                        )
                        break

    # -------------------------------------------------------------------------
    # Basis function visualization
    # -------------------------------------------------------------------------
    def plot_basis_functions(self, save_path: Optional[str] = None) -> None:
        """Visualize all basis functions over maturities.

        For hybrid basis: decomposes into polynomial, Fourier, and B-spline groups,
        showing both per-group overview panels and individual function detail panels.
        For all other basis types: shows every basis function in a dynamic grid.
        """
        cr = self.curve_regressor
        basis_matrix = cr.basis_matrix
        maturities = cr.maturities
        n_basis = basis_matrix.shape[1]

        if cr._basis_type == "hybrid":
            self._plot_hybrid_basis_functions(
                basis_matrix, maturities, n_basis, save_path
            )
            return

        # General case: all basis functions in a dynamic grid
        n_cols = 4
        n_rows = int(np.ceil(n_basis / n_cols))

        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False
        )

        for i in range(n_basis):
            ax = axes[i // n_cols, i % n_cols]
            ax.plot(maturities, basis_matrix[:, i], linewidth=2, color="steelblue")
            ax.fill_between(
                maturities, basis_matrix[:, i], alpha=0.3, color="steelblue"
            )
            ax.set_xlabel("Maturity")
            ax.set_ylabel("Basis value")
            ax.set_title(f"Basis Function {i}")
            ax.grid(True, alpha=0.3)

        for i in range(n_basis, n_rows * n_cols):
            axes[i // n_cols, i % n_cols].axis("off")

        plt.tight_layout()
        if save_path:
            out_path = f"{save_path.rstrip('/')}/basis_functions.png"
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            print(f"✓ Saved basis functions to {out_path}")
        plt.show()

    def _plot_hybrid_basis_functions(
        self,
        basis_matrix: np.ndarray,
        maturities: np.ndarray,
        n_basis: int,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Decomposed visualization for hybrid basis (polynomial + Fourier + B-spline).

        Layout:
            Row 1 (overview): Polynomial group | Fourier group | B-Spline group | Combined
            Rows 2+: All individual basis functions, color-coded by component type.

        Split constants must match HistoricalCurveRegressor._hybrid_basis:
            poly    → first 3 columns   (degree=3)
            fourier → next 8 columns    (n_fourier=8)
            bspline → remaining columns (len(knots) + degree - 1)
        """
        N_POLY = 3
        N_FOURIER = 8
        n_bspline = n_basis - N_POLY - N_FOURIER

        poly_part = basis_matrix[:, :N_POLY]
        fourier_part = basis_matrix[:, N_POLY : N_POLY + N_FOURIER]
        bspline_part = basis_matrix[:, N_POLY + N_FOURIER :]

        n_cols = 4
        n_detail_rows = int(np.ceil(n_basis / n_cols))
        n_total_rows = 1 + n_detail_rows

        fig = plt.figure(figsize=(5 * n_cols, 4 * n_total_rows))

        # ---- Overview row ----
        ax_poly = fig.add_subplot(n_total_rows, n_cols, 1)
        ax_fourier = fig.add_subplot(n_total_rows, n_cols, 2)
        ax_bspline = fig.add_subplot(n_total_rows, n_cols, 3)
        ax_combined = fig.add_subplot(n_total_rows, n_cols, 4)

        colors_poly = plt.cm.Blues(np.linspace(0.45, 0.9, N_POLY))
        colors_fourier = plt.cm.Oranges(np.linspace(0.45, 0.9, N_FOURIER))
        colors_bspline = plt.cm.Greens(np.linspace(0.45, 0.9, max(n_bspline, 1)))

        for i in range(N_POLY):
            ax_poly.plot(
                maturities, poly_part[:, i], color=colors_poly[i], lw=1.5, label=f"P{i}"
            )
        ax_poly.set_title("Polynomial Basis")
        ax_poly.set_xlabel("Maturity")
        ax_poly.set_ylabel("Basis value")
        ax_poly.legend(fontsize=8)
        ax_poly.grid(True, alpha=0.3)

        for i in range(N_FOURIER):
            ax_fourier.plot(
                maturities,
                fourier_part[:, i],
                color=colors_fourier[i],
                lw=1.5,
                label=f"F{i}",
            )
        ax_fourier.set_title("Fourier Basis")
        ax_fourier.set_xlabel("Maturity")
        ax_fourier.set_ylabel("Basis value")
        ax_fourier.legend(fontsize=7, ncol=2)
        ax_fourier.grid(True, alpha=0.3)

        for i in range(n_bspline):
            ax_bspline.plot(
                maturities,
                bspline_part[:, i],
                color=colors_bspline[i],
                lw=1.5,
                label=f"S{i}",
            )
        ax_bspline.set_title(f"B-Spline Basis ({n_bspline} funcs)")
        ax_bspline.set_xlabel("Maturity")
        ax_bspline.set_ylabel("Basis value")
        ax_bspline.legend(fontsize=7, ncol=2)
        ax_bspline.grid(True, alpha=0.3)

        # Combined panel: all groups overlaid with group colors
        for i in range(N_POLY):
            ax_combined.plot(
                maturities,
                poly_part[:, i],
                color="steelblue",
                lw=1.2,
                alpha=0.8,
                label="Polynomial" if i == 0 else "",
            )
        for i in range(N_FOURIER):
            ax_combined.plot(
                maturities,
                fourier_part[:, i],
                color="darkorange",
                lw=1.2,
                alpha=0.8,
                label="Fourier" if i == 0 else "",
            )
        for i in range(n_bspline):
            ax_combined.plot(
                maturities,
                bspline_part[:, i],
                color="forestgreen",
                lw=1.2,
                alpha=0.8,
                label="B-Spline" if i == 0 else "",
            )
        ax_combined.set_title(f"Combined ({n_basis} basis functions)")
        ax_combined.set_xlabel("Maturity")
        ax_combined.set_ylabel("Basis value")
        ax_combined.legend(fontsize=8)
        ax_combined.grid(True, alpha=0.3)

        # ---- Individual function detail rows ----
        group_colors = (
            ["steelblue"] * N_POLY
            + ["darkorange"] * N_FOURIER
            + ["forestgreen"] * n_bspline
        )
        group_labels = (
            [f"Poly {i}" for i in range(N_POLY)]
            + [f"Fourier {i}" for i in range(N_FOURIER)]
            + [f"Spline {i}" for i in range(n_bspline)]
        )

        for i in range(n_basis):
            ax = fig.add_subplot(n_total_rows, n_cols, n_cols + 1 + i)
            ax.plot(maturities, basis_matrix[:, i], linewidth=2, color=group_colors[i])
            ax.fill_between(
                maturities, basis_matrix[:, i], alpha=0.3, color=group_colors[i]
            )
            ax.set_title(group_labels[i], fontsize=9)
            ax.set_xlabel("Maturity", fontsize=8)
            ax.set_ylabel("Basis value", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)

        plt.suptitle(
            "Hybrid Basis: Polynomial + Fourier + B-Spline",
            fontsize=13,
            fontweight="bold",
            y=1.005,
        )
        plt.tight_layout()

        if save_path:
            out_path = f"{save_path.rstrip('/')}/basis_functions_hybrid.png"
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            print(f"✓ Saved hybrid basis functions to {out_path}")
        plt.show()

    # -------------------------------------------------------------------------
    # Hybrid component decomposition
    # -------------------------------------------------------------------------
    def plot_hybrid_curve_decomposition(
        self,
        target_date: Union[pd.Timestamp, dt.datetime],
        save_path: Optional[str] = None,
        figsize: Tuple[float, float] = (18, 10),
    ) -> plt.Figure:
        """
        Decompose a hybrid basis reconstruction into polynomial, Fourier, and B-spline
        contributions for a specified date.

        Only valid for hybrid basis regressors. Shows:
            Row 1: full observed vs reconstructed | component contributions overlaid | full residuals
            Row 2: polynomial-only vs observed | Fourier-only vs observed | B-spline-only vs observed

        The per-component panels isolate each sub-basis contribution so the fraction of curve
        shape explained by trend (polynomial), seasonality (Fourier), and local variation
        (B-spline) can be assessed visually.

        Args:
            target_date: Date to decompose (must be in fitted data).
            save_path: Optional full file path to save the figure.
            figsize: Figure size tuple.

        Returns:
            matplotlib Figure object.

        Raises:
            ValueError: If the regressor is not hybrid, or the date is not found.
        """
        cr = self.curve_regressor

        if cr._basis_type != "hybrid":
            raise ValueError(
                f"plot_hybrid_curve_decomposition requires a hybrid basis regressor, "
                f"got '{cr._basis_type}'."
            )

        if target_date not in cr.date_to_idx:
            raise ValueError(f"Date {target_date} not found in fitted data.")

        # --- Decompose basis matrix and coefficients (must match _hybrid_basis) ---
        N_POLY = 3
        N_FOURIER = 8
        basis_matrix = cr.basis_matrix
        n_basis = basis_matrix.shape[1]
        n_bspline = n_basis - N_POLY - N_FOURIER

        poly_part = basis_matrix[:, :N_POLY]
        fourier_part = basis_matrix[:, N_POLY : N_POLY + N_FOURIER]
        bspline_part = basis_matrix[:, N_POLY + N_FOURIER :]

        date_idx = cr.date_to_idx[target_date]
        coefs = cr.coefficients[date_idx]
        poly_coefs = coefs[:N_POLY]
        fourier_coefs = coefs[N_POLY : N_POLY + N_FOURIER]
        bspline_coefs = coefs[N_POLY + N_FOURIER :]

        # Component reconstructions (additive: full = poly + fourier + bspline)
        poly_fit = poly_part @ poly_coefs
        fourier_fit = fourier_part @ fourier_coefs
        bspline_fit = bspline_part @ bspline_coefs
        full_fit = poly_fit + fourier_fit + bspline_fit

        observed = cr.curve_data[date_idx, :]
        maturities = cr.maturities
        r2 = cr.fit_r2[date_idx]
        rmse = cr.fit_rmse[date_idx]

        def _rmse(a, b):
            return float(np.sqrt(np.mean((a - b) ** 2)))

        rmse_poly = _rmse(observed, poly_fit)
        rmse_fourier = _rmse(observed, fourier_fit)
        rmse_bspline = _rmse(observed, bspline_fit)

        date_str = (
            target_date.strftime("%Y-%m-%d")
            if hasattr(target_date, "strftime")
            else str(target_date)
        )

        fig, axes = plt.subplots(2, 3, figsize=figsize)

        # [0, 0] Observed vs full reconstruction
        ax = axes[0, 0]
        ax.plot(maturities, observed, "o-", color="black", lw=2, ms=5, label="Observed")
        ax.plot(
            maturities,
            full_fit,
            "--",
            color="crimson",
            lw=2,
            label=f"Full fit (R²={r2:.3f}, RMSE={rmse:.3f})",
        )
        ax.fill_between(maturities, observed, full_fit, alpha=0.15, color="crimson")
        ax.set_title(f"Full Reconstruction: {date_str}")
        ax.set_xlabel("Maturity")
        ax.set_ylabel("Price")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # [0, 1] All three component curves overlaid
        ax = axes[0, 1]
        ax.plot(
            maturities,
            observed,
            "o-",
            color="black",
            lw=2,
            ms=5,
            label="Observed",
            zorder=5,
        )
        ax.plot(
            maturities, poly_fit, "-", color="steelblue", lw=1.8, label="Polynomial"
        )
        ax.plot(
            maturities, fourier_fit, "-", color="darkorange", lw=1.8, label="Fourier"
        )
        ax.plot(
            maturities, bspline_fit, "-", color="forestgreen", lw=1.8, label="B-Spline"
        )
        ax.set_title("Component Contributions")
        ax.set_xlabel("Maturity")
        ax.set_ylabel("Price")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # [0, 2] Full fit residuals
        ax = axes[0, 2]
        full_residuals = observed - full_fit
        ax.plot(maturities, full_residuals, "o-", color="crimson", lw=2, ms=4)
        ax.axhline(0, color="black", lw=1, linestyle="--", alpha=0.5)
        ax.fill_between(maturities, full_residuals, alpha=0.2, color="crimson")
        ax.set_title(f"Full Fit Residuals (RMSE={rmse:.3f})")
        ax.set_xlabel("Maturity")
        ax.set_ylabel("Observed − Fitted")
        ax.grid(True, alpha=0.3)

        # [1, 0] Polynomial component only
        ax = axes[1, 0]
        ax.plot(
            maturities,
            observed,
            "o-",
            color="black",
            lw=1.5,
            ms=4,
            alpha=0.4,
            label="Observed",
        )
        ax.plot(
            maturities,
            poly_fit,
            "-",
            color="steelblue",
            lw=2,
            label=f"Polynomial (RMSE={rmse_poly:.3f})",
        )
        ax.fill_between(maturities, observed, poly_fit, alpha=0.15, color="steelblue")
        ax.set_title(f"Polynomial Component ({N_POLY} basis functions)")
        ax.set_xlabel("Maturity")
        ax.set_ylabel("Price")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # [1, 1] Fourier component only
        ax = axes[1, 1]
        ax.plot(
            maturities,
            observed,
            "o-",
            color="black",
            lw=1.5,
            ms=4,
            alpha=0.4,
            label="Observed",
        )
        ax.plot(
            maturities,
            fourier_fit,
            "-",
            color="darkorange",
            lw=2,
            label=f"Fourier (RMSE={rmse_fourier:.3f})",
        )
        ax.fill_between(
            maturities, observed, fourier_fit, alpha=0.15, color="darkorange"
        )
        ax.set_title(f"Fourier Component ({N_FOURIER} basis functions)")
        ax.set_xlabel("Maturity")
        ax.set_ylabel("Price")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # [1, 2] B-Spline component only
        ax = axes[1, 2]
        ax.plot(
            maturities,
            observed,
            "o-",
            color="black",
            lw=1.5,
            ms=4,
            alpha=0.4,
            label="Observed",
        )
        ax.plot(
            maturities,
            bspline_fit,
            "-",
            color="forestgreen",
            lw=2,
            label=f"B-Spline (RMSE={rmse_bspline:.3f})",
        )
        ax.fill_between(
            maturities, observed, bspline_fit, alpha=0.15, color="forestgreen"
        )
        ax.set_title(f"B-Spline Component ({n_bspline} basis functions)")
        ax.set_xlabel("Maturity")
        ax.set_ylabel("Price")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.suptitle(
            f"Hybrid Basis Decomposition: {date_str}",
            fontsize=13,
            fontweight="bold",
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"✓ Saved hybrid decomposition plot: {save_path}")

        plt.show()
        return fig

    # -------------------------------------------------------------------------
    # Macro-predicted coefficient decomposition
    # -------------------------------------------------------------------------
    def plot_macro_predicted_decomposition(
        self,
        feature_df: pd.DataFrame,
        n_pick: int = 6,
        save_path: Optional[str] = None,
        figsize_per_col: Tuple[float, float] = (4.5, 9.0),
    ) -> plt.Figure:
        """
        For n_pick representative dates (sampled at evenly-spaced percentiles of
        curve-level R²), plot two rows per date:

            Row 0: Observed vs OLS-fitted vs macro-predicted full curve.
            Row 1: Macro-predicted curve decomposed into polynomial, Fourier,
                   and B-spline components, with observed overlaid.

        Args:
            feature_df: DateTime-indexed DataFrame of normalised features, one
                        column per feature name registered on the regressor.
            n_pick:     Number of representative dates to show (default 6).
            save_path:  Optional file path to save the figure.
            figsize_per_col: (width, height) per column in inches.

        Returns:
            matplotlib Figure.

        Raises:
            ValueError: If the regressor has not been fitted with
                        fit_parameters_to_coefficients.
        """
        cr = self.curve_regressor

        if cr.multi_output_reg is None:
            raise ValueError(
                "fit_parameters_to_coefficients must be called before plotting."
            )
        if cr._basis_factory.basis_type != "hybrid":
            raise ValueError(
                "plot_macro_predicted_decomposition requires a hybrid basis regressor."
            )

        # Hybrid basis split (mirrors _hybrid_basis)
        N_POLY, N_FOURIER = 3, 8
        basis = cr.basis_matrix
        poly_basis = basis[:, :N_POLY]
        fourier_basis = basis[:, N_POLY : N_POLY + N_FOURIER]
        bspline_basis = basis[:, N_POLY + N_FOURIER :]
        mats = cr.maturities

        # Align feature dates with regressor dates
        common_dates = feature_df.index.intersection(cr.unique_dates)
        common_indices = [cr.date_to_idx[d] for d in common_dates]

        observed_curves = cr.curve_data[common_indices, :]
        predicted_curves = cr.predict_curves_given_features(
            feature_df.loc[common_dates]
        ).values  # (n_common, n_maturities)

        # Per-date curve R² and RMSE
        curve_r2 = np.array(
            [
                r2_score(observed_curves[i], predicted_curves[i])
                for i in range(len(common_indices))
            ]
        )
        curve_rmse = np.array(
            [
                np.sqrt(mean_squared_error(observed_curves[i], predicted_curves[i]))
                for i in range(len(common_indices))
            ]
        )

        # Pick n_pick dates at evenly-spaced percentiles of curve R²
        sorted_idx = np.argsort(curve_r2)
        pick_pos = np.linspace(0, len(sorted_idx) - 1, n_pick, dtype=int)
        pick_indices = sorted_idx[pick_pos]

        # Build percentile labels
        percentiles = np.linspace(0, 100, n_pick, dtype=int)
        labels = [
            "Worst" if p == 0 else ("Best" if p == 100 else f"{p}th pct")
            for p in percentiles
        ]

        fig, axes = plt.subplots(
            2, n_pick, figsize=(figsize_per_col[0] * n_pick, figsize_per_col[1])
        )
        fig.suptitle(
            "Macro-Predicted Coefficients Applied to Basis Functions\n"
            "(OLS fit = dashed orange  |  Macro prediction = solid blue  |  Observed = grey dots)",
            fontsize=12,
            y=1.01,
        )

        for col, (pidx, label) in enumerate(zip(pick_indices, labels)):
            raw_idx = common_indices[pidx]
            date_str = common_dates[pidx].strftime("%Y-%m-%d")
            observed = cr.curve_data[raw_idx, :]

            # OLS coefficients (from fit_basis_coefficients)
            ols_coefs = cr.coefficients[raw_idx]
            ols_fit = basis @ ols_coefs

            # Macro-predicted coefficients (unnormalised via predict_coefficients)
            param_dict = feature_df.loc[common_dates[pidx], cr.feature_names].to_dict()
            mac_coefs = cr.predict_coefficients(param_dict)
            mac_fit = basis @ mac_coefs

            poly_component = poly_basis @ mac_coefs[:N_POLY]
            fourier_component = fourier_basis @ mac_coefs[N_POLY : N_POLY + N_FOURIER]
            bspline_component = bspline_basis @ mac_coefs[N_POLY + N_FOURIER :]

            # --- Row 0: full overlay ---
            ax = axes[0, col]
            ax.plot(mats, observed, "o", color="grey", ms=4, label="Observed", zorder=3)
            ax.plot(mats, ols_fit, "--", color="darkorange", lw=1.5, label="OLS fit")
            ax.plot(mats, mac_fit, "-", color="steelblue", lw=2, label="Macro pred")
            ax.set_title(
                f"{label}\n{date_str}\nR²={curve_r2[pidx]:.3f}  RMSE={curve_rmse[pidx]:.3f}",
                fontsize=9,
            )
            ax.set_xlabel("Maturity")
            ax.set_ylabel("Price")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

            # --- Row 1: component decomposition ---
            ax = axes[1, col]
            ax.plot(mats, observed, "o", color="grey", ms=4, label="Observed", zorder=3)
            ax.plot(mats, poly_component, "-", color="tomato", lw=1.5, label="Poly")
            ax.plot(
                mats,
                fourier_component,
                "--",
                color="mediumorchid",
                lw=1.5,
                label="Fourier",
            )
            ax.plot(
                mats,
                bspline_component,
                "-.",
                color="seagreen",
                lw=1.5,
                label="B-spline",
            )
            ax.plot(
                mats, mac_fit, "-", color="steelblue", lw=2, label="Total", alpha=0.6
            )
            ax.set_title("Component decomposition", fontsize=9)
            ax.set_xlabel("Maturity")
            ax.set_ylabel("Price")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"✓ Saved: {save_path}")

        plt.show()
        return fig
