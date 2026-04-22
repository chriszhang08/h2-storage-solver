"""
BasisFactory: Basis function construction and curve fitting for commodity forward curves.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from scipy.interpolate import BSpline

from constants import EPS


@dataclass
class BasisFactoryConfig:
    """Configuration for basis function construction."""

    n_basis: int = 17
    degree: int = 3
    basis_type: str = "hybrid"
    knots: Optional[List[float]] = None


class BasisFactory:
    """
    Constructs basis function matrices over a maturity grid, fits curves to
    basis coefficients, and stores per-curve fit diagnostics.

    Each column of the returned basis matrix is L2-normalized.
    """

    def __init__(self, config: BasisFactoryConfig):
        self.config = config
        self.n_basis = config.n_basis
        self.degree = config.degree
        self.basis_type = config.basis_type
        self.knots = config.knots

        # Set after build_basis
        self.basis_matrix: Optional[np.ndarray] = None

        # Set after fit_coefficients
        self.coefficients: Optional[np.ndarray] = None
        self.fit_r2: Optional[np.ndarray] = None
        self.fit_rmse: Optional[np.ndarray] = None
        self.fit_mse: Optional[np.ndarray] = None
        self.fit_rss: Optional[np.ndarray] = None
        self.penalized_rmse: Optional[np.ndarray] = None

    def build_basis(self, maturities: np.ndarray) -> np.ndarray:
        """
        Build the basis function matrix for the configured basis type and store it.

        Args:
            maturities: 1-D array of maturity values.

        Returns:
            np.ndarray of shape (len(maturities), n_basis_cols), column-normalised.
        """
        n_maturities = len(maturities)

        mat_norm = (maturities - maturities.min()) / (
            maturities.max() - maturities.min() + EPS
        )

        if self.knots is not None:
            knots_norm = np.array(
                [
                    (k - maturities.min()) / (maturities.max() - maturities.min() + EPS)
                    for k in self.knots
                ]
            )
        else:
            knots_norm = None

        if self.basis_type == "bspline":
            self.basis_matrix = self._bspline_basis(mat_norm, n_maturities, knots_norm)
        elif self.basis_type == "fourier":
            self.basis_matrix = self._fourier_basis(mat_norm, n_maturities)
        elif self.basis_type == "polynomial":
            self.basis_matrix = self._polynomial_basis(mat_norm, n_maturities)
        elif self.basis_type == "hybrid":
            self.basis_matrix = self._hybrid_basis(mat_norm, n_maturities, knots_norm)
        else:
            raise ValueError(f"Unknown basis type: {self.basis_type}")

        return self.basis_matrix

    def fit_coefficients(
        self,
        curve_data: np.ndarray,
        maturities: np.ndarray,
    ) -> np.ndarray:
        """
        Build the basis matrix and fit basis coefficients for every curve in the
        price matrix. Stores per-curve diagnostics (R², RMSE, MSE, RSS).

        Args:
            curve_data: Price matrix of shape (n_dates, n_maturities).
            maturities: 1-D array of maturity values.

        Returns:
            Coefficient matrix of shape (n_dates, n_basis).
        """
        self.build_basis(maturities)

        n_dates = curve_data.shape[0]
        coefficients = np.zeros((n_dates, self.n_basis))

        rss_list: List[float] = []
        mse_list: List[float] = []
        rmse_list: List[float] = []
        r2_list: List[float] = []
        penalized_rmse_list: List[float] = []

        for i in range(n_dates):
            curve_prices = curve_data[i, :]

            if np.any(np.isnan(curve_prices)):
                continue

            coefs, _, _, _ = np.linalg.lstsq(
                self.basis_matrix, curve_prices, rcond=None
            )
            coefficients[i, :] = coefs

            fitted = self.basis_matrix @ coefs
            residuals_vec = curve_prices - fitted

            rss = np.sum(residuals_vec**2)
            rss_list.append(rss)

            mse = np.mean(residuals_vec**2)
            mse_list.append(mse)

            rmse = np.sqrt(mse)
            rmse_list.append(rmse)

            p = 4
            penalized_rmse_list.append((np.mean(np.abs(residuals_vec) ** p)) ** (1 / p))

            ss_tot = np.sum((curve_prices - np.mean(curve_prices)) ** 2)
            r2 = 1 - (rss / ss_tot) if ss_tot > 0 else 0
            r2_list.append(r2)

        self.coefficients = coefficients
        self.fit_rss = np.array(rss_list)
        self.fit_mse = np.array(mse_list)
        self.fit_rmse = np.array(rmse_list)
        self.fit_r2 = np.array(r2_list)
        self.penalized_rmse = np.array(penalized_rmse_list)

        n_fitted = len(rss_list)
        print(f"\n{'=' * 60}")
        print(f"BASIS FITTING SUMMARY")
        print(f"{'=' * 60}")
        print(f"Fitted {n_fitted}/{n_dates} curves ({100 * n_fitted / n_dates:.1f}%)")
        print(f"Mean R²:       {self.fit_r2.mean():.4f} ± {self.fit_r2.std():.4f}")
        print(f"Mean RMSE:     {self.fit_rmse.mean():.4f} ± {self.fit_rmse.std():.4f}")
        print(f"Mean MSE:      {self.fit_mse.mean():.4f} ± {self.fit_mse.std():.4f}")
        print(f"95th pct RMSE: {np.percentile(self.fit_rmse, 95):.4f}")
        print(
            f"Best fit:      R²={self.fit_r2.max():.4f}, RMSE={self.fit_rmse.min():.4f}"
        )
        print(
            f"Worst fit:     R²={self.fit_r2.min():.4f}, RMSE={self.fit_rmse.max():.4f}"
        )
        print(f"✓ Basis fitting complete. Shape: {self.coefficients.shape}")

        return self.coefficients

    def reconstruct(self, coefficients: np.ndarray) -> np.ndarray:
        """Reconstruct price curve(s) from basis coefficients."""
        return self.basis_matrix @ coefficients

    # ── Private basis builders ───────────────────────────────────────────

    def _bspline_basis(
        self,
        maturities_norm: np.ndarray,
        n_maturities: int,
        knots_norm: np.ndarray,
        degree: Optional[int] = None,
    ) -> np.ndarray:
        if degree is None:
            degree = self.degree

        n_spline_basis = len(knots_norm) + degree - 1
        t = np.concatenate((np.zeros(degree), knots_norm, np.ones(degree)))
        basis_matrix = np.zeros((n_maturities, n_spline_basis))

        for i in range(n_spline_basis):
            c = np.zeros(n_spline_basis)
            c[i] = 1
            spl = BSpline(t, c, degree)
            basis_matrix[:, i] = spl(maturities_norm)

        return basis_matrix / (np.linalg.norm(basis_matrix, axis=0) + EPS)

    def _fourier_basis(
        self,
        maturities_norm: np.ndarray,
        n_maturities: int,
        n_harmonics: Optional[int] = None,
        n_fourier: Optional[int] = None,
    ) -> np.ndarray:
        if n_fourier is None:
            n_fourier = self.n_basis
        if n_harmonics is None:
            n_harmonics = n_fourier // 2

        basis_matrix = np.zeros((n_maturities, n_fourier))

        for i in range(1, n_fourier + 1):
            if i % 2 == 1:
                k = (i + 1) // 2
                basis_matrix[:, i - 1] = np.sin(
                    2 * np.pi * k * n_harmonics * maturities_norm
                )
            else:
                k = i // 2
                basis_matrix[:, i - 1] = np.cos(
                    2 * np.pi * k * n_harmonics * maturities_norm
                )

        return basis_matrix / (np.linalg.norm(basis_matrix, axis=0) + EPS)

    def _polynomial_basis(
        self,
        maturities_norm: np.ndarray,
        n_maturities: int,
        degree: Optional[int] = None,
    ) -> np.ndarray:
        if degree is None:
            degree = self.degree

        basis_matrix = np.zeros((n_maturities, degree))
        for i in range(degree):
            basis_matrix[:, i] = maturities_norm**i

        return basis_matrix / (np.linalg.norm(basis_matrix, axis=0) + EPS)

    def _hybrid_basis(
        self,
        maturities_norm: np.ndarray,
        n_maturities: int,
        knots_norm: np.ndarray,
    ) -> np.ndarray:
        poly_basis = self._polynomial_basis(maturities_norm, n_maturities, degree=3)
        fourier_basis = self._fourier_basis(
            maturities_norm, n_maturities, n_harmonics=3, n_fourier=8
        )
        bspline_basis = self._bspline_basis(
            maturities_norm, n_maturities, knots_norm, degree=3
        )
        combined = np.hstack([poly_basis, fourier_basis, bspline_basis])
        return combined / (np.linalg.norm(combined, axis=0) + EPS)
