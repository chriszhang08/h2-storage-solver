"""
ArmaGarchRegressor
==================
Fits an ARMA(p,q) model on a univariate time series and a GARCH(1,1) model on
the residuals.  The model operates on whatever series is passed in — no
log-return transformation is applied internally.  If the caller wants to fit on
log-returns, they should compute np.diff(np.log(series)) before calling fit().

Theory
------
ARMA(p,q):
    y_t = μ + Σ_{i=1}^{p} φ_i y_{t-i} + ε_t + Σ_{j=1}^{q} θ_j ε_{t-j}
    ε_t = σ_t z_t,  z_t ~ N(0,1)

GARCH(1,1) on the residuals ε_t:
    σ_t² = ω + α ε_{t-1}² + β σ_{t-1}²

Simulation:
    1. Draw z_t ~ N(0,1).
    2. Propagate σ_t² using the GARCH recursion.
    3. Compute ε_t = σ_t z_t.
    4. Propagate y_t using the ARMA recursion.
"""

from typing import Optional, Tuple

import numpy as np
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA

from constants import EPS


class ArmaGarchRegressor:
    """
    Univariate ARMA(p,q) + GARCH(1,1) model for simulation and scenario generation.

    Parameters
    ----------
    ar_order : int
        Autoregressive order p for the ARMA mean model (default 1).
    ma_order : int
        Moving-average order q for the ARMA mean model (default 1).

    Attributes (set after fit)
    --------------------------
    _arma_result   : fitted statsmodels ARIMA result (mean model)
    _garch_result  : fitted arch ARCHModelResult (variance model on ARMA residuals)
    _mean_return   : float, unconditional mean of the fitted series
    _resid_std     : float, long-run std of GARCH residuals (used as fallback)
    """

    def __init__(self, ar_order: int = 1, ma_order: int = 1):
        self.ar_order = ar_order
        self.ma_order = ma_order

        self._arma_result = None
        self._garch_result = None
        self._mean_return: float = 0.0
        self._resid_std: float = 1.0

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, series: np.ndarray) -> "ArmaGarchRegressor":
        """
        Fit ARMA(p,q) + GARCH(1,1) on the given series.

        No transformation is applied; pass the series exactly as you want it
        modelled.  To fit on log-returns, compute np.diff(np.log(prices))
        before calling this method.

        Parameters
        ----------
        series : np.ndarray of shape (T,)

        Returns
        -------
        self
        """
        self._mean_return = float(series.mean())

        # --- 1. Fit ARMA mean model ---
        arma = ARIMA(series, order=(self.ar_order, 0, self.ma_order))
        self._arma_result = arma.fit()

        # --- 2. Fit GARCH(1,1) on ARMA residuals ---
        resid = self._arma_result.resid
        self._resid_std = float(resid.std())

        garch = arch_model(resid, vol="GARCH", p=1, q=1, dist="normal", rescale=False)
        self._garch_result = garch.fit(disp="off")

        return self

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate(
        self,
        T: int,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Simulate T steps from the fitted ARMA-GARCH model.

        The output is in the same domain as the series passed to fit() — no
        level conversion is applied.  To recover a price path from log-return
        simulations, apply x0 * np.exp(np.cumsum(result)) yourself.

        Parameters
        ----------
        T    : int
            Number of steps to simulate forward.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        np.ndarray of shape (T,)
        """
        if self._arma_result is None or self._garch_result is None:
            raise RuntimeError("Call fit() before simulate().")

        rng = np.random.default_rng(seed)

        ar_params, ma_params, mu = self._extract_arma_params()
        omega, alpha, beta = self._extract_garch_params()

        # Initialise ARMA lag buffers with unconditional mean
        past_values = np.full(max(self.ar_order, 1), self._mean_return)
        past_resid = np.zeros(max(self.ma_order, 1))

        # Initialise GARCH variance at unconditional long-run variance
        sigma2_uncond = omega / max(1.0 - alpha - beta, EPS)
        sigma2 = sigma2_uncond

        sim = np.empty(T)
        for t in range(T):
            # GARCH variance update
            sigma2 = omega + alpha * past_resid[-1] ** 2 + beta * sigma2
            sigma2 = max(sigma2, EPS)  # numerical floor

            # Draw standardised innovation
            z = rng.standard_normal()
            eps = np.sqrt(sigma2) * z

            # ARMA mean update
            y_mean = mu
            for i, phi in enumerate(ar_params):
                y_mean += phi * past_values[-(i + 1)]
            for j, theta in enumerate(ma_params):
                y_mean += theta * past_resid[-(j + 1)]

            y_t = y_mean + eps
            sim[t] = y_t

            # Roll lag buffers
            past_values = np.roll(past_values, -1)
            past_values[-1] = y_t
            past_resid = np.roll(past_resid, -1)
            past_resid[-1] = eps

        return sim

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_arma_params(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Return (ar_params, ma_params, intercept) from the fitted ARMA result."""
        params = self._arma_result.params
        names = self._arma_result.param_names
        lookup = {n: float(params[i]) for i, n in enumerate(names)}

        mu = lookup.get("const", 0.0)
        ar_params = np.array(
            [lookup.get(f"ar.L{i}", 0.0) for i in range(1, self.ar_order + 1)]
        )
        ma_params = np.array(
            [lookup.get(f"ma.L{i}", 0.0) for i in range(1, self.ma_order + 1)]
        )
        return ar_params, ma_params, mu

    def _extract_garch_params(self) -> Tuple[float, float, float]:
        """Return (omega, alpha[1], beta[1]) from the fitted GARCH result."""
        p = self._garch_result.params
        omega = float(p.get("omega", 1e-6))
        alpha = float(p.get("alpha[1]", 0.05))
        beta = float(p.get("beta[1]", 0.90))
        return omega, alpha, beta

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a combined summary string for both fitted models."""
        if self._arma_result is None:
            return "Model not fitted yet."
        arma_summary = str(self._arma_result.summary())
        garch_summary = str(self._garch_result.summary())
        return f"=== ARMA({self.ar_order},{self.ma_order}) ===\n{arma_summary}\n\n=== GARCH(1,1) ===\n{garch_summary}"
