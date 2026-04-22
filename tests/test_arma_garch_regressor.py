"""
Tests for ArmaGarchRegressor in curve_factory/arma_garch_regressor.py.

ArmaGarchRegressor fits ARMA(p,q) + GARCH(1,1) on whatever series is passed
to fit() — no log-return transformation is applied internally.  Callers are
responsible for any pre-processing (e.g. np.diff(np.log(prices))).

simulate(T, seed) returns a (T,) array in the same domain as the fitted series.

The primary fixture loads a series from a plain-text file on disk.
By default it uses  tests/data/demand.txt  (one value per line, header lines
starting with '#' are ignored).

To run against your own series pass the file path via the --series-path option:

    pytest tests/test_arma_garch_regressor.py --series-path /path/to/series.txt
"""

from pathlib import Path

import numpy as np
import pytest

from curve_factory.arma_garch_regressor import ArmaGarchRegressor

# ---------------------------------------------------------------------------
# CLI option so the user can point at their own persisted series
# ---------------------------------------------------------------------------

DEFAULT_SERIES_PATH = Path(__file__).parent / "data" / "demand.txt"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def series_path(request) -> Path:
    path = Path(request.config.getoption("--series-path"))
    if not path.exists():
        pytest.skip(f"Series file not found: {path}")
    return path


@pytest.fixture(scope="module")
def price_series(series_path) -> np.ndarray:
    """Load the series from the text file on disk (comments skipped)."""
    prices = np.loadtxt(series_path, comments="#")
    assert prices.ndim == 1, "Series file must contain a 1-D array (one value per line)"
    assert (
        len(prices) >= 50
    ), "Series must have at least 50 observations to fit ARMA-GARCH"
    return prices


@pytest.fixture(scope="module")
def input_series(price_series) -> np.ndarray:
    """Pre-process prices to log-returns before passing to the regressor."""
    # return np.diff(np.log(price_series))
    return price_series


@pytest.fixture(scope="module")
def fitted_model(input_series) -> ArmaGarchRegressor:
    """A model fitted on log-returns — shared across tests in this module."""
    model = ArmaGarchRegressor(ar_order=1, ma_order=1)
    model.fit(input_series)
    return model


# ---------------------------------------------------------------------------
# Test: loading and fitting
# ---------------------------------------------------------------------------


class TestFitFromDisk:
    """Verify that the model can be loaded from disk and fitted."""

    def test_series_loads_from_file(self, price_series):
        assert isinstance(price_series, np.ndarray)
        assert price_series.ndim == 1
        assert len(price_series) > 0

    def test_fit_returns_self(self, input_series):
        model = ArmaGarchRegressor()
        result = model.fit(input_series)
        assert result is model

    def test_fit_sets_arma_result(self, fitted_model):
        assert fitted_model._arma_result is not None

    def test_fit_sets_garch_result(self, fitted_model):
        assert fitted_model._garch_result is not None

    def test_mean_is_finite(self, fitted_model):
        assert np.isfinite(fitted_model._mean_return)

    def test_fit_on_raw_levels_also_works(self, price_series):
        """fit() accepts any numeric series — callers choose the domain."""
        model = ArmaGarchRegressor()
        model.fit(price_series)
        assert model._arma_result is not None
        assert model._garch_result is not None


# ---------------------------------------------------------------------------
# Test: ARMA parameter validity
# ---------------------------------------------------------------------------


class TestArmaParams:

    def test_params_are_finite(self, fitted_model):
        ar, ma, mu = fitted_model._extract_arma_params()
        assert np.all(np.isfinite(ar)), "AR params contain non-finite values"
        assert np.all(np.isfinite(ma)), "MA params contain non-finite values"
        assert np.isfinite(mu), "Intercept is not finite"

    def test_ar_param_count(self, fitted_model):
        ar, _, _ = fitted_model._extract_arma_params()
        assert len(ar) == fitted_model.ar_order

    def test_ma_param_count(self, fitted_model):
        _, ma, _ = fitted_model._extract_arma_params()
        assert len(ma) == fitted_model.ma_order


# ---------------------------------------------------------------------------
# Test: GARCH parameter validity
# ---------------------------------------------------------------------------


class TestGarchParams:

    def test_params_are_positive(self, fitted_model):
        omega, alpha, beta = fitted_model._extract_garch_params()
        assert omega > 0, f"omega must be positive, got {omega}"
        assert alpha >= 0, f"alpha must be non-negative, got {alpha}"
        assert beta >= 0, f"beta must be non-negative, got {beta}"

    def test_stationarity_condition(self, fitted_model):
        """GARCH(1,1) is covariance-stationary iff alpha + beta < 1."""
        _, alpha, beta = fitted_model._extract_garch_params()
        assert (
            alpha + beta < 1.0
        ), f"GARCH stationarity violated: alpha ({alpha:.4f}) + beta ({beta:.4f}) >= 1"


# ---------------------------------------------------------------------------
# Test: simulate() output
# ---------------------------------------------------------------------------


class TestSimulate:

    SIM_STEPS = 52

    def test_output_shape(self, fitted_model):
        sim = fitted_model.simulate(T=self.SIM_STEPS, seed=0)
        assert sim.shape == (
            self.SIM_STEPS,
        ), f"Expected ({self.SIM_STEPS},), got {sim.shape}"

    def test_all_values_finite(self, fitted_model):
        sim = fitted_model.simulate(T=self.SIM_STEPS, seed=0)
        assert np.all(np.isfinite(sim)), "Simulated path contains NaN or Inf"

    def test_reproducibility(self, fitted_model):
        sim1 = fitted_model.simulate(T=self.SIM_STEPS, seed=7)
        sim2 = fitted_model.simulate(T=self.SIM_STEPS, seed=7)
        np.testing.assert_array_equal(
            sim1, sim2, err_msg="Same seed must produce identical paths"
        )

    def test_different_seeds_differ(self, fitted_model):
        sim1 = fitted_model.simulate(T=self.SIM_STEPS, seed=1)
        sim2 = fitted_model.simulate(T=self.SIM_STEPS, seed=2)
        assert not np.allclose(
            sim1, sim2
        ), "Different seeds should produce different paths"

    def test_mean_close_to_fitted_mean(self, fitted_model):
        """Over a long simulation the sample mean should be close to _mean_return."""
        sim = fitted_model.simulate(T=5000, seed=0)
        np.testing.assert_allclose(
            sim.mean(),
            fitted_model._mean_return,
            atol=0.05,
            err_msg="Long-run simulated mean should approximate the fitted mean",
        )


# ---------------------------------------------------------------------------
# Test: error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:

    def test_simulate_before_fit_raises(self):
        model = ArmaGarchRegressor()
        with pytest.raises(RuntimeError, match="fit"):
            model.simulate(T=10)


# ---------------------------------------------------------------------------
# Test: summary
# ---------------------------------------------------------------------------


class TestSummary:

    def test_summary_before_fit(self):
        model = ArmaGarchRegressor()
        assert "not fitted" in model.summary().lower()

    def test_summary_after_fit_contains_arma(self, fitted_model):
        s = fitted_model.summary()
        assert "ARMA" in s

    def test_summary_after_fit_contains_garch(self, fitted_model):
        s = fitted_model.summary()
        assert "GARCH" in s
