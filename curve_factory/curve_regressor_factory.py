"""
CommodityCurveRegressor: Regressor Object to Generate Curves from Commodity Parameters
================================================================================
THEORY:
-------
A commodity forward curve F(t,T) represents the future price agreed today
for delivery at maturity T, as seen at time t.

We decompose the curve into basis functions:
    F(t, T) = Σᵢ αᵢ(t) φᵢ(T)

where:
    - αᵢ(t) are time-varying basis coefficients
    - φᵢ(T) are basis functions (splines, Fourier, etc.)

Then regress the basis coefficients on commodity parameters:
    αᵢ(t) = β₀ᵢ + Σⱼ βⱼᵢ Xⱼ(t) + εᵢ(t)

where Xⱼ(t) are commodity parameters like:
    - Slope (short vs long maturity)
    - Volatility (market uncertainty)
    - Curvature (midpoint second derivative)
    - Seasonal factors

This allows us to:
1. Represent curves compactly (10-15 basis functions vs 30+ maturities)
2. Understand curve sensitivity to each parameter
3. Forecast curves from parameter projections
"""

import pandas as pd
import numpy as np
import datetime as dt
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from typing import Dict, List, Optional, Union

from curve_factory import ArmaGarchRegressor, BasisFactory, BasisFactoryConfig
from curve_factory.utils.curve_data_transformers import to_datetime
from curve_factory.utils.feature_generator_utils import seasonal_dummy_var_df


class CurveRegressorFactory:
    """
    Basis expansion regressor for commodity price curves.

    Represents forward curves as basis function expansions (splines, Fourier, PCA)
    and regresses basis coefficients on commodity parameters (spot, vol, convenience yield, etc.).
    """

    def __init__(
        self,
        curve_data_in: np.ndarray,
        dates: List,
    ):
        """
        Initialize commodity curve regressor.

        Args:
            curve_data_in: Price matrix of shape (n_dates, n_maturities).
            dates: Sequence of date labels for each row.
        """
        n_dates_in, n_maturities_in = curve_data_in.shape
        self.curve_data = curve_data_in

        self.unique_dates = [to_datetime(d) for d in dates]
        self.date_to_idx = {d: idx for idx, d in enumerate(self.unique_dates)}

        self.maturities = np.arange(1, n_maturities_in + 1, dtype=float)

        # Basis factory (set by fit_basis_coefficients)
        self._basis_factory: BasisFactory | None = None

        # Macro regression (set by fit_multioutput_reg)
        self.multi_output_reg: Optional[MultiOutputRegressor] = None
        self.regularization = "unset"
        self.alpha = 0.0
        self.scaler_params = StandardScaler()
        self.coef_mean: Optional[np.ndarray] = None
        self.coef_std: Optional[np.ndarray] = None

        # Macro regression diagnostics
        self.coef_r2_scores: Dict[int, float] = {}
        self.coef_rmse: Dict[int, float] = {}
        self.basis_importance: Dict[int, float] = {}

        # Feature interface (set by fit_multioutput_reg)
        self.feature_names: Optional[List[str]] = None
        self.feature_df: Optional[pd.DataFrame] = None

        # ARMA-GARCH feature simulators (set by fit_arma_garch_regressors)
        self._feature_arma_garch: Optional[Dict[str, ArmaGarchRegressor]] = None
        # Columns excluded from ARMA-GARCH fitting (generated deterministically)
        self._deterministic_cols: List[str] = ["seasonal_dummy"]

    # ── Convenience properties delegated to BasisFactory ─────────────────

    @property
    def basis_matrix(self) -> Optional[np.ndarray]:
        return self._basis_factory.basis_matrix

    @property
    def coefficients(self) -> Optional[np.ndarray]:
        return self._basis_factory.coefficients

    @property
    def n_basis(self) -> int:
        return self._basis_factory.n_basis

    @property
    def fit_r2(self) -> Optional[np.ndarray]:
        return self._basis_factory.fit_r2

    @property
    def fit_rmse(self) -> Optional[np.ndarray]:
        return self._basis_factory.fit_rmse

    @property
    def fit_mse(self) -> Optional[np.ndarray]:
        return self._basis_factory.fit_mse

    @property
    def fit_rss(self) -> Optional[np.ndarray]:
        return self._basis_factory.fit_rss

    @property
    def penalized_rmse(self) -> Optional[np.ndarray]:
        return self._basis_factory.penalized_rmse

    # ── Fitting ──────────────────────────────────────────────────────────

    def fit_basis_coefficients(
        self, basis_config: Optional[BasisFactoryConfig] = None
    ) -> "CurveRegressorFactory":
        """
        Fit basis coefficients from historical curves. Delegates entirely to
        the BasisFactory.

        Returns:
            self (for method chaining)
        """
        if basis_config is None:
            basis_config = BasisFactoryConfig()

        self._basis_factory = BasisFactory(basis_config)
        self._basis_factory.fit_coefficients(self.curve_data, self.maturities)

        return self

    def fit_multioutput_reg(
        self,
        parameters: pd.DataFrame,
        alpha: float,
        regularization: str = "ridge",
    ) -> "CurveRegressorFactory":
        """
        Regress basis coefficients on commodity/macro parameters using a single
        MultiOutputRegressor fitted on z-score-normalised coefficient targets.

        Args:
            parameters: DataFrame indexed by date (or with a 'date' column) containing
                        one column per feature.
            alpha: Regularisation strength. Defaults to self.alpha.
            regularization: 'ridge', 'lasso', or 'none'. Defaults to self.regularization.

        Returns:
            self (for method chaining)
        """
        # set auditable state variables
        self.alpha = alpha
        self.regularization = regularization

        self.feature_df = parameters
        self.feature_names = list(parameters.columns)

        if "date" in parameters.columns:
            parameters = parameters.set_index("date")

        common_dates = parameters.index.intersection(self.unique_dates)
        common_indices = [
            self.date_to_idx[d] for d in common_dates if d in self.date_to_idx
        ]

        C = self.coefficients[common_indices, :]
        self.coef_mean = C.mean(axis=0)
        self.coef_std = C.std(axis=0)
        C_norm = (C - self.coef_mean) / self.coef_std

        print(
            f"✓ Aligning {len(common_dates)} common dates between curves and parameters"
        )

        X = parameters.loc[common_dates, self.feature_names].values
        X_scaled = self.scaler_params.fit_transform(X)

        if regularization == "ridge":
            base_model = Ridge(alpha=alpha)
        elif regularization == "lasso":
            base_model = Lasso(alpha=alpha)
        else:
            base_model = LinearRegression()

        self.multi_output_reg = MultiOutputRegressor(base_model)
        self.multi_output_reg.fit(X_scaled, C_norm)

        print(
            f"Fitting MultiOutputRegressor({base_model.__class__.__name__}, alpha={alpha}) "
            f"on {X_scaled.shape[1]} features → {self.n_basis} basis coefficients..."
        )

        C_pred_norm = self.multi_output_reg.predict(X_scaled)
        for i, est in enumerate(self.multi_output_reg.estimators_):
            r2 = r2_score(C_norm[:, i], C_pred_norm[:, i])
            rmse = np.sqrt(mean_squared_error(C_norm[:, i], C_pred_norm[:, i]))
            self.coef_r2_scores[i] = r2
            self.coef_rmse[i] = rmse
            self.basis_importance[i] = float(np.abs(est.coef_).sum())

        mean_r2 = np.mean(list(self.coef_r2_scores.values()))
        mean_rmse = np.mean(list(self.coef_rmse.values()))
        print(f"✓ Fit complete")
        print(f"  Mean R² (normalised coefs) : {mean_r2:.4f}")
        print(f"  Mean RMSE (normalised coefs): {mean_rmse:.4f}")

        return self

    # ── ARMA-GARCH feature simulation ───────────────────────────────────

    def fit_arma_garch_regressors(
        self,
        feature_df: Optional[pd.DataFrame] = None,
        columns: Optional[List[str]] = None,
    ) -> "CurveRegressorFactory":
        """
        Fit one ArmaGarchRegressor per continuous feature column and store
        them for later simulation via generate_feature_df.

        Deterministic columns (e.g. seasonal_dummy) are excluded by default.

        Args:
            feature_df: Feature DataFrame to fit on. Defaults to self.feature_df.
            columns: Subset of column names to fit. Defaults to all columns
                     except those in self._deterministic_cols.

        Returns:
            self (for method chaining)
        """
        if feature_df is None:
            feature_df = self.feature_df
        if feature_df is None:
            raise ValueError(
                "No feature_df available. Call fit_parameters_to_coefficients first "
                "or pass feature_df explicitly."
            )

        if columns is None:
            columns = [
                c for c in self.feature_names if c not in self._deterministic_cols
            ]

        self._feature_arma_garch = {}
        for col in columns:
            model = ArmaGarchRegressor()
            model.fit(feature_df[col].dropna().to_numpy())
            self._feature_arma_garch[col] = model

        print(f"✓ Fitted ARMA-GARCH models for {len(columns)} features: {columns}")
        return self

    def generate_feature_df(
        self,
        T: int,
        seed: int,
        start_date: Union[str, pd.Timestamp] = "2027-01-01",
        freq: str = "B",
    ) -> pd.DataFrame:
        """
        Simulate a new feature DataFrame from the fitted ARMA-GARCH models.

        Continuous features are drawn from their fitted models. The seasonal
        dummy is derived from the synthetic date range via seasonal_dummy_var_df.

        Args:
            T: Number of time steps to simulate.
            seed: Random seed for reproducibility.
            start_date: Start date for the synthetic date range.
            freq: Pandas frequency string for the date range (default "B" for
                  business days).

        Returns:
            DataFrame of shape (T, n_features) indexed by date, with columns
            matching self.feature_names.
        """
        if self._feature_arma_garch is None:
            raise RuntimeError(
                "Call fit_arma_garch_regressors() before generate_feature_df()."
            )

        # Simulate continuous features
        sim_cols = {
            col: model.simulate(T=T, seed=seed)
            for col, model in self._feature_arma_garch.items()
        }
        dates = pd.date_range(start=start_date, periods=T, freq=freq)
        sim_df = pd.DataFrame(sim_cols, index=dates)

        # Construct seasonal dummy aligned to the simulated date range
        seasonal_df = seasonal_dummy_var_df(
            n_samples=T, start_date=str(start_date), freq=freq
        )
        sim_df = pd.concat([sim_df, seasonal_df], axis=1)

        return sim_df[self.feature_names]

    # ── Prediction ───────────────────────────────────────────────────────

    def predict_coefficients(self, parameters: Dict[str, float]) -> np.ndarray:
        """
        Predict basis coefficients (unnormalised) for one set of feature values.
        """
        param_vector = np.array(
            [parameters.get(p, 0.0) for p in self.feature_names]
        ).reshape(1, -1)
        param_scaled = self.scaler_params.transform(param_vector)
        coefs_norm = self.multi_output_reg.predict(param_scaled)[0]
        return coefs_norm * self.coef_std + self.coef_mean

    def predict_curves_given_features(
        self, parameters_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Predict curves for multiple parameter sets.

        Args:
            parameters_df: DataFrame with one row per date and one column per feature.

        Returns:
            DataFrame of shape (n_rows, n_maturities) with predicted prices.
        """
        X = parameters_df[self.feature_names].values
        X_scaled = self.scaler_params.transform(X)

        C_norm = self.multi_output_reg.predict(X_scaled)
        C = C_norm * self.coef_std + self.coef_mean

        predicted = C @ self.basis_matrix.T
        return pd.DataFrame(
            predicted,
            index=parameters_df.index,
            columns=[f"maturity_{m}" for m in self.maturities],
        )

    # ── Reconstruction helpers ───────────────────────────────────────────

    def reconstruct_maturity_for_date(
        self, target_date: Union[pd.Timestamp, dt.datetime]
    ):
        if target_date not in self.date_to_idx:
            raise ValueError(f"Date {target_date} not found in fitted data.")

        date_idx = self.date_to_idx[target_date]
        coefs = self.coefficients[date_idx]

        curve_prices = self.curve_data[date_idx, :]
        fitted_prices = self._basis_factory.reconstruct(coefs)

        return {
            "date": target_date,
            "observed_prices": curve_prices,
            "fitted_prices": fitted_prices,
            "r2_score": self.fit_r2[date_idx],
            "rmse_score": self.fit_rmse[date_idx],
        }

    # ── Serialization ────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """
        Serialize this fitted regressor to disk using joblib.

        Args:
            path: Destination file path (e.g. "models/ttf_regressor.joblib").
                  Parent directories are created automatically.
        """
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        state = {
            # Regressor params
            "regularization": self.regularization,
            "alpha": self.alpha,
            "maturities": self.maturities,
            # Date index
            "unique_dates": self.unique_dates,
            "date_to_idx": self.date_to_idx,
            # BasisFactory (entire object — config + fitted state)
            "_basis_factory": self._basis_factory,
            # Feature / regression interface
            "feature_names": self.feature_names,
            "feature_df": self.feature_df,
            "scaler_params": self.scaler_params,
            "multi_output_reg": self.multi_output_reg,
            "coef_mean": self.coef_mean,
            "coef_std": self.coef_std,
            # Training data (kept for BasisVisualizer and diagnostics)
            "curve_data": self.curve_data,
            # Macro regression diagnostics
            "coef_r2_scores": self.coef_r2_scores,
            "coef_rmse": self.coef_rmse,
            "basis_importance": self.basis_importance,
            # ARMA-GARCH feature simulators
            "_feature_arma_garch": self._feature_arma_garch,
        }
        joblib.dump(state, path, compress=3)
        print(f"Saved HistoricalCurveRegressor to {path}")

    @classmethod
    def load(cls, path: str) -> "CurveRegressorFactory":
        """
        Load a serialized regressor from disk.

        Args:
            path: Path to the saved .joblib file.

        Returns:
            Reconstructed HistoricalCurveRegressor, ready for predict_curve calls.
        """
        import joblib

        state = joblib.load(path)

        obj = cls.__new__(cls)
        obj.curve_data = state["curve_data"]
        obj.unique_dates = state["unique_dates"]
        obj.date_to_idx = state["date_to_idx"]
        obj.regularization = state["regularization"]
        obj.alpha = state["alpha"]
        obj.maturities = state["maturities"]

        # Restore BasisFactory — handle both old and new save formats
        if "_basis_factory" in state:
            obj._basis_factory = state["_basis_factory"]
        else:
            # Backward compat: reconstruct from old flat fields
            config = BasisFactoryConfig(
                n_basis=state["n_basis"],
                degree=state["degree"],
                basis_type=state["_basis_type"],
                knots=state["_knots"],
            )
            obj._basis_factory = BasisFactory(config)
            obj._basis_factory.basis_matrix = state["basis_matrix"]
            obj._basis_factory.coefficients = state["coefficients"]
            obj._basis_factory.fit_r2 = state.get("fit_r2")
            obj._basis_factory.fit_rmse = state.get("fit_rmse")
            obj._basis_factory.fit_mse = state.get("fit_mse")
            obj._basis_factory.fit_rss = state.get("fit_rss")
            obj._basis_factory.penalized_rmse = state.get("penalized_rmse")

        obj.multi_output_reg = state.get("multi_output_reg")
        obj.coef_mean = state.get("coef_mean")
        obj.coef_std = state.get("coef_std")
        obj.scaler_params = state.get("scaler_params")
        obj.feature_names = state.get("feature_names")
        obj.feature_df = state.get("feature_df")
        obj.coef_r2_scores = state.get("coef_r2_scores", {})
        obj.coef_rmse = state.get("coef_rmse", {})
        obj.basis_importance = state.get("basis_importance", {})
        obj._feature_arma_garch = state.get("_feature_arma_garch")
        obj._deterministic_cols = ["seasonal_dummy"]

        print(f"Loaded HistoricalCurveRegressor from {path}")
        print(f"  feature_names : {obj.feature_names}")
        print(
            f"  basis_type    : {obj._basis_factory.basis_type},  n_basis: {obj.n_basis}"
        )
        return obj
