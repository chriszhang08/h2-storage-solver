"""
Evaluation and diagnostics for fitted HistoricalCurveRegressor models.

Provides three levels of analysis:
  1. Coefficient-level: R² and RMSE per basis function.
  2. Curve-level: reconstruction R² and RMSE across dates.
  3. Feature importance: mean absolute regression weight per feature.
"""

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error

from curve_factory import CurveRegressorFactory


# ── Result containers ────────────────────────────────────────────────────────


@dataclass
class CoefficientDiagnostics:
    """Per-basis-function R² and RMSE from the macro→coefficient regression."""

    r2_per_basis: np.ndarray
    rmse_per_basis: np.ndarray

    @property
    def mean_r2(self) -> float:
        return float(self.r2_per_basis.mean())

    @property
    def mean_rmse(self) -> float:
        return float(self.rmse_per_basis.mean())


@dataclass
class CurveDiagnostics:
    """Per-date reconstruction R² and RMSE (predicted vs observed curves)."""

    r2_per_date: np.ndarray
    rmse_per_date: np.ndarray
    dates: pd.DatetimeIndex

    @property
    def mean_r2(self) -> float:
        return float(self.r2_per_date.mean())

    @property
    def mean_rmse(self) -> float:
        return float(self.rmse_per_date.mean())


@dataclass
class FeatureImportance:
    """Mean absolute regression weight per feature, averaged across basis functions."""

    feature_names: List[str]
    importance: np.ndarray

    def sorted_pairs(self) -> List[tuple]:
        """Return (feature_name, importance) pairs sorted descending."""
        return sorted(zip(self.feature_names, self.importance), key=lambda x: -x[1])


@dataclass
class RegressorDiagnostics:
    """Combined diagnostics for a single regressor."""

    commodity: str
    coefficient: CoefficientDiagnostics
    curve: CurveDiagnostics
    feature_importance: FeatureImportance


# ── Computation ──────────────────────────────────────────────────────────────


def evaluate_regressor(
    commodity: str,
    regressor: CurveRegressorFactory,
    feature_df_norm: pd.DataFrame,
) -> RegressorDiagnostics:
    """
    Compute all diagnostics for a fitted regressor.

    Args:
        commodity: Label for this commodity (e.g. "ttf", "nlpwr").
        regressor: A fully fitted HistoricalCurveRegressor (basis + macro stages).
        feature_df_norm: The normalized feature DataFrame used during training.

    Returns:
        RegressorDiagnostics containing coefficient, curve, and feature importance results.
    """
    # 1. Coefficient-level diagnostics (already stored by fit_parameters_to_coefficients)
    coef_diag = CoefficientDiagnostics(
        r2_per_basis=np.array(list(regressor.coef_r2_scores.values())),
        rmse_per_basis=np.array(list(regressor.coef_rmse.values())),
    )

    # 2. Curve-level reconstruction diagnostics
    common_dates = feature_df_norm.index.intersection(regressor.unique_dates)
    common_indices = [regressor.date_to_idx[d] for d in common_dates]

    observed_curves = regressor.curve_data[common_indices, :]
    predicted_curves = regressor.predict_curves_given_features(feature_df_norm).values

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

    curve_diag = CurveDiagnostics(
        r2_per_date=curve_r2,
        rmse_per_date=curve_rmse,
        dates=common_dates,
    )

    # 3. Feature importance (mean |coef| across basis functions)
    coef_matrix = np.array(
        [est.coef_ for est in regressor.multi_output_reg.estimators_]
    )
    importance = np.abs(coef_matrix).mean(axis=0)
    feature_names = list(feature_df_norm.columns)

    feat_imp = FeatureImportance(
        feature_names=feature_names,
        importance=importance,
    )

    return RegressorDiagnostics(
        commodity=commodity,
        coefficient=coef_diag,
        curve=curve_diag,
        feature_importance=feat_imp,
    )


# ── Printing / reporting ────────────────────────────────────────────────────


def print_diagnostics(diag: RegressorDiagnostics) -> None:
    """Print a full diagnostic report for a single commodity regressor."""
    label = diag.commodity.upper()
    coef = diag.coefficient
    curve = diag.curve
    feat = diag.feature_importance

    # Coefficient-level
    print(f"\n{'=' * 60}")
    print(f"MACRO REGRESSOR ({label}) — COEFFICIENT-LEVEL DIAGNOSTICS")
    print(f"{'=' * 60}")
    print(f"{'Basis':>6}  {'R²':>8}  {'RMSE':>10}")
    print("-" * 30)
    for i, (r2, rmse) in enumerate(zip(coef.r2_per_basis, coef.rmse_per_basis)):
        print(f"{i:>6}  {r2:>8.4f}  {rmse:>10.4f}")
    print("-" * 30)
    print(f"{'Mean':>6}  {coef.mean_r2:>8.4f}  {coef.mean_rmse:>10.4f}")
    print(
        f"{'Median':>6}  {np.median(coef.r2_per_basis):>8.4f}"
        f"  {np.median(coef.rmse_per_basis):>10.4f}"
    )

    # Curve-level
    print(f"\n{'=' * 60}")
    print(f"MACRO REGRESSOR ({label}) — CURVE-LEVEL RECONSTRUCTION DIAGNOSTICS")
    print(f"{'=' * 60}")
    print(f"N dates evaluated : {len(curve.r2_per_date)}")
    print(f"Mean curve R²     : {curve.mean_r2:.4f} ± {curve.r2_per_date.std():.4f}")
    print(f"Median curve R²   : {np.median(curve.r2_per_date):.4f}")
    print(
        f"Mean curve RMSE   : {curve.mean_rmse:.4f} ± {curve.rmse_per_date.std():.4f}"
    )
    print(f"95th pct RMSE     : {np.percentile(curve.rmse_per_date, 95):.4f}")
    print(
        f"Worst curve R²    : {curve.r2_per_date.min():.4f}"
        f"  (date: {curve.dates[np.argmin(curve.r2_per_date)].date()})"
    )
    print(
        f"Best  curve R²    : {curve.r2_per_date.max():.4f}"
        f"  (date: {curve.dates[np.argmax(curve.r2_per_date)].date()})"
    )

    # Feature importance
    print(f"\n{'=' * 60}")
    print(f"MACRO REGRESSOR ({label}) — FEATURE IMPORTANCE (mean |coef| across basis)")
    print(f"{'=' * 60}")
    for name, imp in feat.sorted_pairs():
        print(f"  {name:<30}  {imp:.4f}")
