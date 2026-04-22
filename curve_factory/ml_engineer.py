from typing import Optional

import numpy as np
import pandas as pd

from constants import KNOT_VECTOR
from curve_factory import (
    CurveRegressorFactory,
)
from curve_factory.basis_factory import BasisFactoryConfig
from analysis.regressor_analysis import evaluate_regressor, print_diagnostics
from curve_factory.etl.fwd_curve_loader import CurveMatrix

from curve_factory.utils.feature_generator_utils import build_feature_df

np.random.seed(42)


def train_regressor(
    commodity: str,
    curve_matrix: CurveMatrix,
    feature_df_norm: pd.DataFrame,
    version: int,
    basis_config: Optional[BasisFactoryConfig] = None,
    alpha: float = 1.5,
    regularization: str = "ridge",
) -> CurveRegressorFactory:
    """
    Build, fit, and save a HistoricalCurveRegressor for a single commodity.

    Args:
        commodity: Commodity name (used for the saved model filename).
        curve_matrix: CurveMatrix containing .matrix, .maturities, .dates.
        feature_df_norm: Normalized feature DataFrame (from build_feature_df).
        version: Model version number for the saved filename.
        basis_config: BasisFactoryConfig. Defaults to hybrid with 17 basis, degree 3.
        alpha: Regularization strength for the macro→coefficient regression.
        regularization: Regularization type ('ridge', 'lasso', 'none').

    Returns:
        Fitted HistoricalCurveRegressor.
    """
    if basis_config is None:
        basis_config = BasisFactoryConfig(
            n_basis=17, degree=3, basis_type="hybrid", knots=KNOT_VECTOR
        )

    regressor = CurveRegressorFactory(
        curve_data_in=curve_matrix.matrix,
        dates=list(curve_matrix.dates),
    )
    regressor.fit_basis_coefficients(basis_config=basis_config)

    regressor.fit_multioutput_reg(
        parameters=feature_df_norm,
        alpha=alpha,
        regularization=regularization,
    )
    regressor.save(f"models/{commodity}_regressor_v{version}.joblib")

    return regressor
