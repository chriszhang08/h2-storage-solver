from typing import Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

from curve_factory.etl.fwd_curve_loader import CurveMatrix
from curve_factory.utils.curve_data_transformers import z_score_normalize


def seasonal_dummy_var_df(
    n_samples: int,
    start_date: str = "2015-01-01",
    freq: str = "B",  # business days
) -> pd.DataFrame:
    """
    Generate a synthetic DataFrame with seasonal dummy variables.

    Simple seasonal pattern:
    - Months [11-3] => 1 (winter)
    - Months [4-5] => 0 (shoulder)
    - Months [6-8] => -1 (summer)
    - Months [9-10] => 0 (shoulder)

    Args:
    - n_samples: Number of rows (time points) to generate.
    - start_date: Starting date for the index.
    - freq: Frequency string for date range (e.g., 'B' for business days
    """
    date_range = pd.date_range(start=start_date, periods=n_samples, freq=freq)
    month = date_range.month

    # Define the seasonal pattern
    conditions = [
        (month.isin([11, 12, 1, 2, 3])),  # Winter
        (month.isin([4, 5])),  # Shoulder
        (month.isin([6, 7, 8])),  # Summer
        (month.isin([9, 10])),  # Shoulder
    ]
    choices = [1, 0, -1, 0]

    seasonal_dummy = np.select(conditions, choices)

    return pd.DataFrame(
        {"date": date_range, "seasonal_dummy": seasonal_dummy}
    ).set_index("date")


def compute_rolling_volatility(
    price_matrix, window_size, annualize=True, log_returns=True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute rolling mean and standard deviation of futures prices across maturities.

    Theory:
        For a price matrix P of shape (n_dates, n_maturities), compute:
        - r_{t,j} = log(P_t,j / P_{t-1,j})  [log-returns]
        - mean_t,j^(w) = (1/w) * sum_{i=t-w+1}^{t} r_{i,j}
        - std_t,j^(w) = sqrt( (1/(w-1)) * sum_i (r_{i,j} - mean)^2 )

    Args:
        price_matrix: np.ndarray of shape (n_dates, n_maturities)
                      rows = dates, columns = tenors/maturities
        window_size: int, rolling window size (e.g., 20 days)
        annualize: bool, annualize volatility (multiply by sqrt(252))
        log_returns: bool, if True use log-returns; else arithmetic returns

    Returns:
        rolling_mean: np.ndarray of shape (n_dates, n_maturities)
                      rolling mean return for each maturity
        rolling_std: np.ndarray of shape (n_dates, n_maturities)
                     rolling standard deviation (volatility) for each maturity
    """
    if price_matrix.shape[0] < window_size:
        raise ValueError(
            f"Window size {window_size} exceeds data length {price_matrix.shape[0]}"
        )

    n_dates, n_maturities = price_matrix.shape

    # Step 1: Compute log-returns
    if log_returns:
        returns = np.diff(np.log(price_matrix), axis=0)  # (n_dates-1, n_maturities)
    else:
        returns = np.diff(price_matrix, axis=0) / price_matrix[:-1, :]

    # Pad with NaN for alignment (optional, for consistent shape)
    returns = np.vstack(
        [np.full((1, n_maturities), np.nan), returns]
    )  # (n_dates, n_maturities)

    # Step 2: Initialize output arrays
    rolling_mean = np.full((n_dates, n_maturities), np.nan)
    rolling_std = np.full((n_dates, n_maturities), np.nan)

    # Step 3: Compute rolling statistics using pandas (efficient)
    returns_df = pd.DataFrame(returns)

    rolling_mean_df = returns_df.rolling(
        window=window_size, min_periods=window_size
    ).mean()
    rolling_std_df = returns_df.rolling(
        window=window_size, min_periods=window_size
    ).std(ddof=1)

    rolling_mean = rolling_mean_df.values
    rolling_std = rolling_std_df.values

    # Step 4: Annualize if requested
    if annualize:
        rolling_std = rolling_std * np.sqrt(252)
        rolling_mean = rolling_mean * 252

    return rolling_mean, rolling_std


def compute_rolling_curvature(price_matrix: np.ndarray, window_size: int) -> np.ndarray:
    """
    weihong this is probably the worst one
    Compute a time series of futures curve curvature and its rolling average.

    Curvature here is a single scalar per date: the second derivative of the
    cubic spline fit to the curve, evaluated at the middle maturity.

    Args:
        price_matrix: np.ndarray of shape (n_dates, n_maturities)
                      rows = dates, columns = tenors/maturities
        window_size: int, rolling window size (e.g., 20 days)

    Returns:
        rolling_curvature: np.ndarray of shape (n_dates,)
                           rolling mean curvature over the given window.
                           First window_size-1 entries will be NaN.
    """
    n_dates, n_maturities = price_matrix.shape

    if n_maturities < 3:
        raise ValueError("Need at least 3 maturities to compute curvature.")

    # 1) Compute scalar curvature per date
    maturities = np.arange(n_maturities)  # assume equally spaced tenors
    mid_idx = n_maturities // 2  # evaluate curvature at middle maturity

    curvature_arr = np.empty(n_dates, dtype=float)

    for t in range(n_dates):
        prices = price_matrix[t, :]
        cs = CubicSpline(maturities, prices)
        # Second derivative at mid maturity
        curvature_arr[t] = cs(maturities[mid_idx], 2)

    # 2) Compute rolling mean curvature
    curvature_series = pd.Series(curvature_arr)
    rolling_curvature = (
        curvature_series.rolling(window=window_size, min_periods=window_size)
        .mean()
        .to_numpy()
    )

    return rolling_curvature


def compute_curve_slope_series(price_matrix: np.ndarray) -> np.ndarray:
    """
    Compute the term-structure slope for each row (date) in a price matrix.

    Slope definition (per date):
        slope = (P_long - P_short) / (maturity_long - maturity_short)
              = (prices[-1] - prices[0]) / (n_maturities - 1)
    assuming equally spaced maturities.

    Args:
        price_matrix: np.ndarray of shape (n_dates, n_maturities)
                      rows = dates, columns = maturities

    Returns:
        slopes: np.ndarray of shape (n_dates,)
                slope for each date.
    """
    if price_matrix.ndim != 2:
        raise ValueError(f"price_matrix must be 2D, got shape {price_matrix.shape}")

    n_dates, n_maturities = price_matrix.shape

    if n_maturities < 2:
        raise ValueError("Need at least 2 maturities to compute slope.")

    # Short and long end prices per row
    short_end = price_matrix[:, 0]
    long_end = price_matrix[:, -1]

    # Maturity distance (assume equally spaced tenors 0, 1, ..., n_maturities-1)
    maturity_span = float(n_maturities - 1)

    # Vectorized slope computation
    slopes = (long_end - short_end) / maturity_span

    return slopes.astype(float)


def compute_curve_features(
    price_matrix: np.ndarray, dates: np.ndarray, window_size: int
) -> pd.DataFrame:
    """
    Compute rolling features (volatility, curvature, slope) and aggregate by knots.

    Parameters
    ----------
    price_matrix : np.ndarray of shape (n_dates, n_tenors)
    dates : np.ndarray of shape (n_dates,) with date labels corresponding to price_matrix rows
    window_size  : int, rolling window size for volatility and curvature

    Returns
    -------
    DataFrame with columns:
        - 'rolling_slope'
        - 'rolling_curvature'
        - 'rolling_std_{lo}_{hi}' for each knot segment
    """
    rolling_mean, rolling_std = compute_rolling_volatility(price_matrix, window_size)
    rolling_curvature = compute_rolling_curvature(price_matrix, window_size)
    rolling_slope = compute_curve_slope_series(price_matrix)

    rolling_curvature_df = pd.DataFrame(
        data=rolling_curvature,
        index=dates,
        columns=["rolling_curvature"],
    ).dropna()

    slope_series_df = pd.DataFrame(
        data=rolling_slope,
        index=dates,
        columns=["rolling_slope"],
    ).dropna()

    rolling_std_df = (
        pd.DataFrame(
            data=rolling_std,
            index=dates,
        )
        .dropna()
        .mean(axis=1)
        .to_frame(name="rolling_std")
    )

    return pd.concat([rolling_curvature_df, slope_series_df, rolling_std_df], axis=1)


def build_feature_df(
    curve_matrix: CurveMatrix,
    window_size: int = 10,
) -> pd.DataFrame:
    """
    Combine shared macro features with commodity-specific curve shape features
    and a seasonal dummy, then z-score normalize continuous columns.

    The seasonal dummy is constructed from the curve matrix's date range via
    seasonal_dummy_var_df, so it is aligned to the observation dates.

    Args:
        curve_matrix: CurveMatrix for this commodity.
        window_size: Rolling window for curve shape features.

    Returns:
        Normalized feature DataFrame ready for regressor training.
    """
    curve_shape_feats = compute_curve_features(
        price_matrix=curve_matrix.matrix,
        dates=curve_matrix.dates,
        window_size=window_size,
    )

    seasonal_df = seasonal_dummy_var_df(
        n_samples=len(curve_matrix.dates),
        start_date=str(curve_matrix.dates[0]),
        freq="B",
    )

    combined = pd.concat([curve_shape_feats, seasonal_df], axis=1).dropna()

    # Normalize all columns except the seasonal dummy (categorical-like)
    continuous_cols = [c for c in combined.columns if c != "seasonal_dummy"]

    normalized = combined.copy()
    normalized[continuous_cols] = combined[continuous_cols].apply(z_score_normalize)

    return normalized
