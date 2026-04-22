import datetime as dt
import numpy as np
import pandas as pd


def matrix_to_long(
    matrix: np.ndarray,
    dates: np.ndarray,
    maturities: np.ndarray,
) -> pd.DataFrame:
    """
    Convert a (n_dates, n_maturities) price matrix to a long-form DataFrame.

    Args:
        matrix:     Price matrix of shape (n_dates, n_maturities).
        dates:      1-D array of date labels for each row (length n_dates).
        maturities: 1-D array of maturity values for each column (length n_maturities).

    Returns:
        pd.DataFrame with columns ['date', 'maturity', 'price'].
    """
    pivot_df = pd.DataFrame(matrix, index=dates, columns=maturities)
    df = pivot_df.stack().reset_index()
    df.columns = ["date", "maturity", "price"]
    return df


def long_to_matrix(
    df: pd.DataFrame,
    date_col: str = "date",
    maturity_col: str = "maturity",
    price_col: str = "price",
) -> pd.DataFrame:
    """
    Convert a long-form price DataFrame to a wide pivot (date × maturity).

    Args:
        df:           Long-form DataFrame with one row per (date, maturity) pair.
        date_col:     Name of the date column.
        maturity_col: Name of the maturity column.
        price_col:    Name of the price column.

    Returns:
        pd.DataFrame with dates as index, maturities as columns, and prices as
        values, sorted on both axes.  Call `.to_numpy()` / `.index` / `.columns`
        to extract the raw numpy matrix and axis labels.
    """
    pivot = df.pivot_table(
        values=price_col, index=date_col, columns=maturity_col, aggfunc="mean"
    )
    return pivot.sort_index(axis=0).sort_index(axis=1)


def interpolate_forward_curve(
    forward_curve: np.ndarray,
    steps_per_period: int = 30,
    mode: str = "linear",
) -> np.ndarray:
    """
    Interpolate a forward curve from coarse periods to finer steps using piecewise
    interpolation.

    Each input price is treated as an anchor at the start of its period
    (step 0, steps_per_period, 2*steps_per_period, ...).  A dense output grid
    is built and each output step is assigned a price by interpolation between its
    two neighbouring anchors.  Out-of-range steps are filled with the nearest
    boundary price (flat extrapolation).

    Examples:
        Monthly → daily:  steps_per_period=30
        Monthly → weekly: steps_per_period=4  (approx 4 weeks per month)
        Weekly  → daily:  steps_per_period=7

    Args:
        forward_curve:   np.ndarray of shape (n_periods,) with one price per coarse period.
        steps_per_period: Number of fine-grained output steps per input period (default 30).
        mode:            'linear' for piecewise-linear interpolation in price space, or
                         'loglinear' to interpolate in log-price space (requires all
                         prices > 0).

    Returns:
        np.ndarray of shape (n_periods * steps_per_period,) with interpolated prices.
    """
    n_periods = len(forward_curve)
    n_steps = n_periods * steps_per_period

    # Coarse anchors at positions 0, steps_per_period, 2*steps_per_period, ...
    anchor_x = np.arange(n_periods, dtype=float) * steps_per_period
    fine_x = np.arange(n_steps, dtype=float)

    if mode == "loglinear":
        if np.any(forward_curve <= 0):
            raise ValueError("All forward_curve prices must be > 0 for loglinear mode.")
        y = np.log(forward_curve)
    else:
        y = forward_curve.astype(float)

    fine_y = np.empty(n_steps, dtype=float)

    for k in range(n_steps):
        x = fine_x[k]

        if x <= anchor_x[0]:
            fine_y[k] = y[0]
        elif x >= anchor_x[-1]:
            fine_y[k] = y[-1]
        else:
            # find j such that anchor_x[j] <= x < anchor_x[j+1]
            j = int(x // steps_per_period)
            j = min(j, n_periods - 2)  # guard against floating-point edge

            x0, x1 = anchor_x[j], anchor_x[j + 1]
            y0, y1 = y[j], y[j + 1]
            weight = (x - x0) / (x1 - x0)
            fine_y[k] = y0 * (1.0 - weight) + y1 * weight

    return np.exp(fine_y) if mode == "loglinear" else fine_y


# Convert all dates to datetime.datetime objects
def to_datetime(date):
    if isinstance(date, np.datetime64):
        return date.astype("M8[ms]").astype(dt.datetime)
    elif isinstance(dt, str):
        # Try parsing ISO format
        try:
            return dt.datetime.fromisoformat(date)
        except Exception:
            # Fallback: try parsing as date only
            return dt.datetime.strptime(date, "%Y-%m-%d")
    elif isinstance(date, dt.datetime):
        return date
    elif isinstance(date, dt.date):
        return dt.datetime.combine(date, dt.time())
    else:
        raise TypeError(f"Unsupported date type: {type(dt)}")


def z_score_normalize(series: pd.Series) -> pd.Series:
    return (series - series.mean()) / series.std()
