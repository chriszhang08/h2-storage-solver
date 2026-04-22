from typing import List

import numpy as np

from constants import Z_SCORE_CLIP


def _encode_year(year: np.ndarray, offset: int = 0) -> np.ndarray:
    """
    Return [front_month, first_extreme, first_extreme_idx, second_extreme, second_extreme_idx]
    ordered so the earlier-occurring extreme comes first.

    offset: added to both indices so they are absolute positions in the full curve.
    The returned np.ndarray has shape (5,)
    """
    encoding_horizion = 24

    max_idx, min_idx = np.argmax(year), np.argmin(year)

    normed_max_idx = (max_idx + offset) / encoding_horizion
    normed_min_idx = (min_idx + offset) / encoding_horizion

    if max_idx < min_idx:
        extremes = [year[max_idx], normed_max_idx, year[min_idx], normed_min_idx]
    else:
        extremes = [year[min_idx], normed_min_idx, year[max_idx], normed_max_idx]
    return np.array([year[0]] + extremes)


def encode_price_curve(
    curve: np.ndarray, sample_mean: float = 0, sample_std: float = 0
) -> np.ndarray:
    """
    Expresses the forward curve in the same z-scored distribution as norm_curr_spot_price.

    Applies the same transform used by normalize_spot_price/scale_spot_price:
        z = (price - sample_mean) / sample_std

    Returns: np.ndarray with the same shape as curve, clipped to [-Z_SCORE_CLIP, Z_SCORE_CLIP]
    """
    if sample_std == 0:
        return np.zeros_like(curve, dtype=float)

    normed_curve = (curve - sample_mean) / sample_std
    return np.clip(normed_curve, -Z_SCORE_CLIP, Z_SCORE_CLIP)


def normalize_spot_price(price_history: List[float]) -> tuple[float, float, float]:
    """
    Z-scale the current spot price with respect to the price history.

    Requires: the most recent spot price is the last element of the price_history list
    """
    # convert the price history to nd array
    price_history_arr: np.ndarray = np.array(price_history)

    if len(price_history_arr) == 0:
        return 0, 0, 0

    # z normalize it
    std = price_history_arr.std()
    if std == 0:
        return 0, 0, 0
    normed_price_history = (price_history_arr - price_history_arr.mean()) / std

    # clip within bounds
    normed_price_history_clip = np.clip(
        normed_price_history, -Z_SCORE_CLIP, Z_SCORE_CLIP
    )

    return float(normed_price_history_clip[-1]), float(price_history_arr.mean()), float(std)


def scale_spot_price(spot_price: float, sample_mean: float, sample_std: float) -> float:
    """
    Takes an observation and scales it to another samples mean and standard deviation.

    Also clips the observation within the Z_SCORE_CLIP range.
    """
    if sample_std == 0:
        return 0

    z_a = (spot_price - sample_mean) / sample_std

    return np.clip(z_a, -Z_SCORE_CLIP, Z_SCORE_CLIP)