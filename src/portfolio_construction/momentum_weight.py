"""Momentum-weighted portfolio construction."""

import numpy as np
import pandas as pd


def compute_weights(
    returns_df: pd.DataFrame,
    cov_matrix: np.ndarray | None = None,
    mean_returns: np.ndarray | None = None,
    bounds: tuple[float, float] | None = None,
    **kwargs,
) -> np.ndarray:
    """Weight assets by trailing return (momentum tilt).

    Uses the most recent ``lookback`` months of returns to compute
    each asset's average monthly return, then normalises to portfolio
    weights.  Negative trailing returns are floored at a small positive
    value so every asset receives some allocation.

    Parameters
    ----------
    returns_df : pd.DataFrame
        T x N matrix of asset returns.
    cov_matrix, mean_returns, bounds :
        Unused; accepted for interface compatibility.
    **kwargs :
        ``lookback`` (int, default 12): months of trailing return to use.

    Returns
    -------
    np.ndarray
        N-vector of portfolio weights summing to 1.
    """
    lookback = kwargs.get("lookback", 12)
    n = returns_df.shape[1]
    window = returns_df.iloc[-lookback:]
    trailing = window.mean().values

    shifted = trailing - trailing.min() + 1e-8
    weights = shifted / shifted.sum()
    return weights
