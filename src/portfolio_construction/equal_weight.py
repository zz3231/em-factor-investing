"""Equal-weight portfolio construction."""

import numpy as np
import pandas as pd


def compute_weights(
    returns_df: pd.DataFrame,
    cov_matrix: np.ndarray | None = None,
    mean_returns: np.ndarray | None = None,
    bounds: tuple[float, float] | None = None,
    **kwargs,
) -> np.ndarray:
    """Equal weight allocation across all assets.

    Parameters
    ----------
    returns_df : pd.DataFrame
        T x N matrix of asset returns.
    cov_matrix : np.ndarray, optional
        N x N covariance matrix (unused).
    mean_returns : np.ndarray, optional
        N-vector of expected returns (unused).
    bounds : tuple[float, float], optional
        (lower, upper) weight bounds per asset (unused).

    Returns
    -------
    np.ndarray
        N-vector of portfolio weights summing to 1.
    """
    n = returns_df.shape[1]
    return np.ones(n) / n
