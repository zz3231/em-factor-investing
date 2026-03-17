"""Inverse-variance (inverse-volatility) portfolio construction."""

import numpy as np
import pandas as pd


def compute_weights(
    returns_df: pd.DataFrame,
    cov_matrix: np.ndarray | None = None,
    mean_returns: np.ndarray | None = None,
    bounds: tuple[float, float] | None = None,
    **kwargs,
) -> np.ndarray:
    """Inverse-variance weighting: w_i proportional to 1 / sigma_i^2.

    Parameters
    ----------
    returns_df : pd.DataFrame
        T x N matrix of asset returns.
    cov_matrix : np.ndarray, optional
        N x N covariance matrix. If None, estimated from *returns_df*.
    mean_returns : np.ndarray, optional
        N-vector of expected returns (unused).
    bounds : tuple[float, float], optional
        (lower, upper) weight bounds per asset (unused).

    Returns
    -------
    np.ndarray
        N-vector of portfolio weights summing to 1.
    """
    if cov_matrix is None:
        cov_matrix = returns_df.cov().values

    variances = np.diag(cov_matrix)
    inv_var = 1.0 / variances
    weights = inv_var / inv_var.sum()
    return weights
