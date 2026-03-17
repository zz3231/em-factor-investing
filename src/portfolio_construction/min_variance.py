"""Minimum-variance portfolio construction via quadratic optimisation."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


_DEFAULT_BOUNDS = (0.02, 0.20)


def compute_weights(
    returns_df: pd.DataFrame,
    cov_matrix: np.ndarray | None = None,
    mean_returns: np.ndarray | None = None,
    bounds: tuple[float, float] | None = None,
    **kwargs,
) -> np.ndarray:
    """Minimum-variance portfolio: minimise w' Cov w s.t. sum(w) = 1.

    Parameters
    ----------
    returns_df : pd.DataFrame
        T x N matrix of asset returns.
    cov_matrix : np.ndarray, optional
        N x N covariance matrix. If None, estimated from *returns_df*.
    mean_returns : np.ndarray, optional
        N-vector of expected returns (unused).
    bounds : tuple[float, float], optional
        (lower, upper) weight bounds per asset. Defaults to (0.02, 0.20).

    Returns
    -------
    np.ndarray
        N-vector of portfolio weights summing to 1.
    """
    if cov_matrix is None:
        cov_matrix = returns_df.cov().values

    n = cov_matrix.shape[0]
    if bounds is None:
        bounds = _DEFAULT_BOUNDS

    def objective(w):
        return w @ cov_matrix @ w

    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
    x0 = np.ones(n) / n

    result = minimize(
        objective,
        x0,
        method='SLSQP',
        bounds=[bounds] * n,
        constraints=constraints,
        options={'ftol': 1e-12, 'maxiter': 1000},
    )

    if not result.success:
        raise RuntimeError(f"Min-variance optimisation failed: {result.message}")

    return result.x
