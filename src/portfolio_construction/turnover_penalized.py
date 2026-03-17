"""Turnover-penalised mean-variance portfolio construction."""

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
    """MVO with an L1 turnover penalty relative to previous weights.

    Objective: max  mu'w - (lam/2) w'Sigma w - gamma * ||w - w_prev||_1

    Parameters
    ----------
    returns_df : pd.DataFrame
        T x N matrix of asset returns.
    cov_matrix : np.ndarray, optional
        N x N covariance matrix.
    mean_returns : np.ndarray, optional
        Expected return vector.
    bounds : tuple[float, float], optional
        Weight bounds. Defaults to (0.02, 0.20).
    **kwargs :
        ``prev_weights`` (np.ndarray): previous period weights (default 1/N).
        ``gamma`` (float, default 0.5): turnover penalty strength.
        ``risk_aversion`` (float, default 2.5): risk-return trade-off.

    Returns
    -------
    np.ndarray
        N-vector of portfolio weights summing to 1.
    """
    gamma = kwargs.get("gamma", 0.001)
    risk_aversion = kwargs.get("risk_aversion", 2.5)

    if cov_matrix is None:
        cov_matrix = returns_df.cov().values
    if mean_returns is None:
        mean_returns = returns_df.mean().values
    if bounds is None:
        bounds = _DEFAULT_BOUNDS

    n = cov_matrix.shape[0]
    prev_weights = kwargs.get("prev_weights", np.ones(n) / n)

    def objective(w):
        ret = w @ mean_returns
        risk = w @ cov_matrix @ w
        turnover = np.sum(np.abs(w - prev_weights))
        return -(ret - 0.5 * risk_aversion * risk) + gamma * turnover

    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    x0 = prev_weights.copy()

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=[bounds] * n,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000},
    )

    if result.success:
        return result.x
    return np.ones(n) / n
