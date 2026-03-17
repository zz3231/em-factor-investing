"""Maximum Sharpe ratio portfolio construction."""

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
    """Maximum Sharpe ratio portfolio: maximise w'mu / sqrt(w'Cov w).

    Parameters
    ----------
    returns_df : pd.DataFrame
        T x N matrix of asset returns.
    cov_matrix : np.ndarray, optional
        N x N covariance matrix. If None, estimated from *returns_df*.
    mean_returns : np.ndarray, optional
        N-vector of expected returns. If None, estimated from *returns_df*.
    bounds : tuple[float, float], optional
        (lower, upper) weight bounds per asset. Defaults to (0.02, 0.20).

    Returns
    -------
    np.ndarray
        N-vector of portfolio weights summing to 1.
    """
    if cov_matrix is None:
        cov_matrix = returns_df.cov().values
    if mean_returns is None:
        mean_returns = returns_df.mean().values

    n = cov_matrix.shape[0]
    if bounds is None:
        bounds = _DEFAULT_BOUNDS

    def neg_sharpe(w):
        port_return = w @ mean_returns
        port_vol = np.sqrt(w @ cov_matrix @ w)
        if port_vol < 1e-12:
            return 0.0
        return -port_return / port_vol

    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
    x0 = np.ones(n) / n

    result = minimize(
        neg_sharpe,
        x0,
        method='SLSQP',
        bounds=[bounds] * n,
        constraints=constraints,
        options={'ftol': 1e-12, 'maxiter': 1000},
    )

    if not result.success:
        raise RuntimeError(f"Max-Sharpe optimisation failed: {result.message}")

    return result.x
