"""Risk-parity portfolio construction."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


_DEFAULT_BOUNDS = (0.02, 0.20)


def _risk_contribution(w: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Marginal risk contribution of each asset: w_i * (Cov @ w)_i / sigma_p."""
    port_var = w @ cov @ w
    port_vol = np.sqrt(port_var)
    marginal = cov @ w
    return (w * marginal) / port_vol


def compute_weights(
    returns_df: pd.DataFrame,
    cov_matrix: np.ndarray | None = None,
    mean_returns: np.ndarray | None = None,
    bounds: tuple[float, float] | None = None,
    **kwargs,
) -> np.ndarray:
    """Risk-parity portfolio: equalise risk contributions across assets.

    Minimise sum_i (RC_i - sigma_p / N)^2 subject to sum(w) = 1 and bounds,
    where RC_i = w_i * (Cov @ w)_i / sigma_p.

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
        port_vol = np.sqrt(w @ cov_matrix @ w)
        rc = _risk_contribution(w, cov_matrix)
        target = port_vol / n
        return np.sum((rc - target) ** 2)

    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
    x0 = np.ones(n) / n

    result = minimize(
        objective,
        x0,
        method='SLSQP',
        bounds=[bounds] * n,
        constraints=constraints,
        options={'ftol': 1e-14, 'maxiter': 2000},
    )

    if not result.success:
        raise RuntimeError(f"Risk-parity optimisation failed: {result.message}")

    return result.x
