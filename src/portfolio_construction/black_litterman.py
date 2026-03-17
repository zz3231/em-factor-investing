"""Black-Litterman portfolio construction with IC-based views."""

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
    """Black-Litterman allocation using trailing mean returns as views.

    Prior: equal-weight implied returns (risk_aversion * Sigma * w_eq).
    Views: trailing mean returns from the lookback window, with
    confidence diagonal Omega proportional to asset variance.

    Parameters
    ----------
    returns_df : pd.DataFrame
        T x N matrix of asset returns.
    cov_matrix : np.ndarray, optional
        N x N covariance matrix.
    mean_returns : np.ndarray, optional
        Unused (views are derived from trailing returns).
    bounds : tuple[float, float], optional
        Weight bounds for the MVO step.  Defaults to (0.02, 0.20).
    **kwargs :
        ``tau`` (float, default 0.05): uncertainty scaling on the prior.
        ``risk_aversion`` (float, default 2.5): implied-return parameter.

    Returns
    -------
    np.ndarray
        N-vector of portfolio weights summing to 1.
    """
    tau = kwargs.get("tau", 0.05)
    risk_aversion = kwargs.get("risk_aversion", 2.5)

    if cov_matrix is None:
        cov_matrix = returns_df.cov().values
    if bounds is None:
        bounds = _DEFAULT_BOUNDS

    n = cov_matrix.shape[0]
    w_eq = np.ones(n) / n
    pi = risk_aversion * cov_matrix @ w_eq

    Q = returns_df.mean().values
    P = np.eye(n)
    omega = np.diag(np.diag(cov_matrix)) * tau

    tau_sigma = tau * cov_matrix
    inv_tau_sigma = np.linalg.inv(tau_sigma)
    inv_omega = np.linalg.inv(omega)

    posterior_cov = np.linalg.inv(inv_tau_sigma + P.T @ inv_omega @ P)
    posterior_mean = posterior_cov @ (inv_tau_sigma @ pi + P.T @ inv_omega @ Q)

    def neg_utility(w):
        ret = w @ posterior_mean
        risk = w @ cov_matrix @ w
        return -(ret - 0.5 * risk_aversion * risk)

    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    x0 = w_eq.copy()

    result = minimize(
        neg_utility,
        x0,
        method="SLSQP",
        bounds=[bounds] * n,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000},
    )

    if result.success:
        return result.x
    return w_eq
