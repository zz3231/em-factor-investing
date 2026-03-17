"""Mean-CVaR portfolio optimisation via historical simulation."""

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
    """Minimise CVaR (95 %) for a given minimum expected-return target.

    Uses the historical return distribution (no parametric assumption).
    The return target is set to the equal-weight portfolio's mean return
    so the optimiser searches for a lower-tail-risk allocation at
    comparable expected return.

    Parameters
    ----------
    returns_df : pd.DataFrame
        T x N matrix of asset returns.
    cov_matrix : np.ndarray, optional
        Unused (risk measured via empirical CVaR).
    mean_returns : np.ndarray, optional
        Unused.
    bounds : tuple[float, float], optional
        Weight bounds.  Defaults to (0.02, 0.20).
    **kwargs :
        ``alpha`` (float, default 0.05): tail probability for CVaR.

    Returns
    -------
    np.ndarray
        N-vector of portfolio weights summing to 1.
    """
    alpha = kwargs.get("alpha", 0.05)
    if bounds is None:
        bounds = _DEFAULT_BOUNDS

    R = returns_df.values  # T x N
    n = R.shape[1]

    def cvar(w):
        port_rets = R @ w
        cutoff = np.percentile(port_rets, alpha * 100)
        tail = port_rets[port_rets <= cutoff]
        if len(tail) == 0:
            return 0.0
        return -tail.mean()

    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    x0 = np.ones(n) / n

    result = minimize(
        cvar,
        x0,
        method="SLSQP",
        bounds=[bounds] * n,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000},
    )

    if result.success:
        return result.x
    return np.ones(n) / n
