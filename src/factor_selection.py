"""Factor selection and composite score construction.

Provides single-factor selection with stability filtering,
nested cross-validated regularized regression for multi-factor
weight estimation, and rolling multi-factor fitting utilities.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import zscore as _zscore
from sklearn.linear_model import Ridge, ElasticNet


def select_single_factor(
    ic_history: dict[str, float],
    current_factor: str | None = None,
    stability_threshold: float = 0.2,
) -> str:
    """Pick the best factor by absolute average IC, with a stability filter.

    Parameters
    ----------
    ic_history : dict[str, float]
        Mapping from factor name to its average rank IC over the lookback window.
    current_factor : str or None
        Currently selected factor. If provided, switching only occurs when the
        candidate's |IC| exceeds the incumbent by at least ``stability_threshold``
        in relative terms.
    stability_threshold : float
        Minimum relative improvement in |IC| required to trigger a switch
        (e.g. 0.2 means the new factor must be >= 20 % better).

    Returns
    -------
    str
        Name of the selected factor.
    """
    if not ic_history:
        raise ValueError("ic_history is empty; cannot select a factor.")

    best_factor = max(ic_history, key=lambda f: abs(ic_history[f]))

    if current_factor is None or current_factor not in ic_history:
        return best_factor

    current_abs_ic = abs(ic_history[current_factor])
    best_abs_ic = abs(ic_history[best_factor])

    if current_abs_ic == 0:
        return best_factor

    relative_improvement = (best_abs_ic - current_abs_ic) / current_abs_ic
    if relative_improvement >= stability_threshold:
        return best_factor

    return current_factor


def nested_cv_ridge(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    n_lambdas: int = 20,
) -> tuple[float, np.ndarray]:
    """Time-series expanding-window CV for Ridge regression.

    Folds are expanding: fold *k* trains on the first ``k * (n / n_folds)``
    observations and validates on the next block.

    Parameters
    ----------
    X : np.ndarray
        Factor matrix of shape ``(n_obs, n_factors)``.
    y : np.ndarray
        Return vector of shape ``(n_obs,)``.
    n_folds : int
        Number of expanding-window folds.
    n_lambdas : int
        Number of alpha values to search over (log-spaced).

    Returns
    -------
    tuple[float, np.ndarray]
        ``(best_alpha, coefficients_at_best_alpha)`` where coefficients are
        fitted on the full dataset with the chosen regularisation strength.
    """
    n = len(y)
    alphas = np.logspace(-4, 4, n_lambdas)
    block_size = n // (n_folds + 1)

    alpha_scores: dict[float, list[float]] = {a: [] for a in alphas}

    for k in range(1, n_folds + 1):
        train_end = k * block_size
        val_end = min(train_end + block_size, n)
        if val_end <= train_end:
            continue
        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]

        for alpha in alphas:
            model = Ridge(alpha=alpha, fit_intercept=True)
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            mse = float(np.mean((preds - y_val) ** 2))
            alpha_scores[alpha].append(mse)

    avg_mse = {a: np.mean(scores) for a, scores in alpha_scores.items() if scores}
    best_alpha = min(avg_mse, key=avg_mse.get)

    final_model = Ridge(alpha=best_alpha, fit_intercept=True)
    final_model.fit(X, y)
    return best_alpha, final_model.coef_


def nested_cv_elastic_net(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    n_lambdas: int = 20,
    n_l1_ratios: int = 5,
) -> tuple[float, float, np.ndarray]:
    """Time-series expanding-window CV for Elastic Net.

    Searches over a grid of ``alpha`` and ``l1_ratio`` values.

    Parameters
    ----------
    X : np.ndarray
        Factor matrix of shape ``(n_obs, n_factors)``.
    y : np.ndarray
        Return vector of shape ``(n_obs,)``.
    n_folds : int
        Number of expanding-window folds.
    n_lambdas : int
        Number of alpha values to search over (log-spaced).
    n_l1_ratios : int
        Number of l1_ratio values to search over (linearly spaced in (0, 1]).

    Returns
    -------
    tuple[float, float, np.ndarray]
        ``(best_alpha, best_l1_ratio, coefficients)`` fitted on the full
        dataset at the best hyper-parameter combination.
    """
    n = len(y)
    alphas = np.logspace(-4, 4, n_lambdas)
    l1_ratios = np.linspace(0.1, 1.0, n_l1_ratios)
    block_size = n // (n_folds + 1)

    param_scores: dict[tuple[float, float], list[float]] = {
        (a, r): [] for a in alphas for r in l1_ratios
    }

    for k in range(1, n_folds + 1):
        train_end = k * block_size
        val_end = min(train_end + block_size, n)
        if val_end <= train_end:
            continue
        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]

        for alpha in alphas:
            for l1r in l1_ratios:
                model = ElasticNet(
                    alpha=alpha,
                    l1_ratio=l1r,
                    fit_intercept=True,
                    max_iter=10_000,
                )
                model.fit(X_train, y_train)
                preds = model.predict(X_val)
                mse = float(np.mean((preds - y_val) ** 2))
                param_scores[(alpha, l1r)].append(mse)

    avg_mse = {
        k: np.mean(v) for k, v in param_scores.items() if v
    }
    best_params = min(avg_mse, key=avg_mse.get)
    best_alpha, best_l1_ratio = best_params

    final_model = ElasticNet(
        alpha=best_alpha,
        l1_ratio=best_l1_ratio,
        fit_intercept=True,
        max_iter=10_000,
    )
    final_model.fit(X, y)
    return best_alpha, best_l1_ratio, final_model.coef_


def compute_composite_score(
    df: pd.DataFrame,
    factors: list[str],
    weights: dict[str, float],
) -> pd.Series:
    """Compute a weighted z-score composite of multiple factors.

    Each factor is z-scored within the cross-section (the rows of ``df``),
    then multiplied by its weight and summed.

    Parameters
    ----------
    df : pd.DataFrame
        Cross-sectional data containing the factor columns.
    factors : list[str]
        Factor column names to include.
    weights : dict[str, float]
        Mapping from factor name to its weight.

    Returns
    -------
    pd.Series
        Composite score for each row in ``df``.
    """
    composite = pd.Series(0.0, index=df.index)
    for factor in factors:
        raw = df[factor]
        valid = raw.dropna()
        if valid.std() == 0 or len(valid) < 2:
            continue
        z = (raw - valid.mean()) / valid.std()
        composite = composite + weights.get(factor, 0.0) * z
    return composite


def select_composite_factors(
    ic_history: dict[str, pd.Series],
    lookback_dates: list,
    corr_threshold: float = 0.5,
    min_ic: float = 0.01,
    max_factors: int | None = None,
) -> list[tuple[str, int]]:
    """Select factors for an IC-correlation composite.

    Greedily includes factors in descending |IC| order, requiring that
    each new factor's IC time-series has pairwise correlation below
    ``corr_threshold`` with every already-included factor.

    Parameters
    ----------
    ic_history : dict[str, pd.Series]
        Factor name -> monthly IC series (indexed by date).
    lookback_dates : list
        Dates defining the lookback window.
    corr_threshold : float
        Maximum allowed absolute IC correlation for inclusion.
        Set > 1.0 to disable the correlation filter.
    min_ic : float
        Minimum |average IC| to be considered.
        Set to 0.0 to include all factors.
    max_factors : int or None
        Maximum number of factors to include.  None means no limit.

    Returns
    -------
    list[tuple[str, int]]
        ``(factor_name, direction)`` for each included factor.
        Direction is ``+1`` or ``-1`` based on the sign of the average IC.
    """
    avg_ics: dict[str, float] = {}
    for f, ic_series in ic_history.items():
        vals = ic_series.reindex(lookback_dates).dropna()
        avg_ics[f] = float(vals.mean()) if len(vals) > 0 else 0.0

    candidates = {f: ic for f, ic in avg_ics.items() if abs(ic) >= min_ic}
    if not candidates:
        best = max(avg_ics, key=lambda f: abs(avg_ics[f]))
        return [(best, int(np.sign(avg_ics[best])) or 1)]

    ranked = sorted(candidates, key=lambda f: abs(candidates[f]), reverse=True)

    selected = [ranked[0]]
    for f in ranked[1:]:
        if max_factors is not None and len(selected) >= max_factors:
            break
        ic_f = ic_history[f].reindex(lookback_dates).dropna()
        corr_ok = True
        for s in selected:
            ic_s = ic_history[s].reindex(lookback_dates).dropna()
            common = ic_f.index.intersection(ic_s.index)
            if len(common) < 12:
                corr_ok = False
                break
            if abs(ic_f.loc[common].corr(ic_s.loc[common])) >= corr_threshold:
                corr_ok = False
                break
        if corr_ok:
            selected.append(f)

    return [(f, int(np.sign(candidates[f])) or 1) for f in selected]


def build_composite_score(
    cross_section: pd.DataFrame,
    selected_factors: list[tuple[str, int]],
    weighting: str = "equal",
    ic_weights: dict[str, float] | None = None,
) -> pd.Series:
    """Combine direction-adjusted neutralized factors into a single score.

    Parameters
    ----------
    cross_section : pd.DataFrame
        One month of data for one industry.
    selected_factors : list[tuple[str, int]]
        ``(factor_name, direction)`` pairs from :func:`select_composite_factors`.
    weighting : str
        ``'equal'`` or ``'ic_proportional'``.
    ic_weights : dict[str, float] or None
        Average |IC| values used when ``weighting='ic_proportional'``.

    Returns
    -------
    pd.Series
        Composite score (higher is better).
    """
    parts: list[pd.Series] = []
    weights: list[float] = []
    for factor, direction in selected_factors:
        parts.append(direction * cross_section[factor])
        if weighting == "ic_proportional" and ic_weights is not None:
            weights.append(abs(ic_weights.get(factor, 1.0)))
        else:
            weights.append(1.0)

    w_sum = sum(weights)
    if w_sum == 0:
        w_sum = len(weights) or 1.0
        weights = [1.0] * len(parts)
    composite = sum(w / w_sum * s for w, s in zip(weights, parts))
    return composite


def fit_rolling_multi_factor(
    df: pd.DataFrame,
    factors: list[str],
    return_col: str,
    method: str = "ridge",
    window: int = 60,
    date_col: str = "ym",
) -> dict:
    """Estimate factor weights each month via rolling nested CV.

    For each month *t*, uses the preceding ``window`` months of panel data to
    run nested cross-validated regularised regression and extract factor weights.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data with factor columns, returns, and a date column.
    factors : list[str]
        Factor column names.
    return_col : str
        Column name of forward returns.
    method : str
        ``'ridge'`` or ``'elastic_net'``.
    window : int
        Number of trailing months to use for estimation.
    date_col : str
        Column identifying the time period.

    Returns
    -------
    dict
        ``{month: {factor_name: weight, ...}, ...}`` for each month where
        estimation was performed.
    """
    sorted_dates = sorted(df[date_col].unique())
    results: dict = {}

    for i in range(window, len(sorted_dates)):
        current_date = sorted_dates[i]
        lookback_dates = sorted_dates[i - window : i]
        train_df = df[df[date_col].isin(lookback_dates)].dropna(
            subset=factors + [return_col]
        )

        if len(train_df) < window:
            continue

        X = train_df[factors].values
        y = train_df[return_col].values

        if method == "ridge":
            _, coefs = nested_cv_ridge(X, y)
        elif method == "elastic_net":
            _, _, coefs = nested_cv_elastic_net(X, y)
        else:
            raise ValueError(f"Unknown method: {method!r}")

        results[current_date] = dict(zip(factors, coefs))

    return results
