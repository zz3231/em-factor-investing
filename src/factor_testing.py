"""Factor IC and IR computation utilities.

Provides functions to compute information coefficients (IC),
information ratios (IR), and related factor evaluation metrics
for cross-sectional equity factor research.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

MIN_OBS = 10


def rank_ic(scores: pd.Series, returns: pd.Series) -> float:
    """Compute Spearman rank correlation between factor scores and returns.

    Parameters
    ----------
    scores : pd.Series
        Cross-sectional factor scores for one period.
    returns : pd.Series
        Corresponding forward returns.

    Returns
    -------
    float
        Spearman rank IC, or NaN if fewer than ``MIN_OBS`` valid pairs.
    """
    mask = scores.notna() & returns.notna()
    s, r = scores[mask], returns[mask]
    if len(s) < MIN_OBS:
        return np.nan
    corr, _ = spearmanr(s, r)
    return float(corr)


def pearson_ic(scores: pd.Series, returns: pd.Series) -> float:
    """Compute Pearson correlation between factor scores and returns.

    Parameters
    ----------
    scores : pd.Series
        Cross-sectional factor scores for one period.
    returns : pd.Series
        Corresponding forward returns.

    Returns
    -------
    float
        Pearson IC, or NaN if fewer than ``MIN_OBS`` valid pairs.
    """
    mask = scores.notna() & returns.notna()
    s, r = scores[mask], returns[mask]
    if len(s) < MIN_OBS:
        return np.nan
    corr, _ = pearsonr(s, r)
    return float(corr)


def compute_monthly_ic(
    df: pd.DataFrame,
    factor_col: str,
    return_col: str,
    date_col: str = "ym",
    method: str = "rank",
) -> pd.DataFrame:
    """Compute cross-sectional IC for each month.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data with at least ``factor_col``, ``return_col``, and ``date_col``.
    factor_col : str
        Column name of the factor scores.
    return_col : str
        Column name of the forward returns.
    date_col : str
        Column identifying the time period (default ``'ym'``).
    method : str
        ``'rank'`` for Spearman or ``'pearson'`` for Pearson correlation.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``['ym', 'ic']``, one row per month.
    """
    ic_func = rank_ic if method == "rank" else pearson_ic
    records: list[dict] = []
    for ym, group in df.groupby(date_col):
        ic_val = ic_func(group[factor_col], group[return_col])
        records.append({"ym": ym, "ic": ic_val})
    return pd.DataFrame(records)


def compute_ic_summary(ic_series: pd.Series) -> dict:
    """Summarise a time series of monthly ICs.

    Parameters
    ----------
    ic_series : pd.Series
        Monthly IC values (NaNs are dropped before computation).

    Returns
    -------
    dict
        Keys: ``avg_ic``, ``std_ic``, ``ir``, ``t_stat``, ``pct_positive``,
        ``n_months``.
    """
    clean = ic_series.dropna()
    n = len(clean)
    if n == 0:
        return {
            "avg_ic": np.nan,
            "std_ic": np.nan,
            "ir": np.nan,
            "t_stat": np.nan,
            "pct_positive": np.nan,
            "n_months": 0,
        }
    avg = clean.mean()
    std = clean.std(ddof=1)
    ir = avg / std if std > 0 else np.nan
    t_stat = avg / (std / np.sqrt(n)) if std > 0 else np.nan
    pct_pos = (clean > 0).mean()
    return {
        "avg_ic": avg,
        "std_ic": std,
        "ir": ir,
        "t_stat": t_stat,
        "pct_positive": pct_pos,
        "n_months": n,
    }


def compute_ic_correlation(ic_dict: dict[str, pd.Series]) -> pd.DataFrame:
    """Compute pairwise correlation of factor IC time series.

    Parameters
    ----------
    ic_dict : dict[str, pd.Series]
        Mapping from factor name to its monthly IC series (indexed by date).

    Returns
    -------
    pd.DataFrame
        Correlation matrix with factor names as both index and columns.
    """
    ic_df = pd.DataFrame(ic_dict)
    return ic_df.corr()


def factor_summary_table(
    df: pd.DataFrame,
    factors: list[str],
    return_col: str,
    date_col: str = "ym",
    industry_col: str = "industry",
    method: str = "rank",
) -> pd.DataFrame:
    """Build a summary table of IC statistics for every factor x industry pair.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data containing factor columns, returns, dates, and industry labels.
    factors : list[str]
        List of factor column names to evaluate.
    return_col : str
        Column name of forward returns.
    date_col : str
        Column identifying the time period.
    industry_col : str
        Column identifying industry membership.
    method : str
        ``'rank'`` or ``'pearson'``.

    Returns
    -------
    pd.DataFrame
        MultiIndex DataFrame indexed by ``(factor, industry)`` with IC summary
        statistics as columns.
    """
    records: list[dict] = []
    for factor in factors:
        for industry, ind_df in df.groupby(industry_col):
            ic_df = compute_monthly_ic(ind_df, factor, return_col, date_col, method)
            summary = compute_ic_summary(ic_df["ic"])
            summary["factor"] = factor
            summary["industry"] = industry
            records.append(summary)

    result = pd.DataFrame(records)
    result = result.set_index(["factor", "industry"])
    return result
