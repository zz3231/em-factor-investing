"""Beta-neutral hedging utilities.

Provides functions to compute portfolio beta, single-period hedged returns,
and rolling hedged return series for beta-neutral strategy evaluation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_portfolio_beta(
    stock_betas: pd.Series,
    weights: pd.Series,
) -> float:
    """Compute the portfolio beta as a weighted average of stock betas.

    Parameters
    ----------
    stock_betas : pd.Series
        Individual stock betas (aligned index with ``weights``).
    weights : pd.Series
        Portfolio weights (should sum to 1).

    Returns
    -------
    float
        Weighted-average portfolio beta.
    """
    aligned_betas, aligned_weights = stock_betas.align(weights, join="inner")
    return float((aligned_betas * aligned_weights).sum())


def hedge_returns(
    port_ret: float,
    bench_ret: float,
    port_beta: float,
) -> float:
    """Compute single-period beta-hedged return.

    Parameters
    ----------
    port_ret : float
        Gross portfolio return for the period.
    bench_ret : float
        Benchmark return for the period.
    port_beta : float
        Portfolio beta used for hedging.

    Returns
    -------
    float
        Hedged return: ``port_ret - port_beta * bench_ret``.
    """
    return port_ret - port_beta * bench_ret


def rolling_hedge(
    port_ret_series: pd.Series,
    bench_ret_series: pd.Series,
    beta_series: pd.Series,
) -> pd.DataFrame:
    """Apply beta-neutral hedging across a time series.

    All three inputs must share a common index (e.g. ``'ym'`` dates).
    Periods with missing data in any input are dropped.

    Parameters
    ----------
    port_ret_series : pd.Series
        Gross portfolio returns over time.
    bench_ret_series : pd.Series
        Benchmark returns over time.
    beta_series : pd.Series
        Portfolio beta estimates over time.

    Returns
    -------
    pd.DataFrame
        Columns ``['ym', 'gross_return', 'hedge_return', 'net_return', 'beta']``
        where ``net_return = gross_return - beta * benchmark_return``.
    """
    combined = pd.DataFrame(
        {
            "gross_return": port_ret_series,
            "bench_return": bench_ret_series,
            "beta": beta_series,
        }
    ).dropna()

    combined["hedge_return"] = combined["beta"] * combined["bench_return"]
    combined["net_return"] = combined["gross_return"] - combined["hedge_return"]

    result = combined[["gross_return", "hedge_return", "net_return", "beta"]].copy()
    result.index.name = "ym"
    return result.reset_index()
