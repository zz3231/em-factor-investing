"""Industry-level factor portfolio construction and backtesting.

Provides stock selection, portfolio return computation, and full
single-factor / multi-factor backtesting pipelines at the industry level.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data_loader import FACTOR_DIRECTIONS, FACTOR_COLUMNS, SMALL_INDUSTRIES
from .factor_testing import compute_monthly_ic, compute_ic_summary
from .factor_selection import (
    select_single_factor,
    select_composite_factors,
    build_composite_score,
)


def select_top_stocks(
    df: pd.DataFrame,
    score_col: str,
    top_pct: float = 0.2,
    weighting: str = "equal",
) -> pd.DataFrame:
    """Select top-ranked stocks from a single-period, single-industry cross-section.

    Parameters
    ----------
    df : pd.DataFrame
        One month of data for one industry with at least ``score_col``.
    score_col : str
        Column name used for ranking (higher is better).
    top_pct : float
        Fraction of stocks to retain (e.g. 0.2 for top quintile).
    weighting : str
        ``'equal'`` for equal weights or ``'score'`` for score-proportional weights.

    Returns
    -------
    pd.DataFrame
        Subset of ``df`` for selected stocks with an added ``'weight'`` column
        that sums to 1.
    """
    valid = df.dropna(subset=[score_col]).copy()
    if valid.empty:
        return valid.assign(weight=pd.Series(dtype=float))

    n_select = max(1, int(np.ceil(len(valid) * top_pct)))
    top = valid.nlargest(n_select, score_col).copy()

    if weighting == "score":
        scores = top[score_col]
        total = scores.sum()
        if total == 0:
            top["weight"] = 1.0 / len(top)
        else:
            top["weight"] = scores / total
    else:
        top["weight"] = 1.0 / len(top)

    return top


def compute_portfolio_return(
    holdings: pd.DataFrame,
    return_col: str = "mret_bbg",
) -> float:
    """Compute weighted portfolio return for a single period.

    Parameters
    ----------
    holdings : pd.DataFrame
        Must contain ``'weight'`` and ``return_col`` columns.
    return_col : str
        Column with individual stock returns.

    Returns
    -------
    float
        Weighted-average portfolio return, or NaN if no valid data.
    """
    valid = holdings.dropna(subset=["weight", return_col])
    if valid.empty:
        return np.nan
    return float((valid["weight"] * valid[return_col]).sum())


def backtest_single_factor(
    df: pd.DataFrame,
    industry: str,
    factors: list[str] | None = None,
    factor_directions: dict[str, int] | None = None,
    benchmark: pd.Series | None = None,
    window: int = 60,
    top_pct: float | None = None,
    return_col_testing: str = "mret_w",
    return_col_portfolio: str = "mret_bbg",
    date_col: str = "ym",
    id_col: str = "ric",
) -> pd.DataFrame:
    """Full single-factor backtest for one industry with dynamic factor selection.

    At each month *t* (starting from month ``window + 1``):

    1. Compute rolling ``window``-month IC for each candidate factor.
    2. Select the best factor via |IC| with a 20% stability filter.
    3. Rank stocks by the selected factor (direction from sign of rolling IC).
    4. Select top quintile (or tercile for small industries), equal weight.
    5. Record portfolio return at month *t + 1* (using ``return_col_portfolio``).

    No look-ahead bias: signal observed at *t*, return earned at *t + 1*.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data for the given industry.
    industry : str
        Industry label. Used to determine quintile/tercile threshold.
    factors : list[str], optional
        Candidate factor column names. Defaults to ``FACTOR_COLUMNS``.
    factor_directions : dict[str, int], optional
        Per-factor direction mapping. Defaults to ``FACTOR_DIRECTIONS``.
    benchmark : pd.Series, optional
        Benchmark return series (unused in this function but kept for API).
    window : int
        Number of trailing months for IC estimation.
    top_pct : float, optional
        Fraction of stocks to hold. If None, automatically set: 0.33 for
        small industries (HLTHC, RLEST), 0.20 for others.
    return_col_testing : str
        Return column used for IC computation during factor evaluation.
    return_col_portfolio : str
        Return column used for realised portfolio return.
    date_col : str
        Column identifying the time period.
    id_col : str
        Column identifying individual securities.

    Returns
    -------
    pd.DataFrame
        Columns ``['ym', 'return', 'selected_factor', 'n_stocks', 'turnover']``.
    """
    if factors is None:
        factors = FACTOR_COLUMNS
    if factor_directions is None:
        factor_directions = FACTOR_DIRECTIONS
    if top_pct is None:
        top_pct = 0.33 if industry in SMALL_INDUSTRIES else 0.20

    ind_df = df.copy()
    sorted_dates = sorted(ind_df[date_col].unique())

    ic_by_factor: dict[str, pd.Series] = {}
    for fct in factors:
        ic_df = compute_monthly_ic(ind_df, fct, return_col_testing, date_col, method="rank")
        ic_by_factor[fct] = ic_df.set_index("ym")["ic"]

    records: list[dict] = []
    current_selected: str | None = None
    prev_holdings: dict[str, float] = {}

    for t_idx in range(window, len(sorted_dates) - 1):
        t_date = sorted_dates[t_idx]
        t_next = sorted_dates[t_idx + 1]

        lookback_dates = sorted_dates[t_idx - window : t_idx]

        ic_avgs: dict[str, float] = {}
        for fct in factors:
            ic_vals = ic_by_factor[fct].reindex(lookback_dates).dropna()
            avg_ic = ic_vals.mean() if len(ic_vals) > 0 else 0.0
            ic_avgs[fct] = avg_ic if not np.isnan(avg_ic) else 0.0

        current_selected = select_single_factor(
            ic_avgs, current_factor=current_selected
        )

        direction = int(np.sign(ic_avgs[current_selected])) or 1

        cross_section = ind_df[ind_df[date_col] == t_date].copy()
        cross_section["_score"] = direction * cross_section[current_selected]

        holdings = select_top_stocks(cross_section, "_score", top_pct=top_pct)

        curr_holdings = dict(zip(holdings[id_col], holdings["weight"]))
        all_ids = set(prev_holdings) | set(curr_holdings)
        turnover = sum(
            abs(curr_holdings.get(s, 0.0) - prev_holdings.get(s, 0.0))
            for s in all_ids
        ) / 2.0
        prev_holdings = curr_holdings

        next_returns = ind_df[ind_df[date_col] == t_next][[id_col, return_col_portfolio]]
        holdings = holdings.merge(
            next_returns, on=id_col, how="left", suffixes=("", "_next")
        )
        ret_col = (
            return_col_portfolio + "_next"
            if return_col_portfolio + "_next" in holdings.columns
            else return_col_portfolio
        )
        port_ret = compute_portfolio_return(holdings, return_col=ret_col)

        records.append(
            {
                "ym": t_next,
                "return": port_ret,
                "selected_factor": current_selected,
                "n_stocks": len(holdings),
                "turnover": turnover,
            }
        )

    return pd.DataFrame(records)


def backtest_composite_factor(
    df: pd.DataFrame,
    industry: str,
    factors: list[str] | None = None,
    window: int = 60,
    top_pct: float | None = None,
    corr_threshold: float = 0.5,
    weighting: str = "equal",
    min_ic: float = 0.01,
    max_factors: int | None = None,
    return_col_testing: str = "mret_w",
    return_col_portfolio: str = "mret_bbg",
    date_col: str = "ym",
    id_col: str = "ric",
) -> pd.DataFrame:
    """IC-correlation composite factor backtest for one industry.

    At each month *t*:

    1. Compute rolling IC for each factor over the past ``window`` months.
    2. Select factors via greedy IC-correlation screening.
    3. Build a composite score from the selected factors.
    4. Select top quintile (or tercile for small industries), equal weight.
    5. Record portfolio return at month *t + 1*.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data for the given industry.
    industry : str
        Industry label.
    factors : list[str], optional
        Candidate factor column names.
    window : int
        Lookback window in months.
    top_pct : float, optional
        Fraction of stocks to hold.
    corr_threshold : float
        Max pairwise IC correlation for factor inclusion.
        Set > 1.0 to disable the correlation filter.
    weighting : str
        ``'equal'`` or ``'ic_proportional'``.
    min_ic : float
        Minimum |average IC| for a factor to be considered.
        Set to 0.0 to include all factors.
    max_factors : int or None
        Maximum number of factors to include.  None = no limit.
    return_col_testing : str
        Return column for IC computation.
    return_col_portfolio : str
        Return column for realised portfolio return.
    date_col, id_col : str
        Column names.

    Returns
    -------
    pd.DataFrame
        Columns: ``ym``, ``return``, ``selected_factors``, ``n_factors``,
        ``n_stocks``, ``turnover``.
    """
    if factors is None:
        factors = FACTOR_COLUMNS
    if top_pct is None:
        top_pct = 0.33 if industry in SMALL_INDUSTRIES else 0.20

    ind_df = df.copy()
    sorted_dates = sorted(ind_df[date_col].unique())

    ic_by_factor: dict[str, pd.Series] = {}
    for fct in factors:
        ic_df = compute_monthly_ic(
            ind_df, fct, return_col_testing, date_col, method="rank"
        )
        ic_by_factor[fct] = ic_df.set_index("ym")["ic"]

    records: list[dict] = []
    prev_holdings: dict[str, float] = {}

    for t_idx in range(window, len(sorted_dates) - 1):
        t_date = sorted_dates[t_idx]
        t_next = sorted_dates[t_idx + 1]
        lookback_dates = sorted_dates[t_idx - window : t_idx]

        selected = select_composite_factors(
            ic_by_factor, lookback_dates,
            corr_threshold=corr_threshold,
            min_ic=min_ic,
            max_factors=max_factors,
        )

        ic_weights: dict[str, float] = {}
        for fct in factors:
            vals = ic_by_factor[fct].reindex(lookback_dates).dropna()
            ic_weights[fct] = float(vals.mean()) if len(vals) > 0 else 0.0

        cross_section = ind_df[ind_df[date_col] == t_date].copy()
        cross_section["_score"] = build_composite_score(
            cross_section, selected,
            weighting=weighting, ic_weights=ic_weights,
        )

        holdings = select_top_stocks(cross_section, "_score", top_pct=top_pct)

        curr_holdings = dict(zip(holdings[id_col], holdings["weight"]))
        all_ids = set(prev_holdings) | set(curr_holdings)
        turnover = sum(
            abs(curr_holdings.get(s, 0.0) - prev_holdings.get(s, 0.0))
            for s in all_ids
        ) / 2.0
        prev_holdings = curr_holdings

        next_returns = ind_df[ind_df[date_col] == t_next][
            [id_col, return_col_portfolio]
        ]
        holdings = holdings.merge(
            next_returns, on=id_col, how="left", suffixes=("", "_next")
        )
        ret_col = (
            return_col_portfolio + "_next"
            if return_col_portfolio + "_next" in holdings.columns
            else return_col_portfolio
        )
        port_ret = compute_portfolio_return(holdings, return_col=ret_col)

        factor_names = "|".join(f for f, _ in selected)
        records.append(
            {
                "ym": t_next,
                "return": port_ret,
                "selected_factors": factor_names,
                "n_factors": len(selected),
                "n_stocks": len(holdings),
                "turnover": turnover,
            }
        )

    return pd.DataFrame(records)


def backtest_all_industries(
    df: pd.DataFrame,
    benchmark: pd.Series | None = None,
    factors: list[str] | None = None,
    return_col_testing: str = "mret_w",
    return_col_portfolio: str = "mret_bbg",
    window: int = 60,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Run the single-factor backtest across all 11 industries.

    Parameters
    ----------
    df : pd.DataFrame
        Full panel data with an ``'industry'`` column.
    benchmark : pd.Series, optional
        Benchmark return series indexed by date.
    factors : list[str], optional
        Candidate factors. Defaults to ``FACTOR_COLUMNS``.
    return_col_testing : str
        Return column for IC computation.
    return_col_portfolio : str
        Return column for realised portfolio returns.
    window : int
        Rolling lookback window in months.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, pd.DataFrame]]
        1. Wide DataFrame: ``['ym', '<IND>_ret', ...]`` with monthly returns.
        2. Dict mapping industry to its full backtest result DataFrame
           (with selected_factor info).
    """
    from .data_loader import INDUSTRIES

    if factors is None:
        factors = FACTOR_COLUMNS

    industry_results: dict[str, pd.DataFrame] = {}

    for industry in INDUSTRIES:
        ind_df = df[df["industry"] == industry]
        if ind_df.empty:
            continue
        print(f"  Backtesting {industry}...")
        result = backtest_single_factor(
            ind_df,
            industry=industry,
            factors=factors,
            benchmark=benchmark,
            window=window,
            return_col_testing=return_col_testing,
            return_col_portfolio=return_col_portfolio,
        )
        industry_results[industry] = result

    merged = None
    for industry, res in industry_results.items():
        col_name = f"{industry}_ret"
        temp = res[["ym", "return"]].rename(columns={"return": col_name})
        if merged is None:
            merged = temp
        else:
            merged = merged.merge(temp, on="ym", how="outer")

    if merged is not None:
        merged = merged.sort_values("ym").reset_index(drop=True)
    else:
        merged = pd.DataFrame(columns=["ym"])

    return merged, industry_results
