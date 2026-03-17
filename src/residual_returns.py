"""
Beta estimation and residual return computation for EM Industry Factor
Portfolio research.

All computations are strictly backward-looking to avoid lookahead bias.
Rolling OLS betas are computed using only data available up to each date.
"""

import numpy as np
import pandas as pd


def estimate_rolling_beta(
    stock_rets: pd.Series,
    bench_rets: pd.Series,
    window: int = 60,
    min_obs: int = 24,
) -> pd.Series:
    """Compute rolling OLS beta of a single stock against a benchmark.

    Uses an expanding/rolling window of *window* months. For each date,
    only past and current observations are used (no lookahead).

    Parameters
    ----------
    stock_rets:
        Monthly returns for one stock, indexed by date.
    bench_rets:
        Monthly benchmark returns, indexed by date (must cover the
        same date range or broader).
    window:
        Maximum lookback window in months.
    min_obs:
        Minimum number of non-NaN overlapping observations required
        to produce a beta estimate. Dates with fewer observations
        yield NaN.

    Returns
    -------
    pd.Series
        Rolling beta estimates indexed by date. NaN where insufficient
        data is available.
    """
    aligned = pd.DataFrame({"stock": stock_rets, "bench": bench_rets}).dropna()

    if aligned.empty:
        return pd.Series(np.nan, index=stock_rets.index, name="beta")

    betas = pd.Series(np.nan, index=aligned.index, name="beta")

    for i in range(len(aligned)):
        start = max(0, i - window)
        chunk = aligned.iloc[start : i]

        if len(chunk) < min_obs:
            continue

        x = chunk["bench"].values
        y = chunk["stock"].values
        x_dm = x - x.mean()
        denom = (x_dm ** 2).sum()

        if denom == 0:
            continue

        betas.iloc[i] = (x_dm * (y - y.mean())).sum() / denom

    return betas.reindex(stock_rets.index)


def compute_all_betas(
    df: pd.DataFrame,
    benchmark: pd.Series,
    window: int = 60,
    min_obs: int = 24,
    id_col: str = "ric",
    date_col: str = "ym",
    return_col: str = "mret_w",
) -> pd.DataFrame:
    """Compute rolling betas for all stocks in the panel.

    For each unique stock (identified by *id_col*), calls
    :func:`estimate_rolling_beta`. Stocks with NaN beta for a given
    month are filled with the industry-median beta for that month.

    Parameters
    ----------
    df:
        Panel dataframe containing at minimum *id_col*, *date_col*,
        *return_col*, and ``industry``.
    benchmark:
        Monthly benchmark returns (e.g. MSCI EM), indexed by date.
    window:
        Rolling window size for beta estimation.
    min_obs:
        Minimum observations required per stock.
    id_col:
        Column identifying individual stocks.
    date_col:
        Date column name.
    return_col:
        Stock return column to regress on the benchmark.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with a ``beta`` column added. NaN betas are
        filled with industry-month median betas.
    """
    df = df.copy()
    df["beta"] = np.nan

    stock_ids = df[id_col].unique()
    n_total = len(stock_ids)

    for i, stock in enumerate(stock_ids):
        mask = df[id_col] == stock
        stock_rets = df.loc[mask].set_index(date_col)[return_col]

        betas = estimate_rolling_beta(stock_rets, benchmark, window, min_obs)
        df.loc[mask, "beta"] = betas.reindex(df.loc[mask, date_col]).values

        if (i + 1) % 500 == 0:
            print(f"  Beta estimation: {i + 1}/{n_total} stocks processed")

    print(f"  Beta estimation complete: {n_total} stocks")

    n_missing_before = df["beta"].isna().sum()
    industry_median_beta = df.groupby([date_col, "industry"])["beta"].transform("median")
    df["beta"] = df["beta"].fillna(industry_median_beta)
    n_missing_after = df["beta"].isna().sum()

    print(
        f"  Beta NaN fill: {n_missing_before - n_missing_after} filled with "
        f"industry-month median, {n_missing_after} remaining NaN"
    )
    return df


def compute_residual_returns(
    df: pd.DataFrame,
    benchmark: pd.Series,
    date_col: str = "ym",
    return_col: str = "mret_w",
) -> pd.DataFrame:
    """Compute residual returns: stock return minus beta-adjusted benchmark.

    ``resid_ret = return - beta * benchmark_return``

    Parameters
    ----------
    df:
        Panel dataframe with a ``beta`` column (from
        :func:`compute_all_betas`) and *return_col*.
    benchmark:
        Monthly benchmark returns indexed by date.
    date_col:
        Date column name.
    return_col:
        Stock return column.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with a ``resid_ret`` column added.
    """
    df = df.copy()
    bench_map = benchmark.to_dict()
    df["bench_ret"] = df[date_col].map(bench_map)
    df["resid_ret"] = df[return_col] - df["beta"] * df["bench_ret"]
    df.drop(columns=["bench_ret"], inplace=True)
    return df


def aggregate_portfolio_beta(
    stock_betas: pd.Series,
    weights: pd.Series,
) -> float:
    """Compute the weighted-average portfolio beta.

    Parameters
    ----------
    stock_betas:
        Individual stock betas (aligned index with *weights*).
    weights:
        Portfolio weights (should sum to 1, or will be normalised).

    Returns
    -------
    float
        Portfolio beta.
    """
    valid = stock_betas.dropna()
    w = weights.reindex(valid.index).dropna()
    valid = valid.reindex(w.index)

    if w.sum() == 0:
        return np.nan

    w_norm = w / w.sum()
    return float((valid * w_norm).sum())
