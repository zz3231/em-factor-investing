"""Performance measurement, statistics, and plotting for portfolio backtests."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Matplotlib defaults for publication-quality figures
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'figure.figsize': (10, 5),
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

_MONTHS_PER_YEAR = 12


# ===================================================================
# Core analytics
# ===================================================================

def max_drawdown(returns: pd.Series) -> float:
    """Maximum drawdown (as a positive fraction) from peak to trough.

    Parameters
    ----------
    returns : pd.Series
        Time series of periodic (e.g. monthly) simple returns.

    Returns
    -------
    float
        Maximum drawdown expressed as a positive number in [0, 1].
    """
    cum = (1 + returns).cumprod()
    running_max = cum.cummax()
    dd = (cum - running_max) / running_max
    return -dd.min()


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Drawdown at each point in time (non-positive values).

    Parameters
    ----------
    returns : pd.Series
        Time series of periodic simple returns.

    Returns
    -------
    pd.Series
        Drawdown series with same index as *returns*.
    """
    cum = (1 + returns).cumprod()
    running_max = cum.cummax()
    return (cum - running_max) / running_max


def performance_table(
    returns_dict: dict[str, pd.Series],
    rf: float = 0.0,
    benchmark: pd.Series | None = None,
) -> pd.DataFrame:
    """Compute a table of summary statistics for multiple return series.

    Parameters
    ----------
    returns_dict : dict[str, pd.Series]
        Mapping of strategy name to monthly return series.
    rf : float
        Annualised risk-free rate used in Sharpe and Calmar calculations.
    benchmark : pd.Series, optional
        Benchmark return series for IR calculation.  When provided, each
        strategy gets an ``IR`` column = (ann excess return) / (ann TE).

    Returns
    -------
    pd.DataFrame
        Rows = strategies, columns = performance metrics.
    """
    records: list[dict[str, Any]] = []

    for name, rets in returns_dict.items():
        rets = rets.dropna()
        n_months = len(rets)
        ann_mean = rets.mean() * _MONTHS_PER_YEAR
        ann_std = rets.std() * np.sqrt(_MONTHS_PER_YEAR)
        sharpe = (ann_mean - rf) / ann_std if ann_std > 0 else np.nan
        mdd = max_drawdown(rets)
        calmar = (ann_mean - rf) / mdd if mdd > 0 else np.nan
        total_ret = (1 + rets).prod() - 1
        pct_pos = (rets > 0).mean()

        ir = np.nan
        if benchmark is not None:
            common = rets.index.intersection(benchmark.dropna().index)
            if len(common) > 1:
                excess = rets.loc[common] - benchmark.loc[common]
                te = excess.std() * np.sqrt(_MONTHS_PER_YEAR)
                if te > 0:
                    ir = excess.mean() * _MONTHS_PER_YEAR / te

        records.append({
            'Strategy': name,
            'Ann. Mean': ann_mean,
            'Ann. Std': ann_std,
            'Sharpe': sharpe,
            'IR': ir,
            'Max DD': mdd,
            'Calmar': calmar,
            '% Positive': pct_pos,
            'Total Return': total_ret,
            'N Months': n_months,
        })

    return pd.DataFrame(records).set_index('Strategy')


def is_vs_oos_table(
    returns_dict: dict[str, pd.Series],
    split_date: str,
) -> pd.DataFrame:
    """Compute performance separately for in-sample and out-of-sample periods.

    Parameters
    ----------
    returns_dict : dict[str, pd.Series]
        Mapping of strategy name to monthly return series.
    split_date : str
        Date string (e.g. ``'2020-01-01'``) separating IS from OOS.

    Returns
    -------
    pd.DataFrame
        Multi-level columns: (IS / OOS) x metrics.
    """
    split = pd.Timestamp(split_date)
    is_dict = {k: v[v.index < split] for k, v in returns_dict.items()}
    oos_dict = {k: v[v.index >= split] for k, v in returns_dict.items()}

    is_table = performance_table(is_dict).add_prefix('IS ')
    oos_table = performance_table(oos_dict).add_prefix('OOS ')

    return pd.concat([is_table, oos_table], axis=1)


# ===================================================================
# Plotting helpers
# ===================================================================

def _save_or_show(fig: plt.Figure, save_path: str | None) -> None:
    """Save figure as PDF if *save_path* is given, then display inline."""
    if save_path is not None:
        fig.savefig(save_path, format='pdf')
    plt.show()
    plt.close(fig)


def plot_cumulative_returns(
    returns_dict: dict[str, pd.Series],
    title: str = '',
    save_path: str | None = None,
) -> plt.Figure:
    """Plot cumulative wealth curves for one or more return series.

    Parameters
    ----------
    returns_dict : dict[str, pd.Series]
        Mapping of strategy name to monthly return series.
    title : str
        Plot title.
    save_path : str, optional
        If provided, save the figure as a PDF at this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots()
    for name, rets in returns_dict.items():
        cum = (1 + rets).cumprod()
        ax.plot(cum.index, cum.values, label=name, linewidth=1.2)
    ax.set_ylabel('Cumulative Return')
    ax.set_xlabel('Date')
    ax.set_title(title or 'Cumulative Returns')
    ax.legend()
    fig.tight_layout()
    _save_or_show(fig, save_path)
    return fig


def plot_correlation_heatmap(
    returns_df: pd.DataFrame,
    title: str = '',
    save_path: str | None = None,
) -> plt.Figure:
    """Seaborn heatmap of the correlation matrix with annotations.

    Parameters
    ----------
    returns_df : pd.DataFrame
        T x N return matrix whose columns are assets / strategies.
    title : str
        Plot title.
    save_path : str, optional
        If provided, save the figure as a PDF.

    Returns
    -------
    matplotlib.figure.Figure
    """
    corr = returns_df.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        corr,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        ax=ax,
    )
    ax.set_title(title or 'Correlation Matrix')
    fig.tight_layout()
    _save_or_show(fig, save_path)
    return fig


def plot_rolling_sharpe(
    returns_dict: dict[str, pd.Series],
    window: int = 24,
    title: str = '',
    save_path: str | None = None,
) -> plt.Figure:
    """Rolling annualised Sharpe ratio.

    Parameters
    ----------
    returns_dict : dict[str, pd.Series]
        Mapping of strategy name to monthly return series.
    window : int
        Rolling window in months (default 24).
    title : str
        Plot title.
    save_path : str, optional
        If provided, save the figure as a PDF.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots()
    for name, rets in returns_dict.items():
        rolling_mean = rets.rolling(window).mean() * _MONTHS_PER_YEAR
        rolling_std = rets.rolling(window).std() * np.sqrt(_MONTHS_PER_YEAR)
        rolling_sr = rolling_mean / rolling_std
        ax.plot(rolling_sr.index, rolling_sr.values, label=name, linewidth=1.2)
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax.set_ylabel('Rolling Sharpe Ratio')
    ax.set_xlabel('Date')
    ax.set_title(title or f'Rolling {window}-Month Sharpe Ratio')
    ax.legend()
    fig.tight_layout()
    _save_or_show(fig, save_path)
    return fig


def plot_drawdown(
    returns_dict: dict[str, pd.Series],
    title: str = '',
    save_path: str | None = None,
) -> plt.Figure:
    """Drawdown over time for one or more return series.

    Parameters
    ----------
    returns_dict : dict[str, pd.Series]
        Mapping of strategy name to monthly return series.
    title : str
        Plot title.
    save_path : str, optional
        If provided, save the figure as a PDF.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots()
    for name, rets in returns_dict.items():
        dd = drawdown_series(rets)
        ax.fill_between(dd.index, dd.values, 0, alpha=0.25, label=name)
        ax.plot(dd.index, dd.values, linewidth=0.8)
    ax.set_ylabel('Drawdown')
    ax.set_xlabel('Date')
    ax.set_title(title or 'Drawdowns')
    ax.legend()
    fig.tight_layout()
    _save_or_show(fig, save_path)
    return fig


def plot_weight_bars(
    weights_dict: dict[str, np.ndarray],
    labels: list[str],
    title: str = '',
    save_path: str | None = None,
) -> plt.Figure:
    """Grouped bar chart of average weight allocations across strategies.

    Parameters
    ----------
    weights_dict : dict[str, np.ndarray]
        Mapping of strategy name to weight vector.
    labels : list[str]
        Asset / ticker names corresponding to each weight element.
    title : str
        Plot title.
    save_path : str, optional
        If provided, save the figure as a PDF.

    Returns
    -------
    matplotlib.figure.Figure
    """
    df = pd.DataFrame(weights_dict, index=labels)
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.6), 5))
    df.plot.bar(ax=ax, width=0.8)
    ax.set_ylabel('Weight')
    ax.set_xlabel('Asset')
    ax.set_title(title or 'Portfolio Weights')
    ax.legend(title='Strategy')
    ax.set_xticklabels(labels, rotation=45, ha='right')
    fig.tight_layout()
    _save_or_show(fig, save_path)
    return fig
