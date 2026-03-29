"""
Data loading and validation utilities for EM Industry Factor Portfolio research.

Handles ingestion of stock-level signal data, benchmark index prices,
and EEM ETF returns, with validation and consistency checks.
"""

import os
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Factor definitions
# ---------------------------------------------------------------------------
# Each entry: (column_name, display_name, empirical_direction_in_EM)
# Directions reflect empirical rolling IC signs observed in EM data
# (NOT developed-market textbook priors). These are for documentation
# only — the pipeline always uses dynamic rolling IC signs.
FACTORS: dict[str, tuple[str, str, int]] = {
    "log_mktcap": ("log_mktcap", "Size", +1),
    "pb_w": ("pb_w", "Value (P/B)", +1),       # Growth outperforms in EM
    "roe_w": ("roe_w", "Quality (ROE)", +1),
    "mom_11m_w": ("mom_11m_w", "Momentum", +1),
    "ret_vol_w": ("ret_vol_w", "Volatility", -1),
    "div_yield_w": ("div_yield_w", "Dividend Yield", -1),  # High div underperforms in EM
}

FACTOR_COLUMNS: list[str] = list(FACTORS.keys())
FACTOR_NAMES: dict[str, str] = {k: v[1] for k, v in FACTORS.items()}
FACTOR_DIRECTIONS: dict[str, int] = {k: v[2] for k, v in FACTORS.items()}

INDUSTRIES: list[str] = [
    "BMATR", "CODIS", "COSTP", "ENEGY", "FINAN", "HLTHC",
    "INDUS", "RLEST", "TECNO", "TELCM", "UTILS",
]

SMALL_INDUSTRIES: set[str] = {"HLTHC", "RLEST"}

RETURN_COL_TESTING: str = "mret_w"      # winsorized, for factor testing / IC
RETURN_COL_PORTFOLIO: str = "mret_bbg"   # raw Bloomberg, for portfolio returns

# Country-level round-trip transaction costs (bps) — Domowitz, Glen & Madhavan (2001)
# These are conservative upper-bound estimates from the late 1990s.
COUNTRY_TC_BPS: dict[str, int] = {
    "BRAZIL": 60, "CHILE": 55, "CHINA": 50, "COLOMBIA": 60,
    "CZECH REPUBLIC": 45, "EGYPT": 70, "HONG KONG": 20,
    "HUNGARY": 45, "INDIA": 50, "INDONESIA": 65,
    "ISRAEL": 30, "KUWAIT": 45, "MALAYSIA": 40, "MEXICO": 50,
    "MOROCCO": 55, "PAKISTAN": 50, "PERU": 55, "PHILIPPINES": 70,
    "POLAND": 45, "QATAR": 45, "RUSSIAN FEDERATION": 55,
    "SAUDI ARABIA": 45, "SOUTH AFRICA": 55, "SOUTH KOREA": 35,
    "TAIWAN": 30, "THAILAND": 45, "TURKEY": 50,
    "UNITED ARAB EMIRATES": 45,
}
DEFAULT_TC_BPS: int = 55

# Columns that must exist in the signal CSV (before derived columns)
_REQUIRED_CSV_COLS: list[str] = [
    "ric", "ym", "mktcap_m", "country", "industry",
    RETURN_COL_TESTING, RETURN_COL_PORTFOLIO,
    "pb_w", "roe_w", "mom_11m_w", "ret_vol_w", "div_yield_w",
]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_signal_data(data_dir: str) -> pd.DataFrame:
    """Load the stock-level signal dataset and prepare it for analysis.

    Reads ``df_ds_signal.csv``, parses the ``ym`` column as datetime,
    adds a ``log_mktcap`` derived factor, validates that all required
    columns are present, and sorts by (ric, ym).

    Parameters
    ----------
    data_dir:
        Path to the directory containing ``df_ds_signal.csv``.

    Returns
    -------
    pd.DataFrame
        Cleaned and sorted signal dataframe.
    """
    path = os.path.join(data_dir, "df_ds_signal.csv")
    df = pd.read_csv(path, parse_dates=["ym"])

    missing = [c for c in _REQUIRED_CSV_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in signal data: {missing}")

    df["log_mktcap"] = np.log(df["mktcap_m"].clip(lower=1e-8))

    df.sort_values(["ric", "ym"], inplace=True)
    n_before = len(df)
    df.drop_duplicates(subset=["ric", "ym"], keep="first", inplace=True)
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f"  Dropped {n_dropped} duplicate (ric, ym) rows")
    df.reset_index(drop=True, inplace=True)

    print(
        f"Signal data loaded: {df.shape[0]:,} rows x {df.shape[1]} cols | "
        f"{df['ym'].min():%Y-%m} to {df['ym'].max():%Y-%m}"
    )
    return df


def load_benchmark(data_dir: str) -> pd.Series:
    """Load MSCI EM index prices and compute monthly returns.

    Reads ``msci_em_index_price.xlsx``, identifies the date and price
    columns, resamples to month-end frequency, and computes simple
    returns.

    Parameters
    ----------
    data_dir:
        Path to the directory containing ``msci_em_index_price.xlsx``.

    Returns
    -------
    pd.Series
        Monthly returns indexed by ``ym`` (period-start timestamps
        matching the signal data convention).
    """
    path = os.path.join(data_dir, "msci_em_index_price.xlsx")
    raw = pd.read_excel(path, engine="openpyxl")

    date_col = _find_date_column(raw)
    price_col = _find_price_column(raw, exclude=date_col)

    raw[date_col] = pd.to_datetime(raw[date_col])
    raw = raw.sort_values(date_col).set_index(date_col)
    prices = raw[price_col].dropna()

    monthly_prices = prices.resample("MS").last().dropna()
    monthly_ret = monthly_prices.pct_change().dropna()
    monthly_ret.index.name = "ym"
    monthly_ret.name = "bench_ret"

    print(
        f"Benchmark loaded: {len(monthly_ret)} months | "
        f"{monthly_ret.index.min():%Y-%m} to {monthly_ret.index.max():%Y-%m}"
    )
    return monthly_ret


def load_eem_returns(data_dir: str) -> pd.Series:
    """Load EEM ETF monthly returns.

    Reads ``eem_returns_monthly.csv`` and returns a clean Series of
    monthly returns indexed by ``ym``.

    Parameters
    ----------
    data_dir:
        Path to the directory containing ``eem_returns_monthly.csv``.

    Returns
    -------
    pd.Series
        Monthly EEM returns indexed by ``ym``.
    """
    path = os.path.join(data_dir, "eem_returns_monthly.csv")
    raw = pd.read_csv(path)

    raw["caldt"] = pd.to_datetime(raw["caldt"])
    raw["mret"] = pd.to_numeric(raw["mret"], errors="coerce")
    raw = raw.dropna(subset=["mret"])

    # Align date to month-start to match signal data convention
    raw["ym"] = raw["caldt"].dt.to_period("M").dt.to_timestamp()

    eem = raw.set_index("ym")["mret"].sort_index()
    eem.name = "eem_ret"

    print(
        f"EEM returns loaded: {len(eem)} months | "
        f"{eem.index.min():%Y-%m} to {eem.index.max():%Y-%m}"
    )
    return eem


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_data(df: pd.DataFrame) -> dict[str, Any]:
    """Run basic validation checks on the signal dataframe.

    Parameters
    ----------
    df:
        Signal dataframe (output of :func:`load_signal_data`).

    Returns
    -------
    dict
        Keys: ``shape``, ``date_range``, ``n_stocks``, ``n_industries``,
        ``industries``, ``missing_pct`` (nested dict of per-factor,
        per-industry missing percentages).
    """
    missing_pct: dict[str, dict[str, float]] = {}
    valid_industries = df["industry"].dropna().unique()
    for factor in FACTOR_COLUMNS:
        missing_pct[factor] = {}
        for ind in valid_industries:
            mask = df["industry"] == ind
            pct = df.loc[mask, factor].isna().mean() * 100
            missing_pct[factor][ind] = round(pct, 2)

    result = {
        "shape": df.shape,
        "date_range": (df["ym"].min(), df["ym"].max()),
        "n_stocks": df["ric"].nunique(),
        "n_industries": df["industry"].nunique(),
        "industries": sorted(df["industry"].dropna().unique().tolist()),
        "missing_pct": missing_pct,
    }
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_date_column(df: pd.DataFrame) -> str:
    """Heuristically identify the date column in an Excel dataframe."""
    for col in df.columns:
        if "date" in str(col).lower():
            return col
    for col in df.columns:
        try:
            pd.to_datetime(df[col].head(20))
            return col
        except (ValueError, TypeError):
            continue
    raise ValueError(f"Cannot identify date column among: {df.columns.tolist()}")


def _find_price_column(df: pd.DataFrame, exclude: str) -> str:
    """Heuristically identify the price / level column."""
    for col in df.columns:
        if col == exclude:
            continue
        if any(kw in str(col).lower() for kw in ("price", "close", "level", "index")):
            return col
    numeric_cols = [c for c in df.select_dtypes(include="number").columns if c != exclude]
    if numeric_cols:
        return numeric_cols[0]
    raise ValueError(f"Cannot identify price column among: {df.columns.tolist()}")
