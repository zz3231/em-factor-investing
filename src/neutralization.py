"""
Factor neutralization utilities for EM Industry Factor Portfolio research.

Provides country-demeaning, cross-sectional re-standardization, and
median imputation to ensure factors are comparable across countries
within each industry-month group.
"""

import numpy as np
import pandas as pd


def country_demean(df: pd.DataFrame, factor_col: str) -> pd.Series:
    """Subtract the country mean from a factor within a single industry-month.

    Parameters
    ----------
    df:
        Cross-section already filtered to one industry and one month.
        Must contain a ``country`` column and the target *factor_col*.
    factor_col:
        Name of the factor column to demean.

    Returns
    -------
    pd.Series
        Demeaned factor values, preserving the original index.
    """
    country_means = df.groupby("country")[factor_col].transform("mean")
    return df[factor_col] - country_means


def neutralize_cross_section(
    df: pd.DataFrame,
    factors: list[str],
    date_col: str = "ym",
    industry_col: str = "industry",
    country_col: str = "country",
) -> pd.DataFrame:
    """Country-neutralize and re-standardize factors within each industry-month.

    For every (industry, month) group and every factor in *factors*:

    1. Subtract the country mean (country-demean).
    2. Re-standardize to zero mean and unit variance within the group.

    New columns are named ``{factor}_neutral``.

    Parameters
    ----------
    df:
        Full panel dataframe with columns for *date_col*, *industry_col*,
        *country_col*, and all entries in *factors*.
    factors:
        List of factor column names to neutralize.
    date_col:
        Name of the date column.
    industry_col:
        Name of the industry column.
    country_col:
        Name of the country column.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with ``{factor}_neutral`` columns appended.
    """
    df = df.copy()
    group_keys = [date_col, industry_col]

    for factor in factors:
        neutral_col = f"{factor}_neutral"
        df[neutral_col] = np.nan

        for _name, group in df.groupby(group_keys):
            idx = group.index
            demeaned = country_demean(group, factor)

            std = demeaned.std()
            if std == 0 or np.isnan(std):
                z = demeaned * 0.0
            else:
                z = (demeaned - demeaned.mean()) / std

            df.loc[idx, neutral_col] = z

    return df


def impute_median(
    df: pd.DataFrame,
    factor_col: str,
    group_cols: list[str] | None = None,
) -> pd.Series:
    """Fill NaN values with the cross-sectional median within each group.

    Parameters
    ----------
    df:
        Dataframe containing *factor_col* and the grouping columns.
    factor_col:
        Column whose missing values should be filled.
    group_cols:
        Columns defining the cross-sectional groups. Defaults to
        ``['ym', 'industry']``.

    Returns
    -------
    pd.Series
        Factor series with NaN values replaced by group medians.
        If a group is entirely NaN, values remain NaN.
    """
    if group_cols is None:
        group_cols = ["ym", "industry"]

    medians = df.groupby(group_cols)[factor_col].transform("median")
    return df[factor_col].fillna(medians)
