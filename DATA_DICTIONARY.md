# Data Dictionary

## df_ds_signal.csv

Primary dataset. Each row is one stock-month observation for MSCI EM constituents (top 500 by liquidity).

### Identifiers
- `ric`: Reuters Instrument Code (stock identifier)
- `ym`: Year-month date (YYYY-MM-DD, beginning of month)
- `industry`: GICS industry classification (11 industries)
- `country`: Country of domicile

### Return Variables
- `mret_bbg`: Monthly return from Bloomberg (raw, used for portfolio returns)
- `mret_w`: Monthly return, winsorized per-month (used for factor testing/IC)
- `mret_trunc`: Monthly return, truncated full-sample (minor lookahead bias -- avoid)

### Factor Variables (winsorized per-month)
- `pb_w`: Price-to-book ratio (winsorized)
- `roe_w`: Return on equity (winsorized)
- `mom_11m_w`: 11-month momentum (months t-12 to t-2, avoids short-term reversal)
- `ret_vol_w`: Return volatility (winsorized)
- `div_yield_w`: Dividend yield (winsorized)
- `mktcap_w`: Market capitalization (winsorized) -- used to compute log_mktcap

### Derived Columns (computed at load time)
- `log_mktcap`: log(mktcap_w), the size factor

### Other Columns
- `index_rank_lag`: Lagged MSCI EM index rank (filtered to top 500)
- Various `_trunc` columns: Full-sample truncated versions (not used due to lookahead)

## msci_em_index_price.xlsx

MSCI Emerging Markets Index monthly prices. Used to compute benchmark returns and for beta estimation.

## eem_returns_monthly.csv

EEM ETF monthly returns. Alternative benchmark for robustness checks.

## Notes on Data Quality

- All `_w` (winsorized) columns are processed per-month, free of lookahead bias
- All `_trunc` (truncated) columns use full-sample quantiles -- minor lookahead bias
- The universe is filtered to `index_rank_lag < 500` (top 500 MSCI EM constituents)
- Date range: 2004-01 to 2025-03
