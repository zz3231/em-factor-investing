# Data Dictionary

## df_ds_signal.csv

Primary dataset. Each row is one stock-month observation for MSCI EM constituents (top 500 by market cap). 98,742 observations, 953 unique stocks, 36 countries (27 standard EM + 9 non-EM domiciles for cross-listed stocks), 11 GICS industries, 255 months (2004-01 to 2025-03). 894 rows have missing country.

### Identifiers
- `ric`: Reuters Instrument Code (stock identifier, e.g., `0700.HK` for Tencent)
- `ym`: Year-month date (YYYY-MM-DD, beginning of month)
- `industry`: GICS industry classification (11 industries: FINAN, CODIS, INDUS, TECNO, COSTP, BMATR, ENEGY, TELCM, UTILS, RLEST, HLTHC)
- `country`: Country of domicile (e.g., CHINA, SOUTH KOREA, INDIA, TAIWAN, BRAZIL)

### Return Variables
- `mret_bbg`: Monthly total return from Bloomberg (includes dividends/corporate actions). **Primary return variable** used for portfolio return computation.
- `mret`: Monthly USD total return from Datastream. Used for factor testing (IC computation).
- `mret_w`: Monthly return, winsorized per-month at 1st/95th percentiles. Used for factor testing/IC.
- `mret_trunc`: Monthly return, truncated using full-sample quantiles. **Not used** (minor lookahead bias).

### Price and Market Data
- `prc`: Local-currency stock price from Datastream
- `mktcap_m`: Market capitalization in millions (local currency × shares outstanding)
- `mktcap_w`: Market capitalization, winsorized per-month
- `total_ret_index`: Datastream local-currency total return index (includes dividends, in local currency). Used for FX decomposition to isolate local stock returns.
- `fx_rate`: Exchange rate (USD per local currency unit) from Datastream. Used to compute FX return component: `fx_ret = fx_rate(t) / fx_rate(t-1) - 1`.
- `exchange_currency_name`: Currency code for the stock's trading currency

### Volume and Liquidity
- `share_turnover_t`: Monthly trading volume from Datastream (denominator VO). Reported in **shares** (not thousands). Used to compute ADV: `ADV_USD = share_turnover_t × prc × fx_rate / 21` (daily average over ~21 trading days).
- `shrout_t`: Shares outstanding from Datastream (denominator NOSH). Reported in **thousands of shares**.

### Factor Variables (winsorized per-month)
- `pb_w`: Price-to-book ratio, winsorized at 1st/99th percentiles
- `roe_w`: Return on equity (EPS ÷ book value per share), winsorized. Accounting data lagged one quarter to avoid lookahead.
- `mom_11m_w`: 11-month momentum (cumulative return from t-12 to t-2, skipping most recent month to avoid short-term reversal), winsorized
- `ret_vol_w`: Return volatility (std dev of monthly returns over trailing 24 months), winsorized
- `div_yield_w`: Dividend yield, winsorized
- `div_yield`: Raw dividend yield (percentage points, not winsorized). Used in NB12 dividend analysis.

### Derived Columns (computed at load time by `data_loader.py`)
- `log_mktcap`: log(mktcap_w), the size factor

### Other Columns
- `index_rank_lag`: Lagged MSCI EM index rank (filtered to top 500)
- Various `_trunc` columns: Full-sample truncated versions (not used due to lookahead bias)

---

## msci_em_index_price.xlsx

MSCI Emerging Markets Index monthly **price-level** data. Converted to monthly returns for beta estimation and benchmarking. **Important:** This is a price-return index (excludes dividends). Used as secondary benchmark; see NB09 for comparison with EEM.

**Date handling:** Raw dates may include NaT entries and end-of-month timestamps. The loading code drops NaT, sorts chronologically, converts to start-of-month via `.to_period('M').to_timestamp()`, and deduplicates.

---

## eem_returns_monthly.csv

iShares MSCI Emerging Markets ETF (EEM) monthly **total returns** from CRSP (includes dividends/distributions). Used as the **primary hedging instrument** in NB09 and as the main benchmark throughout NB07-NB10.

Columns:
- `date`: Month-end date
- `mret`: Monthly total return (decimal, e.g., 0.05 = 5%)

---

## Key Output Files

### Strategy Returns
- `output/csv/exp4_all_portfolio_returns.csv`: All strategy returns (10 PC methods × 8 factor composites)
- `output/csv/nb08_hedged_returns.csv`: Hedged returns, betas, EEM/MSCI data for holdout period
- `output/csv/nb08_to_mvo_weights.csv`: Monthly TO_MVO industry allocation weights

### Industry-Level Data
- `output/csv/composite_all6_ew_industry_returns.csv`: All6-EW factor portfolio returns per industry
- `output/csv/single_factor_raw_industry_returns.csv`: Single-factor portfolio returns per industry

### Performance Summaries
- `output/csv/exp3b_turnovers.csv`: Stock-level turnover by factor construction method
- `output/csv/exp4_multicriteria_scores.csv`: Multi-criteria strategy scoring (Sharpe, stability, TC, capacity, parsimony)
- `output/csv/nb10_performance_summary.csv`: Comprehensive holdout performance summary

### Capacity and TC Analysis
- `output/csv/nb11_capacity_analysis.csv`: Capacity at various ADV thresholds by month
- `output/csv/nb11_tc_decomposition.csv`: TC breakdown by component and country
- `output/csv/nb11_turnover_decomposition.csv`: Stock-level vs allocation-level turnover

### FX and Hedging
- `output/csv/nb13_fx_decomposition_monthly.csv`: Monthly local/FX/USD return decomposition
- `output/csv/nb13_fx_decomposition_annual.csv`: Annual FX decomposition
- `output/csv/nb13_country_fx_summary.csv`: Country-level FX contribution

### Factor Model Analysis
- `output/csv/nb16_factor_model_results.csv`: Full-sample FF5+Mom regression results (alpha, betas, R² for all strategies)
- `output/csv/nb17_factor_model_results_oos.csv`: OOS-only FF5+Mom regression results (2019-2024)
- `output/csv/nb18_enhanced_hedged_returns.csv`: Hedged returns under multiple beta estimation windows (24m, 36m, 48m, 60m, blended)

### Factor Data
- `data/em_ff6_factors.csv`: Fama-French Emerging Markets 5 Factors + Momentum (monthly, from Ken French's Data Library). Columns: `Mkt-RF`, `SMB`, `HML`, `RMW`, `CMA`, `RF`, `Mom`. Used in NB16-18 for factor model regressions.
- `data/us_ff6_factors.csv`: US Fama-French 5 Factors + Momentum (monthly, from Ken French's Data Library). 751 months from 1963-07 to 2026-01. Columns: `Mkt-RF`, `SMB`, `HML`, `RMW`, `CMA`, `RF`, `Mom`. Returns in decimal (e.g., 0.05 = 5%). Used in NB19 for US factor regressions and diversification analysis.

### US Diversification Analysis (NB19)
- `output/csv/nb19_alpha_results.csv`: US factor regression results (alpha, t-stat, R², hit rate) for all strategies
- `output/csv/nb19_optimal_allocation.csv`: Efficient frontier data (EM weight, return, vol, Sharpe)
- `output/csv/nb19_summary.csv`: Master summary of diversification metrics (hurdle Sharpe, alpha, correlation)

---

## Key Constants (in `src/data_loader.py`)

### Factor Directions
```python
FACTOR_DIRECTIONS = {
    'log_mktcap': +1,   # Larger firms outperform in EM
    'pb_w': +1,          # Growth (high P/B) outperforms
    'roe_w': +1,         # Quality premium
    'mom_11m_w': +1,     # Momentum
    'ret_vol_w': -1,     # Low-vol anomaly
    'div_yield_w': -1,   # Low-dividend outperforms
}
```

### Country Transaction Cost Map (bps, round-trip)
Based on Domowitz et al. (2001). Weighted average across portfolio countries ≈ 45 bps. These estimates are from late-1990s data and are deliberately conservative; current EM costs are likely 20–30% lower.

```python
COUNTRY_TC_BPS = {
    'BRAZIL': 55, 'CHINA': 50, 'INDIA': 45, 'SOUTH KOREA': 35,
    'TAIWAN': 25, 'SOUTH AFRICA': 40, 'MEXICO': 35, 'INDONESIA': 55,
    'THAILAND': 45, 'RUSSIA': 60, 'TURKEY': 50, ...
}
DEFAULT_TC_BPS = 50  # for unlisted countries
```

---

## Notes on Data Quality

- All `_w` (winsorized) columns are processed per-month cross-sectionally, free of lookahead bias
- All `_trunc` (truncated) columns use full-sample quantiles — minor lookahead bias, not used in final pipeline
- The universe is filtered to `index_rank_lag < 500` (top 500 MSCI EM constituents by market cap)
- Date range: 2004-01 to 2025-03 (255 months)
- Missing factor rates are low (<4% for momentum/volatility, <1% for others); imputed with industry-month median
- Country-level TC estimates are from Domowitz et al. (2001) and are deliberately conservative
- ADV computed as `share_turnover_t × prc × fx_rate / 21` (monthly volume ÷ 21 trading days, converted to USD)

---

## Holdings Diagnostic (NB20)

### Output CSVs

| File | Description |
|------|-------------|
| `nb20_stock_holdings.csv` | Reconstructed stock-month holdings panel (date, ric, name, country, industry, weight, return, composite score, factor values) |
| `nb20_stock_contributions.csv` | Per-stock cumulative return contributions (total contribution, avg weight, months held, avg return) |
| `nb20_2023_diagnostic.csv` | 2023 crash diagnostic summary (IC, industry attribution, hedge timing) |

### Output Figures

| Figure | Description |
|--------|-------------|
| `nb20_validation.pdf` | Holdings reconstruction validation vs known A_long returns |
| `nb20_alpha_timeline.pdf` | Monthly hedged alpha, cumulative alpha, and active LO return |
| `nb20_rolling_ic.pdf` | All6-EW composite IC over time and annual averages |
| `nb20_industry_waterfall.pdf` | Industry contribution waterfall for the worst year |
| `nb20_hedge_timing.pdf` | Rolling beta, hedge cost, and ideal vs actual beta |
| `nb20_factor_regime.pdf` | EM factor returns by year and cumulative |
| `nb20_stock_contributors.pdf` | Top 20 and bottom 20 stocks by return contribution |
| `nb20_concentration.pdf` | Return concentration by country and industry |
| `nb20_characteristics.pdf` | Factor characteristics of winners vs losers |
| `nb20_cma_factor.pdf` | CMA factor cumulative returns and correlations |
| `nb20_cma_decomposition.pdf` | CMA loading decomposition by single-factor portfolio |
| `nb20_cma_rolling.pdf` | Rolling 24m CMA loading over time |
