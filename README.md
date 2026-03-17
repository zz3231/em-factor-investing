# EM Industry Factor Portfolio Research

Dynamic factor-based industry portfolios with cross-industry portfolio construction for MSCI Emerging Markets equities.

## Overview

This project constructs long-only factor portfolios within each of 11 GICS industries in the MSCI EM universe, then combines them using various portfolio construction techniques (Equal Weight, Inverse Variance, Minimum Variance, Maximum Sharpe, Risk Parity, and Hierarchical Risk Parity). A beta-neutral hedged variant is also evaluated.

## Project Structure

```
EM/
  data/                     Raw and processed data files
  src/                      Python modules (reusable packages)
    data_loader.py           Data loading and validation
    neutralization.py        Country-demean factor neutralization
    residual_returns.py      Rolling beta estimation, residual returns
    factor_testing.py        IC, IR computation and factor diagnostics
    factor_selection.py      Single-factor and multi-factor selection
    industry_portfolio.py    Industry-level portfolio construction
    hedging.py               Beta-neutral hedging
    performance.py           Performance metrics and plotting
    latex_export.py          LaTeX table/figure/paper generation
    portfolio_construction/  Cross-industry portfolio methods
      equal_weight.py
      inverse_variance.py
      min_variance.py
      max_sharpe.py
      risk_parity.py
      hrp.py
  notebooks/                Jupyter notebooks (experiments)
  output/                   Results (tables, figures, CSV)
  paper/                    LaTeX paper for Overleaf
```

## Timeline

- 2004-2008: Warmup period (rolling windows, beta estimation)
- 2009-2013: Factor portfolio backtesting begins (in-sample for portfolio construction)
- 2014-2025: Full strategy out-of-sample period

## Factors

| Factor | Column | Direction |
|--------|--------|-----------|
| Size | log_mktcap | +1 |
| Value (P/B) | pb_w | -1 |
| Quality (ROE) | roe_w | +1 |
| Momentum | mom_11m_w | +1 |
| Volatility | ret_vol_w | -1 |
| Dividend Yield | div_yield_w | +1 |

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Run notebooks in order: 01 through 07. Each notebook corresponds to one experiment in the research pipeline.
