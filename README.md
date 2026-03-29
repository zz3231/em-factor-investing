# EM Industry Factor Portfolio Research

Dynamic factor-based industry portfolios with cross-industry portfolio construction and market-neutral hedging for MSCI Emerging Markets equities.

## Overview

This project constructs and evaluates a systematic multi-factor strategy for EM equities using six stock-level characteristics across eleven GICS-based industries over 2004–2025. The pipeline covers factor analysis, stock selection, cross-industry allocation, dynamic beta hedging, and comprehensive implementation analysis (capacity, transaction costs, FX decomposition).

**Key result:** The selected strategy (**All6-EW** factor composite + **TO_MVO** allocation + **EEM beta hedge**) delivers:
- Long-only holdout Sharpe: 0.607 (gross), 0.552 (net of 45bp TC)
- Hedged holdout Sharpe: 0.828 (gross), 0.673 (net of 45bp TC)
- Jensen's Alpha: 5.17%/yr (gross), 4.13%/yr (net)
- Maximum drawdown (hedged): ~7%
- Strategy capacity: ~$10M practical ($51M median) at 10% ADV participation

**Experimental design:** True out-of-sample with no data snooping:
- 2004–2008: Warmup (rolling windows, beta estimation)
- 2009–2013: Factor construction selection (→ All6-EW)
- 2014–2018: Portfolio construction selection (→ TO_MVO)
- 2019–2025: True holdout (never touched during model selection)

## Project Structure

```
EM/
  data/                         Raw and processed data files
    df_ds_signal.csv              Primary stock-month dataset (98,742 obs)
    msci_em_index_price.xlsx      MSCI EM Index price data (benchmark)
    eem_returns_monthly.csv       EEM ETF total returns (hedge instrument)
    em_ff6_factors.csv            EM Fama-French 5 Factors + Momentum (Ken French)
    us_ff6_factors.csv            US Fama-French 5 Factors + Momentum (Ken French)
  src/                          Python modules (reusable packages)
    data_loader.py                Data loading, factor definitions, country TC map
    neutralization.py             Country-demean factor neutralization
    residual_returns.py           Rolling beta estimation, residual returns
    factor_testing.py             IC, IR computation and factor diagnostics
    factor_selection.py           Single-factor and multi-factor selection
    industry_portfolio.py         Industry-level portfolio construction
    hedging.py                    Beta-neutral hedging
    performance.py                Performance metrics and plotting
    latex_export.py               LaTeX table/figure/paper generation
    portfolio_construction/       Cross-industry portfolio methods (10 methods)
      equal_weight.py
      inverse_variance.py
      min_variance.py
      max_sharpe.py
      risk_parity.py
      hrp.py
      momentum_weight.py
      black_litterman.py
      mean_cvar.py
      turnover_penalized.py       TO_MVO (turnover-penalized mean-variance)
  notebooks/                    Jupyter notebooks (run in order)
    01_data_validation.ipynb      Universe, coverage, factor distributions
    02_baseline_portfolios.ipynb  EW baseline industry portfolios
    03_ic_analysis.ipynb          IC analysis, raw vs residual, rolling IC
    04_single_factor_portfolios.ipynb   Single-factor industry portfolios
    05_composite_factor.ipynb     Multi-factor composition (8 variants, 2009-2013)
    06_portfolio_construction.ipynb     10 PC methods (2014-2018 dev period)
    06b_parameter_sensitivity.ipynb     TO_MVO parameter tuning (TSCV)
    07_holdout_long_only.ipynb    True holdout long-only (2019-2025)
    08_long_only_attribution.ipynb      Long-only attribution (TO_MVO)
    08b_long_only_attribution_momentum.ipynb  Long-only attribution (Momentum)
    09_hedging.ipynb              Dynamic beta hedging with EEM
    10_hedged_attribution.ipynb   Hedged attribution (TO_MVO)
    10b_hedged_attribution_momentum.ipynb  Hedged attribution (Momentum)
    11_capacity_analysis.ipynb    ADV capacity & Sharpe degradation (TO_MVO)
    11b_capacity_momentum.ipynb   Capacity comparison: TO_MVO vs Momentum
    12_transaction_cost_deep_dive.ipynb  Granular TC decomposition
    13_fx_decomposition.ipynb     FX overlay, local vs currency returns
    14_country_universe_exploration.ipynb  Country dimension analysis
    15_strategy_comparison.ipynb  Head-to-head: TO_MVO vs Momentum (dollar P&L)
    16_factor_model_analysis.ipynb  Full-sample factor model (FF5+Mom) regressions
    17_factor_model_oos.ipynb     OOS-only factor model analysis (2019-2024)
    18_enhanced_hedge_analysis.ipynb  Multi-window beta hedging & cumulative return comparison
    19_us_diversification_analysis.ipynb  US portfolio diversification value analysis
  output/
    csv/                        Intermediate and final result CSVs
    figures/                    All plots (PDF)
    latex/                      LaTeX tables
  paper/
    main.tex                    Detailed research paper (original)
    main_v2.tex                 Comprehensive paper with all analyses
    main_v3.tex                 Full comprehensive paper (94pp, all 97 figures, from scratch)
    main_v4.tex                 Publishable-quality paper (67pp, 33 main-text figs, story-arc structure)
    paper_academic.tex          Concise academic version (journal format)
    references.bib              Bibliography
```

## Research Pipeline

| Stage | Notebook | Period | Description |
|-------|----------|--------|-------------|
| 1. Data | 01, 02 | 2004–2025 | Universe: 953 stocks, 36 countries, 11 industries, 255 months |
| 2. Factor Testing | 03 | 2004–2025 | Rank IC analysis for 6 factors across 11 industries |
| 3. Single Factor | 04 | 2009–2013 | Best single factor per industry via rolling 60m IC |
| 4. Multi-Factor | 05 | 2009–2013 | 8 composition variants; All6-EW selected (zero DoF, diversified) |
| 5. Portfolio Construction | 06, 06b | 2014–2018 | 10 allocation methods; TO_MVO selected (highest TC-adjusted Sharpe) |
| 6. Long-Only Holdout | 07 | 2019–2025 | TO_MVO vs Momentum comparison on true OOS |
| 7. Long-Only Attribution | 08, 08b | 2019–2025 | Sharpe/Treynor/Jensen/Appraisal, style analysis (TO_MVO & Momentum) |
| 8. Beta Hedging | 09 | 2019–2025 | Rolling 60m OLS hedge vs EEM ETF |
| 9. Hedged Attribution | 10, 10b | 2019–2025 | Hedge ratio dynamics, sub-period stability (TO_MVO & Momentum) |
| 10. Capacity | 11, 11b | 2019–2025 | ADV-based capacity, Sharpe degradation comparison |
| 11. TC Deep Dive | 12 | 2019–2025 | Country-weighted, time-varying, decomposed TC |
| 12. FX Decomposition | 13 | 2019–2025 | Local stock return vs FX component, hedged vs unhedged |
| 13. Country Universe | 14 | 2004–2025 | Country coverage, feasibility, cross-tab analysis |
| 14. Strategy Comparison | 15 | 2019–2025 | Head-to-head TO_MVO vs Momentum: dollar P&L, break-even AUM |
| 15. Factor Model (Full) | 16 | 2014–2024 | FF5+Mom regressions, rolling loadings, GRS test (full sample) |
| 16. Factor Model (OOS) | 17 | 2019–2024 | OOS-only factor regressions, alpha comparison, annual sub-periods |
| 17. Enhanced Hedging | 18 | 2019–2024 | Multi-window beta blending, cumulative returns, Sharpe comparison |
| 18. US Diversification | 19 | 2019–2024 | US factor loadings, hurdle Sharpe/alpha, efficient frontier |

## Factors

| Factor | Column | IC Sign | Interpretation |
|--------|--------|---------|----------------|
| Size | log_mktcap | +1 | Larger firms outperform in EM |
| Value (P/B) | pb_w | +1 | Growth (high P/B) outperforms in EM |
| Quality (ROE) | roe_w | +1 | Higher profitability → higher returns |
| Momentum | mom_11m_w | +1 | Past winners continue to outperform |
| Volatility | ret_vol_w | −1 | Low-vol anomaly present in EM |
| Dividend Yield | div_yield_w | −1 | Low-dividend (reinvesting) firms outperform |

## Portfolio Construction Methods (10 tested)

1. Equal Weight (1/N)
2. Inverse Volatility
3. Minimum Variance
4. Maximum Sharpe Ratio
5. Risk Parity
6. Hierarchical Risk Parity (HRP)
7. Momentum-Weighted
8. Black-Litterman
9. Mean-CVaR
10. **Turnover-Penalized MVO** (selected — lowest TC-adjusted turnover, highest TC-adjusted Sharpe)

## Turnover Structure

The strategy has two turnover layers:
- **Stock-level** (18.4%/month): from monthly factor rebalancing — identical across all PC methods
- **PC-level** (0.7%/month for TO_MVO): from industry weight changes — this is where TO_MVO's penalty works
- **Total**: 19.2%/month (2.3x annual) for TO_MVO vs 28.0% for Momentum

TO_MVO has 32% lower total turnover than Momentum, translating to ~46% more AUM capacity.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Run notebooks in order: 01 through 19. Each notebook corresponds to one stage of the research pipeline. Key outputs are saved to `output/csv/` and `output/figures/`.

To compile the paper (comprehensive version):
```bash
cd paper
pdflatex main_v3.tex && bibtex main_v3 && pdflatex main_v3.tex && pdflatex main_v3.tex
```
