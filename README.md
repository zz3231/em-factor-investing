# EM Industry Factor Portfolio Research

Dynamic factor-based industry portfolios with cross-industry allocation, dynamic beta hedging, and quantified US diversification analysis for MSCI Emerging Markets equities.

## Overview

This repository contains the full research pipeline for a systematic EM equity factor strategy built from six stock-level characteristics across eleven GICS-based industries. The stock-level signal panel spans January 2004 to March 2025. Benchmarked, hedged, and US-diversification comparisons are reported on the matched January 2019 to September 2024 out-of-sample window because the EEM total-return series in the working dataset ends in September 2024.

**Selected strategy:** **All6-EW** factor composite + **TO_MVO** cross-industry allocation + rolling **EEM** beta hedge.

**Headline results (matched Jan 2019 to Sep 2024 OOS window, 69 months):**
- Long-only Sharpe: 0.607 gross, 0.552 net of 45 bp transaction costs, versus 0.356 for EEM
- Hedged Sharpe: 0.828 gross, 0.673 net, with net Jensen's alpha of 4.16%/yr and residual beta of 0.049
- Drawdown compression: max drawdown improves from -27.4% long-only to -5.2% hedged
- US diversification value: correlation with US equities is 0.223, hurdle Sharpe is 0.187, and diversification premium is 0.642
- A 10% US+EM allocation reduces portfolio volatility by 9.1% and improves Sharpe from 0.835 to 0.860
- Capacity: ~$10M practical and ~$51M median at 10% ADV participation; hedged Sharpe remains above 0.70 up to roughly $50M AUM

**Experimental design:** Strict temporal split with no data snooping:
- 2004–2008: Warmup (rolling windows, beta estimation)
- 2009–2013: Factor construction selection (→ All6-EW)
- 2014–2018: Portfolio construction selection (→ TO_MVO)
- 2019–Mar 2025: True holdout for stock-level strategy outputs
- Jan 2019–Sep 2024: Matched holdout window for EEM-based benchmark, hedge, and diversification comparisons

## Deliverables

- [Journal-style paper (PDF)](deliverables/papers/em_jpm_paper.pdf)
- [Paper source (LaTeX)](paper/jpm_paper.tex)
- [Presentation deck (PDF)](deliverables/slides/EM_Factor_Investing.pdf)
- [Presentation deck (editable PPTX)](deliverables/slides/EM_Factor_Investing_Final.pptx)

## Diversification Benefit

The diversification result is one of the main takeaways of the project. In the matched out-of-sample window, the hedged EM strategy behaves like a distinct return stream rather than a repackaged US equity factor tilt:

- Corr(EM hedged, US market) = 0.223
- Hurdle Sharpe = 0.187, versus realized hedged Sharpe = 0.828
- Diversification premium = 0.642
- US FF5+Mom factors explain only 15.8% of return variance
- A 10% allocation to the hedged EM sleeve lowers portfolio volatility by 9.1% and lifts portfolio Sharpe from 0.835 to 0.860

## Project Structure

```
EM/
  data/                         Inputs required to run the notebooks (see DATA_DICTIONARY.md)
    df_ds_signal.csv              Primary stock-month panel
    msci_em_index_price.xlsx      MSCI EM index levels (benchmark)
    eem_returns_monthly.csv       EEM ETF monthly total returns (hedge / benchmark)
    em_ff6_factors.csv            EM FF5 + momentum factors (monthly)
    us_ff6_factors.csv            US FF5 + momentum factors (monthly)
    em_ff5.zip, em_mom.zip        Archived factor bundles (optional; CSVs above are canonical)
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
    07_holdout_long_only.ipynb    Matched OOS long-only vs EEM (2019-01 to 2024-09)
    08_long_only_attribution.ipynb      Long-only attribution (TO_MVO, matched OOS)
    08b_long_only_attribution_momentum.ipynb  Long-only attribution (Momentum, matched OOS)
    09_hedging.ipynb              Dynamic beta hedging with EEM total return
    10_hedged_attribution.ipynb   Hedged attribution (TO_MVO, matched OOS)
    10b_hedged_attribution_momentum.ipynb  Hedged attribution (Momentum, matched OOS)
    11_capacity_analysis.ipynb    ADV capacity & Sharpe degradation (TO_MVO, matched OOS)
    11b_capacity_momentum.ipynb   Capacity comparison: TO_MVO vs Momentum
    12_transaction_cost_deep_dive.ipynb  Granular TC decomposition (matched OOS)
    13_fx_decomposition.ipynb     FX overlay, local vs currency returns
    14_country_universe_exploration.ipynb  Country dimension analysis
    15_strategy_comparison.ipynb  Head-to-head: TO_MVO vs Momentum (dollar P&L)
    16_factor_model_analysis.ipynb  Full-sample factor model (FF5+Mom) regressions
    17_factor_model_oos.ipynb     OOS-only factor model analysis (2019-2024)
    18_enhanced_hedge_analysis.ipynb  Multi-window beta hedging & cumulative return comparison
    19_us_diversification_analysis.ipynb  US portfolio diversification value analysis
    20_holdings_diagnostic.ipynb  Holdings-based diagnostic: 2023 crash, top/bottom stocks, CMA mystery
  output/
    csv/                        Intermediate and final result CSVs
    figures/                    All plots (PDF)
    latex/                      LaTeX tables
  deliverables/                 PDFs and slide decks kept out of the repo root
    papers/                     Written deliverables (PDF)
    slides/                     Presentation decks (PPTX and PDF exports)
  paper/                        LaTeX source for the long-form write-up (optional)
```

## Research Pipeline

| Stage | Notebook | Period | Description |
|-------|----------|--------|-------------|
| 1. Data | 01, 02 | 2004–Mar 2025 | Universe: 953 stocks, 36 countries, 11 industries, 255 months |
| 2. Factor Testing | 03 | 2004–Mar 2025 | Rank IC analysis for 6 factors across 11 industries |
| 3. Single Factor | 04 | 2009–2013 | Best single factor per industry via rolling 60m IC |
| 4. Multi-Factor | 05 | 2009–2013 | 8 composition variants; All6-EW selected (zero DoF, diversified) |
| 5. Portfolio Construction | 06, 06b | 2014–2018 | 10 allocation methods; TO_MVO selected (highest TC-adjusted Sharpe) |
| 6. Long-Only Holdout | 07 | 2019–Sep 2024 | TO_MVO vs Momentum comparison on the matched OOS benchmark window |
| 7. Long-Only Attribution | 08, 08b | 2019–Sep 2024 | Sharpe/Treynor/Jensen/Appraisal, style analysis (TO_MVO & Momentum) |
| 8. Beta Hedging | 09 | 2019–Sep 2024 | Rolling 60m OLS hedge vs EEM total return |
| 9. Hedged Attribution | 10, 10b | 2019–Sep 2024 | Hedge ratio dynamics, alpha, and sub-period stability (TO_MVO & Momentum) |
| 10. Capacity | 11, 11b | 2019–Sep 2024 | ADV-based capacity and Sharpe degradation comparison |
| 11. TC Deep Dive | 12 | 2019–Sep 2024 | Country-weighted, time-varying, decomposed TC |
| 12. FX Decomposition | 13 | 2019–Sep 2024 | Local stock return vs FX component, hedged vs unhedged |
| 13. Country Universe | 14 | 2004–Mar 2025 | Country coverage, feasibility, cross-tab analysis |
| 14. Strategy Comparison | 15 | 2019–Sep 2024 | Head-to-head TO_MVO vs Momentum: dollar P&L, break-even AUM |
| 15. Factor Model (Full) | 16 | 2014–2024 | FF5+Mom regressions, rolling loadings, GRS test (full sample) |
| 16. Factor Model (OOS) | 17 | 2019–2024 | OOS-only factor regressions, alpha comparison, annual sub-periods |
| 17. Enhanced Hedging | 18 | 2019–2024 | Multi-window beta blending, cumulative returns, Sharpe comparison |
| 18. US Diversification | 19 | 2019–2024 | US factor loadings, hurdle Sharpe/alpha, efficient frontier |
| 19. Holdings Diagnostic | 20 | 2021–2025 | Stock-level holdings, 2023 crash diagnostic, CMA loading mystery |

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

Run notebooks in order: **01 through 20**. Each notebook corresponds to one stage of the research pipeline. Key outputs are saved to `output/csv/` and `output/figures/`.

Column definitions and file-level notes for `data/` and major `output/csv/` artifacts are in `DATA_DICTIONARY.md`.
