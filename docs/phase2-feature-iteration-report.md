# Phase 2.1 Feature Iteration — Results Report

**Date:** 2026-03-18
**Branch:** feature/phase2.1-feature-iteration
**Previous report:** `docs/phase2-ml-signal-report.md`

---

## Summary

Following the external analysis recommendations, we tested 6 new feature candidates and a 20-day prediction horizon change. The best configuration adds a regime label feature and switches to a 20-day horizon, improving post-2020 Sharpe from 0.60 to 0.68. Targets remain unmet.

---

## What We Tested

### Prediction Horizon Change

The external analysis flagged the 5-day prediction horizon as potentially too noisy relative to the SMA crossover's multi-month holding period. We tested switching the forward return label from 5-day to 20-day, with a corresponding increase in the classification threshold from 0.5% to 1.0%.

### New Feature Candidates

All candidates were tested by adding them individually to the existing 5-feature set and evaluating via walk-forward on 2007-2026 data (34 windows).

| Feature | Source | Computation | Data Availability |
|---|---|---|---|
| `sector_breadth` | 9 SPDR sector ETFs (yfinance) | Fraction of sectors with close > 50-day SMA | 1998+ (used 9 of 11 sectors; excluded XLC and XLRE due to late inception) |
| `skew_level` | CBOE SKEW Index (yfinance ^SKEW) | Raw SKEW index value | 1993+ |
| `skew_change_5d` | CBOE SKEW Index | 5-day change in SKEW index | 1993+ |
| `range_compression_10d` | SPY OHLCV | 10-day rolling mean of (high - low) / close | 1993+ |
| `regime_label` | Derived from SPY | Bull (2) / Choppy (1) / Bear (0) based on 20-day return direction + volatility vs 252-day median | 1993+ |

**Regime label definition:**
- **Bull (2):** log_return_20d > 0 AND volatility_20d < 252-day rolling median volatility
- **Bear (0):** log_return_20d < 0 AND volatility_20d > 252-day rolling median volatility
- **Choppy (1):** everything else

---

## Results: 5-Day Horizon

All new features were first tested on the existing 5-day prediction horizon.

| Feature Set | Full Sharpe | Post-2020 | Delta (Full) |
|---|---|---|---|
| Current 5 features | 0.742 | 0.596 | — |
| + sector_breadth | 0.616 | 0.454 | -0.126 |
| + skew_level | 0.682 | 0.559 | -0.060 |
| + skew_change_5d | 0.452 | 0.387 | -0.290 |
| + range_compression_10d | 0.523 | 0.478 | -0.219 |
| + regime_label | 0.644 | 0.406 | -0.098 |

**Conclusion:** Every new feature degrades performance on the 5-day horizon. The existing 5-feature set is already at the ceiling for 5-day return prediction.

---

## Results: 20-Day Horizon

Switching to 20-day forward return labels changes the picture.

### Horizon Change Alone (No New Features)

| Configuration | Full Sharpe | Post-2020 |
|---|---|---|
| 5 features, 5-day horizon | 0.742 | 0.596 |
| 5 features, 20-day horizon | 0.748 | 0.603 |
| Base 3 features, 5-day, full 1993-2026 | 0.364 | 0.491 |
| Base 3 features, 20-day, full 1993-2026 | 0.540 | 0.588 |

The 20-day horizon is a significant improvement for the base model on full data (+0.18 Sharpe), but provides only a modest improvement for the 5-feature set on 2007+ data (+0.006 full, +0.007 post-2020). The credit_spread and dollar_momentum features already captured most of what the longer horizon provides.

### New Features on 20-Day Horizon

| Feature Set | Full Sharpe | Post-2020 | Delta (Full) |
|---|---|---|---|
| 5 features (20-day) | 0.748 | 0.603 | — |
| + sector_breadth | 0.734 | 0.628 | -0.014 |
| + skew_level | 0.682 | 0.603 | -0.066 |
| + skew_change_5d | 0.701 | 0.634 | -0.047 |
| + range_compression_10d | 0.721 | 0.532 | -0.027 |
| + regime_label | **0.761** | **0.675** | **+0.013** |

**Regime label is the only feature that improves both metrics on the 20-day horizon.** Full Sharpe rises from 0.748 to 0.761, and post-2020 jumps from 0.603 to 0.675 — the largest post-2020 improvement seen in any experiment.

### Stacking Combinations (20-Day Horizon)

| Combo | Full Sharpe | Post-2020 |
|---|---|---|
| 5 features | 0.748 | 0.603 |
| + regime_label | **0.761** | **0.675** |
| + sector_breadth | 0.734 | 0.628 |
| + regime + breadth | 0.758 | 0.558 |

Stacking regime_label and sector_breadth together hurts post-2020 performance — the features interfere with each other. Regime label alone is the best addition.

---

## Best Configuration Found

**6 features, 20-day prediction horizon:**

| Feature | Source | Type |
|---|---|---|
| `log_return_5d` | SPY | Momentum |
| `log_return_20d` | SPY | Trend |
| `volatility_20d` | SPY | Risk regime |
| `credit_spread` | HYG/LQD | Credit risk appetite |
| `dollar_momentum` | UUP | Global risk appetite |
| `regime_label` | Derived (SPY vol + return) | Market regime classification |

**Model configuration:** XGBoost with 20-day forward return labels (UP > +1%, DOWN < -1%, FLAT in between), 0.5 confidence threshold for BUY/SELL signals.

### Performance

| Metric | Best Config | SMA Target | Gap |
|---|---|---|---|
| Full-period Sharpe (2007+) | **0.76** | 0.88 | -0.12 |
| Post-2020 Sharpe | **0.68** | 1.14 | -0.46 |
| Accuracy | 0.48 | — | — |
| Walk-forward windows | 34 | — | — |

---

## Progression Across All Experiments

| Iteration | Full Sharpe | Post-2020 | Key Change |
|---|---|---|---|
| Phase 2 original (buggy) | 0.26 | 0.49 | Cross-window return bug |
| After walk_forward.py (buggy) | 0.63 | 0.59 | Look-ahead bias, truncated data |
| Corrected, base 3 features, full data | **0.36** | **0.49** | First honest result |
| + credit_spread + dollar_momentum (2007+) | 0.74 | 0.60 | Feature selection |
| + 20-day horizon | 0.75 | 0.60 | Horizon change |
| + regime_label (best config) | **0.76** | **0.68** | Best result |

---

## Why the Targets Are Likely Unreachable

The SMA crossover baseline (Sharpe 0.88 full, 1.14 post-2020) is generated by a strategy that made **15 trades in 30 years**. Each trade is a major regime shift (golden cross / death cross) that plays out over months to years. This produces:

- Very few losing trades (high win rate due to long holding periods)
- Large gains per trade (capturing multi-year bull runs)
- Low portfolio turnover (minimal transaction costs)

The XGBoost model generates predictions daily and changes position far more frequently. Even with the 20-day horizon, it produces hundreds of position changes versus the SMA's 15. More trades means more transaction costs, more exposure to noise, and more opportunities to be wrong.

These are fundamentally different strategy profiles. Comparing their Sharpe ratios directly is misleading — the SMA's Sharpe benefits from a very small denominator (low volatility of returns due to infrequent trading), while the XGBoost model's Sharpe is diluted by daily return variance.

A fairer comparison would be risk-adjusted return per trade or total return over the period, but those metrics were not part of the original gate criteria.

---

## Feature Candidates That Did Not Help

| Feature | 5-Day Impact | 20-Day Impact | Assessment |
|---|---|---|---|
| `sector_breadth` | -0.126 | -0.014 | Mild post-2020 help (+0.025) but not enough to offset full-period loss |
| `skew_level` | -0.060 | -0.066 | CBOE SKEW index is too noisy as a raw level; may work better as percentile rank |
| `skew_change_5d` | -0.290 | -0.047 | Severe degradation on 5-day; modest harm on 20-day |
| `range_compression_10d` | -0.219 | -0.027 | Intraday range does not add predictive value beyond close-to-close volatility |

---

## Recommendations

1. **The XGBoost model at Sharpe 0.76 / 0.68 is the best achievable result** with the current approach (daily classifier, momentum + macro features, 20-day horizon). Further feature additions consistently degrade performance.

2. **The SMA crossover targets (0.88 / 1.14) are structurally mismatched** with a daily classifier. Consider either:
   - Revising the gate criteria to compare strategies with similar trade frequencies
   - Accepting that ML shadow mode serves a different purpose (diversification, regime awareness) rather than beating SMA on Sharpe

3. **If pursuing further improvement**, the most promising direction is not more features but a different model architecture — e.g., a regime-switching model that uses the SMA crossover during trending markets and the XGBoost classifier during choppy regimes. This would combine the SMA's strength (trend-following patience) with ML's strength (regime detection).

4. **Deploy the current model to shadow mode** with the 6-feature / 20-day configuration. Collect live shadow data for 4-8 weeks, then compare actual ML signals against SMA signals to assess real-world value. The model may add value in ways not captured by Sharpe alone (e.g., earlier exits before drawdowns, better risk management signals).
