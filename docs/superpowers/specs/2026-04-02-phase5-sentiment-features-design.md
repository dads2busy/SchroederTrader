# Phase 5: Sentiment Features — Design Spec

## Overview

Add three sentiment-derived features to the XGB model using data we already download. These features capture market fear/complacency dynamics that backward-looking price features miss, potentially reducing the model's 95% bullish bias during BEAR regimes. No new data sources needed — all features are derived from existing yfinance downloads (VIX, VIX3M, HYG, LQD, TLT).

## Problem

The XGB model's 6 features are all backward-looking price derivatives. It calls UP 95% of the time during BEAR at 46% accuracy because it has no forward-looking fear/sentiment signals. Sentiment data is a leading indicator — fear builds before crashes, complacency peaks before corrections.

## New Features

| Feature | Formula | Interpretation |
|---------|---------|----------------|
| `vix_term_structure` | `vix_close / vix3m_close` | > 1.0 = inverted (acute panic); < 0.8 = complacent. Inverted term structure historically precedes sharp declines. |
| `credit_spread_level` | `log(hyg_close / lqd_close)` | Absolute credit stress level. Lower = tighter spreads = healthy credit markets. Higher = stress. Complements existing `credit_spread` (20-day change) with the level itself. |
| `tlt_momentum` | `log(tlt_close / tlt_close.shift(20))` | 20-day log return of TLT (long treasury bond ETF). Positive = flight to safety (fear). Negative = risk-on. |

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Data sources | Existing yfinance downloads only | VIX, VIX3M, HYG, LQD, TLT already in features_daily.csv |
| Feature count | 3 new (6 → 9 total) | Minimal addition, model regularization handles feature selection |
| Regime detector | Unchanged | Isolate the change; let XGB learn from features without cascading effects |
| Validation | Walk-forward re-run | Same framework used for all prior model changes |

## Changes

### `backtest/download_features.py`

Add three new derived columns in `compute_derived_features()`:

```python
# VIX term structure (panic indicator)
if "vix_close" in result.columns and "vix3m_close" in result.columns:
    result["vix_term_structure"] = result["vix_close"] / result["vix3m_close"]

# Credit spread level (absolute stress)
if "hyg_close" in result.columns and "lqd_close" in result.columns:
    result["credit_spread_level"] = np.log(result["hyg_close"] / result["lqd_close"])

# TLT momentum (flight to safety)
if "tlt_close" in result.columns:
    result["tlt_momentum"] = np.log(result["tlt_close"] / result["tlt_close"].shift(20))
```

### `src/schroeder_trader/strategy/feature_engineer.py`

Add the 3 new features to `compute_features_extended()`. Merge from `ext_df` the same way `credit_spread` and `dollar_momentum` are currently merged.

### `backtest/train_final_composite.py`

Expand `XGB_FEATURES` from 6 to 9:

```python
XGB_FEATURES = [
    "log_return_5d", "log_return_20d", "volatility_20d",
    "credit_spread", "dollar_momentum", "regime_label",
    "vix_term_structure", "credit_spread_level", "tlt_momentum",
]
```

### `src/schroeder_trader/main.py`

Expand `feature_cols` to include the 3 new features (same list as `XGB_FEATURES`).

### Validation

Re-run `backtest/train_final_composite.py` walk-forward evaluation after adding features. Compare against current baseline:

- Full-period Sharpe: 0.94
- Post-2020 Sharpe: 1.26
- Max Drawdown: 16.1%

**Success gate:** Full-period Sharpe >= 0.94 (no regression).

After validation, retrain the final model and deploy to shadow pipeline.

## What Stays The Same

- Regime detector logic — unchanged
- Composite signal routing — unchanged (BULL→SMA, CHOPPY→XGB, BEAR→flat)
- Kelly sizing, trailing stop, transaction costs — unchanged
- Shadow pipeline structure — unchanged
- Data download schedule — unchanged (same yfinance tickers)

## What This Does NOT Include

- Paid data APIs or new data sources
- News headline sentiment (deferred to Phase 8 LLM layer)
- Changes to regime detection logic
- Put/call ratio or breadth features (can add later if these don't move the needle)
