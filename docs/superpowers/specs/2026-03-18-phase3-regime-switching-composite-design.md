# Phase 3: Regime-Switching Composite Strategy — Design Spec

## Overview

A composite strategy that routes signals through a regime detector: SMA crossover handles trending markets (Bull/Bear), XGBoost classifier handles sideways markets (Choppy). Each component runs only in the regime it's suited for, combining the SMA's trend-following patience with ML's shorter-term edge in uncertain conditions.

**Motivation:** Phase 2.1 experiments showed that regime_label is the only feature that improves the XGBoost model, and it helps most on post-2020 data. This means the model's core weakness is regime-awareness, not predictive power within a regime. The composite addresses this directly by letting each strategy operate where it's strongest.

## Gate Criteria

| Criterion | Threshold | Rationale |
|---|---|---|
| Full-period Sharpe | ≥ 0.88 | Match SMA crossover's proven level |
| Post-2020 Sharpe | ≥ 0.80 | Confirm regime-switching closes the modern gap |
| Max drawdown | ≤ 25% | Composite should improve on SMA's 33.7% |
| Trade frequency | ≥ 30 total trades, with ≥ 10 from XGBoost in Choppy regime | Confirm ML is actively contributing, not just shadowing SMA |

**Diagnostic (not a gate):** Regime residency breakdown — percentage of trading days spent in SMA mode vs XGBoost mode. If >95% of days are SMA mode, the ML is not contributing meaningfully.

## Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Regime detection | Rule-based (vol + 20d return) | Already proven in Phase 2.1 experiments; adding complexity hasn't helped |
| Regime transition | Immediate handoff, position inherits | Simplest; avoids unnecessary transaction costs from flattening |
| XGBoost config | 6 features, 20-day horizon, threshold 0.50 | Validated config from Phase 2.1 (Sharpe 0.76/0.68). Threshold 0.40 showed improvement in sweep but was tested with a custom evaluator; use 0.50 as the conservative default and tune threshold as a post-implementation optimization. |
| SMA config | Existing Phase 1 crossover (SMA 50/200) | Proven strategy, no changes needed |

## Architecture

### Signal Flow

```
Daily pipeline:
  1. Compute regime_label for today
     - Bull:   log_return_20d > 0 AND volatility_20d < 252-day rolling median
     - Bear:   log_return_20d < 0 AND volatility_20d > 252-day rolling median
     - Choppy: everything else

  2. Route to active strategy:
     - Bull or Bear → signal = SMA crossover (golden cross / death cross)
     - Choppy → signal = XGBoost prediction (20-day horizon, threshold 0.50)

  3. Execute signal
     - Position carries across regime transitions
     - Slippage charged only on actual position changes
```

### Regime Detection

Same logic validated in Phase 2.1:

```python
# vol_median MUST be computed as a rolling window up to the current day only
# to avoid look-ahead bias. Never precompute on full dataset.
vol_median = volatility_20d.rolling(252).median()

if log_return_20d > 0 and volatility_20d < vol_median:
    regime = BULL
elif log_return_20d < 0 and volatility_20d > vol_median:
    regime = BEAR
else:
    regime = CHOPPY
```

**Look-ahead prevention:** The 252-day rolling median must be computed strictly from data available up to and including the current day. In walk-forward evaluation, this means computing regime labels incrementally within each test window using only backward-looking data — never precomputing on the full dataset.

**Shared implementation:** The `regime_label` feature used in XGBoost training and the regime detection for signal routing MUST use the same `detect_regime()` function to guarantee consistency.

**Why this works:** Bull markets have positive returns and low volatility. Bear markets have negative returns and high volatility. Choppy markets are everything in between — sideways action, uncertain direction, or volatility without a clear trend. The SMA crossover excels in the first two; the XGBoost model handles the third.

### Regime Transitions

When regime changes (e.g., Choppy → Bull):
- The new strategy takes over immediately
- The current position is inherited — no forced flattening
- If SMA says HOLD and the previous XGBoost position was long, the position stays long until SMA generates a SELL signal

**SMA state on transition:** The SMA crossover is stateless — it only generates BUY/SELL on actual crossover days, and HOLD otherwise. When transitioning from Choppy → Bull/Bear, if no crossover occurs that day, SMA outputs HOLD and the inherited position is maintained. The position will only change when the SMA detects its next crossover event. This is correct behavior — the SMA component should not act on stale crossover state from before the Choppy period.

This avoids unnecessary transaction costs from closing positions just because the regime changed.

### XGBoost Component

Uses the best validated configuration from Phase 2.1:

- **Features:** log_return_5d, log_return_20d, volatility_20d, credit_spread, dollar_momentum, regime_label
- **Prediction horizon:** 20-day forward return
- **Label thresholds:** UP > +1%, DOWN < -1%, FLAT in between
- **Signal threshold:** 0.50 (conservative default; threshold tuning is a post-implementation optimization)
- **Signal mapping:** BUY if argmax=UP and P(UP) > 0.50, SELL if argmax=DOWN and P(DOWN) > 0.50, HOLD otherwise

### SMA Crossover Component

Uses the existing Phase 1 strategy unchanged:

- **Golden cross (SMA50 crosses above SMA200):** BUY
- **Death cross (SMA50 crosses below SMA200):** SELL
- **No crossover:** HOLD

## Walk-Forward Evaluation

The composite needs its own walk-forward evaluator that integrates both strategies within each window.

### Per-Window Logic

For each walk-forward window (2yr train, 6mo test, 6mo roll):

1. **Train XGBoost** on the training window (same as Phase 2 walk-forward)
2. **For each test day:**
   a. Compute regime_label
   b. If Bull or Bear: compute SMA crossover signal from price data
   c. If Choppy: compute XGBoost signal from features
   d. Apply signal to position (previous position × today's return)
   e. Charge slippage on position changes
3. **Aggregate returns** across all test windows

### SMA Signal in Walk-Forward

The SMA crossover signal requires detecting golden/death crosses. For each test day:
- Pass **full price history from dataset start through the current test day** to `generate_signal()` — not just the training window. SMA200 requires at least 200 days of warmup, and crossover detection needs the previous day's SMA values.
- The function returns BUY (golden cross), SELL (death cross), or HOLD (no crossover).
- This ensures SMA signals are computed identically to how they would be in live production.

### Position State Across Windows

Position resets to 0 (flat) at the start of each walk-forward test window. Within a window, position carries across regime transitions. This is a simulation constraint — in live trading, position would be continuous.

### Metrics Computed

- Full-period Sharpe ratio (annualized)
- Post-2020 Sharpe ratio
- Max drawdown
- Total trades
- Regime residency: % of test days in Bull, Bear, Choppy
- Per-regime Sharpe (to verify each component is contributing positively in its regime)

## Project Structure

```
SchroederTrader/
├── src/schroeder_trader/
│   └── strategy/
│       ├── regime_detector.py      # NEW: regime classification
│       └── composite.py            # NEW: signal routing
├── backtest/
│   └── composite_strategy.py       # NEW: walk-forward eval of composite
└── tests/
    ├── test_regime_detector.py     # NEW
    └── test_composite.py           # NEW
```

### New Modules

**`regime_detector.py`** — Stateless regime classification.
- `detect_regime(log_return_20d, volatility_20d, vol_median) -> Regime`
- `compute_regimes(df) -> Series` — vectorized regime labels using rolling 252-day median (backward-looking only)
- `Regime` enum: BULL, BEAR, CHOPPY

**`composite.py`** — Signal routing logic.
- `composite_signal(regime, sma_signal, xgb_signal) -> Signal`
- Routes to SMA in Bull/Bear, XGBoost in Choppy

**`backtest/composite_strategy.py`** — Walk-forward evaluation script.
- Loads SPY + external data
- Runs walk-forward with composite signal routing
- Reports all gate metrics + diagnostics
- Saves results to `backtest/results/composite_results.json`

### Existing Modules Used (No Modifications)

- `strategy/sma_crossover.py` — `generate_signal(df)` for SMA signals
- `strategy/xgboost_classifier.py` — `train_model()`, `predict_signal()` for XGBoost
- `strategy/feature_engineer.py` — `FeaturePipeline` for feature computation
- `backtest/walk_forward.py` — `_compute_sharpe()` utility (reused for metrics)

## Dependencies

No new dependencies. All required packages (xgboost, scikit-learn, pandas, yfinance) are already installed.

## Testing Strategy

- **test_regime_detector.py:** Verify regime classification on synthetic data — bull conditions → BULL, bear → BEAR, ambiguous → CHOPPY. Verify edge cases (zero return, median volatility).
- **test_composite.py:** Verify signal routing — Bull + SMA BUY → BUY, Choppy + XGB SELL → SELL, Bull + XGB BUY → ignore XGB (use SMA). Verify position inheritance logic.
- **composite_strategy.py:** Not unit-tested (research script). Verified by running and inspecting results.

## Error Handling

- If XGBoost prediction fails during a Choppy regime, fall back to HOLD
- If regime detection fails (missing data), default to SMA crossover
- All errors logged but non-fatal (same pattern as Phase 2 shadow mode)
