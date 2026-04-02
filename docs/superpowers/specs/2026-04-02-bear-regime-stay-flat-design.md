# Stay Flat in BEAR Regime — Design Spec

## Overview

Remove the "late BEAR" XGB path from the composite signal routing. In BEAR regime, the system always stays flat (SELL) regardless of bear day count or XGB prediction. This eliminates the -82.4% cumulative loss from XGB calls in late BEAR, where the model calls UP 95% of the time at only 46% accuracy.

## Problem

The current composite routes BEAR regime signals as:
- Days 1-20: Go flat (SELL) — this works well
- Days 21+: Trust XGB at 0.50 threshold — this loses money

XGB has a strong bullish bias and cannot detect bear market direction:
- Calls UP 95% of the time during BEAR
- Market only goes up 46% of those days (worse than coin flip)
- Result: -82.4% cumulative loss after day 20 across all bear periods

## Design Decision

| Decision | Choice | Rationale |
|----------|--------|-----------|
| BEAR behavior | Always flat | XGB fundamentally broken in BEAR; no threshold fixes 95% UP bias |
| BEAR→CHOPPY transition | Immediate switch | Regime detector's 20-day lookback is sufficient buffer; trailing stop protects against false exits |
| Bear-specific model | Not now | Premature; first validate that staying flat is sufficient |

## Changes

### `src/schroeder_trader/strategy/composite.py`

Remove the `bear_days` parameter and late-bear XGB routing from `composite_signal_hybrid()`. BEAR always returns `(Signal.SELL, "FLAT")`.

Simplified routing:
- BULL → SMA signal (source="SMA")
- CHOPPY → XGB at low threshold (source="XGB")
- BEAR → SELL (source="FLAT")

The `count_consecutive_bear_days()` function stays — still useful for shadow signal logging and analysis.

### `src/schroeder_trader/config.py`

Remove `XGB_THRESHOLD_HIGH = 0.50` — no longer needed since XGB is never used in BEAR.

### `src/schroeder_trader/main.py`

- Remove `xgb_high` computation (the high-threshold signal)
- Remove `XGB_THRESHOLD_HIGH` from import
- Update `composite_signal_hybrid()` call to drop `xgb_high` and `bear_days` params

### `backtest/run_backtest_kelly.py`

Update the `composite_signal_hybrid()` call to match the new signature (drop `xgb_high` and `bear_days` params).

### Tests

- `tests/test_composite.py`: Remove `test_hybrid_bear_day_21_routes_to_xgb_high`. Add test confirming BEAR always returns SELL at day 1, day 20, and day 100.
- Update other composite tests as needed for new function signature.

## What Stays The Same

- BULL routing (SMA crossover) — unchanged
- CHOPPY routing (XGB at 0.35 threshold) — unchanged
- Bear day counting — still logged in shadow signals for analysis
- XGB prediction — still logged during BEAR for research
- Kelly sizing — still computed and logged
- Trailing stop — still active

## Expected Impact

- Eliminates -82.4% cumulative BEAR loss (largest drag on Sharpe)
- Simplifies composite routing (one fewer code path)
- May miss some bear market rally profits, but these were net negative overall
- Should improve full-period Sharpe from 0.94 toward 1.0+
