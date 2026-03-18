# Phase 4: Shadow Deployment of Composite Strategy — Design Spec

## Overview

Deploy the validated composite strategy (SMA in Bull, flat/XGBoost in Bear, XGBoost in Choppy) to the daily pipeline's shadow mode. The composite signal runs alongside the live SMA crossover, logging predictions for forward-testing without placing orders.

**Goal:** Get the all-gates-passing composite strategy (Sharpe 0.94/1.26, 16.1% max DD) generating live predictions against real market data.

**Note:** The Sharpe 0.94/1.26 results were obtained using the exact hybrid bear routing described in this spec (flat for first 20 bear days, XGBoost at 0.50 after), not the Phase 3 spec's simpler BEAR→SMA routing.

## What Changes

The current `main.py` shadow Step 10 loads an XGBoost model and logs a single ML prediction. We replace this with the full composite signal routing: regime detection → route to SMA/flat/XGBoost → log the composite signal, regime, and signal source.

## Decisions

| Decision | Choice | Rationale |
|---|---|---|
| External feature fetching | Call `download_features.py` inline at start of Step 10 (skips if <24h old) | Simplest; existing idempotency handles caching |
| Bear day tracking | Compute from price history each run | Regime is deterministic from prices; no new state to manage |
| Model format | 20-day labels, 6 features, saved to `models/xgboost_spy.json` | Replaces old 5-day model; same file path for backward compatibility |
| Shadow table changes | Add `regime` and `signal_source` columns | Needed for monitoring and comparison analysis |

## Daily Pipeline Flow

```
Steps 1-9: UNCHANGED (SMA crossover, orders, alerts, portfolio logging)

Step 10: Composite Shadow Signal (replaces current shadow step)
  1. Fetch/cache external features via download_features.py (skips if <24h old)
  2. Load external features CSV
  3. Load XGBoost model from models/xgboost_spy.json (skip if missing)
  4. Fetch SPY OHLCV with 400 days of history (for SMA200 + feature warmup)
  5. Compute extended features for the available history
  6. Compute regime label for today:
     a. Compute log_return_20d, volatility_20d, vol_median from price history
     b. Call detect_regime() for today's values
  7. Count consecutive bear days:
     a. Compute regimes for the full 400-day history using detect_regime()
     b. Count consecutive BEAR days ending at today (scan backward from last row)
  8. Route signal based on regime:
     - BULL → reuse SMA signal from Step 5 (already computed)
     - BEAR ≤ 20 days → SELL (go flat)
     - BEAR > 20 days → XGBoost prediction at 0.50 confidence threshold
     - CHOPPY → XGBoost prediction at 0.35 confidence threshold
  9. Log to shadow_signals table (composite signal, regime, source, probabilities)
  10. All wrapped in try/except — never affects Steps 1-9
```

## Composite Signal Routing

### `composite_signal_hybrid()`

New function in `composite.py` that replaces the simple `composite_signal()`:

```python
def composite_signal_hybrid(
    regime: Regime,
    sma_signal: Signal,
    xgb_signal_low: Signal,    # XGBoost at 0.35 threshold
    xgb_signal_high: Signal,   # XGBoost at 0.50 threshold
    bear_days: int,
    late_bear_threshold: int = 20,
) -> tuple[Signal, str]:
    """Route signal based on regime and bear duration.

    Returns:
        Tuple of (Signal, source) where source is "SMA", "FLAT", or "XGB".
    """
```

Signal routing:
- `BULL` → `(sma_signal, "SMA")`
- `BEAR` and `bear_days <= late_bear_threshold` → `(Signal.SELL, "FLAT")`
- `BEAR` and `bear_days > late_bear_threshold` → `(xgb_signal_high, "XGB")`
- `CHOPPY` → `(xgb_signal_low, "XGB")`

### Bear Day Counting

Computed from the full 400-day price history each run. The function receives the **entire** features DataFrame (with regime labels already computed for all rows) and counts how many consecutive days ending at the last row are BEAR:

```python
def count_consecutive_bear_days(regimes: pd.Series) -> int:
    """Count consecutive BEAR days ending at the last row.

    Args:
        regimes: Series of Regime enum values (from compute_regimes or equivalent),
                 covering the full 400-day history. NaN values are treated as non-BEAR.

    Returns:
        Number of consecutive BEAR days ending at the last row.
        Returns 0 if the last row is not BEAR.
    """
```

**Important:** The regime labels must be computed from the full 400-day history (not a 30-day slice) because `detect_regime()` requires `vol_median` which is a 252-day rolling median. The full history ensures vol_median is valid for the recent days we care about. The function only needs to scan backward from the last row until it finds a non-BEAR day, so even with 400 rows of regimes it is O(N) where N is the bear streak length (typically small).

## XGBoost Signal Generation

Two signals are generated when the model is loaded — one at each threshold:

```python
proba = model.predict_proba(features_row)[0]
pred_class = int(np.argmax(proba))

# Low threshold signal (for Choppy)
if pred_class == CLASS_UP and proba[CLASS_UP] > 0.35:
    xgb_signal_low = Signal.BUY
elif pred_class == CLASS_DOWN and proba[CLASS_DOWN] > 0.35:
    xgb_signal_low = Signal.SELL
else:
    xgb_signal_low = Signal.HOLD

# High threshold signal (for late Bear)
if pred_class == CLASS_UP and proba[CLASS_UP] > 0.50:
    xgb_signal_high = Signal.BUY
elif pred_class == CLASS_DOWN and proba[CLASS_DOWN] > 0.50:
    xgb_signal_high = Signal.SELL
else:
    xgb_signal_high = Signal.HOLD
```

Both are computed unconditionally; the routing function picks which one to use.

## Shadow Signals Table Update

Add two columns to the existing `shadow_signals` table:

| Column | Type | Description |
|---|---|---|
| regime | TEXT | BULL / BEAR / CHOPPY |
| signal_source | TEXT | SMA / FLAT / XGB |

Existing columns unchanged: id, timestamp, ticker, close_price, predicted_class, predicted_proba, ml_signal, sma_signal.

The `ml_signal` column stores the composite signal (the output of `composite_signal_hybrid`).

**When XGBoost is not consulted (BULL → SMA or early BEAR → FLAT):** `predicted_class` is set to -1 and `predicted_proba` is set to `"{}"` (empty JSON). The XGBoost probabilities are only meaningful when the signal comes from XGB. The `predicted_class` column type changes to allow -1 as a sentinel (it was previously `INTEGER NOT NULL`).

Migration: Add columns defensively in `init_db()`. Since SQLite `ALTER TABLE ADD COLUMN` fails if the column already exists, wrap each addition in a try/except or check `PRAGMA table_info(shadow_signals)` first. Use the same pattern as `CREATE TABLE IF NOT EXISTS` — idempotent, safe to run on every startup. Existing rows get NULL for the new columns.

### Updated Storage Functions

`log_shadow_signal()` gains two new parameters: `regime` and `signal_source`.

`get_shadow_signals()` returns the new columns in its output.

## Model Training

A one-time script to train the final model with 20-day labels:

```bash
uv run python backtest/train_final_composite.py
```

This script:
1. Loads SPY + external data
2. Computes extended features with 20-day forward return labels (1% threshold)
3. Adds regime_label feature (using detect_regime with backward-looking vol_median)
4. Trains XGBoost on all available data (80/20 split for early stopping)
5. Saves to `models/xgboost_spy.json`

This replaces the old 5-day model file. The model must be retrained before the first shadow run with the new composite pipeline.

## Live Feature Computation

Step 10 must produce the same 6 features the model was trained on: `log_return_5d, log_return_20d, volatility_20d, credit_spread, dollar_momentum, regime_label`.

**This is NOT the same as the Phase 2 `compute_features()` method** (which produces `sma_ratio, volume_ratio, rsi_14`). Step 10 uses `FeaturePipeline.compute_features_extended(spy_df, external_df)` which was added in Phase 2.1.

The live computation path:
1. Fetch 400 days of SPY OHLCV via `fetch_daily_bars(TICKER, days=400)`
2. Load external features CSV via `pd.read_csv('backtest/data/features_daily.csv')`
3. Call `pipeline.compute_features_extended(spy_df, external_df)` → produces all 14 extended features
4. Compute regime_label for each day using `detect_regime()` with backward-looking vol_median
5. Extract the last row's 6 feature values for XGBoost prediction

**External features CSV path:** `PROJECT_ROOT / "backtest" / "data" / "features_daily.csv"` — resolved via `config.PROJECT_ROOT`, not relative to `download_features.py`. The `download_features.py` script writes to this location; `main.py` reads from it.

**Date alignment for the last row:** The external CSV may not have today's date yet (depends on when tickers report). Use a left join on SPY dates with forward-fill (limit 1 day) on external features. If today's external data is missing, use yesterday's values — this is acceptable for 20-day rolling features where one day of lag has negligible impact.

`download_features.py` fetches all external tickers and caches to CSV. The pipeline calls it at the start of Step 10 with the existing idempotency check (skip if file <24h old).

## Error Handling

- `download_features.py` fails → log warning, skip entire shadow step
- `models/xgboost_spy.json` missing → log info, skip shadow step
- Feature computation fails → log error, skip shadow step
- Regime detection fails → default to CHOPPY
- XGBoost prediction fails → default signal to HOLD
- Any exception in Step 10 → log error, Steps 1-9 unaffected

## Project Structure

```
SchroederTrader/
├── src/schroeder_trader/
│   ├── strategy/
│   │   ├── composite.py            # MODIFIED: add composite_signal_hybrid, count_consecutive_bear_days
│   │   └── regime_detector.py      # UNCHANGED
│   ├── storage/
│   │   └── trade_log.py            # MODIFIED: add regime + signal_source columns
│   └── main.py                     # MODIFIED: replace shadow Step 10
├── backtest/
│   ├── train_final_composite.py    # NEW: train 20-day model
│   └── download_features.py        # UNCHANGED
├── models/
│   └── xgboost_spy.json            # RETRAINED: 20-day labels
└── tests/
    ├── test_composite.py           # MODIFIED: add hybrid tests
    ├── test_shadow_logging.py      # MODIFIED: test new columns
    └── test_main.py                # MODIFIED: test composite shadow step
```

## Dependencies

No new dependencies. All required packages already installed.

## Testing Strategy

- **test_composite.py:** Add tests for `composite_signal_hybrid()` — BULL routes to SMA, early bear routes to SELL, late bear routes to XGB high threshold, choppy routes to XGB low threshold. Test `count_consecutive_bear_days()` with synthetic data.
- **test_shadow_logging.py:** Test new columns in shadow_signals table (regime, signal_source). Test that existing rows with NULL new columns still work.
- **test_main.py:** Update shadow mode tests — test composite routing skips when no model, test composite logs with regime and source when model exists.
