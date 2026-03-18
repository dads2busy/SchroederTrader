# Phase 4: Shadow Deployment of Composite Strategy — Design Spec

## Overview

Deploy the validated composite strategy (SMA in Bull, flat/XGBoost in Bear, XGBoost in Choppy) to the daily pipeline's shadow mode. The composite signal runs alongside the live SMA crossover, logging predictions for forward-testing without placing orders.

**Goal:** Get the all-gates-passing composite strategy (Sharpe 0.94/1.26, 16.1% max DD) generating live predictions against real market data.

**Note:** The Sharpe 0.94/1.26 results were obtained using the exact hybrid bear routing described in this spec (flat for first 20 bear days, XGBoost at 0.50 after), not the Phase 3 spec's simpler BEAR→SMA routing. The backtest code used **argmax-gated** threshold logic (same as this spec). This has been verified.

## What Changes

The current `main.py` shadow Step 10 loads an XGBoost model and logs a single ML prediction. We replace this with the full composite signal routing: regime detection → route to SMA/flat/XGBoost → log the composite signal, regime, and signal source.

## Decisions

| Decision | Choice | Rationale |
|---|---|---|
| External feature fetching | Call `download_features.py` inline at start of Step 10 (skips if <24h old) | Simplest; existing idempotency handles caching |
| Bear day tracking | Compute from price history each run | Regime is deterministic from prices; no new state to manage |
| Model format | 20-day labels, 6 features, saved to `models/xgboost_spy_20d.json` | New filename avoids silent overwrite by old 5-day training script |
| Shadow table changes | Add `regime`, `signal_source`, `bear_day_count` columns | Needed for monitoring, comparison, and transition diagnostics |
| Signal logic | Argmax-gated thresholds (same as backtest) | Matches the walk-forward code that produced the 0.94 Sharpe result |
| Non-XGB predictions | Store NULL for predicted_class and predicted_proba | Cleaner SQL semantics than sentinel values |

## Daily Pipeline Flow

```
Steps 1-9: UNCHANGED (SMA crossover, orders, alerts, portfolio logging)
  → Step 5 produces: signal (Signal enum), sma_50, sma_200, close_price
  → These are local variables in run_pipeline(), accessible to Step 10

Step 10: Composite Shadow Signal (replaces current shadow step)
  1. Fetch/cache external features via download_features.py (skips if <24h old)
  2. Load external features CSV from PROJECT_ROOT / "backtest" / "data" / "features_daily.csv"
  3. Load XGBoost model from PROJECT_ROOT / "models" / "xgboost_spy_20d.json" (skip if missing)
  4. Validate model class ordering: confirm model.classes_ matches [0, 1, 2] (DOWN, FLAT, UP)
  5. Fetch SPY OHLCV with 400 days of history via fetch_daily_bars(TICKER, days=400)
  6. Compute extended features + regime labels for full history:
     a. Call pipeline.compute_features_extended(spy_df, external_df)
     b. Compute regime labels for all rows using detect_regime() with backward-looking vol_median
     c. Add regime_label as integer feature (BEAR=0, CHOPPY=1, BULL=2)
  7. Extract today's regime from the last row of step 6 result (NOT a separate detect_regime call)
  8. Count consecutive bear days by scanning backward from the last row of step 6 regimes
  9. Generate XGBoost signals (both thresholds) from last row's features:
     a. Use model.predict_proba() on last row's 6 features
     b. Derive class indices from model.classes_ (not hardcoded constants)
     c. Apply argmax-gated threshold at 0.35 → xgb_signal_low
     d. Apply argmax-gated threshold at 0.50 → xgb_signal_high
  10. Route composite signal:
     - BULL → reuse SMA signal from Step 5 (the `signal` variable already in scope)
     - BEAR ≤ 20 days → SELL (go flat)
     - BEAR > 20 days → xgb_signal_high
     - CHOPPY → xgb_signal_low
  11. Log to shadow_signals table
  12. All wrapped in try/except — never affects Steps 1-9
```

**Step 5 SMA signal access:** The `signal` variable from Step 5 is a local in `run_pipeline()` and is already in scope when Step 10 executes (same function body). No pipeline context object or recomputation needed — this is the existing pattern from Phase 2.

## Composite Signal Routing

### `composite_signal_hybrid()`

New function in `composite.py`:

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

**Boundary condition:** `bear_days == 20` routes to FLAT (≤ 20). `bear_days == 21` is the first day routed to XGB. This matches the backtest.

### Bear Day Counting

Computed from the full 400-day price history each run. The function receives the **entire** regimes Series (with regime labels already computed for all rows) and counts how many consecutive days ending at the last row are BEAR:

```python
def count_consecutive_bear_days(regimes: pd.Series) -> int:
    """Count consecutive BEAR days ending at the last row.

    Args:
        regimes: Series of Regime enum values (from compute_regimes or equivalent),
                 covering the full 400-day history. NaN values are treated as non-BEAR.

    Returns:
        Number of consecutive BEAR days ending at the last row.
        Returns 0 if the last row is not BEAR or is NaN.
    """
```

**Important:** The regime labels must be computed from the full 400-day history (not a short slice) because `detect_regime()` requires `vol_median` which is a 252-day rolling median. The full history ensures vol_median is valid for the recent days we care about.

## XGBoost Signal Generation

### Class Index Safety

When the model is loaded, validate and derive class indices from `model.classes_`:

```python
model = load_model(model_path)
class_to_idx = {int(c): i for i, c in enumerate(model.classes_)}
idx_up = class_to_idx[2]    # CLASS_UP = 2
idx_down = class_to_idx[0]  # CLASS_DOWN = 0
```

This prevents silent bugs if the model's internal class ordering ever differs from the assumed [0, 1, 2].

### Argmax-Gated Threshold Logic

Two signals are generated — one at each threshold. The argmax gate ensures the threshold class must also be the plurality class:

```python
proba = model.predict_proba(features_row)[0]
pred_class = int(np.argmax(proba))

# Low threshold signal (for Choppy)
if pred_class == idx_up and proba[idx_up] > 0.35:
    xgb_signal_low = Signal.BUY
elif pred_class == idx_down and proba[idx_down] > 0.35:
    xgb_signal_low = Signal.SELL
else:
    xgb_signal_low = Signal.HOLD

# High threshold signal (for late Bear)
if pred_class == idx_up and proba[idx_up] > 0.50:
    xgb_signal_high = Signal.BUY
elif pred_class == idx_down and proba[idx_down] > 0.50:
    xgb_signal_high = Signal.SELL
else:
    xgb_signal_high = Signal.HOLD
```

**Design note:** The argmax gate means a direction must be BOTH the highest-probability class AND exceed the threshold. This is more conservative than pure threshold (where P(UP)=0.36 with P(FLAT)=0.45 would trigger BUY). This matches the backtest code that produced the 0.94 Sharpe.

## Shadow Signals Table Update

Add three columns to the existing `shadow_signals` table:

| Column | Type | Description |
|---|---|---|
| regime | TEXT | BULL / BEAR / CHOPPY |
| signal_source | TEXT | SMA / FLAT / XGB |
| bear_day_count | INTEGER | Consecutive bear days at time of signal (NULL if not BEAR) |

Existing columns unchanged: id, timestamp, ticker, close_price, predicted_class, predicted_proba, ml_signal, sma_signal.

The `ml_signal` column stores the composite signal (the output of `composite_signal_hybrid`).

**When XGBoost is not consulted (BULL → SMA or early BEAR → FLAT):** `predicted_class` and `predicted_proba` are stored as NULL. This is semantically correct ("not applicable") and avoids downstream JSON parsing issues.

**Migration:** Add columns defensively in `init_db()`. Since SQLite `ALTER TABLE ADD COLUMN` fails if the column already exists, wrap each addition in a try/except. Use the same pattern as `CREATE TABLE IF NOT EXISTS` — idempotent, safe to run on every startup. Existing rows get NULL for the new columns. The `predicted_class` and `predicted_proba` columns are changed to allow NULL (drop NOT NULL constraint by recreating the table if needed, or simply accept that new rows may have NULL).

### Updated Storage Functions

`log_shadow_signal()` gains three new parameters: `regime`, `signal_source`, `bear_day_count`. The `predicted_class` and `predicted_proba` parameters become optional (default NULL).

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
4. Runs a lightweight walk-forward to determine the median optimal `n_estimators` across windows
5. Retrains on **all available data** using the fixed `n_estimators` (no validation split — early stopping hyperparameter already determined)
6. Saves to `models/xgboost_spy_20d.json`
7. Logs training date, n_estimators, and row count for auditability

**Why no 80/20 split:** An 80/20 chronological split excludes the most recent ~4 years of data (2022-2026) from training. Since recent data is the most predictive of near-future behavior, we instead determine `n_estimators` from walk-forward results and retrain on the full dataset.

**Model path:** `models/xgboost_spy_20d.json` (new filename). The old `models/xgboost_spy.json` (5-day model) is left in place but unused. The model path is configured in `config.py` as `COMPOSITE_MODEL_PATH`.

## Live Feature Computation

Step 10 must produce the same 6 features the model was trained on: `log_return_5d, log_return_20d, volatility_20d, credit_spread, dollar_momentum, regime_label`.

**This is NOT the same as the Phase 2 `compute_features()` method** (which produces `sma_ratio, volume_ratio, rsi_14`). Step 10 uses `FeaturePipeline.compute_features_extended(spy_df, external_df)` which was added in Phase 2.1.

The live computation path:
1. Fetch 400 days of SPY OHLCV via `fetch_daily_bars(TICKER, days=400)`
2. Load external features CSV via `pd.read_csv(FEATURES_CSV_PATH)`
3. Call `pipeline.compute_features_extended(spy_df, external_df)` → produces all 14 extended features
4. Compute regime labels for all rows using `detect_regime()` with backward-looking vol_median
5. Extract today's regime from the last row (same computation, guaranteed consistent with routing)
6. Extract the last row's 6 feature values for XGBoost prediction

**Regime consistency guarantee:** The regime used for signal routing (step 7 of pipeline flow) and the regime_label used as an XGBoost feature are both sourced from the same computation in step 6. There is no separate `detect_regime()` call for routing — it reads directly from the last row of the features DataFrame.

**External features CSV path:** `PROJECT_ROOT / "backtest" / "data" / "features_daily.csv"` — resolved via `config.PROJECT_ROOT`.

**Date alignment for the last row:** The external CSV may not have today's date yet (depends on when tickers report). Use a left join on SPY dates with forward-fill (limit 3 days) on external features. This handles multi-day gaps around federal holidays (e.g., Thanksgiving). Log a warning when forward-fill gap exceeds 1 day so data quality issues are visible during shadow monitoring.

`download_features.py` fetches all external tickers and caches to CSV. The pipeline calls it at the start of Step 10 with the existing idempotency check (skip if file <24h old).

## Error Handling

- `download_features.py` fails → log warning, skip entire shadow step
- `models/xgboost_spy_20d.json` missing → log info, skip shadow step
- `model.classes_` doesn't match expected [0, 1, 2] → log error, skip shadow step
- Feature computation fails → log error, skip shadow step
- Regime detection fails → default to CHOPPY
- XGBoost prediction fails → default signal to HOLD
- Any exception in Step 10 → log error, Steps 1-9 unaffected

## Project Structure

```
SchroederTrader/
├── src/schroeder_trader/
│   ├── config.py                     # MODIFIED: add COMPOSITE_MODEL_PATH, FEATURES_CSV_PATH
│   ├── strategy/
│   │   ├── composite.py              # MODIFIED: add composite_signal_hybrid, count_consecutive_bear_days
│   │   └── regime_detector.py        # UNCHANGED
│   ├── storage/
│   │   └── trade_log.py              # MODIFIED: add columns, update log_shadow_signal
│   └── main.py                       # MODIFIED: replace shadow Step 10
├── backtest/
│   ├── train_final_composite.py      # NEW: train 20-day model with walk-forward n_estimators
│   └── download_features.py          # UNCHANGED
├── models/
│   └── xgboost_spy_20d.json          # NEW: 20-day label model
└── tests/
    ├── test_composite.py             # MODIFIED: add hybrid + bear counting tests
    ├── test_shadow_logging.py        # MODIFIED: test new columns + NULL handling
    └── test_main.py                  # MODIFIED: test composite shadow step
```

## Dependencies

No new dependencies. All required packages already installed.

## Testing Strategy

**test_composite.py:**
- `composite_signal_hybrid()`: BULL→SMA, early bear (days≤20)→SELL, late bear (days>20)→XGB high, choppy→XGB low
- Boundary: bear_days=20 → FLAT, bear_days=21 → XGB
- Return type: verify `(Signal, str)` tuple (callers of old `composite_signal()` that unpack single Signal will break)
- `count_consecutive_bear_days()`:
  - All rows BEAR → returns full count
  - Last row NaN → returns 0
  - Alternating BEAR/CHOPPY ending in BEAR → returns 1
  - Single non-BEAR interrupts a long streak → correct termination

**test_shadow_logging.py:**
- New columns (regime, signal_source, bear_day_count) stored and retrieved correctly
- NULL predicted_class and predicted_proba when signal_source is SMA or FLAT
- Existing rows with NULL new columns still returned by get_shadow_signals()

**test_main.py:**
- Step 10 exception does NOT affect Steps 1-9 (assert SMA pipeline completes)
- Shadow step skips when no model file
- Shadow step logs composite signal with regime and source when model exists
- NULL predicted_class/predicted_proba when regime is BULL
