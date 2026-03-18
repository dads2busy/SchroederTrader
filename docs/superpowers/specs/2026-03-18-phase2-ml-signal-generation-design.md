# Phase 2: ML Signal Generation — Design Spec

## Overview

Phase 2 replaces the hard-coded SMA crossover rule with a trained XGBoost classifier that predicts 5-day forward return direction from engineered features. The ML model runs in **shadow mode** — generating predictions alongside the live SMA bot without placing orders — until it demonstrably beats the Phase 1 baseline.

**Gate criterion to advance to Phase 3:** ML shadow predictions produce a higher walk-forward out-of-sample Sharpe ratio than the SMA crossover baseline (0.88 full period, 1.14 post-2020), net of 0.05% transaction costs per trade.

**Phase 1 baseline to beat:**

| Metric | SMA Crossover |
|---|---|
| Full-period Sharpe | 0.88 |
| Post-2020 Sharpe | 1.14 |
| Max Drawdown | 33.7% |
| Win Rate | 78.6% |
| Total Trades (30yr) | 15 |

## Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Model | XGBoost | Higher accuracy than Random Forest, built-in feature importance, standard in quant finance |
| Feature approach | Minimal (6 features), add iteratively via SHAP | Avoids noise from over-featuring, disciplined iteration |
| Prediction horizon | 5-day forward return | Balances signal quality with training sample count |
| Label thresholds | UP > +0.5%, DOWN < -0.5%, FLAT in between | Avoids labeling noise as signal; sub-0.5% moves eaten by costs |
| Validation | Walk-forward (2yr train, 6mo test) | Non-negotiable for time-series; prevents look-ahead bias |
| Deployment mode | Shadow (log predictions, no orders) | Compare against live SMA bot before trusting with capital |
| Coexistence | Additive — SMA pipeline unchanged | If model file missing, shadow step silently skips |

## Feature Engineering

A `FeaturePipeline` class that transforms OHLCV data into ML features.

**Input:** pandas DataFrame with columns `open, high, low, close, volume` (same as `fetch_daily_bars` returns).

**Output:** DataFrame with 6 feature columns, NaN rows dropped (first ~200 rows lost to longest rolling window — SMA 200 for `sma_ratio`).

| Feature | Computation | Signal Type |
|---|---|---|
| `log_return_5d` | `log(close / close.shift(5))` | Momentum |
| `log_return_20d` | `log(close / close.shift(20))` | Trend |
| `volatility_20d` | `close.pct_change().rolling(20).std()` | Risk regime |
| `sma_ratio` | `SMA_50 / SMA_200` | Trend strength |
| `volume_ratio` | `volume / volume.rolling(20).mean()` | Activity |
| `rsi_14` | Standard RSI formula on 14-day window | Mean-reversion |

**Label (training only):** `forward_return_5d_class`

The raw 5-day forward return is classified using these thresholds:
- UP (> +0.5%): encoded as class `2`
- FLAT (between -0.5% and +0.5%): encoded as class `1`
- DOWN (< -0.5%): encoded as class `0`

XGBoost requires contiguous integer class labels starting at 0. The mapping is:
- Class `0` = DOWN
- Class `1` = FLAT
- Class `2` = UP

For human-readable logging and storage, we map back to semantic labels (DOWN/FLAT/UP or -1/0/1).

**Interface:**
- `compute_features(df) -> DataFrame` — for live prediction (no label)
- `compute_features_with_labels(df) -> DataFrame` — for training only (includes future-looking label). Drops trailing rows where the 5-day forward return cannot be computed (last 5 rows of the dataset).

The label uses future data and is only computed during offline training, never during live prediction.

## XGBoost Model

**Configuration (conservative defaults to avoid overfitting):**

```python
{
    "max_depth": 4,
    "n_estimators": 200,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "early_stopping_rounds": 20,
    "random_state": 42,
}
```

**Signal mapping from predicted probabilities:**

XGBoost with `multi:softprob` returns a probability vector `[P(DOWN), P(FLAT), P(UP)]` (indices 0, 1, 2). Signal mapping:
- `argmax == 2` (UP) AND `proba[2] > 0.5` → `BUY`
- `argmax == 0` (DOWN) AND `proba[0] > 0.5` → `SELL`
- Otherwise (FLAT, or no class exceeds 0.5 confidence) → `HOLD`

**Interface:**
- `train_model(X_train, y_train, X_val, y_val) -> XGBClassifier` — train with early stopping. Uses scikit-learn's `XGBClassifier` API.
- `load_model(path) -> XGBClassifier` — load saved model
- `predict_signal(model, features_row) -> (Signal, dict)` — returns signal + class probabilities as `{"DOWN": float, "FLAT": float, "UP": float}`
- `save_model(model, path)` — save to JSON

Note: `predict_signal` is the Phase 2 internal API for shadow mode. A `generate_signal(df)` facade matching the SMA crossover interface will be added at promotion time, not now.

**scikit-learn usage:** Used for XGBoost's sklearn-compatible API (`XGBClassifier`) and for metrics computation (`accuracy_score`, `classification_report`) in the training script.

## Walk-Forward Validation

**Training pipeline** (runs offline via `backtest/train_model.py`):

1. Load full SPY history from `backtest/data/spy_daily.csv`
2. Compute features + labels via `FeaturePipeline.compute_features_with_labels()`
3. Walk-forward loop:
   - 2-year training window, 6-month test window
   - Roll forward in 6-month increments
   - ~48 test windows from 2002 to present
4. Per window: train XGBoost with early stopping (last 20% of train window as validation set for early stopping)
5. Predict on test set, record all out-of-sample predictions
6. After all windows: aggregate predictions, compute metrics

**Metrics computed:**
- Walk-forward Sharpe ratio (the gate metric)
- Max drawdown
- Accuracy per class (DOWN/FLAT/UP)
- Total trades generated
- Win rate
- SHAP feature importance (averaged across windows)

**Outputs saved to `backtest/results/`:**
- `ml_walkforward_results.json` — metrics in same format as Phase 1 backtest
- `ml_vs_sma_comparison.json` — side-by-side comparison
- `shap_importance.png` — feature importance plot

**Final model:** After walk-forward evaluation, train one final model on all available data and save to `models/xgboost_spy.json` for daily shadow predictions.

## Shadow Mode Integration

The orchestrator (`main.py`) gets a new step after the existing pipeline:

```
[Steps 1-9 run unchanged — SMA crossover, orders, alerts]
        │
        ▼
[Step 10: Shadow ML Prediction]
   Load model from models/xgboost_spy.json
   If model file missing: log "no ML model, skipping shadow" and exit
   Fetch OHLCV data with fetch_daily_bars(TICKER, days=400) for sufficient feature warmup
   Compute features from OHLCV data
   Generate ML prediction (class + probabilities)
   Map to signal (BUY/SELL/HOLD)
   Log to shadow_signals table (uses SMA signal from Step 5, already in scope)
   If prediction fails: log error, do not affect SMA pipeline
```

**Data note:** Shadow mode calls `fetch_daily_bars(TICKER, days=400)` (not the default 365) to ensure enough rows for SMA_200 warmup plus usable feature rows. This is a separate call from Step 4's data fetch to keep the SMA pipeline untouched.

**Fail-safe:** If the model file doesn't exist or feature computation fails, shadow mode silently skips. The SMA pipeline is never affected. All shadow code is wrapped in try/except.

## Storage Schema Addition

### `shadow_signals` table (new)
| Column | Type | Description |
|---|---|---|
| id | INTEGER PK | Auto-increment |
| timestamp | DATETIME | When prediction was made |
| ticker | TEXT | "SPY" |
| close_price | REAL | Closing price used |
| predicted_class | INTEGER | 0 (DOWN), 1 (FLAT), or 2 (UP) |
| predicted_proba | TEXT | JSON: `{"DOWN": 0.1, "FLAT": 0.3, "UP": 0.6}` |
| ml_signal | TEXT | BUY / SELL / HOLD |
| sma_signal | TEXT | What the SMA crossover produced (from Step 5) |

Added to `init_db()` — existing tables unchanged.

### New storage functions:
- `log_shadow_signal(conn, timestamp, ticker, close_price, predicted_class, predicted_proba, ml_signal, sma_signal)`
- `get_shadow_signals(conn) -> list[dict]` — for comparison script

## Promotion Script

`backtest/compare_models.py` — run manually to evaluate whether to promote ML to live:

1. Read `signals` table (SMA actual signals) and `shadow_signals` table (ML predictions)
2. For each day with both signals: simulate what each would have done
3. Compute Sharpe, drawdown, accuracy for both over the shadow period
4. Print comparison table
5. If ML Sharpe > SMA Sharpe: print "Ready to promote — ML outperforms SMA"

**Position simulation rules:** ML signals are treated identically to SMA signals for an apples-to-apples comparison:
- BUY = go long (full allocation minus cash buffer)
- SELL = close position (go flat)
- HOLD = maintain current position

This means the ML model's 5-day prediction horizon drives signal timing, but the position management is the same binary all-in/all-out logic as Phase 1. The comparison script simulates returns using actual SPY closing prices from the `signals` table.

This script is informational only — promotion to live is a manual decision.

## Project Structure (new/modified files)

```
SchroederTrader/
├── src/schroeder_trader/
│   ├── strategy/
│   │   ├── sma_crossover.py          # UNCHANGED
│   │   ├── feature_engineer.py       # NEW: FeaturePipeline class
│   │   └── xgboost_classifier.py     # NEW: train, load, predict
│   ├── storage/
│   │   └── trade_log.py              # MODIFIED: add shadow_signals table + functions
│   └── main.py                       # MODIFIED: add shadow prediction step 10
├── backtest/
│   ├── train_model.py                # NEW: walk-forward training script
│   └── compare_models.py            # NEW: SMA vs ML comparison
├── models/                           # NEW: saved models (gitignored — add to .gitignore)
│   └── xgboost_spy.json
└── tests/
    ├── test_feature_engineer.py      # NEW
    ├── test_xgboost_classifier.py    # NEW
    └── test_shadow_logging.py        # NEW
```

## Dependencies (additions to pyproject.toml)

Main dependencies (needed for daily shadow mode):
```
xgboost>=2.0.0
scikit-learn>=1.4.0
```

Optional dependencies (add to `[project.optional-dependencies] backtest`):
```
shap>=0.45.0
```

SHAP is only used in the offline training script for feature importance analysis, not at runtime. Keeping it optional avoids bloating the daily cron environment.

## Testing Strategy

- **test_feature_engineer.py**: verify each feature computation on synthetic data, verify label thresholds (0/1/2 encoding), verify NaN handling, verify trailing row drop in `compute_features_with_labels`
- **test_xgboost_classifier.py**: train on synthetic data, verify signal mapping from probability vectors, verify model save/load round-trip
- **test_shadow_logging.py**: verify shadow_signals table creation, logging, retrieval; verify shadow step skips gracefully when no model file exists
- Existing tests remain unchanged — Phase 1 pipeline is not modified

## Error Handling

- Feature computation failure → log error, skip shadow step
- Model file missing → log info, skip shadow step
- Model prediction failure → log error, skip shadow step
- None of the above affect the SMA pipeline or order execution
- Shadow errors are logged to Python logging (file + console) but do not trigger error alert emails (to avoid noise)

## Future Phase Compatibility

- **Phase 3**: The `FeaturePipeline` class is designed to be extended with additional features (LSTM embeddings, regime labels)
- **Promotion path**: When ML is promoted to live, a `generate_signal(df)` facade will be added to `xgboost_classifier.py` matching the SMA crossover interface — the orchestrator just switches which strategy module it calls
- **Phase 7**: The walk-forward training script becomes the basis for automated retraining triggered by drift detection
