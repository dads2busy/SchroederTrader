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

**Output:** DataFrame with 6 feature columns, NaN rows dropped (first ~50 rows lost to longest rolling window — SMA 200 for `sma_ratio`).

| Feature | Computation | Signal Type |
|---|---|---|
| `log_return_5d` | `log(close / close.shift(5))` | Momentum |
| `log_return_20d` | `log(close / close.shift(20))` | Trend |
| `volatility_20d` | `close.pct_change().rolling(20).std()` | Risk regime |
| `sma_ratio` | `SMA_50 / SMA_200` | Trend strength |
| `volume_ratio` | `volume / volume.rolling(20).mean()` | Activity |
| `rsi_14` | Standard RSI formula on 14-day window | Mean-reversion |

**Label (training only):** `forward_return_5d_class`
- `1` (UP): 5-day forward return > +0.5%
- `-1` (DOWN): 5-day forward return < -0.5%
- `0` (FLAT): between -0.5% and +0.5%

**Interface:**
- `compute_features(df) -> DataFrame` — for live prediction (no label)
- `compute_features_with_labels(df) -> DataFrame` — for training only (includes future-looking label)

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

**Signal mapping from predicted class:**
- Predicted class `1` (UP) with probability > 0.5 → `BUY`
- Predicted class `-1` (DOWN) with probability > 0.5 → `SELL`
- Otherwise (FLAT, or low confidence) → `HOLD`

**Interface:**
- `train_model(X_train, y_train, X_val, y_val) -> XGBClassifier` — train with early stopping
- `load_model(path) -> XGBClassifier` — load saved model
- `predict_signal(model, features_row) -> (Signal, dict)` — returns signal + class probabilities
- `save_model(model, path)` — save to JSON

## Walk-Forward Validation

**Training pipeline** (runs offline via `backtest/train_model.py`):

1. Load full SPY history from `backtest/data/spy_daily.csv`
2. Compute features + labels via `FeaturePipeline.compute_features_with_labels()`
3. Walk-forward loop:
   - 2-year training window, 6-month test window
   - Roll forward in 6-month increments
   - ~45+ test windows from 2000 to present
4. Per window: train XGBoost with early stopping (last 20% of train window as validation set for early stopping)
5. Predict on test set, record all out-of-sample predictions
6. After all windows: aggregate predictions, compute metrics

**Metrics computed:**
- Walk-forward Sharpe ratio (the gate metric)
- Max drawdown
- Accuracy per class (UP/DOWN/FLAT)
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
   Compute features from today's OHLCV data
   Generate ML prediction (class + probabilities)
   Map to signal (BUY/SELL/HOLD)
   Log to shadow_signals table
   If prediction fails: log error, do not affect SMA pipeline
```

**Fail-safe:** If the model file doesn't exist or feature computation fails, shadow mode silently skips. The SMA pipeline is never affected. All shadow code is wrapped in try/except.

## Storage Schema Addition

### `shadow_signals` table (new)
| Column | Type | Description |
|---|---|---|
| id | INTEGER PK | Auto-increment |
| timestamp | DATETIME | When prediction was made |
| ticker | TEXT | "SPY" |
| close_price | REAL | Closing price used |
| predicted_class | INTEGER | -1, 0, or 1 |
| predicted_proba | TEXT | JSON string of class probabilities |
| ml_signal | TEXT | BUY / SELL / HOLD |
| sma_signal | TEXT | What the SMA crossover produced |

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
├── models/                           # NEW: saved models (gitignored)
│   └── xgboost_spy.json
└── tests/
    ├── test_feature_engineer.py      # NEW
    ├── test_xgboost_classifier.py    # NEW
    └── test_shadow_logging.py        # NEW
```

## Dependencies (additions to pyproject.toml)

```
xgboost>=2.0.0
scikit-learn>=1.4.0
shap>=0.45.0
```

Added to main dependencies (not optional) since the shadow mode runs daily.

## Testing Strategy

- **test_feature_engineer.py**: verify each feature computation on synthetic data, verify label thresholds, verify NaN handling
- **test_xgboost_classifier.py**: train on synthetic data, verify signal mapping from predictions, verify model save/load round-trip
- **test_shadow_logging.py**: verify shadow_signals table creation, logging, retrieval; verify shadow step skips gracefully when no model exists
- Existing tests remain unchanged — Phase 1 pipeline is not modified

## Error Handling

- Feature computation failure → log error, skip shadow step
- Model file missing → log info, skip shadow step
- Model prediction failure → log error, skip shadow step
- None of the above affect the SMA pipeline or order execution
- Shadow errors are logged to Python logging (file + console) but do not trigger error alert emails (to avoid noise)

## Future Phase Compatibility

- **Phase 3**: The `FeaturePipeline` class is designed to be extended with additional features (LSTM embeddings, regime labels)
- **Promotion path**: When ML is promoted to live, `xgboost_classifier.py` implements the same `generate_signal()` interface as `sma_crossover.py` — the orchestrator just switches which strategy module it calls
- **Phase 7**: The walk-forward training script becomes the basis for automated retraining triggered by drift detection
