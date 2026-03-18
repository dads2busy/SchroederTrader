# Phase 2 ML Signal Generation — Status Report

**Date:** 2026-03-18
**Branch:** feature/phase2.1-feature-iteration

---

## What We Built

An XGBoost classifier that predicts 5-day forward SPY return direction (DOWN/FLAT/UP), running in shadow mode alongside the existing SMA crossover bot. The ML model logs predictions to a `shadow_signals` table but does not place orders.

### Components

| Module | Path | Purpose |
|---|---|---|
| Feature Engineering | `src/schroeder_trader/strategy/feature_engineer.py` | Computes ML features from SPY OHLCV + external data |
| XGBoost Classifier | `src/schroeder_trader/strategy/xgboost_classifier.py` | Train, predict, save/load model |
| Shadow Storage | `src/schroeder_trader/storage/trade_log.py` | `shadow_signals` table for ML predictions |
| Shadow Orchestration | `src/schroeder_trader/main.py` | Step 10: shadow ML prediction (non-fatal, try/except wrapped) |
| External Data Download | `backtest/download_features.py` | Fetches VIX, ETFs, FRED yields via yfinance + FRED public CSV |
| Walk-Forward Evaluator | `backtest/walk_forward.py` | Reusable walk-forward backtest with per-window returns |
| Feature Selection | `backtest/feature_selection.py` | Greedy forward selection using walk-forward Sharpe |
| Training Script | `backtest/train_model.py` | Walk-forward evaluation + final model export |
| Model Comparison | `backtest/compare_models.py` | SMA vs ML Sharpe comparison on shadow data |

### Test Coverage

67 tests passing across 7 test files:
- 9 feature engineering tests (base features)
- 5 extended feature tests
- 6 XGBoost classifier tests
- 4 shadow logging tests
- 5 orchestrator tests (including shadow mode)
- 38 Phase 1 tests (unchanged)

---

## How the Backtest Works

### Walk-Forward Validation

The model is evaluated using walk-forward validation to prevent look-ahead bias:

1. **Training window:** 2 years of data
2. **Test window:** 6 months of data (out-of-sample)
3. **Roll forward:** 6 months per step
4. **Early stopping:** Last 20% of training window used as validation set; training stops if validation loss doesn't improve for 20 rounds

This produces ~34 test windows on the 2007-2026 dataset (limited by external data availability) or ~61 windows on the full 1993-2026 SPY dataset (base features only).

### Signal Generation

The model outputs a probability vector `[P(DOWN), P(FLAT), P(UP)]` for each day. Signal mapping:

- **BUY (go long):** argmax = UP AND P(UP) > 0.5
- **SELL (go flat):** argmax = DOWN AND P(DOWN) > 0.5
- **HOLD (maintain position):** otherwise (FLAT, or no class exceeds 50% confidence)

This 0.5 confidence threshold is applied identically in both the backtest and shadow mode.

### Return Simulation

For each test window:

1. Position starts at 0 (flat)
2. Each day, the model generates a signal from that day's features
3. The signal updates position, which takes effect on the **next** day's return (no look-ahead)
4. Transaction cost of 0.05% is deducted on each position change
5. Returns are computed within each window independently (no cross-window artifacts)

### Sharpe Calculation

Annualized Sharpe ratio: `sqrt(252) * mean(daily_returns) / std(daily_returns)`

No risk-free rate subtraction (effectively assumes risk-free rate ~ 0).

---

## Selected Features

### Feature Selection Process

Greedy forward selection starting from 3 SHAP-validated base features, testing 11 external candidates one at a time. A candidate is added if it improves full-period Sharpe by at least +0.02 and does not regress post-2020 Sharpe by more than 0.03.

### Final Feature Set (5 features)

| Feature | Source | Computation | Why It Helps |
|---|---|---|---|
| `log_return_5d` | SPY | `log(close / close.shift(5))` | Short-term momentum |
| `log_return_20d` | SPY | `log(close / close.shift(20))` | Medium-term trend |
| `volatility_20d` | SPY | `close.pct_change().rolling(20).std()` | Risk regime |
| `credit_spread` | HYG/LQD (yfinance) | 20-day change of `log(HYG/LQD)` | Credit risk appetite proxy |
| `dollar_momentum` | UUP (yfinance) | `log(UUP / UUP.shift(20))` | Dollar strength / global risk appetite |

### Rejected Features

| Feature | Full Sharpe | Post-2020 | Why Rejected |
|---|---|---|---|
| `drawdown_depth` | 0.690 | 0.507 | Post-2020 regression too large (-0.079) |
| `vix_level` | 0.631 | 0.569 | Improvement below +0.02 threshold |
| `vix_term_structure` | 0.642 | 0.525 | Post-2020 regression too large |
| `vol_of_vol` | 0.637 | 0.561 | Improvement below +0.02 threshold |
| `yield_10y_level` | 0.577 | 0.490 | Worse than base on both metrics |
| `yield_curve_slope` | 0.546 | — | Worse than base |
| `gold_momentum` | 0.480 | — | Worse than base |
| `bond_momentum` | 0.601 | — | Worse than base |
| `em_momentum` | 0.567 | — | Worse than base |

Note: Rejected feature Sharpe values are measured individually (base + that feature), not stacked.

---

## Current Results

### XGBoost Model Performance

| Metric | 2007-2026 (34 windows) | Target |
|---|---|---|
| Full-period Sharpe | **0.74** | 0.88 |
| Post-2020 Sharpe | **0.60** | 1.14 |
| Accuracy | ~0.35 | — |

### Base Features on Full History (for reference)

| Metric | 1993-2026 (61 windows) |
|---|---|
| Full-period Sharpe | 0.36 |
| Post-2020 Sharpe | 0.49 |

The model performs significantly better in the modern regime (2007+) than across the full 30-year history. This is expected — market microstructure, volatility regimes, and cross-asset correlations have changed substantially since the 1990s.

### SMA Crossover Baseline (from Phase 1 backtest)

| Metric | SMA Crossover |
|---|---|
| Full-period Sharpe | 0.88 |
| Post-2020 Sharpe | 1.14 |
| Max Drawdown | 33.7% |
| Win Rate | 78.6% |
| Total Trades (30yr) | 15 |

### Gap to Target

| Metric | XGBoost (2007+) | SMA Target | Gap |
|---|---|---|---|
| Full-period Sharpe | 0.74 | 0.88 | -0.14 |
| Post-2020 Sharpe | 0.60 | 1.14 | -0.54 |

The full-period gap is relatively small (-0.14). The post-2020 gap is large (-0.54), driven by the SMA crossover's anomalously strong performance during the recent bull market (only 15 trades over 30 years, with well-timed entries).

---

## Bugs Found and Fixed During Development

| Bug | Impact | Fix |
|---|---|---|
| Cross-window pct_change | Sharpe computed across date gaps between test windows, producing spurious returns | Compute returns within each window independently (`walk_forward.py`) |
| Look-ahead bias | Today's signal applied to today's return (should apply to tomorrow's) | Use previous position for current return, update position after |
| Backtest/shadow mismatch | Backtest used `model.predict()` (argmax), shadow used 0.5 confidence threshold | Aligned backtest to use same 0.5 threshold as shadow mode |
| Data truncation | `download_features.py` dropped all pre-2007 rows due to VIX3M NaN | Removed global `dropna()`; feature engineer handles NaN per-feature |
| predicted_class mapping | Shadow mode derived class from signal name instead of model output | `predict_signal` now returns actual argmax class |

---

## What Is Not Yet Done

1. **Shadow pipeline not updated** — `main.py` still uses the original 6 base features, not the selected 5. The model file (`models/xgboost_spy.json`) was trained on the old features. Both need updating together before shadow mode uses the new features.

2. **`train_model.py` still has old bugs** — The standalone training script still uses the original Sharpe calculation with cross-window pct_change and look-ahead bias. It should be refactored to use `walk_forward.py` for evaluation.

3. **Targets not met** — The XGBoost model does not beat the SMA crossover baseline on either metric. The model is not ready for promotion to live trading.

---

## Recommendations

The 5-day return prediction approach with momentum + macro features reaches Sharpe 0.74 on 2007+ data but cannot match the SMA crossover's 0.88. Possible next steps:

- **Different prediction horizon** — 5-day may be too noisy; 20-day would align closer to SMA's holding period
- **Regime classification** — predict bull/bear regime instead of return direction
- **Hyperparameter tuning** — modest potential improvement (+0.05-0.10 Sharpe), unlikely to close the gap
- **Accept the result** — SMA crossover is a legitimately strong strategy for SPY with very few trades; ML may not add value for this specific use case
