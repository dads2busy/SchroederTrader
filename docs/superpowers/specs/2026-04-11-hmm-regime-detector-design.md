# HMM Regime Detector with VIX Term Structure and Blended Routing

**Date:** 2026-04-11
**Status:** Approved

## Problem

The current regime detector uses fixed thresholds (positive return + low vol = BULL, negative return + high vol = BEAR, else CHOPPY). This produces hard regime boundaries that cause abrupt position changes and detection lag. In 2023, the detector bounced between BULL and CHOPPY during a sustained rally, causing the strategy to sit out and return only 4.4% vs SPY's 26.7%. The detector also uses only backward-looking indicators, missing the forward-looking information embedded in VIX options pricing.

## Solution

Replace the threshold-based regime detector with a Gaussian Hidden Markov Model (HMM) trained on returns, volatility, VIX, and VIX term structure. The HMM outputs per-state probabilities rather than a single discrete label, enabling the composite router to blend signals proportionally instead of hard-switching.

## Architecture

### HMM Regime Detector

**Inputs (4 features):**
- `log_return_20d` -- 20-day momentum (existing)
- `volatility_20d` -- 20-day rolling std of daily returns (existing)
- `vix_close` -- VIX level (available in features CSV, not currently used)
- `vix_term_structure` -- VIX / VIX3M ratio; >1 = backwardation (fear), <1 = contango (complacency) (new derived feature)

**Model:** `hmmlearn.GaussianHMM` with `covariance_type="full"` to capture feature correlations.

**State count selection:** During each training window, fit with n_states = 2, 3, and 4. Select by lowest BIC (Bayesian Information Criterion). The selected state count is persisted with the model.

**State labeling:** After fitting, sort states by their mean `log_return_20d` emission value. Lowest mean return = BEAR, highest = BULL. For 3 states, the middle = CHOPPY. For 2 states, only BEAR and BULL (CHOPPY probability is always 0). For 4 states, the two middle states both route to XGB in the composite router.

**Output:** For each day, a probability vector across states (e.g., `[0.05, 0.70, 0.25]` for [BEAR, CHOPPY, BULL]).

**Training cadence:**
- Backtest: refit on each walk-forward window (2-year train / 6-month test)
- Production: refit monthly on trailing 2 years of data

**Persistence:** Saved as `models/hmm_regime.pkl` via joblib.

### New Feature: VIX Term Structure

Added to `feature_engineer.py` in `compute_features_extended()`:

```
vix_term_structure = vix_close / vix3m_close
```

Values > 1 indicate backwardation (near-term fear exceeds medium-term), values < 1 indicate contango (normal/complacent). This is a known regime indicator in the literature.

### Blended Composite Router

New function `composite_signal_blended()` replaces the hard-switching logic for shadow/live trading.

**Inputs:**
- `regime_probs` -- dict mapping state label to probability (from HMM)
- `sma_signal` -- Signal enum from SMA crossover
- `xgb_signal` -- Signal enum from XGB at low threshold
- `bear_weakening` -- bool (positive 5-day return while dominant regime is BEAR)
- `current_exposure` -- float, current portfolio exposure (0.0 to 0.98)

**Logic:**

For each regime, determine its target exposure:
- BULL: SMA signal drives exposure. BUY = 0.98, SELL = 0.0, HOLD = current_exposure.
- CHOPPY (and any middle states in 4-state model): XGB signal drives exposure. BUY = 0.98, SELL = 0.0, HOLD = current_exposure.
- BEAR: Default target = 0.0 (flat). If bear_weakening is true, XGB signal drives exposure instead.

Blend: `target_exposure = sum(regime_prob[i] * target_exposure[i] for all states)`

**Output:** A float between 0.0 and 0.98 representing the target portfolio exposure.

**Existing function `composite_signal_hybrid()` is unchanged** for backward compatibility.

## Integration Touchpoints

### Files that change:

1. **`regime_detector.py`** -- Add `HMMRegimeDetector` class. Methods: `fit(features_df)`, `predict_proba(features_df)` (returns regime probabilities), `label_states()` (post-hoc labeling by mean return), `save(path)`, `load(path)`, `select_n_states(features_df)` (BIC comparison). Existing threshold functions stay for backward compatibility.

2. **`composite.py`** -- Add `composite_signal_blended()`. Existing `composite_signal_hybrid()` unchanged.

3. **`main.py`** -- Shadow pipeline (Step 10) loads HMM model, calls `predict_proba()`, passes probabilities to `composite_signal_blended()`. Logs blended exposure and dominant regime. Live SMA-only path untouched.

4. **`feature_engineer.py`** -- Add `vix_term_structure` computation to `compute_features_extended()`.

5. **`train_final_composite.py`** -- Train and persist HMM alongside XGBoost. Add BIC state selection logic.

6. **`config.py`** -- Add `HMM_MODEL_PATH` (default: `models/hmm_regime.pkl`), `HMM_RETRAIN_DAYS` (default: 30).

7. **`pyproject.toml`** -- Add `hmmlearn` to dependencies.

### Files unchanged:

`sma_crossover.py`, `kelly.py`, `risk_manager.py`, `broker.py`, `trade_log.py`, `email_alert.py`, `trailing_stop.py`, `transaction_cost.py`.

### Database:

No schema changes. `shadow_signals.ml_signal` stores the dominant regime's signal value. `shadow_signals.regime` stores the dominant regime label. Blended exposure is logged to `kelly_fraction` column (repurposed, since Kelly sizing is no longer used for trade decisions).

## Backtest & Validation

**Walk-forward structure:** Same 2-year train / 6-month test, 32 windows.

In each window:
1. Fit HMM on training data (BIC selects state count)
2. Fit XGBoost on training data (unchanged)
3. Run composite on test data

**Comparison runs (all with 10% trailing stop):**

| Run | Regime detector | Routing | Sizing |
|-----|----------------|---------|--------|
| Baseline | Threshold (current) | Hard switching | Binary (98% or 0%) |
| HMM hard | HMM + VIX | Hard switching (most probable state) | Binary (98% or 0%) |
| HMM blended | HMM + VIX | Probability-weighted blending | Continuous (0% to 98%, set by blended exposure) |

**Metrics:** Sharpe ratio, max drawdown, annualized return, per-year breakdown (with specific attention to 2023).

**Validation concern:** The choice of 4 HMM input features was informed by knowledge of the full history. This is the same survivorship issue as the current system -- the walk-forward approach prevents look-ahead bias in model fitting, but not in feature/architecture selection.

## Testing

- Unit tests for `HMMRegimeDetector`: fit, predict_proba, state labeling, serialization, BIC selection
- Unit tests for `composite_signal_blended`: verify blending math, edge cases (100% one regime, equal split, bear weakening)
- Integration: verify pipeline runs end-to-end with HMM model loaded
- Existing tests for threshold detector and `composite_signal_hybrid` remain passing
