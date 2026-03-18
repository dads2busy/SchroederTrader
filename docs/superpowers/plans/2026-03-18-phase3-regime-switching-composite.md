# Phase 3: Regime-Switching Composite Strategy — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a composite strategy that routes signals through a regime detector — SMA crossover in trending markets, XGBoost in choppy markets — and evaluate it via walk-forward validation against the Phase 3 gate criteria.

**Architecture:** Three new modules (regime_detector, composite, composite_strategy backtest) that compose existing SMA and XGBoost components. No modifications to existing modules.

**Tech Stack:** XGBoost, scikit-learn, pandas (all already installed)

**Spec:** `docs/superpowers/specs/2026-03-18-phase3-regime-switching-composite-design.md`

---

## File Structure

```
SchroederTrader/
├── src/schroeder_trader/
│   └── strategy/
│       ├── regime_detector.py      # NEW: Regime enum + detection logic
│       └── composite.py            # NEW: signal routing
├── backtest/
│   └── composite_strategy.py       # NEW: walk-forward eval of composite
└── tests/
    ├── test_regime_detector.py     # NEW
    └── test_composite.py           # NEW
```

---

### Task 1: Regime Detector Module

**Files:**
- Create: `src/schroeder_trader/strategy/regime_detector.py`
- Create: `tests/test_regime_detector.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_regime_detector.py`:

```python
import numpy as np
import pandas as pd

from schroeder_trader.strategy.regime_detector import Regime, detect_regime, compute_regimes


def test_detect_regime_bull():
    # Positive return, low volatility → BULL
    assert detect_regime(log_return_20d=0.05, volatility_20d=0.008, vol_median=0.01) == Regime.BULL


def test_detect_regime_bear():
    # Negative return, high volatility → BEAR
    assert detect_regime(log_return_20d=-0.03, volatility_20d=0.02, vol_median=0.01) == Regime.BEAR


def test_detect_regime_choppy_positive_return_high_vol():
    # Positive return but high vol → CHOPPY
    assert detect_regime(log_return_20d=0.02, volatility_20d=0.02, vol_median=0.01) == Regime.CHOPPY


def test_detect_regime_choppy_negative_return_low_vol():
    # Negative return but low vol → CHOPPY
    assert detect_regime(log_return_20d=-0.01, volatility_20d=0.005, vol_median=0.01) == Regime.CHOPPY


def test_detect_regime_edge_zero_return():
    # Zero return → CHOPPY (not > 0 and not < 0)
    assert detect_regime(log_return_20d=0.0, volatility_20d=0.005, vol_median=0.01) == Regime.CHOPPY


def test_detect_regime_edge_equal_vol():
    # Vol equals median → CHOPPY (not strictly < and not strictly >)
    assert detect_regime(log_return_20d=0.05, volatility_20d=0.01, vol_median=0.01) == Regime.CHOPPY


def test_compute_regimes_returns_series():
    np.random.seed(42)
    n = 400
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
    df = pd.DataFrame({"close": close}, index=dates)

    regimes = compute_regimes(df)
    assert isinstance(regimes, pd.Series)
    assert regimes.dtype == object  # Regime enum values
    # Should have some non-NaN values after warmup
    valid = regimes.dropna()
    assert len(valid) > 100
    assert set(valid.unique()).issubset({Regime.BULL, Regime.BEAR, Regime.CHOPPY})


def test_compute_regimes_no_lookahead():
    """Verify regime on day N only uses data up to day N."""
    np.random.seed(42)
    n = 400
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
    df = pd.DataFrame({"close": close}, index=dates)

    full_regimes = compute_regimes(df)
    # Regime at day 300 should be the same whether we pass 400 rows or 301 rows
    partial_regimes = compute_regimes(df.iloc[:301])
    assert full_regimes.iloc[300] == partial_regimes.iloc[300]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd ~/git/SchroederTrader
uv run pytest tests/test_regime_detector.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement regime_detector.py**

Create `src/schroeder_trader/strategy/regime_detector.py`:

```python
import enum

import numpy as np
import pandas as pd


class Regime(enum.Enum):
    BULL = "BULL"
    BEAR = "BEAR"
    CHOPPY = "CHOPPY"


def detect_regime(
    log_return_20d: float,
    volatility_20d: float,
    vol_median: float,
) -> Regime:
    """Classify market regime from a single day's indicators.

    Args:
        log_return_20d: 20-day log return of close prices.
        volatility_20d: 20-day rolling std of daily returns.
        vol_median: 252-day rolling median of volatility_20d.

    Returns:
        Regime enum value.
    """
    if log_return_20d > 0 and volatility_20d < vol_median:
        return Regime.BULL
    elif log_return_20d < 0 and volatility_20d > vol_median:
        return Regime.BEAR
    else:
        return Regime.CHOPPY


def compute_regimes(df: pd.DataFrame) -> pd.Series:
    """Compute regime labels for a DataFrame with a 'close' column.

    Uses only backward-looking data (no look-ahead). The 252-day rolling
    median is computed incrementally so each day's regime only depends on
    data available up to that day.

    Args:
        df: DataFrame with a 'close' column.

    Returns:
        Series of Regime enum values, indexed like df. NaN for warmup rows.
    """
    close = df["close"]
    log_return_20d = np.log(close / close.shift(20))
    volatility_20d = close.pct_change().rolling(20).std()
    vol_median = volatility_20d.rolling(252).median()

    regimes = pd.Series(index=df.index, dtype=object)
    for i in range(len(df)):
        lr = log_return_20d.iloc[i]
        vol = volatility_20d.iloc[i]
        vm = vol_median.iloc[i]
        if pd.isna(lr) or pd.isna(vol) or pd.isna(vm):
            regimes.iloc[i] = np.nan
        else:
            regimes.iloc[i] = detect_regime(lr, vol, vm)

    return regimes
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd ~/git/SchroederTrader
uv run pytest tests/test_regime_detector.py -v
```

Expected: 8 tests PASS

- [ ] **Step 5: Commit**

```bash
cd ~/git/SchroederTrader
git add src/schroeder_trader/strategy/regime_detector.py tests/test_regime_detector.py
git commit -m "feat: regime detector module with bull/bear/choppy classification"
```

---

### Task 2: Composite Signal Router

**Files:**
- Create: `src/schroeder_trader/strategy/composite.py`
- Create: `tests/test_composite.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_composite.py`:

```python
from schroeder_trader.strategy.composite import composite_signal
from schroeder_trader.strategy.regime_detector import Regime
from schroeder_trader.strategy.sma_crossover import Signal


def test_bull_uses_sma():
    assert composite_signal(Regime.BULL, Signal.BUY, Signal.SELL) == Signal.BUY


def test_bear_uses_sma():
    assert composite_signal(Regime.BEAR, Signal.SELL, Signal.BUY) == Signal.SELL


def test_choppy_uses_xgb():
    assert composite_signal(Regime.CHOPPY, Signal.HOLD, Signal.BUY) == Signal.BUY


def test_choppy_uses_xgb_sell():
    assert composite_signal(Regime.CHOPPY, Signal.BUY, Signal.SELL) == Signal.SELL


def test_bull_sma_hold():
    assert composite_signal(Regime.BULL, Signal.HOLD, Signal.BUY) == Signal.HOLD


def test_choppy_xgb_hold():
    assert composite_signal(Regime.CHOPPY, Signal.BUY, Signal.HOLD) == Signal.HOLD
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd ~/git/SchroederTrader
uv run pytest tests/test_composite.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement composite.py**

Create `src/schroeder_trader/strategy/composite.py`:

```python
from schroeder_trader.strategy.regime_detector import Regime
from schroeder_trader.strategy.sma_crossover import Signal


def composite_signal(
    regime: Regime,
    sma_signal: Signal,
    xgb_signal: Signal,
) -> Signal:
    """Route to the appropriate strategy based on regime.

    Args:
        regime: Current market regime.
        sma_signal: Signal from SMA crossover strategy.
        xgb_signal: Signal from XGBoost classifier.

    Returns:
        The signal from the active strategy.
    """
    if regime in (Regime.BULL, Regime.BEAR):
        return sma_signal
    else:
        return xgb_signal
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd ~/git/SchroederTrader
uv run pytest tests/test_composite.py -v
```

Expected: 6 tests PASS

- [ ] **Step 5: Run full test suite**

```bash
cd ~/git/SchroederTrader
uv run pytest tests/ -v
```

Expected: All tests PASS (67 existing + 8 regime + 6 composite = 81)

- [ ] **Step 6: Commit**

```bash
cd ~/git/SchroederTrader
git add src/schroeder_trader/strategy/composite.py tests/test_composite.py
git commit -m "feat: composite signal router (SMA in trends, XGBoost in choppy)"
```

---

### Task 3: Composite Walk-Forward Evaluation Script

**Files:**
- Create: `backtest/composite_strategy.py`

This is the most complex task. It runs the full composite strategy through walk-forward validation and reports all gate metrics.

- [ ] **Step 1: Create the evaluation script**

Create `backtest/composite_strategy.py`:

```python
"""Walk-forward evaluation of the regime-switching composite strategy."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.train_model import load_data
from backtest.walk_forward import _compute_sharpe, TRAIN_YEARS, TEST_MONTHS
from schroeder_trader.config import SLIPPAGE_ESTIMATE
from schroeder_trader.strategy.feature_engineer import (
    FeaturePipeline,
    CLASS_DOWN,
    CLASS_FLAT,
    CLASS_UP,
)
from schroeder_trader.strategy.regime_detector import Regime, detect_regime
from schroeder_trader.strategy.composite import composite_signal
from schroeder_trader.strategy.sma_crossover import Signal, generate_signal
from schroeder_trader.strategy.xgboost_classifier import train_model

DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "results"

# XGBoost feature columns (best validated config from Phase 2.1)
XGB_FEATURES = [
    "log_return_5d", "log_return_20d", "volatility_20d",
    "credit_spread", "dollar_momentum", "regime_label",
]

# 20-day forward return label thresholds
LABEL_THRESHOLD = 0.01  # 1%

# XGBoost confidence threshold
XGB_CONFIDENCE = 0.50


def load_all_data():
    """Load SPY + external data, compute extended features with 20-day labels."""
    spy_df = load_data()

    ext_path = DATA_DIR / "features_daily.csv"
    if not ext_path.exists():
        raise FileNotFoundError(f"No external features at {ext_path}. Run download_features.py first.")
    ext_df = pd.read_csv(ext_path, index_col="date", parse_dates=True)

    pipeline = FeaturePipeline()
    features_df = pipeline.compute_features_extended(spy_df, ext_df)

    # Compute regime labels using backward-looking rolling median
    log_ret_20d = np.log(features_df["close"] / features_df["close"].shift(20))
    vol_20d = features_df["close"].pct_change().rolling(20).std()
    vol_median = vol_20d.rolling(252).median()

    features_df["regime_label_val"] = np.nan
    for i in range(len(features_df)):
        lr = log_ret_20d.iloc[i]
        vol = vol_20d.iloc[i]
        vm = vol_median.iloc[i]
        if not (pd.isna(lr) or pd.isna(vol) or pd.isna(vm)):
            features_df.iloc[i, features_df.columns.get_loc("regime_label_val")] = detect_regime(lr, vol, vm)

    # Map Regime enum to int for XGBoost feature
    regime_map = {Regime.BEAR: 0, Regime.CHOPPY: 1, Regime.BULL: 2}
    features_df["regime_label"] = features_df["regime_label_val"].map(regime_map)

    # 20-day forward return labels
    forward_return = features_df["close"].shift(-20) / features_df["close"] - 1
    features_df = features_df[forward_return.notna()].copy()
    forward_return = forward_return[forward_return.notna()]

    features_df["forward_return_5d_class"] = CLASS_FLAT
    features_df.loc[forward_return > LABEL_THRESHOLD, "forward_return_5d_class"] = CLASS_UP
    features_df.loc[forward_return < -LABEL_THRESHOLD, "forward_return_5d_class"] = CLASS_DOWN
    features_df["forward_return_5d_class"] = features_df["forward_return_5d_class"].astype(int)

    return spy_df, features_df


def run_composite_walkforward():
    """Run walk-forward evaluation of the composite strategy."""
    print("Loading data and computing features...")
    spy_df, features_df = load_all_data()
    print(f"Feature matrix: {len(features_df)} rows, "
          f"{features_df.index.min().date()} to {features_df.index.max().date()}")

    all_returns = []
    all_dates = []
    all_regimes = []
    all_signals = []
    all_sources = []  # "SMA" or "XGB" for each day

    start_date = features_df.index.min()
    end_date = features_df.index.max()
    train_start = start_date
    n_windows = 0

    while True:
        train_end = train_start + pd.DateOffset(years=TRAIN_YEARS)
        test_end = train_end + pd.DateOffset(months=TEST_MONTHS)
        if train_end >= end_date:
            break

        train_data = features_df[(features_df.index >= train_start) & (features_df.index < train_end)]
        test_data = features_df[(features_df.index >= train_end) & (features_df.index < test_end)]

        train_valid = train_data.dropna(subset=XGB_FEATURES)
        test_valid = test_data.dropna(subset=XGB_FEATURES)

        if len(train_valid) < 100 or len(test_valid) < 10:
            train_start += pd.DateOffset(months=TEST_MONTHS)
            continue

        # Train XGBoost on this window
        X_train = train_valid[XGB_FEATURES]
        y_train = train_valid["forward_return_5d_class"]
        val_split = int(len(X_train) * 0.8)

        # Skip window if training data doesn't have all 3 classes
        if len(y_train.unique()) < 3:
            train_start += pd.DateOffset(months=TEST_MONTHS)
            continue

        try:
            xgb_model = train_model(
                X_train[:val_split], y_train[:val_split],
                X_train[val_split:], y_train[val_split:],
            )
        except Exception as e:
            print(f"  Window training failed: {e}")
            train_start += pd.DateOffset(months=TEST_MONTHS)
            continue

        # Evaluate composite on test window
        close = test_valid["close"]
        window_returns = close.pct_change().fillna(0).values

        position = 0
        for i in range(len(test_valid)):
            # Today's return uses previous position (no look-ahead)
            strat_ret = position * window_returns[i]

            # Determine regime
            regime_val = test_valid["regime_label_val"].iloc[i]
            if pd.isna(regime_val):
                regime = Regime.CHOPPY  # fallback
            else:
                regime = regime_val

            # Get SMA signal: pass full price history up to this test day
            test_date = test_valid.index[i]
            spy_up_to_today = spy_df.loc[:test_date]
            if len(spy_up_to_today) >= 201:
                sma_signal, _, _ = generate_signal(spy_up_to_today)
            else:
                sma_signal = Signal.HOLD

            # Get XGBoost signal
            features_row = test_valid[XGB_FEATURES].iloc[[i]]
            proba = xgb_model.predict_proba(features_row)[0]
            pred_class = int(np.argmax(proba))
            if pred_class == CLASS_UP and proba[CLASS_UP] > XGB_CONFIDENCE:
                xgb_signal = Signal.BUY
            elif pred_class == CLASS_DOWN and proba[CLASS_DOWN] > XGB_CONFIDENCE:
                xgb_signal = Signal.SELL
            else:
                xgb_signal = Signal.HOLD

            # Route through composite
            signal = composite_signal(regime, sma_signal, xgb_signal)

            # Update position
            prev_position = position
            if signal == Signal.BUY:
                position = 1
            elif signal == Signal.SELL:
                position = 0
            # HOLD: maintain current position

            # Charge slippage on position change
            if position != prev_position:
                strat_ret -= SLIPPAGE_ESTIMATE

            all_returns.append(strat_ret)
            all_dates.append(test_date)
            all_regimes.append(regime)
            all_signals.append(signal)
            all_sources.append("SMA" if regime in (Regime.BULL, Regime.BEAR) else "XGB")

        n_windows += 1
        print(f"  Window {n_windows}: {train_start.date()}-{train_end.date()} -> "
              f"{train_end.date()}-{test_end.date()}, {len(test_valid)} days")

        train_start += pd.DateOffset(months=TEST_MONTHS)

    if not all_returns:
        print("No windows evaluated!")
        return

    # Compute metrics
    returns = np.array(all_returns)
    dates = pd.DatetimeIndex(all_dates)
    sources = np.array(all_sources)
    regimes = np.array(all_regimes)

    full_sharpe = _compute_sharpe(returns)

    post_2020_mask = dates >= "2020-01-01"
    post_sharpe = _compute_sharpe(returns[post_2020_mask]) if post_2020_mask.any() else 0.0

    # Max drawdown
    cumulative = (1 + returns).cumprod()
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = float(abs(drawdown.min())) * 100

    # Trade count
    position_series = []
    pos = 0
    for s in all_signals:
        if s == Signal.BUY:
            pos = 1
        elif s == Signal.SELL:
            pos = 0
        position_series.append(pos)
    position_arr = np.array(position_series)
    total_trades = int(np.sum(np.abs(np.diff(position_arr))))

    # Per-source trade count
    xgb_trades = 0
    for i in range(1, len(position_arr)):
        if position_arr[i] != position_arr[i - 1] and sources[i] == "XGB":
            xgb_trades += 1

    # Regime residency
    regime_counts = {r: 0 for r in [Regime.BULL, Regime.BEAR, Regime.CHOPPY]}
    for r in regimes:
        if r in regime_counts:
            regime_counts[r] += 1
    total_days = len(regimes)

    # Per-regime Sharpe
    regime_sharpes = {}
    for r in [Regime.BULL, Regime.BEAR, Regime.CHOPPY]:
        mask = np.array([reg == r for reg in regimes])
        if mask.any():
            regime_sharpes[r.value] = round(_compute_sharpe(returns[mask]), 4)
        else:
            regime_sharpes[r.value] = 0.0

    # Print results
    print("\n" + "=" * 60)
    print("COMPOSITE STRATEGY WALK-FORWARD RESULTS")
    print("=" * 60)
    print(f"\nFull-period Sharpe: {full_sharpe:.4f}  (target: >= 0.88)")
    print(f"Post-2020 Sharpe:   {post_sharpe:.4f}  (target: >= 0.80)")
    print(f"Max Drawdown:       {max_drawdown:.2f}%  (target: <= 25%)")
    print(f"Total Trades:       {total_trades}  (target: >= 30)")
    print(f"XGBoost Trades:     {xgb_trades}  (target: >= 10)")
    print(f"Windows Evaluated:  {n_windows}")

    print(f"\nRegime Residency:")
    for r in [Regime.BULL, Regime.BEAR, Regime.CHOPPY]:
        pct = regime_counts[r] / total_days * 100 if total_days > 0 else 0
        source = "SMA" if r in (Regime.BULL, Regime.BEAR) else "XGB"
        print(f"  {r.value:8s}: {regime_counts[r]:5d} days ({pct:5.1f}%) -> {source}  Sharpe={regime_sharpes[r.value]:.4f}")

    # Gate check
    gates = {
        "full_sharpe": full_sharpe >= 0.88,
        "post_2020_sharpe": post_sharpe >= 0.80,
        "max_drawdown": max_drawdown <= 25.0,
        "total_trades": total_trades >= 30,
        "xgb_trades": xgb_trades >= 10,
    }
    all_passed = all(gates.values())

    print(f"\nGate Results:")
    for name, passed in gates.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
    print(f"\nOverall: {'ALL GATES PASSED' if all_passed else 'GATES NOT MET'}")
    print("=" * 60)

    # Save results
    results = {
        "full_sharpe": round(full_sharpe, 4),
        "post_2020_sharpe": round(post_sharpe, 4),
        "max_drawdown_pct": round(max_drawdown, 2),
        "total_trades": total_trades,
        "xgb_trades": xgb_trades,
        "n_windows": n_windows,
        "regime_residency": {r.value: regime_counts[r] for r in regime_counts},
        "regime_sharpes": regime_sharpes,
        "gates_passed": all_passed,
        "gate_details": {k: v for k, v in gates.items()},
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "composite_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_DIR / 'composite_results.json'}")

    return results


if __name__ == "__main__":
    run_composite_walkforward()
```

- [ ] **Step 2: Run the composite evaluation**

```bash
cd ~/git/SchroederTrader
PYTHONPATH=. uv run python backtest/composite_strategy.py
```

This will take 10-20 minutes. Expected output: gate metrics table with PASS/FAIL for each criterion.

- [ ] **Step 3: Commit**

```bash
cd ~/git/SchroederTrader
git add backtest/composite_strategy.py
git commit -m "feat: composite strategy walk-forward evaluation with gate criteria"
```

---

### Task 4: Review Results

- [ ] **Step 1: Inspect the results**

```bash
cd ~/git/SchroederTrader
cat backtest/results/composite_results.json
```

Review:
- Did the composite meet all 4 gate criteria?
- What is the regime residency breakdown?
- Are per-regime Sharpe values sensible (positive in each component's domain)?

- [ ] **Step 2: Report to user**

Present the gate results table and regime breakdown. If all gates pass, recommend deploying to shadow. If not, identify which gates failed and discuss next steps.
