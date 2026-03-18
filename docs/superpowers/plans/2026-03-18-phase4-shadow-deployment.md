# Phase 4: Shadow Deployment — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deploy the validated composite strategy (SMA in Bull, flat/XGBoost in Bear, XGBoost in Choppy) to the daily pipeline's shadow mode for live forward-testing.

**Architecture:** Update composite signal routing to support hybrid bear, add new columns to shadow_signals table, replace shadow Step 10 in main.py with composite routing, train a final 20-day model.

**Tech Stack:** XGBoost, pandas, SQLite (all already installed)

**Spec:** `docs/superpowers/specs/2026-03-18-phase4-shadow-deployment-design.md`

---

## File Structure

```
SchroederTrader/
├── src/schroeder_trader/
│   ├── config.py                     # MODIFIED: add model + features paths
│   ├── strategy/
│   │   └── composite.py              # MODIFIED: add hybrid routing + bear counting
│   ├── storage/
│   │   └── trade_log.py              # MODIFIED: new columns + updated functions
│   └── main.py                       # MODIFIED: replace shadow Step 10
├── backtest/
│   └── train_final_composite.py      # NEW: train 20-day model
└── tests/
    ├── test_composite.py             # MODIFIED: hybrid + bear counting tests
    ├── test_shadow_logging.py        # MODIFIED: new columns tests
    └── test_main.py                  # MODIFIED: composite shadow tests
```

---

### Task 1: Config Paths + Hybrid Composite Routing

**Files:**
- Modify: `src/schroeder_trader/config.py`
- Modify: `src/schroeder_trader/strategy/composite.py`
- Modify: `tests/test_composite.py`

- [ ] **Step 1: Write failing tests for hybrid routing and bear counting**

Add to `tests/test_composite.py`:

```python
import numpy as np
import pandas as pd

from schroeder_trader.strategy.composite import (
    composite_signal,
    composite_signal_hybrid,
    count_consecutive_bear_days,
)
from schroeder_trader.strategy.regime_detector import Regime
from schroeder_trader.strategy.sma_crossover import Signal


# --- Existing tests stay as-is (test_bull_uses_sma, etc.) ---


# --- New: composite_signal_hybrid tests ---

def test_hybrid_bull_routes_to_sma():
    signal, source = composite_signal_hybrid(
        Regime.BULL, Signal.BUY, Signal.SELL, Signal.SELL, bear_days=0,
    )
    assert signal == Signal.BUY
    assert source == "SMA"


def test_hybrid_early_bear_routes_to_flat():
    signal, source = composite_signal_hybrid(
        Regime.BEAR, Signal.HOLD, Signal.BUY, Signal.BUY, bear_days=10,
    )
    assert signal == Signal.SELL
    assert source == "FLAT"


def test_hybrid_bear_day_20_still_flat():
    signal, source = composite_signal_hybrid(
        Regime.BEAR, Signal.HOLD, Signal.BUY, Signal.BUY, bear_days=20,
    )
    assert signal == Signal.SELL
    assert source == "FLAT"


def test_hybrid_bear_day_21_routes_to_xgb_high():
    signal, source = composite_signal_hybrid(
        Regime.BEAR, Signal.HOLD, Signal.SELL, Signal.BUY, bear_days=21,
    )
    assert signal == Signal.BUY  # xgb_signal_high
    assert source == "XGB"


def test_hybrid_choppy_routes_to_xgb_low():
    signal, source = composite_signal_hybrid(
        Regime.CHOPPY, Signal.BUY, Signal.SELL, Signal.HOLD, bear_days=0,
    )
    assert signal == Signal.SELL  # xgb_signal_low
    assert source == "XGB"


def test_hybrid_returns_tuple():
    result = composite_signal_hybrid(
        Regime.BULL, Signal.HOLD, Signal.HOLD, Signal.HOLD, bear_days=0,
    )
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], Signal)
    assert isinstance(result[1], str)


# --- New: count_consecutive_bear_days tests ---

def test_bear_count_all_bear():
    regimes = pd.Series([Regime.BEAR] * 10)
    assert count_consecutive_bear_days(regimes) == 10


def test_bear_count_last_not_bear():
    regimes = pd.Series([Regime.BEAR, Regime.BEAR, Regime.CHOPPY])
    assert count_consecutive_bear_days(regimes) == 0


def test_bear_count_mixed_ending_bear():
    regimes = pd.Series([Regime.BULL, Regime.CHOPPY, Regime.BEAR])
    assert count_consecutive_bear_days(regimes) == 1


def test_bear_count_interrupted_streak():
    regimes = pd.Series([Regime.BEAR, Regime.BEAR, Regime.CHOPPY, Regime.BEAR, Regime.BEAR])
    assert count_consecutive_bear_days(regimes) == 2


def test_bear_count_last_nan():
    regimes = pd.Series([Regime.BEAR, Regime.BEAR, np.nan])
    assert count_consecutive_bear_days(regimes) == 0


def test_bear_count_empty():
    regimes = pd.Series([], dtype=object)
    assert count_consecutive_bear_days(regimes) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd ~/git/SchroederTrader
uv run pytest tests/test_composite.py -v -k "hybrid or bear_count"
```

Expected: FAIL — `ImportError`

- [ ] **Step 3: Add config paths**

Add to `src/schroeder_trader/config.py` after `LOG_DIR`:

```python
# Model paths
COMPOSITE_MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_spy_20d.json"
FEATURES_CSV_PATH = PROJECT_ROOT / "backtest" / "data" / "features_daily.csv"
```

- [ ] **Step 4: Implement hybrid routing and bear counting**

Replace the contents of `src/schroeder_trader/strategy/composite.py` with:

```python
import numpy as np
import pandas as pd

from schroeder_trader.strategy.regime_detector import Regime
from schroeder_trader.strategy.sma_crossover import Signal


def composite_signal(
    regime: Regime,
    sma_signal: Signal,
    xgb_signal: Signal,
) -> Signal:
    """Route to the appropriate strategy based on regime (Phase 3 simple version)."""
    if regime in (Regime.BULL, Regime.BEAR):
        return sma_signal
    else:
        return xgb_signal


def composite_signal_hybrid(
    regime: Regime,
    sma_signal: Signal,
    xgb_signal_low: Signal,
    xgb_signal_high: Signal,
    bear_days: int,
    late_bear_threshold: int = 20,
) -> tuple[Signal, str]:
    """Route signal based on regime and bear duration.

    Args:
        regime: Current market regime.
        sma_signal: Signal from SMA crossover.
        xgb_signal_low: XGBoost signal at 0.35 threshold (for Choppy).
        xgb_signal_high: XGBoost signal at 0.50 threshold (for late Bear).
        bear_days: Consecutive days in BEAR regime.
        late_bear_threshold: Days before switching from flat to XGBoost in bear.

    Returns:
        Tuple of (Signal, source) where source is "SMA", "FLAT", or "XGB".
    """
    if regime == Regime.BULL:
        return sma_signal, "SMA"
    elif regime == Regime.BEAR:
        if bear_days <= late_bear_threshold:
            return Signal.SELL, "FLAT"
        else:
            return xgb_signal_high, "XGB"
    else:
        return xgb_signal_low, "XGB"


def count_consecutive_bear_days(regimes: pd.Series) -> int:
    """Count consecutive BEAR days ending at the last row.

    Args:
        regimes: Series of Regime enum values. NaN treated as non-BEAR.

    Returns:
        Number of consecutive BEAR days ending at the last row.
        Returns 0 if the last row is not BEAR, is NaN, or series is empty.
    """
    if len(regimes) == 0:
        return 0

    last = regimes.iloc[-1]
    if not isinstance(last, Regime) or last != Regime.BEAR:
        return 0

    count = 0
    for i in range(len(regimes) - 1, -1, -1):
        val = regimes.iloc[i]
        if isinstance(val, Regime) and val == Regime.BEAR:
            count += 1
        else:
            break

    return count
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd ~/git/SchroederTrader
uv run pytest tests/test_composite.py -v
```

Expected: All tests PASS (6 existing + 12 new = 18)

- [ ] **Step 6: Commit**

```bash
cd ~/git/SchroederTrader
git add src/schroeder_trader/config.py src/schroeder_trader/strategy/composite.py tests/test_composite.py
git commit -m "feat: hybrid composite routing with late-bear XGBoost and bear day counting"
```

---

### Task 2: Shadow Signals Schema Update

**Files:**
- Modify: `src/schroeder_trader/storage/trade_log.py`
- Modify: `tests/test_shadow_logging.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_shadow_logging.py`:

```python
def test_shadow_signal_with_regime_and_source(tmp_path):
    db_path = tmp_path / "test.db"
    conn = init_db(db_path)
    now = datetime.now(timezone.utc)

    signal_id = log_shadow_signal(
        conn, now, "SPY", 523.10,
        predicted_class=None,
        predicted_proba=None,
        ml_signal="SELL",
        sma_signal="HOLD",
        regime="BEAR",
        signal_source="FLAT",
        bear_day_count=5,
    )
    assert signal_id is not None

    row = conn.execute("SELECT * FROM shadow_signals WHERE id = ?", (signal_id,)).fetchone()
    assert row["regime"] == "BEAR"
    assert row["signal_source"] == "FLAT"
    assert row["bear_day_count"] == 5
    assert row["predicted_class"] is None
    assert row["predicted_proba"] is None
    conn.close()


def test_shadow_signal_xgb_with_proba(tmp_path):
    db_path = tmp_path / "test.db"
    conn = init_db(db_path)
    now = datetime.now(timezone.utc)
    proba = json.dumps({"DOWN": 0.1, "FLAT": 0.3, "UP": 0.6})

    signal_id = log_shadow_signal(
        conn, now, "SPY", 523.10,
        predicted_class=2,
        predicted_proba=proba,
        ml_signal="BUY",
        sma_signal="HOLD",
        regime="CHOPPY",
        signal_source="XGB",
        bear_day_count=None,
    )

    row = conn.execute("SELECT * FROM shadow_signals WHERE id = ?", (signal_id,)).fetchone()
    assert row["predicted_class"] == 2
    assert row["regime"] == "CHOPPY"
    assert row["signal_source"] == "XGB"
    assert row["bear_day_count"] is None
    conn.close()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd ~/git/SchroederTrader
uv run pytest tests/test_shadow_logging.py -v -k "regime_and_source or xgb_with_proba"
```

Expected: FAIL — `TypeError` (unexpected keyword arguments)

- [ ] **Step 3: Update trade_log.py**

In `init_db()`, replace the shadow_signals CREATE TABLE with a version that allows NULLs and has the new columns. Also add defensive ALTER TABLE for existing databases:

Replace the shadow_signals CREATE TABLE block (lines 46-57) with:

```python
    conn.execute("""
        CREATE TABLE IF NOT EXISTS shadow_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            ticker TEXT NOT NULL,
            close_price REAL NOT NULL,
            predicted_class INTEGER,
            predicted_proba TEXT,
            ml_signal TEXT NOT NULL,
            sma_signal TEXT NOT NULL,
            regime TEXT,
            signal_source TEXT,
            bear_day_count INTEGER
        )
    """)
    # Defensive migration for existing databases missing new columns
    for col, col_type in [("regime", "TEXT"), ("signal_source", "TEXT"), ("bear_day_count", "INTEGER")]:
        try:
            conn.execute(f"ALTER TABLE shadow_signals ADD COLUMN {col} {col_type}")
        except sqlite3.OperationalError:
            pass  # column already exists
```

Update `log_shadow_signal()`:

```python
def log_shadow_signal(
    conn: sqlite3.Connection,
    timestamp: datetime,
    ticker: str,
    close_price: float,
    predicted_class: int | None,
    predicted_proba: str | None,
    ml_signal: str,
    sma_signal: str,
    regime: str | None = None,
    signal_source: str | None = None,
    bear_day_count: int | None = None,
) -> int:
    cursor = conn.execute(
        "INSERT INTO shadow_signals (timestamp, ticker, close_price, predicted_class, predicted_proba, ml_signal, sma_signal, regime, signal_source, bear_day_count) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (timestamp.isoformat(), ticker, close_price, predicted_class, predicted_proba, ml_signal, sma_signal, regime, signal_source, bear_day_count),
    )
    conn.commit()
    return cursor.lastrowid
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd ~/git/SchroederTrader
uv run pytest tests/test_shadow_logging.py -v
```

Expected: All tests PASS (4 existing + 2 new = 6)

- [ ] **Step 5: Verify full suite**

```bash
cd ~/git/SchroederTrader
uv run pytest tests/ -v
```

Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
cd ~/git/SchroederTrader
git add src/schroeder_trader/storage/trade_log.py tests/test_shadow_logging.py
git commit -m "feat: add regime, signal_source, bear_day_count columns to shadow_signals"
```

---

### Task 3: Replace Shadow Step 10 with Composite Routing

**Files:**
- Modify: `src/schroeder_trader/main.py`
- Modify: `tests/test_main.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_main.py` (replace the two existing shadow tests with updated versions):

```python
@patch("schroeder_trader.main.load_model")
@patch("schroeder_trader.main.send_daily_summary")
@patch("schroeder_trader.main.get_account")
@patch("schroeder_trader.main.get_position")
@patch("schroeder_trader.main.fetch_daily_bars")
@patch("schroeder_trader.main.is_market_open_today")
@patch("schroeder_trader.main.get_pending_orders")
@patch("schroeder_trader.main.get_signal_by_date")
def test_pipeline_shadow_composite_skips_when_no_model(
    mock_signal_by_date,
    mock_pending,
    mock_market_open,
    mock_fetch,
    mock_position,
    mock_account,
    mock_summary,
    mock_load_model,
    tmp_path,
):
    """Composite shadow should silently skip when no model file exists."""
    mock_signal_by_date.return_value = None
    mock_pending.return_value = []
    mock_market_open.return_value = True
    mock_fetch.return_value = _mock_bars_df()
    mock_position.return_value = 0
    mock_account.return_value = {"portfolio_value": 10000.0, "cash": 10000.0}
    mock_load_model.return_value = None

    db_path = tmp_path / "test.db"
    run_pipeline(db_path=db_path)

    mock_summary.assert_called_once()

    import sqlite3
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    shadow_count = conn.execute("SELECT COUNT(*) FROM shadow_signals").fetchone()[0]
    assert shadow_count == 0
    conn.close()


@patch("schroeder_trader.main.load_model")
@patch("schroeder_trader.main.send_daily_summary")
@patch("schroeder_trader.main.get_account")
@patch("schroeder_trader.main.get_position")
@patch("schroeder_trader.main.fetch_daily_bars")
@patch("schroeder_trader.main.is_market_open_today")
@patch("schroeder_trader.main.get_pending_orders")
@patch("schroeder_trader.main.get_signal_by_date")
def test_pipeline_shadow_exception_does_not_affect_sma(
    mock_signal_by_date,
    mock_pending,
    mock_market_open,
    mock_fetch,
    mock_position,
    mock_account,
    mock_summary,
    mock_load_model,
    tmp_path,
):
    """An exception in shadow Step 10 must not prevent Steps 1-9 from completing."""
    mock_signal_by_date.return_value = None
    mock_pending.return_value = []
    mock_market_open.return_value = True
    mock_fetch.return_value = _mock_bars_df()
    mock_position.return_value = 0
    mock_account.return_value = {"portfolio_value": 10000.0, "cash": 10000.0}
    mock_load_model.side_effect = RuntimeError("model explosion")

    db_path = tmp_path / "test.db"
    run_pipeline(db_path=db_path)

    # SMA pipeline should have completed
    mock_summary.assert_called_once()
    import sqlite3
    conn = sqlite3.connect(str(db_path))
    row = conn.execute("SELECT signal FROM signals").fetchone()
    assert row[0] == "HOLD"
    conn.close()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd ~/git/SchroederTrader
uv run pytest tests/test_main.py -v -k "shadow_composite or shadow_exception"
```

Expected: FAIL (import errors or wrong function signatures)

- [ ] **Step 3: Update main.py**

Replace the imports and shadow Step 10 in `src/schroeder_trader/main.py`.

Update imports — replace the old shadow imports:

```python
import json
from schroeder_trader.config import PROJECT_ROOT
from schroeder_trader.strategy.feature_engineer import FeaturePipeline
from schroeder_trader.strategy.xgboost_classifier import load_model, predict_signal
from schroeder_trader.storage.trade_log import log_shadow_signal
```

With:

```python
import json
import subprocess
import numpy as np
import pandas as pd
from schroeder_trader.config import PROJECT_ROOT, COMPOSITE_MODEL_PATH, FEATURES_CSV_PATH, TICKER
from schroeder_trader.strategy.feature_engineer import FeaturePipeline, CLASS_DOWN, CLASS_UP
from schroeder_trader.strategy.xgboost_classifier import load_model
from schroeder_trader.strategy.regime_detector import Regime, detect_regime
from schroeder_trader.strategy.composite import composite_signal_hybrid, count_consecutive_bear_days
from schroeder_trader.storage.trade_log import log_shadow_signal
```

Remove the old `MODEL_PATH` constant and replace with `COMPOSITE_MODEL_PATH` from config.

Replace the entire Step 10 block (from `# Step 10: Shadow ML prediction` through the except) with:

```python
    # Step 10: Composite shadow signal
    try:
        # Fetch/update external features (idempotent, skips if <24h old)
        try:
            subprocess.run(
                ["uv", "run", "python", str(PROJECT_ROOT / "backtest" / "download_features.py")],
                cwd=str(PROJECT_ROOT), capture_output=True, timeout=120,
            )
        except Exception:
            logger.warning("External feature download failed, using cached data")

        # Load model
        model = load_model(COMPOSITE_MODEL_PATH)
        if model is None:
            logger.info("No composite model at %s, skipping shadow step", COMPOSITE_MODEL_PATH)
        else:
            # Validate class ordering
            expected_classes = [0, 1, 2]
            if list(model.classes_) != expected_classes:
                logger.error("Model classes %s don't match expected %s", model.classes_, expected_classes)
            else:
                class_to_idx = {int(c): i for i, c in enumerate(model.classes_)}
                idx_up = class_to_idx[2]
                idx_down = class_to_idx[0]

                # Load external features
                ext_df = pd.read_csv(str(FEATURES_CSV_PATH), index_col="date", parse_dates=True)

                # Fetch 400 days of SPY data
                shadow_df = fetch_daily_bars(TICKER, days=400)

                # Compute extended features
                pipeline = FeaturePipeline()
                features = pipeline.compute_features_extended(shadow_df, ext_df)

                if len(features) > 0:
                    # Compute regime labels for all rows (backward-looking)
                    log_ret_20d = np.log(features["close"] / features["close"].shift(20))
                    vol_20d = features["close"].pct_change().rolling(20).std()
                    vol_med = vol_20d.rolling(252).median()

                    regime_series = pd.Series(index=features.index, dtype=object)
                    for idx in range(len(features)):
                        lr = log_ret_20d.iloc[idx]
                        vol = vol_20d.iloc[idx]
                        vm = vol_med.iloc[idx]
                        if pd.isna(lr) or pd.isna(vol) or pd.isna(vm):
                            regime_series.iloc[idx] = np.nan
                        else:
                            regime_series.iloc[idx] = detect_regime(lr, vol, vm)

                    # Add regime_label as integer feature
                    regime_map = {Regime.BEAR: 0, Regime.CHOPPY: 1, Regime.BULL: 2}
                    features["regime_label"] = regime_series.map(regime_map)

                    # Get today's data
                    last_row = features.iloc[[-1]]
                    today_regime = regime_series.iloc[-1]
                    if not isinstance(today_regime, Regime):
                        today_regime = Regime.CHOPPY

                    bear_days = count_consecutive_bear_days(regime_series)

                    # Generate XGBoost signals at both thresholds
                    feature_cols = [
                        "log_return_5d", "log_return_20d", "volatility_20d",
                        "credit_spread", "dollar_momentum", "regime_label",
                    ]
                    proba = model.predict_proba(last_row[feature_cols])[0]
                    pred_class = int(np.argmax(proba))
                    proba_dict = {
                        "DOWN": float(proba[idx_down]),
                        "FLAT": float(proba[class_to_idx[1]]),
                        "UP": float(proba[idx_up]),
                    }

                    # Low threshold (0.35) for Choppy
                    if pred_class == idx_up and proba[idx_up] > 0.35:
                        xgb_low = Signal.BUY
                    elif pred_class == idx_down and proba[idx_down] > 0.35:
                        xgb_low = Signal.SELL
                    else:
                        xgb_low = Signal.HOLD

                    # High threshold (0.50) for late Bear
                    if pred_class == idx_up and proba[idx_up] > 0.50:
                        xgb_high = Signal.BUY
                    elif pred_class == idx_down and proba[idx_down] > 0.50:
                        xgb_high = Signal.SELL
                    else:
                        xgb_high = Signal.HOLD

                    # Route composite signal
                    composite_sig, source = composite_signal_hybrid(
                        today_regime, signal, xgb_low, xgb_high, bear_days,
                    )

                    # Log shadow signal
                    log_shadow_signal(
                        conn, now, TICKER, close_price,
                        predicted_class=pred_class if source == "XGB" else None,
                        predicted_proba=json.dumps(proba_dict) if source == "XGB" else None,
                        ml_signal=composite_sig.value,
                        sma_signal=signal.value,
                        regime=today_regime.value,
                        signal_source=source,
                        bear_day_count=bear_days if today_regime == Regime.BEAR else None,
                    )
                    logger.info(
                        "Shadow composite: %s (source=%s, regime=%s, bear_days=%d)",
                        composite_sig.value, source, today_regime.value, bear_days,
                    )
    except Exception:
        logger.exception("Shadow composite prediction failed (non-fatal)")
```

Also add `from schroeder_trader.strategy.sma_crossover import Signal` to imports if not already present (it should be via `generate_signal` but verify).

- [ ] **Step 4: Run tests**

```bash
cd ~/git/SchroederTrader
uv run pytest tests/test_main.py -v
```

Expected: All tests PASS

- [ ] **Step 5: Run full suite**

```bash
cd ~/git/SchroederTrader
uv run pytest tests/ -v
```

Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
cd ~/git/SchroederTrader
git add src/schroeder_trader/main.py tests/test_main.py
git commit -m "feat: replace shadow Step 10 with composite signal routing"
```

---

### Task 4: Train Final Composite Model

**Files:**
- Create: `backtest/train_final_composite.py`

- [ ] **Step 1: Create training script**

Create `backtest/train_final_composite.py`:

```python
"""Train the final XGBoost model for composite shadow deployment.

Uses 20-day forward return labels and the 6-feature set validated in Phase 2.1/3.
Determines optimal n_estimators from walk-forward, then retrains on all data.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.train_model import load_data
from backtest.walk_forward import TRAIN_YEARS, TEST_MONTHS
from schroeder_trader.config import COMPOSITE_MODEL_PATH
from schroeder_trader.strategy.feature_engineer import (
    FeaturePipeline,
    CLASS_DOWN,
    CLASS_FLAT,
    CLASS_UP,
)
from schroeder_trader.strategy.regime_detector import Regime, detect_regime
from schroeder_trader.strategy.xgboost_classifier import save_model
from xgboost import XGBClassifier

DATA_DIR = Path(__file__).parent / "data"

XGB_FEATURES = [
    "log_return_5d", "log_return_20d", "volatility_20d",
    "credit_spread", "dollar_momentum", "regime_label",
]

LABEL_THRESHOLD = 0.01  # 1% for 20-day horizon

XGB_PARAMS = {
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "random_state": 42,
}


def prepare_features() -> pd.DataFrame:
    """Load SPY + external data, compute features with 20-day labels and regime."""
    spy_df = load_data()

    ext_path = DATA_DIR / "features_daily.csv"
    if not ext_path.exists():
        raise FileNotFoundError(f"No external features at {ext_path}. Run download_features.py first.")
    ext_df = pd.read_csv(ext_path, index_col="date", parse_dates=True)

    pipeline = FeaturePipeline()
    features_df = pipeline.compute_features_extended(spy_df, ext_df)

    # Compute regime labels (backward-looking)
    log_ret_20d = np.log(features_df["close"] / features_df["close"].shift(20))
    vol_20d = features_df["close"].pct_change().rolling(20).std()
    vol_median = vol_20d.rolling(252).median()

    regime_vals = []
    for i in range(len(features_df)):
        lr = log_ret_20d.iloc[i]
        vol = vol_20d.iloc[i]
        vm = vol_median.iloc[i]
        if pd.isna(lr) or pd.isna(vol) or pd.isna(vm):
            regime_vals.append(np.nan)
        else:
            regime_vals.append(detect_regime(lr, vol, vm))

    regime_map = {Regime.BEAR: 0, Regime.CHOPPY: 1, Regime.BULL: 2}
    features_df["regime_label"] = pd.Series(regime_vals, index=features_df.index).map(regime_map)

    # 20-day forward return labels
    forward_return = features_df["close"].shift(-20) / features_df["close"] - 1
    features_df = features_df[forward_return.notna()].copy()
    forward_return = forward_return[forward_return.notna()]

    features_df["label"] = CLASS_FLAT
    features_df.loc[forward_return > LABEL_THRESHOLD, "label"] = CLASS_UP
    features_df.loc[forward_return < -LABEL_THRESHOLD, "label"] = CLASS_DOWN
    features_df["label"] = features_df["label"].astype(int)

    features_df = features_df.dropna(subset=XGB_FEATURES)

    return features_df


def find_median_n_estimators(features_df: pd.DataFrame) -> int:
    """Run walk-forward to find median optimal n_estimators."""
    best_iterations = []

    start_date = features_df.index.min()
    end_date = features_df.index.max()
    train_start = start_date

    while True:
        train_end = train_start + pd.DateOffset(years=TRAIN_YEARS)
        test_end = train_end + pd.DateOffset(months=TEST_MONTHS)
        if train_end >= end_date:
            break

        train_data = features_df[(features_df.index >= train_start) & (features_df.index < train_end)]
        train_valid = train_data.dropna(subset=XGB_FEATURES)

        if len(train_valid) < 100 or len(train_valid["label"].unique()) < 3:
            train_start += pd.DateOffset(months=TEST_MONTHS)
            continue

        X = train_valid[XGB_FEATURES]
        y = train_valid["label"]
        val_split = int(len(X) * 0.8)

        model = XGBClassifier(**XGB_PARAMS, n_estimators=500, early_stopping_rounds=20)
        model.fit(X[:val_split], y[:val_split], eval_set=[(X[val_split:], y[val_split:])], verbose=False)
        best_iterations.append(model.best_iteration)

        train_start += pd.DateOffset(months=TEST_MONTHS)

    median_n = int(np.median(best_iterations))
    print(f"Walk-forward best iterations: {best_iterations}")
    print(f"Median n_estimators: {median_n}")
    return median_n


def train_and_save():
    """Train final model on all data and save."""
    print("Preparing features...")
    features_df = prepare_features()
    print(f"Feature matrix: {len(features_df)} rows, "
          f"{features_df.index.min().date()} to {features_df.index.max().date()}")

    print("\nFinding optimal n_estimators via walk-forward...")
    n_estimators = find_median_n_estimators(features_df)

    print(f"\nTraining final model on all {len(features_df)} rows with n_estimators={n_estimators}...")
    X = features_df[XGB_FEATURES]
    y = features_df["label"]

    model = XGBClassifier(**XGB_PARAMS, n_estimators=n_estimators)
    model.fit(X, y, verbose=False)

    COMPOSITE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_model(model, COMPOSITE_MODEL_PATH)
    print(f"Model saved to {COMPOSITE_MODEL_PATH}")
    print(f"Model classes: {model.classes_}")
    print(f"Training rows: {len(X)}, n_estimators: {n_estimators}")


if __name__ == "__main__":
    train_and_save()
```

- [ ] **Step 2: Run the training script**

```bash
cd ~/git/SchroederTrader
PYTHONPATH=. uv run python backtest/train_final_composite.py
```

Expected: Prints walk-forward iterations, median n_estimators, trains final model, saves to `models/xgboost_spy_20d.json`.

- [ ] **Step 3: Verify model loads**

```bash
cd ~/git/SchroederTrader
uv run python -c "
from schroeder_trader.strategy.xgboost_classifier import load_model
from schroeder_trader.config import COMPOSITE_MODEL_PATH
model = load_model(COMPOSITE_MODEL_PATH)
print(f'Model loaded: {model is not None}')
print(f'Classes: {model.classes_}')
"
```

Expected: `Model loaded: True`, `Classes: [0 1 2]`

- [ ] **Step 4: Commit**

```bash
cd ~/git/SchroederTrader
git add backtest/train_final_composite.py
git commit -m "feat: composite model training script with walk-forward n_estimators selection"
```

---

### Task 5: Final Integration Verification

- [ ] **Step 1: Run full test suite**

```bash
cd ~/git/SchroederTrader
uv run pytest tests/ -v
```

Expected: All tests PASS

- [ ] **Step 2: Verify model file exists**

```bash
ls -la models/xgboost_spy_20d.json
```

Expected: File exists

- [ ] **Step 3: Report to user**

Present: test results, model status, readiness for first shadow run.
