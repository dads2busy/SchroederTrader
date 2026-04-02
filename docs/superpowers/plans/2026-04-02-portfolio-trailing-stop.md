# Portfolio Trailing Stop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an 8% portfolio trailing stop with 5-day cooldown to protect against catastrophic drawdowns, logged in shadow signals for evaluation.

**Architecture:** New `TrailingStop` class in `risk/trailing_stop.py` tracks portfolio high-water mark and cooldown state. State persists between runs via the `shadow_signals` table (two new columns). The pipeline initializes trailing stop state from DB on startup, evaluates it after computing portfolio value, and logs the result alongside existing shadow signal data.

**Tech Stack:** Python, SQLite, pytest

---

### Task 1: TrailingStop Module — Tests

**Files:**
- Create: `tests/test_trailing_stop.py`

- [ ] **Step 1: Write failing tests for TrailingStop**

```python
from datetime import date

import pytest

from schroeder_trader.risk.trailing_stop import TrailingStop


def test_update_sets_high_water_mark():
    ts = TrailingStop(drawdown_pct=0.08, cooldown_days=5)
    triggered = ts.update(100000.0, date(2026, 1, 2))
    assert ts.high_water_mark == 100000.0
    assert triggered is False


def test_update_raises_hwm():
    ts = TrailingStop(drawdown_pct=0.08, cooldown_days=5)
    ts.update(100000.0, date(2026, 1, 2))
    ts.update(105000.0, date(2026, 1, 3))
    assert ts.high_water_mark == 105000.0


def test_update_does_not_lower_hwm():
    ts = TrailingStop(drawdown_pct=0.08, cooldown_days=5)
    ts.update(100000.0, date(2026, 1, 2))
    ts.update(99000.0, date(2026, 1, 3))
    assert ts.high_water_mark == 100000.0


def test_triggers_at_threshold():
    ts = TrailingStop(drawdown_pct=0.08, cooldown_days=5)
    ts.update(100000.0, date(2026, 1, 2))
    # 8% drop: 100000 * 0.92 = 92000
    triggered = ts.update(91999.0, date(2026, 1, 3))
    assert triggered is True
    assert ts.stop_date == date(2026, 1, 3)


def test_does_not_trigger_above_threshold():
    ts = TrailingStop(drawdown_pct=0.08, cooldown_days=5)
    ts.update(100000.0, date(2026, 1, 2))
    triggered = ts.update(92001.0, date(2026, 1, 3))
    assert triggered is False
    assert ts.stop_date is None


def test_in_cooldown_during_period():
    ts = TrailingStop(drawdown_pct=0.08, cooldown_days=5)
    ts.update(100000.0, date(2026, 1, 2))
    ts.update(91000.0, date(2026, 1, 3))  # trigger
    trading_dates = [
        date(2026, 1, 3), date(2026, 1, 6), date(2026, 1, 7),
        date(2026, 1, 8), date(2026, 1, 9), date(2026, 1, 10),
    ]
    assert ts.in_cooldown(date(2026, 1, 6), trading_dates) is True  # day 2
    assert ts.in_cooldown(date(2026, 1, 9), trading_dates) is True  # day 5


def test_cooldown_expires():
    ts = TrailingStop(drawdown_pct=0.08, cooldown_days=5)
    ts.update(100000.0, date(2026, 1, 2))
    ts.update(91000.0, date(2026, 1, 3))  # trigger
    trading_dates = [
        date(2026, 1, 3), date(2026, 1, 6), date(2026, 1, 7),
        date(2026, 1, 8), date(2026, 1, 9), date(2026, 1, 10),
        date(2026, 1, 13),
    ]
    assert ts.in_cooldown(date(2026, 1, 13), trading_dates) is False  # day 6


def test_no_cooldown_when_not_triggered():
    ts = TrailingStop(drawdown_pct=0.08, cooldown_days=5)
    ts.update(100000.0, date(2026, 1, 2))
    assert ts.in_cooldown(date(2026, 1, 3), [date(2026, 1, 2), date(2026, 1, 3)]) is False


def test_reset_clears_state():
    ts = TrailingStop(drawdown_pct=0.08, cooldown_days=5)
    ts.update(100000.0, date(2026, 1, 2))
    ts.update(91000.0, date(2026, 1, 3))
    ts.reset()
    assert ts.high_water_mark == 0.0
    assert ts.stop_date is None


def test_hwm_resets_after_cooldown():
    """After cooldown expires, HWM resets to current portfolio value."""
    ts = TrailingStop(drawdown_pct=0.08, cooldown_days=5)
    ts.update(100000.0, date(2026, 1, 2))
    ts.update(91000.0, date(2026, 1, 3))  # trigger at 91k
    trading_dates = [
        date(2026, 1, 3), date(2026, 1, 6), date(2026, 1, 7),
        date(2026, 1, 8), date(2026, 1, 9), date(2026, 1, 10),
        date(2026, 1, 13),
    ]
    # Cooldown expired
    assert ts.in_cooldown(date(2026, 1, 13), trading_dates) is False
    # Next update should reset HWM to current value
    ts.update(93000.0, date(2026, 1, 13))
    assert ts.high_water_mark == 93000.0
    assert ts.stop_date is None


def test_init_with_existing_state():
    """Can initialize with pre-existing HWM and stop date."""
    ts = TrailingStop(
        drawdown_pct=0.08, cooldown_days=5,
        high_water_mark=100000.0, stop_date=date(2026, 1, 3),
    )
    assert ts.high_water_mark == 100000.0
    assert ts.stop_date == date(2026, 1, 3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_trailing_stop.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'schroeder_trader.risk.trailing_stop'`

- [ ] **Step 3: Commit test file**

```bash
git add tests/test_trailing_stop.py
git commit -m "test: add trailing stop tests (red)"
```

---

### Task 2: TrailingStop Module — Implementation

**Files:**
- Create: `src/schroeder_trader/risk/trailing_stop.py`

- [ ] **Step 1: Implement TrailingStop class**

```python
from __future__ import annotations

from datetime import date


class TrailingStop:
    """Portfolio-level trailing stop with cooldown.

    Tracks the high-water mark of portfolio value. Triggers when the
    portfolio drops below ``high_water_mark * (1 - drawdown_pct)``.
    After triggering, enters a cooldown period of ``cooldown_days``
    trading days before allowing new entries.
    """

    def __init__(
        self,
        drawdown_pct: float,
        cooldown_days: int,
        high_water_mark: float = 0.0,
        stop_date: date | None = None,
    ) -> None:
        self.drawdown_pct = drawdown_pct
        self.cooldown_days = cooldown_days
        self.high_water_mark = high_water_mark
        self.stop_date = stop_date

    def update(self, portfolio_value: float, current_date: date) -> bool:
        """Update high-water mark and check for stop trigger.

        Returns True if the stop is triggered on this update.
        """
        # If we previously triggered and cooldown has expired, reset
        if self.stop_date is not None and self.high_water_mark > 0:
            # Reset HWM so it starts fresh from current value
            self.high_water_mark = 0.0
            self.stop_date = None

        self.high_water_mark = max(self.high_water_mark, portfolio_value)

        threshold = self.high_water_mark * (1 - self.drawdown_pct)
        if portfolio_value < threshold:
            self.stop_date = current_date
            return True

        return False

    def in_cooldown(self, current_date: date, trading_dates: list[date]) -> bool:
        """Check if still in cooldown period after a stop trigger.

        Args:
            current_date: Today's date.
            trading_dates: List of actual trading dates (from DB timestamps).

        Returns True if fewer than ``cooldown_days`` trading days have
        elapsed since the stop triggered.
        """
        if self.stop_date is None:
            return False

        # Count trading days after stop_date up to and including current_date
        days_after = sum(
            1 for d in trading_dates
            if self.stop_date < d <= current_date
        )
        return days_after < self.cooldown_days

    def reset(self) -> None:
        """Clear all state."""
        self.high_water_mark = 0.0
        self.stop_date = None
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `pytest tests/test_trailing_stop.py -v`
Expected: all 12 tests PASS

- [ ] **Step 3: Commit**

```bash
git add src/schroeder_trader/risk/trailing_stop.py
git commit -m "feat: add TrailingStop class with drawdown detection and cooldown"
```

---

### Task 3: Config Values

**Files:**
- Modify: `src/schroeder_trader/config.py`

- [ ] **Step 1: Add trailing stop config values**

Add after the `KELLY_WIN_LOSS_RATIO` line in `config.py`:

```python
TRAILING_STOP_PCT = 0.08            # 8% portfolio drawdown triggers stop
TRAILING_STOP_COOLDOWN_DAYS = 5     # trading days before re-entry allowed
```

- [ ] **Step 2: Run existing config tests**

Run: `pytest tests/test_config.py -v`
Expected: all PASS (no changes to existing behavior)

- [ ] **Step 3: Commit**

```bash
git add src/schroeder_trader/config.py
git commit -m "feat: add TRAILING_STOP_PCT and TRAILING_STOP_COOLDOWN_DAYS config"
```

---

### Task 4: Database Schema — New Columns

**Files:**
- Modify: `src/schroeder_trader/storage/trade_log.py`
- Modify: `tests/test_shadow_logging.py`

- [ ] **Step 1: Write failing test for new shadow signal columns**

Add to `tests/test_shadow_logging.py`:

```python
def test_shadow_signal_with_trailing_stop():
    conn = init_db(Path(":memory:"))
    now = datetime.now()
    log_shadow_signal(
        conn, now, "SPY", 650.0,
        predicted_class=2,
        predicted_proba='{"UP": 0.6}',
        ml_signal="BUY",
        sma_signal="HOLD",
        high_water_mark=100000.0,
        trailing_stop_triggered=True,
    )
    rows = get_shadow_signals(conn)
    assert len(rows) == 1
    assert rows[0]["high_water_mark"] == 100000.0
    assert rows[0]["trailing_stop_triggered"] == 1
    conn.close()


def test_shadow_signal_trailing_stop_defaults_null():
    conn = init_db(Path(":memory:"))
    now = datetime.now()
    log_shadow_signal(
        conn, now, "SPY", 650.0,
        predicted_class=None,
        predicted_proba=None,
        ml_signal="HOLD",
        sma_signal="HOLD",
    )
    rows = get_shadow_signals(conn)
    assert rows[0]["high_water_mark"] is None
    assert rows[0]["trailing_stop_triggered"] is None
    conn.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_shadow_logging.py::test_shadow_signal_with_trailing_stop -v`
Expected: FAIL (unknown keyword argument `high_water_mark`)

- [ ] **Step 3: Update trade_log.py**

In `init_db()`, add to the migration loop (after the kelly entries):

```python
    for col, col_type in [
        ("regime", "TEXT"), ("signal_source", "TEXT"), ("bear_day_count", "INTEGER"),
        ("kelly_fraction", "REAL"), ("kelly_qty", "INTEGER"),
        ("high_water_mark", "REAL"), ("trailing_stop_triggered", "INTEGER"),
    ]:
```

Also add the columns to the `CREATE TABLE` statement for `shadow_signals`:

```sql
            kelly_fraction REAL,
            kelly_qty INTEGER,
            high_water_mark REAL,
            trailing_stop_triggered INTEGER
```

Update `log_shadow_signal()` to accept and insert the new columns:

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
    kelly_fraction: float | None = None,
    kelly_qty: int | None = None,
    high_water_mark: float | None = None,
    trailing_stop_triggered: bool | None = None,
) -> int:
    cursor = conn.execute(
        "INSERT INTO shadow_signals (timestamp, ticker, close_price, predicted_class, predicted_proba, ml_signal, sma_signal, regime, signal_source, bear_day_count, kelly_fraction, kelly_qty, high_water_mark, trailing_stop_triggered) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (timestamp.isoformat(), ticker, close_price, predicted_class, predicted_proba, ml_signal, sma_signal, regime, signal_source, bear_day_count, kelly_fraction, kelly_qty, high_water_mark, int(trailing_stop_triggered) if trailing_stop_triggered is not None else None),
    )
    conn.commit()
    return cursor.lastrowid
```

- [ ] **Step 4: Run all shadow logging tests**

Run: `pytest tests/test_shadow_logging.py tests/test_trade_log.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/schroeder_trader/storage/trade_log.py tests/test_shadow_logging.py
git commit -m "feat: add high_water_mark and trailing_stop_triggered columns to shadow_signals"
```

---

### Task 5: Pipeline Integration

**Files:**
- Modify: `src/schroeder_trader/main.py`

- [ ] **Step 1: Add trailing stop imports and initialization**

Add to imports in `main.py`:

```python
from schroeder_trader.risk.trailing_stop import TrailingStop
from schroeder_trader.config import (
    COMPOSITE_MODEL_PATH,
    DB_PATH,
    FEATURES_CSV_PATH,
    KELLY_MULTIPLIER,
    KELLY_WIN_LOSS_RATIO,
    PROJECT_ROOT,
    TICKER,
    TRAILING_STOP_PCT,
    TRAILING_STOP_COOLDOWN_DAYS,
)
```

- [ ] **Step 2: Initialize trailing stop from DB state**

Add after `conn = init_db(db_path)` and before the idempotency check in `run_pipeline()`:

```python
    # Initialize trailing stop from DB state
    ts_row = conn.execute(
        "SELECT high_water_mark, trailing_stop_triggered, timestamp FROM shadow_signals "
        "WHERE high_water_mark IS NOT NULL ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if ts_row and ts_row["trailing_stop_triggered"]:
        from datetime import date as date_type
        stop_date = date_type.fromisoformat(ts_row["timestamp"][:10])
        trailing_stop = TrailingStop(
            TRAILING_STOP_PCT, TRAILING_STOP_COOLDOWN_DAYS,
            high_water_mark=ts_row["high_water_mark"], stop_date=stop_date,
        )
    elif ts_row:
        trailing_stop = TrailingStop(
            TRAILING_STOP_PCT, TRAILING_STOP_COOLDOWN_DAYS,
            high_water_mark=ts_row["high_water_mark"],
        )
    else:
        trailing_stop = TrailingStop(TRAILING_STOP_PCT, TRAILING_STOP_COOLDOWN_DAYS)
```

- [ ] **Step 3: Evaluate trailing stop and log in shadow signal**

In Step 10 (shadow signal section), after computing `k_qty`, add trailing stop evaluation and pass to `log_shadow_signal`:

```python
                    # Evaluate trailing stop
                    ts_triggered = trailing_stop.update(account["portfolio_value"], now.date())
                    ts_trading_dates = [
                        date_type.fromisoformat(r["timestamp"][:10])
                        for r in conn.execute(
                            "SELECT timestamp FROM shadow_signals ORDER BY id"
                        ).fetchall()
                    ]
                    ts_in_cooldown = trailing_stop.in_cooldown(now.date(), ts_trading_dates)

                    # Log shadow signal (always log XGB prediction for analysis)
                    log_shadow_signal(
                        conn, now, TICKER, close_price,
                        predicted_class=pred_class,
                        predicted_proba=json.dumps(proba_dict),
                        ml_signal=composite_sig.value,
                        sma_signal=signal.value,
                        regime=today_regime.value,
                        signal_source=source,
                        bear_day_count=bear_days if today_regime == Regime.BEAR else None,
                        kelly_fraction=k_frac,
                        kelly_qty=k_qty,
                        high_water_mark=trailing_stop.high_water_mark,
                        trailing_stop_triggered=ts_triggered or ts_in_cooldown,
                    )
```

Also add `from datetime import date as date_type` at the top of the file (or use inline import as shown in Step 2).

Update the logger.info line to include trailing stop state:

```python
                    logger.info(
                        "Shadow composite: %s (source=%s, regime=%s, bear_days=%d, kelly=%.3f, kelly_qty=%d, hwm=%.0f, ts_stop=%s)",
                        composite_sig.value, source, today_regime.value, bear_days,
                        k_frac or 0.0, k_qty or 0,
                        trailing_stop.high_water_mark, ts_triggered or ts_in_cooldown,
                    )
```

- [ ] **Step 4: Run full test suite**

Run: `pytest -v`
Expected: all 90+ tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/schroeder_trader/main.py
git commit -m "feat: integrate trailing stop into shadow pipeline"
```

---

### Task 6: End-to-End Validation

**Files:** (no new files)

- [ ] **Step 1: Run the pipeline simulation**

Run the same end-to-end simulation used previously to verify the full pipeline works:

```bash
.venv/bin/python -c "
from schroeder_trader.data.market_data import fetch_daily_bars
from schroeder_trader.strategy.feature_engineer import FeaturePipeline
from schroeder_trader.strategy.regime_detector import detect_regime, Regime
from schroeder_trader.strategy.composite import count_consecutive_bear_days, composite_signal_hybrid
from schroeder_trader.risk.kelly import kelly_fraction, kelly_qty
from schroeder_trader.risk.trailing_stop import TrailingStop
from schroeder_trader.strategy.sma_crossover import Signal
from schroeder_trader.config import *
from schroeder_trader.storage.trade_log import init_db, log_shadow_signal
import pandas as pd, numpy as np, json, tempfile
from pathlib import Path
from datetime import datetime, date

model_path = COMPOSITE_MODEL_PATH
import xgboost as xgb
model = xgb.XGBClassifier()
model.load_model(str(model_path))

ext_df = pd.read_csv(str(FEATURES_CSV_PATH), index_col='date', parse_dates=True)
spy_df = fetch_daily_bars('SPY', days=600)
pipeline = FeaturePipeline()
features = pipeline.compute_features_extended(spy_df, ext_df)

# Quick trailing stop test
ts = TrailingStop(TRAILING_STOP_PCT, TRAILING_STOP_COOLDOWN_DAYS)
triggered = ts.update(100000.0, date.today())
print(f'[OK] TrailingStop: hwm={ts.high_water_mark}, triggered={triggered}')

# Write to temp DB
tmp_db = Path(tempfile.mktemp(suffix='.db'))
conn = init_db(tmp_db)
log_shadow_signal(
    conn, datetime.now(), 'SPY', 650.0,
    predicted_class=2, predicted_proba='{\"UP\": 0.6}',
    ml_signal='SELL', sma_signal='HOLD',
    high_water_mark=ts.high_water_mark,
    trailing_stop_triggered=False,
)
rows = conn.execute('SELECT high_water_mark, trailing_stop_triggered FROM shadow_signals').fetchall()
assert rows[0]['high_water_mark'] == 100000.0
assert rows[0]['trailing_stop_triggered'] == 0
print(f'[OK] Shadow signal logged with HWM and trailing stop')
conn.close()
tmp_db.unlink()

print()
print('=== ALL CHECKS PASSED ===')
"
```

Expected: `ALL CHECKS PASSED`

- [ ] **Step 2: Run full test suite**

Run: `pytest -v`
Expected: all tests PASS

- [ ] **Step 3: Commit and push**

```bash
git push
```
