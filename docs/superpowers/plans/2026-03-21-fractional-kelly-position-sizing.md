# Fractional Kelly Position Sizing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add fractional Kelly position sizing that uses XGBoost probability outputs to scale position size, integrated into shadow logging and validated via backtest.

**Architecture:** New pure-function module `risk/kelly.py` computes Kelly fraction from XGBoost probabilities. Backtest script derives the win/loss ratio (b) from walk-forward data. Shadow pipeline logs Kelly sizing alongside existing signals. No changes to live execution.

**Tech Stack:** Python, XGBoost (existing), pandas, numpy, sqlite3, pytest

**Spec:** `docs/superpowers/specs/2026-03-21-fractional-kelly-position-sizing-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `src/schroeder_trader/risk/kelly.py` | Create | `kelly_fraction()` and `kelly_qty()` pure functions |
| `tests/test_kelly.py` | Create | Unit tests for Kelly functions |
| `src/schroeder_trader/config.py` | Modify | Add `KELLY_MULTIPLIER` and `KELLY_WIN_LOSS_RATIO` |
| `src/schroeder_trader/storage/trade_log.py` | Modify | Add `kelly_fraction`/`kelly_qty` columns + params |
| `tests/test_trade_log.py` | Modify | Test new shadow_signals columns |
| `src/schroeder_trader/main.py` | Modify | Compute and log Kelly in Step 10 |
| `backtest/run_backtest_kelly.py` | Create | Derive win/loss ratio, compare Kelly vs binary sizing |

---

### Task 1: Kelly Module — Core Functions

**Files:**
- Create: `src/schroeder_trader/risk/kelly.py`
- Create: `tests/test_kelly.py`

- [ ] **Step 1: Write failing tests for `kelly_fraction`**

Create `tests/test_kelly.py`:

```python
import pytest

from schroeder_trader.risk.kelly import kelly_fraction


def test_kelly_balanced_confidence():
    """P(UP)=0.6, P(DOWN)=0.2, b=1.5 → positive Kelly."""
    result = kelly_fraction(p_up=0.6, p_down=0.2, win_loss_ratio=1.5, kelly_multiplier=0.5)
    # kelly_pct = (0.6*1.5 - 0.2) / 1.5 = 0.4667
    # fractional = 0.4667 * 0.5 = 0.2333
    assert 0.23 < result < 0.24


def test_kelly_high_flat_prob_gives_near_zero():
    """P(UP)=0.05, P(DOWN)=0.05 → Kelly near zero."""
    result = kelly_fraction(p_up=0.05, p_down=0.05, win_loss_ratio=1.5, kelly_multiplier=0.5)
    assert result < 0.02


def test_kelly_negative_clamped_to_zero():
    """P(DOWN) > P(UP)*b → negative Kelly clamped to 0."""
    result = kelly_fraction(p_up=0.1, p_down=0.8, win_loss_ratio=1.0, kelly_multiplier=0.5)
    assert result == 0.0


def test_kelly_perfect_confidence():
    """P(UP)=1.0, P(DOWN)=0.0 → full Kelly multiplier."""
    result = kelly_fraction(p_up=1.0, p_down=0.0, win_loss_ratio=1.5, kelly_multiplier=0.5)
    assert result == 0.5


def test_kelly_zero_probs():
    """P(UP)=0, P(DOWN)=0 → 0."""
    result = kelly_fraction(p_up=0.0, p_down=0.0, win_loss_ratio=1.5, kelly_multiplier=0.5)
    assert result == 0.0


def test_kelly_clamped_to_one():
    """Full Kelly with extreme confidence clamped to 1.0."""
    result = kelly_fraction(p_up=1.0, p_down=0.0, win_loss_ratio=1.5, kelly_multiplier=1.5)
    assert result == 1.0


def test_kelly_invalid_win_loss_ratio():
    """win_loss_ratio <= 0 raises ValueError."""
    with pytest.raises(ValueError):
        kelly_fraction(p_up=0.5, p_down=0.3, win_loss_ratio=0.0, kelly_multiplier=0.5)
    with pytest.raises(ValueError):
        kelly_fraction(p_up=0.5, p_down=0.3, win_loss_ratio=-1.0, kelly_multiplier=0.5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_kelly.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `kelly_fraction` in `risk/kelly.py`**

Create `src/schroeder_trader/risk/kelly.py`:

```python
import math


def kelly_fraction(
    p_up: float,
    p_down: float,
    win_loss_ratio: float,
    kelly_multiplier: float = 0.5,
) -> float:
    """Compute fractional Kelly position size.

    Args:
        p_up: XGBoost P(UP) — probability of a winning trade.
        p_down: XGBoost P(DOWN) — probability of a losing trade.
        win_loss_ratio: Average win / average loss from backtest. Must be > 0.
        kelly_multiplier: Fractional Kelly safety factor (0.5 = half-Kelly).

    Returns:
        Target position as fraction of available capital [0.0, 1.0].
        Returns 0.0 when Kelly is negative (model predicts net loss).

    Raises:
        ValueError: If win_loss_ratio <= 0.
    """
    if win_loss_ratio <= 0:
        raise ValueError(f"win_loss_ratio must be > 0, got {win_loss_ratio}")

    kelly_pct = (p_up * win_loss_ratio - p_down) / win_loss_ratio
    fractional = kelly_pct * kelly_multiplier
    return min(1.0, max(0.0, fractional))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_kelly.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Write failing tests for `kelly_qty`**

Append to `tests/test_kelly.py`:

```python
from schroeder_trader.risk.kelly import kelly_qty


def test_kelly_qty_basic():
    """50% Kelly fraction, $100k portfolio, SPY at $500."""
    qty = kelly_qty(kelly_frac=0.5, portfolio_value=100000, close_price=500, cash_buffer_pct=0.02)
    # available = 100000 * 0.98 = 98000
    # target = 98000 * 0.5 = 49000
    # qty = floor(49000 / 500) = 98
    assert qty == 98


def test_kelly_qty_zero_fraction():
    qty = kelly_qty(kelly_frac=0.0, portfolio_value=100000, close_price=500)
    assert qty == 0


def test_kelly_qty_small_portfolio():
    """Kelly fraction too small to buy even 1 share."""
    qty = kelly_qty(kelly_frac=0.01, portfolio_value=1000, close_price=500)
    # available = 1000 * 0.98 = 980
    # target = 980 * 0.01 = 9.8
    # qty = floor(9.8 / 500) = 0
    assert qty == 0
```

- [ ] **Step 6: Run tests to verify new tests fail**

Run: `pytest tests/test_kelly.py -v`
Expected: 3 new tests FAIL with `ImportError`

- [ ] **Step 7: Implement `kelly_qty`**

Add to `src/schroeder_trader/risk/kelly.py`:

```python
def kelly_qty(
    kelly_frac: float,
    portfolio_value: float,
    close_price: float,
    cash_buffer_pct: float = 0.02,
) -> int:
    """Convert Kelly fraction to whole share count.

    Args:
        kelly_frac: Kelly fraction [0.0, 1.0] from kelly_fraction().
        portfolio_value: Total portfolio value in dollars.
        close_price: Current price per share.
        cash_buffer_pct: Fraction of portfolio reserved as cash buffer.

    Returns:
        Number of whole shares to hold (always >= 0).
    """
    available = portfolio_value * (1 - cash_buffer_pct)
    target_dollars = available * kelly_frac
    return math.floor(target_dollars / close_price)
```

- [ ] **Step 8: Run all tests to verify they pass**

Run: `pytest tests/test_kelly.py -v`
Expected: All 10 tests PASS

- [ ] **Step 9: Commit**

```bash
git add src/schroeder_trader/risk/kelly.py tests/test_kelly.py
git commit -m "feat: add fractional Kelly position sizing module"
```

---

### Task 2: Config — Add Kelly Parameters

**Files:**
- Modify: `src/schroeder_trader/config.py:21-22`

- [ ] **Step 1: Add Kelly config values**

Add after line 22 (`SLIPPAGE_ESTIMATE = 0.0005`) in `config.py`:

```python
KELLY_MULTIPLIER = 0.5           # half-Kelly default (configurable)
KELLY_WIN_LOSS_RATIO = 1.0       # placeholder — derived in Task 6
```

The win/loss ratio starts at 1.0 as a placeholder. Task 6 (backtest) will compute the real value.

- [ ] **Step 2: Run full test suite**

Run: `pytest tests/ -x -q`
Expected: All tests pass (config additions are additive)

- [ ] **Step 3: Commit**

```bash
git add src/schroeder_trader/config.py
git commit -m "feat: add KELLY_MULTIPLIER and KELLY_WIN_LOSS_RATIO config"
```

---

### Task 3: Database — Add Kelly Columns to shadow_signals

**Files:**
- Modify: `src/schroeder_trader/storage/trade_log.py:46-66` (CREATE TABLE + migration)
- Modify: `src/schroeder_trader/storage/trade_log.py:152-170` (log_shadow_signal)
- Modify: `tests/test_trade_log.py`

- [ ] **Step 1: Write failing test for kelly columns in shadow signal logging**

Add to the imports at the top of `tests/test_trade_log.py`:

```python
import pytest

from schroeder_trader.storage.trade_log import (
    init_db,
    log_signal,
    log_order,
    log_portfolio,
    get_signal_by_date,
    get_pending_orders,
    update_order_fill,
    log_shadow_signal,
    get_shadow_signals,
)
```

Then add the test functions at the bottom:

```python
def test_log_shadow_signal_with_kelly(tmp_path):
    conn = init_db(tmp_path / "test.db")
    log_shadow_signal(
        conn,
        timestamp=datetime(2026, 3, 21, 16, 30),
        ticker="SPY",
        close_price=650.0,
        predicted_class=2,
        predicted_proba='{"DOWN": 0.2, "FLAT": 0.2, "UP": 0.6}',
        ml_signal="BUY",
        sma_signal="HOLD",
        regime="CHOPPY",
        signal_source="XGB",
        bear_day_count=None,
        kelly_fraction=0.233,
        kelly_qty=35,
    )
    rows = get_shadow_signals(conn)
    assert len(rows) == 1
    assert rows[0]["kelly_fraction"] == pytest.approx(0.233)
    assert rows[0]["kelly_qty"] == 35
    conn.close()


def test_log_shadow_signal_kelly_null_for_non_xgb(tmp_path):
    conn = init_db(tmp_path / "test.db")
    log_shadow_signal(
        conn,
        timestamp=datetime(2026, 3, 21, 16, 30),
        ticker="SPY",
        close_price=650.0,
        predicted_class=None,
        predicted_proba=None,
        ml_signal="SELL",
        sma_signal="HOLD",
        regime="BEAR",
        signal_source="FLAT",
        bear_day_count=3,
    )
    rows = get_shadow_signals(conn)
    assert len(rows) == 1
    assert rows[0]["kelly_fraction"] is None
    assert rows[0]["kelly_qty"] is None
    conn.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_trade_log.py::test_log_shadow_signal_with_kelly -v`
Expected: FAIL (unexpected keyword argument `kelly_fraction`)

- [ ] **Step 3: Update `log_shadow_signal` signature and INSERT**

In `trade_log.py`, modify `log_shadow_signal` (lines 152-170):

Add `kelly_fraction: float | None = None` and `kelly_qty: int | None = None` parameters.
Update the INSERT to include the two new columns.

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
) -> int:
    cursor = conn.execute(
        "INSERT INTO shadow_signals (timestamp, ticker, close_price, predicted_class, predicted_proba, ml_signal, sma_signal, regime, signal_source, bear_day_count, kelly_fraction, kelly_qty) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (timestamp.isoformat(), ticker, close_price, predicted_class, predicted_proba, ml_signal, sma_signal, regime, signal_source, bear_day_count, kelly_fraction, kelly_qty),
    )
    conn.commit()
    return cursor.lastrowid
```

- [ ] **Step 4: Update CREATE TABLE and migration in `init_db`**

Add `kelly_fraction` and `kelly_qty` to the CREATE TABLE statement (after `bear_day_count INTEGER`):

```sql
            kelly_fraction REAL,
            kelly_qty INTEGER
```

Add to the migration loop (lines 62-66):

```python
    for col, col_type in [
        ("regime", "TEXT"), ("signal_source", "TEXT"), ("bear_day_count", "INTEGER"),
        ("kelly_fraction", "REAL"), ("kelly_qty", "INTEGER"),
    ]:
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_trade_log.py -v`
Expected: All tests PASS (existing + 2 new)

- [ ] **Step 6: Run full test suite**

Run: `pytest tests/ -x -q`
Expected: All tests pass

- [ ] **Step 7: Commit**

```bash
git add src/schroeder_trader/storage/trade_log.py tests/test_trade_log.py
git commit -m "feat: add kelly_fraction and kelly_qty columns to shadow_signals"
```

---

### Task 4: Shadow Pipeline — Compute and Log Kelly

**Files:**
- Modify: `src/schroeder_trader/main.py:254-264` (log_shadow_signal call in Step 10)

- [ ] **Step 1: Add Kelly import to main.py**

Add to the imports at the top of `main.py`:

```python
from schroeder_trader.risk.kelly import kelly_fraction as compute_kelly_fraction, kelly_qty as compute_kelly_qty
from schroeder_trader.config import KELLY_MULTIPLIER, KELLY_WIN_LOSS_RATIO
```

- [ ] **Step 2: Compute Kelly values before log_shadow_signal call**

In the shadow pipeline section (after the composite signal is computed, before `log_shadow_signal`), add Kelly computation when source is XGB:

```python
                    # Compute Kelly sizing (XGB sources only)
                    k_frac = None
                    k_qty = None
                    if source == "XGB":
                        k_frac = compute_kelly_fraction(
                            p_up=proba[idx_up],
                            p_down=proba[idx_down],
                            win_loss_ratio=KELLY_WIN_LOSS_RATIO,
                            kelly_multiplier=KELLY_MULTIPLIER,
                        )
                        k_qty = compute_kelly_qty(k_frac, account["portfolio_value"], close_price)
```

- [ ] **Step 3: Add Kelly params to log_shadow_signal call**

Update the existing `log_shadow_signal(...)` call (lines 254-264) to include:

```python
                        kelly_fraction=k_frac,
                        kelly_qty=k_qty,
```

- [ ] **Step 4: Add Kelly info to the logger.info line**

Update the shadow composite log message to include Kelly info:

```python
                    logger.info(
                        "Shadow composite: %s (source=%s, regime=%s, bear_days=%d, kelly=%.3f, kelly_qty=%d)",
                        composite_sig.value, source, today_regime.value, bear_days,
                        k_frac or 0.0, k_qty or 0,
                    )
```

- [ ] **Step 5: Run full test suite**

Run: `pytest tests/ -x -q`
Expected: All tests pass

- [ ] **Step 6: Commit**

```bash
git add src/schroeder_trader/main.py
git commit -m "feat: compute and log Kelly sizing in shadow pipeline"
```

---

### Task 5: Manual Verification — Reconstruct Kelly for Recent Days

**Files:** None (manual check only)

- [ ] **Step 1: Run reconstruction script to verify Kelly computes for recent data**

Run a Python script similar to the earlier reconstruction that fetches 600 days of SPY, computes composite signals, and prints Kelly fraction + qty for the last 5 trading days. Verify:
- Kelly fraction is a reasonable number (0.0–0.5 range for half-Kelly)
- Kelly qty is a reasonable share count
- Non-XGB source days show None/None

- [ ] **Step 2: Verify database migration works on existing trades.db**

Run: `.venv/bin/python -c "from schroeder_trader.storage.trade_log import init_db; from pathlib import Path; conn = init_db(Path('data/trades.db')); print('Migration OK'); conn.close()"`

Expected: "Migration OK" (ALTER TABLE adds new columns to existing DB)

---

### Task 6: Backtest — Derive Win/Loss Ratio and Validate Kelly Sizing

**Files:**
- Create: `backtest/run_backtest_kelly.py`
- Modify: `src/schroeder_trader/config.py` (update `KELLY_WIN_LOSS_RATIO` with real value)

- [ ] **Step 1: Create backtest script**

Create `backtest/run_backtest_kelly.py` that:

1. Imports `from backtest.train_final_composite import prepare_features, XGB_FEATURES, XGB_PARAMS, TRAIN_YEARS, TEST_MONTHS` and calls `prepare_features()` to get the labeled feature matrix
2. Runs walk-forward windows (same 2yr train / 6mo test as existing)
3. In each test window:
   - Trains XGBoost, gets `predict_proba` for test rows
   - Computes composite signals via `composite_signal_hybrid`
   - For XGB-sourced BUY signals: records the actual 20-day forward return
   - Tracks wins (forward return > 0) and losses (forward return < 0)
4. Computes `win_loss_ratio = mean(winning_returns) / abs(mean(losing_returns))`
5. Simulates two portfolios through the test windows:
   - **Binary**: 100% in on BUY, 100% out on SELL (current behavior)
   - **Kelly**: sized by `kelly_fraction()` using each day's probabilities
6. Reports comparative metrics: Sharpe, max drawdown, total return

The script should print the derived `KELLY_WIN_LOSS_RATIO` value clearly.

- [ ] **Step 2: Run the backtest**

Run: `cd /Users/ads7fg/git/SchroederTrader && .venv/bin/python -m backtest.run_backtest_kelly`

Expected output: win/loss ratio, comparative Sharpe/DD metrics

- [ ] **Step 3: Update config with real win/loss ratio**

Replace the placeholder `KELLY_WIN_LOSS_RATIO = 1.0` in `config.py` with the value derived from the backtest.

- [ ] **Step 4: Verify success criteria**

Check:
- Kelly Sharpe >= binary Sharpe (0.94 full-period)
- Kelly max drawdown <= binary max drawdown (16.1%)
- win_loss_ratio is > 0 and reasonable (typically 1.0–2.0)

- [ ] **Step 5: Run full test suite**

Run: `pytest tests/ -x -q`
Expected: All tests pass

- [ ] **Step 6: Commit**

```bash
git add backtest/run_backtest_kelly.py src/schroeder_trader/config.py
git commit -m "feat: backtest-derived Kelly win/loss ratio and validation"
```

---

### Task 7: Final Verification

- [ ] **Step 1: Run full test suite**

Run: `pytest tests/ -x -q`
Expected: All tests pass

- [ ] **Step 2: Verify shadow pipeline end-to-end**

Manually run the pipeline reconstruction for recent days to confirm Kelly values appear in output.

- [ ] **Step 3: Final commit if any cleanup needed**

```bash
git add -A
git commit -m "chore: Kelly position sizing cleanup"
```
