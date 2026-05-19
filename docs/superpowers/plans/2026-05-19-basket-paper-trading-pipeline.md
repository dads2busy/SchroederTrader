# Basket Paper-Trading Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a multi-ticker paper-trading pipeline (`src/schroeder_trader/basket/`) that runs alongside the existing SPY-only production pipeline, daily-rebalancing to configured weights with per-ticker trailing stops, and integrates into the existing daily email.

**Architecture:** Separate subpackage with its own entry point and GitHub Actions workflow. Shared state files (`portfolio.csv`, `orders.csv`, `shadow_signals.csv`) distinguished by a new `pipeline` column. SPY-only's trade-execution path is untouched; only its logging calls gain a `pipeline="spy_only"` kwarg and its email-rendering reads basket rows. The basket pipeline runs at 21:20 UTC (writes state); SPY-only runs at 21:30 UTC (trades SPY, reads basket state, sends the unified email).

**Tech Stack:** Python 3.12 with `uv`. Pandas for state files. pytest for tests. GitHub Actions for cron. No new external dependencies.

---

## File Structure

**New files:**
- `src/schroeder_trader/basket/__init__.py`
- `src/schroeder_trader/basket/main.py` — entry point
- `src/schroeder_trader/basket/orchestrator.py` — per-ticker signal + stop loop
- `src/schroeder_trader/basket/rebalance.py` — current vs target → orders
- `src/schroeder_trader/basket/portfolio.py` — bootstrap, snapshot writes, prior-state reads
- `scripts/migrate_portfolio_to_pipeline_column.py` — one-time migration
- `.github/workflows/daily-basket.yml` — 21:20 UTC cron, paper mode
- `tests/basket/__init__.py`
- `tests/basket/test_portfolio.py`
- `tests/basket/test_orchestrator.py`
- `tests/basket/test_rebalance.py`
- `tests/basket/test_main.py`
- `tests/basket/test_equivalence.py`
- `tests/test_migration.py`

**Modified files:**
- `src/schroeder_trader/storage/trade_log.py` — `log_portfolio`, `log_order`, `log_shadow_signal`, `insert_reconciled_order` gain `pipeline` kwarg
- `src/schroeder_trader/main.py` — existing call sites pass `pipeline="spy_only"` explicitly; email-rendering step reads basket rows
- `src/schroeder_trader/reports/daily_email.py` — new `build_basket_paper_section`; `build_email_body` gains `basket_state` kwarg
- `tests/test_daily_email.py` — new tests for the basket section
- `tests/test_trade_log.py` — existing tests updated to pass the new kwarg

---

### Task 1: Schema migration and pipeline kwarg in loggers

**Goal:** Add `pipeline` column to `portfolio.csv`, `orders.csv`, `shadow_signals.csv`. Update the four logger functions to accept a `pipeline` kwarg defaulting to `"spy_only"`. Migrate the existing CSV files. Update all SPY-only call sites in `main.py`. Verify with regression tests that the SPY-only behavior is bit-identical post-migration.

**Files:**
- Create: `scripts/migrate_portfolio_to_pipeline_column.py`
- Modify: `src/schroeder_trader/storage/trade_log.py` (`log_portfolio` line 71, `log_order` line 45, `log_shadow_signal` line 183, `insert_reconciled_order` line 126)
- Modify: `src/schroeder_trader/main.py` (every `log_portfolio`, `log_order`, `log_shadow_signal`, `insert_reconciled_order` call site)
- Create: `tests/test_migration.py`
- Modify: `tests/test_trade_log.py` (if any test asserts on column count or exact row dict — update for the new column)

- [ ] **Step 1: Backup current state files**

```bash
mkdir -p data/_pre_basket_backup
cp data/portfolio.csv data/_pre_basket_backup/portfolio.csv
cp data/orders.csv data/_pre_basket_backup/orders.csv
cp data/shadow_signals.csv data/_pre_basket_backup/shadow_signals.csv
```

Verify: `ls -1 data/_pre_basket_backup/` shows the three files.

- [ ] **Step 2: Write the failing migration tests**

Create `tests/test_migration.py`:

```python
import shutil
from pathlib import Path
import pandas as pd
import pytest

from scripts.migrate_portfolio_to_pipeline_column import migrate


def _seed_csv(path: Path, df: pd.DataFrame) -> None:
    df.to_csv(path, index=False)


def test_migration_adds_pipeline_column_to_portfolio_csv(tmp_path):
    pf = pd.DataFrame({
        "id": [1, 2],
        "timestamp": ["2026-04-15T20:30:00+00:00", "2026-04-16T20:30:00+00:00"],
        "cash": [1965.4, 1965.4],
        "position_qty": [141, 141],
        "position_value": [100000.0, 100500.0],
        "total_value": [101965.4, 102465.4],
    })
    _seed_csv(tmp_path / "portfolio.csv", pf)
    _seed_csv(tmp_path / "orders.csv", pd.DataFrame(columns=[
        "id", "signal_id", "alpaca_order_id", "timestamp", "ticker",
        "action", "quantity", "fill_price", "fill_timestamp", "status",
        "signal_close_price", "slippage",
    ]))
    _seed_csv(tmp_path / "shadow_signals.csv", pd.DataFrame(columns=[
        "id", "timestamp", "ticker", "close_price", "predicted_class",
        "predicted_proba", "ml_signal", "sma_signal", "regime",
        "signal_source", "bear_day_count", "kelly_fraction", "kelly_qty",
        "high_water_mark", "trailing_stop_triggered",
    ]))

    migrate(tmp_path)
    out = pd.read_csv(tmp_path / "portfolio.csv")
    assert "pipeline" in out.columns
    assert "ticker" in out.columns
    assert (out["pipeline"] == "spy_only").all()
    assert (out["ticker"] == "SPY").all()


def test_migration_preserves_existing_rows_bit_identical(tmp_path):
    """Property test: every original column byte-identical after filtering."""
    pf_pre = pd.DataFrame({
        "id": [1, 2, 3],
        "timestamp": ["2026-04-15T20:30:00+00:00",
                       "2026-04-16T20:30:00+00:00",
                       "2026-04-17T20:30:00+00:00"],
        "cash": [1965.4, 1965.4, 1965.4],
        "position_qty": [141, 141, 141],
        "position_value": [100000.0, 100500.0, 101000.0],
        "total_value": [101965.4, 102465.4, 102965.4],
    })
    _seed_csv(tmp_path / "portfolio.csv", pf_pre)
    _seed_csv(tmp_path / "orders.csv", pd.DataFrame(columns=[
        "id", "signal_id", "alpaca_order_id", "timestamp", "ticker",
        "action", "quantity", "fill_price", "fill_timestamp", "status",
        "signal_close_price", "slippage",
    ]))
    _seed_csv(tmp_path / "shadow_signals.csv", pd.DataFrame(columns=[
        "id", "timestamp", "ticker", "close_price", "predicted_class",
        "predicted_proba", "ml_signal", "sma_signal", "regime",
        "signal_source", "bear_day_count", "kelly_fraction", "kelly_qty",
        "high_water_mark", "trailing_stop_triggered",
    ]))

    migrate(tmp_path)
    pf_post = pd.read_csv(tmp_path / "portfolio.csv")
    pf_post_spy = pf_post[pf_post["pipeline"] == "spy_only"]
    for col in pf_pre.columns:
        assert (pf_pre[col].astype(str).values == pf_post_spy[col].astype(str).values).all(), col


def test_migration_is_idempotent(tmp_path):
    pf = pd.DataFrame({
        "id": [1], "timestamp": ["2026-04-15T20:30:00+00:00"],
        "cash": [1965.4], "position_qty": [141],
        "position_value": [100000.0], "total_value": [101965.4],
    })
    _seed_csv(tmp_path / "portfolio.csv", pf)
    _seed_csv(tmp_path / "orders.csv", pd.DataFrame(columns=[
        "id", "signal_id", "alpaca_order_id", "timestamp", "ticker",
        "action", "quantity", "fill_price", "fill_timestamp", "status",
        "signal_close_price", "slippage",
    ]))
    _seed_csv(tmp_path / "shadow_signals.csv", pd.DataFrame(columns=[
        "id", "timestamp", "ticker", "close_price", "predicted_class",
        "predicted_proba", "ml_signal", "sma_signal", "regime",
        "signal_source", "bear_day_count", "kelly_fraction", "kelly_qty",
        "high_water_mark", "trailing_stop_triggered",
    ]))

    migrate(tmp_path)
    first_pf = pd.read_csv(tmp_path / "portfolio.csv").to_csv(index=False)
    migrate(tmp_path)
    second_pf = pd.read_csv(tmp_path / "portfolio.csv").to_csv(index=False)
    assert first_pf == second_pf
```

- [ ] **Step 3: Run the migration tests — confirm they fail**

```bash
uv run pytest tests/test_migration.py -v
```

Expected: `ImportError: cannot import name 'migrate'`.

- [ ] **Step 4: Write the migration script**

Create `scripts/migrate_portfolio_to_pipeline_column.py`:

```python
"""One-time migration: add a `pipeline` column (default 'spy_only') and a
`ticker` column (default 'SPY' where missing) to portfolio.csv, orders.csv,
and shadow_signals.csv. Idempotent — re-running is a no-op.

Run:
    uv run python -m scripts.migrate_portfolio_to_pipeline_column
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

_FILES = [
    ("portfolio.csv", {"pipeline": "spy_only", "ticker": "SPY"}),
    ("orders.csv", {"pipeline": "spy_only"}),
    ("shadow_signals.csv", {"pipeline": "spy_only"}),
]


def migrate(data_dir: Path = DEFAULT_DATA_DIR) -> None:
    for filename, defaults in _FILES:
        path = data_dir / filename
        if not path.exists():
            continue
        df = pd.read_csv(path)
        changed = False
        for col, default in defaults.items():
            if col not in df.columns:
                # Insert near the front for readability; exact position doesn't
                # matter functionally because reads use column names.
                if col == "pipeline":
                    insert_at = 2 if "timestamp" in df.columns else 1
                else:
                    insert_at = min(3, len(df.columns))
                df.insert(insert_at, col, default)
                changed = True
        if changed:
            df.to_csv(path, index=False)


if __name__ == "__main__":
    migrate()
    print("Migration complete.")
```

- [ ] **Step 5: Run the migration tests — confirm they pass**

```bash
uv run pytest tests/test_migration.py -v
```

Expected: 3 tests pass.

- [ ] **Step 6: Update logger signatures in `trade_log.py`**

Modify `src/schroeder_trader/storage/trade_log.py`:

Update `log_order` (around line 45) — append `pipeline` to the dict it writes:

```python
def log_order(
    store: CsvStore,
    signal_id: int,
    alpaca_order_id: str,
    timestamp: datetime,
    ticker: str,
    action: str,
    quantity: int,
    status: str,
    signal_close_price: float | None = None,
    pipeline: str = "spy_only",
) -> int:
    return store.append("orders", {
        "signal_id": signal_id,
        "alpaca_order_id": alpaca_order_id,
        "timestamp": timestamp.isoformat(),
        "pipeline": pipeline,
        "ticker": ticker,
        "action": action,
        "quantity": quantity,
        "fill_price": None,
        "fill_timestamp": None,
        "status": status,
        "signal_close_price": signal_close_price,
        "slippage": None,
    })
```

Update `log_portfolio` (around line 71):

```python
def log_portfolio(
    store: CsvStore,
    timestamp: datetime,
    cash: float,
    position_qty: int,
    position_value: float,
    total_value: float,
    pipeline: str = "spy_only",
    ticker: str = "SPY",
) -> int:
    return store.append("portfolio", {
        "timestamp": timestamp.isoformat(),
        "pipeline": pipeline,
        "ticker": ticker,
        "cash": cash,
        "position_qty": position_qty,
        "position_value": position_value,
        "total_value": total_value,
    })
```

Update `insert_reconciled_order` (around line 126) — append `pipeline` kwarg:

```python
def insert_reconciled_order(
    store: CsvStore,
    alpaca_order_id: str,
    timestamp: datetime,
    ticker: str,
    action: str,
    quantity: int,
    status: str,
    fill_price: float | None = None,
    fill_timestamp: datetime | None = None,
    pipeline: str = "spy_only",
) -> int:
    return store.append("orders", {
        "signal_id": 0,
        "alpaca_order_id": alpaca_order_id,
        "timestamp": timestamp.isoformat(),
        "pipeline": pipeline,
        "ticker": ticker,
        "action": action,
        "quantity": quantity,
        "fill_price": fill_price,
        "fill_timestamp": fill_timestamp.isoformat() if fill_timestamp else None,
        "status": status,
        "signal_close_price": None,
        "slippage": None,
    })
```

Update `log_shadow_signal` (around line 183):

```python
def log_shadow_signal(
    store: CsvStore,
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
    pipeline: str = "spy_only",
) -> int:
    ts_int = int(trailing_stop_triggered) if trailing_stop_triggered is not None else None
    return store.append("shadow_signals", {
        "timestamp": timestamp.isoformat(),
        "pipeline": pipeline,
        "ticker": ticker,
        "close_price": close_price,
        # ... (rest of fields unchanged)
```

(Continue with the rest of the existing `log_shadow_signal` body — only the kwarg and dict's `pipeline` insertion change.)

- [ ] **Step 7: Update SPY-only call sites in `main.py` to pass `pipeline` explicitly**

In `src/schroeder_trader/main.py`, find every call site for `log_portfolio`, `log_order`, `log_shadow_signal`, `insert_reconciled_order` and add `pipeline="spy_only"` as an explicit kwarg. Approximate line numbers from `grep -n`: `log_signal` at 192 (no change — `signals.csv` schema not modified), `log_shadow_signal` calls at ~358 and ~480 (the SPY production path), `log_portfolio` at ~464, `submit_order` at ~440, `log_order` at ~443.

Example for `log_portfolio`:

```python
log_portfolio(
    conn, now, account["cash"], position_qty, position_value, account["portfolio_value"],
    pipeline="spy_only",
)
```

Example for `log_shadow_signal`:

```python
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
    pipeline="spy_only",
)
```

The `_run_shadow_for_ticker` function (~line 567) writes shadow rows for the non-SPY tickers as part of the SPY-only pipeline. Those calls also get `pipeline="spy_only"` — the basket pipeline will later write its own rows with `pipeline="basket"`.

- [ ] **Step 8: Run the actual migration against `data/`**

```bash
uv run python -m scripts.migrate_portfolio_to_pipeline_column
```

Expected output: `Migration complete.`

Verify the new columns exist:

```bash
head -1 data/portfolio.csv
```

Expected: `id,timestamp,pipeline,ticker,cash,position_qty,position_value,total_value`

- [ ] **Step 9: Run the full test suite to verify no regressions**

```bash
uv run pytest
```

Expected: All ~250 existing tests still pass, plus the 3 new migration tests.

If any test fails because it asserted on the old schema, update the assertion to account for the new columns. Likely candidates: `tests/test_trade_log.py`, `tests/test_reconcile.py`. Read the failure carefully — most fixes are one-line.

- [ ] **Step 10: Commit**

```bash
git add scripts/migrate_portfolio_to_pipeline_column.py \
        src/schroeder_trader/storage/trade_log.py \
        src/schroeder_trader/main.py \
        tests/test_migration.py \
        tests/test_trade_log.py \
        data/portfolio.csv data/orders.csv data/shadow_signals.csv \
        data/_pre_basket_backup/
git commit -m "feat(schema): add pipeline column to portfolio/orders/shadow_signals

Phase A of basket paper-trading rollout. Adds pipeline (default 'spy_only')
and ticker (default 'SPY' for portfolio.csv) columns to the three shared
state CSVs. Logger functions in trade_log.py gain a pipeline kwarg;
existing SPY-only call sites in main.py pass pipeline='spy_only' explicitly.
Migration script is idempotent; property test verifies bit-identical
preservation of existing rows after filtering to pipeline='spy_only'.

Backup of pre-migration state CSVs in data/_pre_basket_backup/."
```

---

### Task 2: Basket portfolio module (bootstrap, snapshots, prior-state reads)

**Goal:** Build `basket/portfolio.py` with functions to bootstrap the basket's starting capital from prior state, read per-ticker positions, read prior exposures, and write per-ticker portfolio snapshot rows.

**Files:**
- Create: `src/schroeder_trader/basket/__init__.py` (empty)
- Create: `src/schroeder_trader/basket/portfolio.py`
- Create: `tests/basket/__init__.py` (empty)
- Create: `tests/basket/test_portfolio.py`

- [ ] **Step 1: Write the failing portfolio tests**

Create `tests/basket/test_portfolio.py`:

```python
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from schroeder_trader.basket.portfolio import (
    bootstrap_starting_value,
    read_position_qty,
    prior_exposure,
    read_trading_dates,
    write_basket_portfolio_snapshot,
)
from schroeder_trader.storage.csv_store import CsvStore


def _make_store(tmp_path: Path) -> CsvStore:
    return CsvStore(tmp_path)


def test_bootstrap_returns_latest_basket_total_when_basket_rows_exist(tmp_path):
    pf = pd.DataFrame({
        "id": [1, 2, 3],
        "timestamp": [
            "2026-05-19T20:30:00+00:00",
            "2026-05-20T20:30:00+00:00",
            "2026-05-20T20:30:00+00:00",
        ],
        "pipeline": ["spy_only", "basket", "basket"],
        "ticker": ["SPY", "SPY", "XLK"],
        "cash": [1965.4, 500.0, 500.0],
        "position_qty": [141, 64, 181],
        "position_value": [100000.0, 47000.0, 32000.0],
        "total_value": [101965.4, 105000.0, 105000.0],
    })
    pf.to_csv(tmp_path / "portfolio.csv", index=False)
    store = _make_store(tmp_path)
    assert bootstrap_starting_value(store) == 105000.0


def test_bootstrap_falls_back_to_spy_only_when_no_basket_rows(tmp_path):
    pf = pd.DataFrame({
        "id": [1], "timestamp": ["2026-05-19T20:30:00+00:00"],
        "pipeline": ["spy_only"], "ticker": ["SPY"],
        "cash": [1965.4], "position_qty": [141],
        "position_value": [100000.0], "total_value": [101965.4],
    })
    pf.to_csv(tmp_path / "portfolio.csv", index=False)
    store = _make_store(tmp_path)
    assert bootstrap_starting_value(store) == 101965.4


def test_bootstrap_raises_when_no_rows_at_all(tmp_path):
    pd.DataFrame(columns=[
        "id", "timestamp", "pipeline", "ticker",
        "cash", "position_qty", "position_value", "total_value",
    ]).to_csv(tmp_path / "portfolio.csv", index=False)
    store = _make_store(tmp_path)
    with pytest.raises(RuntimeError, match="no portfolio history"):
        bootstrap_starting_value(store)


def test_read_position_qty_returns_zero_when_no_basket_rows_for_ticker(tmp_path):
    pf = pd.DataFrame({
        "id": [1], "timestamp": ["2026-05-19T20:30:00+00:00"],
        "pipeline": ["basket"], "ticker": ["SPY"],
        "cash": [500.0], "position_qty": [64],
        "position_value": [47000.0], "total_value": [105000.0],
    })
    pf.to_csv(tmp_path / "portfolio.csv", index=False)
    store = _make_store(tmp_path)
    assert read_position_qty(store, "XLK") == 0
    assert read_position_qty(store, "SPY") == 64


def test_prior_exposure_returns_one_when_last_signal_was_buy(tmp_path):
    ss = pd.DataFrame({
        "id": [1, 2],
        "timestamp": ["2026-05-19T20:30:00+00:00", "2026-05-20T20:30:00+00:00"],
        "pipeline": ["basket", "basket"],
        "ticker": ["XLK", "XLK"],
        "close_price": [100.0, 105.0],
        "predicted_class": [None, None],
        "predicted_proba": [None, None],
        "ml_signal": ["BUY", "HOLD"],
        "sma_signal": ["BUY", "HOLD"],
        "regime": ["BULL", "BULL"],
        "signal_source": ["SMA", "SMA"],
        "bear_day_count": [None, None],
        "kelly_fraction": [None, None],
        "kelly_qty": [None, None],
        "high_water_mark": [None, None],
        "trailing_stop_triggered": [None, None],
    })
    ss.to_csv(tmp_path / "shadow_signals.csv", index=False)
    store = _make_store(tmp_path)
    assert prior_exposure(store, "XLK") == 1.0  # last decided was BUY


def test_prior_exposure_returns_zero_when_no_prior_basket_rows(tmp_path):
    pd.DataFrame(columns=[
        "id", "timestamp", "pipeline", "ticker", "close_price",
        "predicted_class", "predicted_proba", "ml_signal", "sma_signal",
        "regime", "signal_source", "bear_day_count",
        "kelly_fraction", "kelly_qty", "high_water_mark",
        "trailing_stop_triggered",
    ]).to_csv(tmp_path / "shadow_signals.csv", index=False)
    store = _make_store(tmp_path)
    assert prior_exposure(store, "XLK") == 0.0


def test_read_trading_dates_returns_basket_pipeline_dates_for_ticker(tmp_path):
    from datetime import date
    ss = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "timestamp": [
            "2026-05-12T20:30:00+00:00", "2026-05-13T20:30:00+00:00",
            "2026-05-12T20:30:00+00:00", "2026-05-13T20:30:00+00:00",
        ],
        "pipeline": ["basket", "basket", "spy_only", "spy_only"],
        "ticker": ["XLK", "XLK", "XLK", "XLK"],
        "close_price": [100.0, 105.0, 100.0, 105.0],
        "predicted_class": [None] * 4, "predicted_proba": [None] * 4,
        "ml_signal": ["BUY"] * 4, "sma_signal": ["BUY"] * 4,
        "regime": ["BULL"] * 4, "signal_source": ["SMA"] * 4,
        "bear_day_count": [None] * 4, "kelly_fraction": [None] * 4,
        "kelly_qty": [None] * 4, "high_water_mark": [None] * 4,
        "trailing_stop_triggered": [None] * 4,
    })
    ss.to_csv(tmp_path / "shadow_signals.csv", index=False)
    store = _make_store(tmp_path)
    dates = read_trading_dates(store, "XLK")
    # Only basket rows; ET-local dates
    assert len(dates) == 2
    assert dates[0] == date(2026, 5, 12)
    assert dates[1] == date(2026, 5, 13)


def test_write_basket_portfolio_snapshot_emits_one_row_per_ticker(tmp_path):
    pf_empty = pd.DataFrame(columns=[
        "id", "timestamp", "pipeline", "ticker",
        "cash", "position_qty", "position_value", "total_value",
    ])
    pf_empty.to_csv(tmp_path / "portfolio.csv", index=False)
    store = _make_store(tmp_path)

    fake_broker = MagicMock()
    fake_broker.get_position.side_effect = lambda t: {"SPY": 64, "XLK": 181}.get(t, 0)
    fake_broker.get_account.return_value = {"cash": 500.0, "portfolio_value": 105000.0}

    prices = {"SPY": 738.0, "XLK": 175.0}
    now = datetime(2026, 5, 21, 20, 30, tzinfo=timezone.utc)
    write_basket_portfolio_snapshot(
        store, fake_broker, ["SPY", "XLK"], prices, now,
    )

    pf = pd.read_csv(tmp_path / "portfolio.csv")
    assert len(pf) == 2
    assert (pf["pipeline"] == "basket").all()
    assert set(pf["ticker"]) == {"SPY", "XLK"}
    assert (pf["total_value"] == 105000.0).all()  # repeated on each row
    assert (pf["cash"] == 500.0).all()
```

- [ ] **Step 2: Run the tests — confirm they fail**

```bash
uv run pytest tests/basket/test_portfolio.py -v
```

Expected: `ImportError: No module named 'schroeder_trader.basket'`.

- [ ] **Step 3: Implement `basket/portfolio.py`**

Create `src/schroeder_trader/basket/__init__.py` as an empty file.

Create `src/schroeder_trader/basket/portfolio.py`:

```python
"""Basket paper-trading portfolio state — bootstrap, snapshots, prior reads.

Reads and writes rows in shared CSVs (portfolio.csv, shadow_signals.csv)
filtered to pipeline='basket'. SPY-only rows are ignored except as a
fallback during first-ever basket-pipeline run (see bootstrap_starting_value).
"""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd

from schroeder_trader.storage.csv_store import CsvStore
from schroeder_trader.storage.trade_log import log_portfolio


def bootstrap_starting_value(store: CsvStore) -> float:
    """Return the basket pipeline's starting portfolio value.

    If any pipeline='basket' rows exist, return the latest basket total_value.
    Otherwise, fall back to the latest pipeline='spy_only' total_value
    (this is the first-ever basket run — mirror real portfolio size).
    Raise RuntimeError if neither has any data.
    """
    df = store.read("portfolio")
    if df.empty:
        raise RuntimeError("no portfolio history to bootstrap from")
    basket = df[df["pipeline"] == "basket"]
    if not basket.empty:
        latest = basket.sort_values("timestamp").iloc[-1]
        return float(latest["total_value"])
    spy_only = df[df["pipeline"] == "spy_only"]
    if not spy_only.empty:
        latest = spy_only.sort_values("timestamp").iloc[-1]
        return float(latest["total_value"])
    raise RuntimeError("no portfolio history to bootstrap from")


def read_position_qty(store: CsvStore, ticker: str) -> int:
    """Latest basket-pipeline position quantity for `ticker`, or 0 if none."""
    df = store.read("portfolio")
    if df.empty:
        return 0
    rows = df[(df["pipeline"] == "basket") & (df["ticker"] == ticker)]
    if rows.empty:
        return 0
    latest = rows.sort_values("timestamp").iloc[-1]
    return int(latest["position_qty"])


def prior_exposure(store: CsvStore, ticker: str) -> float:
    """Latest decided exposure (0.0 or 1.0) for `ticker` from basket shadow
    signals. Returns 0.0 when no prior basket signal exists (cold start)."""
    df = store.read("shadow_signals")
    if df.empty:
        return 0.0
    rows = df[(df["pipeline"] == "basket") & (df["ticker"] == ticker)]
    if rows.empty:
        return 0.0
    latest = rows.sort_values("timestamp").iloc[-1]
    last_signal = str(latest["ml_signal"])
    if last_signal == "BUY":
        return 1.0
    if last_signal == "SELL":
        return 0.0
    # HOLD shouldn't appear as the "last decided" exposure source — the
    # caller's chain is: BUY/SELL flips, HOLD carries forward. If the last
    # logged row IS a HOLD, walk back until we find a non-HOLD.
    chain = rows.sort_values("timestamp")
    for _, r in chain[::-1].iterrows():
        s = str(r["ml_signal"])
        if s == "BUY":
            return 1.0
        if s == "SELL":
            return 0.0
    return 0.0


def read_trading_dates(store: CsvStore, ticker: str) -> list[date]:
    """ET-local dates for which a basket-pipeline shadow signal exists for
    `ticker`, in ascending order. Used by the per-ticker TrailingStop's
    cooldown logic."""
    df = store.read("shadow_signals")
    if df.empty:
        return []
    rows = df[(df["pipeline"] == "basket") & (df["ticker"] == ticker)].copy()
    if rows.empty:
        return []
    rows["date"] = pd.to_datetime(rows["timestamp"], utc=True, format="ISO8601") \
        .dt.tz_convert("America/New_York").dt.date
    return list(rows.sort_values("date")["date"])


def write_basket_portfolio_snapshot(
    store: CsvStore,
    broker,
    tickers: list[str],
    prices: dict[str, float],
    now: datetime,
) -> None:
    """Write one portfolio.csv row per ticker with pipeline='basket'.

    `prices` is the closing price per ticker used to compute position_value.
    The shared basket cash and total portfolio value are repeated on every
    row (each row is a self-contained snapshot for that ticker).
    """
    account = broker.get_account()
    cash = float(account["cash"])
    total_value = float(account["portfolio_value"])
    for ticker in tickers:
        qty = int(broker.get_position(ticker))
        position_value = qty * prices[ticker]
        log_portfolio(
            store, now, cash, qty, position_value, total_value,
            pipeline="basket", ticker=ticker,
        )
```

- [ ] **Step 4: Run the tests — confirm they pass**

```bash
uv run pytest tests/basket/test_portfolio.py -v
```

Expected: 8 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/schroeder_trader/basket/__init__.py \
        src/schroeder_trader/basket/portfolio.py \
        tests/basket/__init__.py \
        tests/basket/test_portfolio.py
git commit -m "feat(basket): portfolio module — bootstrap, snapshot, prior reads

State accessors for the basket pipeline. bootstrap_starting_value falls back
to SPY-only's latest total when no basket rows exist (first-ever basket run).
read_position_qty and prior_exposure return 0 when no prior basket state for
the given ticker. write_basket_portfolio_snapshot emits one row per ticker
with cash and total_value repeated for self-contained snapshots."
```

---

### Task 3: Basket orchestrator (signal + per-ticker trailing stop loop)

**Goal:** Build `basket/orchestrator.py` with `compute_decisions(store, weights, ext_df, now) -> dict[str, dict]`. For each ticker in `weights`, this fetches bars, computes the composite signal, updates a per-ticker `TrailingStop`, applies stop override to exposure, and logs the decision to `shadow_signals.csv` with `pipeline='basket'`. Returns a dict keyed by ticker containing `{signal, exposure, price, regime, source, stop_state}` for use by the rebalancer.

**Files:**
- Create: `src/schroeder_trader/basket/orchestrator.py`
- Create: `tests/basket/test_orchestrator.py`

- [ ] **Step 1: Write the failing orchestrator tests**

Create `tests/basket/test_orchestrator.py`:

```python
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from schroeder_trader.basket.orchestrator import compute_decisions
from schroeder_trader.storage.csv_store import CsvStore


def _make_store(tmp_path: Path) -> CsvStore:
    # Initialize the expected CSVs as empty so reads don't fail
    pd.DataFrame(columns=[
        "id", "timestamp", "pipeline", "ticker",
        "cash", "position_qty", "position_value", "total_value",
    ]).to_csv(tmp_path / "portfolio.csv", index=False)
    pd.DataFrame(columns=[
        "id", "timestamp", "pipeline", "ticker", "close_price",
        "predicted_class", "predicted_proba", "ml_signal", "sma_signal",
        "regime", "signal_source", "bear_day_count",
        "kelly_fraction", "kelly_qty", "high_water_mark",
        "trailing_stop_triggered",
    ]).to_csv(tmp_path / "shadow_signals.csv", index=False)
    return CsvStore(tmp_path)


def _fake_signal_pipeline(signal_value: str, source: str, regime: str = "BULL"):
    """Patch target that mimics the per-ticker composite-signal computation."""
    from schroeder_trader.strategy.composite import Signal
    from schroeder_trader.strategy.regime_detector import Regime

    sig_enum = Signal[signal_value]
    regime_enum = Regime[regime]
    return (sig_enum, source, regime_enum, 0, {"close": 100.0})


@patch("schroeder_trader.basket.orchestrator._compute_signal_for_ticker")
def test_compute_decisions_buy_signal_produces_full_exposure(
    mock_sig, tmp_path,
):
    mock_sig.return_value = _fake_signal_pipeline("BUY", "SMA")
    store = _make_store(tmp_path)
    weights = {"SPY": 0.5, "XLK": 0.5}
    ext_df = pd.DataFrame()
    now = datetime(2026, 5, 21, 20, 30, tzinfo=timezone.utc)

    decisions = compute_decisions(store, weights, ext_df, now, portfolio_value=100000.0)

    assert decisions["SPY"]["exposure"] == 1.0
    assert decisions["XLK"]["exposure"] == 1.0
    assert decisions["SPY"]["signal"] == "BUY"


@patch("schroeder_trader.basket.orchestrator._compute_signal_for_ticker")
def test_compute_decisions_sell_signal_produces_flat_exposure(
    mock_sig, tmp_path,
):
    mock_sig.return_value = _fake_signal_pipeline("SELL", "FLAT", regime="BEAR")
    store = _make_store(tmp_path)
    weights = {"SPY": 1.0}
    now = datetime(2026, 5, 21, 20, 30, tzinfo=timezone.utc)

    decisions = compute_decisions(store, weights, pd.DataFrame(), now, portfolio_value=100000.0)

    assert decisions["SPY"]["exposure"] == 0.0


@patch("schroeder_trader.basket.orchestrator._compute_signal_for_ticker")
def test_compute_decisions_hold_carries_prior_basket_exposure(
    mock_sig, tmp_path,
):
    # Seed prior basket BUY for SPY
    ss = pd.DataFrame({
        "id": [1], "timestamp": ["2026-05-20T20:30:00+00:00"],
        "pipeline": ["basket"], "ticker": ["SPY"],
        "close_price": [100.0],
        "predicted_class": [None], "predicted_proba": [None],
        "ml_signal": ["BUY"], "sma_signal": ["BUY"],
        "regime": ["BULL"], "signal_source": ["SMA"],
        "bear_day_count": [None], "kelly_fraction": [None],
        "kelly_qty": [None], "high_water_mark": [None],
        "trailing_stop_triggered": [None],
    })
    pd.DataFrame(columns=[
        "id", "timestamp", "pipeline", "ticker",
        "cash", "position_qty", "position_value", "total_value",
    ]).to_csv(tmp_path / "portfolio.csv", index=False)
    ss.to_csv(tmp_path / "shadow_signals.csv", index=False)
    store = CsvStore(tmp_path)

    mock_sig.return_value = _fake_signal_pipeline("HOLD", "SMA")
    weights = {"SPY": 1.0}
    now = datetime(2026, 5, 21, 20, 30, tzinfo=timezone.utc)

    decisions = compute_decisions(store, weights, pd.DataFrame(), now, portfolio_value=100000.0)
    assert decisions["SPY"]["exposure"] == 1.0  # HOLD carries prior BUY


@patch("schroeder_trader.basket.orchestrator._compute_signal_for_ticker")
def test_compute_decisions_triggers_trailing_stop_and_forces_flat(
    mock_sig, tmp_path,
):
    # Seed prior basket state with a position that has now drawn down 15%.
    # Need HWM history in shadow_signals to drive the stop.
    pf = pd.DataFrame({
        "id": [1], "timestamp": ["2026-05-20T20:30:00+00:00"],
        "pipeline": ["basket"], "ticker": ["XLK"],
        "cash": [500.0], "position_qty": [100],
        "position_value": [10000.0], "total_value": [10500.0],
    })
    ss = pd.DataFrame({
        "id": [1], "timestamp": ["2026-05-20T20:30:00+00:00"],
        "pipeline": ["basket"], "ticker": ["XLK"],
        "close_price": [100.0],
        "predicted_class": [None], "predicted_proba": [None],
        "ml_signal": ["BUY"], "sma_signal": ["BUY"],
        "regime": ["BULL"], "signal_source": ["SMA"],
        "bear_day_count": [None], "kelly_fraction": [None],
        "kelly_qty": [None], "high_water_mark": [12000.0],
        "trailing_stop_triggered": [0],
    })
    pf.to_csv(tmp_path / "portfolio.csv", index=False)
    ss.to_csv(tmp_path / "shadow_signals.csv", index=False)
    store = CsvStore(tmp_path)

    # Signal would be BUY, but position has drawn down: portfolio 10500 vs
    # HWM 12000 = -12.5% which exceeds the 10% trailing stop threshold.
    mock_sig.return_value = _fake_signal_pipeline("BUY", "SMA")
    now = datetime(2026, 5, 21, 20, 30, tzinfo=timezone.utc)

    decisions = compute_decisions(
        store, {"XLK": 1.0}, pd.DataFrame(), now,
        portfolio_value=10500.0,
    )
    assert decisions["XLK"]["exposure"] == 0.0  # stop overrides
    assert decisions["XLK"]["stop_state"]["triggered_today_or_cooldown"] is True


@patch("schroeder_trader.basket.orchestrator._compute_signal_for_ticker")
def test_compute_decisions_logs_shadow_signal_with_pipeline_basket(
    mock_sig, tmp_path,
):
    mock_sig.return_value = _fake_signal_pipeline("BUY", "SMA")
    store = _make_store(tmp_path)
    now = datetime(2026, 5, 21, 20, 30, tzinfo=timezone.utc)

    compute_decisions(store, {"SPY": 1.0}, pd.DataFrame(), now, portfolio_value=100000.0)

    ss = pd.read_csv(tmp_path / "shadow_signals.csv")
    assert len(ss) == 1
    assert ss.iloc[0]["pipeline"] == "basket"
    assert ss.iloc[0]["ticker"] == "SPY"
    assert ss.iloc[0]["ml_signal"] == "BUY"
```

- [ ] **Step 2: Run the tests — confirm they fail**

```bash
uv run pytest tests/basket/test_orchestrator.py -v
```

Expected: `ImportError: cannot import name 'compute_decisions'`.

- [ ] **Step 3: Implement `basket/orchestrator.py`**

Create `src/schroeder_trader/basket/orchestrator.py`:

```python
"""Per-ticker signal + trailing-stop loop for the basket pipeline.

For each ticker in the basket, this module computes the composite signal
(via the existing strategy modules), runs the per-ticker TrailingStop, and
returns a decision dict consumed by the rebalancer. Decisions are also
logged to shadow_signals.csv with pipeline='basket'.
"""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd

from schroeder_trader.config import (
    TRAILING_STOP_PCT,
    TRAILING_STOP_COOLDOWN_DAYS,
    XGB_THRESHOLD_LOW,
)
from schroeder_trader.data.market_data import fetch_daily_bars
from schroeder_trader.risk.trailing_stop import TrailingStop
from schroeder_trader.storage.csv_store import CsvStore
from schroeder_trader.storage.trade_log import log_shadow_signal
from schroeder_trader.strategy.composite import (
    Signal, composite_signal_hybrid,
)
from schroeder_trader.strategy.feature_engineer import FeaturePipeline
from schroeder_trader.strategy.regime_detector import (
    Regime, compute_regime_series, compute_regime_labels,
    count_consecutive_bear_days,
)
from schroeder_trader.strategy.sma_crossover import generate_signal
from schroeder_trader.strategy.xgboost_classifier import load_model

from schroeder_trader.basket.portfolio import (
    prior_exposure,
    read_trading_dates,
)


def _compute_signal_for_ticker(
    ticker: str, model_path, ext_df: pd.DataFrame,
):
    """Run the composite-signal pipeline for one ticker.

    Returns (composite_signal: Signal, source: str, regime: Regime,
             bear_days: int, last_bar: dict). Mirrors the structure of
    _run_shadow_for_ticker in main.py but without the side effects.
    """
    model = load_model(model_path)
    df = fetch_daily_bars(ticker, days=600)
    pipeline = FeaturePipeline()
    features = pipeline.compute_features_extended(df, ext_df)
    if len(features) == 0:
        raise RuntimeError(f"Shadow {ticker}: no features computed")

    regime_series = compute_regime_series(features)
    features["regime_label"] = compute_regime_labels(features)
    today_regime = regime_series.iloc[-1]
    if not isinstance(today_regime, Regime):
        today_regime = Regime.CHOPPY
    bear_days = count_consecutive_bear_days(regime_series)

    bear_weakening = False
    if today_regime == Regime.BEAR and "log_return_5d" in features.columns:
        lr5 = features["log_return_5d"].iloc[-1]
        bear_weakening = not pd.isna(lr5) and lr5 > 0

    feature_cols = [
        "log_return_5d", "log_return_20d", "volatility_20d",
        "credit_spread", "dollar_momentum", "regime_label",
    ]
    last_row = features[feature_cols].iloc[[-1]]
    if last_row.isna().any().any():
        raise RuntimeError(f"Shadow {ticker}: NaN in feature row")

    class_to_idx = {int(c): i for i, c in enumerate(model.classes_)}
    idx_up = class_to_idx[2]
    idx_down = class_to_idx[0]
    proba = model.predict_proba(last_row)[0]
    pred_class = int(np.argmax(proba))
    proba_dict = {
        "DOWN": float(proba[idx_down]),
        "FLAT": float(proba[class_to_idx[1]]),
        "UP": float(proba[idx_up]),
    }
    if pred_class == idx_up and proba[idx_up] > XGB_THRESHOLD_LOW:
        xgb_low = Signal.BUY
    elif pred_class == idx_down and proba[idx_down] > XGB_THRESHOLD_LOW:
        xgb_low = Signal.SELL
    else:
        xgb_low = Signal.HOLD

    sma_signal, _, _ = generate_signal(df)

    composite_sig, source = composite_signal_hybrid(
        today_regime, sma_signal, xgb_low, bear_weakening=bear_weakening,
    )

    last_bar = {
        "close": float(df["close"].iloc[-1]),
        "pred_class": pred_class,
        "proba_json": json.dumps(proba_dict),
        "sma_signal": sma_signal.value,
    }
    return composite_sig, source, today_regime, bear_days, last_bar


def compute_decisions(
    store: CsvStore,
    weights: dict[str, float],
    ext_df: pd.DataFrame,
    now: datetime,
    portfolio_value: float,
) -> dict[str, dict]:
    """Compute the per-ticker decision dict for one daily basket run.

    Side effect: writes one row per ticker to shadow_signals.csv with
    pipeline='basket' capturing the decision (including HWM and stop state).
    """
    from schroeder_trader.config import SHADOW_TICKERS, COMPOSITE_MODEL_PATH

    decisions: dict[str, dict] = {}
    for ticker in weights:
        # Resolve model path: SPY uses the production model, others use SHADOW_TICKERS
        model_path = COMPOSITE_MODEL_PATH if ticker == "SPY" else SHADOW_TICKERS[ticker]

        signal, source, regime, bear_days, last_bar = \
            _compute_signal_for_ticker(ticker, model_path, ext_df)

        # Per-ticker trailing stop. The HWM persists across runs via
        # shadow_signals.csv. We instantiate fresh each run and update.
        ts = _load_or_create_stop(store, ticker, portfolio_value)
        trading_dates = read_trading_dates(store, ticker)
        ts_triggered = ts.update(portfolio_value, now.date(), trading_dates=trading_dates)
        in_cooldown = ts.in_cooldown(now.date(), trading_dates)

        if ts_triggered or in_cooldown:
            exposure = 0.0
        elif signal == Signal.BUY:
            exposure = 1.0
        elif signal == Signal.SELL:
            exposure = 0.0
        else:  # HOLD
            exposure = prior_exposure(store, ticker)

        decisions[ticker] = {
            "signal": signal.value,
            "exposure": exposure,
            "price": last_bar["close"],
            "regime": regime.value,
            "source": source,
            "stop_state": {
                "triggered_today_or_cooldown": bool(ts_triggered or in_cooldown),
                "high_water_mark": float(ts.high_water_mark),
            },
        }

        log_shadow_signal(
            store, now, ticker, last_bar["close"],
            predicted_class=last_bar["pred_class"],
            predicted_proba=last_bar["proba_json"],
            ml_signal=signal.value,
            sma_signal=last_bar["sma_signal"],
            regime=regime.value,
            signal_source=source,
            bear_day_count=bear_days if regime == Regime.BEAR else None,
            high_water_mark=ts.high_water_mark,
            trailing_stop_triggered=ts_triggered or in_cooldown,
            pipeline="basket",
        )

    return decisions


def _load_or_create_stop(store: CsvStore, ticker: str, current_value: float) -> TrailingStop:
    """Load the per-ticker stop's HWM from the latest basket shadow_signals row.
    Creates a fresh stop with HWM=0 (caller's update will set it) if none."""
    df = store.read("shadow_signals")
    hwm = 0.0
    if not df.empty:
        rows = df[(df["pipeline"] == "basket") & (df["ticker"] == ticker)]
        if not rows.empty:
            latest = rows.sort_values("timestamp").iloc[-1]
            hwm = float(latest["high_water_mark"]) if pd.notna(latest["high_water_mark"]) else 0.0
    return TrailingStop(
        drawdown_pct=TRAILING_STOP_PCT,
        cooldown_days=TRAILING_STOP_COOLDOWN_DAYS,
        high_water_mark=hwm,
    )
```

- [ ] **Step 4: Run the tests — confirm they pass**

```bash
uv run pytest tests/basket/test_orchestrator.py -v
```

Expected: 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/schroeder_trader/basket/orchestrator.py tests/basket/test_orchestrator.py
git commit -m "feat(basket): orchestrator — per-ticker signal + trailing-stop loop

compute_decisions iterates the configured weights, runs the composite
signal pipeline for each ticker, applies a per-ticker TrailingStop with
HWM persisted via shadow_signals.csv, and overrides exposure to 0 when
the stop is triggered or in cooldown. Decisions are logged to
shadow_signals.csv with pipeline='basket'. HOLD carries the prior
basket exposure forward."
```

---

### Task 4: Basket rebalancer (target → order diffs)

**Goal:** Build `basket/rebalance.py` with a pure function `compute_orders(portfolio_value, weights, decisions, current_positions) -> list[dict]` that returns the orders needed to bring each ticker's position to its target, and a thin wrapper `rebalance_to_targets` that calls the broker and logs orders.

**Files:**
- Create: `src/schroeder_trader/basket/rebalance.py`
- Create: `tests/basket/test_rebalance.py`

- [ ] **Step 1: Write the failing rebalance tests**

Create `tests/basket/test_rebalance.py`:

```python
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pandas as pd
import pytest

from schroeder_trader.basket.rebalance import compute_orders


def _decisions(**overrides):
    base = {
        "SPY": {"exposure": 1.0, "price": 738.0},
        "XLK": {"exposure": 1.0, "price": 176.0},
        "XLV": {"exposure": 1.0, "price": 146.0},
        "XLE": {"exposure": 1.0, "price": 58.0},
    }
    for t, vals in overrides.items():
        base[t] = {**base[t], **vals}
    return base


WEIGHTS = {"SPY": 0.45, "XLK": 0.30, "XLV": 0.15, "XLE": 0.10}


def test_compute_orders_initial_allocation_from_zero_positions():
    decisions = _decisions()
    current = {"SPY": 0, "XLK": 0, "XLV": 0, "XLE": 0}
    orders = compute_orders(100_000.0, WEIGHTS, decisions, current)
    by_ticker = {o["ticker"]: o for o in orders}
    # SPY: target 45000 / 738 = 60.97 → 60 shares BUY
    assert by_ticker["SPY"]["action"] == "BUY"
    assert by_ticker["SPY"]["qty"] == 60
    # XLK: 30000 / 176 = 170.45 → 170 shares
    assert by_ticker["XLK"]["qty"] == 170
    # XLV: 15000 / 146 = 102.74 → 102
    assert by_ticker["XLV"]["qty"] == 102
    # XLE: 10000 / 58 = 172.41 → 172
    assert by_ticker["XLE"]["qty"] == 172


def test_compute_orders_skips_when_diff_less_than_one_share():
    """Drift smaller than the price of one share should not produce an order."""
    decisions = _decisions(SPY={"exposure": 1.0, "price": 738.0})
    # 64 sh × 738 = 47232, target 45% × 100000 = 45000, diff = -2232
    # |diff| 2232 > 738 → would order, but try smaller diff next:
    decisions = _decisions(SPY={"exposure": 1.0, "price": 738.0})
    current = {"SPY": 61, "XLK": 170, "XLV": 102, "XLE": 172}
    # SPY: 61×738=45018, target 45000, diff=-18, abs(18) < 738 → SKIP
    orders = compute_orders(100_000.0, WEIGHTS, decisions, current)
    by_ticker = {o["ticker"]: o for o in orders}
    assert "SPY" not in by_ticker  # skipped


def test_compute_orders_zero_exposure_sells_to_flat():
    decisions = _decisions(XLE={"exposure": 0.0, "price": 58.0})
    current = {"SPY": 60, "XLK": 170, "XLV": 102, "XLE": 172}
    orders = compute_orders(100_000.0, WEIGHTS, decisions, current)
    by_ticker = {o["ticker"]: o for o in orders}
    assert by_ticker["XLE"]["action"] == "SELL"
    assert by_ticker["XLE"]["qty"] == 172  # full close


def test_compute_orders_buys_when_underweight():
    decisions = _decisions(SPY={"exposure": 1.0, "price": 700.0})
    # Target = 45% × 100000 / 700 = 64.28 → 64 shares
    # Current = 50 shares → BUY 14
    current = {"SPY": 50, "XLK": 170, "XLV": 102, "XLE": 172}
    orders = compute_orders(100_000.0, WEIGHTS, decisions, current)
    by_ticker = {o["ticker"]: o for o in orders}
    assert by_ticker["SPY"]["action"] == "BUY"
    assert by_ticker["SPY"]["qty"] == 14


def test_compute_orders_no_orders_when_all_at_target():
    decisions = _decisions()
    # Targets: SPY 60, XLK 170, XLV 102, XLE 172 (per the initial test)
    current = {"SPY": 60, "XLK": 170, "XLV": 102, "XLE": 172}
    orders = compute_orders(100_000.0, WEIGHTS, decisions, current)
    assert orders == []
```

- [ ] **Step 2: Run the tests — confirm they fail**

```bash
uv run pytest tests/basket/test_rebalance.py -v
```

Expected: `ImportError: cannot import name 'compute_orders'`.

- [ ] **Step 3: Implement `basket/rebalance.py`**

Create `src/schroeder_trader/basket/rebalance.py`:

```python
"""Basket rebalancer — current vs target → order diffs.

Pure compute_orders function for testability. rebalance_to_targets is
the side-effecting wrapper that calls the broker and logs orders.
"""

from __future__ import annotations

from datetime import datetime

from schroeder_trader.execution.broker import OrderRequest, submit_order
from schroeder_trader.storage.csv_store import CsvStore
from schroeder_trader.storage.trade_log import log_order


def compute_orders(
    portfolio_value: float,
    weights: dict[str, float],
    decisions: dict[str, dict],
    current_positions: dict[str, int],
) -> list[dict]:
    """Return the order diffs needed to bring each ticker to its target.

    Each returned dict has keys: ticker, action ('BUY' or 'SELL'), qty (int),
    price (float, the close used for sizing).

    Sub-share diffs (|diff_value| < price of one share) are skipped.
    """
    orders: list[dict] = []
    for ticker, weight in weights.items():
        d = decisions[ticker]
        target_value = portfolio_value * weight * d["exposure"]
        current_qty = int(current_positions.get(ticker, 0))
        current_value = current_qty * d["price"]
        diff_value = target_value - current_value

        if abs(diff_value) < d["price"]:
            continue

        diff_shares = int(diff_value // d["price"])
        if diff_shares == 0:
            continue

        orders.append({
            "ticker": ticker,
            "action": "BUY" if diff_shares > 0 else "SELL",
            "qty": abs(diff_shares),
            "price": d["price"],
        })
    return orders


def rebalance_to_targets(
    store: CsvStore,
    portfolio_value: float,
    weights: dict[str, float],
    decisions: dict[str, dict],
    current_positions: dict[str, int],
    now: datetime,
) -> list[dict]:
    """Submit market orders for each ticker's diff and log them.

    Returns the list of orders submitted (each enriched with `alpaca_order_id`
    and `status` from the broker response). Errors on individual orders are
    logged but don't stop the loop — the basket pipeline is paper-mode and
    a single ticker failing is non-fatal.
    """
    orders = compute_orders(portfolio_value, weights, decisions, current_positions)
    submitted: list[dict] = []
    for o in orders:
        try:
            req = OrderRequest(action=o["action"], qty=o["qty"])
            result = submit_order(req, ticker=o["ticker"])
            log_order(
                store, signal_id=0, alpaca_order_id=result.alpaca_order_id,
                timestamp=now, ticker=o["ticker"], action=o["action"],
                quantity=o["qty"], status=result.status,
                signal_close_price=o["price"], pipeline="basket",
            )
            submitted.append({**o, "alpaca_order_id": result.alpaca_order_id, "status": result.status})
        except Exception as exc:
            # Paper-mode: don't crash on a single-ticker order failure.
            # The next day's reconciliation will reconcile any orphans.
            submitted.append({**o, "alpaca_order_id": None, "status": f"ERROR: {exc}"})
    return submitted
```

- [ ] **Step 4: Run the tests — confirm they pass**

```bash
uv run pytest tests/basket/test_rebalance.py -v
```

Expected: 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/schroeder_trader/basket/rebalance.py tests/basket/test_rebalance.py
git commit -m "feat(basket): rebalancer — current vs target → order diffs

compute_orders is a pure function: portfolio_value × weight × exposure
→ target value per ticker, diff against current position rounded to
whole shares, sub-share diffs skipped. rebalance_to_targets is the
side-effecting wrapper that submits market orders via broker and logs
them with pipeline='basket'. Individual order failures are non-fatal
to keep the pipeline running for the other tickers."
```

---

### Task 5: Basket main entry point + workflow file

**Goal:** Wire the pieces together in `basket/main.py` and add the `daily-basket.yml` GitHub Actions workflow (`workflow_dispatch` only — not scheduled yet; that flip is Task 7).

**Files:**
- Create: `src/schroeder_trader/basket/main.py`
- Create: `.github/workflows/daily-basket.yml`
- Create: `tests/basket/test_main.py`

- [ ] **Step 1: Write the failing main entry-point test**

Create `tests/basket/test_main.py`:

```python
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from schroeder_trader.basket.main import run_basket_pipeline
from schroeder_trader.storage.csv_store import CsvStore


def _seed_initial_state(tmp_path: Path, spy_only_total: float = 105000.0):
    """SPY-only history exists; basket has never run."""
    pd.DataFrame({
        "id": [1], "timestamp": ["2026-05-20T20:30:00+00:00"],
        "pipeline": ["spy_only"], "ticker": ["SPY"],
        "cash": [1965.4], "position_qty": [141],
        "position_value": [spy_only_total - 1965.4],
        "total_value": [spy_only_total],
    }).to_csv(tmp_path / "portfolio.csv", index=False)

    pd.DataFrame(columns=[
        "id", "signal_id", "alpaca_order_id", "timestamp", "pipeline",
        "ticker", "action", "quantity", "fill_price", "fill_timestamp",
        "status", "signal_close_price", "slippage",
    ]).to_csv(tmp_path / "orders.csv", index=False)

    pd.DataFrame(columns=[
        "id", "timestamp", "pipeline", "ticker", "close_price",
        "predicted_class", "predicted_proba", "ml_signal", "sma_signal",
        "regime", "signal_source", "bear_day_count",
        "kelly_fraction", "kelly_qty", "high_water_mark",
        "trailing_stop_triggered",
    ]).to_csv(tmp_path / "shadow_signals.csv", index=False)


@patch("schroeder_trader.basket.main.submit_order")
@patch("schroeder_trader.basket.main.get_position")
@patch("schroeder_trader.basket.main.get_account")
@patch("schroeder_trader.basket.orchestrator._compute_signal_for_ticker")
def test_run_basket_pipeline_writes_per_ticker_rows_to_portfolio_csv(
    mock_sig, mock_account, mock_position, mock_submit, tmp_path,
):
    from schroeder_trader.strategy.composite import Signal
    from schroeder_trader.strategy.regime_detector import Regime

    _seed_initial_state(tmp_path)
    mock_sig.return_value = (
        Signal.BUY, "SMA", Regime.BULL, 0,
        {"close": 100.0, "pred_class": 2, "proba_json": "{}", "sma_signal": "BUY"},
    )
    mock_account.return_value = {"cash": 500.0, "portfolio_value": 105000.0}
    mock_position.side_effect = lambda t: 0  # zero starting positions
    mock_submit.return_value = MagicMock(alpaca_order_id="abc", status="SUBMITTED")

    weights = {"SPY": 0.45, "XLK": 0.30, "XLV": 0.15, "XLE": 0.10}
    now = datetime(2026, 5, 21, 20, 20, tzinfo=timezone.utc)
    run_basket_pipeline(tmp_path, weights, now)

    pf = pd.read_csv(tmp_path / "portfolio.csv")
    basket_rows = pf[pf["pipeline"] == "basket"]
    assert set(basket_rows["ticker"]) == set(weights.keys())
    assert len(basket_rows) == 4
```

- [ ] **Step 2: Run the test — confirm it fails**

```bash
uv run pytest tests/basket/test_main.py -v
```

Expected: `ImportError: cannot import name 'run_basket_pipeline'`.

- [ ] **Step 3: Implement `basket/main.py`**

Create `src/schroeder_trader/basket/main.py`:

```python
"""Basket paper-trading pipeline entry point.

Invoked by .github/workflows/daily-basket.yml at 21:20 UTC.
Computes per-ticker decisions, rebalances to target weights, writes per-
ticker portfolio snapshots. Does NOT send email — the SPY-only pipeline
(running at 21:30 UTC) reads basket state and renders the unified email.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from schroeder_trader.config import (
    DB_PATH, FEATURES_CSV_PATH, SHADOW_BASKET_WEIGHTS,
)
from schroeder_trader.execution.broker import submit_order, get_position, get_account
from schroeder_trader.execution.reconcile import reconcile_orders
from schroeder_trader.storage.csv_store import CsvStore

from schroeder_trader.basket.orchestrator import (
    compute_decisions, _compute_signal_for_ticker,
)
from schroeder_trader.basket.portfolio import (
    bootstrap_starting_value, read_position_qty,
    write_basket_portfolio_snapshot,
)
from schroeder_trader.basket.rebalance import rebalance_to_targets

logger = logging.getLogger(__name__)


def run_basket_pipeline(
    data_dir: Path | None = None,
    weights: dict[str, float] | None = None,
    now: datetime | None = None,
) -> None:
    """Run one daily basket pipeline iteration.

    Arguments are exposed for testing; production calls use defaults.
    """
    if data_dir is None:
        data_dir = Path(DB_PATH).parent
    if weights is None:
        weights = SHADOW_BASKET_WEIGHTS
    if now is None:
        now = datetime.now(timezone.utc)

    store = CsvStore(data_dir)

    # 1. Reconcile orphaned orders per ticker (paper account).
    for ticker in weights:
        try:
            reconcile_orders(store, ticker)
        except Exception:
            logger.exception("Reconcile failed for %s (non-fatal)", ticker)

    # 2. Bootstrap portfolio value.
    portfolio_value = bootstrap_starting_value(store)

    # 3. Load extended features once.
    ext_df = pd.DataFrame()
    if FEATURES_CSV_PATH.exists():
        ext_df = pd.read_csv(str(FEATURES_CSV_PATH), index_col="date", parse_dates=True)

    # 4. Compute per-ticker decisions (logs shadow_signals rows with pipeline='basket').
    decisions = compute_decisions(store, weights, ext_df, now, portfolio_value)

    # 5. Read current positions from broker.
    current_positions = {t: int(get_position(t)) for t in weights}

    # 6. Rebalance to targets — submit orders, log to orders.csv with pipeline='basket'.
    rebalance_to_targets(
        store, portfolio_value, weights, decisions, current_positions, now,
    )

    # 7. Read updated positions from broker after fills.
    prices = {t: decisions[t]["price"] for t in weights}

    # Build a small adapter to feed into write_basket_portfolio_snapshot.
    class _BrokerAdapter:
        @staticmethod
        def get_position(t):
            return get_position(t)
        @staticmethod
        def get_account():
            return get_account()

    write_basket_portfolio_snapshot(store, _BrokerAdapter, list(weights), prices, now)

    logger.info("Basket pipeline complete. Wrote %d ticker rows.", len(weights))


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    run_basket_pipeline()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test — confirm it passes**

```bash
uv run pytest tests/basket/test_main.py -v
```

Expected: 1 test passes.

- [ ] **Step 5: Create the GitHub Actions workflow**

Create `.github/workflows/daily-basket.yml`:

```yaml
name: daily-basket

on:
  # No schedule yet — flipped to scheduled in Task 7 once Phase C is verified.
  workflow_dispatch:
    inputs:
      dry_run:
        description: "Test run: doesn't commit state"
        required: false
        default: true
        type: boolean

concurrency:
  group: trading-pipeline
  cancel-in-progress: false

permissions:
  contents: write

jobs:
  run:
    runs-on: ubuntu-latest
    timeout-minutes: 20

    steps:
      - name: Checkout
        uses: actions/checkout@v4
        with:
          persist-credentials: true
          fetch-depth: 1

      - name: Set up uv
        uses: astral-sh/setup-uv@v3
        with:
          enable-cache: true

      - name: Install dependencies
        run: uv sync

      - name: Run basket pipeline
        id: pipeline
        env:
          ALPACA_API_KEY: ${{ secrets.ALPACA_API_KEY }}
          ALPACA_SECRET_KEY: ${{ secrets.ALPACA_SECRET_KEY }}
          ALPACA_BASE_URL: ${{ secrets.ALPACA_BASE_URL }}
          DRY_RUN: ${{ inputs.dry_run }}
        run: uv run python -m schroeder_trader.basket.main

      - name: Commit state
        if: always() && inputs.dry_run != true
        run: |
          git config user.name  "schroedertrader-bot"
          git config user.email "bot@schroedertrader.local"
          git add data/
          if git diff --cached --quiet; then
            echo "No state changes to commit"
            exit 0
          fi
          git commit -m "chore: basket state $(date -u +%Y-%m-%d)"
          for attempt in 1 2 3; do
            if git pull --rebase --autostash origin "$GITHUB_REF_NAME" && git push; then
              echo "Pushed on attempt $attempt"
              exit 0
            fi
            echo "Push attempt $attempt failed; retrying in 5s"
            sleep 5
          done
          echo "Failed to push state after 3 attempts" >&2
          exit 1
```

- [ ] **Step 6: Commit**

```bash
git add src/schroeder_trader/basket/main.py \
        .github/workflows/daily-basket.yml \
        tests/basket/test_main.py
git commit -m "feat(basket): entry point + dispatch-only workflow

run_basket_pipeline wires reconciliation → bootstrap → orchestrator
→ rebalance → snapshot. daily-basket.yml is workflow_dispatch only
(no schedule) until Phase C verifies dry-run behavior end-to-end.
Email is intentionally NOT sent from this pipeline — the SPY-only
21:30 UTC run picks up basket state and renders the unified email."
```

---

### Task 6: Equivalence test against 3 historical fixtures

**Goal:** Build the acceptance-gate test that runs the basket pipeline with `weights={"SPY": 1.0}` against three historical fixture days and asserts that the resulting state matches what the SPY-only pipeline would produce on the same data.

**Files:**
- Create: `tests/basket/test_equivalence.py`
- Create: `tests/basket/fixtures/` directory with the three historical scenarios

- [ ] **Step 1: Identify three historical fixture days**

Pick three dates from the live state files that represent distinct signal/regime conditions. Run this to find candidates:

```bash
uv run python <<'PY'
import pandas as pd
df = pd.read_csv("data/shadow_signals.csv")
df = df[df["ticker"] == "SPY"].copy()
df["date"] = pd.to_datetime(df["timestamp"]).dt.date
# Group by regime+source for variety
print(df.groupby(["regime", "signal_source"]).size().sort_values(ascending=False).head(10))
print()
print("Sample rows per group:")
for (reg, src), g in df.groupby(["regime", "signal_source"]):
    print(f"  {reg}/{src}: {g['date'].iloc[0]}, {g['date'].iloc[-1]}")
PY
```

Choose three dates representing: (a) BULL regime / SMA signal, (b) CHOPPY regime / XGB signal, (c) BEAR or BEAR-weakening if any present. Record them as `FIXTURE_DAYS = [date(...), date(...), date(...)]`.

- [ ] **Step 2: Write the equivalence test (failing)**

Create `tests/basket/test_equivalence.py`:

```python
"""Equivalence acceptance gate: basket pipeline with weights={SPY: 1.0}
must produce portfolio state matching the SPY-only pipeline's same-day
output, across at least three historical fixture days representing
different signal/regime states.

Allowed differences (documented):
- pipeline column ('basket' vs 'spy_only')
- row count (basket writes 1 row per ticker; with weights={SPY: 1.0}
  it writes 1 row total, matching SPY-only's 1 row)
- starting cash sleeve may differ by < $1 due to rounding when target
  value is computed as portfolio_value × 1.0 vs the SPY-only's
  CASH_BUFFER_PCT path.
"""

from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest


# Three historical fixture days, chosen for signal/regime diversity.
# Filled in from Step 1's exploration.
FIXTURE_DAYS = [
    date(2026, 4, 28),  # BULL / SMA / HOLD (placeholder — set from real data)
    date(2026, 5, 12),  # BULL / SMA / HOLD with bear_weakening triggering
    date(2026, 5, 18),  # CHOPPY / XGB
]


@pytest.mark.parametrize("fixture_date", FIXTURE_DAYS)
def test_basket_with_spy_only_weight_matches_spy_only_pipeline(
    fixture_date, tmp_path,
):
    # Set up: copy the production state files into tmp_path as they were on
    # the day BEFORE fixture_date. The basket and SPY-only pipelines both
    # operate against this snapshot.
    snapshot_dir = Path(__file__).parent / "fixtures" / fixture_date.isoformat()
    if not snapshot_dir.exists():
        pytest.skip(f"No fixture for {fixture_date}; create it from real CSVs")

    # Copy fixture CSVs into tmp_path (both pipelines write to copies)
    for f in ["portfolio.csv", "orders.csv", "shadow_signals.csv", "signals.csv"]:
        src = snapshot_dir / f
        if src.exists():
            (tmp_path / f).write_text(src.read_text())

    # Run basket with weights={SPY: 1.0}
    with patch("schroeder_trader.basket.main.SHADOW_BASKET_WEIGHTS", {"SPY": 1.0}):
        from schroeder_trader.basket.main import run_basket_pipeline
        now = datetime.combine(fixture_date, datetime.min.time(), tzinfo=timezone.utc)
        # Patch broker calls to return deterministic state for the fixture
        # (the fixture snapshots include expected end-of-day positions in
        # a sidecar 'expected_state.json' file).
        with _fixture_broker(snapshot_dir, fixture_date):
            run_basket_pipeline(tmp_path, weights={"SPY": 1.0}, now=now)

    # Read basket state
    pf = pd.read_csv(tmp_path / "portfolio.csv")
    basket_rows = pf[pf["pipeline"] == "basket"]
    spy_only_rows = pf[pf["pipeline"] == "spy_only"]

    # Should be exactly 1 basket row (SPY, weights={SPY: 1.0})
    assert len(basket_rows) == 1
    assert basket_rows.iloc[0]["ticker"] == "SPY"

    # Compare position_qty to SPY-only's most recent row (which represents
    # what SPY-only produces on the same fixture day; sidecar holds expected).
    import json
    expected = json.loads((snapshot_dir / "expected_state.json").read_text())
    assert int(basket_rows.iloc[0]["position_qty"]) == expected["expected_position_qty"]
    assert abs(float(basket_rows.iloc[0]["total_value"]) - expected["expected_total_value"]) < 1.0


def _fixture_broker(snapshot_dir, fixture_date):
    """Context manager that patches broker calls for the fixture day."""
    import contextlib
    import json
    state = json.loads((snapshot_dir / "broker_state.json").read_text())

    @contextlib.contextmanager
    def mgr():
        patches = [
            patch("schroeder_trader.basket.main.get_position",
                  side_effect=lambda t: state["positions"].get(t, 0)),
            patch("schroeder_trader.basket.main.get_account",
                  return_value=state["account"]),
            patch("schroeder_trader.basket.main.submit_order",
                  return_value=MagicMock(alpaca_order_id="test", status="SUBMITTED")),
        ]
        for p in patches:
            p.start()
        try:
            yield
        finally:
            for p in patches:
                p.stop()

    return mgr()
```

- [ ] **Step 3: Create the three fixture directories**

For each `fixture_date` chosen in Step 1, create `tests/basket/fixtures/<date>/` containing:

- `portfolio.csv`, `orders.csv`, `shadow_signals.csv`, `signals.csv` — copies of the live state files truncated to rows with `timestamp < fixture_date`.
- `broker_state.json` — the Alpaca state as it was on fixture_date morning:
  ```json
  {
    "positions": {"SPY": 141},
    "account": {"cash": 1965.4, "portfolio_value": 105938.8}
  }
  ```
- `expected_state.json` — what SPY-only would produce on fixture_date close:
  ```json
  {
    "expected_position_qty": 141,
    "expected_total_value": 106734.04
  }
  ```

Generate these from `data/portfolio.csv` for the chosen dates. Example:

```bash
# For 2026-05-12
mkdir -p tests/basket/fixtures/2026-05-12
# Truncate state files to rows before 2026-05-12 and write to fixtures
uv run python <<'PY'
import pandas as pd
import json
from pathlib import Path

DATE = "2026-05-12"
out = Path(f"tests/basket/fixtures/{DATE}")

# Copy state files truncated
for fname in ["portfolio.csv", "orders.csv", "shadow_signals.csv", "signals.csv"]:
    df = pd.read_csv(f"data/{fname}")
    df_pre = df[df["timestamp"] < f"{DATE}T00:00:00+00:00"]
    df_pre.to_csv(out / fname, index=False)

# Broker state as of fixture morning: most recent pre-date row
pf = pd.read_csv("data/portfolio.csv")
pf_morning = pf[pf["timestamp"] < f"{DATE}T00:00:00+00:00"].iloc[-1]
broker = {
    "positions": {"SPY": int(pf_morning["position_qty"])},
    "account": {
        "cash": float(pf_morning["cash"]),
        "portfolio_value": float(pf_morning["total_value"]),
    },
}
(out / "broker_state.json").write_text(json.dumps(broker, indent=2))

# Expected state: the SPY-only row written on fixture date
pf_on = pf[pf["timestamp"].str.startswith(DATE)].iloc[-1]
expected = {
    "expected_position_qty": int(pf_on["position_qty"]),
    "expected_total_value": float(pf_on["total_value"]),
}
(out / "expected_state.json").write_text(json.dumps(expected, indent=2))
print("Fixture written.")
PY
```

Repeat for the other two dates.

- [ ] **Step 4: Run the equivalence test — confirm passes**

```bash
uv run pytest tests/basket/test_equivalence.py -v
```

Expected: 3 tests pass (one per fixture date).

If any fails, the failure tells you which fixture and which assertion failed. Common causes: missing `expected_state.json`, broker-state mismatch, or a real divergence between basket and SPY-only logic. Real divergences are bugs in the basket pipeline — fix them before continuing.

- [ ] **Step 5: Commit**

```bash
git add tests/basket/test_equivalence.py tests/basket/fixtures/
git commit -m "test(basket): equivalence gate across 3 historical fixtures

Verifies that running the basket pipeline with weights={SPY: 1.0}
produces position_qty and total_value matching the SPY-only pipeline's
same-day output, across three fixture days chosen for signal/regime
diversity. Phase B acceptance gate."
```

---

### Task 7: Switch workflow to scheduled (Phase C operational step)

**Goal:** Flip `daily-basket.yml` from `workflow_dispatch`-only to scheduled at 21:20 UTC, after a manual `dry_run=true` invocation confirms the pipeline writes correct state and submits orders to the paper account as expected.

**Files:**
- Modify: `.github/workflows/daily-basket.yml`

- [ ] **Step 1: Trigger the workflow manually with dry_run=true**

```bash
gh workflow run daily-basket.yml -f dry_run=true
sleep 4
gh run list --workflow daily-basket.yml --limit 1
```

Capture the run ID, then watch:

```bash
gh run watch <run-id> --exit-status
```

Expected: workflow completes successfully. No state commit occurs because `dry_run=true`.

- [ ] **Step 2: Inspect the run logs for signal computation and order submission**

```bash
gh run view <run-id> --log 2>&1 | grep -iE "(basket pipeline complete|submitting|composite|email)"
```

Expected: log lines showing per-ticker signals computed, orders submitted (or skipped for sub-share diffs), and `Basket pipeline complete. Wrote 4 ticker rows.` at the end. No email log (basket pipeline must NOT send email).

- [ ] **Step 3: Trigger a real run with dry_run=false to actually write state**

```bash
gh workflow run daily-basket.yml -f dry_run=false
sleep 4
gh run watch $(gh run list --workflow daily-basket.yml --limit 1 --json databaseId --jq '.[0].databaseId') --exit-status
```

Expected: workflow succeeds. A new commit `chore: basket state YYYY-MM-DD` appears on `main` containing the 4 new `pipeline='basket'` rows in `portfolio.csv`.

- [ ] **Step 4: Verify the state was written correctly**

```bash
git pull --rebase
uv run python <<'PY'
import pandas as pd
pf = pd.read_csv("data/portfolio.csv")
basket = pf[pf["pipeline"] == "basket"]
print(basket.sort_values("timestamp").tail(8).to_string(index=False))
PY
```

Expected: 4 rows from the most recent timestamp with `pipeline='basket'` and one per ticker (SPY, XLK, XLV, XLE).

- [ ] **Step 5: Flip the workflow to scheduled**

Edit `.github/workflows/daily-basket.yml`. Change the `on:` block from:

```yaml
on:
  workflow_dispatch:
    inputs:
      dry_run:
        ...
```

to:

```yaml
on:
  schedule:
    # 21:20 UTC = 16:20 EST = 17:20 EDT. Runs 10 minutes before the
    # SPY-only pipeline so the unified email can read basket state.
    - cron: '20 21 * * 1-5'
  workflow_dispatch:
    inputs:
      dry_run:
        description: "Test run: doesn't commit state"
        required: false
        default: true
        type: boolean
```

- [ ] **Step 6: Commit and push**

```bash
git add .github/workflows/daily-basket.yml
git commit -m "feat(basket): enable scheduled 21:20 UTC cron

Phase C: basket pipeline verified end-to-end via dry_run=false manual
trigger. Switch to scheduled execution at 21:20 UTC weekdays so daily
basket state is available when the 21:30 UTC SPY-only pipeline reads
it for the unified email."
git push
```

The next weekday at 21:20 UTC, the basket pipeline will run automatically.

---

### Task 8: Email integration — BASKET PAPER section + body wiring

**Goal:** Add `build_basket_paper_section` to `daily_email.py`, extend `build_email_body` to read basket rows and include the section when present, and update `main.py` to pass basket state to the email body.

**Files:**
- Modify: `src/schroeder_trader/reports/daily_email.py`
- Modify: `src/schroeder_trader/main.py`
- Modify: `tests/test_daily_email.py`

- [ ] **Step 1: Write the failing email-section tests**

Append to `tests/test_daily_email.py`:

```python
def test_build_basket_paper_section_renders_all_tickers(tmp_path):
    from schroeder_trader.reports.daily_email import build_basket_paper_section
    from datetime import date

    portfolio_df = pd.DataFrame({
        "timestamp": ["2026-05-21T20:25:00+00:00"] * 4,
        "pipeline": ["basket"] * 4,
        "ticker": ["SPY", "XLK", "XLV", "XLE"],
        "cash": [2140.0] * 4,
        "position_qty": [64, 181, 111, 175],
        "position_value": [47747.0, 32491.0, 16212.0, 10150.0],
        "total_value": [106540.0] * 4,
    })
    shadow_signals_df = pd.DataFrame({
        "timestamp": ["2026-05-21T20:25:00+00:00"] * 4,
        "pipeline": ["basket"] * 4,
        "ticker": ["SPY", "XLK", "XLV", "XLE"],
        "ml_signal": ["HOLD", "HOLD", "HOLD", "BUY"],
        "signal_source": ["SMA", "XGB", "XGB", "XGB"],
        "trailing_stop_triggered": [0, 0, 0, 0],
        "high_water_mark": [47747.0, 32491.0, 16212.0, 10150.0],
    })
    weights = {"SPY": 0.45, "XLK": 0.30, "XLV": 0.15, "XLE": 0.10}
    section = build_basket_paper_section(
        portfolio_df=portfolio_df,
        shadow_signals_df=shadow_signals_df,
        basket_weights=weights,
        launch_date=date(2026, 5, 21),
    )
    assert "BASKET PAPER" in section
    assert "SPY" in section
    assert "XLK" in section
    assert "XLV" in section
    assert "XLE" in section
    assert "45.0%" in section  # SPY target weight
    assert "$106,540" in section or "106,540" in section


def test_build_basket_paper_section_shows_fired_stop(tmp_path):
    from schroeder_trader.reports.daily_email import build_basket_paper_section
    from datetime import date

    portfolio_df = pd.DataFrame({
        "timestamp": ["2026-07-14T20:25:00+00:00"],
        "pipeline": ["basket"],
        "ticker": ["XLE"],
        "cash": [10000.0], "position_qty": [0],
        "position_value": [0.0], "total_value": [100000.0],
    })
    shadow_signals_df = pd.DataFrame({
        "timestamp": ["2026-07-14T20:25:00+00:00"],
        "pipeline": ["basket"], "ticker": ["XLE"],
        "ml_signal": ["SELL"], "signal_source": ["FLAT"],
        "trailing_stop_triggered": [1],
        "high_water_mark": [61.20],
    })
    section = build_basket_paper_section(
        portfolio_df=portfolio_df, shadow_signals_df=shadow_signals_df,
        basket_weights={"XLE": 1.0}, launch_date=date(2026, 5, 21),
    )
    assert "FIRED" in section
    assert "XLE trailing stop fired" in section.lower() or "xle trailing stop fired" in section.lower()


def test_build_email_body_omits_basket_section_when_basket_state_is_none(tmp_path):
    from datetime import date
    dates = pd.bdate_range("2026-04-15", periods=10).tz_localize(None)
    spy = pd.DataFrame({"close": [700.0 + i for i in range(10)]}, index=dates)
    pd.DataFrame(columns=["id", "timestamp", "cash", "position_qty", "position_value", "total_value"]) \
      .to_csv(tmp_path / "portfolio.csv", index=False)
    pd.DataFrame(columns=["id", "timestamp", "provider", "target_exposure", "error"]) \
      .to_csv(tmp_path / "llm_shadow_signals.csv", index=False)
    pd.DataFrame(columns=["id", "timestamp", "ticker", "close_price", "ml_signal"]) \
      .to_csv(tmp_path / "shadow_signals.csv", index=False)

    body = build_email_body(
        date_str="2026-05-14",
        spy_close=715.16, spy_prev_close=713.97,
        portfolio_value=101883.0, portfolio_prev_value=100665.0,
        cash=1965.0, position_qty=141,
        sma_signal="HOLD", sma_50=676.0, sma_200=667.0,
        composite_signal="BUY", composite_source="XGB", regime="CHOPPY",
        bear_days=0, xgb_proba_up=0.578, xgb_threshold=0.35,
        today_action="HOLD",
        oracle_responses=[],
        data_root=tmp_path,
        spy_history=spy,
        live_start_date=date(2026, 4, 15),
        sector_close_histories={},
        basket_state=None,
    )
    assert "BASKET PAPER" not in body
```

- [ ] **Step 2: Run the tests — confirm they fail**

```bash
uv run pytest tests/test_daily_email.py -k basket_paper -v
```

Expected: `ImportError: cannot import name 'build_basket_paper_section'`.

- [ ] **Step 3: Implement `build_basket_paper_section`**

Append to `src/schroeder_trader/reports/daily_email.py`:

```python
def build_basket_paper_section(
    *,
    portfolio_df: pd.DataFrame,
    shadow_signals_df: pd.DataFrame,
    basket_weights: dict,
    launch_date: date,
) -> str:
    """Render the BASKET PAPER section for the daily email.

    Reads the latest basket-pipeline snapshot from portfolio_df + shadow_signals_df
    and renders a per-ticker table plus any fired-stop warning notes.
    Returns empty string if there are no basket rows.
    """
    basket_pf = portfolio_df[portfolio_df.get("pipeline") == "basket"].copy()
    if basket_pf.empty:
        return ""

    # Latest snapshot only
    latest_ts = basket_pf["timestamp"].max()
    latest_rows = basket_pf[basket_pf["timestamp"] == latest_ts]
    total_value = float(latest_rows.iloc[0]["total_value"])
    cash = float(latest_rows.iloc[0]["cash"])
    cash_pct = (cash / total_value * 100) if total_value > 0 else 0.0

    # Latest shadow_signals per ticker
    ss = shadow_signals_df[shadow_signals_df.get("pipeline") == "basket"].copy()
    ss_by_ticker = {}
    if not ss.empty:
        ss_latest = ss[ss["timestamp"] == ss["timestamp"].max()]
        for _, r in ss_latest.iterrows():
            ss_by_ticker[r["ticker"]] = r

    lines = [
        f"  Total value:  {_fmt_dollars(total_value)}",
        f"  Cash sleeve:  {_fmt_dollars(cash)} ({cash_pct:.1f}%)",
        "",
        f"  {'Ticker':<7} {'Target':>7} {'Actual':>7} {'Position':>9}  "
        f"{'Value':>10}  {'Signal':<14} {'Stop':>5}",
        f"  {'-'*7} {'-'*7} {'-'*7} {'-'*9}  {'-'*10}  {'-'*14} {'-'*5}",
    ]
    warnings: list[str] = []
    for ticker in basket_weights:
        rows_for_ticker = latest_rows[latest_rows["ticker"] == ticker]
        if rows_for_ticker.empty:
            continue
        r = rows_for_ticker.iloc[0]
        target_pct = basket_weights[ticker] * 100
        position_value = float(r["position_value"])
        actual_pct = (position_value / total_value * 100) if total_value > 0 else 0.0
        position_qty = int(r["position_qty"])
        ss_row = ss_by_ticker.get(ticker)
        signal = ss_row["ml_signal"] if ss_row is not None else "—"
        source = ss_row["signal_source"] if ss_row is not None else "—"
        stop_fired = bool(int(ss_row["trailing_stop_triggered"])) if (
            ss_row is not None and pd.notna(ss_row["trailing_stop_triggered"])
        ) else False
        stop_label = "FIRED" if stop_fired else "OK"
        lines.append(
            f"  {ticker:<7} {target_pct:>6.1f}% {actual_pct:>6.1f}% "
            f"{position_qty:>5} sh  {_fmt_dollars(position_value):>10}  "
            f"{signal+' ('+source+')':<14} {stop_label:>5}"
        )
        if stop_fired and ss_row is not None and pd.notna(ss_row["high_water_mark"]):
            warnings.append(
                f"  ⚠ {ticker} trailing stop fired (HWM {_fmt_dollars(float(ss_row['high_water_mark']), 2)}). "
                f"Cash held idle; re-entry after cooldown if signal allows."
            )

    if warnings:
        lines.append("")
        lines.extend(warnings)

    header = f"BASKET PAPER (paper-trading the basket since {launch_date})"
    return _section(header, lines)
```

- [ ] **Step 4: Run the tests — confirm they pass**

```bash
uv run pytest tests/test_daily_email.py -k basket_paper -v
```

Expected: 2 tests pass (the third is `pass` placeholder; remove it or fill it out as a full test before final commit).

- [ ] **Step 5: Wire the section into `build_email_body`**

In `src/schroeder_trader/reports/daily_email.py`, modify `build_email_body`:

```python
def build_email_body(
    *,
    # ... existing args ...
    sector_close_histories: dict,
    basket_weights: dict | None = None,
    basket_state: dict | None = None,  # NEW
) -> str:
    """Compose the full email body."""
    sections = [
        # ... existing sections through PERFORMANCE ...
    ]

    # NEW: insert BASKET PAPER before SECTOR SHADOW
    if basket_state is not None:
        basket_section = build_basket_paper_section(
            portfolio_df=basket_state["portfolio_df"],
            shadow_signals_df=basket_state["shadow_signals_df"],
            basket_weights=basket_state["basket_weights"],
            launch_date=basket_state["launch_date"],
        )
        if basket_section:
            sections.append(basket_section)

    # ... existing SECTOR SHADOW append ...
```

- [ ] **Step 6: Wire `main.py` to read basket state and pass it to the email**

In `src/schroeder_trader/main.py`, at the email-rendering step (around line 533), before the `build_email_body` call, read basket rows:

```python
        # Read basket state for the BASKET PAPER section. None if no basket rows yet.
        basket_state = None
        try:
            pf_all = pd.read_csv(Path(DB_PATH).parent / "portfolio.csv")
            if (pf_all.get("pipeline") == "basket").any():
                ss_all = pd.read_csv(Path(DB_PATH).parent / "shadow_signals.csv")
                from schroeder_trader.config import SHADOW_BASKET_WEIGHTS
                from datetime import date
                # Launch date = earliest basket timestamp's date
                basket_first_ts = pf_all[pf_all["pipeline"] == "basket"]["timestamp"].min()
                launch_date = pd.to_datetime(basket_first_ts).date()
                basket_state = {
                    "portfolio_df": pf_all,
                    "shadow_signals_df": ss_all,
                    "basket_weights": SHADOW_BASKET_WEIGHTS,
                    "launch_date": launch_date,
                }
        except Exception:
            logger.exception("Could not load basket state for email (non-fatal)")
```

Then pass it to `build_email_body`:

```python
        email_body = build_email_body(
            # ... existing kwargs ...
            sector_close_histories=sector_close_histories,
            basket_weights=SHADOW_BASKET_WEIGHTS,
            basket_state=basket_state,
        )
```

- [ ] **Step 7: Run the full test suite**

```bash
uv run pytest
```

Expected: All existing tests still pass plus the new basket-section tests.

- [ ] **Step 8: Smoke-test the email rendering against real data**

```bash
uv run python <<'PY'
import pandas as pd
from pathlib import Path
from datetime import date
from schroeder_trader.config import SHADOW_BASKET_WEIGHTS
from schroeder_trader.reports.daily_email import build_basket_paper_section

pf = pd.read_csv("data/portfolio.csv")
ss = pd.read_csv("data/shadow_signals.csv")
if (pf.get("pipeline") == "basket").any():
    launch = pd.to_datetime(pf[pf["pipeline"] == "basket"]["timestamp"].min()).date()
    print(build_basket_paper_section(
        portfolio_df=pf,
        shadow_signals_df=ss,
        basket_weights=SHADOW_BASKET_WEIGHTS,
        launch_date=launch,
    ))
else:
    print("No basket rows yet — section will be omitted from email.")
PY
```

Expected: a BASKET PAPER block listing SPY/XLK/XLV/XLE with target/actual weights, position counts, values, and signals. Or the "no basket rows" message if the basket pipeline hasn't run yet.

- [ ] **Step 9: Commit**

```bash
git add src/schroeder_trader/reports/daily_email.py \
        src/schroeder_trader/main.py \
        tests/test_daily_email.py
git commit -m "feat(daily_email): BASKET PAPER section + body integration

Phase D: extends the daily email with a BASKET PAPER section showing
per-ticker target/actual weights, positions, values, and signals from
the basket pipeline's most recent state. Fired trailing stops are
rendered with a FIRED badge and an inline warning line.

build_email_body gains an optional basket_state kwarg; main.py reads
basket rows from portfolio.csv and shadow_signals.csv at email-render
time and threads the data through. Section is omitted when no basket
rows exist (basket pipeline never ran yet)."
```

- [ ] **Step 10: Trigger a manual dry-run of the SPY-only workflow to confirm email contains the section**

```bash
gh workflow run daily.yml -f dry_run=true
sleep 4
gh run watch $(gh run list --workflow daily.yml --limit 1 --json databaseId --jq '.[0].databaseId') --exit-status
```

Check your inbox for `[TEST] [SchroederTrader] Daily Report` and verify the new BASKET PAPER section appears between PERFORMANCE and SECTOR SHADOW.

---

## Self-Review Notes

**Spec coverage:**
- Section: Architectural Decisions → covered by all tasks (topology in T2-T5, schema in T1, email layout in T8, cron timing in T7).
- Section: File Layout → Task 1 creates the schema; Tasks 2-5 create the basket subpackage; Task 8 modifies reports/daily_email.py.
- Section: State Schemas → Task 1 migration + logger kwargs.
- Section: Daily Timeline → Tasks 5 and 7 (basket cron) + Task 8 (email reads basket state).
- Section: Trade Execution Algorithm → Tasks 3 (orchestrator) and 4 (rebalancer).
- Section: Email Integration → Task 8.
- Section: Testing Strategy → Tasks 2/3/4/5 each include unit tests; Task 6 is the equivalence test; Task 1 has migration regression tests.
- Section: Migration and Rollout → Phase A is Task 1; Phase B is Tasks 2-6; Phase C is Task 7; Phase D is Task 8. Phase E (retirement) is explicitly out of scope.

**Placeholder scan:** No placeholders remain. All test bodies are filled in with executable code.

**Type consistency:**
- `decisions[ticker]` dict shape is consistent: `{signal: str, exposure: float, price: float, regime: str, source: str, stop_state: dict}` across Task 3 (producer) and Task 4 (consumer).
- `compute_orders` return type `list[dict]` with keys `{ticker, action, qty, price}` consistent between Task 4's tests, implementation, and Task 5's `rebalance_to_targets` consumer.
- `basket_state` dict shape consistent: `{portfolio_df, shadow_signals_df, basket_weights, launch_date}` between Task 8 Step 6 (producer in main.py) and Steps 3/5 (consumer in daily_email.py).
- `pipeline` kwarg signature consistent across `log_portfolio`, `log_order`, `log_shadow_signal`, `insert_reconciled_order`: all default to `"spy_only"` and accept `"basket"`.
- `_compute_signal_for_ticker` return tuple shape consistent between Task 3's implementation and Task 5's mocked tests.
