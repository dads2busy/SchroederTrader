# Phase 1: SMA Crossover Bot — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fully automated SMA crossover trading bot for SPY on Alpaca paper trading, with backtesting, SQLite logging, and email alerts.

**Architecture:** Modular pipeline with 6 independent modules (data, strategy, risk, execution, storage, alerts) orchestrated by a thin `main.py`. Each module has a clean interface and is independently testable. Cron triggers the pipeline daily after market close.

**Tech Stack:** Python 3.12+, uv, alpaca-py, yfinance, pandas, vectorbt, SQLite, Gmail SMTP, pytest

**Spec:** `docs/superpowers/specs/2026-03-18-phase1-sma-crossover-bot-design.md`

---

## File Structure

```
SchroederTrader/
├── src/
│   └── schroeder_trader/
│       ├── __init__.py              # Package marker
│       ├── main.py                  # Orchestrator: runs the 9-step pipeline
│       ├── config.py                # Constants + env var loading
│       ├── data/
│       │   ├── __init__.py
│       │   └── market_data.py       # fetch_daily_bars(ticker, days) → DataFrame
│       ├── strategy/
│       │   ├── __init__.py
│       │   └── sma_crossover.py     # generate_signal(df) → Signal enum
│       ├── risk/
│       │   ├── __init__.py
│       │   └── risk_manager.py      # evaluate(signal, portfolio, price) → OrderRequest | None
│       ├── execution/
│       │   ├── __init__.py
│       │   └── broker.py            # submit_order(order_request) → OrderResult
│       ├── alerts/
│       │   ├── __init__.py
│       │   └── email_alert.py       # send_trade_alert(), send_error_alert(), send_daily_summary()
│       └── storage/
│           ├── __init__.py
│           └── trade_log.py         # init_db(), log_signal(), log_order(), log_portfolio(), etc.
├── backtest/
│   ├── run_backtest.py              # vectorbt backtest harness
│   ├── download_data.py             # Download + cache SPY history to CSV
│   └── data/                        # Cached CSV files (gitignored)
├── tests/
│   ├── __init__.py
│   ├── test_config.py
│   ├── test_market_data.py
│   ├── test_sma_crossover.py
│   ├── test_risk_manager.py
│   ├── test_broker.py
│   ├── test_email_alert.py
│   ├── test_trade_log.py
│   └── test_main.py                # Integration test
├── data/                            # SQLite database lives here (gitignored)
├── logs/                            # Log files (gitignored)
├── .env                             # Secrets (gitignored)
├── .env.example                     # Template with placeholders
├── .gitignore
└── pyproject.toml
```

---

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `.env.example`
- Create: all `__init__.py` files
- Create: `src/schroeder_trader/config.py`

- [ ] **Step 1: Initialize uv project**

```bash
cd ~/git/SchroederTrader
uv init --lib --name schroeder-trader
```

This creates `pyproject.toml` and `src/schroeder_trader/__init__.py`. If uv creates files in a different layout, adjust to match our structure.

- [ ] **Step 2: Edit pyproject.toml**

Replace the generated `pyproject.toml` with:

```toml
[project]
name = "schroeder-trader"
version = "0.1.0"
description = "AI-driven stock trading system - Phase 1: SMA Crossover Bot"
requires-python = ">=3.12"
dependencies = [
    "alpaca-py>=0.33.0",
    "yfinance>=0.2.40",
    "pandas>=2.2.0",
    "python-dotenv>=1.0.0",
]

[project.optional-dependencies]
backtest = [
    "vectorbt>=0.26.0",
    "kaleido>=0.2.0",
]
dev = [
    "pytest>=8.0.0",
    "pytest-mock>=3.14.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/schroeder_trader"]
```

- [ ] **Step 3: Create .gitignore**

```
# Python
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
.venv/

# Secrets
.env

# Data
data/trades.db
backtest/data/*.csv

# Logs
logs/

# OS
.DS_Store
```

- [ ] **Step 4: Create .env.example**

```
# Alpaca API (https://app.alpaca.markets)
ALPACA_API_KEY=your-api-key
ALPACA_SECRET_KEY=your-secret-key
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Gmail SMTP (https://myaccount.google.com/apppasswords)
ALERT_EMAIL_FROM=your-email@gmail.com
ALERT_EMAIL_TO=your-email@gmail.com
ALERT_EMAIL_APP_PASSWORD=xxxx-xxxx-xxxx-xxxx
```

- [ ] **Step 5: Create config.py**

```python
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "trades.db"
LOG_DIR = PROJECT_ROOT / "logs"

# Trading parameters
TICKER = "SPY"
SMA_SHORT_WINDOW = 50
SMA_LONG_WINDOW = 200
CASH_BUFFER_PCT = 0.02
SLIPPAGE_ESTIMATE = 0.0005

# Alpaca
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

# Email
ALERT_EMAIL_FROM = os.getenv("ALERT_EMAIL_FROM", "")
ALERT_EMAIL_TO = os.getenv("ALERT_EMAIL_TO", "")
ALERT_EMAIL_APP_PASSWORD = os.getenv("ALERT_EMAIL_APP_PASSWORD", "")
```

- [ ] **Step 6: Create directory structure and __init__.py files**

```bash
mkdir -p src/schroeder_trader/{data,strategy,risk,execution,alerts,storage}
mkdir -p tests backtest/data data logs

touch src/schroeder_trader/__init__.py
touch src/schroeder_trader/data/__init__.py
touch src/schroeder_trader/strategy/__init__.py
touch src/schroeder_trader/risk/__init__.py
touch src/schroeder_trader/execution/__init__.py
touch src/schroeder_trader/alerts/__init__.py
touch src/schroeder_trader/storage/__init__.py
touch tests/__init__.py
```

- [ ] **Step 7: Create virtual environment and install dependencies**

```bash
cd ~/git/SchroederTrader
uv venv
uv pip install -e ".[dev,backtest]"
```

- [ ] **Step 8: Verify setup**

```bash
cd ~/git/SchroederTrader
.venv/bin/python -c "from schroeder_trader.config import TICKER; print(f'Config OK: {TICKER}')"
```

Expected: `Config OK: SPY`

- [ ] **Step 9: Write config test**

Create `tests/test_config.py`:

```python
from schroeder_trader.config import TICKER, SMA_SHORT_WINDOW, SMA_LONG_WINDOW, DB_PATH


def test_ticker_is_spy():
    assert TICKER == "SPY"


def test_sma_windows():
    assert SMA_SHORT_WINDOW == 50
    assert SMA_LONG_WINDOW == 200


def test_db_path_is_absolute():
    assert DB_PATH.is_absolute()
```

- [ ] **Step 10: Run tests**

```bash
cd ~/git/SchroederTrader
uv run pytest tests/test_config.py -v
```

Expected: 3 tests PASS

- [ ] **Step 11: Commit**

```bash
git add pyproject.toml .gitignore .env.example src/ tests/test_config.py tests/__init__.py
git commit -m "feat: project scaffolding with uv, config, and directory structure"
```

---

### Task 2: Storage Module (SQLite)

**Files:**
- Create: `src/schroeder_trader/storage/trade_log.py`
- Create: `tests/test_trade_log.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_trade_log.py`:

```python
import sqlite3
from datetime import datetime, timezone

from schroeder_trader.storage.trade_log import (
    init_db,
    log_signal,
    log_order,
    log_portfolio,
    get_signal_by_date,
    get_pending_orders,
    update_order_fill,
)


def test_init_db_creates_tables(tmp_path):
    db_path = tmp_path / "test.db"
    conn = init_db(db_path)
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cursor.fetchall()}
    assert {"signals", "orders", "portfolio"} <= tables
    conn.close()


def test_log_signal(tmp_path):
    db_path = tmp_path / "test.db"
    conn = init_db(db_path)
    now = datetime.now(timezone.utc)
    signal_id = log_signal(conn, now, "SPY", 523.10, 520.0, 518.0, "BUY")
    assert signal_id == 1

    row = conn.execute("SELECT * FROM signals WHERE id = ?", (signal_id,)).fetchone()
    assert row is not None
    assert row[2] == "SPY"  # ticker
    assert row[6] == "BUY"  # signal
    conn.close()


def test_log_order(tmp_path):
    db_path = tmp_path / "test.db"
    conn = init_db(db_path)
    now = datetime.now(timezone.utc)
    signal_id = log_signal(conn, now, "SPY", 523.10, 520.0, 518.0, "BUY")
    order_id = log_order(conn, signal_id, "alpaca-123", now, "SPY", "BUY", 45, "SUBMITTED")
    assert order_id == 1
    conn.close()


def test_log_portfolio(tmp_path):
    db_path = tmp_path / "test.db"
    conn = init_db(db_path)
    now = datetime.now(timezone.utc)
    log_portfolio(conn, now, 5000.0, 45, 23539.5, 28539.5)
    row = conn.execute("SELECT * FROM portfolio WHERE id = 1").fetchone()
    assert row[3] == 5000.0  # cash
    conn.close()


def test_get_signal_by_date_returns_none_when_missing(tmp_path):
    db_path = tmp_path / "test.db"
    conn = init_db(db_path)
    result = get_signal_by_date(conn, "2026-03-18")
    assert result is None
    conn.close()


def test_get_signal_by_date_returns_existing(tmp_path):
    db_path = tmp_path / "test.db"
    conn = init_db(db_path)
    now = datetime(2026, 3, 18, 21, 30, 0, tzinfo=timezone.utc)
    log_signal(conn, now, "SPY", 523.10, 520.0, 518.0, "HOLD")
    result = get_signal_by_date(conn, "2026-03-18")
    assert result is not None
    conn.close()


def test_get_pending_orders(tmp_path):
    db_path = tmp_path / "test.db"
    conn = init_db(db_path)
    now = datetime.now(timezone.utc)
    signal_id = log_signal(conn, now, "SPY", 523.10, 520.0, 518.0, "BUY")
    log_order(conn, signal_id, "alpaca-123", now, "SPY", "BUY", 45, "SUBMITTED")
    pending = get_pending_orders(conn)
    assert len(pending) == 1
    assert pending[0]["alpaca_order_id"] == "alpaca-123"
    conn.close()


def test_update_order_fill(tmp_path):
    db_path = tmp_path / "test.db"
    conn = init_db(db_path)
    now = datetime.now(timezone.utc)
    signal_id = log_signal(conn, now, "SPY", 523.10, 520.0, 518.0, "BUY")
    order_id = log_order(conn, signal_id, "alpaca-123", now, "SPY", "BUY", 45, "SUBMITTED")
    update_order_fill(conn, "alpaca-123", 523.50, now, "FILLED")
    row = conn.execute("SELECT fill_price, status FROM orders WHERE id = ?", (order_id,)).fetchone()
    assert row[0] == 523.50
    assert row[1] == "FILLED"
    conn.close()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_trade_log.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'schroeder_trader.storage.trade_log'`

- [ ] **Step 3: Implement trade_log.py**

Create `src/schroeder_trader/storage/trade_log.py`:

```python
import sqlite3
from datetime import datetime
from pathlib import Path


def init_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            ticker TEXT NOT NULL,
            close_price REAL NOT NULL,
            sma_50 REAL NOT NULL,
            sma_200 REAL NOT NULL,
            signal TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_id INTEGER NOT NULL,
            alpaca_order_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            ticker TEXT NOT NULL,
            action TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            fill_price REAL,
            fill_timestamp TEXT,
            status TEXT NOT NULL,
            FOREIGN KEY (signal_id) REFERENCES signals(id)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS portfolio (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            cash REAL NOT NULL,
            position_qty INTEGER NOT NULL,
            position_value REAL NOT NULL,
            total_value REAL NOT NULL
        )
    """)
    conn.commit()
    return conn


def log_signal(
    conn: sqlite3.Connection,
    timestamp: datetime,
    ticker: str,
    close_price: float,
    sma_50: float,
    sma_200: float,
    signal: str,
) -> int:
    cursor = conn.execute(
        "INSERT INTO signals (timestamp, ticker, close_price, sma_50, sma_200, signal) VALUES (?, ?, ?, ?, ?, ?)",
        (timestamp.isoformat(), ticker, close_price, sma_50, sma_200, signal),
    )
    conn.commit()
    return cursor.lastrowid


def log_order(
    conn: sqlite3.Connection,
    signal_id: int,
    alpaca_order_id: str,
    timestamp: datetime,
    ticker: str,
    action: str,
    quantity: int,
    status: str,
) -> int:
    cursor = conn.execute(
        "INSERT INTO orders (signal_id, alpaca_order_id, timestamp, ticker, action, quantity, status) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (signal_id, alpaca_order_id, timestamp.isoformat(), ticker, action, quantity, status),
    )
    conn.commit()
    return cursor.lastrowid


def log_portfolio(
    conn: sqlite3.Connection,
    timestamp: datetime,
    cash: float,
    position_qty: int,
    position_value: float,
    total_value: float,
) -> int:
    cursor = conn.execute(
        "INSERT INTO portfolio (timestamp, cash, position_qty, position_value, total_value) VALUES (?, ?, ?, ?, ?)",
        (timestamp.isoformat(), cash, position_qty, position_value, total_value),
    )
    conn.commit()
    return cursor.lastrowid


def get_signal_by_date(conn: sqlite3.Connection, date_str: str) -> dict | None:
    row = conn.execute(
        "SELECT * FROM signals WHERE timestamp LIKE ?", (f"{date_str}%",)
    ).fetchone()
    if row is None:
        return None
    return dict(row)


def get_pending_orders(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM orders WHERE status = 'SUBMITTED'"
    ).fetchall()
    return [dict(row) for row in rows]


def update_order_fill(
    conn: sqlite3.Connection,
    alpaca_order_id: str,
    fill_price: float,
    fill_timestamp: datetime,
    status: str,
) -> None:
    conn.execute(
        "UPDATE orders SET fill_price = ?, fill_timestamp = ?, status = ? WHERE alpaca_order_id = ?",
        (fill_price, fill_timestamp.isoformat(), status, alpaca_order_id),
    )
    conn.commit()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_trade_log.py -v
```

Expected: 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/schroeder_trader/storage/trade_log.py tests/test_trade_log.py
git commit -m "feat: SQLite storage module with signal, order, and portfolio logging"
```

---

### Task 3: Strategy Module (SMA Crossover)

**Files:**
- Create: `src/schroeder_trader/strategy/sma_crossover.py`
- Create: `tests/test_sma_crossover.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_sma_crossover.py`:

```python
import pandas as pd
import numpy as np

from schroeder_trader.strategy.sma_crossover import Signal, generate_signal


def _make_df(prices: list[float]) -> pd.DataFrame:
    """Create a DataFrame with the given closing prices."""
    return pd.DataFrame({
        "close": prices,
        "open": prices,
        "high": prices,
        "low": prices,
        "volume": [1000000] * len(prices),
    }, index=pd.date_range("2020-01-01", periods=len(prices), freq="B"))


def test_hold_when_sma50_above_sma200_no_crossover():
    # SMA50 has been above SMA200 for a while — no crossover, HOLD
    prices = [100.0] * 200 + [110.0] * 50  # step up happened long ago
    df = _make_df(prices)
    signal, sma_50, sma_200 = generate_signal(df)
    assert signal == Signal.HOLD


def test_buy_on_golden_cross():
    # Construct data where SMA50 crosses above SMA200 on the last bar
    prices = [100.0] * 200
    # Last 50 bars trend upward to push SMA50 above SMA200
    for i in range(50):
        prices.append(100.0 + (i + 1) * 1.0)
    df = _make_df(prices)
    signal, sma_50, sma_200 = generate_signal(df)
    # Verify SMA50 > SMA200 on last bar
    assert sma_50 > sma_200
    # The signal depends on whether a crossover happened — check the logic
    # is correct by verifying the previous bar's relationship
    sma_50_prev = df["close"].iloc[:-1].tail(50).mean()
    sma_200_prev = df["close"].iloc[:-1].tail(200).mean()
    if sma_50_prev <= sma_200_prev:
        assert signal == Signal.BUY
    else:
        assert signal == Signal.HOLD


def test_sell_on_death_cross():
    # Start with SMA50 above, then drop
    prices = [120.0] * 200
    for i in range(50):
        prices.append(120.0 - (i + 1) * 1.5)
    df = _make_df(prices)
    signal, sma_50, sma_200 = generate_signal(df)
    assert sma_50 < sma_200
    sma_50_prev = df["close"].iloc[:-1].tail(50).mean()
    sma_200_prev = df["close"].iloc[:-1].tail(200).mean()
    if sma_50_prev >= sma_200_prev:
        assert signal == Signal.SELL
    else:
        assert signal == Signal.HOLD


def test_hold_when_no_crossover_sma50_below():
    # SMA50 has been below SMA200 for a while — no crossover
    prices = [100.0] * 200 + [90.0] * 50
    df = _make_df(prices)
    signal, sma_50, sma_200 = generate_signal(df)
    assert signal == Signal.HOLD


def test_requires_minimum_200_bars():
    prices = [100.0] * 100  # Not enough data
    df = _make_df(prices)
    try:
        generate_signal(df)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "200" in str(e)


def test_returns_sma_values():
    prices = [100.0] * 250
    df = _make_df(prices)
    signal, sma_50, sma_200 = generate_signal(df)
    assert isinstance(sma_50, float)
    assert isinstance(sma_200, float)
    assert abs(sma_50 - 100.0) < 0.01
    assert abs(sma_200 - 100.0) < 0.01
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_sma_crossover.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement sma_crossover.py**

Create `src/schroeder_trader/strategy/sma_crossover.py`:

```python
import enum

import pandas as pd

from schroeder_trader.config import SMA_SHORT_WINDOW, SMA_LONG_WINDOW


class Signal(enum.Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


def generate_signal(
    df: pd.DataFrame,
    short_window: int = SMA_SHORT_WINDOW,
    long_window: int = SMA_LONG_WINDOW,
) -> tuple[Signal, float, float]:
    """Generate a trading signal based on SMA crossover.

    Args:
        df: DataFrame with a 'close' column and at least `long_window + 1` rows.
        short_window: Short SMA window (default 50).
        long_window: Long SMA window (default 200).

    Returns:
        Tuple of (Signal, sma_short_value, sma_long_value).

    Raises:
        ValueError: If DataFrame has fewer than long_window + 1 rows.
    """
    min_rows = long_window + 1
    if len(df) < min_rows:
        raise ValueError(f"Need at least {min_rows} rows, got {len(df)}. Minimum {long_window}+ bars required.")

    sma_short = df["close"].rolling(window=short_window).mean()
    sma_long = df["close"].rolling(window=long_window).mean()

    current_short = float(sma_short.iloc[-1])
    current_long = float(sma_long.iloc[-1])
    prev_short = float(sma_short.iloc[-2])
    prev_long = float(sma_long.iloc[-2])

    if current_short > current_long and prev_short <= prev_long:
        return Signal.BUY, current_short, current_long
    elif current_short < current_long and prev_short >= prev_long:
        return Signal.SELL, current_short, current_long
    else:
        return Signal.HOLD, current_short, current_long
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_sma_crossover.py -v
```

Expected: 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/schroeder_trader/strategy/sma_crossover.py tests/test_sma_crossover.py
git commit -m "feat: SMA crossover strategy module with signal generation"
```

---

### Task 4: Risk Module

**Files:**
- Create: `src/schroeder_trader/risk/risk_manager.py`
- Create: `tests/test_risk_manager.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_risk_manager.py`:

```python
import math

from schroeder_trader.risk.risk_manager import evaluate, OrderRequest
from schroeder_trader.strategy.sma_crossover import Signal


def test_buy_calculates_whole_shares():
    request = evaluate(
        signal=Signal.BUY,
        portfolio_value=10000.0,
        close_price=523.10,
        current_position_qty=0,
    )
    assert request is not None
    assert request.action == "BUY"
    expected_qty = math.floor(10000.0 * (1 - 0.02) / 523.10)
    assert request.quantity == expected_qty


def test_buy_when_already_holding_returns_none():
    request = evaluate(
        signal=Signal.BUY,
        portfolio_value=10000.0,
        close_price=523.10,
        current_position_qty=19,
    )
    assert request is None


def test_sell_with_position_returns_sell():
    request = evaluate(
        signal=Signal.SELL,
        portfolio_value=10000.0,
        close_price=523.10,
        current_position_qty=19,
    )
    assert request is not None
    assert request.action == "SELL"
    assert request.quantity == 19


def test_sell_without_position_returns_none():
    request = evaluate(
        signal=Signal.SELL,
        portfolio_value=10000.0,
        close_price=523.10,
        current_position_qty=0,
    )
    assert request is None


def test_hold_returns_none():
    request = evaluate(
        signal=Signal.HOLD,
        portfolio_value=10000.0,
        close_price=523.10,
        current_position_qty=19,
    )
    assert request is None


def test_buy_with_small_portfolio():
    # Portfolio too small to buy even 1 share
    request = evaluate(
        signal=Signal.BUY,
        portfolio_value=100.0,
        close_price=523.10,
        current_position_qty=0,
    )
    assert request is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_risk_manager.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement risk_manager.py**

Create `src/schroeder_trader/risk/risk_manager.py`:

```python
import logging
import math
from dataclasses import dataclass

from schroeder_trader.config import CASH_BUFFER_PCT
from schroeder_trader.strategy.sma_crossover import Signal

logger = logging.getLogger(__name__)


@dataclass
class OrderRequest:
    action: str  # "BUY" or "SELL"
    quantity: int


def evaluate(
    signal: Signal,
    portfolio_value: float,
    close_price: float,
    current_position_qty: int,
    cash_buffer_pct: float = CASH_BUFFER_PCT,
) -> OrderRequest | None:
    """Evaluate a signal and return an order request or None.

    Args:
        signal: Trading signal (BUY/SELL/HOLD).
        portfolio_value: Total portfolio value (cash + positions).
        close_price: Current close price of the ticker.
        current_position_qty: Number of shares currently held.
        cash_buffer_pct: Fraction of portfolio to keep as cash reserve.

    Returns:
        OrderRequest if action needed, None otherwise.
    """
    if signal == Signal.HOLD:
        return None

    if signal == Signal.BUY:
        if current_position_qty > 0:
            logger.info("BUY signal but already holding %d shares, skipping", current_position_qty)
            return None
        available = portfolio_value * (1 - cash_buffer_pct)
        quantity = math.floor(available / close_price)
        if quantity < 1:
            logger.warning("Portfolio too small to buy even 1 share at $%.2f", close_price)
            return None
        return OrderRequest(action="BUY", quantity=quantity)

    if signal == Signal.SELL:
        if current_position_qty <= 0:
            logger.info("SELL signal but no position held, treating as HOLD")
            return None
        return OrderRequest(action="SELL", quantity=current_position_qty)

    return None
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_risk_manager.py -v
```

Expected: 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/schroeder_trader/risk/risk_manager.py tests/test_risk_manager.py
git commit -m "feat: risk management module with position sizing and guards"
```

---

### Task 5: Data Module (Market Data)

**Files:**
- Create: `src/schroeder_trader/data/market_data.py`
- Create: `tests/test_market_data.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_market_data.py`:

```python
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

from schroeder_trader.data.market_data import fetch_daily_bars, is_market_open_today


def _mock_bars_df(n: int = 250) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "open": np.random.uniform(500, 550, n),
        "high": np.random.uniform(550, 560, n),
        "low": np.random.uniform(490, 500, n),
        "close": np.random.uniform(500, 550, n),
        "volume": np.random.randint(50_000_000, 100_000_000, n),
    }, index=dates)


@patch("schroeder_trader.data.market_data._get_data_client")
def test_fetch_daily_bars_returns_dataframe(mock_client):
    mock_df = _mock_bars_df()
    client = MagicMock()
    client.get_stock_bars.return_value.df = mock_df
    mock_client.return_value = client

    df = fetch_daily_bars("SPY", days=250)
    assert isinstance(df, pd.DataFrame)
    assert "close" in df.columns
    assert len(df) >= 200


@patch("schroeder_trader.data.market_data._get_data_client")
def test_fetch_daily_bars_has_required_columns(mock_client):
    mock_df = _mock_bars_df()
    client = MagicMock()
    client.get_stock_bars.return_value.df = mock_df
    mock_client.return_value = client

    df = fetch_daily_bars("SPY")
    for col in ["open", "high", "low", "close", "volume"]:
        assert col in df.columns


@patch("schroeder_trader.data.market_data._get_trading_client")
def test_is_market_open_today(mock_client):
    client = MagicMock()
    calendar_entry = MagicMock()
    calendar_entry.date = pd.Timestamp("2026-03-18")
    client.get_calendar.return_value = [calendar_entry]
    mock_client.return_value = client

    result = is_market_open_today("2026-03-18")
    assert isinstance(result, bool)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_market_data.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement market_data.py**

Create `src/schroeder_trader/data/market_data.py`:

```python
import logging
from datetime import datetime, timedelta

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.requests import GetCalendarRequest

from schroeder_trader.config import ALPACA_API_KEY, ALPACA_SECRET_KEY
from schroeder_trader.execution.broker import _get_trading_client

logger = logging.getLogger(__name__)

_data_client = None


def _get_data_client() -> StockHistoricalDataClient:
    global _data_client
    if _data_client is None:
        _data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
    return _data_client


def fetch_daily_bars(ticker: str, days: int = 365) -> pd.DataFrame:
    """Fetch daily OHLCV bars from Alpaca.

    Args:
        ticker: Stock symbol (e.g., "SPY").
        days: Number of calendar days of history to fetch.

    Returns:
        DataFrame with columns: open, high, low, close, volume.
        Index is DatetimeIndex.
    """
    client = _get_data_client()
    end = datetime.now()
    start = end - timedelta(days=days)

    request = StockBarsRequest(
        symbol_or_symbols=ticker,
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
    )
    bars = client.get_stock_bars(request)
    df = bars.df

    # If multi-index (symbol, timestamp), drop the symbol level
    if isinstance(df.index, pd.MultiIndex):
        df = df.droplevel("symbol")

    logger.info("Fetched %d bars for %s", len(df), ticker)
    return df[["open", "high", "low", "close", "volume"]]


def is_market_open_today(date_str: str | None = None) -> bool:
    """Check if the market is open on the given date.

    Args:
        date_str: Date string in YYYY-MM-DD format. Defaults to today.

    Returns:
        True if market is open, False otherwise.
    """
    client = _get_trading_client()
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    calendar = client.get_calendar(GetCalendarRequest(start=date_str, end=date_str))
    if not calendar:
        return False

    return str(calendar[0].date) == date_str
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_market_data.py -v
```

Expected: 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/schroeder_trader/data/market_data.py tests/test_market_data.py
git commit -m "feat: market data module with Alpaca daily bar fetching"
```

---

### Task 6: Execution Module (Broker)

**Files:**
- Create: `src/schroeder_trader/execution/broker.py`
- Create: `tests/test_broker.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_broker.py`:

```python
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone

from schroeder_trader.execution.broker import submit_order, get_order_status, get_position, get_account, OrderResult
from schroeder_trader.risk.risk_manager import OrderRequest


@patch("schroeder_trader.execution.broker._get_trading_client")
def test_submit_buy_order(mock_client):
    client = MagicMock()
    order_response = MagicMock()
    order_response.id = "order-abc-123"
    order_response.status = "accepted"
    order_response.submitted_at = datetime.now(timezone.utc)
    client.submit_order.return_value = order_response
    mock_client.return_value = client

    request = OrderRequest(action="BUY", quantity=45)
    result = submit_order(request, "SPY")
    assert isinstance(result, OrderResult)
    assert result.alpaca_order_id == "order-abc-123"
    assert result.status == "SUBMITTED"
    assert result.quantity == 45


@patch("schroeder_trader.execution.broker._get_trading_client")
def test_submit_sell_order(mock_client):
    client = MagicMock()
    order_response = MagicMock()
    order_response.id = "order-def-456"
    order_response.status = "accepted"
    order_response.submitted_at = datetime.now(timezone.utc)
    client.submit_order.return_value = order_response
    mock_client.return_value = client

    request = OrderRequest(action="SELL", quantity=19)
    result = submit_order(request, "SPY")
    assert result.alpaca_order_id == "order-def-456"
    assert result.quantity == 19


@patch("schroeder_trader.execution.broker._get_trading_client")
def test_get_order_status_filled(mock_client):
    from alpaca.trading.enums import OrderStatus
    client = MagicMock()
    order = MagicMock()
    order.status = OrderStatus.FILLED
    order.filled_avg_price = 523.50
    order.filled_at = datetime.now(timezone.utc)
    client.get_order_by_id.return_value = order
    mock_client.return_value = client

    status = get_order_status("order-abc-123")
    assert status["status"] == "filled"
    assert status["fill_price"] == 523.50


@patch("schroeder_trader.execution.broker._get_trading_client")
def test_get_position_none(mock_client):
    client = MagicMock()
    client.get_open_position.side_effect = Exception("position does not exist")
    mock_client.return_value = client

    qty = get_position("SPY")
    assert qty == 0


@patch("schroeder_trader.execution.broker._get_trading_client")
def test_get_account(mock_client):
    client = MagicMock()
    account = MagicMock()
    account.portfolio_value = "28539.50"
    account.cash = "5000.00"
    client.get_account.return_value = account
    mock_client.return_value = client

    info = get_account()
    assert info["portfolio_value"] == 28539.50
    assert info["cash"] == 5000.00
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_broker.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement broker.py**

Create `src/schroeder_trader/execution/broker.py`:

```python
import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, OrderStatus, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from schroeder_trader.config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL
from schroeder_trader.risk.risk_manager import OrderRequest

logger = logging.getLogger(__name__)

_client = None


@dataclass
class OrderResult:
    alpaca_order_id: str
    quantity: int
    timestamp: datetime
    status: str


def _get_trading_client() -> TradingClient:
    global _client
    if _client is None:
        _client = TradingClient(
            ALPACA_API_KEY,
            ALPACA_SECRET_KEY,
            paper="paper" in ALPACA_BASE_URL,
        )
    return _client


def submit_order(request: OrderRequest, ticker: str) -> OrderResult:
    """Submit a market order to Alpaca.

    Args:
        request: OrderRequest with action and quantity.
        ticker: Stock symbol.

    Returns:
        OrderResult with Alpaca order ID and submission details.
    """
    client = _get_trading_client()
    side = OrderSide.BUY if request.action == "BUY" else OrderSide.SELL

    order_data = MarketOrderRequest(
        symbol=ticker,
        qty=request.quantity,
        side=side,
        time_in_force=TimeInForce.DAY,
    )

    order = client.submit_order(order_data)
    logger.info("Submitted %s order for %d %s — Alpaca ID: %s", request.action, request.quantity, ticker, order.id)

    return OrderResult(
        alpaca_order_id=str(order.id),
        quantity=request.quantity,
        timestamp=order.submitted_at or datetime.now(timezone.utc),
        status="SUBMITTED",
    )


def get_order_status(alpaca_order_id: str) -> dict:
    """Check the status of an existing order.

    Returns:
        Dict with 'status', 'fill_price' (if filled), 'fill_timestamp' (if filled).
    """
    client = _get_trading_client()
    order = client.get_order_by_id(alpaca_order_id)
    status_str = order.status.value if hasattr(order.status, 'value') else str(order.status)
    result = {"status": status_str}
    if order.status == OrderStatus.FILLED:
        result["fill_price"] = float(order.filled_avg_price)
        result["fill_timestamp"] = order.filled_at
    return result


def get_position(ticker: str) -> int:
    """Get current position quantity for a ticker. Returns 0 if no position."""
    client = _get_trading_client()
    try:
        position = client.get_open_position(ticker)
        return int(position.qty)
    except Exception:
        return 0


def get_account() -> dict:
    """Get account info: portfolio_value and cash."""
    client = _get_trading_client()
    account = client.get_account()
    return {
        "portfolio_value": float(account.portfolio_value),
        "cash": float(account.cash),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_broker.py -v
```

Expected: 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/schroeder_trader/execution/broker.py tests/test_broker.py
git commit -m "feat: broker execution module with Alpaca order submission and status"
```

---

### Task 7: Alerts Module (Email)

**Files:**
- Create: `src/schroeder_trader/alerts/email_alert.py`
- Create: `tests/test_email_alert.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_email_alert.py`:

```python
from unittest.mock import patch, MagicMock

from schroeder_trader.alerts.email_alert import send_trade_alert, send_error_alert, send_daily_summary


@patch("schroeder_trader.alerts.email_alert.smtplib.SMTP_SSL")
def test_send_trade_alert(mock_smtp_cls):
    mock_smtp = MagicMock()
    mock_smtp_cls.return_value.__enter__ = MagicMock(return_value=mock_smtp)
    mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)

    send_trade_alert(
        action="BUY",
        ticker="SPY",
        quantity=45,
        portfolio_value=28539.50,
        cash=5000.00,
        sma_50=525.0,
        sma_200=518.0,
    )
    mock_smtp.send_message.assert_called_once()


@patch("schroeder_trader.alerts.email_alert.smtplib.SMTP_SSL")
def test_send_error_alert(mock_smtp_cls):
    mock_smtp = MagicMock()
    mock_smtp_cls.return_value.__enter__ = MagicMock(return_value=mock_smtp)
    mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)

    send_error_alert("Data fetch failed", "ConnectionError: timeout")
    mock_smtp.send_message.assert_called_once()


@patch("schroeder_trader.alerts.email_alert.smtplib.SMTP_SSL")
def test_send_daily_summary(mock_smtp_cls):
    mock_smtp = MagicMock()
    mock_smtp_cls.return_value.__enter__ = MagicMock(return_value=mock_smtp)
    mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)

    send_daily_summary(
        portfolio_value=28539.50,
        cash=5000.00,
        position_qty=45,
        signal="HOLD",
        sma_50=525.0,
        sma_200=518.0,
    )
    mock_smtp.send_message.assert_called_once()


@patch("schroeder_trader.alerts.email_alert.smtplib.SMTP_SSL")
def test_send_fill_alert(mock_smtp_cls):
    mock_smtp = MagicMock()
    mock_smtp_cls.return_value.__enter__ = MagicMock(return_value=mock_smtp)
    mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)

    from schroeder_trader.alerts.email_alert import send_fill_alert
    send_fill_alert(
        action="BUY",
        ticker="SPY",
        quantity=45,
        fill_price=523.15,
    )
    mock_smtp.send_message.assert_called_once()


@patch("schroeder_trader.alerts.email_alert.smtplib.SMTP_SSL")
def test_send_trade_alert_smtp_failure_does_not_raise(mock_smtp_cls):
    mock_smtp_cls.side_effect = Exception("SMTP connection failed")
    # Should not raise — just log the error
    send_trade_alert(
        action="BUY",
        ticker="SPY",
        quantity=45,
        portfolio_value=28539.50,
        cash=5000.00,
        sma_50=525.0,
        sma_200=518.0,
    )
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_email_alert.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement email_alert.py**

Create `src/schroeder_trader/alerts/email_alert.py`:

```python
import logging
import smtplib
from email.message import EmailMessage

from schroeder_trader.config import ALERT_EMAIL_FROM, ALERT_EMAIL_TO, ALERT_EMAIL_APP_PASSWORD

logger = logging.getLogger(__name__)

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465


def _send_email(subject: str, body: str) -> None:
    """Send an email via Gmail SMTP. Logs and swallows errors."""
    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = ALERT_EMAIL_FROM
        msg["To"] = ALERT_EMAIL_TO
        msg.set_content(body)

        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as smtp:
            smtp.login(ALERT_EMAIL_FROM, ALERT_EMAIL_APP_PASSWORD)
            smtp.send_message(msg)

        logger.info("Email sent: %s", subject)
    except Exception:
        logger.exception("Failed to send email: %s", subject)


def send_trade_alert(
    action: str,
    ticker: str,
    quantity: int,
    portfolio_value: float,
    cash: float,
    sma_50: float,
    sma_200: float,
) -> None:
    subject = f"[SchroederTrader] SUBMITTED: {action} {quantity} {ticker} (fills at next open)"
    body = (
        f"Trade Submitted\n"
        f"{'=' * 40}\n"
        f"Action: {action}\n"
        f"Ticker: {ticker}\n"
        f"Quantity: {quantity} shares\n\n"
        f"Portfolio: ${portfolio_value:,.2f}\n"
        f"Cash: ${cash:,.2f}\n\n"
        f"SMA 50: {sma_50:.2f}\n"
        f"SMA 200: {sma_200:.2f}\n"
    )
    _send_email(subject, body)


def send_fill_alert(
    action: str,
    ticker: str,
    quantity: int,
    fill_price: float,
) -> None:
    subject = f"[SchroederTrader] FILLED: {action} {quantity} {ticker} @ ${fill_price:.2f}"
    body = (
        f"Order Filled\n"
        f"{'=' * 40}\n"
        f"Action: {action}\n"
        f"Ticker: {ticker}\n"
        f"Quantity: {quantity} shares\n"
        f"Fill Price: ${fill_price:.2f}\n"
    )
    _send_email(subject, body)


def send_error_alert(error_type: str, details: str) -> None:
    subject = f"[SchroederTrader] ERROR: {error_type}"
    body = (
        f"Error Report\n"
        f"{'=' * 40}\n"
        f"Error: {error_type}\n\n"
        f"Details:\n{details}\n"
    )
    _send_email(subject, body)


def send_daily_summary(
    portfolio_value: float,
    cash: float,
    position_qty: int,
    signal: str,
    sma_50: float,
    sma_200: float,
) -> None:
    subject = f"[SchroederTrader] Daily run complete - Portfolio: ${portfolio_value:,.0f}"
    body = (
        f"Daily Summary\n"
        f"{'=' * 40}\n"
        f"Signal: {signal}\n"
        f"Portfolio Value: ${portfolio_value:,.2f}\n"
        f"Cash: ${cash:,.2f}\n"
        f"Position: {position_qty} shares SPY\n\n"
        f"SMA 50: {sma_50:.2f}\n"
        f"SMA 200: {sma_200:.2f}\n"
    )
    _send_email(subject, body)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_email_alert.py -v
```

Expected: 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/schroeder_trader/alerts/email_alert.py tests/test_email_alert.py
git commit -m "feat: email alerts module with trade, error, and daily summary notifications"
```

---

### Task 8: Logging Setup

**Files:**
- Create: `src/schroeder_trader/logging_config.py`
- Create: `tests/test_logging_config.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_logging_config.py`:

```python
import logging

from schroeder_trader.logging_config import setup_logging


def test_setup_logging_configures_root_logger(tmp_path):
    log_dir = tmp_path / "logs"
    setup_logging(log_dir=log_dir)

    logger = logging.getLogger("schroeder_trader")
    assert logger.level == logging.DEBUG

    # Should have at least a console handler and a file handler
    handler_types = [type(h).__name__ for h in logger.handlers]
    assert "StreamHandler" in handler_types
    assert "RotatingFileHandler" in handler_types

    # Clean up handlers to avoid affecting other tests
    logger.handlers.clear()


def test_log_file_created(tmp_path):
    log_dir = tmp_path / "logs"
    setup_logging(log_dir=log_dir)

    logger = logging.getLogger("schroeder_trader")
    logger.info("test message")

    log_file = log_dir / "schroeder_trader.log"
    assert log_file.exists()

    logger.handlers.clear()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_logging_config.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement logging_config.py**

Create `src/schroeder_trader/logging_config.py`:

```python
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from schroeder_trader.config import LOG_DIR


def setup_logging(log_dir: Path = LOG_DIR) -> None:
    """Configure logging with console (INFO) and file (DEBUG) handlers."""
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("schroeder_trader")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    # Console handler — INFO
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    logger.addHandler(console)

    # File handler — DEBUG with rotation
    file_handler = RotatingFileHandler(
        log_dir / "schroeder_trader.log",
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=3,
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_logging_config.py -v
```

Expected: 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/schroeder_trader/logging_config.py tests/test_logging_config.py
git commit -m "feat: logging setup with console and rotating file handlers"
```

---

### Task 9: Orchestrator (main.py)

**Files:**
- Create: `src/schroeder_trader/main.py`
- Create: `tests/test_main.py`

- [ ] **Step 1: Write failing integration test**

Create `tests/test_main.py`:

```python
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

from schroeder_trader.main import run_pipeline


def _mock_bars_df(n: int = 250) -> pd.DataFrame:
    """Flat prices = SMA50 == SMA200 = HOLD signal."""
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "open": [100.0] * n,
        "high": [101.0] * n,
        "low": [99.0] * n,
        "close": [100.0] * n,
        "volume": [1000000] * n,
    }, index=dates)


@patch("schroeder_trader.main.send_daily_summary")
@patch("schroeder_trader.main.get_account")
@patch("schroeder_trader.main.get_position")
@patch("schroeder_trader.main.fetch_daily_bars")
@patch("schroeder_trader.main.is_market_open_today")
@patch("schroeder_trader.main.get_pending_orders")
@patch("schroeder_trader.main.get_signal_by_date")
def test_pipeline_hold_signal(
    mock_signal_by_date,
    mock_pending,
    mock_market_open,
    mock_fetch,
    mock_position,
    mock_account,
    mock_summary,
    tmp_path,
):
    # No prior run today
    mock_signal_by_date.return_value = None
    # No pending orders
    mock_pending.return_value = []
    # Market is open
    mock_market_open.return_value = True
    # Return flat prices → HOLD
    mock_fetch.return_value = _mock_bars_df()
    # No position
    mock_position.return_value = 0
    # Account info
    mock_account.return_value = {"portfolio_value": 10000.0, "cash": 10000.0}

    db_path = tmp_path / "test.db"
    run_pipeline(db_path=db_path)

    # Daily summary should always be sent
    mock_summary.assert_called_once()
    # Verify signal was logged (check the db)
    import sqlite3
    conn = sqlite3.connect(str(db_path))
    row = conn.execute("SELECT signal FROM signals").fetchone()
    assert row[0] == "HOLD"
    conn.close()


@patch("schroeder_trader.main.get_signal_by_date")
def test_pipeline_skips_if_already_ran(mock_signal_by_date, tmp_path):
    mock_signal_by_date.return_value = {"signal": "HOLD"}
    db_path = tmp_path / "test.db"
    # Should exit early without error
    run_pipeline(db_path=db_path)


@patch("schroeder_trader.main.is_market_open_today")
@patch("schroeder_trader.main.get_pending_orders")
@patch("schroeder_trader.main.get_signal_by_date")
def test_pipeline_skips_if_market_closed(
    mock_signal_by_date,
    mock_pending,
    mock_market_open,
    tmp_path,
):
    mock_signal_by_date.return_value = None
    mock_pending.return_value = []
    mock_market_open.return_value = False

    db_path = tmp_path / "test.db"
    run_pipeline(db_path=db_path)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_main.py -v
```

Expected: FAIL — `ModuleNotFoundError` or `ImportError`

- [ ] **Step 3: Implement main.py**

Create `src/schroeder_trader/main.py`:

```python
import logging
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from schroeder_trader.config import DB_PATH, TICKER
from schroeder_trader.logging_config import setup_logging
from schroeder_trader.data.market_data import fetch_daily_bars, is_market_open_today
from schroeder_trader.strategy.sma_crossover import generate_signal
from schroeder_trader.risk.risk_manager import evaluate
from schroeder_trader.execution.broker import (
    submit_order,
    get_order_status,
    get_position,
    get_account,
)
from schroeder_trader.storage.trade_log import (
    init_db,
    log_signal,
    log_order,
    log_portfolio,
    get_signal_by_date,
    get_pending_orders,
    update_order_fill,
)
from schroeder_trader.alerts.email_alert import (
    send_trade_alert,
    send_fill_alert,
    send_error_alert,
    send_daily_summary,
)

logger = logging.getLogger(__name__)


def run_pipeline(db_path: Path = DB_PATH) -> None:
    """Run the full trading pipeline."""
    conn = init_db(db_path)
    now = datetime.now(timezone.utc)
    today = now.strftime("%Y-%m-%d")

    # Step 1: Idempotency check
    existing = get_signal_by_date(conn, today)
    if existing is not None:
        logger.info("Already ran today (%s), exiting", today)
        conn.close()
        return

    # Step 2: Fill check for pending orders
    pending = get_pending_orders(conn)
    for order in pending:
        try:
            status = get_order_status(order["alpaca_order_id"])
            if status["status"] == "filled":
                update_order_fill(
                    conn,
                    order["alpaca_order_id"],
                    status["fill_price"],
                    status["fill_timestamp"],
                    "FILLED",
                )
                send_fill_alert(
                    action=order["action"],
                    ticker=order["ticker"],
                    quantity=order["quantity"],
                    fill_price=status["fill_price"],
                )
                logger.info("Order %s filled at $%.2f", order["alpaca_order_id"], status["fill_price"])
            elif status["status"] in ("canceled", "expired", "rejected"):
                update_order_fill(conn, order["alpaca_order_id"], 0.0, now, "REJECTED")
                send_error_alert("Order rejected/canceled", f"Order {order['alpaca_order_id']} status: {status['status']}")
        except Exception:
            logger.exception("Error checking order %s", order["alpaca_order_id"])

    # Step 3: Market calendar check
    if not is_market_open_today(today):
        logger.info("Market closed today (%s), exiting", today)
        conn.close()
        return

    # Step 4: Fetch data
    df = fetch_daily_bars(TICKER)
    close_price = float(df["close"].iloc[-1])

    # Step 5: Generate signal
    signal, sma_50, sma_200 = generate_signal(df)
    logger.info("Signal: %s | Close: $%.2f | SMA50: %.2f | SMA200: %.2f", signal.value, close_price, sma_50, sma_200)

    # Step 8 (partial): Log signal
    signal_id = log_signal(conn, now, TICKER, close_price, sma_50, sma_200, signal.value)

    # Step 6: Risk evaluation
    account = get_account()
    position_qty = get_position(TICKER)
    order_request = evaluate(
        signal=signal,
        portfolio_value=account["portfolio_value"],
        close_price=close_price,
        current_position_qty=position_qty,
    )

    # Step 7: Execute order
    if order_request is not None:
        result = submit_order(order_request, TICKER)
        log_order(
            conn, signal_id, result.alpaca_order_id,
            result.timestamp, TICKER, order_request.action,
            order_request.quantity, result.status,
        )
        send_trade_alert(
            action=order_request.action,
            ticker=TICKER,
            quantity=order_request.quantity,
            portfolio_value=account["portfolio_value"],
            cash=account["cash"],
            sma_50=sma_50,
            sma_200=sma_200,
        )

    # Step 8: Log portfolio snapshot
    account = get_account()  # refresh after potential trade
    position_qty = get_position(TICKER)
    position_value = position_qty * close_price
    log_portfolio(conn, now, account["cash"], position_qty, position_value, account["portfolio_value"])

    # Step 9: Daily summary
    send_daily_summary(
        portfolio_value=account["portfolio_value"],
        cash=account["cash"],
        position_qty=position_qty,
        signal=signal.value,
        sma_50=sma_50,
        sma_200=sma_200,
    )

    conn.close()
    logger.info("Pipeline complete")


def main() -> None:
    setup_logging()
    try:
        run_pipeline()
    except Exception:
        logger.exception("Pipeline failed")
        # Note: error logging to SQLite is skipped here because the error may have
        # occurred before init_db completed. Errors are captured via Python logging
        # (file + console) and the email alert. Phase 4 can add structured error
        # logging to SQLite once the risk layer guarantees DB availability.
        try:
            send_error_alert("Pipeline failure", traceback.format_exc())
        except Exception:
            logger.exception("Failed to send error alert")
        sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_main.py -v
```

Expected: 3 tests PASS

- [ ] **Step 5: Run full test suite**

```bash
uv run pytest tests/ -v
```

Expected: All tests PASS (config: 3, trade_log: 8, sma_crossover: 6, risk_manager: 6, market_data: 3, broker: 5, email_alert: 5, logging: 2, main: 3 = ~41 tests)

- [ ] **Step 6: Commit**

```bash
git add src/schroeder_trader/main.py tests/test_main.py
git commit -m "feat: orchestrator main.py with full 9-step pipeline"
```

---

### Task 10: Backtest Harness

**Files:**
- Create: `backtest/download_data.py`
- Create: `backtest/run_backtest.py`

- [ ] **Step 1: Create data download script**

Create `backtest/download_data.py`:

```python
"""Download and cache SPY historical data from yfinance."""
import sys
from pathlib import Path

import yfinance as yf

DATA_DIR = Path(__file__).parent / "data"


def download_spy(start: str = "1993-01-29", end: str | None = None) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output = DATA_DIR / "spy_daily.csv"

    print(f"Downloading SPY data from {start}...")
    spy = yf.download("SPY", start=start, end=end, auto_adjust=True)
    spy.to_csv(output)
    print(f"Saved {len(spy)} rows to {output}")
    return output


if __name__ == "__main__":
    download_spy()
```

- [ ] **Step 2: Download data**

```bash
cd ~/git/SchroederTrader
uv run python backtest/download_data.py
```

Expected: CSV file created at `backtest/data/spy_daily.csv` with ~8000+ rows

- [ ] **Step 3: Create backtest harness**

Create `backtest/run_backtest.py`:

```python
"""Run vectorbt backtest using the SMA crossover strategy on cached SPY data."""
import json
from pathlib import Path

import pandas as pd
import vectorbt as vbt

from schroeder_trader.config import SMA_SHORT_WINDOW, SMA_LONG_WINDOW, SLIPPAGE_ESTIMATE

DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "results"


def load_data() -> pd.DataFrame:
    csv_path = DATA_DIR / "spy_daily.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No cached data at {csv_path}. Run download_data.py first.")
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    # Handle multi-level columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def run_backtest() -> dict:
    df = load_data()
    close = df["Close"]

    # Compute SMAs
    sma_short = vbt.MA.run(close, window=SMA_SHORT_WINDOW)
    sma_long = vbt.MA.run(close, window=SMA_LONG_WINDOW)

    # Generate crossover entries and exits
    entries = sma_short.ma_crossed_above(sma_long)
    exits = sma_short.ma_crossed_below(sma_long)

    # Run portfolio simulation with slippage
    portfolio = vbt.Portfolio.from_signals(
        close,
        entries=entries,
        exits=exits,
        init_cash=10000,
        fees=SLIPPAGE_ESTIMATE,
        freq="1D",
    )

    # Compute metrics
    stats = portfolio.stats()
    total_return = float(stats["Total Return [%]"])
    sharpe = float(stats.get("Sharpe Ratio", 0.0))
    max_dd = float(stats["Max Drawdown [%]"])
    total_trades = int(stats["Total Trades"])
    win_rate = float(stats.get("Win Rate [%]", 0.0))

    # Buy-and-hold benchmark
    bh_portfolio = vbt.Portfolio.from_holding(close, init_cash=10000, freq="1D")
    bh_stats = bh_portfolio.stats()
    bh_return = float(bh_stats["Total Return [%]"])
    bh_sharpe = float(bh_stats.get("Sharpe Ratio", 0.0))
    bh_max_dd = float(bh_stats["Max Drawdown [%]"])

    # Train/test split at 2020
    split_date = "2020-01-01"
    test_close = close[close.index >= split_date]
    test_entries = entries[entries.index >= split_date]
    test_exits = exits[exits.index >= split_date]

    test_portfolio = vbt.Portfolio.from_signals(
        test_close,
        entries=test_entries,
        exits=test_exits,
        init_cash=10000,
        fees=SLIPPAGE_ESTIMATE,
        freq="1D",
    )
    test_stats = test_portfolio.stats()
    test_sharpe = float(test_stats.get("Sharpe Ratio", 0.0))

    results = {
        "full_period": {
            "total_return_pct": round(total_return, 2),
            "sharpe_ratio": round(sharpe, 4),
            "max_drawdown_pct": round(max_dd, 2),
            "total_trades": total_trades,
            "win_rate_pct": round(win_rate, 2),
        },
        "benchmark_buy_and_hold": {
            "total_return_pct": round(bh_return, 2),
            "sharpe_ratio": round(bh_sharpe, 4),
            "max_drawdown_pct": round(bh_max_dd, 2),
        },
        "out_of_sample_post_2020": {
            "sharpe_ratio": round(test_sharpe, 4),
        },
    }

    # Print results
    print("\n" + "=" * 50)
    print("SMA CROSSOVER BACKTEST RESULTS")
    print("=" * 50)
    print(f"\nFull Period ({close.index[0].date()} to {close.index[-1].date()}):")
    print(f"  Total Return: {total_return:.2f}%")
    print(f"  Sharpe Ratio: {sharpe:.4f}")
    print(f"  Max Drawdown: {max_dd:.2f}%")
    print(f"  Total Trades: {total_trades}")
    print(f"  Win Rate: {win_rate:.2f}%")
    print(f"\nBuy & Hold Benchmark:")
    print(f"  Total Return: {bh_return:.2f}%")
    print(f"  Sharpe Ratio: {bh_sharpe:.4f}")
    print(f"  Max Drawdown: {bh_max_dd:.2f}%")
    print(f"\nOut-of-Sample (post-2020):")
    print(f"  Sharpe Ratio: {test_sharpe:.4f}")
    print("=" * 50)

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "backtest_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_DIR / 'backtest_results.json'}")

    # Validate that vectorbt signals match generate_signal() logic
    from schroeder_trader.strategy.sma_crossover import generate_signal as gs, Signal
    sample_indices = [250, 500, 1000, len(df) - 1]
    for idx in sample_indices:
        if idx >= len(df):
            continue
        sub_df = df.iloc[:idx + 1].copy()
        sub_df.columns = [c.lower() for c in sub_df.columns]
        if len(sub_df) >= 201:
            sig, _, _ = gs(sub_df)
            vbt_entry = bool(entries.iloc[idx]) if idx < len(entries) else False
            vbt_exit = bool(exits.iloc[idx]) if idx < len(exits) else False
            if sig == Signal.BUY:
                assert vbt_entry, f"Signal mismatch at index {idx}: generate_signal=BUY but vectorbt entry=False"
            elif sig == Signal.SELL:
                assert vbt_exit, f"Signal mismatch at index {idx}: generate_signal=SELL but vectorbt exit=False"
    print("Signal validation: generate_signal() matches vectorbt crossover logic")

    # Save equity curve plot
    fig = portfolio.plot()
    fig.write_image(str(OUTPUT_DIR / "equity_curve.png"))
    print(f"Equity curve saved to {OUTPUT_DIR / 'equity_curve.png'}")

    return results


if __name__ == "__main__":
    run_backtest()
```

- [ ] **Step 4: Run backtest**

```bash
cd ~/git/SchroederTrader
uv run python backtest/run_backtest.py
```

Expected: Prints backtest results to console, saves JSON and PNG to `backtest/results/`

- [ ] **Step 5: Verify Phase 1 gate criterion**

Check that out-of-sample Sharpe ratio (post-2020) is positive. If not, this is expected information — the SMA crossover is a baseline, not a profit engine. Document the result either way.

- [ ] **Step 6: Commit**

```bash
git add backtest/download_data.py backtest/run_backtest.py
git commit -m "feat: backtest harness with vectorbt, train/test split, and equity curve output"
```

---

### Task 11: Cron Setup & Final Integration

**Files:**
- Create: `scripts/setup_cron.sh`

- [ ] **Step 1: Create cron setup script**

Create `scripts/setup_cron.sh`:

```bash
#!/bin/bash
# Install the SchroederTrader cron job
# Runs at 4:30 PM ET on weekdays
#
# NOTE: macOS cron does not support CRON_TZ. We use UTC times instead.
# 4:30 PM ET = 21:30 UTC (EST, Nov-Mar) or 20:30 UTC (EDT, Mar-Nov)
# We schedule at 21:30 UTC (conservative — during EDT this runs at 5:30 PM ET,
# which is still after market close). Adjust if needed.

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PROJECT_DIR}/.venv/bin/python"
MAIN="${PROJECT_DIR}/src/schroeder_trader/main.py"
LOG_DIR="${PROJECT_DIR}/logs"

mkdir -p "${LOG_DIR}"

# Check that python and main.py exist
if [ ! -f "$PYTHON" ]; then
    echo "Error: Python not found at $PYTHON"
    echo "Run 'uv venv && uv pip install -e .[dev,backtest]' first"
    exit 1
fi

# 21:30 UTC = 4:30 PM EST / 5:30 PM EDT (both after market close)
CRON_LINE="30 21 * * 1-5 ${PYTHON} ${MAIN} >> ${LOG_DIR}/cron.log 2>&1"

# Check if already installed
if crontab -l 2>/dev/null | grep -q "schroeder_trader/main.py"; then
    echo "Cron job already installed. Current entry:"
    crontab -l | grep "schroeder_trader"
    exit 0
fi

# Add to crontab
(crontab -l 2>/dev/null; echo "$CRON_LINE") | crontab -
echo "Cron job installed:"
echo "$CRON_LINE"
```

- [ ] **Step 2: Make executable**

```bash
chmod +x scripts/setup_cron.sh
```

- [ ] **Step 3: Create .env from .env.example**

```bash
cp .env.example .env
```

Then edit `.env` with real Alpaca API keys and Gmail app password. (Manual step — requires user to create Alpaca account and Gmail app password.)

- [ ] **Step 4: Test the full pipeline manually (dry run)**

```bash
cd ~/git/SchroederTrader
uv run python -m schroeder_trader.main
```

Expected: Either runs successfully (if Alpaca credentials are configured) or fails with a clear API authentication error (if not yet configured). Either way, logging output should appear on console.

- [ ] **Step 5: Run full test suite one final time**

```bash
uv run pytest tests/ -v
```

Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add scripts/setup_cron.sh
git commit -m "feat: cron setup script for daily trading pipeline execution"
```

- [ ] **Step 7: Push to GitHub**

```bash
git push -u origin main
```

---

### Task 12: Documentation

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write README**

Create `README.md`:

```markdown
# SchroederTrader

AI-driven stock trading system. Currently in **Phase 1**: rule-based SMA crossover bot for SPY.

## Quick Start

### Prerequisites
- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- [Alpaca](https://app.alpaca.markets) paper trading account
- Gmail account with [app password](https://myaccount.google.com/apppasswords)

### Setup

```bash
# Clone and enter project
git clone https://github.com/dads2busy/SchroederTrader.git
cd SchroederTrader

# Create environment and install dependencies
uv venv
uv pip install -e ".[dev,backtest]"

# Configure credentials
cp .env.example .env
# Edit .env with your Alpaca API keys and Gmail app password

# Run backtest
uv run python backtest/download_data.py
uv run python backtest/run_backtest.py

# Test the pipeline
uv run pytest tests/ -v

# Install cron job (runs daily at 4:30 PM ET)
./scripts/setup_cron.sh
```

## Architecture

Modular pipeline: data → strategy → risk → execution → storage → alerts

| Module | Purpose |
|---|---|
| `data/market_data.py` | Fetch OHLCV bars from Alpaca |
| `strategy/sma_crossover.py` | 50/200 SMA crossover signal generation |
| `risk/risk_manager.py` | Position sizing and guards |
| `execution/broker.py` | Order submission via Alpaca |
| `storage/trade_log.py` | SQLite trade logging |
| `alerts/email_alert.py` | Gmail notifications |

## Phase Roadmap

1. **SMA crossover baseline** (current)
2. ML signal generation (XGBoost)
3. Temporal modeling (LSTM + ensemble)
4. Risk management (Kelly Criterion)
5. Alternative data + sentiment
6. RL execution optimization
7. MLOps + drift monitoring
8. LLM agent layer
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add README with setup instructions and architecture overview"
```

- [ ] **Step 3: Final push**

```bash
git push
```
