# Phase 1: SMA Crossover Bot — Design Spec

## Overview

Phase 1 of SchroederTrader builds a fully automated, rule-based SMA crossover trading bot for SPY on daily bars. The goal is not alpha generation — it's building a complete, tested data → signal → risk → execution → logging → alerting pipeline that serves as the infrastructure foundation and performance baseline for all subsequent phases.

**Gate criterion to advance to Phase 2:** Positive out-of-sample Sharpe ratio on paper account.

## Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Broker | Alpaca (paper trading) | Purpose-built for algo trading, clean API, instant paper accounts |
| Ticker | SPY | Most liquid ETF, 30+ years of clean data, no survivorship bias |
| Strategy | 50/200 SMA crossover | Well-studied baseline, infrequent trades, low noise |
| Timeframe | Daily bars | Matches SMA windows, one evaluation per day |
| Backtesting | vectorbt | Vectorized (fast), actively maintained, pandas-native |
| Storage | SQLite | Zero-config, sufficient for single-ticker Phase 1 |
| Scheduling | Cron (4:30 PM ET, weekdays) | Daily strategy needs only one evaluation per day |
| Alerts | Gmail SMTP | Built-in Python stdlib, no extra dependencies |
| Python env | uv with local .venv | Fast, modern package management |
| Architecture | Modular pipeline | Independently testable modules, swappable for later phases |

## Project Structure

```
SchroederTrader/
├── src/
│   └── schroeder_trader/
│       ├── __init__.py
│       ├── main.py              # Orchestrator: fetch → signal → risk → execute → log → alert
│       ├── config.py            # Central config (API keys, thresholds, ticker, SMA windows)
│       ├── data/
│       │   ├── __init__.py
│       │   └── market_data.py   # Fetch OHLCV from Alpaca (live) + yfinance (backtest history)
│       ├── strategy/
│       │   ├── __init__.py
│       │   └── sma_crossover.py # 50/200 SMA signal generation → BUY / SELL / HOLD
│       ├── risk/
│       │   ├── __init__.py
│       │   └── risk_manager.py  # Position sizing, max exposure check
│       ├── execution/
│       │   ├── __init__.py
│       │   └── broker.py        # Place orders via Alpaca API (paper or live, same interface)
│       ├── alerts/
│       │   ├── __init__.py
│       │   └── email_alert.py   # Gmail SMTP notifications on trades and errors
│       └── storage/
│           ├── __init__.py
│           └── trade_log.py     # SQLite: log signals, orders, portfolio state
├── backtest/
│   └── run_backtest.py          # vectorbt backtest harness using the same strategy module
├── tests/
│   └── ...                      # Unit tests per module
├── .env                         # API keys, email credentials (gitignored)
├── .env.example                 # Placeholder template (committed)
├── pyproject.toml               # uv project config, dependencies
└── .gitignore
```

## Data Flow

The pipeline runs as a single sequential flow, triggered by cron daily at 4:30 PM ET on weekdays:

```
[Cron triggers main.py]
        │
        ▼
[1. Market Calendar Check]
   Query Alpaca market calendar
   If market closed today (holiday): log and exit
        │
        ▼
[2. Data Module]
   Fetch 200+ daily bars for SPY via Alpaca API
   Returns: pandas DataFrame (date, open, high, low, close, volume)
        │
        ▼
[3. Strategy Module]
   Compute SMA_50 and SMA_200
   Detect crossover event vs previous day
   Returns: Signal enum (BUY, SELL, HOLD)
        │
        ▼
[4. Risk Module]
   Check current position via Alpaca
   If BUY: calculate shares = (portfolio_value * (1 - cash_buffer)) / close_price
   If SELL: confirm we hold a position to close
   If HOLD: no action
   Returns: OrderRequest (action, quantity) or None
        │
        ▼
[5. Execution Module]
   Submit market order to Alpaca paper account
   Wait for fill confirmation
   Returns: OrderResult (fill_price, quantity, timestamp, status)
        │
        ▼
[6. Storage Module]
   Log signal, order, and portfolio snapshot to SQLite
        │
        ▼
[7. Alerts Module]
   Trade placed → email summary
   Error at any step → email error details
   HOLD / no action → no email
   Daily run complete → email portfolio summary
```

## Strategy Logic

- **SMA_50** = rolling mean of last 50 closing prices
- **SMA_200** = rolling mean of last 200 closing prices
- **BUY signal**: SMA_50 crosses above SMA_200 (golden cross) — today `SMA_50 > SMA_200` AND yesterday `SMA_50 <= SMA_200`
- **SELL signal**: SMA_50 crosses below SMA_200 (death cross) — today `SMA_50 < SMA_200` AND yesterday `SMA_50 >= SMA_200`
- **HOLD**: no crossover occurred

**Position logic:**
- On BUY: go long SPY with full allocation minus 2% cash buffer
- On SELL: close entire position (flat, no shorting)
- Binary state: either fully in SPY or fully in cash

## Storage Schema

### `signals` table
| Column | Type | Description |
|---|---|---|
| id | INTEGER PK | Auto-increment |
| timestamp | DATETIME | When the evaluation ran |
| ticker | TEXT | "SPY" |
| close_price | REAL | Closing price used |
| sma_50 | REAL | 50-day SMA value |
| sma_200 | REAL | 200-day SMA value |
| signal | TEXT | BUY / SELL / HOLD |

### `orders` table
| Column | Type | Description |
|---|---|---|
| id | INTEGER PK | Auto-increment |
| signal_id | INTEGER FK | Links to the signal that triggered it |
| timestamp | DATETIME | Order submission time |
| ticker | TEXT | "SPY" |
| action | TEXT | BUY / SELL |
| quantity | INTEGER | Shares ordered |
| fill_price | REAL | Actual fill price (null until filled) |
| status | TEXT | SUBMITTED / FILLED / REJECTED / ERROR |

### `portfolio` table
| Column | Type | Description |
|---|---|---|
| id | INTEGER PK | Auto-increment |
| timestamp | DATETIME | Snapshot time |
| cash | REAL | Cash balance |
| position_qty | INTEGER | Shares of SPY held |
| position_value | REAL | Market value of position |
| total_value | REAL | cash + position_value |

## Alerts

**Gmail SMTP** via app password, using Python stdlib (`smtplib`, `email`).

| Event | Email? | Subject Format |
|---|---|---|
| Trade placed | Yes | `[SchroederTrader] BUY 45 SPY @ $523.10` |
| Trade filled | Yes | `[SchroederTrader] FILLED: BUY 45 SPY @ $523.15` |
| Error | Yes | `[SchroederTrader] ERROR: Data fetch failed` |
| HOLD (no action) | No | — |
| Daily run complete | Yes | `[SchroederTrader] Daily run complete - Portfolio: $10,245` |

Email content includes: action, ticker, quantity, price, portfolio value, cash balance, SMA values. Plain text format.

If the email itself fails to send, log the error to SQLite but do not block the pipeline.

## Configuration

**`.env` (gitignored) — secrets:**
```
ALPACA_API_KEY=your-key
ALPACA_SECRET_KEY=your-secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets

ALERT_EMAIL_FROM=youraddress@gmail.com
ALERT_EMAIL_TO=youraddress@gmail.com
ALERT_EMAIL_APP_PASSWORD=xxxx-xxxx-xxxx-xxxx
```

**`config.py` — non-secret parameters:**
```python
TICKER = "SPY"
SMA_SHORT_WINDOW = 50
SMA_LONG_WINDOW = 200
SLIPPAGE_ESTIMATE = 0.0005      # 0.05% for backtest cost modeling
CASH_BUFFER_PCT = 0.02          # Keep 2% cash reserve
DB_PATH = "data/trades.db"
```

Switching from paper to live trading is a single env var change (`ALPACA_BASE_URL`).

## Backtesting

vectorbt backtest harness that imports the same `sma_crossover.py` module as the live pipeline.

- **Data**: yfinance historical SPY data from 1993 to present (~30 years)
- **Metrics**: total return, Sharpe ratio, max drawdown, win rate, trade count — all net of 0.05% slippage per trade
- **Validation**: walk-forward with rolling 1-year out-of-sample windows
- **Benchmark**: buy-and-hold SPY over the same period
- **Output**: summary report (console + JSON), equity curve plot (PNG)

## Dependencies

```
alpaca-py            # Alpaca broker API
yfinance             # Historical OHLCV data
pandas               # Data manipulation
vectorbt             # Backtesting
python-dotenv        # Load .env
pytest               # Testing
```

`smtplib`, `sqlite3`, `email` are Python stdlib — no extra packages needed.

## Scheduling

```
30 16 * * 1-5  /path/to/.venv/bin/python /path/to/main.py >> /path/to/logs/cron.log 2>&1
```

- 4:30 PM ET, weekdays only (after 4:00 PM market close)
- Market calendar check at pipeline start skips holidays
- Orders execute as market orders at next day's open

## Error Handling

Each module raises typed exceptions. The orchestrator (`main.py`):
1. Catches all exceptions
2. Logs the error to SQLite (storage module)
3. Sends an error alert email (alerts module)
4. Exits with non-zero status code

No silent failures. A failed alert email does not block trade execution or logging.

## Phase 1 Gate Criteria

Before advancing to Phase 2:
1. Backtest shows positive out-of-sample Sharpe ratio across walk-forward windows
2. Paper trading runs successfully for 6+ weeks with no pipeline errors
3. All metrics (Sharpe, drawdown, trade log) are being captured correctly
4. Performance baseline is documented for Phase 2 comparison

## Future Phase Compatibility

The modular design supports Phase 2+ without rewriting:
- **Phase 2**: swap `strategy/sma_crossover.py` for `strategy/xgboost_classifier.py` — same interface
- **Phase 4**: expand `risk/risk_manager.py` with Kelly Criterion sizing
- **Phase 5**: add `data/sentiment.py` alongside `data/market_data.py`
- **Cron → long-running**: wrap `main.py` orchestrator in a scheduler loop
