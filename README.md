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
