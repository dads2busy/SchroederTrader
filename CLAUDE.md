# CLAUDE.md

## Project Overview

AI-driven SPY trading system using XGBoost regime classification with composite signal routing. Runs daily via cron, trades on Alpaca (paper), sends email summaries.

## Commands

- **Run tests:** `uv run pytest` (175 tests, ~21s)
- **Run single test:** `uv run pytest tests/test_composite.py -k test_name`
- **Run pipeline:** `uv run python -m schroeder_trader.main`
- **Install deps:** `uv sync` (use `uv sync --extra backtest --extra dev` for all)

## Architecture

- `src/schroeder_trader/` — main package
  - `main.py` — daily pipeline entry point (cron-driven)
  - `config.py` — all tunables (thresholds, sizing, stops) + env vars
  - `strategy/composite.py` — regime routing: BULL/BEAR/CHOPPY signal logic
  - `strategy/regime_detector.py` — threshold-based regime detection (HMM tested and rejected)
  - `strategy/xgboost_classifier.py` — XGBoost model for 20-day forward returns
  - `strategy/feature_engineer.py` — feature pipeline for model inputs
  - `risk/` — Kelly criterion (shadow only), trailing stop, transaction costs
  - `execution/broker.py` — Alpaca API wrapper
  - `agents/daily_report.py` — LLM-generated daily intelligence report
  - `alerts/email_alert.py` — daily summary emails
- `backtest/` — backtesting scripts and data
- `models/` — serialized model files (not in git)
- `data/` — SQLite trade log, feature CSVs

## Composite Signal Routing (CRITICAL)

The system uses **regime-based signal routing** — different signal sources control trading depending on market regime. This is the core trading logic and must be preserved exactly.

### Regime Detection (`strategy/regime_detector.py`)
- **BULL:** 20-day log return > 0 AND volatility <= 252-day rolling median
- **BEAR:** 20-day log return < 0 AND volatility > 252-day rolling median
- **CHOPPY:** everything else

### Signal Routing (`strategy/composite.py → composite_signal_hybrid()`)

| Regime | Signal Source | Source Label | Logic |
|--------|-------------|-------------|-------|
| **BULL** | SMA crossover | `"SMA"` | Trust trend-following when trend is up + vol is normal |
| **BEAR** | Flat (forced SELL) | `"FLAT"` | Stay defensive, go to cash |
| **BEAR + weakening** | XGBoost (low threshold) | `"XGB_BEAR_WEAK"` | Tactical re-entry when 5-day return turns positive |
| **CHOPPY** | XGBoost (low threshold) | `"XGB"` | ML handles sideways markets better than momentum |

### Bear Weakening
Detected when `regime == BEAR` AND `5-day log return > 0`. Switches BEAR from defensive flat to tactical XGBoost routing.

### Overrides (applied after routing)
- **Stale cash override:** Force BUY after 7+ days in cash if SMA50 > SMA200 (source: `"STALE_CASH"`)
- **Trailing stop:** 10% portfolio drawdown triggers stop, 5-day cooldown before re-entry

### Key Principle
The SMA signal is NOT replaced by XGBoost everywhere — it is the primary signal in BULL regime. XGBoost only controls in CHOPPY and BEAR+weakening. In plain BEAR, nothing trades (flat). This regime routing is what produces Sharpe 2.53.

## Key Design Decisions

- **Binary position sizing** (98% in or 0% out) — outperformed Kelly fractional sizing
- **Threshold-based regime detection** — simpler and higher Sharpe (2.53) than HMM (1.02)
- **Stale cash override** — re-enters BULL after 7 days if SMA50 > SMA200
- **10% trailing stop** with 5-day cooldown

## Environment Variables

Stored in `.env` (not committed): `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, `ALPACA_BASE_URL`, `ALERT_EMAIL_FROM`, `ALERT_EMAIL_TO`, `ALERT_EMAIL_APP_PASSWORD`, `ANTHROPIC_API_KEY`

## Style

- Python 3.12+, managed with `uv`
- Conventional commit messages: `feat:`, `fix:`, `chore:`, `docs:`
- Tests mirror source structure: `test_<module>.py`
