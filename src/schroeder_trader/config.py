import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "trades.db"
LOG_DIR = PROJECT_ROOT / "logs"

# Model paths
COMPOSITE_MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_spy_20d.json"
HMM_MODEL_PATH = PROJECT_ROOT / "models" / "hmm_regime.pkl"
# Shadow-only ticker models (logged but not traded) — see SHADOW_TICKERS below
XLK_MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_xlk_20d.json"
XLV_MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_xlv_20d.json"
XLE_MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_xle_20d.json"
HMM_RETRAIN_DAYS = 30
FEATURES_CSV_PATH = PROJECT_ROOT / "backtest" / "data" / "features_daily.csv"

# Trading parameters
TICKER = "SPY"
SMA_SHORT_WINDOW = 50
SMA_LONG_WINDOW = 200
CASH_BUFFER_PCT = 0.02
POSITION_SIZING = "binary"       # binary (98% in / 0% out) — regime routing handles risk
KELLY_MULTIPLIER = 2.0           # retained for shadow log analysis only
KELLY_WIN_LOSS_RATIO = 0.76      # derived from walk-forward backtest
XGB_THRESHOLD_LOW = 0.35         # confidence threshold for Choppy regime
STALE_CASH_DAYS = 7              # re-enter BULL if in cash this many days + SMA50 > SMA200
TRAILING_STOP_PCT = 0.10            # 10% portfolio drawdown triggers stop
TRAILING_STOP_COOLDOWN_DAYS = 5     # trading days before re-entry allowed

# Shadow tickers compute composite signals but do not trade. Each entry maps a
# ticker symbol to its trained XGBoost model file. Logged to shadow_signals.csv
# with the ticker column populated, alongside SPY's production rows.
SHADOW_TICKERS: dict[str, "Path"] = {
    "XLK": XLK_MODEL_PATH,
    "XLV": XLV_MODEL_PATH,
    "XLE": XLE_MODEL_PATH,
}

# Basket weights for the SECTOR SHADOW row in the daily email. Daily-rebalanced
# weighted combination of per-ticker composite strategies. 45/30/15/10 picked
# from a Pareto-frontier grid search (in-sample Sharpe 2.40) and confirmed by
# walk-forward validation (OOS Sharpe 2.34, MaxDD -8.76%, best of tested mixes).
# See backtest/optimize_basket_weights.py and backtest/walk_forward_basket.py.
SHADOW_BASKET_WEIGHTS: dict[str, float] = {
    "SPY": 0.45,
    "XLK": 0.30,
    "XLV": 0.15,
    "XLE": 0.10,
}

# Alpaca
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

# Email
ALERT_EMAIL_FROM = os.getenv("ALERT_EMAIL_FROM", "")
ALERT_EMAIL_TO = os.getenv("ALERT_EMAIL_TO", "")
ALERT_EMAIL_APP_PASSWORD = os.getenv("ALERT_EMAIL_APP_PASSWORD", "")

# LLM
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = "claude-haiku-4-5-20251001"  # daily report narrator (cheap)
CLAUDE_ORACLE_MODEL = "claude-opus-4-7"  # trading oracle comparison (premium)
OPENAI_ORACLE_MODEL = "gpt-5.4"          # trading oracle comparison (premium)
