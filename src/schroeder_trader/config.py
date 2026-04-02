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
FEATURES_CSV_PATH = PROJECT_ROOT / "backtest" / "data" / "features_daily.csv"

# Trading parameters
TICKER = "SPY"
SMA_SHORT_WINDOW = 50
SMA_LONG_WINDOW = 200
CASH_BUFFER_PCT = 0.02
KELLY_MULTIPLIER = 0.5           # half-Kelly default (configurable)
KELLY_WIN_LOSS_RATIO = 0.88      # derived from walk-forward backtest
XGB_THRESHOLD_LOW = 0.35         # confidence threshold for Choppy regime
TRAILING_STOP_PCT = 0.08            # 8% portfolio drawdown triggers stop
TRAILING_STOP_COOLDOWN_DAYS = 5     # trading days before re-entry allowed

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
LLM_MODEL = "claude-haiku-4-5-20251001"
