# Phase 2: ML Signal Generation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an XGBoost classifier running in shadow mode alongside the existing SMA crossover bot, with walk-forward validation and a model comparison pipeline.

**Architecture:** Three new modules (feature_engineer, xgboost_classifier, shadow logging) plus two offline scripts (train_model, compare_models). The existing SMA pipeline is never modified — shadow mode is purely additive with silent fail-safe behavior.

**Tech Stack:** XGBoost, scikit-learn, SHAP (optional), pandas

**Prerequisites:** `backtest/data/spy_daily.csv` must exist (run `uv run python backtest/download_data.py` from Phase 1 if needed)

**Spec:** `docs/superpowers/specs/2026-03-18-phase2-ml-signal-generation-design.md`

---

## File Structure

```
SchroederTrader/
├── src/schroeder_trader/
│   ├── strategy/
│   │   ├── sma_crossover.py          # UNCHANGED
│   │   ├── feature_engineer.py       # NEW: FeaturePipeline class
│   │   └── xgboost_classifier.py     # NEW: train, load, predict, save
│   ├── storage/
│   │   └── trade_log.py              # MODIFIED: add shadow_signals table + 2 functions
│   └── main.py                       # MODIFIED: add shadow prediction step 10
├── backtest/
│   ├── train_model.py                # NEW: walk-forward training + final model
│   └── compare_models.py            # NEW: SMA vs ML comparison
├── models/                           # NEW directory (gitignored)
└── tests/
    ├── test_feature_engineer.py      # NEW
    ├── test_xgboost_classifier.py    # NEW
    └── test_shadow_logging.py        # NEW
```

---

### Task 1: Dependencies & Project Setup

**Files:**
- Modify: `pyproject.toml`
- Modify: `.gitignore`

- [ ] **Step 1: Add new dependencies to pyproject.toml**

Add `xgboost` and `scikit-learn` to main dependencies, `shap` to backtest extras:

```toml
dependencies = [
    "alpaca-py>=0.33.0",
    "yfinance>=0.2.40",
    "pandas>=2.2.0",
    "python-dotenv>=1.0.0",
    "xgboost>=2.0.0",
    "scikit-learn>=1.4.0",
]

[project.optional-dependencies]
backtest = [
    "vectorbt>=0.26.0",
    "kaleido>=0.2.0",
    "shap>=0.45.0",
]
```

- [ ] **Step 2: Add models/ to .gitignore**

Append to `.gitignore`:

```
# ML models
models/
```

- [ ] **Step 3: Create models directory**

```bash
mkdir -p models
```

- [ ] **Step 4: Install new dependencies**

```bash
cd ~/git/SchroederTrader
uv pip install -e ".[dev,backtest]"
```

- [ ] **Step 5: Verify existing tests still pass**

```bash
uv run pytest tests/ -v
```

Expected: 41 tests PASS

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .gitignore
git commit -m "chore: add xgboost, scikit-learn, shap dependencies for Phase 2"
```

---

### Task 2: Feature Engineering Module

**Files:**
- Create: `src/schroeder_trader/strategy/feature_engineer.py`
- Create: `tests/test_feature_engineer.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_feature_engineer.py`:

```python
import numpy as np
import pandas as pd

from schroeder_trader.strategy.feature_engineer import FeaturePipeline

FEATURE_COLUMNS = [
    "log_return_5d",
    "log_return_20d",
    "volatility_20d",
    "sma_ratio",
    "volume_ratio",
    "rsi_14",
]


def _make_ohlcv(n: int = 300) -> pd.DataFrame:
    """Create synthetic OHLCV data with a trend."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "open": close - np.random.uniform(0, 1, n),
        "high": close + np.random.uniform(0, 2, n),
        "low": close - np.random.uniform(0, 2, n),
        "close": close,
        "volume": np.random.randint(1_000_000, 10_000_000, n).astype(float),
    }, index=dates)


def test_compute_features_returns_all_columns():
    df = _make_ohlcv(300)
    pipeline = FeaturePipeline()
    result = pipeline.compute_features(df)
    for col in FEATURE_COLUMNS:
        assert col in result.columns, f"Missing feature: {col}"


def test_compute_features_drops_nan_rows():
    df = _make_ohlcv(300)
    pipeline = FeaturePipeline()
    result = pipeline.compute_features(df)
    assert not result[FEATURE_COLUMNS].isna().any().any()
    # Should lose ~200 rows to SMA_200 warmup
    assert len(result) < len(df)
    assert len(result) > 50


def test_compute_features_no_label_column():
    df = _make_ohlcv(300)
    pipeline = FeaturePipeline()
    result = pipeline.compute_features(df)
    assert "forward_return_5d_class" not in result.columns


def test_compute_features_with_labels_has_label():
    df = _make_ohlcv(300)
    pipeline = FeaturePipeline()
    result = pipeline.compute_features_with_labels(df)
    assert "forward_return_5d_class" in result.columns


def test_label_encoding_values():
    df = _make_ohlcv(300)
    pipeline = FeaturePipeline()
    result = pipeline.compute_features_with_labels(df)
    labels = result["forward_return_5d_class"].unique()
    # All labels should be 0, 1, or 2
    assert set(labels).issubset({0, 1, 2})


def test_labels_drop_trailing_rows():
    df = _make_ohlcv(300)
    pipeline = FeaturePipeline()
    features_only = pipeline.compute_features(df)
    with_labels = pipeline.compute_features_with_labels(df)
    # With-labels should have fewer rows (last 5 dropped)
    assert len(with_labels) <= len(features_only) - 5


def test_sma_ratio_positive():
    df = _make_ohlcv(300)
    pipeline = FeaturePipeline()
    result = pipeline.compute_features(df)
    assert (result["sma_ratio"] > 0).all()


def test_volume_ratio_positive():
    df = _make_ohlcv(300)
    pipeline = FeaturePipeline()
    result = pipeline.compute_features(df)
    assert (result["volume_ratio"] > 0).all()


def test_rsi_between_0_and_100():
    df = _make_ohlcv(300)
    pipeline = FeaturePipeline()
    result = pipeline.compute_features(df)
    assert (result["rsi_14"] >= 0).all()
    assert (result["rsi_14"] <= 100).all()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_feature_engineer.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement feature_engineer.py**

Create `src/schroeder_trader/strategy/feature_engineer.py`:

```python
import numpy as np
import pandas as pd

from schroeder_trader.config import SMA_SHORT_WINDOW, SMA_LONG_WINDOW

# Label encoding: XGBoost requires 0-indexed contiguous integers
CLASS_DOWN = 0
CLASS_FLAT = 1
CLASS_UP = 2

CLASS_NAMES = {CLASS_DOWN: "DOWN", CLASS_FLAT: "FLAT", CLASS_UP: "UP"}

# Threshold for classifying 5-day forward returns
RETURN_THRESHOLD = 0.005  # 0.5%


class FeaturePipeline:
    """Compute ML features from OHLCV data."""

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute features for live prediction (no label).

        Args:
            df: DataFrame with open, high, low, close, volume columns.

        Returns:
            DataFrame with feature columns added, NaN rows dropped.
        """
        result = df.copy()

        # Momentum
        result["log_return_5d"] = np.log(result["close"] / result["close"].shift(5))
        result["log_return_20d"] = np.log(result["close"] / result["close"].shift(20))

        # Risk regime
        result["volatility_20d"] = result["close"].pct_change().rolling(20).std()

        # Trend strength
        sma_short = result["close"].rolling(SMA_SHORT_WINDOW).mean()
        sma_long = result["close"].rolling(SMA_LONG_WINDOW).mean()
        result["sma_ratio"] = sma_short / sma_long

        # Activity
        result["volume_ratio"] = result["volume"] / result["volume"].rolling(20).mean()

        # Mean-reversion (RSI 14)
        result["rsi_14"] = self._compute_rsi(result["close"], 14)

        # Drop rows with NaN features
        feature_cols = [
            "log_return_5d", "log_return_20d", "volatility_20d",
            "sma_ratio", "volume_ratio", "rsi_14",
        ]
        result = result.dropna(subset=feature_cols)

        return result

    def compute_features_with_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute features + forward-looking label for training.

        Args:
            df: DataFrame with open, high, low, close, volume columns.

        Returns:
            DataFrame with features and forward_return_5d_class column.
            Trailing rows where the 5-day forward return cannot be computed are dropped.
        """
        result = self.compute_features(df)

        # Forward 5-day return (uses future data — training only!)
        forward_return = result["close"].shift(-5) / result["close"] - 1

        # Drop trailing rows where forward return is NaN (last 5 rows)
        result = result[forward_return.notna()].copy()
        forward_return = forward_return[forward_return.notna()]

        # Classify
        result["forward_return_5d_class"] = CLASS_FLAT  # default
        result.loc[forward_return > RETURN_THRESHOLD, "forward_return_5d_class"] = CLASS_UP
        result.loc[forward_return < -RETURN_THRESHOLD, "forward_return_5d_class"] = CLASS_DOWN
        result["forward_return_5d_class"] = result["forward_return_5d_class"].astype(int)

        return result

    @staticmethod
    def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Compute RSI (Relative Strength Index)."""
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_feature_engineer.py -v
```

Expected: 9 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/schroeder_trader/strategy/feature_engineer.py tests/test_feature_engineer.py
git commit -m "feat: feature engineering pipeline with 6 ML features and label encoding"
```

---

### Task 3: XGBoost Classifier Module

**Files:**
- Create: `src/schroeder_trader/strategy/xgboost_classifier.py`
- Create: `tests/test_xgboost_classifier.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_xgboost_classifier.py`:

```python
import numpy as np
import pandas as pd

from schroeder_trader.strategy.xgboost_classifier import (
    train_model,
    predict_signal,
    save_model,
    load_model,
)
from schroeder_trader.strategy.sma_crossover import Signal


def _make_training_data(n: int = 500):
    """Create synthetic training data with 3 classes."""
    np.random.seed(42)
    X = pd.DataFrame({
        "f1": np.random.randn(n),
        "f2": np.random.randn(n),
        "f3": np.random.randn(n),
    })
    # Labels correlated with f1: positive f1 → UP, negative → DOWN
    y = pd.Series(np.where(X["f1"] > 0.5, 2, np.where(X["f1"] < -0.5, 0, 1)))
    return X, y


def test_train_model_returns_classifier():
    X, y = _make_training_data()
    split = int(len(X) * 0.8)
    model = train_model(X[:split], y[:split], X[split:], y[split:])
    assert model is not None
    assert hasattr(model, "predict_proba")


def test_predict_signal_returns_signal_and_proba():
    X, y = _make_training_data()
    split = int(len(X) * 0.8)
    model = train_model(X[:split], y[:split], X[split:], y[split:])

    signal, proba = predict_signal(model, X.iloc[[0]])
    assert isinstance(signal, Signal)
    assert "DOWN" in proba
    assert "FLAT" in proba
    assert "UP" in proba
    assert abs(sum(proba.values()) - 1.0) < 0.01


def test_predict_signal_buy_on_high_up_proba():
    X, y = _make_training_data()
    split = int(len(X) * 0.8)
    model = train_model(X[:split], y[:split], X[split:], y[split:])

    # Find a row with strong UP signal (f1 >> 0.5)
    strong_up = pd.DataFrame({"f1": [3.0], "f2": [0.0], "f3": [0.0]})
    signal, proba = predict_signal(model, strong_up)
    # With clear signal, should predict BUY
    assert signal == Signal.BUY or proba["UP"] > 0.3  # model may not be perfect on synthetic


def test_predict_signal_sell_on_high_down_proba():
    X, y = _make_training_data()
    split = int(len(X) * 0.8)
    model = train_model(X[:split], y[:split], X[split:], y[split:])

    strong_down = pd.DataFrame({"f1": [-3.0], "f2": [0.0], "f3": [0.0]})
    signal, proba = predict_signal(model, strong_down)
    assert signal == Signal.SELL or proba["DOWN"] > 0.3


def test_save_and_load_model(tmp_path):
    X, y = _make_training_data()
    split = int(len(X) * 0.8)
    model = train_model(X[:split], y[:split], X[split:], y[split:])

    model_path = tmp_path / "test_model.json"
    save_model(model, model_path)
    assert model_path.exists()

    loaded = load_model(model_path)
    # Verify loaded model produces same predictions
    signal_orig, proba_orig = predict_signal(model, X.iloc[[0]])
    signal_loaded, proba_loaded = predict_signal(loaded, X.iloc[[0]])
    assert signal_orig == signal_loaded
    assert abs(proba_orig["UP"] - proba_loaded["UP"]) < 0.001


def test_load_model_nonexistent_returns_none(tmp_path):
    result = load_model(tmp_path / "nonexistent.json")
    assert result is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_xgboost_classifier.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement xgboost_classifier.py**

Create `src/schroeder_trader/strategy/xgboost_classifier.py`:

```python
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from schroeder_trader.strategy.feature_engineer import CLASS_DOWN, CLASS_FLAT, CLASS_UP, CLASS_NAMES
from schroeder_trader.strategy.sma_crossover import Signal

logger = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    "max_depth": 4,
    "n_estimators": 200,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "random_state": 42,
}


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    early_stopping_rounds: int = 20,
) -> XGBClassifier:
    """Train an XGBoost classifier with early stopping.

    Args:
        X_train: Training features.
        y_train: Training labels (0=DOWN, 1=FLAT, 2=UP).
        X_val: Validation features for early stopping.
        y_val: Validation labels.
        early_stopping_rounds: Stop if no improvement for N rounds.

    Returns:
        Trained XGBClassifier.
    """
    model = XGBClassifier(
        **DEFAULT_PARAMS,
        early_stopping_rounds=early_stopping_rounds,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    logger.info("Model trained: %d trees, best iteration %d", model.n_estimators, model.best_iteration)
    return model


def predict_signal(model: XGBClassifier, features_row: pd.DataFrame) -> tuple[Signal, dict]:
    """Generate a trading signal from model prediction.

    Args:
        model: Trained XGBClassifier.
        features_row: Single-row DataFrame of features.

    Returns:
        Tuple of (Signal, probability_dict).
        probability_dict has keys "DOWN", "FLAT", "UP".
    """
    proba = model.predict_proba(features_row)[0]
    proba_dict = {
        "DOWN": float(proba[CLASS_DOWN]),
        "FLAT": float(proba[CLASS_FLAT]),
        "UP": float(proba[CLASS_UP]),
    }

    predicted_class = int(np.argmax(proba))

    if predicted_class == CLASS_UP and proba[CLASS_UP] > 0.5:
        return Signal.BUY, proba_dict
    elif predicted_class == CLASS_DOWN and proba[CLASS_DOWN] > 0.5:
        return Signal.SELL, proba_dict
    else:
        return Signal.HOLD, proba_dict


def save_model(model: XGBClassifier, path: Path) -> None:
    """Save model to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))
    logger.info("Model saved to %s", path)


def load_model(path: Path) -> XGBClassifier | None:
    """Load model from JSON file. Returns None if file doesn't exist."""
    if not path.exists():
        logger.info("No model file at %s", path)
        return None
    model = XGBClassifier()
    model.load_model(str(path))
    logger.info("Model loaded from %s", path)
    return model
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_xgboost_classifier.py -v
```

Expected: 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/schroeder_trader/strategy/xgboost_classifier.py tests/test_xgboost_classifier.py
git commit -m "feat: XGBoost classifier module with train, predict, save/load"
```

---

### Task 4: Shadow Signal Storage

**Files:**
- Modify: `src/schroeder_trader/storage/trade_log.py`
- Create: `tests/test_shadow_logging.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_shadow_logging.py`:

```python
import json
from datetime import datetime, timezone

from schroeder_trader.storage.trade_log import (
    init_db,
    log_shadow_signal,
    get_shadow_signals,
)


def test_init_db_creates_shadow_signals_table(tmp_path):
    db_path = tmp_path / "test.db"
    conn = init_db(db_path)
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cursor.fetchall()}
    assert "shadow_signals" in tables
    conn.close()


def test_log_shadow_signal(tmp_path):
    db_path = tmp_path / "test.db"
    conn = init_db(db_path)
    now = datetime.now(timezone.utc)
    proba = json.dumps({"DOWN": 0.1, "FLAT": 0.3, "UP": 0.6})

    signal_id = log_shadow_signal(
        conn, now, "SPY", 523.10, 2, proba, "BUY", "HOLD"
    )
    assert signal_id == 1

    row = conn.execute("SELECT * FROM shadow_signals WHERE id = ?", (signal_id,)).fetchone()
    assert row["ticker"] == "SPY"
    assert row["ml_signal"] == "BUY"
    assert row["sma_signal"] == "HOLD"
    assert row["predicted_class"] == 2
    conn.close()


def test_get_shadow_signals(tmp_path):
    db_path = tmp_path / "test.db"
    conn = init_db(db_path)
    now = datetime.now(timezone.utc)
    proba = json.dumps({"DOWN": 0.1, "FLAT": 0.3, "UP": 0.6})

    log_shadow_signal(conn, now, "SPY", 523.10, 2, proba, "BUY", "HOLD")
    log_shadow_signal(conn, now, "SPY", 524.00, 0, json.dumps({"DOWN": 0.7, "FLAT": 0.2, "UP": 0.1}), "SELL", "HOLD")

    signals = get_shadow_signals(conn)
    assert len(signals) == 2
    assert signals[0]["ml_signal"] == "BUY"
    assert signals[1]["ml_signal"] == "SELL"
    conn.close()


def test_existing_tables_unchanged(tmp_path):
    """Verify Phase 1 tables still work after shadow_signals addition."""
    db_path = tmp_path / "test.db"
    conn = init_db(db_path)
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cursor.fetchall()}
    assert {"signals", "orders", "portfolio", "shadow_signals"} <= tables
    conn.close()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_shadow_logging.py -v
```

Expected: FAIL — `ImportError: cannot import name 'log_shadow_signal'`

- [ ] **Step 3: Add shadow_signals table and functions to trade_log.py**

Add to `init_db()` in `src/schroeder_trader/storage/trade_log.py` (after the portfolio table creation):

```python
    conn.execute("""
        CREATE TABLE IF NOT EXISTS shadow_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            ticker TEXT NOT NULL,
            close_price REAL NOT NULL,
            predicted_class INTEGER NOT NULL,
            predicted_proba TEXT NOT NULL,
            ml_signal TEXT NOT NULL,
            sma_signal TEXT NOT NULL
        )
    """)
```

Add two new functions at the end of the file:

```python
def log_shadow_signal(
    conn: sqlite3.Connection,
    timestamp: datetime,
    ticker: str,
    close_price: float,
    predicted_class: int,
    predicted_proba: str,
    ml_signal: str,
    sma_signal: str,
) -> int:
    cursor = conn.execute(
        "INSERT INTO shadow_signals (timestamp, ticker, close_price, predicted_class, predicted_proba, ml_signal, sma_signal) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (timestamp.isoformat(), ticker, close_price, predicted_class, predicted_proba, ml_signal, sma_signal),
    )
    conn.commit()
    return cursor.lastrowid


def get_shadow_signals(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute("SELECT * FROM shadow_signals ORDER BY id").fetchall()
    return [dict(row) for row in rows]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_shadow_logging.py -v
```

Expected: 4 tests PASS

- [ ] **Step 5: Verify existing tests still pass**

```bash
uv run pytest tests/ -v
```

Expected: 41 + 4 = 45+ tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/schroeder_trader/storage/trade_log.py tests/test_shadow_logging.py
git commit -m "feat: shadow_signals table for ML prediction logging"
```

---

### Task 5: Shadow Mode in Orchestrator

**Files:**
- Modify: `src/schroeder_trader/main.py`
- Modify: `tests/test_main.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_main.py`:

```python
@patch("schroeder_trader.main.send_daily_summary")
@patch("schroeder_trader.main.get_account")
@patch("schroeder_trader.main.get_position")
@patch("schroeder_trader.main.fetch_daily_bars")
@patch("schroeder_trader.main.is_market_open_today")
@patch("schroeder_trader.main.get_pending_orders")
@patch("schroeder_trader.main.get_signal_by_date")
def test_pipeline_shadow_mode_skips_when_no_model(
    mock_signal_by_date,
    mock_pending,
    mock_market_open,
    mock_fetch,
    mock_position,
    mock_account,
    mock_summary,
    tmp_path,
):
    """Shadow mode should silently skip when no model file exists."""
    mock_signal_by_date.return_value = None
    mock_pending.return_value = []
    mock_market_open.return_value = True
    mock_fetch.return_value = _mock_bars_df()
    mock_position.return_value = 0
    mock_account.return_value = {"portfolio_value": 10000.0, "cash": 10000.0}

    db_path = tmp_path / "test.db"
    run_pipeline(db_path=db_path)

    # Pipeline should complete without error
    mock_summary.assert_called_once()

    # No shadow signals should be logged (no model file)
    import sqlite3
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    shadow_count = conn.execute("SELECT COUNT(*) FROM shadow_signals").fetchone()[0]
    assert shadow_count == 0
    conn.close()
```

Also add a test for the happy path (model exists):

```python
@patch("schroeder_trader.main.load_model")
@patch("schroeder_trader.main.send_daily_summary")
@patch("schroeder_trader.main.get_account")
@patch("schroeder_trader.main.get_position")
@patch("schroeder_trader.main.fetch_daily_bars")
@patch("schroeder_trader.main.is_market_open_today")
@patch("schroeder_trader.main.get_pending_orders")
@patch("schroeder_trader.main.get_signal_by_date")
def test_pipeline_shadow_mode_logs_when_model_exists(
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
    """Shadow mode should log a prediction when a model is available."""
    mock_signal_by_date.return_value = None
    mock_pending.return_value = []
    mock_market_open.return_value = True
    mock_fetch.return_value = _mock_bars_df(300)
    mock_position.return_value = 0
    mock_account.return_value = {"portfolio_value": 10000.0, "cash": 10000.0}

    # Create a mock model that returns a HOLD prediction
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.2, 0.6, 0.2]])
    mock_load_model.return_value = mock_model

    db_path = tmp_path / "test.db"
    run_pipeline(db_path=db_path)

    import sqlite3
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    shadow_count = conn.execute("SELECT COUNT(*) FROM shadow_signals").fetchone()[0]
    assert shadow_count == 1
    shadow = conn.execute("SELECT * FROM shadow_signals").fetchone()
    assert shadow["ml_signal"] == "HOLD"
    conn.close()
```

Note: This test needs `from unittest.mock import MagicMock` and `import numpy as np` added to the test file imports if not already present.

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_main.py -v
```

Expected: FAIL (shadow_signals table may not exist yet in test, or import errors)

- [ ] **Step 3: Add shadow mode step to main.py**

Add these imports to the top of `src/schroeder_trader/main.py`:

```python
import json
from schroeder_trader.strategy.feature_engineer import FeaturePipeline
from schroeder_trader.strategy.xgboost_classifier import load_model, predict_signal
from schroeder_trader.storage.trade_log import log_shadow_signal
```

Add a new constant near the top:

```python
MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_spy.json"
```

(You'll need to import `PROJECT_ROOT` from config if not already imported.)

Add Step 10 at the end of `run_pipeline()`, just before `conn.close()`:

```python
    # Step 10: Shadow ML prediction
    try:
        model = load_model(MODEL_PATH)
        if model is not None:
            shadow_df = fetch_daily_bars(TICKER, days=400)
            pipeline = FeaturePipeline()
            features = pipeline.compute_features(shadow_df)
            if len(features) > 0:
                last_row = features.iloc[[-1]]
                feature_cols = [
                    "log_return_5d", "log_return_20d", "volatility_20d",
                    "sma_ratio", "volume_ratio", "rsi_14",
                ]
                ml_signal, proba = predict_signal(model, last_row[feature_cols])
                class_map = {"SELL": 0, "HOLD": 1, "BUY": 2}
                pred_class = class_map.get(ml_signal.value, 1)
                log_shadow_signal(
                    conn, now, TICKER, close_price,
                    predicted_class=pred_class,
                    predicted_proba=json.dumps(proba),
                    ml_signal=ml_signal.value,
                    sma_signal=signal.value,
                )
                logger.info("Shadow ML signal: %s (UP=%.2f, FLAT=%.2f, DOWN=%.2f)",
                           ml_signal.value, proba["UP"], proba["FLAT"], proba["DOWN"])
    except Exception:
        logger.exception("Shadow ML prediction failed (non-fatal)")

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_main.py -v
```

Expected: 5 tests PASS (3 existing + 2 new)

- [ ] **Step 5: Run full test suite**

```bash
uv run pytest tests/ -v
```

Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/schroeder_trader/main.py tests/test_main.py
git commit -m "feat: add shadow ML prediction step to orchestrator pipeline"
```

---

### Task 6: Walk-Forward Training Script

**Files:**
- Create: `backtest/train_model.py`

- [ ] **Step 1: Create training script**

Create `backtest/train_model.py`:

```python
"""Walk-forward training and evaluation of XGBoost classifier on SPY data."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

from schroeder_trader.config import SMA_SHORT_WINDOW, SMA_LONG_WINDOW, SLIPPAGE_ESTIMATE
from schroeder_trader.strategy.feature_engineer import (
    FeaturePipeline,
    CLASS_DOWN,
    CLASS_FLAT,
    CLASS_UP,
    CLASS_NAMES,
)
from schroeder_trader.strategy.xgboost_classifier import train_model, predict_signal, save_model

DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "results"
MODEL_DIR = Path(__file__).parent.parent / "models"

FEATURE_COLUMNS = [
    "log_return_5d", "log_return_20d", "volatility_20d",
    "sma_ratio", "volume_ratio", "rsi_14",
]

# Walk-forward parameters
TRAIN_YEARS = 2
TEST_MONTHS = 6


def load_data() -> pd.DataFrame:
    csv_path = DATA_DIR / "spy_daily.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No cached data at {csv_path}. Run download_data.py first.")
    df = pd.read_csv(csv_path, skiprows=3, names=["Date", "Close", "High", "Low", "Open", "Volume"])
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    df.columns = [c.lower() for c in df.columns]
    return df


def walk_forward_evaluate() -> dict:
    """Run walk-forward validation and return aggregate metrics."""
    df = load_data()
    pipeline = FeaturePipeline()
    features_df = pipeline.compute_features_with_labels(df)

    all_predictions = []
    all_actuals = []
    all_dates = []
    window_results = []

    # Walk-forward loop
    start_date = features_df.index.min()
    end_date = features_df.index.max()

    train_start = start_date
    while True:
        train_end = train_start + pd.DateOffset(years=TRAIN_YEARS)
        test_end = train_end + pd.DateOffset(months=TEST_MONTHS)

        if train_end >= end_date:
            break

        train_data = features_df[(features_df.index >= train_start) & (features_df.index < train_end)]
        test_data = features_df[(features_df.index >= train_end) & (features_df.index < test_end)]

        if len(train_data) < 100 or len(test_data) < 10:
            train_start = train_start + pd.DateOffset(months=TEST_MONTHS)
            continue

        X_train = train_data[FEATURE_COLUMNS]
        y_train = train_data["forward_return_5d_class"]

        X_test = test_data[FEATURE_COLUMNS]
        y_test = test_data["forward_return_5d_class"]

        # Split last 20% of train for early stopping validation
        val_split = int(len(X_train) * 0.8)
        X_tr, X_val = X_train[:val_split], X_train[val_split:]
        y_tr, y_val = y_train[:val_split], y_train[val_split:]

        model = train_model(X_tr, y_tr, X_val, y_val)

        # Predict on test set
        preds = model.predict(X_test)
        accuracy = accuracy_score(y_test, preds)

        all_predictions.extend(preds)
        all_actuals.extend(y_test.values)
        all_dates.extend(test_data.index)

        window_results.append({
            "train_start": str(train_start.date()),
            "train_end": str(train_end.date()),
            "test_end": str(test_end.date()),
            "accuracy": round(float(accuracy), 4),
            "n_train": len(X_train),
            "n_test": len(X_test),
        })

        print(f"Window {len(window_results)}: train {train_start.date()}-{train_end.date()}, "
              f"test {train_end.date()}-{test_end.date()}, accuracy={accuracy:.4f}")

        train_start = train_start + pd.DateOffset(months=TEST_MONTHS)

    # Aggregate metrics
    all_predictions = np.array(all_predictions)
    all_actuals = np.array(all_actuals)
    all_dates = pd.DatetimeIndex(all_dates)

    overall_accuracy = accuracy_score(all_actuals, all_predictions)

    # Simulate returns for Sharpe calculation
    # BUY (pred=2) → long, SELL (pred=0) → flat, HOLD (pred=1) → maintain
    close_prices = features_df.loc[all_dates, "close"]
    daily_returns = close_prices.pct_change().fillna(0)

    position = 0  # 0 = flat, 1 = long
    strategy_returns = []
    for i, (pred, ret) in enumerate(zip(all_predictions, daily_returns)):
        if pred == CLASS_UP:
            position = 1
        elif pred == CLASS_DOWN:
            position = 0
        # FLAT keeps current position

        strategy_return = position * ret
        # Deduct transaction cost on position changes
        if i > 0 and ((pred == CLASS_UP and all_predictions[i-1] != CLASS_UP) or
                       (pred == CLASS_DOWN and all_predictions[i-1] != CLASS_DOWN)):
            strategy_return -= SLIPPAGE_ESTIMATE

        strategy_returns.append(strategy_return)

    strategy_returns = np.array(strategy_returns)
    sharpe = np.sqrt(252) * strategy_returns.mean() / (strategy_returns.std() + 1e-10)
    total_return = (1 + strategy_returns).prod() - 1

    # Max drawdown
    cumulative = (1 + strategy_returns).cumprod()
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = float(abs(drawdown.min())) * 100

    # Trade count and win rate
    position_changes = np.diff(np.where(all_predictions == CLASS_UP, 1, 0))
    total_trades = int(np.sum(np.abs(position_changes)))

    # Win rate: fraction of trades with positive return
    wins = sum(1 for r in strategy_returns if r > 0)
    trading_days = sum(1 for r in strategy_returns if r != 0)
    win_rate = wins / trading_days if trading_days > 0 else 0.0

    # Post-2020 Sharpe
    post_2020_mask = all_dates >= "2020-01-01"
    if post_2020_mask.any():
        post_returns = strategy_returns[post_2020_mask]
        post_sharpe = float(np.sqrt(252) * post_returns.mean() / (post_returns.std() + 1e-10))
    else:
        post_sharpe = 0.0

    results = {
        "full_period": {
            "total_return_pct": round(float(total_return * 100), 2),
            "sharpe_ratio": round(float(sharpe), 4),
            "max_drawdown_pct": round(max_drawdown, 2),
            "total_trades": total_trades,
            "win_rate_pct": round(float(win_rate * 100), 2),
            "accuracy": round(float(overall_accuracy), 4),
        },
        "out_of_sample_post_2020": {
            "sharpe_ratio": round(post_sharpe, 4),
        },
        "windows": window_results,
    }

    # Print results
    print("\n" + "=" * 50)
    print("ML WALK-FORWARD RESULTS")
    print("=" * 50)
    print(f"\nOverall Accuracy: {overall_accuracy:.4f}")
    print(f"Total Return: {total_return * 100:.2f}%")
    print(f"Sharpe Ratio: {sharpe:.4f}")
    print(f"Max Drawdown: {max_drawdown:.2f}%")
    print(f"Total Trades: {total_trades}")
    print(f"Win Rate: {win_rate * 100:.2f}%")
    print(f"Post-2020 Sharpe: {post_sharpe:.4f}")
    print(f"Windows evaluated: {len(window_results)}")
    print("=" * 50)

    print("\n" + classification_report(
        all_actuals, all_predictions,
        target_names=["DOWN", "FLAT", "UP"],
    ))

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "ml_walkforward_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {OUTPUT_DIR / 'ml_walkforward_results.json'}")

    # SHAP feature importance (optional — requires shap package)
    try:
        import shap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Train a model on recent data for SHAP analysis
        recent = features_df.tail(1000)
        X_shap = recent[FEATURE_COLUMNS]
        y_shap = recent["forward_return_5d_class"]
        split = int(len(X_shap) * 0.8)
        shap_model = train_model(X_shap[:split], y_shap[:split], X_shap[split:], y_shap[split:])

        explainer = shap.TreeExplainer(shap_model)
        shap_values = explainer.shap_values(X_shap[:200])
        shap.summary_plot(shap_values, X_shap[:200], show=False)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "shap_importance.png", dpi=150)
        plt.close()
        print(f"SHAP importance plot saved to {OUTPUT_DIR / 'shap_importance.png'}")
    except ImportError:
        print("SHAP not installed — skipping feature importance plot (install with: uv pip install shap)")
    except Exception as e:
        print(f"SHAP analysis failed: {e}")

    return results


def train_final_model() -> None:
    """Train final model on all available data and save for shadow mode."""
    df = load_data()
    pipeline = FeaturePipeline()
    features_df = pipeline.compute_features_with_labels(df)

    X = features_df[FEATURE_COLUMNS]
    y = features_df["forward_return_5d_class"]

    # Use last 20% as validation for early stopping
    split = int(len(X) * 0.8)
    model = train_model(X[:split], y[:split], X[split:], y[split:])

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "xgboost_spy.json"
    save_model(model, model_path)
    print(f"\nFinal model saved to {model_path}")


if __name__ == "__main__":
    results = walk_forward_evaluate()
    print("\nTraining final model for shadow deployment...")
    train_final_model()
```

- [ ] **Step 2: Run the training script**

```bash
cd ~/git/SchroederTrader
uv run python backtest/train_model.py
```

Expected: Prints walk-forward results across ~48 windows, saves results JSON, trains and saves final model to `models/xgboost_spy.json`.

- [ ] **Step 3: Commit**

```bash
git add backtest/train_model.py
git commit -m "feat: walk-forward training script with XGBoost evaluation and final model export"
```

---

### Task 7: Model Comparison Script

**Files:**
- Create: `backtest/compare_models.py`

- [ ] **Step 1: Create comparison script**

Create `backtest/compare_models.py`:

```python
"""Compare SMA crossover signals vs ML shadow signals."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from schroeder_trader.config import DB_PATH, SLIPPAGE_ESTIMATE
from schroeder_trader.storage.trade_log import init_db, get_shadow_signals

OUTPUT_DIR = Path(__file__).parent / "results"


def compare() -> None:
    conn = init_db(DB_PATH)

    # Get shadow signals
    shadow = get_shadow_signals(conn)
    if not shadow:
        print("No shadow signals found. Run the pipeline with a trained model first.")
        conn.close()
        return

    # Get SMA signals for the same dates
    sma_rows = conn.execute("SELECT * FROM signals ORDER BY id").fetchall()
    sma_signals = [dict(row) for row in sma_rows]
    conn.close()

    if not sma_signals:
        print("No SMA signals found.")
        return

    print(f"Shadow signals: {len(shadow)}")
    print(f"SMA signals: {len(sma_signals)}")

    # Build daily signal series
    shadow_df = pd.DataFrame(shadow)
    shadow_df["date"] = pd.to_datetime(shadow_df["timestamp"]).dt.date

    sma_df = pd.DataFrame(sma_signals)
    sma_df["date"] = pd.to_datetime(sma_df["timestamp"]).dt.date

    # Merge on date
    merged = pd.merge(shadow_df, sma_df, on="date", suffixes=("_ml", "_sma"))

    if merged.empty:
        print("No overlapping dates between SMA and ML signals.")
        return

    print(f"Overlapping days: {len(merged)}")

    # Signal agreement
    agreement = (merged["ml_signal"] == merged["signal_sma"]).mean()
    print(f"\nSignal agreement: {agreement:.1%}")

    # Count signal distributions
    print("\nML signal distribution:")
    print(merged["ml_signal"].value_counts().to_string())
    print("\nSMA signal distribution:")
    print(merged["signal_sma"].value_counts().to_string())

    # Simulate returns for both
    close_prices = merged["close_price_ml"].values
    daily_returns = np.diff(close_prices) / close_prices[:-1]

    for name, signal_col in [("SMA", "signal_sma"), ("ML", "ml_signal")]:
        signals = merged[signal_col].values[:-1]  # align with returns
        position = 0
        strat_returns = []
        prev_position = 0

        for sig, ret in zip(signals, daily_returns):
            if sig == "BUY":
                position = 1
            elif sig == "SELL":
                position = 0

            strat_ret = position * ret
            if position != prev_position:
                strat_ret -= SLIPPAGE_ESTIMATE
            prev_position = position
            strat_returns.append(strat_ret)

        strat_returns = np.array(strat_returns)
        if len(strat_returns) > 0 and strat_returns.std() > 0:
            sharpe = np.sqrt(252) * strat_returns.mean() / strat_returns.std()
        else:
            sharpe = 0.0

        total_ret = (1 + strat_returns).prod() - 1
        cumulative = (1 + strat_returns).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_dd = float(abs(drawdown.min())) * 100 if len(drawdown) > 0 else 0.0

        print(f"\n{name} Strategy:")
        print(f"  Sharpe Ratio: {sharpe:.4f}")
        print(f"  Total Return: {total_ret * 100:.2f}%")
        print(f"  Max Drawdown: {max_dd:.2f}%")

    # Save comparison results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    comparison = {"sma": {}, "ml": {}}
    for name, signal_col in [("sma", "signal_sma"), ("ml", "ml_signal")]:
        signals = merged[signal_col].values[:-1]
        position = 0
        strat_returns = []
        prev_position = 0
        for sig, ret in zip(signals, daily_returns):
            if sig == "BUY":
                position = 1
            elif sig == "SELL":
                position = 0
            strat_ret = position * ret
            if position != prev_position:
                strat_ret -= SLIPPAGE_ESTIMATE
            prev_position = position
            strat_returns.append(strat_ret)
        sr = np.array(strat_returns)
        comparison[name] = {
            "sharpe_ratio": round(float(np.sqrt(252) * sr.mean() / (sr.std() + 1e-10)), 4),
            "total_return_pct": round(float(((1 + sr).prod() - 1) * 100), 2),
        }

    with open(OUTPUT_DIR / "ml_vs_sma_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nComparison saved to {OUTPUT_DIR / 'ml_vs_sma_comparison.json'}")

    # Promotion recommendation
    if comparison["ml"]["sharpe_ratio"] > comparison["sma"]["sharpe_ratio"]:
        print("\n*** Ready to promote — ML outperforms SMA ***")
    else:
        print("\n*** Not ready — SMA still outperforms ML ***")

    print("\n" + "=" * 50)
    print("Note: This comparison is only meaningful with several weeks of shadow data.")
    print("=" * 50)


if __name__ == "__main__":
    compare()
```

- [ ] **Step 2: Commit**

```bash
git add backtest/compare_models.py
git commit -m "feat: SMA vs ML model comparison script for promotion decisions"
```

---

### Task 8: Final Integration & Push

- [ ] **Step 1: Run full test suite**

```bash
cd ~/git/SchroederTrader
uv run pytest tests/ -v
```

Expected: All tests PASS (41 Phase 1 + 9 feature + 6 xgboost + 4 shadow + 2 shadow-main = ~62 tests)

- [ ] **Step 2: Verify shadow mode works end-to-end**

```bash
cd ~/git/SchroederTrader
uv run python -c "
from schroeder_trader.strategy.xgboost_classifier import load_model
from pathlib import Path
model = load_model(Path('models/xgboost_spy.json'))
print(f'Model loaded: {model is not None}')
"
```

Expected: `Model loaded: True` (if train_model.py was run in Task 6)

- [ ] **Step 3: Run the pipeline manually to test shadow mode**

```bash
uv run python src/schroeder_trader/main.py
```

Expected: Pipeline completes with shadow ML prediction logged. Check logs for "Shadow ML signal:" line.

- [ ] **Step 4: Push to GitHub**

```bash
git push
```

- [ ] **Step 5: Verify shadow signal was logged**

```bash
uv run python -c "
from schroeder_trader.config import DB_PATH
from schroeder_trader.storage.trade_log import init_db, get_shadow_signals
conn = init_db(DB_PATH)
shadows = get_shadow_signals(conn)
for s in shadows:
    print(f'{s[\"timestamp\"]}: ML={s[\"ml_signal\"]}, SMA={s[\"sma_signal\"]}, proba={s[\"predicted_proba\"]}')
conn.close()
"
```

Expected: At least one shadow signal entry showing ML prediction alongside SMA signal.
