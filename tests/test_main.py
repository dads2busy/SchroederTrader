from datetime import datetime, timezone
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

from schroeder_trader.main import run_pipeline


def _mock_bars_df(n: int = 250) -> pd.DataFrame:
    """Oscillating prices around 100 — SMA50 ≈ SMA200 = HOLD signal; RSI is computable."""
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    # Small sinusoidal variation keeps SMA50 ≈ SMA200 (both ≈ 100) while
    # ensuring gains and losses exist so RSI is not NaN.
    close = 100.0 + np.sin(np.arange(n) * 2 * np.pi / 20)
    return pd.DataFrame({
        "open": close - 0.5,
        "high": close + 0.5,
        "low": close - 0.5,
        "close": close,
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
    mock_signal_by_date.return_value = None
    mock_pending.return_value = []
    mock_market_open.return_value = True
    mock_fetch.return_value = _mock_bars_df()
    mock_position.return_value = 0
    mock_account.return_value = {"portfolio_value": 10000.0, "cash": 10000.0}

    db_path = tmp_path / "test.db"
    run_pipeline(db_path=db_path)

    mock_summary.assert_called_once()
    import sqlite3
    conn = sqlite3.connect(str(db_path))
    row = conn.execute("SELECT signal FROM signals").fetchone()
    assert row[0] == "HOLD"
    conn.close()


@patch("schroeder_trader.main.get_signal_by_date")
def test_pipeline_skips_if_already_ran(mock_signal_by_date, tmp_path):
    mock_signal_by_date.return_value = {"signal": "HOLD"}
    db_path = tmp_path / "test.db"
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
