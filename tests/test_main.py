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
