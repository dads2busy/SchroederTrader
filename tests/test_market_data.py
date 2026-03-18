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
