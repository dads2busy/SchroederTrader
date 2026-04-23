from datetime import datetime, timezone
from unittest.mock import patch
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
@patch("schroeder_trader.main.get_portfolio_by_date")
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
    import pandas as pd
    signals = pd.read_csv(tmp_path / "signals.csv")
    assert signals.iloc[0]["signal"] == "HOLD"


@patch("schroeder_trader.main.get_portfolio_by_date")
def test_pipeline_skips_if_already_ran(mock_signal_by_date, tmp_path):
    mock_signal_by_date.return_value = {"signal": "HOLD"}
    db_path = tmp_path / "test.db"
    run_pipeline(db_path=db_path)


@patch("schroeder_trader.main.is_market_open_today")
@patch("schroeder_trader.main.get_pending_orders")
@patch("schroeder_trader.main.get_portfolio_by_date")
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


@patch("schroeder_trader.main.load_model")
@patch("schroeder_trader.main.send_daily_summary")
@patch("schroeder_trader.main.get_account")
@patch("schroeder_trader.main.get_position")
@patch("schroeder_trader.main.fetch_daily_bars")
@patch("schroeder_trader.main.is_market_open_today")
@patch("schroeder_trader.main.get_pending_orders")
@patch("schroeder_trader.main.get_portfolio_by_date")
def test_pipeline_shadow_composite_skips_when_no_model(
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
    """Composite shadow should silently skip when no model file exists."""
    mock_signal_by_date.return_value = None
    mock_pending.return_value = []
    mock_market_open.return_value = True
    mock_fetch.return_value = _mock_bars_df()
    mock_position.return_value = 0
    mock_account.return_value = {"portfolio_value": 10000.0, "cash": 10000.0}
    mock_load_model.return_value = None

    db_path = tmp_path / "test.db"
    run_pipeline(db_path=db_path)

    mock_summary.assert_called_once()

    shadow_csv = tmp_path / "shadow_signals.csv"
    if shadow_csv.exists():
        import pandas as pd
        assert len(pd.read_csv(shadow_csv)) == 0


@patch("schroeder_trader.main.load_model")
@patch("schroeder_trader.main.send_daily_summary")
@patch("schroeder_trader.main.get_account")
@patch("schroeder_trader.main.get_position")
@patch("schroeder_trader.main.fetch_daily_bars")
@patch("schroeder_trader.main.is_market_open_today")
@patch("schroeder_trader.main.get_pending_orders")
@patch("schroeder_trader.main.get_portfolio_by_date")
def test_pipeline_shadow_exception_does_not_affect_sma(
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
    """An exception in shadow Step 10 must not prevent Steps 1-9 from completing."""
    mock_signal_by_date.return_value = None
    mock_pending.return_value = []
    mock_market_open.return_value = True
    mock_fetch.return_value = _mock_bars_df()
    mock_position.return_value = 0
    mock_account.return_value = {"portfolio_value": 10000.0, "cash": 10000.0}
    mock_load_model.side_effect = RuntimeError("model explosion")

    db_path = tmp_path / "test.db"
    run_pipeline(db_path=db_path)

    # SMA pipeline should have completed
    mock_summary.assert_called_once()
    import pandas as pd
    signals = pd.read_csv(tmp_path / "signals.csv")
    assert signals.iloc[0]["signal"] == "HOLD"
