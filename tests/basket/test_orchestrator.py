from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from schroeder_trader.basket.orchestrator import compute_decisions
from schroeder_trader.storage.csv_store import CsvStore


def _make_store(tmp_path: Path) -> CsvStore:
    # Initialize the expected CSVs as empty so reads don't fail
    pd.DataFrame(columns=[
        "id", "timestamp", "pipeline", "ticker",
        "cash", "position_qty", "position_value", "total_value",
    ]).to_csv(tmp_path / "portfolio.csv", index=False)
    pd.DataFrame(columns=[
        "id", "timestamp", "pipeline", "ticker", "close_price",
        "predicted_class", "predicted_proba", "ml_signal", "sma_signal",
        "regime", "signal_source", "bear_day_count",
        "kelly_fraction", "kelly_qty", "high_water_mark",
        "trailing_stop_triggered",
    ]).to_csv(tmp_path / "shadow_signals.csv", index=False)
    return CsvStore(tmp_path)


def _fake_signal_pipeline(signal_value: str, source: str, regime: str = "BULL"):
    """Mimics the per-ticker composite-signal computation return tuple."""
    from schroeder_trader.strategy.composite import Signal
    from schroeder_trader.strategy.regime_detector import Regime

    sig_enum = Signal[signal_value]
    regime_enum = Regime[regime]
    return (sig_enum, source, regime_enum, 0,
            {"close": 100.0, "pred_class": 2, "proba_json": "{}", "sma_signal": "BUY"})


@patch("schroeder_trader.basket.orchestrator._compute_signal_for_ticker")
def test_compute_decisions_buy_signal_produces_full_exposure(
    mock_sig, tmp_path,
):
    mock_sig.return_value = _fake_signal_pipeline("BUY", "SMA")
    store = _make_store(tmp_path)
    weights = {"SPY": 0.5, "XLK": 0.5}
    ext_df = pd.DataFrame()
    now = datetime(2026, 5, 21, 20, 30, tzinfo=timezone.utc)

    decisions = compute_decisions(store, weights, ext_df, now, portfolio_value=100000.0)

    assert decisions["SPY"]["exposure"] == 1.0
    assert decisions["XLK"]["exposure"] == 1.0
    assert decisions["SPY"]["signal"] == "BUY"


@patch("schroeder_trader.basket.orchestrator._compute_signal_for_ticker")
def test_compute_decisions_sell_signal_produces_flat_exposure(
    mock_sig, tmp_path,
):
    mock_sig.return_value = _fake_signal_pipeline("SELL", "FLAT", regime="BEAR")
    store = _make_store(tmp_path)
    weights = {"SPY": 1.0}
    now = datetime(2026, 5, 21, 20, 30, tzinfo=timezone.utc)

    decisions = compute_decisions(store, weights, pd.DataFrame(), now, portfolio_value=100000.0)

    assert decisions["SPY"]["exposure"] == 0.0


@patch("schroeder_trader.basket.orchestrator._compute_signal_for_ticker")
def test_compute_decisions_hold_carries_prior_basket_exposure(
    mock_sig, tmp_path,
):
    # Seed prior basket BUY for SPY
    ss = pd.DataFrame({
        "id": [1], "timestamp": ["2026-05-20T20:30:00+00:00"],
        "pipeline": ["basket"], "ticker": ["SPY"],
        "close_price": [100.0],
        "predicted_class": [None], "predicted_proba": [None],
        "ml_signal": ["BUY"], "sma_signal": ["BUY"],
        "regime": ["BULL"], "signal_source": ["SMA"],
        "bear_day_count": [None], "kelly_fraction": [None],
        "kelly_qty": [None], "high_water_mark": [None],
        "trailing_stop_triggered": [None],
    })
    pd.DataFrame(columns=[
        "id", "timestamp", "pipeline", "ticker",
        "cash", "position_qty", "position_value", "total_value",
    ]).to_csv(tmp_path / "portfolio.csv", index=False)
    ss.to_csv(tmp_path / "shadow_signals.csv", index=False)
    store = CsvStore(tmp_path)

    mock_sig.return_value = _fake_signal_pipeline("HOLD", "SMA")
    weights = {"SPY": 1.0}
    now = datetime(2026, 5, 21, 20, 30, tzinfo=timezone.utc)

    decisions = compute_decisions(store, weights, pd.DataFrame(), now, portfolio_value=100000.0)
    assert decisions["SPY"]["exposure"] == 1.0  # HOLD carries prior BUY


@patch("schroeder_trader.basket.orchestrator._compute_signal_for_ticker")
def test_compute_decisions_triggers_trailing_stop_and_forces_flat(
    mock_sig, tmp_path,
):
    # Seed prior basket state with a high HWM that today's portfolio_value will
    # drop more than 10% below (the trailing-stop threshold).
    pf = pd.DataFrame({
        "id": [1], "timestamp": ["2026-05-20T20:30:00+00:00"],
        "pipeline": ["basket"], "ticker": ["XLK"],
        "cash": [500.0], "position_qty": [100],
        "position_value": [10000.0], "total_value": [10500.0],
    })
    ss = pd.DataFrame({
        "id": [1], "timestamp": ["2026-05-20T20:30:00+00:00"],
        "pipeline": ["basket"], "ticker": ["XLK"],
        "close_price": [100.0],
        "predicted_class": [None], "predicted_proba": [None],
        "ml_signal": ["BUY"], "sma_signal": ["BUY"],
        "regime": ["BULL"], "signal_source": ["SMA"],
        "bear_day_count": [None], "kelly_fraction": [None],
        "kelly_qty": [None], "high_water_mark": [12000.0],
        "trailing_stop_triggered": [0],
    })
    pf.to_csv(tmp_path / "portfolio.csv", index=False)
    ss.to_csv(tmp_path / "shadow_signals.csv", index=False)
    store = CsvStore(tmp_path)

    # Signal would be BUY, but HWM 12000 and current 10500 = -12.5% > 10% stop
    mock_sig.return_value = _fake_signal_pipeline("BUY", "SMA")
    now = datetime(2026, 5, 21, 20, 30, tzinfo=timezone.utc)

    decisions = compute_decisions(
        store, {"XLK": 1.0}, pd.DataFrame(), now,
        portfolio_value=10500.0,
    )
    assert decisions["XLK"]["exposure"] == 0.0  # stop overrides BUY
    assert decisions["XLK"]["stop_state"]["triggered_today_or_cooldown"] is True


@patch("schroeder_trader.basket.orchestrator._compute_signal_for_ticker")
def test_compute_decisions_respects_cooldown_after_prior_stop(
    mock_sig, tmp_path,
):
    """A stop fired yesterday should keep exposure flat today even if the
    portfolio value has recovered — the 5-day cooldown forces flat."""
    # Seed a prior basket row from yesterday where the stop fired.
    ss = pd.DataFrame({
        "id": [1], "timestamp": ["2026-05-20T20:30:00+00:00"],
        "pipeline": ["basket"], "ticker": ["XLK"],
        "close_price": [100.0],
        "predicted_class": [None], "predicted_proba": [None],
        "ml_signal": ["SELL"], "sma_signal": ["SELL"],
        "regime": ["BEAR"], "signal_source": ["FLAT"],
        "bear_day_count": [None], "kelly_fraction": [None],
        "kelly_qty": [None], "high_water_mark": [12000.0],
        "trailing_stop_triggered": [1],  # KEY: stop fired yesterday
    })
    pd.DataFrame(columns=[
        "id", "timestamp", "pipeline", "ticker",
        "cash", "position_qty", "position_value", "total_value",
    ]).to_csv(tmp_path / "portfolio.csv", index=False)
    ss.to_csv(tmp_path / "shadow_signals.csv", index=False)
    store = CsvStore(tmp_path)

    # Today's signal would be BUY, portfolio has fully recovered.
    # But yesterday's stop is still within the 5-day cooldown window.
    mock_sig.return_value = _fake_signal_pipeline("BUY", "SMA")
    now = datetime(2026, 5, 21, 20, 30, tzinfo=timezone.utc)

    decisions = compute_decisions(
        store, {"XLK": 1.0}, pd.DataFrame(), now,
        portfolio_value=15000.0,  # recovered above HWM
    )
    assert decisions["XLK"]["exposure"] == 0.0  # still in cooldown
    assert decisions["XLK"]["stop_state"]["triggered_today_or_cooldown"] is True


@patch("schroeder_trader.basket.orchestrator._compute_signal_for_ticker")
def test_compute_decisions_logs_shadow_signal_with_pipeline_basket(
    mock_sig, tmp_path,
):
    mock_sig.return_value = _fake_signal_pipeline("BUY", "SMA")
    store = _make_store(tmp_path)
    now = datetime(2026, 5, 21, 20, 30, tzinfo=timezone.utc)

    compute_decisions(store, {"SPY": 1.0}, pd.DataFrame(), now, portfolio_value=100000.0)

    ss = pd.read_csv(tmp_path / "shadow_signals.csv")
    assert len(ss) == 1
    assert ss.iloc[0]["pipeline"] == "basket"
    assert ss.iloc[0]["ticker"] == "SPY"
    assert ss.iloc[0]["ml_signal"] == "BUY"
