from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from schroeder_trader.basket.portfolio import (
    bootstrap_starting_value,
    read_position_qty,
    prior_exposure,
    read_trading_dates,
    write_basket_portfolio_snapshot,
)
from schroeder_trader.storage.csv_store import CsvStore


def _make_store(tmp_path: Path) -> CsvStore:
    return CsvStore(tmp_path)


def test_bootstrap_returns_latest_basket_total_when_basket_rows_exist(tmp_path):
    pf = pd.DataFrame({
        "id": [1, 2, 3],
        "timestamp": [
            "2026-05-19T20:30:00+00:00",
            "2026-05-20T20:30:00+00:00",
            "2026-05-20T20:30:00+00:00",
        ],
        "pipeline": ["spy_only", "basket", "basket"],
        "ticker": ["SPY", "SPY", "XLK"],
        "cash": [1965.4, 500.0, 500.0],
        "position_qty": [141, 64, 181],
        "position_value": [100000.0, 47000.0, 32000.0],
        "total_value": [101965.4, 105000.0, 105000.0],
    })
    pf.to_csv(tmp_path / "portfolio.csv", index=False)
    store = _make_store(tmp_path)
    assert bootstrap_starting_value(store) == 105000.0


def test_bootstrap_falls_back_to_spy_only_when_no_basket_rows(tmp_path):
    pf = pd.DataFrame({
        "id": [1], "timestamp": ["2026-05-19T20:30:00+00:00"],
        "pipeline": ["spy_only"], "ticker": ["SPY"],
        "cash": [1965.4], "position_qty": [141],
        "position_value": [100000.0], "total_value": [101965.4],
    })
    pf.to_csv(tmp_path / "portfolio.csv", index=False)
    store = _make_store(tmp_path)
    assert bootstrap_starting_value(store) == 101965.4


def test_bootstrap_raises_when_no_rows_at_all(tmp_path):
    pd.DataFrame(columns=[
        "id", "timestamp", "pipeline", "ticker",
        "cash", "position_qty", "position_value", "total_value",
    ]).to_csv(tmp_path / "portfolio.csv", index=False)
    store = _make_store(tmp_path)
    with pytest.raises(RuntimeError, match="no portfolio history"):
        bootstrap_starting_value(store)


def test_read_position_qty_returns_zero_when_no_basket_rows_for_ticker(tmp_path):
    pf = pd.DataFrame({
        "id": [1], "timestamp": ["2026-05-19T20:30:00+00:00"],
        "pipeline": ["basket"], "ticker": ["SPY"],
        "cash": [500.0], "position_qty": [64],
        "position_value": [47000.0], "total_value": [105000.0],
    })
    pf.to_csv(tmp_path / "portfolio.csv", index=False)
    store = _make_store(tmp_path)
    assert read_position_qty(store, "XLK") == 0
    assert read_position_qty(store, "SPY") == 64


def test_prior_exposure_returns_one_when_last_signal_was_buy(tmp_path):
    ss = pd.DataFrame({
        "id": [1, 2],
        "timestamp": ["2026-05-19T20:30:00+00:00", "2026-05-20T20:30:00+00:00"],
        "pipeline": ["basket", "basket"],
        "ticker": ["XLK", "XLK"],
        "close_price": [100.0, 105.0],
        "predicted_class": [None, None],
        "predicted_proba": [None, None],
        "ml_signal": ["BUY", "HOLD"],
        "sma_signal": ["BUY", "HOLD"],
        "regime": ["BULL", "BULL"],
        "signal_source": ["SMA", "SMA"],
        "bear_day_count": [None, None],
        "kelly_fraction": [None, None],
        "kelly_qty": [None, None],
        "high_water_mark": [None, None],
        "trailing_stop_triggered": [None, None],
    })
    ss.to_csv(tmp_path / "shadow_signals.csv", index=False)
    store = _make_store(tmp_path)
    assert prior_exposure(store, "XLK") == 1.0  # HOLD walks back to BUY


def test_prior_exposure_returns_zero_when_last_decided_was_sell_then_hold(tmp_path):
    ss = pd.DataFrame({
        "id": [1, 2],
        "timestamp": ["2026-05-19T20:30:00+00:00", "2026-05-20T20:30:00+00:00"],
        "pipeline": ["basket", "basket"],
        "ticker": ["XLK", "XLK"],
        "close_price": [100.0, 105.0],
        "predicted_class": [None, None],
        "predicted_proba": [None, None],
        "ml_signal": ["SELL", "HOLD"],
        "sma_signal": ["SELL", "HOLD"],
        "regime": ["BEAR", "BEAR"],
        "signal_source": ["FLAT", "FLAT"],
        "bear_day_count": [None, None],
        "kelly_fraction": [None, None],
        "kelly_qty": [None, None],
        "high_water_mark": [None, None],
        "trailing_stop_triggered": [None, None],
    })
    ss.to_csv(tmp_path / "shadow_signals.csv", index=False)
    store = _make_store(tmp_path)
    assert prior_exposure(store, "XLK") == 0.0  # HOLD walks back to SELL


def test_prior_exposure_returns_zero_when_no_prior_basket_rows(tmp_path):
    pd.DataFrame(columns=[
        "id", "timestamp", "pipeline", "ticker", "close_price",
        "predicted_class", "predicted_proba", "ml_signal", "sma_signal",
        "regime", "signal_source", "bear_day_count",
        "kelly_fraction", "kelly_qty", "high_water_mark",
        "trailing_stop_triggered",
    ]).to_csv(tmp_path / "shadow_signals.csv", index=False)
    store = _make_store(tmp_path)
    assert prior_exposure(store, "XLK") == 0.0


def test_read_trading_dates_returns_basket_pipeline_dates_for_ticker(tmp_path):
    from datetime import date
    ss = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "timestamp": [
            "2026-05-12T20:30:00+00:00", "2026-05-13T20:30:00+00:00",
            "2026-05-12T20:30:00+00:00", "2026-05-13T20:30:00+00:00",
        ],
        "pipeline": ["basket", "basket", "spy_only", "spy_only"],
        "ticker": ["XLK", "XLK", "XLK", "XLK"],
        "close_price": [100.0, 105.0, 100.0, 105.0],
        "predicted_class": [None] * 4, "predicted_proba": [None] * 4,
        "ml_signal": ["BUY"] * 4, "sma_signal": ["BUY"] * 4,
        "regime": ["BULL"] * 4, "signal_source": ["SMA"] * 4,
        "bear_day_count": [None] * 4, "kelly_fraction": [None] * 4,
        "kelly_qty": [None] * 4, "high_water_mark": [None] * 4,
        "trailing_stop_triggered": [None] * 4,
    })
    ss.to_csv(tmp_path / "shadow_signals.csv", index=False)
    store = _make_store(tmp_path)
    dates = read_trading_dates(store, "XLK")
    # Only basket rows; ET-local dates
    assert len(dates) == 2
    assert dates[0] == date(2026, 5, 12)
    assert dates[1] == date(2026, 5, 13)


def test_write_basket_portfolio_snapshot_emits_one_row_per_ticker(tmp_path):
    pf_empty = pd.DataFrame(columns=[
        "id", "timestamp", "pipeline", "ticker",
        "cash", "position_qty", "position_value", "total_value",
    ])
    pf_empty.to_csv(tmp_path / "portfolio.csv", index=False)
    store = _make_store(tmp_path)

    fake_broker = MagicMock()
    fake_broker.get_position.side_effect = lambda t: {"SPY": 64, "XLK": 181}.get(t, 0)
    fake_broker.get_account.return_value = {"cash": 500.0, "portfolio_value": 105000.0}

    prices = {"SPY": 738.0, "XLK": 175.0}
    now = datetime(2026, 5, 21, 20, 30, tzinfo=timezone.utc)
    write_basket_portfolio_snapshot(
        store, fake_broker, ["SPY", "XLK"], prices, now,
    )

    pf = pd.read_csv(tmp_path / "portfolio.csv")
    assert len(pf) == 2
    assert (pf["pipeline"] == "basket").all()
    assert set(pf["ticker"]) == {"SPY", "XLK"}
    assert (pf["total_value"] == 105000.0).all()  # repeated on each row
    assert (pf["cash"] == 500.0).all()
