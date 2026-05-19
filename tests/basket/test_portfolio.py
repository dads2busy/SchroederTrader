from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from schroeder_trader.basket.portfolio import (
    SimulatedBroker,
    bootstrap_starting_value,
    load_basket_broker,
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

    broker = SimulatedBroker(
        cash=500.0,
        positions={"SPY": 64, "XLK": 181},
        prices={"SPY": 738.0, "XLK": 175.0},
    )

    now = datetime(2026, 5, 21, 20, 30, tzinfo=timezone.utc)
    write_basket_portfolio_snapshot(store, broker, ["SPY", "XLK"], now)

    pf = pd.read_csv(tmp_path / "portfolio.csv")
    assert len(pf) == 2
    assert (pf["pipeline"] == "basket").all()
    assert set(pf["ticker"]) == {"SPY", "XLK"}
    expected_total = 500.0 + 64 * 738.0 + 181 * 175.0
    assert all(abs(v - expected_total) < 0.01 for v in pf["total_value"])
    assert (pf["cash"] == 500.0).all()


def test_simulated_broker_submit_order_updates_position_and_cash():
    broker = SimulatedBroker(cash=10000.0, positions={"SPY": 0}, prices={"SPY": 100.0})
    result = broker.submit_order("SPY", "BUY", 50)
    assert result["status"] == "FILLED"
    assert broker.get_position("SPY") == 50
    assert broker.cash == 5000.0  # 10000 - 50*100
    # SELL flow
    broker.submit_order("SPY", "SELL", 20)
    assert broker.get_position("SPY") == 30
    assert broker.cash == 7000.0  # 5000 + 20*100


def test_load_basket_broker_cold_start_uses_spy_only_total(tmp_path):
    pd.DataFrame({
        "id": [1], "timestamp": ["2026-05-19T20:30:00+00:00"],
        "pipeline": ["spy_only"], "ticker": ["SPY"],
        "cash": [1965.4], "position_qty": [141],
        "position_value": [100000.0], "total_value": [101965.4],
    }).to_csv(tmp_path / "portfolio.csv", index=False)
    store = _make_store(tmp_path)
    broker = load_basket_broker(store, prices={"SPY": 700.0, "XLK": 175.0})
    assert broker.cash == 101965.4
    assert broker.get_position("SPY") == 0
    assert broker.get_position("XLK") == 0


def test_load_basket_broker_warm_start_uses_latest_basket_rows(tmp_path):
    pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "timestamp": [
            "2026-05-18T20:30:00+00:00",
            "2026-05-19T20:30:00+00:00",
            "2026-05-19T20:30:00+00:00",
            "2026-05-19T20:30:00+00:00",
            "2026-05-19T20:30:00+00:00",
        ],
        "pipeline": ["spy_only", "basket", "basket", "basket", "basket"],
        "ticker": ["SPY", "SPY", "XLK", "XLV", "XLE"],
        "cash": [1965.4, 500.0, 500.0, 500.0, 500.0],
        "position_qty": [141, 64, 181, 111, 175],
        "position_value": [100000.0, 47000.0, 32000.0, 16000.0, 10000.0],
        "total_value": [101965.4, 105500.0, 105500.0, 105500.0, 105500.0],
    }).to_csv(tmp_path / "portfolio.csv", index=False)
    store = _make_store(tmp_path)
    broker = load_basket_broker(store, prices={"SPY": 738.0, "XLK": 175.0, "XLV": 146.0, "XLE": 58.0})
    assert broker.cash == 500.0
    assert broker.get_position("SPY") == 64
    assert broker.get_position("XLK") == 181
    assert broker.get_position("XLV") == 111
    assert broker.get_position("XLE") == 175
