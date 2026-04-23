import json
from datetime import datetime, timezone

from schroeder_trader.storage.trade_log import (
    init_db,
    log_shadow_signal,
    get_shadow_signals,
)


def test_init_db_creates_store(tmp_path):
    store = init_db(tmp_path / "test.db")
    assert store.root == tmp_path
    store.close()


def test_log_shadow_signal(tmp_path):
    store = init_db(tmp_path / "test.db")
    now = datetime.now(timezone.utc)
    proba = json.dumps({"DOWN": 0.1, "FLAT": 0.3, "UP": 0.6})

    signal_id = log_shadow_signal(store, now, "SPY", 523.10, 2, proba, "BUY", "HOLD")
    assert signal_id == 1

    rows = get_shadow_signals(store)
    assert len(rows) == 1
    assert rows[0]["ticker"] == "SPY"
    assert rows[0]["ml_signal"] == "BUY"
    assert rows[0]["sma_signal"] == "HOLD"
    assert rows[0]["predicted_class"] == 2
    store.close()


def test_get_shadow_signals(tmp_path):
    store = init_db(tmp_path / "test.db")
    now = datetime.now(timezone.utc)
    proba_up = json.dumps({"DOWN": 0.1, "FLAT": 0.3, "UP": 0.6})
    proba_down = json.dumps({"DOWN": 0.7, "FLAT": 0.2, "UP": 0.1})

    log_shadow_signal(store, now, "SPY", 523.10, 2, proba_up, "BUY", "HOLD")
    log_shadow_signal(store, now, "SPY", 524.00, 0, proba_down, "SELL", "HOLD")

    signals = get_shadow_signals(store)
    assert len(signals) == 2
    assert signals[0]["ml_signal"] == "BUY"
    assert signals[1]["ml_signal"] == "SELL"
    store.close()


def test_shadow_signal_with_regime_and_source(tmp_path):
    store = init_db(tmp_path / "test.db")
    now = datetime.now(timezone.utc)

    signal_id = log_shadow_signal(
        store, now, "SPY", 523.10,
        predicted_class=None,
        predicted_proba=None,
        ml_signal="SELL",
        sma_signal="HOLD",
        regime="BEAR",
        signal_source="FLAT",
        bear_day_count=5,
    )
    assert signal_id is not None

    rows = get_shadow_signals(store)
    row = rows[0]
    assert row["regime"] == "BEAR"
    assert row["signal_source"] == "FLAT"
    assert row["bear_day_count"] == 5
    assert row["predicted_class"] is None
    assert row["predicted_proba"] is None
    store.close()


def test_shadow_signal_with_trailing_stop(tmp_path):
    store = init_db(tmp_path / "test.db")
    now = datetime.now()
    log_shadow_signal(
        store, now, "SPY", 650.0,
        predicted_class=2,
        predicted_proba='{"UP": 0.6}',
        ml_signal="BUY",
        sma_signal="HOLD",
        high_water_mark=100000.0,
        trailing_stop_triggered=True,
    )
    rows = get_shadow_signals(store)
    assert len(rows) == 1
    assert rows[0]["high_water_mark"] == 100000.0
    assert rows[0]["trailing_stop_triggered"] == 1
    store.close()


def test_shadow_signal_trailing_stop_defaults_null(tmp_path):
    store = init_db(tmp_path / "test.db")
    now = datetime.now()
    log_shadow_signal(
        store, now, "SPY", 650.0,
        predicted_class=None,
        predicted_proba=None,
        ml_signal="HOLD",
        sma_signal="HOLD",
    )
    rows = get_shadow_signals(store)
    assert rows[0]["high_water_mark"] is None
    assert rows[0]["trailing_stop_triggered"] is None
    store.close()


def test_shadow_signal_xgb_with_proba(tmp_path):
    store = init_db(tmp_path / "test.db")
    now = datetime.now(timezone.utc)
    proba = json.dumps({"DOWN": 0.1, "FLAT": 0.3, "UP": 0.6})

    log_shadow_signal(
        store, now, "SPY", 523.10,
        predicted_class=2,
        predicted_proba=proba,
        ml_signal="BUY",
        sma_signal="HOLD",
        regime="CHOPPY",
        signal_source="XGB",
        bear_day_count=None,
    )

    rows = get_shadow_signals(store)
    row = rows[0]
    assert row["predicted_class"] == 2
    assert row["regime"] == "CHOPPY"
    assert row["signal_source"] == "XGB"
    assert row["bear_day_count"] is None
    store.close()
