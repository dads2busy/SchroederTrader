import json
from datetime import datetime, timezone
from pathlib import Path

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


def test_shadow_signal_with_regime_and_source(tmp_path):
    db_path = tmp_path / "test.db"
    conn = init_db(db_path)
    now = datetime.now(timezone.utc)

    signal_id = log_shadow_signal(
        conn, now, "SPY", 523.10,
        predicted_class=None,
        predicted_proba=None,
        ml_signal="SELL",
        sma_signal="HOLD",
        regime="BEAR",
        signal_source="FLAT",
        bear_day_count=5,
    )
    assert signal_id is not None

    row = conn.execute("SELECT * FROM shadow_signals WHERE id = ?", (signal_id,)).fetchone()
    assert row["regime"] == "BEAR"
    assert row["signal_source"] == "FLAT"
    assert row["bear_day_count"] == 5
    assert row["predicted_class"] is None
    assert row["predicted_proba"] is None
    conn.close()


def test_shadow_signal_with_trailing_stop():
    conn = init_db(Path(":memory:"))
    now = datetime.now()
    log_shadow_signal(
        conn, now, "SPY", 650.0,
        predicted_class=2,
        predicted_proba='{"UP": 0.6}',
        ml_signal="BUY",
        sma_signal="HOLD",
        high_water_mark=100000.0,
        trailing_stop_triggered=True,
    )
    rows = get_shadow_signals(conn)
    assert len(rows) == 1
    assert rows[0]["high_water_mark"] == 100000.0
    assert rows[0]["trailing_stop_triggered"] == 1
    conn.close()


def test_shadow_signal_trailing_stop_defaults_null():
    conn = init_db(Path(":memory:"))
    now = datetime.now()
    log_shadow_signal(
        conn, now, "SPY", 650.0,
        predicted_class=None,
        predicted_proba=None,
        ml_signal="HOLD",
        sma_signal="HOLD",
    )
    rows = get_shadow_signals(conn)
    assert rows[0]["high_water_mark"] is None
    assert rows[0]["trailing_stop_triggered"] is None
    conn.close()


def test_shadow_signal_xgb_with_proba(tmp_path):
    db_path = tmp_path / "test.db"
    conn = init_db(db_path)
    now = datetime.now(timezone.utc)
    proba = json.dumps({"DOWN": 0.1, "FLAT": 0.3, "UP": 0.6})

    signal_id = log_shadow_signal(
        conn, now, "SPY", 523.10,
        predicted_class=2,
        predicted_proba=proba,
        ml_signal="BUY",
        sma_signal="HOLD",
        regime="CHOPPY",
        signal_source="XGB",
        bear_day_count=None,
    )

    row = conn.execute("SELECT * FROM shadow_signals WHERE id = ?", (signal_id,)).fetchone()
    assert row["predicted_class"] == 2
    assert row["regime"] == "CHOPPY"
    assert row["signal_source"] == "XGB"
    assert row["bear_day_count"] is None
    conn.close()
