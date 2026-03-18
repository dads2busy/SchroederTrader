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
