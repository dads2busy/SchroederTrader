import sqlite3
from datetime import datetime, timezone

import pytest

from schroeder_trader.storage.trade_log import (
    init_db,
    log_signal,
    log_order,
    log_portfolio,
    get_signal_by_date,
    get_pending_orders,
    update_order_fill,
    log_shadow_signal,
    get_shadow_signals,
    log_llm_signal,
    get_llm_signals,
)


def test_init_db_creates_tables(tmp_path):
    db_path = tmp_path / "test.db"
    conn = init_db(db_path)
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cursor.fetchall()}
    assert {"signals", "orders", "portfolio"} <= tables
    conn.close()


def test_log_signal(tmp_path):
    db_path = tmp_path / "test.db"
    conn = init_db(db_path)
    now = datetime.now(timezone.utc)
    signal_id = log_signal(conn, now, "SPY", 523.10, 520.0, 518.0, "BUY")
    assert signal_id == 1

    row = conn.execute("SELECT * FROM signals WHERE id = ?", (signal_id,)).fetchone()
    assert row is not None
    assert row[2] == "SPY"  # ticker
    assert row[6] == "BUY"  # signal
    conn.close()


def test_log_order(tmp_path):
    db_path = tmp_path / "test.db"
    conn = init_db(db_path)
    now = datetime.now(timezone.utc)
    signal_id = log_signal(conn, now, "SPY", 523.10, 520.0, 518.0, "BUY")
    order_id = log_order(conn, signal_id, "alpaca-123", now, "SPY", "BUY", 45, "SUBMITTED")
    assert order_id == 1
    conn.close()


def test_log_portfolio(tmp_path):
    db_path = tmp_path / "test.db"
    conn = init_db(db_path)
    now = datetime.now(timezone.utc)
    log_portfolio(conn, now, 5000.0, 45, 23539.5, 28539.5)
    row = conn.execute("SELECT * FROM portfolio WHERE id = 1").fetchone()
    assert row[2] == 5000.0  # cash
    conn.close()


def test_get_signal_by_date_returns_none_when_missing(tmp_path):
    db_path = tmp_path / "test.db"
    conn = init_db(db_path)
    result = get_signal_by_date(conn, "2026-03-18")
    assert result is None
    conn.close()


def test_get_signal_by_date_returns_existing(tmp_path):
    db_path = tmp_path / "test.db"
    conn = init_db(db_path)
    now = datetime(2026, 3, 18, 21, 30, 0, tzinfo=timezone.utc)
    log_signal(conn, now, "SPY", 523.10, 520.0, 518.0, "HOLD")
    result = get_signal_by_date(conn, "2026-03-18")
    assert result is not None
    conn.close()


def test_get_pending_orders(tmp_path):
    db_path = tmp_path / "test.db"
    conn = init_db(db_path)
    now = datetime.now(timezone.utc)
    signal_id = log_signal(conn, now, "SPY", 523.10, 520.0, 518.0, "BUY")
    log_order(conn, signal_id, "alpaca-123", now, "SPY", "BUY", 45, "SUBMITTED")
    pending = get_pending_orders(conn)
    assert len(pending) == 1
    assert pending[0]["alpaca_order_id"] == "alpaca-123"
    conn.close()


def test_update_order_fill(tmp_path):
    db_path = tmp_path / "test.db"
    conn = init_db(db_path)
    now = datetime.now(timezone.utc)
    signal_id = log_signal(conn, now, "SPY", 523.10, 520.0, 518.0, "BUY")
    order_id = log_order(conn, signal_id, "alpaca-123", now, "SPY", "BUY", 45, "SUBMITTED")
    update_order_fill(conn, "alpaca-123", 523.50, now, "FILLED")
    row = conn.execute("SELECT fill_price, status FROM orders WHERE id = ?", (order_id,)).fetchone()
    assert row[0] == 523.50
    assert row[1] == "FILLED"
    conn.close()


def test_log_shadow_signal_with_kelly(tmp_path):
    conn = init_db(tmp_path / "test.db")
    log_shadow_signal(
        conn,
        timestamp=datetime(2026, 3, 21, 16, 30),
        ticker="SPY",
        close_price=650.0,
        predicted_class=2,
        predicted_proba='{"DOWN": 0.2, "FLAT": 0.2, "UP": 0.6}',
        ml_signal="BUY",
        sma_signal="HOLD",
        regime="CHOPPY",
        signal_source="XGB",
        bear_day_count=None,
        kelly_fraction=0.233,
        kelly_qty=35,
    )
    rows = get_shadow_signals(conn)
    assert len(rows) == 1
    assert rows[0]["kelly_fraction"] == pytest.approx(0.233)
    assert rows[0]["kelly_qty"] == 35
    conn.close()


def test_log_llm_signal_roundtrip(tmp_path):
    conn = init_db(tmp_path / "test.db")
    now = datetime(2026, 4, 20, 20, 30, tzinfo=timezone.utc)
    log_llm_signal(
        conn, now, "SPY", 710.14,
        provider="claude", model="claude-opus-4-7",
        action="BUY", confidence="HIGH", regime_assessment="BULL",
        key_drivers=["momentum", "vix low"], reasoning="trend intact",
        raw_response='{"action": "BUY"}',
    )
    rows = get_llm_signals(conn)
    assert len(rows) == 1
    assert rows[0]["provider"] == "claude"
    assert rows[0]["action"] == "BUY"
    assert rows[0]["key_drivers"] == '["momentum", "vix low"]'
    assert rows[0]["error"] is None
    conn.close()


def test_log_llm_signal_records_error(tmp_path):
    conn = init_db(tmp_path / "test.db")
    now = datetime(2026, 4, 20, 20, 30, tzinfo=timezone.utc)
    log_llm_signal(
        conn, now, "SPY", 710.14,
        provider="openai", model="gpt-5",
        action="HOLD", confidence="LOW", regime_assessment="CHOPPY",
        key_drivers=[], reasoning="",
        raw_response="", error="TimeoutError: read timed out",
    )
    rows = get_llm_signals(conn)
    assert rows[0]["error"] == "TimeoutError: read timed out"
    conn.close()


def test_log_shadow_signal_kelly_null_for_non_xgb(tmp_path):
    conn = init_db(tmp_path / "test.db")
    log_shadow_signal(
        conn,
        timestamp=datetime(2026, 3, 21, 16, 30),
        ticker="SPY",
        close_price=650.0,
        predicted_class=None,
        predicted_proba=None,
        ml_signal="SELL",
        sma_signal="HOLD",
        regime="BEAR",
        signal_source="FLAT",
        bear_day_count=3,
    )
    rows = get_shadow_signals(conn)
    assert len(rows) == 1
    assert rows[0]["kelly_fraction"] is None
    assert rows[0]["kelly_qty"] is None
    conn.close()
