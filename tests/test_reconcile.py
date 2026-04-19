from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from alpaca.trading.enums import OrderSide, OrderStatus

from schroeder_trader.execution.reconcile import reconcile_orders
from schroeder_trader.storage.trade_log import (
    get_order_by_alpaca_id,
    init_db,
    log_order,
    log_signal,
)


def _make_alpaca_order(
    id, status, side, qty, submitted_at, filled_avg_price=None, filled_at=None
):
    order = MagicMock()
    order.id = id
    order.status = status
    order.side = OrderSide.BUY if side == "BUY" else OrderSide.SELL
    order.qty = qty
    order.submitted_at = submitted_at
    order.filled_avg_price = filled_avg_price
    order.filled_at = filled_at
    return order


@patch("schroeder_trader.execution.reconcile.list_recent_orders")
def test_reconcile_inserts_open_orphan_as_submitted(mock_list, tmp_path):
    conn = init_db(tmp_path / "test.db")
    submitted_at = datetime(2026, 4, 18, 20, tzinfo=timezone.utc)
    mock_list.return_value = [
        _make_alpaca_order("orphan-1", OrderStatus.NEW, "BUY", 10, submitted_at),
    ]
    assert reconcile_orders(conn, "SPY") == ["orphan-1"]
    row = get_order_by_alpaca_id(conn, "orphan-1")
    assert row["status"] == "SUBMITTED"
    assert row["action"] == "BUY"
    assert row["quantity"] == 10
    assert row["signal_id"] == 0
    assert row["fill_price"] is None
    conn.close()


@patch("schroeder_trader.execution.reconcile.list_recent_orders")
def test_reconcile_inserts_filled_orphan_with_fill_data(mock_list, tmp_path):
    conn = init_db(tmp_path / "test.db")
    submitted = datetime(2026, 4, 18, 13, tzinfo=timezone.utc)
    filled = datetime(2026, 4, 18, 14, tzinfo=timezone.utc)
    mock_list.return_value = [
        _make_alpaca_order(
            "orphan-2", OrderStatus.FILLED, "SELL", 5, submitted,
            filled_avg_price=700.0, filled_at=filled,
        ),
    ]
    assert reconcile_orders(conn, "SPY") == ["orphan-2"]
    row = get_order_by_alpaca_id(conn, "orphan-2")
    assert row["status"] == "FILLED"
    assert row["action"] == "SELL"
    assert row["fill_price"] == 700.0
    assert row["fill_timestamp"] == filled.isoformat()
    conn.close()


@patch("schroeder_trader.execution.reconcile.list_recent_orders")
def test_reconcile_inserts_cancelled_orphan_as_rejected(mock_list, tmp_path):
    conn = init_db(tmp_path / "test.db")
    submitted = datetime(2026, 4, 18, 13, tzinfo=timezone.utc)
    mock_list.return_value = [
        _make_alpaca_order("orphan-3", OrderStatus.CANCELED, "BUY", 10, submitted),
    ]
    assert reconcile_orders(conn, "SPY") == ["orphan-3"]
    row = get_order_by_alpaca_id(conn, "orphan-3")
    assert row["status"] == "REJECTED"
    conn.close()


@patch("schroeder_trader.execution.reconcile.list_recent_orders")
def test_reconcile_skips_orders_already_in_db(mock_list, tmp_path):
    conn = init_db(tmp_path / "test.db")
    now = datetime(2026, 4, 18, 14, tzinfo=timezone.utc)
    signal_id = log_signal(conn, now, "SPY", 700.0, 695.0, 690.0, "BUY")
    log_order(conn, signal_id, "known-1", now, "SPY", "BUY", 10, "SUBMITTED")
    mock_list.return_value = [
        _make_alpaca_order("known-1", OrderStatus.NEW, "BUY", 10, now),
    ]
    assert reconcile_orders(conn, "SPY") == []
    conn.close()


@patch("schroeder_trader.execution.reconcile.list_recent_orders")
def test_reconcile_empty_response(mock_list, tmp_path):
    conn = init_db(tmp_path / "test.db")
    mock_list.return_value = []
    assert reconcile_orders(conn, "SPY") == []
    conn.close()


@patch("schroeder_trader.execution.reconcile.list_recent_orders")
def test_reconcile_mixed_known_and_orphan(mock_list, tmp_path):
    conn = init_db(tmp_path / "test.db")
    now = datetime(2026, 4, 18, 14, tzinfo=timezone.utc)
    signal_id = log_signal(conn, now, "SPY", 700.0, 695.0, 690.0, "BUY")
    log_order(conn, signal_id, "known-1", now, "SPY", "BUY", 10, "SUBMITTED")
    mock_list.return_value = [
        _make_alpaca_order("known-1", OrderStatus.NEW, "BUY", 10, now),
        _make_alpaca_order("orphan-x", OrderStatus.FILLED, "BUY", 5, now,
                           filled_avg_price=700.5, filled_at=now),
    ]
    assert reconcile_orders(conn, "SPY") == ["orphan-x"]
    conn.close()
