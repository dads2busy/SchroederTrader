import logging

from alpaca.trading.enums import OrderSide, OrderStatus

from schroeder_trader.execution.broker import list_recent_orders
from schroeder_trader.storage.trade_log import (
    get_order_by_alpaca_id,
    insert_reconciled_order,
)

logger = logging.getLogger(__name__)


_TERMINAL_FAIL_STATUSES = {
    OrderStatus.CANCELED,
    OrderStatus.EXPIRED,
    OrderStatus.REJECTED,
}


def reconcile_orders(conn, ticker: str, lookback_days: int = 7) -> list[str]:
    """Find Alpaca orders not in our DB and insert them.

    Catches the "submit_order returned at broker but crashed before log_order" gap:
    an order exists at Alpaca that we have no record of. Without this, a subsequent
    BUY evaluation could double-fire against a position Alpaca already reflects.

    Returns list of alpaca_order_ids that were newly reconciled.
    """
    alpaca_orders = list_recent_orders(ticker, lookback_days=lookback_days)
    reconciled: list[str] = []
    for order in alpaca_orders:
        alpaca_id = str(order.id)
        if get_order_by_alpaca_id(conn, alpaca_id):
            continue

        action = "BUY" if order.side == OrderSide.BUY else "SELL"

        if order.status == OrderStatus.FILLED:
            db_status = "FILLED"
            fill_price = float(order.filled_avg_price) if order.filled_avg_price else None
            fill_ts = order.filled_at
        elif order.status in _TERMINAL_FAIL_STATUSES:
            db_status = "REJECTED"
            fill_price = None
            fill_ts = None
        else:
            db_status = "SUBMITTED"
            fill_price = None
            fill_ts = None

        insert_reconciled_order(
            conn,
            alpaca_order_id=alpaca_id,
            timestamp=order.submitted_at,
            ticker=ticker,
            action=action,
            quantity=int(order.qty),
            status=db_status,
            fill_price=fill_price,
            fill_timestamp=fill_ts,
        )
        logger.warning(
            "Reconciled orphaned Alpaca order %s (%s %s %d, status=%s)",
            alpaca_id, action, ticker, int(order.qty), db_status,
        )
        reconciled.append(alpaca_id)
    return reconciled
