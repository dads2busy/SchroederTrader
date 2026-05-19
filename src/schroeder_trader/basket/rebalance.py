"""Basket rebalancer — current vs target → order diffs.

Pure compute_orders function for testability. rebalance_to_targets is
the side-effecting wrapper that calls the broker and logs orders.
"""

from __future__ import annotations

import logging
from datetime import datetime

from schroeder_trader.execution.broker import OrderResult, submit_order
from schroeder_trader.risk.risk_manager import OrderRequest
from schroeder_trader.storage.csv_store import CsvStore
from schroeder_trader.storage.trade_log import log_order

logger = logging.getLogger(__name__)


def compute_orders(
    portfolio_value: float,
    weights: dict[str, float],
    decisions: dict[str, dict],
    current_positions: dict[str, int],
) -> list[dict]:
    """Return the order diffs needed to bring each ticker to its target.

    Each returned dict has keys: ticker, action ('BUY' or 'SELL'), qty (int),
    price (float, the close used for sizing).

    Sub-share diffs (|diff_value| < price of one share) are skipped.
    """
    orders: list[dict] = []
    for ticker, weight in weights.items():
        d = decisions[ticker]
        target_value = portfolio_value * weight * d["exposure"]
        current_qty = int(current_positions.get(ticker, 0))
        current_value = current_qty * d["price"]
        diff_value = target_value - current_value

        if abs(diff_value) < d["price"]:
            continue

        diff_shares = int(diff_value // d["price"])
        if diff_shares == 0:
            continue

        orders.append({
            "ticker": ticker,
            "action": "BUY" if diff_shares > 0 else "SELL",
            "qty": abs(diff_shares),
            "price": d["price"],
        })
    return orders


def rebalance_to_targets(
    store: CsvStore,
    portfolio_value: float,
    weights: dict[str, float],
    decisions: dict[str, dict],
    current_positions: dict[str, int],
    now: datetime,
) -> list[dict]:
    """Submit market orders for each ticker's diff and log them.

    Returns the list of orders submitted (each enriched with `alpaca_order_id`
    and `status` from the broker response). Errors on individual orders are
    logged but don't stop the loop — the basket pipeline is paper-mode and
    a single ticker failing is non-fatal.
    """
    orders = compute_orders(portfolio_value, weights, decisions, current_positions)
    submitted: list[dict] = []
    for o in orders:
        try:
            req = OrderRequest(action=o["action"], quantity=o["qty"])
            result = submit_order(req, ticker=o["ticker"])
            log_order(
                store, signal_id=0, alpaca_order_id=result.alpaca_order_id,
                timestamp=now, ticker=o["ticker"], action=o["action"],
                quantity=o["qty"], status=result.status,
                signal_close_price=o["price"], pipeline="basket",
            )
            submitted.append({**o, "alpaca_order_id": result.alpaca_order_id, "status": result.status})
        except Exception:
            logger.exception("Basket order failed for %s (non-fatal)", o["ticker"])
            submitted.append({**o, "alpaca_order_id": None, "status": "ERROR"})
    return submitted
