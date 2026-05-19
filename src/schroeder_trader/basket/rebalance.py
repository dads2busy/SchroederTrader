"""Basket rebalancer — current vs target → order diffs."""

from __future__ import annotations

import logging
from datetime import datetime

from schroeder_trader.basket.portfolio import SimulatedBroker
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
    broker: SimulatedBroker,
    weights: dict[str, float],
    decisions: dict[str, dict],
    now: datetime,
) -> list[dict]:
    """Submit simulated orders to update broker state, log to orders.csv."""
    portfolio_value = broker.get_account()["portfolio_value"]
    current_positions = {t: broker.get_position(t) for t in weights}
    orders = compute_orders(portfolio_value, weights, decisions, current_positions)
    submitted: list[dict] = []
    for o in orders:
        try:
            result = broker.submit_order(o["ticker"], o["action"], o["qty"])
            log_order(
                store, signal_id=0, alpaca_order_id=result["alpaca_order_id"],
                timestamp=now, ticker=o["ticker"], action=o["action"],
                quantity=o["qty"], status=result["status"],
                signal_close_price=o["price"], pipeline="basket",
            )
            submitted.append({**o, "alpaca_order_id": result["alpaca_order_id"], "status": result["status"]})
        except Exception:
            logger.exception("Basket simulated order failed for %s (non-fatal)", o["ticker"])
            submitted.append({**o, "alpaca_order_id": None, "status": "ERROR"})
    return submitted
