import logging
import math
from dataclasses import dataclass

from schroeder_trader.config import CASH_BUFFER_PCT
from schroeder_trader.strategy.sma_crossover import Signal

logger = logging.getLogger(__name__)


@dataclass
class OrderRequest:
    action: str  # "BUY" or "SELL"
    quantity: int


def evaluate(
    signal: Signal,
    portfolio_value: float,
    close_price: float,
    current_position_qty: int,
    cash_buffer_pct: float = CASH_BUFFER_PCT,
) -> OrderRequest | None:
    if signal == Signal.HOLD:
        return None

    if signal == Signal.BUY:
        if current_position_qty > 0:
            logger.info("BUY signal but already holding %d shares, skipping", current_position_qty)
            return None
        available = portfolio_value * (1 - cash_buffer_pct)
        quantity = math.floor(available / close_price)
        if quantity < 1:
            logger.warning("Portfolio too small to buy even 1 share at $%.2f", close_price)
            return None
        return OrderRequest(action="BUY", quantity=quantity)

    if signal == Signal.SELL:
        if current_position_qty <= 0:
            logger.info("SELL signal but no position held, treating as HOLD")
            return None
        return OrderRequest(action="SELL", quantity=current_position_qty)

    return None
