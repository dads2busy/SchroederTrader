import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import wraps

from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, OrderStatus, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest
from requests.exceptions import ConnectionError as RequestsConnectionError

from schroeder_trader.config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL
from schroeder_trader.risk.risk_manager import OrderRequest

logger = logging.getLogger(__name__)

_client = None


def _retry_on_connection_error(retries: int = 2, delay: float = 3.0):
    """Decorator: retry on requests ConnectionError (e.g. RemoteDisconnected).

    Alpaca occasionally drops idle connections. Retrying with a fresh socket
    almost always succeeds within 1-2 attempts. Only apply to idempotent reads;
    do NOT apply to submit_order (would risk double-fire without client_order_id).
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            last = None
            for attempt in range(retries + 1):
                try:
                    return fn(*args, **kwargs)
                except RequestsConnectionError as e:
                    last = e
                    if attempt < retries:
                        logger.warning(
                            "%s connection error (attempt %d/%d): %s — retrying in %.1fs",
                            fn.__name__, attempt + 1, retries + 1, e, delay,
                        )
                        time.sleep(delay)
            raise last
        return wrapper
    return decorator


@dataclass
class OrderResult:
    alpaca_order_id: str
    quantity: int
    timestamp: datetime
    status: str


def _get_trading_client() -> TradingClient:
    global _client
    if _client is None:
        _client = TradingClient(
            ALPACA_API_KEY,
            ALPACA_SECRET_KEY,
            paper="paper" in ALPACA_BASE_URL,
        )
    return _client


def submit_order(request: OrderRequest, ticker: str) -> OrderResult:
    client = _get_trading_client()
    side = OrderSide.BUY if request.action == "BUY" else OrderSide.SELL

    order_data = MarketOrderRequest(
        symbol=ticker,
        qty=request.quantity,
        side=side,
        time_in_force=TimeInForce.DAY,
    )

    order = client.submit_order(order_data)
    logger.info("Submitted %s order for %d %s — Alpaca ID: %s", request.action, request.quantity, ticker, order.id)

    return OrderResult(
        alpaca_order_id=str(order.id),
        quantity=request.quantity,
        timestamp=order.submitted_at or datetime.now(timezone.utc),
        status="SUBMITTED",
    )


@_retry_on_connection_error()
def get_order_status(alpaca_order_id: str) -> dict:
    client = _get_trading_client()
    order = client.get_order_by_id(alpaca_order_id)
    status_str = order.status.value if hasattr(order.status, 'value') else str(order.status)
    result = {"status": status_str}
    if order.status == OrderStatus.FILLED:
        result["fill_price"] = float(order.filled_avg_price)
        result["fill_timestamp"] = order.filled_at
    return result


@_retry_on_connection_error()
def get_position(ticker: str) -> int:
    client = _get_trading_client()
    try:
        position = client.get_open_position(ticker)
        return int(position.qty)
    except APIError as e:
        if e.status_code == 404:
            return 0  # no position held
        raise


@_retry_on_connection_error()
def get_account() -> dict:
    client = _get_trading_client()
    account = client.get_account()
    return {
        "portfolio_value": float(account.portfolio_value),
        "cash": float(account.cash),
    }


@_retry_on_connection_error()
def list_recent_orders(ticker: str, lookback_days: int = 7) -> list:
    """Return Alpaca orders for ticker across all statuses within lookback window."""
    client = _get_trading_client()
    after = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    request = GetOrdersRequest(
        status=QueryOrderStatus.ALL,
        symbols=[ticker],
        after=after,
        limit=500,
    )
    return list(client.get_orders(filter=request))
