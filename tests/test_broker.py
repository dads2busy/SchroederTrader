from unittest.mock import patch, MagicMock
from datetime import datetime, timezone

from schroeder_trader.execution.broker import submit_order, get_order_status, get_position, get_account, OrderResult
from schroeder_trader.risk.risk_manager import OrderRequest


@patch("schroeder_trader.execution.broker._get_trading_client")
def test_submit_buy_order(mock_client):
    client = MagicMock()
    order_response = MagicMock()
    order_response.id = "order-abc-123"
    order_response.status = "accepted"
    order_response.submitted_at = datetime.now(timezone.utc)
    client.submit_order.return_value = order_response
    mock_client.return_value = client

    request = OrderRequest(action="BUY", quantity=45)
    result = submit_order(request, "SPY")
    assert isinstance(result, OrderResult)
    assert result.alpaca_order_id == "order-abc-123"
    assert result.status == "SUBMITTED"
    assert result.quantity == 45


@patch("schroeder_trader.execution.broker._get_trading_client")
def test_submit_sell_order(mock_client):
    client = MagicMock()
    order_response = MagicMock()
    order_response.id = "order-def-456"
    order_response.status = "accepted"
    order_response.submitted_at = datetime.now(timezone.utc)
    client.submit_order.return_value = order_response
    mock_client.return_value = client

    request = OrderRequest(action="SELL", quantity=19)
    result = submit_order(request, "SPY")
    assert result.alpaca_order_id == "order-def-456"
    assert result.quantity == 19


@patch("schroeder_trader.execution.broker._get_trading_client")
def test_get_order_status_filled(mock_client):
    from alpaca.trading.enums import OrderStatus
    client = MagicMock()
    order = MagicMock()
    order.status = OrderStatus.FILLED
    order.filled_avg_price = 523.50
    order.filled_at = datetime.now(timezone.utc)
    client.get_order_by_id.return_value = order
    mock_client.return_value = client

    status = get_order_status("order-abc-123")
    assert status["status"] == "filled"
    assert status["fill_price"] == 523.50


@patch("schroeder_trader.execution.broker._get_trading_client")
def test_get_position_none(mock_client):
    from alpaca.common.exceptions import APIError
    from requests import HTTPError, Response
    client = MagicMock()
    resp = Response()
    resp.status_code = 404
    err = APIError("position does not exist", http_error=HTTPError(response=resp))
    client.get_open_position.side_effect = err
    mock_client.return_value = client

    qty = get_position("SPY")
    assert qty == 0


@patch("schroeder_trader.execution.broker._get_trading_client")
def test_get_account(mock_client):
    client = MagicMock()
    account = MagicMock()
    account.portfolio_value = "28539.50"
    account.cash = "5000.00"
    client.get_account.return_value = account
    mock_client.return_value = client

    info = get_account()
    assert info["portfolio_value"] == 28539.50
    assert info["cash"] == 5000.00


def test_retry_decorator_retries_then_succeeds(monkeypatch):
    from requests.exceptions import ConnectionError as RequestsConnectionError
    from schroeder_trader.execution.broker import _retry_on_connection_error

    monkeypatch.setattr("schroeder_trader.execution.broker.time.sleep", lambda _: None)
    calls = {"n": 0}

    @_retry_on_connection_error(retries=2, delay=0)
    def flaky():
        calls["n"] += 1
        if calls["n"] < 2:
            raise RequestsConnectionError("boom")
        return "ok"

    assert flaky() == "ok"
    assert calls["n"] == 2


def test_retry_decorator_reraises_after_exhaustion(monkeypatch):
    from requests.exceptions import ConnectionError as RequestsConnectionError
    from schroeder_trader.execution.broker import _retry_on_connection_error

    monkeypatch.setattr("schroeder_trader.execution.broker.time.sleep", lambda _: None)

    @_retry_on_connection_error(retries=2, delay=0)
    def always_fails():
        raise RequestsConnectionError("persistent")

    with __import__("pytest").raises(RequestsConnectionError):
        always_fails()


@patch("schroeder_trader.execution.broker._get_trading_client")
def test_list_recent_orders_filters_by_ticker_and_window(mock_client):
    from schroeder_trader.execution.broker import list_recent_orders
    from alpaca.trading.enums import QueryOrderStatus

    client = MagicMock()
    order1 = MagicMock()
    order2 = MagicMock()
    client.get_orders.return_value = [order1, order2]
    mock_client.return_value = client

    result = list_recent_orders("SPY", lookback_days=3)
    assert result == [order1, order2]

    call_kwargs = client.get_orders.call_args.kwargs
    request = call_kwargs["filter"]
    assert request.symbols == ["SPY"]
    assert request.status == QueryOrderStatus.ALL
    assert request.limit == 500
