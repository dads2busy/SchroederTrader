import math

from schroeder_trader.risk.risk_manager import evaluate, OrderRequest
from schroeder_trader.strategy.sma_crossover import Signal


def test_buy_calculates_whole_shares():
    request = evaluate(
        signal=Signal.BUY,
        portfolio_value=10000.0,
        close_price=523.10,
        current_position_qty=0,
    )
    assert request is not None
    assert request.action == "BUY"
    expected_qty = math.floor(10000.0 * (1 - 0.02) / 523.10)
    assert request.quantity == expected_qty


def test_buy_when_already_holding_returns_none():
    request = evaluate(
        signal=Signal.BUY,
        portfolio_value=10000.0,
        close_price=523.10,
        current_position_qty=19,
    )
    assert request is None


def test_sell_with_position_returns_sell():
    request = evaluate(
        signal=Signal.SELL,
        portfolio_value=10000.0,
        close_price=523.10,
        current_position_qty=19,
    )
    assert request is not None
    assert request.action == "SELL"
    assert request.quantity == 19


def test_sell_without_position_returns_none():
    request = evaluate(
        signal=Signal.SELL,
        portfolio_value=10000.0,
        close_price=523.10,
        current_position_qty=0,
    )
    assert request is None


def test_hold_returns_none():
    request = evaluate(
        signal=Signal.HOLD,
        portfolio_value=10000.0,
        close_price=523.10,
        current_position_qty=19,
    )
    assert request is None


def test_buy_with_small_portfolio():
    request = evaluate(
        signal=Signal.BUY,
        portfolio_value=100.0,
        close_price=523.10,
        current_position_qty=0,
    )
    assert request is None
