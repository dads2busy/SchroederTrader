from datetime import datetime, timezone
from unittest.mock import MagicMock

import pandas as pd
import pytest

from schroeder_trader.basket.rebalance import compute_orders


def _decisions(**overrides):
    base = {
        "SPY": {"exposure": 1.0, "price": 738.0},
        "XLK": {"exposure": 1.0, "price": 176.0},
        "XLV": {"exposure": 1.0, "price": 146.0},
        "XLE": {"exposure": 1.0, "price": 58.0},
    }
    for t, vals in overrides.items():
        base[t] = {**base[t], **vals}
    return base


WEIGHTS = {"SPY": 0.45, "XLK": 0.30, "XLV": 0.15, "XLE": 0.10}


def test_compute_orders_initial_allocation_from_zero_positions():
    decisions = _decisions()
    current = {"SPY": 0, "XLK": 0, "XLV": 0, "XLE": 0}
    orders = compute_orders(100_000.0, WEIGHTS, decisions, current)
    by_ticker = {o["ticker"]: o for o in orders}
    # SPY: target 45000 / 738 = 60.97 → 60 shares BUY
    assert by_ticker["SPY"]["action"] == "BUY"
    assert by_ticker["SPY"]["qty"] == 60
    # XLK: 30000 / 176 = 170.45 → 170 shares
    assert by_ticker["XLK"]["qty"] == 170
    # XLV: 15000 / 146 = 102.74 → 102
    assert by_ticker["XLV"]["qty"] == 102
    # XLE: 10000 / 58 = 172.41 → 172
    assert by_ticker["XLE"]["qty"] == 172


def test_compute_orders_skips_when_diff_less_than_one_share():
    """Drift smaller than the price of one share should not produce an order."""
    decisions = _decisions(SPY={"exposure": 1.0, "price": 738.0})
    current = {"SPY": 61, "XLK": 170, "XLV": 102, "XLE": 172}
    # SPY: 61×738=45018, target 45000, diff=-18, abs(18) < 738 → SKIP
    orders = compute_orders(100_000.0, WEIGHTS, decisions, current)
    by_ticker = {o["ticker"]: o for o in orders}
    assert "SPY" not in by_ticker  # skipped


def test_compute_orders_zero_exposure_sells_to_flat():
    decisions = _decisions(XLE={"exposure": 0.0, "price": 58.0})
    current = {"SPY": 60, "XLK": 170, "XLV": 102, "XLE": 172}
    orders = compute_orders(100_000.0, WEIGHTS, decisions, current)
    by_ticker = {o["ticker"]: o for o in orders}
    assert by_ticker["XLE"]["action"] == "SELL"
    assert by_ticker["XLE"]["qty"] == 172  # full close


def test_compute_orders_buys_when_underweight():
    decisions = _decisions(SPY={"exposure": 1.0, "price": 700.0})
    # Target = 45% × 100000 / 700 = 64.28 → 64 shares
    # Current = 50 shares → BUY 14
    current = {"SPY": 50, "XLK": 170, "XLV": 102, "XLE": 172}
    orders = compute_orders(100_000.0, WEIGHTS, decisions, current)
    by_ticker = {o["ticker"]: o for o in orders}
    assert by_ticker["SPY"]["action"] == "BUY"
    assert by_ticker["SPY"]["qty"] == 14


def test_compute_orders_no_orders_when_all_at_target():
    decisions = _decisions()
    # Targets: SPY 60, XLK 170, XLV 102, XLE 172 (per the initial test)
    current = {"SPY": 60, "XLK": 170, "XLV": 102, "XLE": 172}
    orders = compute_orders(100_000.0, WEIGHTS, decisions, current)
    assert orders == []
