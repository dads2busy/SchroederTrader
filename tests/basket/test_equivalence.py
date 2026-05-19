"""Equivalence acceptance gate: basket pipeline with weights={SPY: 1.0}
must produce SPY position_qty and total_value matching the SPY-only
pipeline's same-day output across three fixture days.

Fixtures are limited to BEAR/FLAT days where neither pipeline trades.
This is the only scenario where exact equivalence holds by design:

- SPY-only uses CASH_BUFFER_PCT=0.02 (98% in); basket targets 100%.
  On a BUY/HOLD day where the morning position is 141 shares, basket
  would compute a ~2-share higher target and diverge. Trading days have
  a structural ~2% share-count divergence that is intentional, not a bug.
- On BEAR/FLAT days both pipelines receive a SELL signal, set exposure=0,
  submit no orders, and end with all-cash portfolios. Exact equivalence
  is both meaningful and verifiable on these days.

The stateful broker mock updates positions on submit_order and computes
get_account from post-fill state. close_price is read from the recorded
shadow_signals.csv value on each fixture date (not derived).

Fixture dates chosen:
- 2026-03-31: BEAR/FLAT, qty=0, close=650.34 (mid-bear-run)
- 2026-04-02: BEAR/FLAT, qty=0, close=653.78 (two days later)
- 2026-04-07: BEAR/FLAT, qty=0, close=659.22 (end of bear period)
"""

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


_FIXTURES = Path(__file__).parent / "fixtures"

FIXTURE_DAYS = sorted(d.name for d in _FIXTURES.iterdir() if d.is_dir()) if _FIXTURES.exists() else []


class _StatefulBroker:
    def __init__(self, initial_positions: dict, cash: float, close_price: float):
        self.positions = dict(initial_positions)
        self.cash = cash
        self.close_price = close_price

    def get_position(self, ticker):
        return self.positions.get(ticker, 0)

    def get_account(self):
        equity = sum(qty * self.close_price for qty in self.positions.values())
        return {"cash": self.cash, "portfolio_value": self.cash + equity}

    def submit_order(self, request, ticker):
        signed_qty = request.quantity if request.action == "BUY" else -request.quantity
        self.positions[ticker] = self.positions.get(ticker, 0) + signed_qty
        self.cash -= signed_qty * self.close_price
        return MagicMock(alpaca_order_id=f"test-{ticker}-{signed_qty}", status="FILLED")


@pytest.mark.parametrize("fixture_date", FIXTURE_DAYS)
def test_basket_with_spy_only_weight_matches_spy_only_pipeline(
    fixture_date, tmp_path,
):
    """On HOLD/FLAT days where neither pipeline trades, basket pipeline with
    weights={SPY: 1.0} must produce the same position_qty and total_value
    as the SPY-only pipeline's recorded output for that date."""
    snapshot_dir = _FIXTURES / fixture_date

    for f in ["portfolio.csv", "orders.csv", "shadow_signals.csv", "signals.csv"]:
        src = snapshot_dir / f
        if src.exists():
            (tmp_path / f).write_text(src.read_text())

    broker_state = json.loads((snapshot_dir / "broker_state.json").read_text())
    expected = json.loads((snapshot_dir / "expected_state.json").read_text())

    from schroeder_trader.strategy.composite import Signal
    from schroeder_trader.strategy.regime_detector import Regime

    recorded_signal = Signal[expected["ml_signal"]] if expected["ml_signal"] else Signal.HOLD
    recorded_regime = Regime[expected["regime"]] if expected["regime"] else Regime.BULL
    recorded_source = expected["signal_source"]
    close_price = float(expected["close_price"])

    broker = _StatefulBroker(
        initial_positions=broker_state["positions"],
        cash=float(broker_state["account"]["cash"]),
        close_price=close_price,
    )

    from schroeder_trader.basket.main import run_basket_pipeline

    now = datetime.fromisoformat(f"{fixture_date}T20:25:00+00:00")

    with patch("schroeder_trader.basket.orchestrator._compute_signal_for_ticker") as mock_sig, \
         patch("schroeder_trader.basket.main.get_position",
               side_effect=lambda t: broker.get_position(t)), \
         patch("schroeder_trader.basket.main.get_account",
               side_effect=lambda: broker.get_account()), \
         patch("schroeder_trader.basket.rebalance.submit_order",
               side_effect=lambda req, ticker: broker.submit_order(req, ticker)), \
         patch("schroeder_trader.basket.main.reconcile_orders", return_value=[]):
        mock_sig.return_value = (
            recorded_signal,
            recorded_source,
            recorded_regime,
            0,
            {
                "close": close_price,
                "pred_class": 2,
                "proba_json": "{}",
                "sma_signal": expected.get("sma_signal", "HOLD") or "HOLD",
            },
        )
        run_basket_pipeline(tmp_path, weights={"SPY": 1.0}, now=now)

    pf = pd.read_csv(tmp_path / "portfolio.csv")
    basket_rows = pf[pf["pipeline"] == "basket"]

    assert len(basket_rows) == 1, \
        f"Expected 1 basket row, got {len(basket_rows)}"
    assert basket_rows.iloc[0]["ticker"] == "SPY"

    assert int(basket_rows.iloc[0]["position_qty"]) == expected["expected_position_qty"], (
        f"Position mismatch on {fixture_date}: "
        f"basket got {basket_rows.iloc[0]['position_qty']}, "
        f"SPY-only had {expected['expected_position_qty']}"
    )

    assert abs(float(basket_rows.iloc[0]["total_value"]) - expected["expected_total_value"]) < 1.0, (
        f"Total value mismatch on {fixture_date}: "
        f"basket got {basket_rows.iloc[0]['total_value']}, "
        f"SPY-only had {expected['expected_total_value']}"
    )
