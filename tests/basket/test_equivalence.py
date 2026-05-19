"""Equivalence acceptance gate: basket pipeline with weights={SPY: 1.0}
must produce SPY position_qty and total_value matching the SPY-only
pipeline's same-day output, across three historical fixture days
representing different signal/regime states.

The basket pipeline's signal computation is mocked to return the exact
same composite signal the SPY-only pipeline recorded on the fixture day.
The broker is mocked: get_position returns the morning position; get_account
returns end-of-day values (from eod_account in broker_state.json). The
basket then runs its rebalance + snapshot path, and we assert the resulting
basket SPY row matches the historical SPY-only row within $1 of total_value
and exactly on position_qty.

Fixture days chosen for signal/regime variety:
- 2026-04-02: BEAR regime, FLAT signal_source, position=0 (stays flat)
- 2026-04-28: CHOPPY regime, XGB signal_source, BUY signal, position=141
- 2026-05-13: BULL regime, SMA signal_source, HOLD signal, position=141
  (shadow_signals.csv seeded with prior basket BUY row so prior_exposure=1.0)

close_proxy for invested days uses portfolio_value/position_qty so the
rebalancer computes zero diff and submits no orders (HOLD behaviour).

Allowed differences (documented):
- pipeline column: 'basket' vs 'spy_only'
- Trailing stop starts at HWM=0 for fixture days with no prior basket
  history. No fixture day has a stop fire, so this is benign.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


_FIXTURES = Path(__file__).parent / "fixtures"

FIXTURE_DAYS = sorted(d.name for d in _FIXTURES.iterdir() if d.is_dir()) if _FIXTURES.exists() else []


@pytest.mark.parametrize("fixture_date", FIXTURE_DAYS)
def test_basket_with_spy_only_weight_matches_spy_only_pipeline(
    fixture_date, tmp_path,
):
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
    recorded_source = "SMA" if recorded_regime == Regime.BULL else "XGB"

    # close_proxy: for invested days use portfolio_value/qty to produce zero diff
    # in the rebalancer; for flat days use a reasonable fallback.
    morning_qty = broker_state["positions"].get("SPY", 0)
    morning_pv = broker_state["account"]["portfolio_value"]
    if morning_qty > 0:
        close_proxy = morning_pv / morning_qty
    else:
        close_proxy = 700.0

    # EOD account values for snapshot step
    eod_account = broker_state.get("eod_account", broker_state["account"])

    from schroeder_trader.basket.main import run_basket_pipeline

    now = datetime.fromisoformat(f"{fixture_date}T20:25:00+00:00")

    with patch("schroeder_trader.basket.orchestrator._compute_signal_for_ticker") as mock_sig, \
         patch("schroeder_trader.basket.main.get_position",
               side_effect=lambda t: broker_state["positions"].get(t, 0)), \
         patch("schroeder_trader.basket.main.get_account",
               return_value=eod_account), \
         patch("schroeder_trader.basket.rebalance.submit_order",
               return_value=MagicMock(alpaca_order_id="test", status="SUBMITTED")), \
         patch("schroeder_trader.basket.main.reconcile_orders", return_value=[]):
        mock_sig.return_value = (
            recorded_signal,
            recorded_source,
            recorded_regime,
            0,
            {
                "close": close_proxy,
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
