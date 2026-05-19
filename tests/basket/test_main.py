from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from schroeder_trader.basket.main import run_basket_pipeline


def _seed_initial_state(tmp_path: Path, spy_only_total: float = 105000.0):
    """SPY-only history exists; basket has never run."""
    pd.DataFrame({
        "id": [1], "timestamp": ["2026-05-20T20:30:00+00:00"],
        "pipeline": ["spy_only"], "ticker": ["SPY"],
        "cash": [1965.4], "position_qty": [141],
        "position_value": [spy_only_total - 1965.4],
        "total_value": [spy_only_total],
    }).to_csv(tmp_path / "portfolio.csv", index=False)

    pd.DataFrame(columns=[
        "id", "signal_id", "alpaca_order_id", "timestamp", "pipeline",
        "ticker", "action", "quantity", "fill_price", "fill_timestamp",
        "status", "signal_close_price", "slippage",
    ]).to_csv(tmp_path / "orders.csv", index=False)

    pd.DataFrame(columns=[
        "id", "timestamp", "pipeline", "ticker", "close_price",
        "predicted_class", "predicted_proba", "ml_signal", "sma_signal",
        "regime", "signal_source", "bear_day_count",
        "kelly_fraction", "kelly_qty", "high_water_mark",
        "trailing_stop_triggered",
    ]).to_csv(tmp_path / "shadow_signals.csv", index=False)


@patch("schroeder_trader.basket.main.subprocess.run")
@patch("schroeder_trader.basket.orchestrator._compute_signal_for_ticker")
def test_run_basket_pipeline_writes_per_ticker_rows_to_portfolio_csv(
    mock_sig, mock_subproc, tmp_path,
):
    from schroeder_trader.strategy.composite import Signal
    from schroeder_trader.strategy.regime_detector import Regime

    _seed_initial_state(tmp_path)
    mock_sig.return_value = (
        Signal.BUY, "SMA", Regime.BULL, 0,
        {"close": 100.0, "pred_class": 2, "proba_json": "{}", "sma_signal": "BUY"},
    )
    mock_subproc.return_value = None  # feature download is a no-op

    weights = {"SPY": 0.45, "XLK": 0.30, "XLV": 0.15, "XLE": 0.10}
    now = datetime(2026, 5, 21, 20, 20, tzinfo=timezone.utc)
    run_basket_pipeline(tmp_path, weights, now)

    pf = pd.read_csv(tmp_path / "portfolio.csv")
    basket_rows = pf[pf["pipeline"] == "basket"]
    assert set(basket_rows["ticker"]) == set(weights.keys())
    assert len(basket_rows) == 4

    # Total value should equal starting capital ($105K) within rounding
    assert abs(float(basket_rows.iloc[0]["total_value"]) - 105000.0) < 1000.0
