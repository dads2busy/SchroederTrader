import shutil
from pathlib import Path
import pandas as pd
import pytest

from scripts.migrate_portfolio_to_pipeline_column import migrate


def _seed_csv(path: Path, df: pd.DataFrame) -> None:
    df.to_csv(path, index=False)


def test_migration_adds_pipeline_column_to_portfolio_csv(tmp_path):
    pf = pd.DataFrame({
        "id": [1, 2],
        "timestamp": ["2026-04-15T20:30:00+00:00", "2026-04-16T20:30:00+00:00"],
        "cash": [1965.4, 1965.4],
        "position_qty": [141, 141],
        "position_value": [100000.0, 100500.0],
        "total_value": [101965.4, 102465.4],
    })
    _seed_csv(tmp_path / "portfolio.csv", pf)
    _seed_csv(tmp_path / "orders.csv", pd.DataFrame(columns=[
        "id", "signal_id", "alpaca_order_id", "timestamp", "ticker",
        "action", "quantity", "fill_price", "fill_timestamp", "status",
        "signal_close_price", "slippage",
    ]))
    _seed_csv(tmp_path / "shadow_signals.csv", pd.DataFrame(columns=[
        "id", "timestamp", "ticker", "close_price", "predicted_class",
        "predicted_proba", "ml_signal", "sma_signal", "regime",
        "signal_source", "bear_day_count", "kelly_fraction", "kelly_qty",
        "high_water_mark", "trailing_stop_triggered",
    ]))

    migrate(tmp_path)
    out = pd.read_csv(tmp_path / "portfolio.csv")
    assert "pipeline" in out.columns
    assert "ticker" in out.columns
    assert (out["pipeline"] == "spy_only").all()
    assert (out["ticker"] == "SPY").all()


def test_migration_preserves_existing_rows_bit_identical(tmp_path):
    """Property test: every original column byte-identical after filtering."""
    pf_pre = pd.DataFrame({
        "id": [1, 2, 3],
        "timestamp": ["2026-04-15T20:30:00+00:00",
                       "2026-04-16T20:30:00+00:00",
                       "2026-04-17T20:30:00+00:00"],
        "cash": [1965.4, 1965.4, 1965.4],
        "position_qty": [141, 141, 141],
        "position_value": [100000.0, 100500.0, 101000.0],
        "total_value": [101965.4, 102465.4, 102965.4],
    })
    _seed_csv(tmp_path / "portfolio.csv", pf_pre)
    _seed_csv(tmp_path / "orders.csv", pd.DataFrame(columns=[
        "id", "signal_id", "alpaca_order_id", "timestamp", "ticker",
        "action", "quantity", "fill_price", "fill_timestamp", "status",
        "signal_close_price", "slippage",
    ]))
    _seed_csv(tmp_path / "shadow_signals.csv", pd.DataFrame(columns=[
        "id", "timestamp", "ticker", "close_price", "predicted_class",
        "predicted_proba", "ml_signal", "sma_signal", "regime",
        "signal_source", "bear_day_count", "kelly_fraction", "kelly_qty",
        "high_water_mark", "trailing_stop_triggered",
    ]))

    migrate(tmp_path)
    pf_post = pd.read_csv(tmp_path / "portfolio.csv")
    pf_post_spy = pf_post[pf_post["pipeline"] == "spy_only"]
    for col in pf_pre.columns:
        assert (pf_pre[col].astype(str).values == pf_post_spy[col].astype(str).values).all(), col


def test_migration_is_idempotent(tmp_path):
    pf = pd.DataFrame({
        "id": [1], "timestamp": ["2026-04-15T20:30:00+00:00"],
        "cash": [1965.4], "position_qty": [141],
        "position_value": [100000.0], "total_value": [101965.4],
    })
    _seed_csv(tmp_path / "portfolio.csv", pf)
    _seed_csv(tmp_path / "orders.csv", pd.DataFrame(columns=[
        "id", "signal_id", "alpaca_order_id", "timestamp", "ticker",
        "action", "quantity", "fill_price", "fill_timestamp", "status",
        "signal_close_price", "slippage",
    ]))
    _seed_csv(tmp_path / "shadow_signals.csv", pd.DataFrame(columns=[
        "id", "timestamp", "ticker", "close_price", "predicted_class",
        "predicted_proba", "ml_signal", "sma_signal", "regime",
        "signal_source", "bear_day_count", "kelly_fraction", "kelly_qty",
        "high_water_mark", "trailing_stop_triggered",
    ]))

    migrate(tmp_path)
    first_pf = pd.read_csv(tmp_path / "portfolio.csv").to_csv(index=False)
    migrate(tmp_path)
    second_pf = pd.read_csv(tmp_path / "portfolio.csv").to_csv(index=False)
    assert first_pf == second_pf
