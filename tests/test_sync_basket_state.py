"""Tests for the basket-state sync helper (scripts/sync_basket_state.py)."""
import importlib.util
from pathlib import Path

import pandas as pd

_spec = importlib.util.spec_from_file_location(
    "sync_basket_state",
    Path(__file__).resolve().parent.parent / "scripts" / "sync_basket_state.py",
)
sync_basket_state = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sync_basket_state)
merge_basket_rows = sync_basket_state.merge_basket_rows


def _row(pipeline, ticker, ts, total=100.0):
    return {"timestamp": ts, "pipeline": pipeline, "ticker": ticker, "total_value": total}


def test_merge_replaces_stale_basket_and_preserves_spy_only():
    # Local: full spy_only history (incl. fresh 6/03) + STALE basket (5/19 only)
    local = pd.DataFrame([
        _row("spy_only", "SPY", "2026-05-18T20:30:00+00:00"),
        _row("spy_only", "SPY", "2026-06-03T20:30:00+00:00"),  # fresh local write
        _row("basket", "SPY", "2026-05-19T20:38:00+00:00"),    # stale basket
    ])
    # Origin: spy_only frozen at 5/18 + FRESH basket through 6/02
    origin = pd.DataFrame([
        _row("spy_only", "SPY", "2026-05-18T20:30:00+00:00"),
        _row("basket", "SPY", "2026-05-19T20:38:00+00:00"),
        _row("basket", "SPY", "2026-06-02T22:11:00+00:00"),    # fresh basket
    ])

    merged = merge_basket_rows(local, origin)

    spy = merged[merged["pipeline"] == "spy_only"]["timestamp"].tolist()
    bkt = merged[merged["pipeline"] == "basket"]["timestamp"].tolist()
    # Local spy_only rows preserved, including the fresh 6/03 write
    assert "2026-06-03T20:30:00+00:00" in spy
    assert len(spy) == 2
    # Basket replaced with origin's fresh set (latest 6/02 present)
    assert "2026-06-02T22:11:00+00:00" in bkt
    assert len(bkt) == 2  # not 3 — the stale local basket row was dropped, not duplicated


def test_merge_noop_when_origin_has_no_basket():
    local = pd.DataFrame([_row("spy_only", "SPY", "2026-06-03T20:30:00+00:00")])
    origin = pd.DataFrame([_row("spy_only", "SPY", "2026-05-18T20:30:00+00:00")])
    assert merge_basket_rows(local, origin).equals(local)


def test_merge_noop_when_no_pipeline_column():
    local = pd.DataFrame([{"timestamp": "x", "ticker": "SPY"}])
    origin = pd.DataFrame([{"timestamp": "y", "ticker": "SPY"}])
    assert merge_basket_rows(local, origin).equals(local)
