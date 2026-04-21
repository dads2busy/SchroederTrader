from datetime import date

import pandas as pd

import sys as _sys
from pathlib import Path
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import oracle_comparison as oc


def _make_spy(closes, start=date(2026, 1, 2)):
    dates = pd.bdate_range(start=start, periods=len(closes)).date
    return pd.Series(closes, index=list(dates), name="close")


def test_simulate_flat_exposure_no_change():
    spy = _make_spy([100.0, 101.0, 102.0, 103.0])
    values = oc.simulate(spy, target_by_date={}, start_value=100_000.0)
    assert values.iloc[0] == 100_000.0
    assert values.iloc[-1] == 100_000.0  # no exposure ever


def test_simulate_full_long_tracks_spy():
    spy = _make_spy([100.0, 110.0])
    # Target set on day 0 to 1.0, carries to day 1
    values = oc.simulate(spy, target_by_date={spy.index[0]: 1.0}, start_value=100_000.0)
    # 10% SPY move at 100% exposure = 10% portfolio move
    assert abs(values.iloc[-1] - 110_000.0) < 0.01


def test_simulate_half_exposure():
    spy = _make_spy([100.0, 110.0])
    values = oc.simulate(spy, target_by_date={spy.index[0]: 0.5}, start_value=100_000.0)
    assert abs(values.iloc[-1] - 105_000.0) < 0.01  # 10% × 0.5 = 5%


def test_simulate_exposure_change_midway():
    spy = _make_spy([100.0, 110.0, 99.0])  # +10% then -10%
    targets = {spy.index[0]: 1.0, spy.index[1]: 0.0}
    values = oc.simulate(spy, target_by_date=targets, start_value=100_000.0)
    # Day 0 → 1: full long, +10% → 110k
    # Day 1 → 2: flat (rebalanced at day-1 close), -10% SPY but 0 exposure → stays 110k
    assert abs(values.iloc[1] - 110_000.0) < 0.01
    assert abs(values.iloc[2] - 110_000.0) < 0.01


def test_simulate_carries_forward_missing_targets():
    spy = _make_spy([100.0, 110.0, 121.0])
    # Only day 0 has a target; day 1 should carry it forward
    values = oc.simulate(spy, target_by_date={spy.index[0]: 1.0}, start_value=100_000.0)
    # Full compounding: 100k → 110k → 121k
    assert abs(values.iloc[-1] - 121_000.0) < 0.01


def test_summary_stats_positive_return():
    series = pd.Series([100_000.0, 105_000.0, 110_000.0])
    stats = oc.summary_stats(series)
    assert abs(stats["total_return_pct"] - 10.0) < 0.01
    assert stats["max_dd_pct"] == 0.0  # monotonically rising


def test_summary_stats_drawdown():
    series = pd.Series([100_000.0, 120_000.0, 90_000.0, 100_000.0])
    stats = oc.summary_stats(series)
    # Peak 120, trough 90 → -25% drawdown
    assert abs(stats["max_dd_pct"] - (-25.0)) < 0.01
