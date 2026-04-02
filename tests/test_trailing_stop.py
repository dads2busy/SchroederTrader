from datetime import date

import pytest

from schroeder_trader.risk.trailing_stop import TrailingStop


def test_update_sets_high_water_mark():
    ts = TrailingStop(drawdown_pct=0.08, cooldown_days=5)
    triggered = ts.update(100000.0, date(2026, 1, 2))
    assert ts.high_water_mark == 100000.0
    assert triggered is False


def test_update_raises_hwm():
    ts = TrailingStop(drawdown_pct=0.08, cooldown_days=5)
    ts.update(100000.0, date(2026, 1, 2))
    ts.update(105000.0, date(2026, 1, 3))
    assert ts.high_water_mark == 105000.0


def test_update_does_not_lower_hwm():
    ts = TrailingStop(drawdown_pct=0.08, cooldown_days=5)
    ts.update(100000.0, date(2026, 1, 2))
    ts.update(99000.0, date(2026, 1, 3))
    assert ts.high_water_mark == 100000.0


def test_triggers_at_threshold():
    ts = TrailingStop(drawdown_pct=0.08, cooldown_days=5)
    ts.update(100000.0, date(2026, 1, 2))
    # 8% drop: 100000 * 0.92 = 92000
    triggered = ts.update(91999.0, date(2026, 1, 3))
    assert triggered is True
    assert ts.stop_date == date(2026, 1, 3)


def test_does_not_trigger_above_threshold():
    ts = TrailingStop(drawdown_pct=0.08, cooldown_days=5)
    ts.update(100000.0, date(2026, 1, 2))
    triggered = ts.update(92001.0, date(2026, 1, 3))
    assert triggered is False
    assert ts.stop_date is None


def test_in_cooldown_during_period():
    ts = TrailingStop(drawdown_pct=0.08, cooldown_days=5)
    ts.update(100000.0, date(2026, 1, 2))
    ts.update(91000.0, date(2026, 1, 3))  # trigger
    trading_dates = [
        date(2026, 1, 3), date(2026, 1, 6), date(2026, 1, 7),
        date(2026, 1, 8), date(2026, 1, 9), date(2026, 1, 10),
    ]
    assert ts.in_cooldown(date(2026, 1, 6), trading_dates) is True  # day 2
    assert ts.in_cooldown(date(2026, 1, 9), trading_dates) is True  # day 5


def test_cooldown_expires():
    ts = TrailingStop(drawdown_pct=0.08, cooldown_days=5)
    ts.update(100000.0, date(2026, 1, 2))
    ts.update(91000.0, date(2026, 1, 3))  # trigger
    trading_dates = [
        date(2026, 1, 3), date(2026, 1, 6), date(2026, 1, 7),
        date(2026, 1, 8), date(2026, 1, 9), date(2026, 1, 10),
        date(2026, 1, 13),
    ]
    assert ts.in_cooldown(date(2026, 1, 13), trading_dates) is False  # day 6


def test_no_cooldown_when_not_triggered():
    ts = TrailingStop(drawdown_pct=0.08, cooldown_days=5)
    ts.update(100000.0, date(2026, 1, 2))
    assert ts.in_cooldown(date(2026, 1, 3), [date(2026, 1, 2), date(2026, 1, 3)]) is False


def test_reset_clears_state():
    ts = TrailingStop(drawdown_pct=0.08, cooldown_days=5)
    ts.update(100000.0, date(2026, 1, 2))
    ts.update(91000.0, date(2026, 1, 3))
    ts.reset()
    assert ts.high_water_mark == 0.0
    assert ts.stop_date is None


def test_hwm_resets_after_cooldown():
    """After cooldown expires, HWM resets to current portfolio value."""
    ts = TrailingStop(drawdown_pct=0.08, cooldown_days=5)
    ts.update(100000.0, date(2026, 1, 2))
    ts.update(91000.0, date(2026, 1, 3))  # trigger at 91k
    trading_dates = [
        date(2026, 1, 3), date(2026, 1, 6), date(2026, 1, 7),
        date(2026, 1, 8), date(2026, 1, 9), date(2026, 1, 10),
        date(2026, 1, 13),
    ]
    # Cooldown expired
    assert ts.in_cooldown(date(2026, 1, 13), trading_dates) is False
    # Next update with trading_dates should reset HWM to current value
    ts.update(93000.0, date(2026, 1, 13), trading_dates=trading_dates)
    assert ts.high_water_mark == 93000.0
    assert ts.stop_date is None


def test_update_during_cooldown_preserves_state():
    """Calling update() during cooldown does NOT reset state."""
    ts = TrailingStop(drawdown_pct=0.08, cooldown_days=5)
    ts.update(100000.0, date(2026, 1, 2))
    ts.update(91000.0, date(2026, 1, 3))  # trigger
    trading_dates = [
        date(2026, 1, 3), date(2026, 1, 6), date(2026, 1, 7),
    ]
    # Day 2 of cooldown — update should return False and preserve stop_date
    result = ts.update(95000.0, date(2026, 1, 6), trading_dates=trading_dates)
    assert result is False
    assert ts.stop_date == date(2026, 1, 3)  # preserved, not cleared
    assert ts.high_water_mark == 100000.0  # preserved, not reset


def test_update_without_trading_dates_preserves_state():
    """Calling update() without trading_dates after trigger is safe (no reset)."""
    ts = TrailingStop(drawdown_pct=0.08, cooldown_days=5)
    ts.update(100000.0, date(2026, 1, 2))
    ts.update(91000.0, date(2026, 1, 3))  # trigger
    # No trading_dates — should preserve state (safe default)
    result = ts.update(95000.0, date(2026, 1, 6))
    assert result is False
    assert ts.stop_date == date(2026, 1, 3)


def test_init_with_existing_state():
    """Can initialize with pre-existing HWM and stop date."""
    ts = TrailingStop(
        drawdown_pct=0.08, cooldown_days=5,
        high_water_mark=100000.0, stop_date=date(2026, 1, 3),
    )
    assert ts.high_water_mark == 100000.0
    assert ts.stop_date == date(2026, 1, 3)
