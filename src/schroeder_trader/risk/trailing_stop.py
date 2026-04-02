from __future__ import annotations

from datetime import date


class TrailingStop:
    """Portfolio-level trailing stop with cooldown.

    Tracks the high-water mark of portfolio value. Triggers when the
    portfolio drops below ``high_water_mark * (1 - drawdown_pct)``.
    After triggering, enters a cooldown period of ``cooldown_days``
    trading days before allowing new entries.
    """

    def __init__(
        self,
        drawdown_pct: float,
        cooldown_days: int,
        high_water_mark: float = 0.0,
        stop_date: date | None = None,
    ) -> None:
        self.drawdown_pct = drawdown_pct
        self.cooldown_days = cooldown_days
        self.high_water_mark = high_water_mark
        self.stop_date = stop_date

    def update(
        self, portfolio_value: float, current_date: date,
        trading_dates: list[date] | None = None,
    ) -> bool:
        """Update high-water mark and check for stop trigger.

        Args:
            portfolio_value: Current portfolio value.
            current_date: Today's date.
            trading_dates: List of past trading dates, needed to check
                cooldown expiry. If None and a stop was previously
                triggered, the stop state is preserved (safe default).

        Returns True if the stop is triggered on this update.
        """
        # If previously triggered, only reset after cooldown expires
        if self.stop_date is not None:
            if trading_dates is not None and not self.in_cooldown(current_date, trading_dates):
                self.high_water_mark = 0.0
                self.stop_date = None
            else:
                # Still in cooldown (or can't verify) — don't update HWM
                return False

        self.high_water_mark = max(self.high_water_mark, portfolio_value)

        threshold = self.high_water_mark * (1 - self.drawdown_pct)
        if portfolio_value < threshold:
            self.stop_date = current_date
            return True

        return False

    def in_cooldown(self, current_date: date, trading_dates: list[date]) -> bool:
        """Check if still in cooldown period after a stop trigger.

        Args:
            current_date: Today's date.
            trading_dates: List of actual trading dates (from DB timestamps).

        Returns True if fewer than ``cooldown_days`` trading days have
        elapsed since the stop triggered.
        """
        if self.stop_date is None:
            return False

        # Count trading days after stop_date up to and including current_date
        days_after = sum(
            1 for d in trading_dates
            if self.stop_date < d <= current_date
        )
        return days_after < self.cooldown_days

    def reset(self) -> None:
        """Clear all state."""
        self.high_water_mark = 0.0
        self.stop_date = None
