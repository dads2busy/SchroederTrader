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

    def update(self, portfolio_value: float, current_date: date) -> bool:
        """Update high-water mark and check for stop trigger.

        Returns True if the stop is triggered on this update.
        """
        # If we previously triggered and cooldown has expired, reset
        if self.stop_date is not None and self.high_water_mark > 0:
            # Reset HWM so it starts fresh from current value
            self.high_water_mark = 0.0
            self.stop_date = None

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
