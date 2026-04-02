# Portfolio Trailing Stop — Design Spec

## Overview

Add a portfolio-level trailing stop that tracks the high-water mark of the portfolio. If the portfolio drops 8% from its peak, liquidate all positions and enter a 5 trading day cooldown before allowing new entries. This is a catastrophic risk circuit breaker that sits above the composite signal routing.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| What to track | Portfolio value (not SPY price) | Accounts for position sizing; what actually matters for real money |
| Drawdown threshold | 8% | Conservative for a single-asset system still proving itself live |
| Post-trigger behavior | Cooldown (5 trading days) | Prevents whipsaw re-entry without requiring manual intervention |
| Cooldown length | 5 trading days | Short enough to catch recovery, long enough to avoid immediate re-entry into decline |

## New Module: `src/schroeder_trader/risk/trailing_stop.py`

### `TrailingStop` class

```python
class TrailingStop:
    def __init__(self, drawdown_pct: float, cooldown_days: int, high_water_mark: float = 0.0, stop_date: date | None = None):
        ...

    def update(self, portfolio_value: float, current_date: date) -> bool:
        """Update high-water mark. Return True if stop is triggered."""

    def in_cooldown(self, current_date: date, trading_dates: list[date]) -> bool:
        """Return True if still within cooldown period after a stop trigger."""

    def reset(self) -> None:
        """Clear all state. For testing or manual override."""
```

**Behavior:**
- `update()` sets `high_water_mark = max(high_water_mark, portfolio_value)`. If `portfolio_value < high_water_mark * (1 - drawdown_pct)`, trigger the stop and record `stop_date`.
- `in_cooldown()` counts trading days since `stop_date`. Returns True if fewer than `cooldown_days` trading days have elapsed. Uses the list of actual trading dates (from shadow_signals timestamps) rather than calendar days.
- After cooldown expires, `high_water_mark` resets to current portfolio value to start tracking fresh.

## New Config Values

```python
TRAILING_STOP_PCT = 0.08          # 8% drawdown triggers stop
TRAILING_STOP_COOLDOWN_DAYS = 5   # 5 trading days before re-entry
```

## Integration

### Shadow Pipeline (`main.py`)

Log trailing stop state in shadow signals. New columns in `shadow_signals`:
- `high_water_mark` (REAL) — current HWM at time of signal
- `trailing_stop_triggered` (INTEGER, 0/1) — whether stop fired today

On pipeline startup, initialize `TrailingStop` by reading the most recent `high_water_mark` from the database. This provides state persistence between daily runs.

### Risk Manager

Add a trailing stop check to `evaluate()`:
- If trailing stop is triggered or in cooldown, override BUY signals to return None (no action).
- SELL signals pass through unchanged (allow exits during cooldown).
- This is additive — the existing signal-based logic remains unchanged.

### Backtest

Add trailing stop logic to the backtest runner to measure impact:
- Compare Sharpe ratio, max drawdown, and total return with and without trailing stop.
- Count how many times the stop would have triggered historically.
- Measure average cooldown cost (missed gains during cooldown periods).

## Database Schema Change

```sql
ALTER TABLE shadow_signals ADD COLUMN high_water_mark REAL;
ALTER TABLE shadow_signals ADD COLUMN trailing_stop_triggered INTEGER DEFAULT 0;
```

Use the same migration pattern as existing column additions in `init_db()`.

## State Persistence

The `TrailingStop` needs to survive between daily pipeline runs:
1. On startup: query `SELECT MAX(high_water_mark) FROM shadow_signals` and the most recent `trailing_stop_triggered` + `timestamp` to reconstruct state.
2. On each run: log updated HWM and trigger status to shadow_signals.

## What This Does NOT Include

- Per-position stops (only portfolio-level)
- Manual re-enable requirement (automatic after cooldown)
- Changes to live order execution (shadow logging only for now)
- Volatility-scaled stop levels (fixed 8% regardless of regime)
