# Fractional Kelly Position Sizing Design

**Date:** 2026-03-21
**Phase:** 4, Track 3 (Risk Management)
**Status:** Approved

## Summary

Add fractional Kelly position sizing to the composite trading system. Kelly sizing uses XGBoost probability outputs to scale position size based on model confidence, replacing binary (100% or 0%) allocation when XGBoost is the signal source. SMA and FLAT signal sources remain binary.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Kelly fraction | Configurable, default half-Kelly (0.5) | Can tune during shadow period |
| Signal sources | XGB-only | Only XGB provides real-time probabilities |
| Win/loss ratio (b) | Fixed from backtest | Solid walk-forward data available; stable 20-day horizon |
| Position cap | 1 - CASH_BUFFER_PCT (98%) of portfolio | Half-Kelly is naturally conservative and will rarely approach this cap; trailing stop handles catastrophic risk |
| Validation | Backtest + shadow logging | No live execution changes yet |

## Kelly Formula

Standard Kelly criterion adapted for 3-class XGBoost output:

```
p = P(UP)               # probability of winning trade
q = P(DOWN)              # probability of losing trade
b = avg_win / avg_loss   # payoff ratio, derived from backtest (must be > 0)

kelly_pct = (p * b - q) / b
fractional_kelly = kelly_pct * KELLY_MULTIPLIER
target_pct = max(0.0, fractional_kelly)
```

**3-class treatment:** FLAT outcomes are modeled as a "push" — principal returned with zero P&L. Since the 20-day horizon classifies FLAT as returns within +/-1% of zero, this is a reasonable approximation. When P(FLAT) is high, both p and q shrink, pushing Kelly toward 0 (small or no position).

**Negative Kelly:** When q > p*b (model predicts losses more likely than gains), the formula produces a negative value. This is clamped to 0 (no position). The unclamped value is still logged for analysis — persistently negative Kelly would indicate the model is predicting the wrong direction.

No shorting: Kelly is clamped to >= 0.
No leverage: position size capped at available capital after cash buffer.

## New Module: `risk/kelly.py`

Single pure function:

```python
def kelly_fraction(
    p_up: float,
    p_down: float,
    win_loss_ratio: float,
    kelly_multiplier: float = 0.5,
) -> float:
    """Compute fractional Kelly position size.

    Args:
        p_up: XGBoost P(UP) — probability of a winning trade.
        p_down: XGBoost P(DOWN) — probability of a losing trade.
        win_loss_ratio: Average win / average loss from backtest. Must be > 0.
        kelly_multiplier: Fractional Kelly safety factor (0.5 = half-Kelly).

    Returns:
        Target position as fraction of available capital [0.0, 1.0].

    Raises:
        ValueError: If win_loss_ratio <= 0.
    """
```

And a helper to convert fraction to share quantity:

```python
def kelly_qty(
    kelly_frac: float,
    portfolio_value: float,
    close_price: float,
    cash_buffer_pct: float = 0.02,
) -> int:
    """Convert Kelly fraction to whole share count."""
```

## Config Additions

Two new values in `config.py`:

```python
KELLY_MULTIPLIER = 0.5       # half-Kelly default
KELLY_WIN_LOSS_RATIO = TBD   # derived from backtest (step 1 of implementation)
```

**Known limitation:** `KELLY_WIN_LOSS_RATIO` is a static value derived from backtest. If market regime shifts change the actual win/loss ratio, this value becomes stale. Acceptable for Phase 4 shadow testing; can be made adaptive in a future iteration.

## Database Changes

Two new columns on `shadow_signals` table:

| Column | Type | Description |
|--------|------|-------------|
| `kelly_fraction` | REAL | Unclamped Kelly % (may be negative; NULL when source != XGB) |
| `kelly_qty` | INTEGER | Number of shares Kelly would size after clamping (NULL when source != XGB) |

**Migration:** Follow the existing defensive ALTER TABLE pattern in `trade_log.py` `init_db`:

```python
# Add to the existing migration loop (lines 62-66):
for col, col_type in [("kelly_fraction", "REAL"), ("kelly_qty", "INTEGER")]:
    try:
        conn.execute(f"ALTER TABLE shadow_signals ADD COLUMN {col} {col_type}")
    except sqlite3.OperationalError:
        pass  # column already exists
```

Also update the CREATE TABLE statement to include the new columns for fresh databases.

**`log_shadow_signal` changes:** Add `kelly_fraction: float | None = None` and `kelly_qty: int | None = None` parameters, and include them in the INSERT statement.

## Integration Points

### 1. Shadow Pipeline (main.py Step 10)

When `signal_source == "XGB"` (regardless of whether the composite signal is BUY, SELL, or HOLD):
- Extract P(UP) and P(DOWN) using the existing `class_to_idx` mapping (`proba[idx_up]`, `proba[idx_down]`) — do not use naive array indexing
- Call `kelly_fraction()` to compute target allocation
- Call `kelly_qty()` with current portfolio value and close price
- Log both values to `shadow_signals` table

Kelly is logged even when the composite signal is HOLD (e.g., XGB confidence below threshold). This allows analysis of what Kelly would have sized when the threshold filter suppressed the signal.

When `signal_source != "XGB"`:
- Set both columns to NULL

### 2. Backtest Validation (`backtest/run_backtest_kelly.py`)

New script that:
1. Replays walk-forward composite signals from existing backtest infrastructure
2. Extracts per-day P(UP)/P(DOWN) from XGBoost predictions
3. Computes win/loss ratio (b) from walk-forward results — this becomes `KELLY_WIN_LOSS_RATIO`
4. Simulates Kelly-sized positions vs binary baseline
5. Reports comparative metrics: Sharpe, max drawdown, total return, trade count

### 3. Live Execution (NOT in scope)

No changes to `risk_manager.py` or `execution/broker.py`. The existing binary `evaluate()` function continues to drive live SMA trading. Kelly stays in shadow until forward-test validation is complete.

## What's NOT Included

- Shorting (Kelly clamped >= 0)
- Leverage (capped at available capital)
- Kelly for SMA or FLAT signals
- Trailing stop / circuit breaker (separate Track 3 item)
- Volatility scaling (future enhancement)
- Changes to live order execution

## Success Criteria

1. Backtest Kelly Sharpe >= binary Sharpe (0.94 full-period)
2. Backtest Kelly max drawdown <= binary max drawdown (16.1%)
3. Shadow pipeline logs kelly_fraction and kelly_qty without errors
4. Win/loss ratio (b) derived and validated from walk-forward data
