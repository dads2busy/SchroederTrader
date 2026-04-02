# Volatility-Scaled Transaction Costs — Design Spec

## Overview

Replace the fixed 5 bps slippage estimate with a VIX-based tiered cost model. This makes backtests more realistic by accounting for wider spreads during volatile markets and tighter spreads during calm periods. Backtest-only change — no impact on live pipeline.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Where to apply | Backtest only | Live execution uses real Alpaca costs; shadow pipeline doesn't need it |
| Cost model | VIX-tiered | VIX correlates with bid-ask spreads; already in our feature data |
| Market impact | Not modeled | SPY too liquid, positions too small (~22 shares) |
| Fixed slippage config | Remove entirely | Was always a placeholder; one cost model avoids confusion |

## New Module: `src/schroeder_trader/risk/transaction_cost.py`

```python
def estimate_slippage(vix: float) -> float:
    """Estimate transaction cost based on VIX level.

    Returns slippage as a fraction of trade value.
    """
```

### VIX Tiers

| VIX Range | Slippage | Label |
|-----------|----------|-------|
| < 15 | 0.0003 (3 bps) | Low vol — tight spreads |
| 15 to 25 | 0.0005 (5 bps) | Normal conditions |
| 25 to 35 | 0.0010 (10 bps) | Elevated vol — wider spreads |
| > 35 | 0.0015 (15 bps) | Crisis — wide spreads, thin books |

Thresholds are exclusive on the lower bound: VIX of exactly 15 falls in the 15-25 tier, exactly 25 falls in the 25-35 tier, exactly 35 falls in the >35 tier.

## Config Change

Remove `SLIPPAGE_ESTIMATE = 0.0005` from `src/schroeder_trader/config.py`.

## Backtest Script Changes

Three files need updating:

### `backtest/train_model.py`
- Remove `SLIPPAGE_ESTIMATE` import
- Load VIX data from features CSV
- On each trade (position change), call `estimate_slippage(vix_on_trade_day)` instead of using the fixed constant

### `backtest/compare_models.py`
- Remove `SLIPPAGE_ESTIMATE` import
- Load VIX data
- Use `estimate_slippage(vix)` per trade

### `backtest/run_backtest.py`
- Remove `SLIPPAGE_ESTIMATE` import
- This script uses `vbt.Portfolio.from_signals(fees=...)` which expects a scalar or array
- Pass a per-day fee array computed from VIX using `estimate_slippage()`

## VIX Data Source

VIX close prices are already in `backtest/data/features_daily.csv` as the `vix_close` column. No new data download needed.

## What This Does NOT Include

- No shadow pipeline changes
- No Kelly sizing adjustment for costs
- No market impact modeling
- No continuous cost function (tiers are simpler and more interpretable)
- No changes to live order execution
