# Basket Paper-Trading Pipeline — Design

**Date:** 2026-05-19
**Status:** Approved, ready for plan

## Problem

The system trades SPY-only with real (paper-broker) money. The long-term destination is a multi-ticker basket strategy (currently 45/30/15/10 across SPY/XLK/XLV/XLE, validated by 27-year backtest with walk-forward Sharpe 2.34). The existing trading pipeline is hardcoded to a single `TICKER` constant — no per-ticker orchestration, no weight-based sizing, no rebalancing, no per-ticker trailing stops.

We need the multi-ticker execution architecture in place so the basket strategy can be paper-traded end-to-end *alongside* the existing SPY-only pipeline. Paper-trading the basket is the only way to measure live-execution friction (slippage on smaller per-ticker orders, daily rebalance drag, multi-order fill timing) that the backtest's frictionless math doesn't expose. The architecture is being built now even though the switch from SPY-only to basket-as-production won't happen for months — the gating data accumulates faster than the architecture can be built, so building first avoids becoming the bottleneck later.

## Goal

Build a new `src/schroeder_trader/basket/` subpackage that runs a multi-ticker paper-trading pipeline alongside the existing SPY-only production pipeline. State files are shared (distinguished by a new `pipeline` column). The existing daily email is extended with a new `BASKET PAPER` section. The SPY-only pipeline's trade-execution behavior is untouched.

The headline acceptance gate: setting `BASKET_PAPER_WEIGHTS = {"SPY": 1.0}` and running only the basket pipeline must produce portfolio state byte-identical (or trivially close, with documented allowed differences) to what the SPY-only pipeline produces on the same day.

## Non-Goals

- No changes to the SPY-only pipeline's trade-execution behavior. The only SPY-only touches are: schema-aware logging (one column added, kwarg passed), and email-rendering reads from the new shared schema.
- No live-money basket trading. Paper mode only until retirement of SPY-only is explicitly approved.
- No MOC-cutover work. The basket runs at 21:20 UTC with market orders, mirroring the current SPY-only timing. MOC cutover for the basket is deferred under the existing MOC transition design.
- No new shadow tickers. The basket uses exactly the four tickers already trained (SPY, XLK, XLV, XLE).
- No drift-triggered or weekly rebalancing. Daily rebalancing only, matching the backtest's assumption.
- No Kelly sizing for the basket. Binary 0/1 exposure per ticker × configured weights × portfolio value. Matches production.
- No basket-level oracle integration (Claude/OpenAI LLM signals). Deferred; oracles remain SPY-only for now.
- No retirement of the SPY-only pipeline. Phase E criteria documented but execution is a separate later decision.

## Architectural Decisions

| Decision | Choice | Why |
|---|---|---|
| Topology | Separate `basket/` subpackage + separate cron entry (`daily-basket.yml`) | Strongest regression isolation. SPY-only trade-execution path literally untouched. Cleanest delete-later structure. |
| Rebalance frequency | Daily, every run | Matches the 27-year backtest's daily-rebalanced math exactly. Slippage costs already baked into the backtest's Sharpe via `risk/transaction_cost.py`. |
| Per-ticker trailing stop | Yes; reuses existing `risk/trailing_stop.py` with one instance per ticker | Matches the per-strategy independence the backtest assumed. |
| Cash handling on stop trigger | Cash sits idle, not redistributed to other tickers | Matches backtest's per-strategy independent equity curves. |
| State files | Shared CSVs with a new `pipeline` column (`'spy_only'` or `'basket'`) | Single-query cross-pipeline comparison. Long-format rows scale to more tickers without schema changes. Continuous data lineage across the eventual retirement. |
| Email layout | Single email, new `BASKET PAPER` section in the existing email between `PERFORMANCE` and `SECTOR SHADOW` | One email per day. Reuses existing infrastructure. |
| Cron timing | Basket at 21:20 UTC, SPY-only at 21:30 UTC (SPY-only renders the unified email last) | Basket writes state before SPY-only reads it for the email. 10-minute gap is ample. |
| Paper-mode starting capital | Mirror the SPY-only portfolio's `total_value` on first basket-pipeline run | Dollar P&L between the two pipelines is directly comparable in the email. |
| Order type | Market orders at 21:20 UTC (fills at next-day open) | Mirrors current SPY-only behavior. MOC cutover handled separately later. |
| Aggregate row in `portfolio.csv` | Not stored; computed on read | Storing a synthetic `ticker='TOTAL'` row conflates ticker holdings with derived sums. |

## File Layout

```
src/schroeder_trader/
  main.py                      # SPY-only (existing). Minimal changes:
                               #   - log_portfolio/log_order/log_shadow_signal calls add pipeline="spy_only"
                               #   - email rendering reads pipeline='basket' rows for the new section
                               #   - trade-execution code is unchanged
  basket/                      # NEW subpackage
    __init__.py
    main.py                    # entry point invoked by daily-basket.yml
    orchestrator.py            # per-ticker signal → exposure → trailing-stop loop
    rebalance.py               # current vs target → submit diff orders
    portfolio.py               # write per-ticker portfolio snapshot rows
  risk/
    trailing_stop.py           # existing; instantiated once per ticker by orchestrator
  reports/
    daily_email.py             # add build_basket_paper_section; build_email_body gains basket_state kwarg
  storage/
    trade_log.py               # log_portfolio, log_order, log_shadow_signal gain `pipeline` kwarg
data/
  portfolio.csv                # SCHEMA CHANGE: add pipeline, ticker columns
  orders.csv                   # SCHEMA CHANGE: add pipeline column (ticker already present)
  shadow_signals.csv           # SCHEMA CHANGE: add pipeline column (ticker already present)
  signals.csv                  # untouched (SMA log, SPY-only specific)
  trades.db                    # untouched
.github/workflows/
  daily.yml                    # existing — 21:30 UTC SPY-only
  daily-basket.yml             # NEW — 21:20 UTC basket paper-trading
scripts/
  migrate_portfolio_to_pipeline_column.py  # NEW — one-time migration
tests/
  basket/
    test_orchestrator.py
    test_rebalance.py
    test_portfolio.py
    test_trailing_stop.py
    test_equivalence.py        # SPY=100% equivalence test
  test_daily_email.py          # extended for basket section
  test_migration.py            # migration regression tests
backtest/                      # untouched
```

## State Schemas

After migration, all three shared CSVs carry a `pipeline` column:

### `portfolio.csv`

```
id, timestamp, pipeline, ticker, cash, position_qty, position_value, total_value
```

- **SPY-only rows:** `pipeline='spy_only', ticker='SPY', cash=portfolio cash, position_qty/position_value for SPY, total_value=portfolio total`. One row per run.
- **Basket rows:** `pipeline='basket', ticker=<each in basket>, cash=basket shared cash (repeated on every basket row), position_qty/position_value for that ticker, total_value=basket portfolio total (repeated)`. N rows per run (N = number of tickers in `SHADOW_BASKET_WEIGHTS`).
- **Aggregate row:** not stored; basket totals computed on read by filtering to the latest timestamp + `pipeline='basket'`, then taking `total_value` from any of the N rows (all identical) and summing per-ticker `position_value`s.

### `orders.csv`

```
id, timestamp, pipeline, ticker, action, qty, price, order_id, status
```

The existing schema already has a `ticker` column. Add `pipeline`. SPY-only rows backfill to `pipeline='spy_only'`.

### `shadow_signals.csv`

```
id, timestamp, pipeline, ticker, close_price, predicted_class, predicted_proba,
ml_signal, sma_signal, regime, signal_source, bear_day_count,
kelly_fraction, kelly_qty, high_water_mark, trailing_stop_triggered
```

Existing schema already has `ticker`. Add `pipeline`. SPY-only rows backfill to `pipeline='spy_only'`. The basket pipeline writes its own rows here with `pipeline='basket'` — same composite-signal logging it would produce in shadow mode.

## Daily Timeline

```
21:20 UTC  basket cron fires:
  1. Reconcile per-ticker orphaned orders.
  2. For each ticker in SHADOW_BASKET_WEIGHTS:
       fetch_daily_bars, compute composite signal,
       update per-ticker TrailingStop,
       log decision to shadow_signals.csv with pipeline='basket'.
  3. Compute target_value[t] = portfolio_value × weight[t] × exposure[t]
       where exposure is 0/1 from signal (HOLD carries prior),
       overridden to 0 if stop fired or in cooldown.
  4. For each ticker, submit market order for the share-diff between
       current and target positions. Skip if |diff| < one share's value.
       Log orders to orders.csv with pipeline='basket'.
  5. Read updated positions from Alpaca paper account.
  6. Write N per-ticker portfolio snapshot rows to portfolio.csv
       with pipeline='basket'. Exit. No email.

21:30 UTC  SPY-only cron fires (existing):
  1-N. Unchanged trade-execution path. Logs use pipeline='spy_only'.
  N+1. At email-rendering time:
       - Read portfolio.csv filtered to pipeline='spy_only' for existing sections.
       - Read portfolio.csv filtered to pipeline='basket' for BASKET PAPER section.
       - Render unified email.
       - Send single email.
```

### Edge cases

- **First basket run:** Bootstrap starting capital from the latest `pipeline='spy_only'` `total_value` row. All positions start at zero; rebalance buys them to targets.
- **Basket cron fails:** SPY-only still runs at 21:30. The BASKET PAPER section shows last-known basket state with a "stale data" tag (last `pipeline='basket'` timestamp visible).
- **SPY-only cron fails:** Basket state already written. No email sent that day. Next day picks up from state.
- **Weights changed in `config.py`:** Basket rebalances toward new weights on its next run. Drift correction does the work over a few days; no special transition logic.
- **Per-ticker stop fires:** Exposure forced to 0 for that ticker. Cash from the close stays in basket's shared cash pool (other tickers do NOT redistribute up). Cooldown advances by trading-date count; re-entry on the first run after cooldown ends if signal allows.

## Trade Execution Algorithm

Orchestrator pseudocode:

```python
def run_basket_pipeline(conn, broker, weights, now):
    # 1. Bootstrap portfolio value (prior basket row, or fall back to SPY-only).
    portfolio_value = read_starting_value(conn)

    decisions = {}
    for ticker, weight in weights.items():
        df = fetch_daily_bars(ticker, days=600)
        close = float(df["close"].iloc[-1])

        signal, source, regime = composite_signal_for_ticker(ticker, df, ext_df)

        stop = load_or_create_stop(conn, ticker, pipeline="basket")
        ticker_pos_value = read_position_value(conn, ticker, pipeline="basket")
        stop_triggered = stop.update(
            ticker_pos_value, now.date(),
            trading_dates=read_trading_dates(conn, pipeline="basket", ticker=ticker),
        )
        in_cooldown = stop.in_cooldown(now.date(), read_trading_dates(...))

        if stop_triggered or in_cooldown:
            exposure = 0.0
        elif signal == "BUY":
            exposure = 1.0
        elif signal == "SELL":
            exposure = 0.0
        else:  # HOLD
            exposure = prior_exposure(conn, ticker, pipeline="basket")

        decisions[ticker] = {
            "signal": signal, "exposure": exposure, "price": close,
            "regime": regime, "source": source,
            "stop_state": (stop_triggered, in_cooldown, stop.high_water_mark),
        }

        log_shadow_signal(
            conn, now, ticker, close,
            ml_signal=signal, signal_source=source, regime=regime,
            high_water_mark=stop.high_water_mark,
            trailing_stop_triggered=stop_triggered or in_cooldown,
            pipeline="basket",
        )

    rebalance_to_targets(conn, broker, portfolio_value, weights, decisions, now)
    write_basket_portfolio_snapshot(conn, broker, weights.keys(), now)
```

Rebalance pseudocode:

```python
def rebalance_to_targets(conn, broker, portfolio_value, weights, decisions, now):
    for ticker, weight in weights.items():
        d = decisions[ticker]
        target_value = portfolio_value * weight * d["exposure"]
        current_qty = broker.get_position(ticker)
        current_value = current_qty * d["price"]
        diff_value = target_value - current_value

        if abs(diff_value) < d["price"]:
            continue

        diff_shares = int(diff_value // d["price"])
        if diff_shares == 0:
            continue

        action = "BUY" if diff_shares > 0 else "SELL"
        order = broker.submit_order(
            OrderRequest(action=action, qty=abs(diff_shares)),
            ticker=ticker,
        )
        log_order(
            conn, now, ticker, action, abs(diff_shares),
            d["price"], order.id, order.status, pipeline="basket",
        )
```

### Key invariants

- **Cash sits idle when stop fires.** `exposure=0` → `target_value=0` → rebalance sells to flat. Other tickers' targets use the full `portfolio_value` × their configured weight, so they do not redistribute upward.
- **HOLD carries forward.** Reuses the validated semantics from `_exposure_from_decisions` in `daily_email.py`.
- **Sub-share rebalances skipped.** Drift smaller than one share's value produces no order. Daily rebalancing is dominated by signal flips, not drift correction, given share-price granularity.
- **Reconciliation per ticker** at the top of every basket run.
- **Order type:** market orders submitted at 21:20 UTC, fills at next-day open.

## Email Integration

New `BASKET PAPER` section between `PERFORMANCE` and `SECTOR SHADOW`. Reads `pipeline='basket'` rows from `portfolio.csv` and `shadow_signals.csv`.

```
BASKET PAPER (paper-trading the 45/30/15/10 basket since 2026-05-21)

  Total value:  $106,540   (+0.42% today, +0.51% since launch)
  Cash sleeve: $2,140  (2.0%)

  Ticker  Target  Actual  Position    Value    Today      Signal     Stop
  ------  ------  ------  --------    -----    -----      ------     ----
  SPY     45.0%   44.8%   64 sh    $47,747   +0.31%   HOLD (SMA)    OK
  XLK     30.0%   30.5%   181 sh   $32,491   +0.78%   HOLD (XGB)    OK
  XLV     15.0%   15.2%   111 sh   $16,212   -0.12%   HOLD (XGB)    OK
  XLE     10.0%    9.5%   175 sh   $10,150   +1.45%   BUY  (XGB)    OK

  Orders today: 0  (no rebalance crossed share boundary, no signal flips)
  Trailing stops armed: 4/4   |   In cooldown: 0
```

When a stop fires, the ticker's Stop column shows `FIRED` and a warning note is appended:

```
  XLE     10.0%    0.0%    0 sh        $0     -    SELL (FLAT)   FIRED
  ⚠ XLE trailing stop fired 2026-07-14 (HWM $61.20 → triggered at $55.08).
    Cash held idle; re-entry after 5 trading days if signal allows.
```

Code touch-points:

1. `reports/daily_email.py` — new `build_basket_paper_section(*, portfolio_df, shadow_signals_df, basket_weights, launch_date) -> str`.
2. `reports/daily_email.py: build_email_body` — gains optional `basket_state: dict | None` kwarg. Section is omitted when `basket_state` is None (basket pipeline has never run).
3. `main.py` at the email-rendering step — reads `portfolio.csv` and `shadow_signals.csv` filtered to `pipeline='basket'` (or None if no rows exist yet). Passes to `build_email_body`.
4. `tests/test_daily_email.py` — three new tests (basic, fired-stop, omitted-when-empty).

A small bonus that falls out: the existing `PERFORMANCE` section can gain a `Basket (paper)` row, computed the same way `System (real)` is — same `live_start_date`, just filter on `pipeline='basket'`.

## Testing Strategy

**Equivalence test (acceptance gate):** With `BASKET_PAPER_WEIGHTS = {"SPY": 1.0}`, the basket pipeline's resulting SPY position and `total_value` match the SPY-only pipeline's on the same fixture day, within 0.01 dollars.

**Per-component unit tests:**
- `test_orchestrator.py` — BUY/HOLD/SELL/FIRED-stop/COOLDOWN combinations produce correct exposures.
- `test_rebalance.py` — current+target+price produces correct order list. Sub-share skips, signal flips, drift, no-op days.
- `test_portfolio.py` — N-row snapshot writer with consistent total_value. Bootstrap from SPY-only row when no basket rows exist.
- `test_trailing_stop.py` — four independent per-ticker stops; one ticker stopping doesn't affect others' HWM or cooldown.

**Migration regression tests:**
- `test_migration_idempotent` — running twice = running once.
- `test_migration_preserves_spy_only_rows` — every original column byte-identical after filtering to `pipeline='spy_only'`.
- `test_log_portfolio_with_explicit_pipeline_kwarg` — the modified logger writes the column correctly.

**Email rendering tests:**
- `test_build_basket_paper_section_basic` — happy path, four tickers, all OK stops.
- `test_build_basket_paper_section_with_fired_stop` — XLE shows FIRED, warning line appears.
- `test_build_email_body_omits_basket_section_when_no_basket_rows` — section absent when CSV has no `pipeline='basket'` rows.

**Coverage gate before merging:** 100% of new `basket/` module functions covered. Equivalence test passes. All existing tests (~250) still pass. Migration tests pass against a copy of the real `portfolio.csv`.

## Migration and Rollout

**Phase A — preparatory, no behavior change.** Add `pipeline` kwarg to `log_portfolio`, `log_order`, `log_shadow_signal` (default `'spy_only'`). Update existing SPY-only `main.py` call sites to pass `pipeline='spy_only'` explicitly. Backup CSVs to `data/_pre_basket_backup/`. Run migration script. Property test verifies bit-identical preservation. Commit, push, watch one scheduled SPY-only run. Acceptance: the next email is byte-identical in user-visible content to the day before.

**Phase B — basket pipeline code, dormant.** Build `src/schroeder_trader/basket/` modules with full test coverage. The equivalence test (`BASKET_PAPER_WEIGHTS={"SPY": 1.0}` matches SPY-only) must pass across at least three distinct historical fixture days representing different signal/regime states (e.g., one all-BUY day, one with a signal flip, one with bear-weakening). Add `daily-basket.yml` workflow with `workflow_dispatch` only — not scheduled yet. Commit. Push. SPY-only continues unchanged.

**Phase C — basket pipeline paper-trading, no email integration.** Manually trigger `daily-basket.yml` once. Verify `pipeline='basket'` rows in `portfolio.csv`, orders submitted to the paper account, position matches the rebalance target. Run the equivalence test against the day's actual data in a side branch. If clean, switch `daily-basket.yml` to scheduled at 21:20 UTC. Basket writes state silently.

**Phase D — email integration.** Add `build_basket_paper_section` with full tests. Wire it into `build_email_body` (conditional on `pipeline='basket'` rows). SPY-only's `main.py` reads basket rows at email-render time. Commit, push. Next 21:30 UTC run produces the unified email.

**Phase E — retirement criteria (documented, executed later, OUT OF SCOPE for this spec).** Retire SPY-only when (a) basket has been writing state for 60+ trading days, (b) at least one regime transition observed across the basket window, (c) basket cumulative return falls within ±2σ of the 27-year backtest distribution for the same regime mix, (d) equivalence test still passes (CI gate), and (e) user explicitly approves.

**Rollback plan:**
- Phase A: restore `data/_pre_basket_backup/`, revert `log_*` signature changes and SPY-only call sites.
- Phase C: set `daily-basket.yml` back to `workflow_dispatch` only. Basket subpackage stays dormant.

## Risks and Mitigations

- **Migration corrupts live state files.** Mitigation: backup before migration; property test for bit-identical preservation; idempotent script.
- **Equivalence test passes coincidentally for a quiet day but reveals divergence on a signal-flip day.** Mitigation: run the equivalence test across at least three distinct historical days representing different signal/regime states before declaring Phase B done.
- **Basket pipeline writes to `portfolio.csv` while SPY-only is reading it.** Mitigation: 10-minute cron gap; SPY-only reads at 21:32 (post-trade-execution), basket finishes by ~21:25 at latest. File-level write atomicity (CsvStore appends complete rows) handles overlap if it occurs.
- **Per-ticker trailing stop state has no migration path.** Mitigation: basket pipeline's first run initializes all four `TrailingStop`s with `high_water_mark=0` (will be updated by first portfolio value read). No legacy stop state to migrate; this is a fresh state.
- **Memory drift on the basket-weight choice.** Mitigation: `feedback_resist_reoptimizing_basket_weights.md` is in memory. The plan honors it (uses the configured weights, no inline re-optimization).
