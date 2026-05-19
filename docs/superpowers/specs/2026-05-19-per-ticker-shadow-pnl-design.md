# Per-Ticker Shadow P&L Tracking — Design

**Date:** 2026-05-19
**Status:** Approved, ready for plan

## Problem

The system shadow-trades sector ETFs (XLK, XLV, XLE) alongside its live SPY position to test whether the composite strategy generalizes beyond the broad market. Today the daily email shows only SPY performance. To judge sector shadow results without running ad-hoc SQL, the email needs a per-ticker P&L view that compounds signal-driven returns rather than raw price drift.

## Goal

Add a `SECTOR SHADOW` section to the daily email showing, for each non-SPY ticker logged to `shadow_signals.csv`, the cumulative return of following its composite-routed signal compared to buy-and-hold of that same ticker.

## Non-Goals

- No schema change to `shadow_signals.csv`. Composite decision is reconstructed at render time.
- No Kelly-sized shadow variant. Production trades binary; the shadow view mirrors that.
- No trailing-stop simulation. Phase 4 deliberately keeps trailing-stop accounting production-only.
- No new shadow tickers. The mechanism works for whatever is already being logged.
- No backfill of pre-shadow history. Window starts at each ticker's first row in `shadow_signals.csv`.

## Composite Signal Source

The `ml_signal` column in `shadow_signals.csv` is misnamed — `_run_shadow_for_ticker` (main.py:635) and the SPY production path (main.py:358) both write `composite_sig.value` to it, not the raw XGBoost output. So the composite decision is already in the CSV; no reconstruction needed.

Translate `ml_signal` to next-day target exposure: `BUY → 1.0`, `SELL → 0.0`, `HOLD → carry forward previous`. This matches production's binary sizing (98% in or 0% out, approximated as 100/0 for sim purposes — the LLM oracle sim uses the same convention).

The `signal_source` column is shown in the email for context (so a reader can see which sub-signal drove the decision) but is not used in P&L math.

## Simulation

Reuse the existing `_simulate_close_to_close` helper in `daily_email.py`. It already implements "signal on day T applies to T+1 close-to-close return," which is the correct semantics. Each ticker is simulated independently:

1. Filter `shadow_signals.csv` to the ticker.
2. Derive `{date: exposure}` map from composite reconstruction.
3. Fetch close-price history for the ticker covering inception → today.
4. Run `_simulate_close_to_close(closes, exposure_by_date, start_value=100.0)`.
5. Cumulative composite return = `(final / 100.0 - 1) * 100`.
6. Buy-and-hold return = `(closes[-1] / closes[0] - 1) * 100`.
7. Edge = `composite_return_pct - bnh_return_pct` (in percentage points).

Use `start_value=100.0` (not `100_000`) so the section is a pure percentage view — no dollar amounts for shadow tickers.

## Output Format

New section, appended after the existing PERFORMANCE block:

```
SECTOR SHADOW (composite signal, not trading)
  Ticker  Since        Sessions  Composite   B&H       Edge
  ------  ----------   --------  ---------   -------   --------
  XLK     2026-05-12          5    +0.45%   -0.48%   +0.93 pp
  XLV     2026-05-12          5    -0.08%   -0.08%   +0.00 pp
  XLE     2026-05-12          5    +5.19%   +5.19%   +0.00 pp

  Note: composite signal logged but not traded; numbers reflect what
        binary exposure would have earned vs buy-and-hold of each ETF.
```

Section is omitted entirely if there are zero non-SPY tickers in `shadow_signals.csv`.

## Code Touch-Points

1. **`src/schroeder_trader/reports/daily_email.py`**
   - Add `_exposure_from_decisions(decisions_by_date) -> dict[date, float]` translating BUY/HOLD/SELL to 0.0/1.0 with HOLD carry-forward.
   - Add `build_sector_shadow_section(shadow_signals_path, ticker_close_histories) -> str`.
   - Thread `sector_close_histories: dict[str, pd.DataFrame]` through `build_email_body`.

2. **`src/schroeder_trader/main.py`**
   - The sector shadow loop already fetches per-ticker `market_data` for signals. Collect those DataFrames into a dict keyed by ticker and pass to `build_email_body`.

3. **`tests/test_daily_email.py`** (extend existing file)
   - Test exposure translation: HOLD carry-forward (BUY → HOLD → HOLD → SELL → HOLD).
   - Test full section render against a hand-rolled three-day fixture for one ticker.
   - Test multi-ticker render (two tickers with different inception dates).
   - Test empty-shadow case (no non-SPY tickers) returns empty string and is omitted from body.
   - Test single-session ticker is skipped (cannot compute return from one bar).

## Edge Cases

- **Single shadow day for a ticker:** Sessions=1. Cannot compute close-to-close return from one bar. Show `—` in Composite/B&H/Edge columns or skip that row. **Decision:** skip the row; show only tickers with ≥2 sessions.
- **Ticker has signal logged but no matching price in fetched history:** This would only happen if `main.py` fails to fetch market data for a ticker but still logs a shadow row (unlikely). **Decision:** skip the row, log a warning.
- **First row's signal is SELL/HOLD with no prior position:** Exposure starts at 0 until first BUY. This is the same convention used by the LLM oracle sim.
- **Same date appears twice in shadow_signals.csv for one ticker:** Use last entry (matches existing oracle sim `drop_duplicates(["date", "ticker"], keep="last")`).

## Risks & Mitigations

- **Misleading column name.** The `ml_signal` column actually stores the composite decision. A future contributor refactoring shadow logging might "fix" this by routing the raw XGB signal there instead, silently breaking the P&L sim. **Mitigation:** comment in `build_sector_shadow_section` explaining the column name vs. content mismatch, with a pointer to the two writer sites.
- **Spec creep.** This adds a section; it does not change trading, logging, or model behavior. Keep the diff to email rendering + main wiring.

## Acceptance

- New section appears in the next scheduled email containing at least one non-SPY ticker with ≥2 sessions.
- Composite/B&H/Edge columns match a manual spreadsheet computation against the same `shadow_signals.csv`.
- All existing daily-email tests still pass.
