# Per-Ticker Shadow P&L Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a SECTOR SHADOW section to the daily email showing each non-SPY ticker's composite-signal cumulative return vs buy-and-hold, computed at render time from `data/shadow_signals.csv` with no schema changes.

**Architecture:** Three new helpers and one section builder in `src/schroeder_trader/reports/daily_email.py`. Wiring change in `src/schroeder_trader/main.py` to collect per-ticker close history from the existing sector shadow loop and pass it through `build_email_body`. Unit tests extend `tests/test_daily_email.py`. Composite decision is already stored in the `ml_signal` column of `shadow_signals.csv` — no reconstruction needed.

**Tech Stack:** Python 3.12, pandas, pytest. Run tests with `uv run pytest`.

---

## File Structure

- **Modify** `src/schroeder_trader/reports/daily_email.py` — add `_exposure_from_decisions`, `_compute_ticker_shadow_pnl`, `build_sector_shadow_section`; thread new arg through `build_email_body`.
- **Modify** `src/schroeder_trader/main.py` — capture per-ticker `df` from `_run_shadow_for_ticker` into a dict, pass to `build_email_body`.
- **Modify** `tests/test_daily_email.py` — add five tests for the new helpers and section.

Single-responsibility split inside `daily_email.py`: pure logic helpers (translation, math) are testable without I/O; only `build_sector_shadow_section` touches the CSV.

---

### Task 1: Exposure translation helper

**Files:**
- Modify: `src/schroeder_trader/reports/daily_email.py` (add new helper, place after `_simulate_close_to_close`)
- Test: `tests/test_daily_email.py`

The composite decision sequence is read from the `ml_signal` column of `shadow_signals.csv` (the column is misnamed — it stores the composite output, not the raw XGB signal; see `src/schroeder_trader/main.py:358` and `:635`). This helper translates that sequence into a target-exposure dict suitable for `_simulate_close_to_close`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_daily_email.py`:

```python
from datetime import date as _date

from schroeder_trader.reports.daily_email import _exposure_from_decisions


def test_exposure_from_decisions_buy_hold_sell_carry_forward():
    decisions = {
        _date(2026, 5, 12): "BUY",
        _date(2026, 5, 13): "HOLD",
        _date(2026, 5, 14): "HOLD",
        _date(2026, 5, 15): "SELL",
        _date(2026, 5, 18): "HOLD",
    }
    exposure = _exposure_from_decisions(decisions)
    assert exposure[_date(2026, 5, 12)] == 1.0
    assert exposure[_date(2026, 5, 13)] == 1.0  # HOLD carries BUY forward
    assert exposure[_date(2026, 5, 14)] == 1.0
    assert exposure[_date(2026, 5, 15)] == 0.0  # SELL flattens
    assert exposure[_date(2026, 5, 18)] == 0.0  # HOLD carries SELL forward


def test_exposure_from_decisions_starts_flat_if_first_is_hold():
    decisions = {
        _date(2026, 5, 12): "HOLD",
        _date(2026, 5, 13): "BUY",
    }
    exposure = _exposure_from_decisions(decisions)
    assert exposure[_date(2026, 5, 12)] == 0.0  # nothing to carry forward
    assert exposure[_date(2026, 5, 13)] == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_daily_email.py::test_exposure_from_decisions_buy_hold_sell_carry_forward tests/test_daily_email.py::test_exposure_from_decisions_starts_flat_if_first_is_hold -v`
Expected: FAIL with `ImportError: cannot import name '_exposure_from_decisions'`.

- [ ] **Step 3: Write minimal implementation**

Add to `src/schroeder_trader/reports/daily_email.py` (after the existing `_simulate_close_to_close` function around line 205):

```python
def _exposure_from_decisions(decisions_by_date: dict) -> dict:
    """Translate BUY/HOLD/SELL sequence into a 0/1 exposure map.

    HOLD carries the previous exposure forward. A leading HOLD (no prior BUY)
    starts at 0.0 — matches "you can't hold what you don't have."
    """
    out: dict = {}
    current = 0.0
    for d in sorted(decisions_by_date):
        decision = decisions_by_date[d]
        if decision == "BUY":
            current = 1.0
        elif decision == "SELL":
            current = 0.0
        # HOLD: leave `current` unchanged
        out[d] = current
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_daily_email.py::test_exposure_from_decisions_buy_hold_sell_carry_forward tests/test_daily_email.py::test_exposure_from_decisions_starts_flat_if_first_is_hold -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/schroeder_trader/reports/daily_email.py tests/test_daily_email.py
git commit -m "feat(daily_email): exposure-from-decisions helper for shadow P&L

Translates composite BUY/HOLD/SELL sequence into a 0/1 next-day
target-exposure map, with HOLD carrying the prior exposure forward."
```

---

### Task 2: Single-ticker P&L computation

**Files:**
- Modify: `src/schroeder_trader/reports/daily_email.py` (add new helper)
- Test: `tests/test_daily_email.py`

Computes a single ticker's composite return, B&H return, edge, and session count from a filtered shadow_signals DataFrame plus that ticker's close history. Skips tickers with <2 sessions.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_daily_email.py`:

```python
from schroeder_trader.reports.daily_email import _compute_ticker_shadow_pnl


def test_compute_ticker_shadow_pnl_basic():
    # Three days for XLK: BUY at 100, HOLD at 110, SELL at 105
    # Expected: exposure 1.0 from day 1 onward.
    #   day 1 → day 2: 1.0 * (110/100 - 1) = +10%
    #   day 2 → day 3: 1.0 * (105/110 - 1) = -4.55%
    #   composite final = 100 * 1.10 * (105/110) = 105.0  →  +5.00%
    # B&H: (105/100 - 1) = +5.00%  →  edge 0.00pp (signals never sold in time)
    shadow_df = pd.DataFrame({
        "timestamp": [
            "2026-05-12T20:30:00+00:00",
            "2026-05-13T20:30:00+00:00",
            "2026-05-14T20:30:00+00:00",
        ],
        "ticker": ["XLK", "XLK", "XLK"],
        "close_price": [100.0, 110.0, 105.0],
        "ml_signal": ["BUY", "HOLD", "SELL"],
    })
    closes = pd.Series(
        [100.0, 110.0, 105.0],
        index=pd.to_datetime(["2026-05-12", "2026-05-13", "2026-05-14"]),
        name="close",
    )
    result = _compute_ticker_shadow_pnl(shadow_df, closes)
    assert result is not None
    assert result["sessions"] == 3
    assert result["inception"] == _date(2026, 5, 12)
    assert abs(result["composite_return_pct"] - 5.0) < 1e-6
    assert abs(result["bnh_return_pct"] - 5.0) < 1e-6
    assert abs(result["edge_pp"] - 0.0) < 1e-6


def test_compute_ticker_shadow_pnl_skips_single_session():
    shadow_df = pd.DataFrame({
        "timestamp": ["2026-05-12T20:30:00+00:00"],
        "ticker": ["XLK"],
        "close_price": [100.0],
        "ml_signal": ["BUY"],
    })
    closes = pd.Series(
        [100.0], index=pd.to_datetime(["2026-05-12"]), name="close",
    )
    assert _compute_ticker_shadow_pnl(shadow_df, closes) is None


def test_compute_ticker_shadow_pnl_sell_then_buy_captures_partial_run():
    # Day 1 BUY @100, Day 2 SELL @110 (signal applied next day), Day 3 BUY @105, Day 4 HOLD @120
    # Composite exposures: d1→d2 = 1.0, d2→d3 = 0.0, d3→d4 = 1.0
    #   d1→d2: 1.0 * (110/100 - 1) = +10%   → 110.0
    #   d2→d3: 0.0 * (105/110 - 1) = 0       → 110.0
    #   d3→d4: 1.0 * (120/105 - 1) = +14.29% → 125.71
    # Composite return = +25.71%; B&H = +20.00%; edge ≈ +5.71pp
    shadow_df = pd.DataFrame({
        "timestamp": [
            "2026-05-12T20:30:00+00:00",
            "2026-05-13T20:30:00+00:00",
            "2026-05-14T20:30:00+00:00",
            "2026-05-15T20:30:00+00:00",
        ],
        "ticker": ["XLK"] * 4,
        "close_price": [100.0, 110.0, 105.0, 120.0],
        "ml_signal": ["BUY", "SELL", "BUY", "HOLD"],
    })
    closes = pd.Series(
        [100.0, 110.0, 105.0, 120.0],
        index=pd.to_datetime(["2026-05-12", "2026-05-13", "2026-05-14", "2026-05-15"]),
        name="close",
    )
    result = _compute_ticker_shadow_pnl(shadow_df, closes)
    assert result["sessions"] == 4
    assert abs(result["composite_return_pct"] - 25.7142857) < 1e-4
    assert abs(result["bnh_return_pct"] - 20.0) < 1e-6
    assert abs(result["edge_pp"] - 5.7142857) < 1e-4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_daily_email.py -k compute_ticker_shadow -v`
Expected: FAIL with `ImportError: cannot import name '_compute_ticker_shadow_pnl'`.

- [ ] **Step 3: Write minimal implementation**

Add to `src/schroeder_trader/reports/daily_email.py` (after `_exposure_from_decisions`):

```python
def _compute_ticker_shadow_pnl(
    shadow_df: pd.DataFrame, closes: pd.Series,
) -> dict | None:
    """Compute composite vs B&H return for a single ticker.

    shadow_df: rows from shadow_signals.csv already filtered to one ticker.
               The 'ml_signal' column stores the composite decision
               (BUY/HOLD/SELL), not the raw XGB signal — see
               src/schroeder_trader/main.py:358 and :635 where
               composite_sig.value is written into that column.
    closes:    daily close prices for the ticker, indexed by date or datetime,
               covering at least the shadow window.

    Returns None if there are fewer than 2 distinct shadow sessions.
    """
    if shadow_df.empty:
        return None

    df = shadow_df.copy()
    df["date"] = pd.to_datetime(df["timestamp"], utc=True, format="ISO8601") \
        .dt.tz_convert("America/New_York").dt.date
    # Same-date dedupe (latest entry wins; matches oracle sim convention).
    df = df.drop_duplicates(["date"], keep="last").sort_values("date")
    if len(df) < 2:
        return None

    decisions_by_date = dict(zip(df["date"], df["ml_signal"].astype(str)))
    exposure_by_date = _exposure_from_decisions(decisions_by_date)

    # Normalize closes index to tz-naive dates so it lines up with the
    # decisions map (which uses ET-local dates).
    closes = closes.copy()
    if isinstance(closes.index, pd.DatetimeIndex) and closes.index.tz is not None:
        closes.index = closes.index.tz_localize(None)
    inception = df["date"].iloc[0]
    last = df["date"].iloc[-1]
    closes_window = closes[
        (closes.index.date >= inception) & (closes.index.date <= last)
    ]
    if len(closes_window) < 2:
        return None

    values = _simulate_close_to_close(
        closes_window, exposure_by_date, start_value=100.0,
    )
    composite_pct = (float(values.iloc[-1]) / 100.0 - 1) * 100
    bnh_pct = (float(closes_window.iloc[-1]) / float(closes_window.iloc[0]) - 1) * 100

    return {
        "inception": inception,
        "sessions": len(df),
        "composite_return_pct": composite_pct,
        "bnh_return_pct": bnh_pct,
        "edge_pp": composite_pct - bnh_pct,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_daily_email.py -k compute_ticker_shadow -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/schroeder_trader/reports/daily_email.py tests/test_daily_email.py
git commit -m "feat(daily_email): per-ticker shadow P&L computation

Computes composite vs buy-and-hold cumulative return for a single ticker
from its shadow_signals rows plus close-price history. Skips tickers with
fewer than two sessions. Reuses _simulate_close_to_close for compounding."
```

---

### Task 3: Section builder

**Files:**
- Modify: `src/schroeder_trader/reports/daily_email.py` (add `build_sector_shadow_section`)
- Test: `tests/test_daily_email.py`

Reads `shadow_signals.csv`, filters out SPY, computes per-ticker P&L via Task 2's helper, formats as a fixed-width table. Returns empty string if no eligible tickers (caller decides whether to omit).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_daily_email.py`:

```python
from schroeder_trader.reports.daily_email import build_sector_shadow_section


def test_sector_shadow_section_two_tickers(tmp_path):
    # XLK: 3 sessions, composite=BUY all → matches B&H
    # XLE: 3 sessions, BUY then SELL then BUY → sits out the middle day
    shadow = pd.DataFrame({
        "timestamp": [
            "2026-05-12T20:30:00+00:00", "2026-05-13T20:30:00+00:00", "2026-05-14T20:30:00+00:00",
            "2026-05-12T20:30:00+00:00", "2026-05-13T20:30:00+00:00", "2026-05-14T20:30:00+00:00",
            "2026-05-12T20:30:00+00:00", "2026-05-13T20:30:00+00:00", "2026-05-14T20:30:00+00:00",
        ],
        "ticker": ["XLK", "XLK", "XLK", "XLE", "XLE", "XLE", "SPY", "SPY", "SPY"],
        "close_price": [
            100.0, 110.0, 121.0,
             50.0,  55.0,  52.0,
            700.0, 710.0, 720.0,
        ],
        "ml_signal": ["BUY"] * 3 + ["BUY", "SELL", "BUY"] + ["HOLD"] * 3,
    })
    shadow.to_csv(tmp_path / "shadow_signals.csv", index=False)

    xlk_closes = pd.Series(
        [100.0, 110.0, 121.0],
        index=pd.to_datetime(["2026-05-12", "2026-05-13", "2026-05-14"]),
    )
    xle_closes = pd.Series(
        [50.0, 55.0, 52.0],
        index=pd.to_datetime(["2026-05-12", "2026-05-13", "2026-05-14"]),
    )

    section = build_sector_shadow_section(
        shadow_signals_path=tmp_path / "shadow_signals.csv",
        ticker_close_histories={"XLK": xlk_closes, "XLE": xle_closes},
    )
    assert "SECTOR SHADOW" in section
    assert "XLK" in section
    assert "XLE" in section
    assert "SPY" not in section  # SPY excluded
    # XLK composite return = B&H = +21.00%
    assert "+21.00%" in section
    # XLE composite: exposures = [1.0, 1.0, 0.0] → value sequence 100, 110, 110 → +10.00%
    assert "+10.00%" in section


def test_sector_shadow_section_empty_when_only_spy(tmp_path):
    shadow = pd.DataFrame({
        "timestamp": ["2026-05-12T20:30:00+00:00"],
        "ticker": ["SPY"],
        "close_price": [700.0],
        "ml_signal": ["HOLD"],
    })
    shadow.to_csv(tmp_path / "shadow_signals.csv", index=False)

    section = build_sector_shadow_section(
        shadow_signals_path=tmp_path / "shadow_signals.csv",
        ticker_close_histories={},
    )
    assert section == ""


def test_sector_shadow_section_missing_file_returns_empty(tmp_path):
    section = build_sector_shadow_section(
        shadow_signals_path=tmp_path / "does-not-exist.csv",
        ticker_close_histories={},
    )
    assert section == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_daily_email.py -k sector_shadow_section -v`
Expected: FAIL with `ImportError: cannot import name 'build_sector_shadow_section'`.

- [ ] **Step 3: Write minimal implementation**

Add to `src/schroeder_trader/reports/daily_email.py` (after `_compute_ticker_shadow_pnl`):

```python
def build_sector_shadow_section(
    *,
    shadow_signals_path: Path,
    ticker_close_histories: dict,
) -> str:
    """Build the SECTOR SHADOW table from shadow_signals.csv.

    Excludes SPY (already shown elsewhere). Tickers with fewer than two
    sessions or with no matching close history are skipped. Returns an
    empty string if no rows qualify — caller should omit the section.
    """
    if not shadow_signals_path.exists():
        return ""
    shadow = pd.read_csv(shadow_signals_path)
    if shadow.empty:
        return ""

    rows: list[tuple[str, dict]] = []
    for ticker in sorted(t for t in shadow["ticker"].unique() if t != "SPY"):
        closes = ticker_close_histories.get(ticker)
        if closes is None or len(closes) == 0:
            continue
        ticker_df = shadow[shadow["ticker"] == ticker]
        result = _compute_ticker_shadow_pnl(ticker_df, closes)
        if result is not None:
            rows.append((ticker, result))

    if not rows:
        return ""

    lines = [
        f"  {'Ticker':<7} {'Since':<11} {'Sessions':>8}  {'Composite':>9}  {'B&H':>9}  {'Edge':>9}",
        f"  {'-'*7} {'-'*11} {'-'*8}  {'-'*9}  {'-'*9}  {'-'*9}",
    ]
    for ticker, r in rows:
        comp = _fmt_pct(r["composite_return_pct"])
        bnh = _fmt_pct(r["bnh_return_pct"])
        edge = f"{r['edge_pp']:+.2f} pp"
        lines.append(
            f"  {ticker:<7} {str(r['inception']):<11} {r['sessions']:>8}  {comp:>9}  {bnh:>9}  {edge:>9}"
        )
    lines.append("")
    lines.append("  Note: composite signal logged but not traded; numbers reflect")
    lines.append("        what binary exposure would have earned vs buy-and-hold.")
    return _section("SECTOR SHADOW (composite signal, not trading)", lines)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_daily_email.py -k sector_shadow_section -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/schroeder_trader/reports/daily_email.py tests/test_daily_email.py
git commit -m "feat(daily_email): SECTOR SHADOW section builder

Reads shadow_signals.csv, excludes SPY, renders per-ticker composite
vs B&H table. Empty string when no eligible non-SPY tickers."
```

---

### Task 4: Thread through `build_email_body`

**Files:**
- Modify: `src/schroeder_trader/reports/daily_email.py` (extend `build_email_body` signature; call the new builder)
- Test: `tests/test_daily_email.py` (extend `test_build_email_body_full`)

- [ ] **Step 1: Write the failing test**

Replace the existing `test_build_email_body_full` in `tests/test_daily_email.py` with a version that exercises the new arg, and add a new test for the multi-ticker case:

```python
def test_build_email_body_full(tmp_path):
    # Synthetic SPY history covering the live-start window
    dates = pd.bdate_range("2026-04-15", periods=10).tz_localize(None)
    spy = pd.DataFrame({"close": [700.0 + i for i in range(10)]}, index=dates)

    # Synthetic portfolio.csv (system real P&L)
    pf = pd.DataFrame({
        "id": [1, 2],
        "timestamp": ["2026-04-15T20:30:00+00:00", "2026-04-28T20:30:00+00:00"],
        "cash": [1965.0, 1965.0],
        "position_qty": [141, 141],
        "position_value": [98700.0, 99918.0],
        "total_value": [100665.0, 101883.0],
    })
    pf.to_csv(tmp_path / "portfolio.csv", index=False)

    # Empty llm_shadow_signals.csv (forces sim sections to be N/A)
    pd.DataFrame(columns=[
        "id", "timestamp", "provider", "target_exposure", "error"
    ]).to_csv(tmp_path / "llm_shadow_signals.csv", index=False)

    # Empty shadow_signals.csv (no sector shadow section)
    pd.DataFrame(columns=[
        "id", "timestamp", "ticker", "close_price", "ml_signal",
    ]).to_csv(tmp_path / "shadow_signals.csv", index=False)

    body = build_email_body(
        date_str="2026-04-28",
        spy_close=715.16,
        spy_prev_close=713.97,
        portfolio_value=101883.0,
        portfolio_prev_value=100665.0,
        cash=1965.0,
        position_qty=141,
        sma_signal="HOLD",
        sma_50=676.0,
        sma_200=667.0,
        composite_signal="BUY",
        composite_source="XGB",
        regime="CHOPPY",
        bear_days=0,
        xgb_proba_up=0.578,
        xgb_threshold=0.35,
        today_action="HOLD",
        oracle_responses=[_fake_oracle("claude", "SELL", 0.85)],
        data_root=tmp_path,
        spy_history=spy,
        live_start_date=date(2026, 4, 15),
        sector_close_histories={},
    )
    assert "SchroederTrader Daily Report — 2026-04-28" in body
    assert "TODAY" in body
    assert "SYSTEM" in body
    assert "LLM ORACLES" in body
    assert "PERFORMANCE" in body
    # No sector shadow section when csv is empty
    assert "SECTOR SHADOW" not in body
    assert "System (real)" in body


def test_build_email_body_includes_sector_shadow(tmp_path):
    dates = pd.bdate_range("2026-04-15", periods=10).tz_localize(None)
    spy = pd.DataFrame({"close": [700.0 + i for i in range(10)]}, index=dates)
    pd.DataFrame(columns=["id", "timestamp", "cash", "position_qty", "position_value", "total_value"]) \
      .to_csv(tmp_path / "portfolio.csv", index=False)
    pd.DataFrame(columns=["id", "timestamp", "provider", "target_exposure", "error"]) \
      .to_csv(tmp_path / "llm_shadow_signals.csv", index=False)
    pd.DataFrame({
        "timestamp": [
            "2026-05-12T20:30:00+00:00", "2026-05-13T20:30:00+00:00", "2026-05-14T20:30:00+00:00",
        ],
        "ticker": ["XLK"] * 3,
        "close_price": [100.0, 110.0, 121.0],
        "ml_signal": ["BUY", "HOLD", "HOLD"],
    }).to_csv(tmp_path / "shadow_signals.csv", index=False)

    xlk_closes = pd.Series(
        [100.0, 110.0, 121.0],
        index=pd.to_datetime(["2026-05-12", "2026-05-13", "2026-05-14"]),
    )

    body = build_email_body(
        date_str="2026-05-14",
        spy_close=715.16, spy_prev_close=713.97,
        portfolio_value=101883.0, portfolio_prev_value=100665.0,
        cash=1965.0, position_qty=141,
        sma_signal="HOLD", sma_50=676.0, sma_200=667.0,
        composite_signal="BUY", composite_source="XGB", regime="CHOPPY",
        bear_days=0, xgb_proba_up=0.578, xgb_threshold=0.35,
        today_action="HOLD",
        oracle_responses=[],
        data_root=tmp_path,
        spy_history=spy,
        live_start_date=date(2026, 4, 15),
        sector_close_histories={"XLK": xlk_closes},
    )
    assert "SECTOR SHADOW" in body
    assert "XLK" in body
    assert "+21.00%" in body
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_daily_email.py::test_build_email_body_full tests/test_daily_email.py::test_build_email_body_includes_sector_shadow -v`
Expected: FAIL — `build_email_body()` does not accept `sector_close_histories`.

- [ ] **Step 3: Modify `build_email_body`**

In `src/schroeder_trader/reports/daily_email.py`, change the `build_email_body` signature and body:

```python
def build_email_body(
    *,
    date_str: str,
    spy_close: float,
    spy_prev_close: float | None,
    portfolio_value: float,
    portfolio_prev_value: float | None,
    cash: float,
    position_qty: int,
    sma_signal: str,
    sma_50: float,
    sma_200: float,
    composite_signal: str | None,
    composite_source: str | None,
    regime: str | None,
    bear_days: int | None,
    xgb_proba_up: float | None,
    xgb_threshold: float,
    today_action: str,
    oracle_responses: list,
    data_root: Path,
    spy_history: pd.DataFrame,
    live_start_date: date,
    sector_close_histories: dict,
) -> str:
    """Compose the full email body."""
    sections = [
        f"SchroederTrader Daily Report — {date_str}",
        "=" * 60,
        build_today_section(
            date_str=date_str,
            spy_close=spy_close,
            spy_prev_close=spy_prev_close,
            portfolio_value=portfolio_value,
            portfolio_prev_value=portfolio_prev_value,
            cash=cash,
            position_qty=position_qty,
        ),
        build_system_section(
            sma_signal=sma_signal,
            sma_50=sma_50,
            sma_200=sma_200,
            composite_signal=composite_signal,
            composite_source=composite_source,
            regime=regime,
            bear_days=bear_days,
            xgb_proba_up=xgb_proba_up,
            xgb_threshold=xgb_threshold,
            today_action=today_action,
        ),
        build_oracles_section(oracle_responses),
        build_performance_section(
            data_root=data_root,
            spy_history=spy_history,
            start_date=live_start_date,
        ),
    ]
    sector_section = build_sector_shadow_section(
        shadow_signals_path=data_root / "shadow_signals.csv",
        ticker_close_histories=sector_close_histories,
    )
    if sector_section:
        sections.append(sector_section)
    return "\n\n".join(sections) + "\n"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_daily_email.py -v`
Expected: PASS — all daily-email tests including the two updated/added body tests.

- [ ] **Step 5: Commit**

```bash
git add src/schroeder_trader/reports/daily_email.py tests/test_daily_email.py
git commit -m "feat(daily_email): wire sector shadow section into email body

Adds required sector_close_histories arg to build_email_body; appends the
SECTOR SHADOW section after PERFORMANCE when non-empty."
```

---

### Task 5: Capture sector close history in `main.py` and pass to email

**Files:**
- Modify: `src/schroeder_trader/main.py:567-650` (`_run_shadow_for_ticker`) — return the fetched DataFrame.
- Modify: `src/schroeder_trader/main.py:500-510` (shadow loop) — collect returned DataFrames into a dict.
- Modify: `src/schroeder_trader/main.py:528-551` (`build_email_body` call) — pass `sector_close_histories`.

No new test — this is wiring. The existing email-body tests cover the contract; an end-to-end main test is out of scope (`main.py` already lacks one).

- [ ] **Step 1: Update `_run_shadow_for_ticker` to return its close history**

Change the function signature and add a return statement at the end. In `src/schroeder_trader/main.py` around line 567:

```python
def _run_shadow_for_ticker(
    conn, now: datetime, ticker: str, model_path: Path, ext_df,
) -> pd.DataFrame | None:
    """Compute composite signal for a non-trading ticker; log to shadow_signals.

    No trailing stop, no Kelly, no oracles — just the regime + SMA + XGB +
    composite routing, written with ticker=ticker so it lives alongside SPY's
    production rows. Failures are non-fatal.

    Returns the fetched daily-bars DataFrame (indexed by date, 'close' column)
    so the caller can pass it to the email body for P&L rendering. Returns
    None if the function exits early (no model, no features, etc.).
    """
```

At every early-return point in the function, change `return` to `return None`. At the very end of the function (after the final `logger.info(...)` call around line 650), add:

```python
    return df
```

- [ ] **Step 2: Update the shadow loop to collect returned histories**

In `src/schroeder_trader/main.py` around line 500-510, change the loop to keep a dict of returned DataFrames:

```python
    sector_close_histories: dict[str, pd.DataFrame] = {}
    if FEATURES_CSV_PATH.exists():
        try:
            ext_df_shadow = pd.read_csv(str(FEATURES_CSV_PATH), index_col="date", parse_dates=True)
            for shadow_ticker, shadow_model_path in SHADOW_TICKERS.items():
                try:
                    ticker_df = _run_shadow_for_ticker(
                        conn, now, shadow_ticker, shadow_model_path, ext_df_shadow,
                    )
                    if ticker_df is not None and "close" in ticker_df.columns:
                        sector_close_histories[shadow_ticker] = ticker_df[["close"]]
                except Exception:
                    logger.exception("Shadow ticker %s failed (non-fatal)", shadow_ticker)
        except Exception:
            logger.exception("Shadow tickers block failed (non-fatal)")
```

- [ ] **Step 3: Pass the dict to `build_email_body`**

In `src/schroeder_trader/main.py` inside the `build_email_body(...)` call (around line 528-551), add one new keyword argument immediately after the existing `live_start_date=date(2026, 4, 15),` line:

```python
            sector_close_histories=sector_close_histories,
```

The full tail of the call should now read:

```python
            live_start_date=date(2026, 4, 15),  # first day of paper trading
            sector_close_histories=sector_close_histories,
        )
```

- [ ] **Step 4: Run the full test suite**

Run: `uv run pytest`
Expected: PASS — all 175+ existing tests plus the new ones from Tasks 1–4.

- [ ] **Step 5: Smoke-test by rendering against real data**

Run a quick interactive check to confirm the section renders sensibly against the actual `data/shadow_signals.csv`:

```bash
uv run python - <<'PY'
from pathlib import Path
import pandas as pd
from schroeder_trader.reports.daily_email import build_sector_shadow_section
from schroeder_trader.data.market_data import fetch_daily_bars

shadow_csv = Path("data/shadow_signals.csv")
tickers = sorted(t for t in pd.read_csv(shadow_csv)["ticker"].unique() if t != "SPY")
histories = {t: fetch_daily_bars(t, days=60)[["close"]] for t in tickers}
print(build_sector_shadow_section(shadow_signals_path=shadow_csv, ticker_close_histories=histories))
PY
```

Expected: a SECTOR SHADOW block listing XLK, XLE, XLV with composite/B&H/edge values that match the May 12–18 numbers we computed earlier (XLK ~-0.48% B&H, XLE ~+5.19% B&H, XLV ~-0.08% B&H). The composite returns should match the raw price drift on days where the signal stayed BUY, and differ on days where SELL/HOLD pulled exposure to 0.

- [ ] **Step 6: Commit**

```bash
git add src/schroeder_trader/main.py
git commit -m "feat(main): pass sector close histories to daily email

Threads each shadow ticker's fetched daily bars through the email body
so the SECTOR SHADOW section can compute composite vs B&H returns
without re-fetching. Failures in the shadow loop continue to be
non-fatal — missing histories just omit that ticker from the table."
```

---

### Task 6: Verification

**Files:** none modified

- [ ] **Step 1: Final test run**

Run: `uv run pytest`
Expected: all tests pass, ~21 seconds, no warnings about the new code.

- [ ] **Step 2: Type-check / lint sanity**

If a project linter is configured, run it. (At time of writing the repo has no enforced linter step; if `ruff check` or `mypy` is configured in `pyproject.toml`, run the same command CI runs.)

Check: `uv run ruff check src/schroeder_trader/reports/daily_email.py src/schroeder_trader/main.py 2>/dev/null || echo "no ruff config — skip"`

- [ ] **Step 3: Verify the next scheduled email contains the new section**

The pipeline runs at 21:30 UTC on weekdays via `.github/workflows/daily.yml`. After this change merges to `main`, the next scheduled run's email should include a SECTOR SHADOW block with XLK/XLV/XLE rows.

If you want to confirm without waiting, trigger the workflow manually with `gh workflow run daily.yml -f dry_run=true` (this sends a `[TEST]` email and does not commit state). Check the resulting email body for the new section.

- [ ] **Step 4: Update memory**

After the live email confirms the section renders correctly, add a project memory noting that sector shadow P&L is now visible in the daily email — this becomes a relevant pointer for future "what's the latest on sector shadow?" questions. Use a project-type memory at `~/.claude/projects/-Users-ads7fg-git-SchroederTrader/memory/` and add a line to MEMORY.md.

---

## Self-Review Notes

**Spec coverage:**
- Composite signal source (spec §"Composite Signal Source"): Task 2 reads `ml_signal` directly; the comment in `_compute_ticker_shadow_pnl` documents the misleading-column-name caveat.
- Simulation (spec §"Simulation"): Task 2 reuses `_simulate_close_to_close` with `start_value=100.0`.
- Output format (spec §"Output Format"): Task 3 produces a matching table; Task 4 conditionally appends it.
- Edge cases (spec §"Edge Cases"): single-session skip → Task 2 `test_compute_ticker_shadow_pnl_skips_single_session`; same-date duplicates → handled by `drop_duplicates(["date"], keep="last")` in Task 2; first row SELL/HOLD → Task 1 `test_exposure_from_decisions_starts_flat_if_first_is_hold`; missing price history → Task 3 `if closes is None or len(closes) == 0: continue`.
- Risks (spec §"Risks & Mitigations"): misleading column-name comment included in Task 2's implementation.

**Placeholder scan:** No TBDs. Every step contains either the actual code to add or the exact command to run.

**Type consistency:** `build_sector_shadow_section` keyword args are spelled `shadow_signals_path` and `ticker_close_histories` in Tasks 3, 4, and 5 alike. `sector_close_histories` is the name of the dict passed into `build_email_body` (matching the variable name in `main.py`).
