"""Build the structured daily email body.

Pure formatting — no network calls, no LLM. Reads state CSVs to compute
cumulative performance stats. Output is a plain-text block sectioned for
quick scanning: Today / System / Oracles / Performance.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd


def _fmt_pct(x: float | None, digits: int = 2) -> str:
    if x is None or pd.isna(x):
        return "N/A"
    return f"{x:+.{digits}f}%"


def _fmt_dollars(x: float | None, digits: int = 0) -> str:
    if x is None or pd.isna(x):
        return "N/A"
    return f"${x:,.{digits}f}"


def _fmt_target(x) -> str:
    if x is None or pd.isna(x):
        return "—"
    try:
        return f"{float(x):.2f}"
    except (TypeError, ValueError):
        return "—"


def _section(title: str, lines: list[str]) -> str:
    return f"{title}\n" + "\n".join(f"  {ln}" for ln in lines)


def build_today_section(
    *,
    date_str: str,
    spy_close: float,
    spy_prev_close: float | None,
    portfolio_value: float,
    portfolio_prev_value: float | None,
    cash: float,
    position_qty: int,
) -> str:
    spy_change_pct = None
    if spy_prev_close and spy_prev_close > 0:
        spy_change_pct = (spy_close / spy_prev_close - 1) * 100

    pf_change = None
    pf_change_pct = None
    if portfolio_prev_value and portfolio_prev_value > 0:
        pf_change = portfolio_value - portfolio_prev_value
        pf_change_pct = (portfolio_value / portfolio_prev_value - 1) * 100

    position_value = position_qty * spy_close
    exposure_pct = (position_value / portfolio_value * 100) if portfolio_value > 0 else 0.0

    lines = [
        f"Date:        {date_str}",
        f"SPY close:   {_fmt_dollars(spy_close, 2)}  ({_fmt_pct(spy_change_pct)} vs prev close)",
        f"Portfolio:   {_fmt_dollars(portfolio_value)}  ({_fmt_dollars(pf_change)} / {_fmt_pct(pf_change_pct)} today)",
        f"Position:    {position_qty} shares SPY  ({_fmt_dollars(position_value)}, {exposure_pct:.0f}% exposure)",
        f"Cash:        {_fmt_dollars(cash)}",
    ]
    return _section("TODAY", lines)


def build_system_section(
    *,
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
) -> str:
    lines = [
        f"Today's action:    {today_action}",
        f"Composite signal:  {composite_signal or 'N/A'}  (source: {composite_source or 'N/A'})",
        f"Market regime:     {regime or 'N/A'}  (bear days: {bear_days if bear_days is not None else 'N/A'})",
    ]
    if xgb_proba_up is not None:
        lines.append(f"XGB UP probability: {xgb_proba_up*100:.1f}%  (BUY threshold: {xgb_threshold*100:.0f}%)")
    lines.append(f"SMA 50 / 200:      {_fmt_dollars(sma_50, 2)} / {_fmt_dollars(sma_200, 2)}")
    return _section("SYSTEM (decides what to trade)", lines)


def build_oracles_section(oracle_responses: list) -> str:
    if not oracle_responses:
        return _section("LLM ORACLES (shadow only)", ["No oracle data."])
    lines: list[str] = []
    for resp in oracle_responses:
        label = resp.provider.upper()
        if getattr(resp, "error", None):
            lines.append(f"{label}: ERROR — {resp.error}")
            continue
        lines.append(
            f"{label}:  {resp.action}  target={_fmt_target(resp.target_exposure)}  "
            f"(regime: {resp.regime_assessment}, conf: {resp.confidence})"
        )
        if resp.key_drivers:
            lines.append(f"   drivers: {', '.join(resp.key_drivers)}")
        if resp.reasoning:
            lines.append(f"   reasoning: {resp.reasoning}")
    return _section("LLM ORACLES (shadow only — not trading)", lines)


def _compute_performance(
    *,
    data_root: Path,
    spy_history: pd.DataFrame,
    start_date: date,
) -> dict:
    """Compute cumulative performance for system (real), oracles (sim), and SPY B&H.

    spy_history: DataFrame indexed by date with a 'close' column covering
                 start_date through today.
    """
    out = {
        "start_date": start_date,
        "sessions": 0,
        "system_real": None,
        "system_real_return_pct": None,
        "claude_sim": None,
        "claude_sim_return_pct": None,
        "openai_sim": None,
        "openai_sim_return_pct": None,
        "spy_return_pct": None,
    }

    # Filter SPY to window
    if not isinstance(spy_history.index, pd.DatetimeIndex):
        spy_history.index = pd.to_datetime(spy_history.index)
    spy_history = spy_history.copy()
    spy_history.index = spy_history.index.tz_localize(None) if spy_history.index.tz is not None else spy_history.index
    spy_window = spy_history[spy_history.index.date >= start_date]
    if len(spy_window) < 2:
        return out
    out["sessions"] = len(spy_window)
    out["spy_return_pct"] = (spy_window["close"].iloc[-1] / spy_window["close"].iloc[0] - 1) * 100

    # System real: from portfolio.csv
    pf_path = data_root / "portfolio.csv"
    if pf_path.exists():
        pf = pd.read_csv(pf_path)
        if not pf.empty:
            pf["date"] = pd.to_datetime(pf["timestamp"], utc=True, format="ISO8601").dt.tz_convert("America/New_York").dt.date
            pf_window = pf[pf["date"] >= start_date].sort_values("date")
            if len(pf_window) >= 2:
                start_val = float(pf_window["total_value"].iloc[0])
                end_val = float(pf_window["total_value"].iloc[-1])
                out["system_real"] = end_val
                out["system_real_return_pct"] = (end_val / start_val - 1) * 100

    # Oracle sim: simulate close-to-close rebalancing from start_date
    llm_path = data_root / "llm_shadow_signals.csv"
    if llm_path.exists():
        llm = pd.read_csv(llm_path)
        if not llm.empty:
            llm["date"] = pd.to_datetime(llm["timestamp"], utc=True, format="ISO8601").dt.tz_convert("America/New_York").dt.date
            llm_window = llm[
                (llm["date"] >= start_date)
                & llm["target_exposure"].notna()
                & llm["error"].isna()
            ].drop_duplicates(["date", "provider"], keep="last")

            for provider, key_final, key_pct in [
                ("claude", "claude_sim", "claude_sim_return_pct"),
                ("openai", "openai_sim", "openai_sim_return_pct"),
            ]:
                provider_df = llm_window[llm_window["provider"] == provider]
                if provider_df.empty:
                    continue
                target_by_date = dict(zip(provider_df["date"], provider_df["target_exposure"].astype(float)))
                values = _simulate_close_to_close(spy_window["close"], target_by_date)
                out[key_final] = float(values.iloc[-1])
                out[key_pct] = (float(values.iloc[-1]) / 100_000.0 - 1) * 100

    return out


def _simulate_close_to_close(closes: pd.Series, targets_by_date: dict, start_value: float = 100_000.0) -> pd.Series:
    """Carry forward target exposure across days; rebalance at each close."""
    values = [start_value]
    current_target = 0.0
    dates = [d.date() if hasattr(d, "date") else d for d in closes.index]
    for i in range(1, len(dates)):
        prev_date = dates[i - 1]
        if prev_date in targets_by_date:
            current_target = float(targets_by_date[prev_date])
        spy_ret = (closes.iloc[i] - closes.iloc[i - 1]) / closes.iloc[i - 1]
        values.append(values[-1] * (1.0 + current_target * spy_ret))
    return pd.Series(values, index=closes.index, name="value")


def _exposure_from_decisions(decisions_by_date: dict) -> dict:
    """Translate BUY/HOLD/SELL sequence into a 0/1 exposure map.

    HOLD carries the previous exposure forward. A leading HOLD (no prior BUY)
    starts at 0.0 — matches "you can't hold what you don't have." Unknown
    decision strings raise ValueError rather than silently passing through.
    """
    out: dict = {}
    current = 0.0
    for d in sorted(decisions_by_date):
        decision = decisions_by_date[d]
        if decision == "BUY":
            current = 1.0
        elif decision == "SELL":
            current = 0.0
        elif decision == "HOLD":
            pass  # carry current forward
        else:
            raise ValueError(
                f"Unknown decision {decision!r}; expected BUY, HOLD, or SELL"
            )
        out[d] = current
    return out


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
    if isinstance(closes.index, pd.DatetimeIndex):
        if closes.index.tz is not None:
            closes.index = closes.index.tz_localize(None)
        idx_dates = closes.index.date
    else:
        # Plain object Index of datetime.date values — already date-typed.
        idx_dates = closes.index.to_numpy()

    inception = df["date"].iloc[0]
    last = df["date"].iloc[-1]
    closes_window = closes[(idx_dates >= inception) & (idx_dates <= last)]
    if len(closes_window) < 2:
        return None

    values = _simulate_close_to_close(
        closes_window, exposure_by_date, start_value=100.0,
    )
    composite_pct = (float(values.iloc[-1]) / 100.0 - 1) * 100
    bnh_pct = (float(closes_window.iloc[-1]) / float(closes_window.iloc[0]) - 1) * 100

    # B&H value series rebased to 100 at inception, for basket-math reuse.
    bnh_values = closes_window / float(closes_window.iloc[0]) * 100.0

    return {
        "inception": inception,
        "sessions": len(df),
        "composite_return_pct": composite_pct,
        "bnh_return_pct": bnh_pct,
        "edge_pp": composite_pct - bnh_pct,
        "composite_value_series": values,
        "bnh_value_series": bnh_values,
    }


def _compute_basket_pnl(
    per_ticker_results: dict, weights: dict,
) -> dict | None:
    """Daily-rebalanced basket of per-ticker shadow results.

    per_ticker_results: {ticker: dict from _compute_ticker_shadow_pnl(...)}.
                        Each dict must contain 'composite_value_series' and
                        'bnh_value_series'.
    weights: {ticker: float} summing to 1.0.

    Returns None if any weighted ticker is missing or the aligned window has
    fewer than 2 sessions.
    """
    missing = [t for t in weights if t not in per_ticker_results]
    if missing:
        return None

    composite_panel = pd.DataFrame(
        {t: per_ticker_results[t]["composite_value_series"] for t in weights}
    ).dropna(how="any")
    bnh_panel = pd.DataFrame(
        {t: per_ticker_results[t]["bnh_value_series"] for t in weights}
    ).dropna(how="any")
    if len(composite_panel) < 2 or len(bnh_panel) < 2:
        return None

    w = pd.Series(weights)

    def _basket_return(panel: pd.DataFrame) -> float:
        daily = panel.pct_change().dropna()
        weighted = (daily * w).sum(axis=1)
        return (float((1.0 + weighted).cumprod().iloc[-1]) - 1) * 100

    composite_pct = _basket_return(composite_panel)
    bnh_pct = _basket_return(bnh_panel)
    sessions = len(composite_panel)
    inception = composite_panel.index[0]
    if hasattr(inception, "date"):
        inception = inception.date()

    return {
        "inception": inception,
        "sessions": sessions,
        "composite_return_pct": composite_pct,
        "bnh_return_pct": bnh_pct,
        "edge_pp": composite_pct - bnh_pct,
    }


def _fmt_edge(edge_pp: float | None) -> str:
    if edge_pp is None:
        return "—"
    # Snap sub-cent edges to 0 so float accumulation noise doesn't render
    # as "-0.00 pp" when composite and B&H are numerically identical.
    val = edge_pp if abs(edge_pp) >= 0.005 else 0.0
    return f"{val:+.2f} pp"


def build_sector_shadow_section(
    *,
    shadow_signals_path: Path,
    ticker_close_histories: dict,
    basket_weights: dict | None = None,
) -> str:
    """Build the SECTOR SHADOW table from shadow_signals.csv.

    Per-ticker rows show non-SPY tickers (SPY's real P&L is in PERFORMANCE
    above). When basket_weights is supplied AND every weighted ticker has
    enough shadow history, a BASKET row is appended showing the daily-
    rebalanced weighted combination.
    """
    if not shadow_signals_path.exists():
        return ""
    shadow = pd.read_csv(shadow_signals_path)
    if shadow.empty:
        return ""

    # Compute per-ticker results for everything in shadow (including SPY) so
    # the basket math has all ingredients. SPY's own row is filtered out below.
    all_results: dict[str, dict] = {}
    for ticker in sorted(shadow["ticker"].dropna().unique()):
        closes = ticker_close_histories.get(ticker)
        if closes is None or len(closes) == 0:
            continue
        ticker_df = shadow[shadow["ticker"] == ticker]
        result = _compute_ticker_shadow_pnl(ticker_df, closes)
        if result is not None:
            all_results[ticker] = result

    rows: list[tuple[str, dict]] = [
        (t, r) for t, r in all_results.items() if t != "SPY"
    ]

    basket_row: dict | None = None
    if basket_weights:
        basket_row = _compute_basket_pnl(all_results, basket_weights)

    if not rows and basket_row is None:
        return ""

    header = (
        f"  {'Ticker':<7} {'Since':<11} {'Sessions':>8}  "
        f"{'Composite':>9}  {'B&H':>9}  {'Edge':>9}"
    )
    sep = (
        f"  {'-'*7} {'-'*11} {'-'*8}  "
        f"{'-'*9}  {'-'*9}  {'-'*9}"
    )
    lines = [header, sep]
    for ticker, r in rows:
        lines.append(
            f"  {ticker:<7} {str(r['inception']):<11} {r['sessions']:>8}  "
            f"{_fmt_pct(r['composite_return_pct']):>9}  "
            f"{_fmt_pct(r['bnh_return_pct']):>9}  "
            f"{_fmt_edge(r['edge_pp']):>9}"
        )

    if basket_row is not None:
        weight_label = "/".join(
            f"{int(round(basket_weights[t]*100))}" for t in basket_weights
        )
        # Insert a separator before the basket so it visually stands out.
        lines.append(sep)
        lines.append(
            f"  {'BASKET':<7} {str(basket_row['inception']):<11} "
            f"{basket_row['sessions']:>8}  "
            f"{_fmt_pct(basket_row['composite_return_pct']):>9}  "
            f"{_fmt_pct(basket_row['bnh_return_pct']):>9}  "
            f"{_fmt_edge(basket_row['edge_pp']):>9}"
        )
        lines.append(
            f"           weights: {'/'.join(basket_weights.keys())} = {weight_label}"
        )

    lines.append("")
    lines.append("  Note: composite signal logged but not traded; numbers reflect")
    lines.append("        what binary exposure would have earned vs buy-and-hold.")
    return _section("SECTOR SHADOW (composite signal, not trading)", lines)


def build_performance_section(
    *,
    data_root: Path,
    spy_history: pd.DataFrame,
    start_date: date,
) -> str:
    perf = _compute_performance(data_root=data_root, spy_history=spy_history, start_date=start_date)
    if perf["sessions"] < 2:
        return _section(
            f"PERFORMANCE — Live trading since {start_date}",
            ["Not enough sessions yet for a meaningful comparison."],
        )

    rows = []
    rows.append(("System (real)", perf["system_real"], perf["system_real_return_pct"]))
    rows.append(("Claude (sim)",  perf["claude_sim"],  perf["claude_sim_return_pct"]))
    rows.append(("OpenAI (sim)",  perf["openai_sim"],  perf["openai_sim_return_pct"]))

    spy_pct = perf["spy_return_pct"]
    lines = [
        f"Start: {start_date}  ({perf['sessions']} sessions through today)",
        "",
        f"  {'Strategy':<14} {'Final':>12}  {'Return':>9}  {'vs SPY':>9}",
        f"  {'-'*14} {'-'*12}  {'-'*9}  {'-'*9}",
    ]
    for name, final_val, pct in rows:
        if final_val is None:
            lines.append(f"  {name:<14} {'N/A':>12}  {'N/A':>9}  {'N/A':>9}")
        else:
            vs_spy = (pct - spy_pct) if pct is not None and spy_pct is not None else None
            vs_spy_str = f"{vs_spy:+.2f} pp" if vs_spy is not None else "N/A"
            lines.append(
                f"  {name:<14} {_fmt_dollars(final_val):>12}  {_fmt_pct(pct):>9}  {vs_spy_str:>9}"
            )
    lines.append(f"  {'SPY B&H':<14} {'—':>12}  {_fmt_pct(spy_pct):>9}  {'—':>9}")
    lines.append("")
    lines.append("  Note: Claude/OpenAI are theoretical (close-to-close rebalancing).")
    lines.append("        Only 'System (real)' is your actual Alpaca P&L.")
    return _section(f"PERFORMANCE — Live trading since {start_date}", lines)


def build_basket_paper_section(
    *,
    portfolio_df: pd.DataFrame,
    shadow_signals_df: pd.DataFrame,
    basket_weights: dict,
    launch_date: date,
) -> str:
    """Render the BASKET PAPER section for the daily email.

    Reads the latest basket-pipeline snapshot from portfolio_df + shadow_signals_df
    and renders a per-ticker table plus any fired-stop warning notes.
    Returns empty string if there are no basket rows.
    """
    basket_pf = portfolio_df[portfolio_df["pipeline"] == "basket"].copy() if not portfolio_df.empty else portfolio_df
    if basket_pf.empty:
        return ""

    # Latest snapshot only
    latest_ts = basket_pf["timestamp"].max()
    latest_rows = basket_pf[basket_pf["timestamp"] == latest_ts]
    total_value = float(latest_rows.iloc[0]["total_value"])
    cash = float(latest_rows.iloc[0]["cash"])
    cash_pct = (cash / total_value * 100) if total_value > 0 else 0.0

    # Latest shadow_signals per ticker
    ss_by_ticker: dict = {}
    if not shadow_signals_df.empty:
        ss = shadow_signals_df[shadow_signals_df["pipeline"] == "basket"].copy()
        if not ss.empty:
            ss_latest = ss[ss["timestamp"] == ss["timestamp"].max()]
            for _, r in ss_latest.iterrows():
                ss_by_ticker[r["ticker"]] = r

    lines = [
        f"  Total value:  {_fmt_dollars(total_value)}",
        f"  Cash sleeve:  {_fmt_dollars(cash)} ({cash_pct:.1f}%)",
        "",
        f"  {'Ticker':<7} {'Target':>7} {'Actual':>7} {'Position':>9}  "
        f"{'Value':>10}  {'Signal':<14} {'Stop':>5}",
        f"  {'-'*7} {'-'*7} {'-'*7} {'-'*9}  {'-'*10}  {'-'*14} {'-'*5}",
    ]
    warnings: list[str] = []
    for ticker in basket_weights:
        rows_for_ticker = latest_rows[latest_rows["ticker"] == ticker]
        if rows_for_ticker.empty:
            continue
        r = rows_for_ticker.iloc[0]
        target_pct = basket_weights[ticker] * 100
        position_value = float(r["position_value"])
        actual_pct = (position_value / total_value * 100) if total_value > 0 else 0.0
        position_qty = int(r["position_qty"])
        ss_row = ss_by_ticker.get(ticker)
        signal = ss_row["ml_signal"] if ss_row is not None else "—"
        source = ss_row["signal_source"] if ss_row is not None else "—"
        stop_fired = bool(int(ss_row["trailing_stop_triggered"])) if (
            ss_row is not None and pd.notna(ss_row["trailing_stop_triggered"])
        ) else False
        stop_label = "FIRED" if stop_fired else "OK"
        lines.append(
            f"  {ticker:<7} {target_pct:>6.1f}% {actual_pct:>6.1f}% "
            f"{position_qty:>5} sh  {_fmt_dollars(position_value):>10}  "
            f"{signal+' ('+source+')':<14} {stop_label:>5}"
        )
        if stop_fired and ss_row is not None and pd.notna(ss_row["high_water_mark"]):
            warnings.append(
                f"  ⚠ {ticker} trailing stop fired (HWM {_fmt_dollars(float(ss_row['high_water_mark']), 2)}). "
                f"Cash held idle; re-entry after cooldown if signal allows."
            )

    if warnings:
        lines.append("")
        lines.extend(warnings)

    header = f"BASKET PAPER (paper-trading the basket since {launch_date})"
    return _section(header, lines)


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
    basket_weights: dict | None = None,
    basket_state: dict | None = None,
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

    if basket_state is not None:
        basket_section = build_basket_paper_section(
            portfolio_df=basket_state["portfolio_df"],
            shadow_signals_df=basket_state["shadow_signals_df"],
            basket_weights=basket_state["basket_weights"],
            launch_date=basket_state["launch_date"],
        )
        if basket_section:
            sections.append(basket_section)

    sector_section = build_sector_shadow_section(
        shadow_signals_path=data_root / "shadow_signals.csv",
        ticker_close_histories=sector_close_histories,
        basket_weights=basket_weights,
    )
    if sector_section:
        sections.append(sector_section)
    return "\n\n".join(sections) + "\n"
