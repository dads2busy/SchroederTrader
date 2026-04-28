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
    return "\n\n".join(sections) + "\n"
