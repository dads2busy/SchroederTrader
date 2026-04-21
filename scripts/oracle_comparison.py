"""Compare system vs LLM oracle strategies using close-to-close simulation.

Reads target exposures from:
  - llm_shadow_signals (Claude + ChatGPT target_exposure per day)
  - shadow_signals (system's composite signal, mapped to 0.98/0.0)

Simulates each strategy starting at $100k at the earliest overlapping date.
At each day's close, rebalance to the strategy's target_exposure. Days without
a target carry forward the previous exposure.

Usage:
    uv run python scripts/oracle_comparison.py
"""

import sqlite3
import sys
from datetime import datetime

import pandas as pd

from schroeder_trader.config import DB_PATH, TICKER
from schroeder_trader.data.market_data import fetch_daily_bars


_START_VALUE = 100_000.0
_SYSTEM_TARGET_BY_SIGNAL = {"BUY": 0.98, "SELL": 0.0}


def _to_et_date(ts_series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(ts_series, utc=True)
    return ts.dt.tz_convert("America/New_York").dt.date


def load_llm_targets(conn) -> pd.DataFrame:
    df = pd.read_sql(
        "SELECT timestamp, provider, target_exposure FROM llm_shadow_signals "
        "WHERE target_exposure IS NOT NULL AND error IS NULL",
        conn,
    )
    if df.empty:
        return df.assign(date=pd.Series(dtype="object"))
    df["date"] = _to_et_date(df["timestamp"])
    return df[["date", "provider", "target_exposure"]].drop_duplicates(["date", "provider"], keep="last")


def load_system_targets(conn) -> pd.DataFrame:
    df = pd.read_sql("SELECT timestamp, ml_signal FROM shadow_signals", conn)
    if df.empty:
        return df.assign(date=pd.Series(dtype="object"), target_exposure=pd.Series(dtype=float))
    df["date"] = _to_et_date(df["timestamp"])
    df = df.sort_values("date")
    df["target_exposure"] = df["ml_signal"].map(_SYSTEM_TARGET_BY_SIGNAL)
    df["target_exposure"] = df["target_exposure"].ffill().fillna(0.0)
    return df[["date", "target_exposure"]].drop_duplicates("date", keep="last")


def simulate(spy_closes: pd.Series, target_by_date: dict, start_value: float = _START_VALUE) -> pd.Series:
    """Rebalance to target at each day's close. Returns per-day portfolio value."""
    values = [start_value]
    current_target = 0.0
    dates = spy_closes.index.tolist()
    for i in range(1, len(dates)):
        prev_date = dates[i - 1]
        if prev_date in target_by_date:
            current_target = float(target_by_date[prev_date])
        spy_ret = (spy_closes.iloc[i] - spy_closes.iloc[i - 1]) / spy_closes.iloc[i - 1]
        values.append(values[-1] * (1.0 + current_target * spy_ret))
    return pd.Series(values, index=dates, name="value")


def summary_stats(series: pd.Series) -> dict:
    if len(series) < 2:
        return {"total_return_pct": 0.0, "max_dd_pct": 0.0, "days": len(series)}
    total_return = (series.iloc[-1] / series.iloc[0] - 1.0) * 100
    peaks = series.expanding().max()
    dd = (series / peaks - 1.0) * 100
    return {
        "total_return_pct": total_return,
        "max_dd_pct": dd.min(),
        "days": len(series),
    }


def main():
    conn = sqlite3.connect(DB_PATH)
    llm_df = load_llm_targets(conn)
    sys_df = load_system_targets(conn)
    conn.close()

    all_dates = pd.concat([llm_df["date"], sys_df["date"]]) if not llm_df.empty or not sys_df.empty else pd.Series(dtype="object")
    if all_dates.empty:
        print("No signal data in DB yet.")
        sys.exit(0)

    start_date = min(all_dates)
    today = datetime.now().date()

    days_back = (today - start_date).days + 10
    spy = fetch_daily_bars(TICKER, days=max(days_back, 30))
    spy.index = pd.to_datetime(spy.index).tz_convert("America/New_York").date
    spy = spy[(spy.index >= start_date) & (spy.index <= today)]
    if len(spy) < 2:
        print(f"Not enough SPY bars in range {start_date} → {today} (got {len(spy)}).")
        sys.exit(0)

    spy_closes = spy["close"].astype(float)

    system_map = dict(zip(sys_df["date"], sys_df["target_exposure"]))
    claude_map = dict(zip(
        llm_df[llm_df["provider"] == "claude"]["date"],
        llm_df[llm_df["provider"] == "claude"]["target_exposure"],
    ))
    openai_map = dict(zip(
        llm_df[llm_df["provider"] == "openai"]["date"],
        llm_df[llm_df["provider"] == "openai"]["target_exposure"],
    ))

    strategies = [
        ("System", simulate(spy_closes, system_map)),
        ("Claude", simulate(spy_closes, claude_map)),
        ("OpenAI", simulate(spy_closes, openai_map)),
    ]

    spy_ret_pct = (spy_closes.iloc[-1] / spy_closes.iloc[0] - 1.0) * 100

    print()
    print(f"=== Oracle Comparison  {spy_closes.index[0]} → {spy_closes.index[-1]}  ({len(spy_closes)} sessions) ===")
    print(f"Baseline ${_START_VALUE:,.0f}.  Close-to-close rebalancing (theoretical).")
    print(f"SPY: ${spy_closes.iloc[0]:.2f} → ${spy_closes.iloc[-1]:.2f}  ({spy_ret_pct:+.2f}%)")
    print()

    # Data-coverage note
    first_signal = {
        "System": min(system_map) if system_map else None,
        "Claude": min(claude_map) if claude_map else None,
        "OpenAI": min(openai_map) if openai_map else None,
    }
    coverage_notes = [
        f"{name} first signal: {d or 'no data'}" for name, d in first_signal.items()
    ]
    print("Data coverage: " + "  |  ".join(coverage_notes))
    print()

    # Per-day table
    header = f"{'Date':<12} {'SPY':>8} {'Sys%':>5} {'System $':>11} {'Cla%':>5} {'Claude $':>11} {'OAI%':>5} {'OpenAI $':>11}"
    print(header)
    print("-" * len(header))

    sys_prev, cla_prev, oai_prev = 0.0, 0.0, 0.0
    sys_values, cla_values, oai_values = strategies[0][1], strategies[1][1], strategies[2][1]
    for d in spy_closes.index:
        sys_prev = system_map.get(d, sys_prev)
        cla_prev = claude_map.get(d, cla_prev)
        oai_prev = openai_map.get(d, oai_prev)
        print(
            f"{str(d):<12} "
            f"${spy_closes.loc[d]:>7.2f} "
            f"{sys_prev*100:>4.0f}% "
            f"${sys_values.loc[d]:>9,.0f} "
            f"{cla_prev*100:>4.0f}% "
            f"${cla_values.loc[d]:>9,.0f} "
            f"{oai_prev*100:>4.0f}% "
            f"${oai_values.loc[d]:>9,.0f}"
        )

    print()
    summary_header = f"{'Strategy':<10} {'Final':>11} {'Return':>9} {'MaxDD':>9} {'Sessions':>10}"
    print(summary_header)
    print("-" * len(summary_header))
    for name, series in strategies:
        stats = summary_stats(series)
        print(
            f"{name:<10} "
            f"${series.iloc[-1]:>9,.0f} "
            f"{stats['total_return_pct']:>8.2f}% "
            f"{stats['max_dd_pct']:>8.2f}% "
            f"{stats['days']:>10d}"
        )
    print(f"{'SPY B&H':<10} {'':>11} {spy_ret_pct:>8.2f}%")
    print()

    if any(v is None for v in first_signal.values()):
        print("⚠ Some strategies have no signals yet — comparison is preliminary.")
    elif max((d for d in first_signal.values() if d is not None)) > min((d for d in first_signal.values() if d is not None)):
        latest_start = max(d for d in first_signal.values() if d is not None)
        print(f"⚠ Strategies don't share a start date — earliest common session is {latest_start}.")
        print("   Comparison will firm up as more LLM signals accumulate.")
    print()


if __name__ == "__main__":
    main()
