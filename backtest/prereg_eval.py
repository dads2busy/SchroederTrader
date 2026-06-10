"""One-shot evaluation for PREREGISTRATION_levered_brake_v1.md.

Implements the frozen spec exactly: six exposure rules on SPY, lag-1 honest
timing, slippage + financing + cash-yield cost model, validity canaries, pass
gates, and the kill criterion. Run once; results stand.

    uv run python backtest/prereg_eval.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from schroeder_trader.risk.transaction_cost import estimate_slippage  # noqa: E402

DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results" / "prereg_v1"

LEVERAGE_CAP = 1.3
SOFT_FLOOR = 0.65
VOL_TARGET = 0.15
SMA_LEN = 200
VOL_LEN = 20
NO_TRADE_BAND = 0.10
FINANCING_SPREAD = 0.0040  # 40bp over DTB3, annual
DD_GATE = -0.20
TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_inputs() -> pd.DataFrame:
    spy = pd.read_csv(
        DATA_DIR / "spy_daily.csv", skiprows=3, index_col=0, parse_dates=True,
        names=["Date", "Close", "High", "Low", "Open", "Volume"],
    )["Close"].astype(float).sort_index()

    vix = pd.read_csv(DATA_DIR.parent / "data" / "features_daily.csv",
                      index_col="date", parse_dates=True)["vix_close"]
    dtb3 = pd.read_csv(DATA_DIR / "dtb3.csv", index_col="date", parse_dates=True)["dtb3"]

    df = pd.DataFrame({"close": spy})
    df["vix"] = vix.reindex(df.index).ffill()
    df["dtb3"] = dtb3.reindex(df.index).ffill() / 100.0  # percent -> fraction
    df["sma200"] = df["close"].rolling(SMA_LEN).mean()
    logret = np.log(df["close"] / df["close"].shift(1))
    df["vol20"] = logret.rolling(VOL_LEN).std() * np.sqrt(TRADING_DAYS)
    df["ret"] = df["close"].pct_change()
    # Evaluation window: first date all inputs exist (spec section 3)
    return df.dropna(subset=["vix", "dtb3", "sma200", "vol20"])


# ---------------------------------------------------------------------------
# Exposure rules (spec section 2) — each takes the input frame, returns the
# target exposure series decided at each close.
# ---------------------------------------------------------------------------

def targets(df: pd.DataFrame) -> dict[str, pd.Series]:
    above = df["close"] > df["sma200"]
    vol_exp = (VOL_TARGET / df["vol20"]).clip(0.0, LEVERAGE_CAP)
    return {
        "S0_buy_hold": pd.Series(1.0, index=df.index),
        "S1_faber_null": above.map({True: 1.0, False: 0.0}),
        "S2_trend_hard": above.map({True: LEVERAGE_CAP, False: 0.0}),
        "S3_trend_soft": above.map({True: LEVERAGE_CAP, False: SOFT_FLOOR}),
        "S4_vol_target": vol_exp,
        "S5_combo": vol_exp.where(above, 0.0),
    }


# ---------------------------------------------------------------------------
# Engine (spec section 3)
# ---------------------------------------------------------------------------

def simulate(df: pd.DataFrame, target: pd.Series, lag: int = 1) -> pd.DataFrame:
    """lag=1 (honest): exposure decided at close[t] earns close[t]->close[t+1].
    lag=0 (canary A only): exposure decided at close[t] earns the day-t return."""
    closes = df["close"].values
    vixs = df["vix"].values
    rates = df["dtb3"].values
    tgts = target.values
    rets = df["ret"].values  # ret[t] = close[t]/close[t-1] - 1

    n = len(df)
    value = 1.0
    exposure = 0.0
    trades = 0
    values = np.empty(n)
    exposures = np.empty(n)

    for t in range(n):
        if lag == 0 and t > 0:
            # look-ahead: today's decision earns today's return
            new = tgts[t]
            if abs(new - exposure) >= NO_TRADE_BAND:
                value *= 1 - estimate_slippage(vixs[t]) * abs(new - exposure)
                exposure = new
                trades += 1
            value *= 1 + exposure * rets[t]
            value *= 1 + rates[t] / TRADING_DAYS * max(0.0, 1 - exposure)
            value *= 1 - (rates[t] + FINANCING_SPREAD) / TRADING_DAYS * max(0.0, exposure - 1)
        elif lag == 1:
            # honest: earn today's return on yesterday's exposure first
            if t > 0:
                value *= 1 + exposure * rets[t]
                value *= 1 + rates[t - 1] / TRADING_DAYS * max(0.0, 1 - exposure)
                value *= 1 - (rates[t - 1] + FINANCING_SPREAD) / TRADING_DAYS * max(0.0, exposure - 1)
            new = tgts[t]
            if abs(new - exposure) >= NO_TRADE_BAND:
                value *= 1 - estimate_slippage(vixs[t]) * abs(new - exposure)
                exposure = new
                trades += 1
        values[t] = value
        exposures[t] = exposure

    out = pd.DataFrame({"value": values, "exposure": exposures}, index=df.index)
    out.attrs["trades"] = trades
    return out


# ---------------------------------------------------------------------------
# Metrics (spec section 3, reported set)
# ---------------------------------------------------------------------------

def metrics(curve: pd.DataFrame) -> dict:
    v = curve["value"]
    years = len(v) / TRADING_DAYS
    cagr = float(v.iloc[-1] ** (1 / years) - 1)
    daily = v.pct_change().dropna()
    sharpe = float(daily.mean() / daily.std() * np.sqrt(TRADING_DAYS)) if daily.std() > 0 else 0.0
    peak = v.cummax()
    dd = v / peak - 1
    underwater = (dd < 0).astype(int)
    # longest underwater spell (trading days)
    longest = 0
    run = 0
    for u in underwater:
        run = run + 1 if u else 0
        longest = max(longest, run)
    roll12 = (v / v.shift(TRADING_DAYS) - 1).dropna()
    return {
        "terminal_multiple": round(float(v.iloc[-1]), 2),
        "cagr_pct": round(cagr * 100, 2),
        "sharpe": round(sharpe, 2),
        "max_dd_pct": round(float(dd.min()) * 100, 2),
        "longest_underwater_days": int(longest),
        "worst_12m_pct": round(float(roll12.min()) * 100, 2) if len(roll12) else None,
        "trades": int(curve.attrs["trades"]),
        "time_above_1x_pct": round(float((curve["exposure"] > 1.0).mean()) * 100, 1),
    }


def sub_cagr(curve: pd.DataFrame, start: str, end: str | None = None) -> float | None:
    v = curve["value"].loc[start:end] if end else curve["value"].loc[start:]
    if len(v) < TRADING_DAYS:
        return None
    years = len(v) / TRADING_DAYS
    return round(float((v.iloc[-1] / v.iloc[0]) ** (1 / years) - 1) * 100, 2)


# ---------------------------------------------------------------------------
# Main: canaries first (spec section 5), then the one-shot evaluation
# ---------------------------------------------------------------------------

def main() -> None:
    df = load_inputs()
    print(f"Window: {df.index[0].date()} -> {df.index[-1].date()}  ({len(df)} sessions)\n")
    tgt = targets(df)

    # Canary A: S2 at lag-0 must beat S2 at lag-1
    a0 = metrics(simulate(df, tgt["S2_trend_hard"], lag=0))["cagr_pct"]
    a1 = metrics(simulate(df, tgt["S2_trend_hard"], lag=1))["cagr_pct"]
    ok_a = a0 > a1
    print(f"Canary A (look-ahead sensitivity): lag0 CAGR {a0:.2f}% vs lag1 {a1:.2f}%  -> {'PASS' if ok_a else 'FAIL'}")

    # Canary B: 100 seeded random exposures, median CAGR < B&H
    bh_curve = simulate(df, tgt["S0_buy_hold"], lag=1)
    bh = metrics(bh_curve)
    rand_cagrs = []
    for seed in range(100):
        rng = np.random.default_rng(seed)
        r = pd.Series(rng.choice([0.0, LEVERAGE_CAP], size=len(df)), index=df.index)
        rand_cagrs.append(metrics(simulate(df, r, lag=1))["cagr_pct"])
    med = float(np.median(rand_cagrs))
    ok_b = med < bh["cagr_pct"]
    print(f"Canary B (no free lunch): random median CAGR {med:.2f}% vs B&H {bh['cagr_pct']:.2f}%  -> {'PASS' if ok_b else 'FAIL'}")

    if not (ok_a and ok_b):
        print("\nVALIDITY CHECK FAILED — results must not be read. Investigate the harness.")
        sys.exit(1)

    # One-shot evaluation
    print("\n" + "=" * 100)
    results = {}
    curves = {}
    for name, t in tgt.items():
        curve = simulate(df, t, lag=1)
        curves[name] = curve
        results[name] = metrics(curve)

    header = f"{'Strategy':<16}{'Terminal':>9}{'CAGR':>8}{'Sharpe':>8}{'MaxDD':>9}{'Worst12m':>10}{'Underwater':>11}{'Trades':>8}{'>1x':>7}"
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        print(f"{name:<16}{m['terminal_multiple']:>8.1f}x{m['cagr_pct']:>7.2f}%{m['sharpe']:>8.2f}"
              f"{m['max_dd_pct']:>8.2f}%{m['worst_12m_pct']:>9.2f}%{m['longest_underwater_days']:>10}d"
              f"{m['trades']:>8}{m['time_above_1x_pct']:>6.1f}%")

    # Per-decade diagnostics (non-binding)
    print("\nPer-period CAGR % (diagnostic only):")
    periods = [("1994-2003", "1994-01-01", "2003-12-31"), ("2004-2013", "2004-01-01", "2013-12-31"),
               ("2014-end", "2014-01-01", None), ("2010-end", "2010-01-01", None)]
    print(f"{'Strategy':<16}" + "".join(f"{p[0]:>12}" for p in periods))
    for name, curve in curves.items():
        print(f"{name:<16}" + "".join(
            f"{(sub_cagr(curve, p[1], p[2]) if sub_cagr(curve, p[1], p[2]) is not None else float('nan')):>11.2f}%"
            for p in periods))

    # Gates (spec section 4)
    print("\n" + "=" * 100)
    print("GATES (all binding, full sample):")
    bh_cagr = results["S0_buy_hold"]["cagr_pct"]
    null_cagr = results["S1_faber_null"]["cagr_pct"]
    passers = []
    for name, m in results.items():
        if name == "S0_buy_hold":
            continue
        g1 = m["max_dd_pct"] >= DD_GATE * 100
        g2 = m["cagr_pct"] > bh_cagr
        g3 = m["cagr_pct"] > null_cagr or name == "S1_faber_null"
        verdict = "PASS" if (g1 and g2 and g3) else "fail"
        print(f"  {name:<16} DD>={DD_GATE*100:.0f}%: {'Y' if g1 else 'N'}   CAGR>B&H({bh_cagr:.2f}%): {'Y' if g2 else 'N'}   "
              f"CAGR>null({null_cagr:.2f}%): {'Y' if g3 else 'N'}   -> {verdict}")
        if g1 and g2 and g3:
            passers.append(name)

    if not passers:
        print("\nKILL CRITERION MET: no candidate passes. Program answer is NO —")
        print("adopt B&H with a written risk policy; SPY-timing research ends (spec section 4).")
    else:
        best = sorted(passers, key=lambda nm: (-results[nm]["cagr_pct"], results[nm]["trades"]))
        # tie-break within 0.25pp: fewer trades
        top = [nm for nm in best if results[best[0]]["cagr_pct"] - results[nm]["cagr_pct"] <= 0.25]
        winner = sorted(top, key=lambda nm: results[nm]["trades"])[0]
        print(f"\nWINNER: {winner}  (CAGR {results[winner]['cagr_pct']:.2f}% vs B&H {bh_cagr:.2f}%)")
        w2010 = sub_cagr(curves[winner], "2010-01-01")
        b2010 = sub_cagr(curves["S0_buy_hold"], "2010-01-01")
        fragile = w2010 is not None and b2010 is not None and w2010 < b2010
        print(f"Fragility flag (2010-end winner {w2010}% vs B&H {b2010}%): "
              f"{'FLAGGED - paper-trade gate extends to 12 months' if fragile else 'not flagged (6-month gate)'}")

    # Persist + hash
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {"window": [str(df.index[0].date()), str(df.index[-1].date())],
           "canaries": {"A_lag0_vs_lag1": [a0, a1], "B_random_median_vs_bh": [med, bh["cagr_pct"]]},
           "results": results, "passers": passers}
    path = RESULTS_DIR / "evaluation.json"
    path.write_text(json.dumps(out, indent=2))
    for name, curve in curves.items():
        curve.to_csv(RESULTS_DIR / f"{name}.csv")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    print(f"\nResults written to {path}  (sha256 {digest})")


if __name__ == "__main__":
    main()
