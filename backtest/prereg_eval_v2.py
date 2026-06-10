"""One-shot evaluation for PREREGISTRATION_levered_diversified_v2.md.

Static multi-asset portfolios at fixed leverage, monthly rebalancing, honest
timing, slippage + financing + cash-yield cost model. Run once; results stand.

    uv run python backtest/prereg_eval_v2.py
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
RESULTS_DIR = Path(__file__).parent / "results" / "prereg_v2"

ASSETS = ["SPY", "IEF", "TLT", "GLD"]
FINANCING_SPREAD = 0.0040
EMERGENCY_GROSS = 1.40
DD_GATE = -20.0
TIE_BAND_PP = 0.25
TRADING_DAYS = 252

CANDIDATES: dict[str, tuple[dict[str, float], float]] = {
    # name: (target weights summing to 1.0, leverage)
    "S0_spy_bh":      ({"SPY": 1.00}, 1.0),
    "S1_6040_null":   ({"SPY": 0.60, "IEF": 0.40}, 1.0),
    "S2_6040_13x":    ({"SPY": 0.60, "IEF": 0.40}, 1.3),
    "S3_ief_gold_13x": ({"SPY": 0.55, "IEF": 0.35, "GLD": 0.10}, 1.3),
    "S4_tlt_13x":     ({"SPY": 0.60, "TLT": 0.40}, 1.3),
    "S5_tlt_gold_13x": ({"SPY": 0.55, "TLT": 0.35, "GLD": 0.10}, 1.3),
}
DIAGNOSTIC = {"DIAG_spy_13x": ({"SPY": 1.00}, 1.3)}  # not a candidate (spec section 2)


def _load_close(ticker: str) -> pd.Series:
    df = pd.read_csv(
        DATA_DIR / f"{ticker.lower()}_daily.csv", skiprows=3, index_col=0,
        parse_dates=True, names=["Date", "Close", "High", "Low", "Open", "Volume"],
    )["Close"].astype(float).sort_index()
    return df


def load_inputs() -> pd.DataFrame:
    closes = pd.DataFrame({t: _load_close(t) for t in ASSETS})
    vix = pd.read_csv(DATA_DIR / "features_daily.csv", index_col="date",
                      parse_dates=True)["vix_close"]
    dtb3 = pd.read_csv(DATA_DIR / "dtb3.csv", index_col="date", parse_dates=True)["dtb3"]
    df = closes.copy()
    df["vix"] = vix.reindex(df.index).ffill()
    df["dtb3"] = dtb3.reindex(df.index).ffill() / 100.0
    df = df.dropna()  # window start: first date all inputs exist (spec section 3)
    # Window end: latest COMMON completed date — truncate at the last raw (non-ffilled)
    # VIX close so an in-progress intraday asset bar can't enter the one-shot run.
    df = df.loc[:vix.last_valid_index()]
    return df


def simulate(df: pd.DataFrame, weights: dict[str, float], leverage: float) -> pd.DataFrame:
    """Monthly-rebalanced static mix at fixed leverage, honest timing.

    Day-t asset returns accrue to the notional weights held coming into day t;
    a rebalance decided at close[t] (calendar or emergency, both knowable at t)
    takes effect for day t+1. Financing/cash use DTB3[t-1] (knowable).
    """
    rets = df[ASSETS].pct_change().values
    vixs = df["vix"].values
    rates = df["dtb3"].values
    months = df.index.month
    n = len(df)

    target = np.array([weights.get(a, 0.0) * leverage for a in ASSETS])
    nw = target.copy()  # notional weights as fraction of equity
    value = 1.0
    rebalances = 0
    values = np.empty(n)
    gross_hist = np.empty(n)

    for t in range(n):
        if t > 0:
            gross = nw.sum()
            r_assets = float(np.dot(nw, np.nan_to_num(rets[t])))
            financing = (rates[t - 1] + FINANCING_SPREAD) / TRADING_DAYS * max(0.0, gross - 1)
            cash = rates[t - 1] / TRADING_DAYS * max(0.0, 1 - gross)
            r_p = r_assets - financing + cash
            value *= 1 + r_p
            # drift: new equity scales by (1+r_p); notional_i scales by (1+r_i)
            nw = nw * (1 + np.nan_to_num(rets[t])) / (1 + r_p)

            first_of_month = months[t] != months[t - 1]
            if first_of_month or nw.sum() >= EMERGENCY_GROSS:
                turnover = float(np.abs(target - nw).sum())
                if turnover > 0:
                    value *= 1 - estimate_slippage(vixs[t]) * turnover
                    rebalances += 1
                nw = target.copy()
        values[t] = value
        gross_hist[t] = nw.sum()

    out = pd.DataFrame({"value": values, "gross": gross_hist}, index=df.index)
    out.attrs["rebalances"] = rebalances
    return out


def metrics(curve: pd.DataFrame) -> dict:
    v = curve["value"]
    years = len(v) / TRADING_DAYS
    cagr = float(v.iloc[-1] ** (1 / years) - 1)
    daily = v.pct_change().dropna()
    sharpe = float(daily.mean() / daily.std() * np.sqrt(TRADING_DAYS)) if daily.std() > 0 else 0.0
    dd = v / v.cummax() - 1
    roll12 = (v / v.shift(TRADING_DAYS) - 1).dropna()
    return {
        "terminal_multiple": round(float(v.iloc[-1]), 2),
        "cagr_pct": round(cagr * 100, 2),
        "sharpe": round(sharpe, 2),
        "max_dd_pct": round(float(dd.min()) * 100, 2),
        "worst_12m_pct": round(float(roll12.min()) * 100, 2) if len(roll12) else None,
        "rebalances": int(curve.attrs["rebalances"]),
        "avg_gross": round(float(curve["gross"].mean()), 3),
    }


def sub_cagr(curve: pd.DataFrame, start: str, end: str | None = None) -> float | None:
    v = curve["value"].loc[start:end] if end else curve["value"].loc[start:]
    if len(v) < TRADING_DAYS:
        return None
    years = len(v) / TRADING_DAYS
    return round(float((v.iloc[-1] / v.iloc[0]) ** (1 / years) - 1) * 100, 2)


def main() -> None:
    df = load_inputs()
    print(f"Window: {df.index[0].date()} -> {df.index[-1].date()}  ({len(df)} sessions)\n")

    # Validity checks (spec section 5) BEFORE results are read
    s0_curve = simulate(df, *CANDIDATES["S0_spy_bh"])
    raw = float(df["SPY"].iloc[-1] / df["SPY"].iloc[0])
    eng = float(s0_curve["value"].iloc[-1])
    ok_v1 = abs(eng / raw - 1) < 0.001
    print(f"V1 (engine identity): engine {eng:.4f} vs raw {raw:.4f}  -> {'PASS' if ok_v1 else 'FAIL'}")

    diag_curve = simulate(df, *DIAGNOSTIC["DIAG_spy_13x"])
    diag_m = metrics(diag_curve)
    s0_m = metrics(s0_curve)
    ok_v2 = (abs(diag_m["avg_gross"] - 1.3) <= 0.05) and (diag_m["max_dd_pct"] < s0_m["max_dd_pct"])
    print(f"V2 (leverage realism): avg gross {diag_m['avg_gross']:.3f}, "
          f"DD {diag_m['max_dd_pct']:.2f}% vs SPY {s0_m['max_dd_pct']:.2f}%  -> {'PASS' if ok_v2 else 'FAIL'}")

    if not (ok_v1 and ok_v2):
        print("\nVALIDITY CHECK FAILED — results must not be read.")
        sys.exit(1)

    print("\n" + "=" * 96)
    curves = {"S0_spy_bh": s0_curve}
    results = {"S0_spy_bh": s0_m}
    for name, (w, lev) in list(CANDIDATES.items())[1:]:
        curves[name] = simulate(df, w, lev)
        results[name] = metrics(curves[name])
    diag = {"DIAG_spy_13x": diag_m}

    header = f"{'Strategy':<17}{'Terminal':>9}{'CAGR':>8}{'Sharpe':>8}{'MaxDD':>9}{'Worst12m':>10}{'Rebals':>8}{'AvgGross':>10}"
    print(header)
    print("-" * len(header))
    for name, m in {**results, **diag}.items():
        print(f"{name:<17}{m['terminal_multiple']:>8.1f}x{m['cagr_pct']:>7.2f}%{m['sharpe']:>8.2f}"
              f"{m['max_dd_pct']:>8.2f}%{m['worst_12m_pct']:>9.2f}%{m['rebalances']:>8}{m['avg_gross']:>10.3f}")

    print("\nPer-period CAGR % (diagnostic only):")
    periods = [("2004-2013", "2004-01-01", "2013-12-31"), ("2014-end", "2014-01-01", None),
               ("2015-end", "2015-01-01", None)]
    print(f"{'Strategy':<17}" + "".join(f"{p[0]:>12}" for p in periods))
    for name, curve in curves.items():
        print(f"{name:<17}" + "".join(f"{sub_cagr(curve, p[1], p[2]):>11.2f}%" for p in periods))

    print("\n" + "=" * 96)
    print("GATES (all binding, full sample):")
    bh = results["S0_spy_bh"]["cagr_pct"]
    null = results["S1_6040_null"]["cagr_pct"]
    passers = []
    for name, m in results.items():
        if name == "S0_spy_bh":
            continue
        g1 = m["max_dd_pct"] >= DD_GATE
        g2 = m["cagr_pct"] > bh
        g3 = m["cagr_pct"] > null or name == "S1_6040_null"
        ok = g1 and g2 and g3
        print(f"  {name:<17} DD>={DD_GATE:.0f}%: {'Y' if g1 else 'N'}   CAGR>S0({bh:.2f}%): {'Y' if g2 else 'N'}   "
              f"CAGR>S1({null:.2f}%): {'Y' if g3 else 'N'}   -> {'PASS' if ok else 'fail'}")
        if ok:
            passers.append(name)

    if not passers:
        print("\nKILL CRITERION MET: no candidate passes. Registered conclusion: the")
        print("constraint pair (beat SPY, maxDD <= -25%, leverage <= 1.3x) is infeasible")
        print("by static allocation with these assets (spec section 4).")
    else:
        best = sorted(passers, key=lambda nm: -results[nm]["cagr_pct"])
        top = [nm for nm in best if results[best[0]]["cagr_pct"] - results[nm]["cagr_pct"] <= TIE_BAND_PP]
        winner = sorted(top, key=lambda nm: results[nm]["rebalances"])[0]
        print(f"\nWINNER: {winner}  (CAGR {results[winner]['cagr_pct']:.2f}% vs SPY {bh:.2f}%)")
        w15, b15 = sub_cagr(curves[winner], "2015-01-01"), sub_cagr(curves["S0_spy_bh"], "2015-01-01")
        fragile = w15 is not None and b15 is not None and w15 < b15
        print(f"Fragility flag (2015-end {w15}% vs SPY {b15}%): "
              f"{'FLAGGED - 12-month paper gate' if fragile else 'not flagged (6-month gate)'}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {"window": [str(df.index[0].date()), str(df.index[-1].date())],
           "validity": {"V1_engine_vs_raw": [eng, raw], "V2_diag": diag_m},
           "results": results, "diagnostic": diag, "passers": passers}
    path = RESULTS_DIR / "evaluation.json"
    path.write_text(json.dumps(out, indent=2))
    for name, curve in curves.items():
        curve.to_csv(RESULTS_DIR / f"{name}.csv")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    print(f"\nResults written to {path}  (sha256 {digest})")


if __name__ == "__main__":
    main()
