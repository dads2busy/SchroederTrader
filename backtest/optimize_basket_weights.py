"""Grid-search optimal basket weights across SPY + sector strategy curves.

Loads the four equity curves saved by backtest/sector_backtest.py, computes
each strategy's daily returns, then sweeps weight combinations on the simplex
(step 5%) under daily rebalancing. Reports per-objective winners and the
Pareto frontier on (Sharpe, total return, max drawdown).

Run:
    uv run python -m backtest.optimize_basket_weights

Optional --step 0.025 for a finer sweep. Default step 0.05 finishes in ~1s.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent / "results"
TICKERS = ["SPY", "XLK", "XLV", "XLE"]
ANNUALIZATION = 252


def load_returns() -> pd.DataFrame:
    """Return a (sessions, tickers) DataFrame of daily strategy returns."""
    series = {}
    for t in TICKERS:
        curve = pd.read_csv(
            RESULTS_DIR / f"sector_{t.lower()}_curve.csv",
            parse_dates=["date"],
        ).set_index("date")
        series[t] = curve["value"].pct_change()
    df = pd.DataFrame(series).dropna()
    return df


def enumerate_weights(step: float) -> np.ndarray:
    """All non-negative weight vectors over TICKERS that sum to 1, on a step grid."""
    steps = int(round(1.0 / step))
    out = []
    for a in range(steps + 1):
        for b in range(steps + 1 - a):
            for c in range(steps + 1 - a - b):
                d = steps - a - b - c
                out.append([a, b, c, d])
    return np.array(out, dtype=float) / steps


def metrics(daily_basket: pd.Series) -> tuple[float, float, float]:
    """Return (sharpe, total_return_pct, max_dd_pct) for a daily-return series."""
    mu = daily_basket.mean()
    sd = daily_basket.std(ddof=0)
    sharpe = (mu / sd) * np.sqrt(ANNUALIZATION) if sd > 0 else 0.0
    equity = (1.0 + daily_basket).cumprod()
    total = (equity.iloc[-1] - 1) * 100
    peak = equity.cummax()
    dd = (equity / peak - 1) * 100
    return float(sharpe), float(total), float(dd.min())


def is_pareto_optimal(row: np.ndarray, matrix: np.ndarray) -> bool:
    """row is (sharpe, ret, dd). Dominated iff exists r' with
    sharpe' >= sharpe AND ret' >= ret AND dd' >= dd AND strictly better on one."""
    sharpe, ret, dd = row
    s, r, d = matrix[:, 0], matrix[:, 1], matrix[:, 2]
    dominated = (s >= sharpe) & (r >= ret) & (d >= dd) & (
        (s > sharpe) | (r > ret) | (d > dd)
    )
    return not dominated.any()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=float, default=0.05,
                    help="Weight grid step (default 0.05 → 1771 combos for 4 assets).")
    ap.add_argument("--pareto-limit", type=int, default=20,
                    help="Max rows of Pareto frontier to print.")
    args = ap.parse_args()

    returns = load_returns()
    print(f"Loaded {len(returns)} aligned sessions for {TICKERS} "
          f"({returns.index.min().date()} → {returns.index.max().date()})")
    print()

    # Per-ticker baseline for reference
    print("Per-ticker baselines (system, 100% allocation):")
    print(f"  {'Ticker':<8}{'Sharpe':>8}{'Return %':>11}{'MaxDD %':>10}")
    for t in TICKERS:
        sharpe, ret, dd = metrics(returns[t])
        print(f"  {t:<8}{sharpe:>8.2f}{ret:>11.1f}{dd:>10.2f}")
    print()

    weights = enumerate_weights(args.step)
    n_combos = len(weights)
    print(f"Sweeping {n_combos} weight combinations (step={args.step})...")

    R = returns[TICKERS].to_numpy()             # (sessions, 4)
    basket_returns = R @ weights.T              # (sessions, n_combos)
    # Compute metrics column-wise
    mu = basket_returns.mean(axis=0)
    sd = basket_returns.std(axis=0, ddof=0)
    sharpe = np.where(sd > 0, (mu / sd) * np.sqrt(ANNUALIZATION), 0.0)
    equity = (1.0 + basket_returns).cumprod(axis=0)
    total = (equity[-1] - 1) * 100
    peak = np.maximum.accumulate(equity, axis=0)
    dd = (equity / peak - 1) * 100
    max_dd = dd.min(axis=0)

    summary = pd.DataFrame({
        **{t: weights[:, i] for i, t in enumerate(TICKERS)},
        "sharpe": sharpe,
        "return_pct": total,
        "max_dd_pct": max_dd,
    })

    # Single-objective winners
    print("\n=== Single-objective winners ===")
    for label, col, ascending in [
        ("Max Sharpe",     "sharpe",     False),
        ("Max Return",     "return_pct", False),
        ("Min Drawdown",   "max_dd_pct", False),  # max_dd is negative; "smallest" = closest to 0 = largest value
    ]:
        row = summary.sort_values(col, ascending=ascending).iloc[0]
        w = " / ".join(f"{t} {row[t]*100:>4.1f}%" for t in TICKERS)
        print(f"  {label:<14} {w}   "
              f"Sharpe {row['sharpe']:.2f}  Return {row['return_pct']:.1f}%  "
              f"MaxDD {row['max_dd_pct']:.2f}%")

    # Pareto frontier on (sharpe, return, -max_dd) where higher is better on all three.
    # Convert max_dd to "higher is better" by negating.
    obj = np.stack([sharpe, total, max_dd], axis=1)  # (n, 3); higher max_dd (less negative) = better
    pareto_mask = np.array([
        is_pareto_optimal(obj[i], obj) for i in range(len(obj))
    ])
    frontier = summary[pareto_mask].sort_values("sharpe", ascending=False)
    print(f"\n=== Pareto frontier ({len(frontier)} combinations, top {args.pareto_limit} by Sharpe) ===")
    print(f"  {'SPY':>5}{'XLK':>6}{'XLV':>6}{'XLE':>6}  "
          f"{'Sharpe':>8}{'Return %':>11}{'MaxDD %':>10}")
    for _, row in frontier.head(args.pareto_limit).iterrows():
        print(f"  {row['SPY']*100:>5.1f}{row['XLK']*100:>6.1f}{row['XLV']*100:>6.1f}{row['XLE']*100:>6.1f}  "
              f"{row['sharpe']:>8.2f}{row['return_pct']:>11.1f}{row['max_dd_pct']:>10.2f}")

    # Persist full sweep so the user can slice it themselves
    out_path = RESULTS_DIR / "basket_weight_sweep.csv"
    summary.to_csv(out_path, index=False)
    print(f"\nFull sweep saved to {out_path.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
