"""Walk-forward validation of basket-weight optimization.

Two questions answered side-by-side:

1. Does the optimization PROCESS generalize? — For each rolling window,
   pick max-Sharpe weights on past TRAIN_YEARS of history, hold them for
   the next EVAL_MONTHS, stitch the out-of-sample returns into one
   equity curve. If walk-forward Sharpe ≈ in-sample Sharpe, the process
   generalizes.

2. Does the fixed 40/40/10/10 mix hold up? — Apply that exact mix to the
   same out-of-sample evaluation windows. Compare against the re-optimized
   walk-forward path and the per-window in-sample picks.

Run:
    uv run python -m backtest.walk_forward_basket
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.optimize_basket_weights import (
    ANNUALIZATION,
    RESULTS_DIR,
    TICKERS,
    enumerate_weights,
    load_returns,
    metrics,
)

TRAIN_YEARS = 5
EVAL_MONTHS = 12       # length of each evaluation window
STEP_MONTHS = 6        # advance origin by this many months between windows
STEP = 0.05
FIXED_WEIGHTS = np.array([0.45, 0.30, 0.15, 0.10])  # SPY, XLK, XLV, XLE


@dataclass
class Window:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    eval_start: pd.Timestamp
    eval_end: pd.Timestamp


def build_windows(index: pd.DatetimeIndex) -> list[Window]:
    """Rolling-origin windows: TRAIN_YEARS train → EVAL_MONTHS evaluate, step STEP_MONTHS.

    When STEP_MONTHS < EVAL_MONTHS the evaluation windows overlap — each per-window
    Sharpe is computed on an independent 12-month slice but adjacent windows share
    months. The stitched OOS curve is built from non-overlapping sub-windows so the
    cumulative return number stays interpretable.
    """
    out: list[Window] = []
    start = index[0]
    cursor = start + pd.DateOffset(years=TRAIN_YEARS)
    while True:
        train_start = start
        train_end = cursor - pd.Timedelta(days=1)
        eval_start = cursor
        eval_end = cursor + pd.DateOffset(months=EVAL_MONTHS) - pd.Timedelta(days=1)
        if eval_end > index[-1]:
            break
        out.append(Window(train_start, train_end, eval_start, eval_end))
        cursor = cursor + pd.DateOffset(months=STEP_MONTHS)
    return out


def pick_max_sharpe_weights(returns: pd.DataFrame, weights_grid: np.ndarray) -> np.ndarray:
    """Return the weight vector with the highest Sharpe over the given returns."""
    R = returns[TICKERS].to_numpy()
    basket = R @ weights_grid.T
    mu = basket.mean(axis=0)
    sd = basket.std(axis=0, ddof=0)
    sharpe = np.where(sd > 0, (mu / sd) * np.sqrt(ANNUALIZATION), 0.0)
    return weights_grid[int(np.argmax(sharpe))]


def main():
    returns = load_returns()
    weights_grid = enumerate_weights(STEP)
    windows = build_windows(returns.index)

    print(f"Walk-forward setup: train={TRAIN_YEARS}y, eval={EVAL_MONTHS}m, "
          f"step={STEP_MONTHS}m, weight grid step={STEP}")
    print(f"Data: {returns.index.min().date()} → {returns.index.max().date()} "
          f"({len(returns)} sessions)")
    print(f"Windows: {len(windows)}  "
          f"(first eval {windows[0].eval_start.date()}, "
          f"last eval ends {windows[-1].eval_end.date()})")
    print()

    rows: list[dict] = []
    optimized_oos = []
    fixed_oos = []

    for w in windows:
        train_mask = (returns.index >= w.train_start) & (returns.index <= w.train_end)
        eval_mask = (returns.index >= w.eval_start) & (returns.index <= w.eval_end)
        train_ret = returns.loc[train_mask]
        eval_ret = returns.loc[eval_mask]
        if len(train_ret) < 252 or eval_ret.empty:
            continue

        chosen = pick_max_sharpe_weights(train_ret, weights_grid)
        opt_eval_returns = eval_ret[TICKERS].to_numpy() @ chosen
        fix_eval_returns = eval_ret[TICKERS].to_numpy() @ FIXED_WEIGHTS

        opt_sharpe = metrics(pd.Series(opt_eval_returns))[0]
        fix_sharpe = metrics(pd.Series(fix_eval_returns))[0]

        rows.append({
            "eval_start": w.eval_start.date(),
            "eval_end": w.eval_end.date(),
            **{f"w_{t}": chosen[i] for i, t in enumerate(TICKERS)},
            "opt_sharpe_oos": opt_sharpe,
            "fixed_sharpe_oos": fix_sharpe,
        })
        # For the stitched OOS curve, take only the first STEP_MONTHS of each
        # window so adjacent windows don't double-count overlapping months.
        stitch_end = w.eval_start + pd.DateOffset(months=STEP_MONTHS) - pd.Timedelta(days=1)
        stitch_mask = np.asarray(
            (eval_ret.index >= w.eval_start) & (eval_ret.index <= stitch_end)
        )
        if stitch_mask.any():
            stitch_idx = eval_ret.index[stitch_mask]
            optimized_oos.append(pd.Series(opt_eval_returns[stitch_mask], index=stitch_idx))
            fixed_oos.append(pd.Series(fix_eval_returns[stitch_mask], index=stitch_idx))

    df_windows = pd.DataFrame(rows)
    print("=== Per-window picks (max-Sharpe on training window) ===")
    print(f"  {'Eval period':<25} {'SPY':>5}{'XLK':>6}{'XLV':>6}{'XLE':>6}  "
          f"{'OptOOS':>8}{'FixedOOS':>10}")
    for _, r in df_windows.iterrows():
        print(f"  {str(r['eval_start'])} → {str(r['eval_end'])}  "
              f"{r['w_SPY']*100:>5.0f}{r['w_XLK']*100:>6.0f}"
              f"{r['w_XLV']*100:>6.0f}{r['w_XLE']*100:>6.0f}  "
              f"{r['opt_sharpe_oos']:>8.2f}{r['fixed_sharpe_oos']:>10.2f}")
    print()

    opt_curve = pd.concat(optimized_oos).sort_index()
    fix_curve = pd.concat(fixed_oos).sort_index()

    # Per-window standard error of the difference (fixed - re-optimized).
    # If |mean_diff| > 2 SE, the fixed mix is meaningfully better at α≈0.05.
    diff = df_windows["fixed_sharpe_oos"] - df_windows["opt_sharpe_oos"]
    mean_diff = diff.mean()
    se_diff = diff.std(ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else float("nan")
    wins = int((diff > 0).sum())
    print(f"  Per-window Sharpe(fixed) − Sharpe(re-opt):  "
          f"mean {mean_diff:+.3f}  SE {se_diff:.3f}  "
          f"(fixed wins {wins}/{len(diff)})")
    print()

    print("=== Stitched out-of-sample performance (non-overlapping sub-windows) ===")
    print(f"  {'Strategy':<32}{'Sharpe':>8}{'Return %':>12}{'MaxDD %':>10}")
    candidate_curves = [
        ("Walk-forward re-optimized", opt_curve),
        (f"Fixed {'/'.join(f'{int(round(w*100))}' for w in FIXED_WEIGHTS)}", fix_curve),
    ]
    # Also run each comparison candidate over the same stitched index
    extra_candidates = {
        "40/40/10/10 (max in-sample return)": np.array([0.40, 0.40, 0.10, 0.10]),
        "35/35/20/10 (in-sample max-Sharpe)": np.array([0.35, 0.35, 0.20, 0.10]),
        "Equal weight 25/25/25/25":           np.array([0.25, 0.25, 0.25, 0.25]),
    }
    stitch_index = opt_curve.index
    aligned_returns = returns.loc[stitch_index]
    for label, w in extra_candidates.items():
        candidate_curves.append((
            label,
            pd.Series(aligned_returns[TICKERS].to_numpy() @ w, index=stitch_index),
        ))
    for label, series in candidate_curves:
        s, r, dd = metrics(series)
        print(f"  {label:<32}{s:>8.2f}{r:>12.1f}{dd:>10.2f}")

    # In-sample benchmark on the same OOS date range
    oos_index = opt_curve.index
    in_sample_section = returns.loc[oos_index]
    chosen_in_sample = pick_max_sharpe_weights(returns, weights_grid)
    in_sample_curve = pd.Series(
        in_sample_section[TICKERS].to_numpy() @ chosen_in_sample,
        index=oos_index,
    )
    s, r, dd = metrics(in_sample_curve)
    in_sample_w = " / ".join(f"{t} {chosen_in_sample[i]*100:>2.0f}%" for i, t in enumerate(TICKERS))
    print(f"\n  In-sample optimal ({in_sample_w}) over same window:  "
          f"Sharpe {s:.2f}  Return {r:.1f}%  MaxDD {dd:.2f}%")

    # Save artifacts
    df_windows.to_csv(RESULTS_DIR / "walk_forward_windows.csv", index=False)
    pd.DataFrame({"optimized": opt_curve, "fixed_40_40_10_10": fix_curve}).to_csv(
        RESULTS_DIR / "walk_forward_oos_returns.csv"
    )
    print(f"\nSaved: walk_forward_windows.csv, walk_forward_oos_returns.csv")


if __name__ == "__main__":
    main()
