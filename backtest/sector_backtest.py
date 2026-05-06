"""Backtest the composite (threshold + SMA + XGBoost) system on sector ETFs.

For each ticker:
  1. Download adjusted history from yfinance (cached).
  2. Compute the same 6-feature set used by the composite model.
  3. Run walk-forward XGBoost training (2-year train, 6-month test, retrain).
  4. Combine with SMA crossover and the threshold regime detector via the
     same composite_signal_hybrid as production.
  5. Simulate a portfolio with 98% binary sizing and a 10% trailing stop.
  6. Report Sharpe, total return, max drawdown.

Saves per-ticker equity curves to backtest/results/sector_<ticker>_curve.csv
so we can inspect drawdown timing across sectors.

Usage:
    uv run python backtest/sector_backtest.py SPY XLK XLE XLV
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from xgboost import XGBClassifier

from schroeder_trader.risk.transaction_cost import estimate_slippage
from schroeder_trader.strategy.composite import (
    composite_signal_hybrid,
    count_consecutive_bear_days,
)
from schroeder_trader.strategy.feature_engineer import (
    CLASS_DOWN,
    CLASS_FLAT,
    CLASS_UP,
    FeaturePipeline,
)
from schroeder_trader.strategy.regime_detector import (
    Regime,
    compute_regime_labels,
    compute_regime_series,
)
from schroeder_trader.strategy.sma_crossover import Signal


# Hyperparameters — match the SPY production config
XGB_FEATURES = [
    "log_return_5d", "log_return_20d", "volatility_20d",
    "credit_spread", "dollar_momentum", "regime_label",
]
XGB_PARAMS = {
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "random_state": 42,
}
LABEL_THRESHOLD = 0.01      # 1% over 20 trading days
TRAIN_YEARS = 2
TEST_MONTHS = 6
XGB_THRESHOLD = 0.35
CASH_BUFFER = 0.02
INITIAL = 100_000.0
TRAILING_STOP_PCT = 0.10

DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_history(ticker: str, start: str = "1999-01-01") -> pd.DataFrame:
    """Download (or load cached) daily OHLCV for ticker. Adjusted close."""
    cache = DATA_DIR / f"{ticker.lower()}_daily.csv"
    if not cache.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        print(f"  Downloading {ticker}…")
        df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
        df.to_csv(cache)
    df = pd.read_csv(
        cache,
        skiprows=3,
        index_col=0,
        parse_dates=True,
        names=["Date", "Close", "High", "Low", "Open", "Volume"],
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    return df.sort_index()


def load_external_features() -> pd.DataFrame:
    path = DATA_DIR / "features_daily.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"No external features at {path}. Run: uv run python backtest/download_features.py"
        )
    return pd.read_csv(path, index_col="date", parse_dates=True)


# ---------------------------------------------------------------------------
# Feature prep + label
# ---------------------------------------------------------------------------

def prepare_features(price_df: pd.DataFrame, ext_df: pd.DataFrame) -> pd.DataFrame:
    pipeline = FeaturePipeline()
    features = pipeline.compute_features_extended(price_df, ext_df)
    features["regime_label"] = compute_regime_labels(features)
    forward_return = features["close"].shift(-20) / features["close"] - 1
    features = features[forward_return.notna()].copy()
    forward_return = forward_return[forward_return.notna()]
    features["label"] = CLASS_FLAT
    features.loc[forward_return > LABEL_THRESHOLD, "label"] = CLASS_UP
    features.loc[forward_return < -LABEL_THRESHOLD, "label"] = CLASS_DOWN
    features["label"] = features["label"].astype(int)
    return features.dropna(subset=XGB_FEATURES)


def _sma_crossover_signal(close_window: pd.Series) -> Signal:
    if len(close_window) < 201:
        return Signal.HOLD
    sma50 = close_window.rolling(50).mean()
    sma200 = close_window.rolling(200).mean()
    if pd.isna(sma50.iloc[-1]) or pd.isna(sma200.iloc[-1]) or pd.isna(sma50.iloc[-2]) or pd.isna(sma200.iloc[-2]):
        return Signal.HOLD
    if sma50.iloc[-1] > sma200.iloc[-1] and sma50.iloc[-2] <= sma200.iloc[-2]:
        return Signal.BUY
    if sma50.iloc[-1] < sma200.iloc[-1] and sma50.iloc[-2] >= sma200.iloc[-2]:
        return Signal.SELL
    return Signal.HOLD


def _xgb_signal(proba: np.ndarray, class_order, threshold: float = XGB_THRESHOLD) -> Signal:
    idx_up = list(class_order).index(CLASS_UP)
    idx_down = list(class_order).index(CLASS_DOWN)
    pred = int(np.argmax(proba))
    if pred == idx_up and proba[idx_up] > threshold:
        return Signal.BUY
    if pred == idx_down and proba[idx_down] > threshold:
        return Signal.SELL
    return Signal.HOLD


# ---------------------------------------------------------------------------
# Walk-forward generation of records
# ---------------------------------------------------------------------------

def walk_forward_records(features: pd.DataFrame, close: pd.Series) -> list[dict]:
    """Generate one record per test day across walk-forward folds."""
    records: list[dict] = []
    train_days = TRAIN_YEARS * 252
    test_days = TEST_MONTHS * 21

    if len(features) < train_days + test_days:
        raise ValueError(
            f"Not enough history ({len(features)} rows) for walk-forward "
            f"(needs {train_days + test_days})"
        )

    regime_series = compute_regime_series(features)

    start_idx = train_days
    while start_idx + test_days <= len(features):
        train_slice = features.iloc[start_idx - train_days : start_idx]
        test_slice = features.iloc[start_idx : start_idx + test_days]

        X_train = train_slice[XGB_FEATURES].values
        y_train = train_slice["label"].values
        # Skip fold if train labels are degenerate
        if len(np.unique(y_train)) < 3:
            start_idx += test_days
            continue

        model = XGBClassifier(**XGB_PARAMS)
        model.fit(X_train, y_train)
        class_order = list(model.classes_)

        for ts in test_slice.index:
            row = features.loc[ts]
            proba = model.predict_proba(row[XGB_FEATURES].values.reshape(1, -1))[0]

            xgb_sig = _xgb_signal(proba, class_order)

            # SMA crossover from raw close window through this date
            close_window = close.loc[:ts]
            sma_sig = _sma_crossover_signal(close_window)

            today_regime = regime_series.loc[ts] if ts in regime_series.index else Regime.CHOPPY
            if not isinstance(today_regime, Regime):
                today_regime = Regime.CHOPPY

            bear_weakening = False
            if today_regime == Regime.BEAR:
                lr5 = features.loc[ts, "log_return_5d"]
                bear_weakening = (not pd.isna(lr5)) and lr5 > 0

            composite_sig, source = composite_signal_hybrid(
                today_regime, sma_sig, xgb_sig, bear_weakening=bear_weakening,
            )

            records.append({
                "date": ts,
                "close": float(row["close"]),
                "regime": today_regime.value,
                "composite_signal": composite_sig,
                "source": source,
            })

        start_idx += test_days

    return records


# ---------------------------------------------------------------------------
# Portfolio simulation
# ---------------------------------------------------------------------------

def simulate(
    records: list[dict],
    close: pd.Series,
    vix: pd.Series | None,
) -> pd.DataFrame:
    """Run the trailing-stop portfolio simulation. Returns a DataFrame
    indexed by date with 'value' and 'exposure' columns."""
    daily_returns = close.pct_change()

    value = INITIAL
    invested = 0.0
    peak = INITIAL
    stop_triggered = False
    cooldown_left = 0

    rows = []
    for rec in records:
        dt = rec["date"]
        daily_ret = daily_returns.get(dt, 0.0)
        if pd.isna(daily_ret):
            daily_ret = 0.0

        sig = rec["composite_signal"]
        if sig == Signal.BUY:
            target = 1.0 - CASH_BUFFER
        elif sig == Signal.SELL:
            target = 0.0
        else:
            target = invested

        # Trailing stop logic
        if value > peak:
            peak = value
        if not stop_triggered and value > 0 and (value / peak - 1.0) <= -TRAILING_STOP_PCT:
            stop_triggered = True
            target = 0.0
            cooldown_left = 5
        if stop_triggered:
            if cooldown_left > 0:
                target = 0.0
                cooldown_left -= 1
            else:
                stop_triggered = False
                peak = value  # reset

        # Apply slippage on rebalance
        if abs(target - invested) > 1e-6:
            v = vix.get(dt, 20.0) if vix is not None else 20.0
            slip = estimate_slippage(v) * abs(target - invested)
            value *= (1 - slip)

        invested = target
        value *= (1 + invested * daily_ret)

        rows.append({"date": dt, "value": value, "exposure": invested})

    return pd.DataFrame(rows).set_index("date")


def metrics(values: np.ndarray) -> dict:
    if len(values) < 2:
        return {"sharpe": 0.0, "max_dd_pct": 0.0, "total_return_pct": 0.0}
    daily = np.diff(values) / values[:-1]
    sharpe = float(np.sqrt(252) * daily.mean() / (daily.std() + 1e-12))
    peak = np.maximum.accumulate(values)
    dd = (values / peak - 1.0)
    return {
        "sharpe": round(sharpe, 3),
        "max_dd_pct": round(float(dd.min()) * 100, 2),
        "total_return_pct": round(float(values[-1] / values[0] - 1.0) * 100, 2),
    }


def buy_and_hold_metrics(close: pd.Series, dates: pd.Index) -> dict:
    """Buy-and-hold over the same date range, no slippage."""
    series = close.reindex(dates).ffill().dropna()
    if len(series) < 2:
        return {"sharpe": 0.0, "max_dd_pct": 0.0, "total_return_pct": 0.0}
    norm = (series / series.iloc[0]).values * INITIAL
    return metrics(norm)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_for_ticker(ticker: str, ext_df: pd.DataFrame) -> dict:
    print(f"\n=== {ticker} ===")
    price_df = load_history(ticker)
    print(f"  History: {price_df.index[0].date()} → {price_df.index[-1].date()} ({len(price_df)} rows)")

    features = prepare_features(price_df, ext_df)
    print(f"  Features: {len(features)} rows after labels/regime")

    records = walk_forward_records(features, price_df["close"])
    print(f"  Walk-forward records: {len(records)}")

    vix = ext_df["vix_close"] if "vix_close" in ext_df.columns else None
    sim = simulate(records, price_df["close"], vix)

    sys_metrics = metrics(sim["value"].values)
    bh_metrics = buy_and_hold_metrics(price_df["close"], sim.index)

    print(f"  Strategy:    Sharpe {sys_metrics['sharpe']:>5.2f}  "
          f"return {sys_metrics['total_return_pct']:>+7.2f}%  "
          f"maxDD {sys_metrics['max_dd_pct']:>+7.2f}%")
    print(f"  Buy & hold:  Sharpe {bh_metrics['sharpe']:>5.2f}  "
          f"return {bh_metrics['total_return_pct']:>+7.2f}%  "
          f"maxDD {bh_metrics['max_dd_pct']:>+7.2f}%")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sim.to_csv(RESULTS_DIR / f"sector_{ticker.lower()}_curve.csv")

    return {
        "ticker": ticker,
        "system": sys_metrics,
        "buy_and_hold": bh_metrics,
        "sessions": len(sim),
        "start": str(sim.index[0].date()),
        "end": str(sim.index[-1].date()),
    }


def print_comparison(results: list[dict]) -> None:
    print()
    print("=" * 80)
    print(f"{'Ticker':<8}{'System Sharpe':>14}{'B&H Sharpe':>13}{'Sys Return':>14}{'B&H Return':>14}{'Sys MaxDD':>13}{'B&H MaxDD':>13}")
    print("-" * 89)
    for r in results:
        s, b = r["system"], r["buy_and_hold"]
        print(
            f"{r['ticker']:<8}"
            f"{s['sharpe']:>14.2f}"
            f"{b['sharpe']:>13.2f}"
            f"{s['total_return_pct']:>13.2f}%"
            f"{b['total_return_pct']:>13.2f}%"
            f"{s['max_dd_pct']:>12.2f}%"
            f"{b['max_dd_pct']:>12.2f}%"
        )
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tickers", nargs="+")
    args = ap.parse_args()

    ext_df = load_external_features()

    results = []
    for ticker in args.tickers:
        try:
            results.append(run_for_ticker(ticker, ext_df))
        except Exception as exc:
            print(f"  FAILED: {exc}")

    print_comparison(results)

    summary_path = RESULTS_DIR / "sector_summary.json"
    summary_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
