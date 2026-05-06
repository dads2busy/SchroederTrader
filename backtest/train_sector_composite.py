"""Train a production XGBoost composite model for any sector ETF.

Mirrors backtest/train_final_composite.py (which is hardcoded to SPY) but
parameterized by ticker. Walks forward to find median optimal n_estimators,
then retrains on full history. Saves to models/xgboost_<ticker>_20d.json.

Usage:
    uv run python backtest/train_sector_composite.py XLK
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from backtest.sector_backtest import (
    XGB_FEATURES,
    XGB_PARAMS,
    LABEL_THRESHOLD,
    TRAIN_YEARS,
    TEST_MONTHS,
    load_history,
    load_external_features,
    prepare_features,
)
from schroeder_trader.strategy.xgboost_classifier import save_model


MODELS_DIR = Path(__file__).parent.parent / "models"


def find_median_n_estimators(features_df: pd.DataFrame) -> int:
    """Walk-forward to find median optimal n_estimators with early stopping."""
    best_iterations: list[int] = []

    start_date = features_df.index.min()
    end_date = features_df.index.max()
    train_start = start_date

    while True:
        train_end = train_start + pd.DateOffset(years=TRAIN_YEARS)
        if train_end >= end_date:
            break

        train_data = features_df[(features_df.index >= train_start) & (features_df.index < train_end)]
        train_valid = train_data.dropna(subset=XGB_FEATURES)
        if len(train_valid) < 100 or train_valid["label"].nunique() < 3:
            train_start += pd.DateOffset(months=TEST_MONTHS)
            continue

        X = train_valid[XGB_FEATURES]
        y = train_valid["label"]
        val_split = int(len(X) * 0.8)

        model = XGBClassifier(**XGB_PARAMS, n_estimators=500, early_stopping_rounds=20)
        model.fit(
            X[:val_split], y[:val_split],
            eval_set=[(X[val_split:], y[val_split:])],
            verbose=False,
        )
        best_iterations.append(model.best_iteration)
        train_start += pd.DateOffset(months=TEST_MONTHS)

    if not best_iterations:
        raise RuntimeError("No valid walk-forward windows produced an n_estimators value.")
    median_n = int(np.median(best_iterations))
    print(f"  Walk-forward best iterations: median={median_n}, n_windows={len(best_iterations)}")
    return median_n


def train_for_ticker(ticker: str) -> Path:
    print(f"\n=== Training composite for {ticker} ===")
    price_df = load_history(ticker)
    print(f"  Loaded {len(price_df)} bars  ({price_df.index[0].date()} → {price_df.index[-1].date()})")

    ext_df = load_external_features()
    features_df = prepare_features(price_df, ext_df)
    print(f"  Feature matrix: {len(features_df)} rows after labels/regime")

    print("  Walk-forward to find median n_estimators…")
    n_estimators = find_median_n_estimators(features_df)

    print(f"  Training final model on full history with n_estimators={n_estimators}…")
    X = features_df[XGB_FEATURES]
    y = features_df["label"]
    model = XGBClassifier(**XGB_PARAMS, n_estimators=n_estimators)
    model.fit(X, y, verbose=False)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out = MODELS_DIR / f"xgboost_{ticker.lower()}_20d.json"
    save_model(model, out)
    print(f"  Saved → {out}")
    print(f"  Classes: {list(model.classes_)}  |  Training rows: {len(X)}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tickers", nargs="+")
    args = ap.parse_args()
    for ticker in args.tickers:
        train_for_ticker(ticker.upper())


if __name__ == "__main__":
    main()
