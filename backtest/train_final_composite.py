"""Train the final XGBoost model for composite shadow deployment.

Uses 20-day forward return labels and the 6-feature set validated in Phase 2.1/3.
Determines optimal n_estimators from walk-forward, then retrains on all data.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.train_model import load_data
from schroeder_trader.config import COMPOSITE_MODEL_PATH
from schroeder_trader.strategy.feature_engineer import (
    FeaturePipeline,
    CLASS_DOWN,
    CLASS_FLAT,
    CLASS_UP,
)
from schroeder_trader.strategy.regime_detector import Regime, detect_regime
from schroeder_trader.strategy.xgboost_classifier import save_model
from xgboost import XGBClassifier

DATA_DIR = Path(__file__).parent / "data"

XGB_FEATURES = [
    "log_return_5d", "log_return_20d", "volatility_20d",
    "credit_spread", "dollar_momentum", "regime_label",
]

LABEL_THRESHOLD = 0.01  # 1% for 20-day horizon

TRAIN_YEARS = 2
TEST_MONTHS = 6

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


def prepare_features() -> pd.DataFrame:
    """Load SPY + external data, compute features with 20-day labels and regime."""
    spy_df = load_data()

    ext_path = DATA_DIR / "features_daily.csv"
    if not ext_path.exists():
        raise FileNotFoundError(f"No external features at {ext_path}. Run download_features.py first.")
    ext_df = pd.read_csv(ext_path, index_col="date", parse_dates=True)

    pipeline = FeaturePipeline()
    features_df = pipeline.compute_features_extended(spy_df, ext_df)

    # Compute regime labels (backward-looking)
    log_ret_20d = np.log(features_df["close"] / features_df["close"].shift(20))
    vol_20d = features_df["close"].pct_change().rolling(20).std()
    vol_median = vol_20d.rolling(252).median()

    regime_vals = []
    for i in range(len(features_df)):
        lr = log_ret_20d.iloc[i]
        vol = vol_20d.iloc[i]
        vm = vol_median.iloc[i]
        if pd.isna(lr) or pd.isna(vol) or pd.isna(vm):
            regime_vals.append(np.nan)
        else:
            regime_vals.append(detect_regime(lr, vol, vm))

    regime_map = {Regime.BEAR: 0, Regime.CHOPPY: 1, Regime.BULL: 2}
    features_df["regime_label"] = pd.Series(regime_vals, index=features_df.index).map(regime_map)

    # 20-day forward return labels
    forward_return = features_df["close"].shift(-20) / features_df["close"] - 1
    features_df = features_df[forward_return.notna()].copy()
    forward_return = forward_return[forward_return.notna()]

    features_df["label"] = CLASS_FLAT
    features_df.loc[forward_return > LABEL_THRESHOLD, "label"] = CLASS_UP
    features_df.loc[forward_return < -LABEL_THRESHOLD, "label"] = CLASS_DOWN
    features_df["label"] = features_df["label"].astype(int)

    features_df = features_df.dropna(subset=XGB_FEATURES)

    return features_df


def find_median_n_estimators(features_df: pd.DataFrame) -> int:
    """Run walk-forward to find median optimal n_estimators."""
    best_iterations = []

    start_date = features_df.index.min()
    end_date = features_df.index.max()
    train_start = start_date

    while True:
        train_end = train_start + pd.DateOffset(years=TRAIN_YEARS)
        test_end = train_end + pd.DateOffset(months=TEST_MONTHS)
        if train_end >= end_date:
            break

        train_data = features_df[(features_df.index >= train_start) & (features_df.index < train_end)]
        train_valid = train_data.dropna(subset=XGB_FEATURES)

        if len(train_valid) < 100 or len(train_valid["label"].unique()) < 3:
            train_start += pd.DateOffset(months=TEST_MONTHS)
            continue

        X = train_valid[XGB_FEATURES]
        y = train_valid["label"]
        val_split = int(len(X) * 0.8)

        model = XGBClassifier(**XGB_PARAMS, n_estimators=500, early_stopping_rounds=20)
        model.fit(X[:val_split], y[:val_split], eval_set=[(X[val_split:], y[val_split:])], verbose=False)
        best_iterations.append(model.best_iteration)
        print(f"  Window {len(best_iterations)}: best_iteration={model.best_iteration}")

        train_start += pd.DateOffset(months=TEST_MONTHS)

    median_n = int(np.median(best_iterations))
    print(f"\nWalk-forward best iterations: {best_iterations}")
    print(f"Median n_estimators: {median_n}")
    return median_n


def train_and_save():
    """Train final model on all data and save."""
    print("Preparing features...")
    features_df = prepare_features()
    print(f"Feature matrix: {len(features_df)} rows, "
          f"{features_df.index.min().date()} to {features_df.index.max().date()}")

    print("\nFinding optimal n_estimators via walk-forward...")
    n_estimators = find_median_n_estimators(features_df)

    print(f"\nTraining final model on all {len(features_df)} rows with n_estimators={n_estimators}...")
    X = features_df[XGB_FEATURES]
    y = features_df["label"]

    model = XGBClassifier(**XGB_PARAMS, n_estimators=n_estimators)
    model.fit(X, y, verbose=False)

    COMPOSITE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_model(model, COMPOSITE_MODEL_PATH)
    print(f"\nModel saved to {COMPOSITE_MODEL_PATH}")
    print(f"Model classes: {model.classes_}")
    print(f"Training rows: {len(X)}, n_estimators: {n_estimators}")


if __name__ == "__main__":
    train_and_save()
