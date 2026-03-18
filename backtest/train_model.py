"""Walk-forward training and evaluation of XGBoost classifier on SPY data."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

from schroeder_trader.config import SMA_SHORT_WINDOW, SMA_LONG_WINDOW, SLIPPAGE_ESTIMATE
from schroeder_trader.strategy.feature_engineer import (
    FeaturePipeline,
    CLASS_DOWN,
    CLASS_FLAT,
    CLASS_UP,
    CLASS_NAMES,
)
from schroeder_trader.strategy.xgboost_classifier import train_model, predict_signal, save_model

DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "results"
MODEL_DIR = Path(__file__).parent.parent / "models"

FEATURE_COLUMNS = [
    "log_return_5d", "log_return_20d", "volatility_20d",
    "sma_ratio", "volume_ratio", "rsi_14",
]

# Walk-forward parameters
TRAIN_YEARS = 2
TEST_MONTHS = 6


def load_data() -> pd.DataFrame:
    csv_path = DATA_DIR / "spy_daily.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No cached data at {csv_path}. Run download_data.py first.")
    # yfinance new-style CSV: 3 header rows (Price, Ticker, Date), then data.
    # Column order: Date, Close, High, Low, Open, Volume
    df = pd.read_csv(
        csv_path,
        skiprows=3,
        names=["Date", "close", "high", "low", "open", "volume"],
    )
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    # Cast to float (they come in as object strings in some pandas versions)
    for col in ["close", "high", "low", "open", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna()
    return df


def walk_forward_evaluate() -> dict:
    """Run walk-forward validation and return aggregate metrics."""
    df = load_data()
    pipeline = FeaturePipeline()
    features_df = pipeline.compute_features_with_labels(df)

    all_predictions = []
    all_actuals = []
    all_dates = []
    window_results = []

    # Walk-forward loop
    start_date = features_df.index.min()
    end_date = features_df.index.max()

    train_start = start_date
    while True:
        train_end = train_start + pd.DateOffset(years=TRAIN_YEARS)
        test_end = train_end + pd.DateOffset(months=TEST_MONTHS)

        if train_end >= end_date:
            break

        train_data = features_df[(features_df.index >= train_start) & (features_df.index < train_end)]
        test_data = features_df[(features_df.index >= train_end) & (features_df.index < test_end)]

        if len(train_data) < 100 or len(test_data) < 10:
            train_start = train_start + pd.DateOffset(months=TEST_MONTHS)
            continue

        X_train = train_data[FEATURE_COLUMNS]
        y_train = train_data["forward_return_5d_class"]

        X_test = test_data[FEATURE_COLUMNS]
        y_test = test_data["forward_return_5d_class"]

        # Split last 20% of train for early stopping validation
        val_split = int(len(X_train) * 0.8)
        X_tr, X_val = X_train[:val_split], X_train[val_split:]
        y_tr, y_val = y_train[:val_split], y_train[val_split:]

        model = train_model(X_tr, y_tr, X_val, y_val)

        # Predict on test set
        preds = model.predict(X_test)
        accuracy = accuracy_score(y_test, preds)

        all_predictions.extend(preds)
        all_actuals.extend(y_test.values)
        all_dates.extend(test_data.index)

        window_results.append({
            "train_start": str(train_start.date()),
            "train_end": str(train_end.date()),
            "test_end": str(test_end.date()),
            "accuracy": round(float(accuracy), 4),
            "n_train": len(X_train),
            "n_test": len(X_test),
        })

        print(f"Window {len(window_results)}: train {train_start.date()}-{train_end.date()}, "
              f"test {train_end.date()}-{test_end.date()}, accuracy={accuracy:.4f}")

        train_start = train_start + pd.DateOffset(months=TEST_MONTHS)

    # Aggregate metrics
    all_predictions = np.array(all_predictions)
    all_actuals = np.array(all_actuals)
    all_dates = pd.DatetimeIndex(all_dates)

    overall_accuracy = accuracy_score(all_actuals, all_predictions)

    # Simulate returns for Sharpe calculation
    close_prices = features_df.loc[all_dates, "close"]
    daily_returns = close_prices.pct_change().fillna(0)

    position = 0  # 0 = flat, 1 = long
    strategy_returns = []
    for i, (pred, ret) in enumerate(zip(all_predictions, daily_returns)):
        if pred == CLASS_UP:
            position = 1
        elif pred == CLASS_DOWN:
            position = 0

        strategy_return = position * ret
        if i > 0 and ((pred == CLASS_UP and all_predictions[i-1] != CLASS_UP) or
                       (pred == CLASS_DOWN and all_predictions[i-1] != CLASS_DOWN)):
            strategy_return -= SLIPPAGE_ESTIMATE

        strategy_returns.append(strategy_return)

    strategy_returns = np.array(strategy_returns)
    sharpe = np.sqrt(252) * strategy_returns.mean() / (strategy_returns.std() + 1e-10)
    total_return = (1 + strategy_returns).prod() - 1

    # Max drawdown
    cumulative = (1 + strategy_returns).cumprod()
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = float(abs(drawdown.min())) * 100

    # Trade count and win rate
    position_changes = np.diff(np.where(all_predictions == CLASS_UP, 1, 0))
    total_trades = int(np.sum(np.abs(position_changes)))

    wins = sum(1 for r in strategy_returns if r > 0)
    trading_days = sum(1 for r in strategy_returns if r != 0)
    win_rate = wins / trading_days if trading_days > 0 else 0.0

    # Post-2020 Sharpe
    post_2020_mask = all_dates >= "2020-01-01"
    if post_2020_mask.any():
        post_returns = strategy_returns[post_2020_mask]
        post_sharpe = float(np.sqrt(252) * post_returns.mean() / (post_returns.std() + 1e-10))
    else:
        post_sharpe = 0.0

    results = {
        "full_period": {
            "total_return_pct": round(float(total_return * 100), 2),
            "sharpe_ratio": round(float(sharpe), 4),
            "max_drawdown_pct": round(max_drawdown, 2),
            "total_trades": total_trades,
            "win_rate_pct": round(float(win_rate * 100), 2),
            "accuracy": round(float(overall_accuracy), 4),
        },
        "out_of_sample_post_2020": {
            "sharpe_ratio": round(post_sharpe, 4),
        },
        "windows": window_results,
    }

    print("\n" + "=" * 50)
    print("ML WALK-FORWARD RESULTS")
    print("=" * 50)
    print(f"\nOverall Accuracy: {overall_accuracy:.4f}")
    print(f"Total Return: {total_return * 100:.2f}%")
    print(f"Sharpe Ratio: {sharpe:.4f}")
    print(f"Max Drawdown: {max_drawdown:.2f}%")
    print(f"Total Trades: {total_trades}")
    print(f"Win Rate: {win_rate * 100:.2f}%")
    print(f"Post-2020 Sharpe: {post_sharpe:.4f}")
    print(f"Windows evaluated: {len(window_results)}")
    print("=" * 50)

    print("\n" + classification_report(
        all_actuals, all_predictions,
        target_names=["DOWN", "FLAT", "UP"],
    ))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "ml_walkforward_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {OUTPUT_DIR / 'ml_walkforward_results.json'}")

    # SHAP feature importance (optional)
    try:
        import shap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        recent = features_df.tail(1000)
        X_shap = recent[FEATURE_COLUMNS]
        y_shap = recent["forward_return_5d_class"]
        split = int(len(X_shap) * 0.8)
        shap_model = train_model(X_shap[:split], y_shap[:split], X_shap[split:], y_shap[split:])

        explainer = shap.TreeExplainer(shap_model)
        shap_values = explainer.shap_values(X_shap[:200])
        shap.summary_plot(shap_values, X_shap[:200], show=False)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "shap_importance.png", dpi=150)
        plt.close()
        print(f"SHAP importance plot saved to {OUTPUT_DIR / 'shap_importance.png'}")
    except ImportError:
        print("SHAP not installed — skipping feature importance plot (install with: uv pip install shap)")
    except Exception as e:
        print(f"SHAP analysis failed: {e}")

    return results


def train_final_model() -> None:
    """Train final model on all available data and save for shadow mode."""
    df = load_data()
    pipeline = FeaturePipeline()
    features_df = pipeline.compute_features_with_labels(df)

    X = features_df[FEATURE_COLUMNS]
    y = features_df["forward_return_5d_class"]

    split = int(len(X) * 0.8)
    model = train_model(X[:split], y[:split], X[split:], y[split:])

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "xgboost_spy.json"
    save_model(model, model_path)
    print(f"\nFinal model saved to {model_path}")


if __name__ == "__main__":
    results = walk_forward_evaluate()
    print("\nTraining final model for shadow deployment...")
    train_final_model()
