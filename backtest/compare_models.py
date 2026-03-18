"""Compare SMA crossover signals vs ML shadow signals."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from schroeder_trader.config import DB_PATH, SLIPPAGE_ESTIMATE
from schroeder_trader.storage.trade_log import init_db, get_shadow_signals

OUTPUT_DIR = Path(__file__).parent / "results"


def compare() -> None:
    conn = init_db(DB_PATH)

    # Get shadow signals
    shadow = get_shadow_signals(conn)
    if not shadow:
        print("No shadow signals found. Run the pipeline with a trained model first.")
        conn.close()
        return

    # Get SMA signals for the same dates
    sma_rows = conn.execute("SELECT * FROM signals ORDER BY id").fetchall()
    sma_signals = [dict(row) for row in sma_rows]
    conn.close()

    if not sma_signals:
        print("No SMA signals found.")
        return

    print(f"Shadow signals: {len(shadow)}")
    print(f"SMA signals: {len(sma_signals)}")

    # Build daily signal series
    shadow_df = pd.DataFrame(shadow)
    shadow_df["date"] = pd.to_datetime(shadow_df["timestamp"]).dt.date

    sma_df = pd.DataFrame(sma_signals)
    sma_df["date"] = pd.to_datetime(sma_df["timestamp"]).dt.date

    # Merge on date
    merged = pd.merge(shadow_df, sma_df, on="date", suffixes=("_ml", "_sma"))

    if merged.empty:
        print("No overlapping dates between SMA and ML signals.")
        return

    print(f"Overlapping days: {len(merged)}")

    # Signal agreement
    agreement = (merged["ml_signal"] == merged["signal"]).mean()
    print(f"\nSignal agreement: {agreement:.1%}")

    # Count signal distributions
    print("\nML signal distribution:")
    print(merged["ml_signal"].value_counts().to_string())
    print("\nSMA signal distribution:")
    print(merged["signal"].value_counts().to_string())

    # Simulate returns for both
    close_prices = merged["close_price_ml"].values
    daily_returns = np.diff(close_prices) / close_prices[:-1]

    for name, signal_col in [("SMA", "signal"), ("ML", "ml_signal")]:
        signals = merged[signal_col].values[:-1]
        position = 0
        strat_returns = []
        prev_position = 0

        for sig, ret in zip(signals, daily_returns):
            if sig == "BUY":
                position = 1
            elif sig == "SELL":
                position = 0

            strat_ret = position * ret
            if position != prev_position:
                strat_ret -= SLIPPAGE_ESTIMATE
            prev_position = position
            strat_returns.append(strat_ret)

        strat_returns = np.array(strat_returns)
        if len(strat_returns) > 0 and strat_returns.std() > 0:
            sharpe = np.sqrt(252) * strat_returns.mean() / strat_returns.std()
        else:
            sharpe = 0.0

        total_ret = (1 + strat_returns).prod() - 1
        cumulative = (1 + strat_returns).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_dd = float(abs(drawdown.min())) * 100 if len(drawdown) > 0 else 0.0

        print(f"\n{name} Strategy:")
        print(f"  Sharpe Ratio: {sharpe:.4f}")
        print(f"  Total Return: {total_ret * 100:.2f}%")
        print(f"  Max Drawdown: {max_dd:.2f}%")

    # Save comparison results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    comparison = {"sma": {}, "ml": {}}
    for name, signal_col in [("sma", "signal"), ("ml", "ml_signal")]:
        signals = merged[signal_col].values[:-1]
        position = 0
        strat_returns = []
        prev_position = 0
        for sig, ret in zip(signals, daily_returns):
            if sig == "BUY":
                position = 1
            elif sig == "SELL":
                position = 0
            strat_ret = position * ret
            if position != prev_position:
                strat_ret -= SLIPPAGE_ESTIMATE
            prev_position = position
            strat_returns.append(strat_ret)
        sr = np.array(strat_returns)
        comparison[name] = {
            "sharpe_ratio": round(float(np.sqrt(252) * sr.mean() / (sr.std() + 1e-10)), 4),
            "total_return_pct": round(float(((1 + sr).prod() - 1) * 100), 2),
        }

    with open(OUTPUT_DIR / "ml_vs_sma_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nComparison saved to {OUTPUT_DIR / 'ml_vs_sma_comparison.json'}")

    # Promotion recommendation
    if comparison["ml"]["sharpe_ratio"] > comparison["sma"]["sharpe_ratio"]:
        print("\n*** Ready to promote — ML outperforms SMA ***")
    else:
        print("\n*** Not ready — SMA still outperforms ML ***")

    print("\n" + "=" * 50)
    print("Note: This comparison is only meaningful with several weeks of shadow data.")
    print("=" * 50)


if __name__ == "__main__":
    compare()
