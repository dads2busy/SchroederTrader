"""Run vectorbt backtest using the SMA crossover strategy on cached SPY data."""
import json
from pathlib import Path

import pandas as pd
import vectorbt as vbt

from schroeder_trader.config import SMA_SHORT_WINDOW, SMA_LONG_WINDOW, SLIPPAGE_ESTIMATE

DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "results"


def load_data() -> pd.DataFrame:
    csv_path = DATA_DIR / "spy_daily.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No cached data at {csv_path}. Run download_data.py first.")
    # yfinance CSVs have 3 header rows (Price, Ticker, Date) before actual data
    df = pd.read_csv(
        csv_path,
        skiprows=3,
        index_col=0,
        parse_dates=True,
        names=["Date", "Close", "High", "Low", "Open", "Volume"],
    )
    # Handle multi-level columns from yfinance (older yfinance versions)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def run_backtest() -> dict:
    df = load_data()
    close = df["Close"]

    # Compute SMAs
    sma_short = vbt.MA.run(close, window=SMA_SHORT_WINDOW)
    sma_long = vbt.MA.run(close, window=SMA_LONG_WINDOW)

    # Generate crossover entries and exits
    entries = sma_short.ma_crossed_above(sma_long)
    exits = sma_short.ma_crossed_below(sma_long)

    # Run portfolio simulation with slippage
    portfolio = vbt.Portfolio.from_signals(
        close,
        entries=entries,
        exits=exits,
        init_cash=10000,
        fees=SLIPPAGE_ESTIMATE,
        freq="1D",
    )

    # Compute metrics
    stats = portfolio.stats()
    total_return = float(stats["Total Return [%]"])
    sharpe = float(stats.get("Sharpe Ratio", 0.0))
    max_dd = float(stats["Max Drawdown [%]"])
    total_trades = int(stats["Total Trades"])
    win_rate = float(stats.get("Win Rate [%]", 0.0))

    # Buy-and-hold benchmark
    bh_portfolio = vbt.Portfolio.from_holding(close, init_cash=10000, freq="1D")
    bh_stats = bh_portfolio.stats()
    bh_return = float(bh_stats["Total Return [%]"])
    bh_sharpe = float(bh_stats.get("Sharpe Ratio", 0.0))
    bh_max_dd = float(bh_stats["Max Drawdown [%]"])

    # Train/test split at 2020
    split_date = "2020-01-01"
    test_close = close[close.index >= split_date]
    test_entries = entries[entries.index >= split_date]
    test_exits = exits[exits.index >= split_date]

    test_portfolio = vbt.Portfolio.from_signals(
        test_close,
        entries=test_entries,
        exits=test_exits,
        init_cash=10000,
        fees=SLIPPAGE_ESTIMATE,
        freq="1D",
    )
    test_stats = test_portfolio.stats()
    test_sharpe = float(test_stats.get("Sharpe Ratio", 0.0))

    results = {
        "full_period": {
            "total_return_pct": round(total_return, 2),
            "sharpe_ratio": round(sharpe, 4),
            "max_drawdown_pct": round(max_dd, 2),
            "total_trades": total_trades,
            "win_rate_pct": round(win_rate, 2),
        },
        "benchmark_buy_and_hold": {
            "total_return_pct": round(bh_return, 2),
            "sharpe_ratio": round(bh_sharpe, 4),
            "max_drawdown_pct": round(bh_max_dd, 2),
        },
        "out_of_sample_post_2020": {
            "sharpe_ratio": round(test_sharpe, 4),
        },
    }

    # Print results
    print("\n" + "=" * 50)
    print("SMA CROSSOVER BACKTEST RESULTS")
    print("=" * 50)
    print(f"\nFull Period ({close.index[0].date()} to {close.index[-1].date()}):")
    print(f"  Total Return: {total_return:.2f}%")
    print(f"  Sharpe Ratio: {sharpe:.4f}")
    print(f"  Max Drawdown: {max_dd:.2f}%")
    print(f"  Total Trades: {total_trades}")
    print(f"  Win Rate: {win_rate:.2f}%")
    print(f"\nBuy & Hold Benchmark:")
    print(f"  Total Return: {bh_return:.2f}%")
    print(f"  Sharpe Ratio: {bh_sharpe:.4f}")
    print(f"  Max Drawdown: {bh_max_dd:.2f}%")
    print(f"\nOut-of-Sample (post-2020):")
    print(f"  Sharpe Ratio: {test_sharpe:.4f}")
    print("=" * 50)

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "backtest_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_DIR / 'backtest_results.json'}")

    # Validate that vectorbt signals match generate_signal() logic
    from schroeder_trader.strategy.sma_crossover import generate_signal as gs, Signal
    sample_indices = [250, 500, 1000, len(df) - 1]
    for idx in sample_indices:
        if idx >= len(df):
            continue
        sub_df = df.iloc[:idx + 1].copy()
        sub_df.columns = [c.lower() for c in sub_df.columns]
        if len(sub_df) >= 201:
            sig, _, _ = gs(sub_df)
            vbt_entry = bool(entries.iloc[idx]) if idx < len(entries) else False
            vbt_exit = bool(exits.iloc[idx]) if idx < len(exits) else False
            if sig == Signal.BUY:
                assert vbt_entry, f"Signal mismatch at index {idx}: generate_signal=BUY but vectorbt entry=False"
            elif sig == Signal.SELL:
                assert vbt_exit, f"Signal mismatch at index {idx}: generate_signal=SELL but vectorbt exit=False"
    print("Signal validation: generate_signal() matches vectorbt crossover logic")

    # Save equity curve plot
    try:
        fig = portfolio.plot()
        fig.write_image(str(OUTPUT_DIR / "equity_curve.png"))
        print(f"Equity curve saved to {OUTPUT_DIR / 'equity_curve.png'}")
    except Exception as e:
        print(f"Warning: Could not save equity curve image: {e}")

    return results


if __name__ == "__main__":
    run_backtest()
