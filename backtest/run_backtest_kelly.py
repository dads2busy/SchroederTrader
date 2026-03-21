"""Backtest comparing Kelly-sized vs binary (100% in/out) positions.

Derives the win/loss ratio from walk-forward XGB signals and their actual
20-day forward returns, then simulates both strategies side-by-side.
"""
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from backtest.train_final_composite import (
    prepare_features,
    XGB_FEATURES,
    XGB_PARAMS,
    TRAIN_YEARS,
    TEST_MONTHS,
)
from schroeder_trader.risk.kelly import kelly_fraction
from schroeder_trader.strategy.composite import composite_signal_hybrid
from schroeder_trader.strategy.feature_engineer import CLASS_DOWN, CLASS_UP
from schroeder_trader.strategy.regime_detector import Regime
from schroeder_trader.strategy.sma_crossover import Signal


CASH_BUFFER = 0.02
INITIAL_CAPITAL = 100_000.0


def _regime_from_label(label: float) -> Regime:
    """Convert numeric regime_label back to Regime enum."""
    mapping = {0: Regime.BEAR, 1: Regime.CHOPPY, 2: Regime.BULL}
    return mapping.get(int(label), Regime.CHOPPY)


def _xgb_signals_from_proba(proba, class_order, threshold):
    """Produce a Signal from XGB probabilities at a given threshold."""
    idx_up = list(class_order).index(CLASS_UP)
    idx_down = list(class_order).index(CLASS_DOWN)
    pred_class = int(np.argmax(proba))
    if pred_class == idx_up and proba[idx_up] > threshold:
        return Signal.BUY
    elif pred_class == idx_down and proba[idx_down] > threshold:
        return Signal.SELL
    else:
        return Signal.HOLD


def _sma_signal_for_day(close_series):
    """Compute SMA crossover signal from a close series ending at current day.

    Needs at least 201 rows.
    """
    if len(close_series) < 201:
        return Signal.HOLD
    sma50 = close_series.rolling(50).mean()
    sma200 = close_series.rolling(200).mean()
    cur_short, cur_long = sma50.iloc[-1], sma200.iloc[-1]
    prev_short, prev_long = sma50.iloc[-2], sma200.iloc[-2]
    if cur_short > cur_long and prev_short <= prev_long:
        return Signal.BUY
    elif cur_short < cur_long and prev_short >= prev_long:
        return Signal.SELL
    else:
        return Signal.HOLD


def run():
    print("Preparing features...")
    features_df = prepare_features()
    print(f"Feature matrix: {len(features_df)} rows, "
          f"{features_df.index.min().date()} to {features_df.index.max().date()}")

    # We also need the raw close prices for SMA and daily returns.
    # close is already in features_df from prepare_features.
    close = features_df["close"]

    # Compute 20-day forward return for win/loss analysis
    fwd_return_20d = close.shift(-20) / close - 1

    # ── Walk-forward: collect signals and XGB metadata ──────────────
    print("\nRunning walk-forward windows...")
    start_date = features_df.index.min()
    end_date = features_df.index.max()
    train_start = start_date

    # Collect per-day records for all test windows
    records = []  # list of dicts per test day
    xgb_buy_returns = []  # 20-day returns for XGB-sourced BUY signals

    window_num = 0
    while True:
        train_end = train_start + pd.DateOffset(years=TRAIN_YEARS)
        test_end = train_end + pd.DateOffset(months=TEST_MONTHS)
        if train_end >= end_date:
            break

        train_data = features_df[
            (features_df.index >= train_start) & (features_df.index < train_end)
        ].dropna(subset=XGB_FEATURES)

        test_data = features_df[
            (features_df.index >= train_end) & (features_df.index < test_end)
        ].dropna(subset=XGB_FEATURES)

        if len(train_data) < 100 or len(train_data["label"].unique()) < 3:
            train_start += pd.DateOffset(months=TEST_MONTHS)
            continue

        if len(test_data) == 0:
            train_start += pd.DateOffset(months=TEST_MONTHS)
            continue

        window_num += 1

        # Train XGB with early stopping
        X_train = train_data[XGB_FEATURES]
        y_train = train_data["label"]
        val_split = int(len(X_train) * 0.8)

        model = XGBClassifier(**XGB_PARAMS, n_estimators=500, early_stopping_rounds=20)
        model.fit(
            X_train[:val_split], y_train[:val_split],
            eval_set=[(X_train[val_split:], y_train[val_split:])],
            verbose=False,
        )
        print(f"  Window {window_num}: train {train_data.index.min().date()}-"
              f"{train_data.index.max().date()}, "
              f"test {test_data.index.min().date()}-{test_data.index.max().date()}, "
              f"best_iter={model.best_iteration}")

        class_order = model.classes_

        # Track consecutive bear days across test window using full history
        for i, (dt, row) in enumerate(test_data.iterrows()):
            X_row = row[XGB_FEATURES].values.reshape(1, -1)
            proba = model.predict_proba(pd.DataFrame(X_row, columns=XGB_FEATURES))[0]

            xgb_low = _xgb_signals_from_proba(proba, class_order, 0.35)
            xgb_high = _xgb_signals_from_proba(proba, class_order, 0.50)

            regime = _regime_from_label(row["regime_label"])

            # Count consecutive bear days up to this point using all data
            hist_end = features_df.index.get_loc(dt)
            lookback = min(hist_end + 1, 252)
            regime_slice = features_df["regime_label"].iloc[hist_end - lookback + 1:hist_end + 1]
            bear_count = 0
            for val in reversed(regime_slice.values):
                if val == 0:  # BEAR
                    bear_count += 1
                else:
                    break

            # SMA signal from historical close up to this day
            close_up_to = close.iloc[:hist_end + 1]
            sma_signal = _sma_signal_for_day(close_up_to)

            signal, source = composite_signal_hybrid(
                regime, sma_signal, xgb_low, xgb_high, bear_count,
            )

            # p_up and p_down for Kelly
            idx_up = list(class_order).index(CLASS_UP)
            idx_down = list(class_order).index(CLASS_DOWN)
            p_up = float(proba[idx_up])
            p_down = float(proba[idx_down])

            fwd_ret = fwd_return_20d.get(dt, np.nan)

            records.append({
                "date": dt,
                "signal": signal,
                "source": source,
                "p_up": p_up,
                "p_down": p_down,
                "fwd_return_20d": fwd_ret,
                "daily_return": None,  # filled later
            })

            # Track XGB-sourced BUY signals for win/loss ratio
            if source == "XGB" and signal == Signal.BUY and not np.isnan(fwd_ret):
                xgb_buy_returns.append(fwd_ret)

        train_start += pd.DateOffset(months=TEST_MONTHS)

    if not records:
        print("ERROR: No test records generated.")
        return

    # ── Compute win/loss ratio ─────────────────────────────────────
    xgb_buy_returns = np.array(xgb_buy_returns)
    wins = xgb_buy_returns[xgb_buy_returns > 0]
    losses = xgb_buy_returns[xgb_buy_returns < 0]

    print(f"\nXGB-sourced BUY signals: {len(xgb_buy_returns)}")
    print(f"  Wins: {len(wins)}, Losses: {len(losses)}")

    if len(wins) == 0 or len(losses) == 0:
        print("ERROR: Cannot compute win/loss ratio (no wins or no losses).")
        return

    mean_win = np.mean(np.abs(wins))
    mean_loss = np.mean(np.abs(losses))
    win_loss_ratio = mean_win / mean_loss
    print(f"  Mean win: {mean_win:.4f}, Mean loss: {mean_loss:.4f}")
    print(f"  Win/Loss ratio (b): {win_loss_ratio:.4f}")

    # ── Simulate portfolios ────────────────────────────────────────
    # Build a DataFrame of test days in order, with daily returns
    signals_df = pd.DataFrame(records)
    signals_df = signals_df.sort_values("date").reset_index(drop=True)

    # Compute daily SPY returns for the test dates
    daily_returns = close.pct_change()

    # Binary and Kelly portfolio simulation
    binary_value = INITIAL_CAPITAL
    kelly_value = INITIAL_CAPITAL
    binary_invested_frac = 0.0  # fraction of portfolio invested
    kelly_invested_frac = 0.0

    binary_values = []
    kelly_values = []

    for _, row in signals_df.iterrows():
        dt = row["date"]
        sig = row["signal"]
        source = row["source"]
        p_up = row["p_up"]
        p_down = row["p_down"]

        daily_ret = daily_returns.get(dt, 0.0)
        if np.isnan(daily_ret):
            daily_ret = 0.0

        # ── Update positions based on signal ──
        # Binary strategy
        if sig == Signal.BUY:
            binary_invested_frac = 1.0 - CASH_BUFFER
        elif sig == Signal.SELL:
            binary_invested_frac = 0.0
        # HOLD: maintain current position

        # Kelly strategy
        if sig == Signal.BUY:
            if source == "XGB":
                kf = kelly_fraction(p_up, p_down, win_loss_ratio)
                kelly_invested_frac = kf * (1.0 - CASH_BUFFER)
            else:
                # SMA-sourced BUY: go full position
                kelly_invested_frac = 1.0 - CASH_BUFFER
        elif sig == Signal.SELL:
            kelly_invested_frac = 0.0
        # HOLD: maintain current position

        # ── Mark to market ──
        binary_value *= (1.0 + binary_invested_frac * daily_ret)
        kelly_value *= (1.0 + kelly_invested_frac * daily_ret)

        binary_values.append(binary_value)
        kelly_values.append(kelly_value)

    binary_values = np.array(binary_values)
    kelly_values = np.array(kelly_values)

    # ── Performance metrics ────────────────────────────────────────
    def compute_metrics(values):
        daily_rets = np.diff(values) / values[:-1]
        sharpe = np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(252) if np.std(daily_rets) > 0 else 0.0
        peak = np.maximum.accumulate(values)
        drawdown = (values - peak) / peak
        max_dd = float(np.min(drawdown))
        total_return = (values[-1] / values[0]) - 1.0
        return sharpe, max_dd, total_return

    b_sharpe, b_dd, b_ret = compute_metrics(binary_values)
    k_sharpe, k_dd, k_ret = compute_metrics(kelly_values)

    # ── Report ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    print(f"\nWin/Loss ratio (b): {win_loss_ratio:.4f}")
    print(f"  → config value (rounded): {round(win_loss_ratio, 2)}")
    print(f"\nXGB BUY win rate: {len(wins) / len(xgb_buy_returns) * 100:.1f}%")

    print(f"\n{'Metric':<25} {'Binary':>12} {'Kelly':>12}")
    print("-" * 50)
    print(f"{'Annualized Sharpe':<25} {b_sharpe:>12.3f} {k_sharpe:>12.3f}")
    print(f"{'Max Drawdown':<25} {b_dd:>11.1%} {k_dd:>11.1%}")
    print(f"{'Total Return':<25} {b_ret:>11.1%} {k_ret:>11.1%}")
    print(f"{'Final Value':<25} ${binary_values[-1]:>11,.0f} ${kelly_values[-1]:>11,.0f}")

    print(f"\n→ Set KELLY_WIN_LOSS_RATIO = {round(win_loss_ratio, 2)} in config.py")

    return round(win_loss_ratio, 2)


if __name__ == "__main__":
    run()
