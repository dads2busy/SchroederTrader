"""Walk-forward backtest comparing three strategies:

1. Baseline (threshold + binary): threshold regime detection with hard switching
2. HMM hard: HMM regime detection with hard switching (most-probable state)
3. HMM blended: HMM regime detection with probability-weighted blending

All strategies use a 10% trailing stop.
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
from schroeder_trader.strategy.composite import (
    composite_signal_hybrid,
    composite_signal_blended,
)
from schroeder_trader.strategy.feature_engineer import CLASS_DOWN, CLASS_UP
from schroeder_trader.strategy.regime_detector import (
    Regime,
    HMMRegimeDetector,
    HMM_FEATURES,
)
from schroeder_trader.strategy.sma_crossover import Signal
from schroeder_trader.risk.transaction_cost import estimate_slippage

CASH_BUFFER = 0.02
INITIAL = 100_000.0
TRAILING_STOP_PCT = 0.10


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _regime_from_label(label: float) -> Regime:
    """Map numeric regime_label → Regime enum (0=BEAR, 1=CHOPPY, 2=BULL)."""
    mapping = {0: Regime.BEAR, 1: Regime.CHOPPY, 2: Regime.BULL}
    return mapping.get(int(label), Regime.CHOPPY)


def _xgb_signal(proba, class_order, threshold: float = 0.35) -> Signal:
    """Produce a Signal from XGB probability vector at the given threshold."""
    idx_up = list(class_order).index(CLASS_UP)
    idx_down = list(class_order).index(CLASS_DOWN)
    pred_class = int(np.argmax(proba))
    if pred_class == idx_up and proba[idx_up] > threshold:
        return Signal.BUY
    elif pred_class == idx_down and proba[idx_down] > threshold:
        return Signal.SELL
    else:
        return Signal.HOLD


def _sma_signal(close_series: pd.Series) -> Signal:
    """SMA 50/200 crossover signal; needs at least 201 rows."""
    if len(close_series) < 201:
        return Signal.HOLD
    sma50 = close_series.rolling(50).mean()
    sma200 = close_series.rolling(200).mean()
    cur_short = sma50.iloc[-1]
    cur_long = sma200.iloc[-1]
    prev_short = sma50.iloc[-2]
    prev_long = sma200.iloc[-2]
    if cur_short > cur_long and prev_short <= prev_long:
        return Signal.BUY
    elif cur_short < cur_long and prev_short >= prev_long:
        return Signal.SELL
    else:
        return Signal.HOLD


def compute_metrics(values: np.ndarray):
    """Return (sharpe, max_dd, total_return) from a portfolio value array."""
    daily_rets = np.diff(values) / values[:-1]
    if np.std(daily_rets) > 0:
        sharpe = float(np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(252))
    else:
        sharpe = 0.0
    peak = np.maximum.accumulate(values)
    drawdown = (values - peak) / peak
    max_dd = float(np.min(drawdown))
    total_return = float(values[-1] / values[0] - 1.0)
    return sharpe, max_dd, total_return


# ---------------------------------------------------------------------------
# Trailing stop simulation helper
# ---------------------------------------------------------------------------

def simulate_strategy(
    records: list[dict],
    close: pd.Series,
    strategy: str,
    vix: pd.Series | None = None,
) -> np.ndarray:
    """Simulate one strategy and return array of daily portfolio values.

    Args:
        records: List of per-day dicts with signal info.
        close: Full SPY close price series.
        strategy: One of "threshold", "hmm_hard", "hmm_blended".
        vix: VIX close series for slippage estimation.

    Returns:
        np.ndarray of portfolio values (one per record row).
    """
    daily_returns = close.pct_change()

    value = INITIAL
    invested = 0.0  # fraction of portfolio currently invested
    peak = INITIAL  # for trailing stop
    stop_triggered = False

    values = []

    for rec in records:
        dt = rec["date"]
        daily_ret = daily_returns.get(dt, 0.0)
        if pd.isna(daily_ret):
            daily_ret = 0.0

        # Determine signal and target exposure for this strategy
        if strategy == "threshold":
            sig = rec["threshold_signal"]
            if sig == Signal.BUY:
                target = 1.0 - CASH_BUFFER
            elif sig == Signal.SELL:
                target = 0.0
            else:
                target = invested  # HOLD: maintain

        elif strategy == "hmm_hard":
            sig = rec["hmm_hard_signal"]
            if sig is None:
                # Fallback to threshold
                sig = rec["threshold_signal"]
            if sig == Signal.BUY:
                target = 1.0 - CASH_BUFFER
            elif sig == Signal.SELL:
                target = 0.0
            else:
                target = invested

        elif strategy == "hmm_blended":
            regime_probs = rec["hmm_regime_probs"]
            if regime_probs is None:
                # Fallback to threshold
                sig = rec["threshold_signal"]
                if sig == Signal.BUY:
                    target = 1.0 - CASH_BUFFER
                elif sig == Signal.SELL:
                    target = 0.0
                else:
                    target = invested
            else:
                target = composite_signal_blended(
                    regime_probs=regime_probs,
                    sma_signal=rec["sma_signal"],
                    xgb_signal=rec["xgb_signal"],
                    bear_weakening=rec["bear_weakening"],
                    current_exposure=invested,
                )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Apply trailing stop: if portfolio peaked and has dropped 10%, go flat
        if value > peak:
            peak = value

        if not stop_triggered and (value / peak - 1.0) <= -TRAILING_STOP_PCT:
            stop_triggered = True
            target = 0.0
            invested = 0.0

        # Re-entry: allow re-entry once price recovers above stop threshold
        # Simple rule: allow re-entry on next BUY signal after stop
        if stop_triggered:
            if strategy == "hmm_blended":
                regime_probs = rec["hmm_regime_probs"]
                if regime_probs is None:
                    sig = rec["threshold_signal"]
                else:
                    # Use dominant regime for re-entry check
                    dominant_label = max(regime_probs, key=regime_probs.get)
                    if dominant_label.startswith("CHOPPY"):
                        dom_regime = Regime.CHOPPY
                    else:
                        dom_regime = Regime[dominant_label]
                    sig, _ = composite_signal_hybrid(
                        dom_regime,
                        rec["sma_signal"],
                        rec["xgb_signal"],
                        bear_weakening=rec["bear_weakening"],
                    )
            elif strategy == "hmm_hard":
                sig = rec["hmm_hard_signal"]
                if sig is None:
                    sig = rec["threshold_signal"]
            else:
                sig = rec["threshold_signal"]

            if sig == Signal.BUY:
                stop_triggered = False
                target = 1.0 - CASH_BUFFER
            else:
                target = 0.0

        # Apply transaction cost on position changes
        if vix is not None and invested != target:
            v = vix.get(dt, 20.0)
            if pd.isna(v):
                v = 20.0
            slippage = estimate_slippage(v)
            value -= value * abs(target - invested) * slippage

        invested = target

        # Mark to market
        value *= 1.0 + invested * daily_ret
        values.append(value)

    return np.array(values)


# ---------------------------------------------------------------------------
# Main run function
# ---------------------------------------------------------------------------

def run():
    print("Preparing features...")
    features_df = prepare_features()
    print(
        f"Feature matrix: {len(features_df)} rows, "
        f"{features_df.index.min().date()} to {features_df.index.max().date()}"
    )

    close = features_df["close"]

    # ── Load VIX data and join into features_df ─────────────────────────────
    from pathlib import Path
    data_dir = Path(__file__).parent / "data"
    ext_path = data_dir / "features_daily.csv"
    ext_df = pd.read_csv(ext_path, index_col="date", parse_dates=True)

    for col in ["vix_close", "vix3m_close"]:
        if col in ext_df.columns:
            vix_series = ext_df[col].copy()
            if hasattr(vix_series.index, "tz") and vix_series.index.tz is not None:
                vix_series.index = vix_series.index.tz_localize(None)
            vix_series.index = vix_series.index.normalize()
            features_df[col] = vix_series.reindex(features_df.index).ffill()

    if "vix_close" in features_df.columns and "vix3m_close" in features_df.columns:
        features_df["vix_term_structure"] = (
            features_df["vix_close"] / features_df["vix3m_close"]
        )

    # ── Walk-forward loop ────────────────────────────────────────────────────
    print("\nRunning walk-forward windows...")
    start_date = features_df.index.min()
    end_date = features_df.index.max()
    train_start = start_date

    records = []
    window_num = 0
    hmm_failures = 0

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

        # ── Train XGBoost ──────────────────────────────────────────────────
        X_train = train_data[XGB_FEATURES]
        y_train = train_data["label"]
        val_split = int(len(X_train) * 0.8)

        model = XGBClassifier(**XGB_PARAMS, n_estimators=500, early_stopping_rounds=20)
        model.fit(
            X_train[:val_split],
            y_train[:val_split],
            eval_set=[(X_train[val_split:], y_train[val_split:])],
            verbose=False,
        )
        class_order = model.classes_
        print(
            f"  Window {window_num}: train {train_data.index.min().date()}-"
            f"{train_data.index.max().date()}, "
            f"test {test_data.index.min().date()}-{test_data.index.max().date()}, "
            f"best_iter={model.best_iteration}"
        )

        # ── Train HMM ──────────────────────────────────────────────────────
        hmm_detector = None
        hmm_train = train_data.dropna(subset=HMM_FEATURES)
        if len(hmm_train) >= 100:
            try:
                hmm_detector = HMMRegimeDetector()
                hmm_detector.fit(hmm_train)
            except Exception as e:
                print(f"    NOTE: HMM training failed for window {window_num}: {e}")
                hmm_detector = None
                hmm_failures += 1
        else:
            print(f"    NOTE: Not enough HMM data for window {window_num} ({len(hmm_train)} rows)")
            hmm_failures += 1

        # ── Per-test-day predictions ───────────────────────────────────────
        for dt, row in test_data.iterrows():
            X_row = row[XGB_FEATURES].values.reshape(1, -1)
            proba = model.predict_proba(
                pd.DataFrame(X_row, columns=XGB_FEATURES)
            )[0]

            xgb_sig = _xgb_signal(proba, class_order, threshold=0.35)

            # Threshold regime
            regime = _regime_from_label(row["regime_label"])

            # SMA signal from historical close up to this date
            hist_end = features_df.index.get_loc(dt)
            close_up_to = close.iloc[: hist_end + 1]
            sma_sig = _sma_signal(close_up_to)

            # Bear weakening flag
            bear_weakening = False
            if regime == Regime.BEAR:
                lr5 = row.get("log_return_5d")
                if lr5 is not None and not np.isnan(float(lr5)) and float(lr5) > 0:
                    bear_weakening = True

            # Threshold composite signal
            threshold_signal, _ = composite_signal_hybrid(
                regime, sma_sig, xgb_sig, bear_weakening=bear_weakening
            )

            # HMM signals
            hmm_hard_signal = None
            hmm_regime_probs = None

            if hmm_detector is not None:
                hmm_row = row[HMM_FEATURES]
                if not hmm_row.isna().any():
                    try:
                        hmm_df_row = pd.DataFrame([hmm_row.values], columns=HMM_FEATURES)
                        probs_list = hmm_detector.predict_proba(hmm_df_row)
                        hmm_regime_probs = probs_list[0]

                        # HMM hard: use dominant regime through composite_signal_hybrid
                        dom_regime = hmm_detector.dominant_regime(hmm_regime_probs)
                        hmm_hard_signal, _ = composite_signal_hybrid(
                            dom_regime, sma_sig, xgb_sig, bear_weakening=bear_weakening
                        )
                    except Exception:
                        hmm_hard_signal = None
                        hmm_regime_probs = None

            records.append(
                {
                    "date": dt,
                    "sma_signal": sma_sig,
                    "xgb_signal": xgb_sig,
                    "bear_weakening": bear_weakening,
                    "threshold_signal": threshold_signal,
                    "hmm_hard_signal": hmm_hard_signal,
                    "hmm_regime_probs": hmm_regime_probs,
                }
            )

        train_start += pd.DateOffset(months=TEST_MONTHS)

    if not records:
        print("ERROR: No test records generated.")
        return

    if hmm_failures > 0:
        print(f"\n  Total HMM window failures (fell back to threshold): {hmm_failures}")

    print(f"\nTotal test-day records: {len(records)}")

    # ── Sort records by date ─────────────────────────────────────────────────
    records = sorted(records, key=lambda r: r["date"])

    # ── Simulate three strategies ────────────────────────────────────────────
    print("\nSimulating strategies (with VIX-based slippage)...")
    vix_series = features_df.get("vix_close")
    threshold_vals = simulate_strategy(records, close, "threshold", vix=vix_series)
    hmm_hard_vals = simulate_strategy(records, close, "hmm_hard", vix=vix_series)
    hmm_blended_vals = simulate_strategy(records, close, "hmm_blended", vix=vix_series)

    # ── SPY buy-and-hold benchmark ───────────────────────────────────────────
    test_dates = [r["date"] for r in records]
    first_dt = test_dates[0]
    last_dt = test_dates[-1]

    spy_slice = close[close.index >= first_dt].loc[:last_dt]
    if len(spy_slice) > 1:
        spy_vals = INITIAL * (spy_slice / spy_slice.iloc[0]).values
    else:
        spy_vals = np.array([INITIAL])

    # ── Compute overall metrics ──────────────────────────────────────────────
    years = (last_dt - first_dt).days / 365.25

    def annualized_return(total_ret: float, years: float) -> float:
        if years <= 0:
            return 0.0
        return (1.0 + total_ret) ** (1.0 / years) - 1.0

    t_sharpe, t_dd, t_ret = compute_metrics(threshold_vals)
    h_sharpe, h_dd, h_ret = compute_metrics(hmm_hard_vals)
    b_sharpe, b_dd, b_ret = compute_metrics(hmm_blended_vals)
    s_sharpe, s_dd, s_ret = compute_metrics(spy_vals)

    t_ann = annualized_return(t_ret, years)
    h_ann = annualized_return(h_ret, years)
    b_ann = annualized_return(b_ret, years)
    s_ann = annualized_return(s_ret, years)

    # ── Print summary table ──────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("HMM vs THRESHOLD BACKTEST COMPARISON")
    print(f"Period: {first_dt.date()} to {last_dt.date()} ({years:.1f} years)")
    print("=" * 75)
    header = f"{'Metric':<25} {'Threshold':>12} {'HMM Hard':>12} {'HMM Blended':>13} {'SPY B&H':>10}"
    print(header)
    print("-" * 75)
    print(f"{'Sharpe Ratio':<25} {t_sharpe:>12.3f} {h_sharpe:>12.3f} {b_sharpe:>13.3f} {s_sharpe:>10.3f}")
    print(f"{'Max Drawdown':<25} {t_dd:>11.1%}  {h_dd:>11.1%}  {b_dd:>12.1%}  {s_dd:>9.1%}")
    print(f"{'Ann. Return':<25} {t_ann:>11.1%}  {h_ann:>11.1%}  {b_ann:>12.1%}  {s_ann:>9.1%}")
    print(f"{'Final Value':<25} ${threshold_vals[-1]:>11,.0f} ${hmm_hard_vals[-1]:>11,.0f} ${hmm_blended_vals[-1]:>12,.0f} ${spy_vals[-1]:>9,.0f}")
    print("=" * 75)

    # ── Per-year breakdown ───────────────────────────────────────────────────
    records_df = pd.DataFrame(
        {
            "date": test_dates,
            "threshold": threshold_vals,
            "hmm_hard": hmm_hard_vals,
            "hmm_blended": hmm_blended_vals,
        }
    ).set_index("date")

    spy_series = pd.Series(spy_vals, index=spy_slice.index[: len(spy_vals)])

    print("\nPer-Year Breakdown")
    print("-" * 75)
    yr_header = f"{'Year':<8} {'Threshold':>10} {'HMM Hard':>10} {'HMM Blended':>13} {'SPY B&H':>10}"
    print(yr_header)
    print("-" * 75)

    all_years = sorted(records_df.index.year.unique())
    for yr in all_years:
        yr_mask = records_df.index.year == yr
        yr_df = records_df[yr_mask]
        if len(yr_df) < 2:
            continue

        def year_return(vals):
            return vals.iloc[-1] / vals.iloc[0] - 1.0

        t_yr = year_return(yr_df["threshold"])
        h_yr = year_return(yr_df["hmm_hard"])
        b_yr = year_return(yr_df["hmm_blended"])

        spy_yr_slice = spy_series[spy_series.index.year == yr]
        if len(spy_yr_slice) >= 2:
            s_yr = float(spy_yr_slice.iloc[-1] / spy_yr_slice.iloc[0] - 1.0)
        else:
            s_yr = float("nan")

        print(
            f"{yr:<8} {t_yr:>+9.1%}  {h_yr:>+9.1%}  {b_yr:>+12.1%}  {s_yr:>+9.1%}"
        )

    print("=" * 75)
    print("\nKey questions:")
    print(f"  Does HMM blended beat threshold?  Sharpe: {b_sharpe:.3f} vs {t_sharpe:.3f}")
    print(f"  Max drawdown under 10%?  Threshold: {t_dd:.1%}  HMM hard: {h_dd:.1%}  HMM blended: {b_dd:.1%}")
    print(f"  Blended vs threshold Sharpe improvement: {b_sharpe - t_sharpe:+.3f}")


if __name__ == "__main__":
    run()
