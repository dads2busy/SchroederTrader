"""Replay composite signal + LLM oracles for a specific historical close.

Usage:
    uv run python scripts/replay_date.py YYYY-MM-DD [POSITION_QTY] [PORTFOLIO_VALUE]

- Computes what the composite model would have recommended at that day's close.
- Queries both LLM oracles with that date's data and logs rows to llm_shadow_signals.
- Does NOT write to signals / shadow_signals / portfolio (won't contaminate real data).

Defaults: POSITION_QTY=0, PORTFOLIO_VALUE=100000.
"""

import sys
from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from schroeder_trader.agents.llm_oracle import OracleInput, query_all
from schroeder_trader.config import (
    COMPOSITE_MODEL_PATH,
    DB_PATH,
    FEATURES_CSV_PATH,
    TICKER,
    XGB_THRESHOLD_LOW,
)
from schroeder_trader.data.market_data import fetch_daily_bars
from schroeder_trader.logging_config import setup_logging
from schroeder_trader.storage.trade_log import init_db, log_llm_signal
from schroeder_trader.strategy.composite import (
    composite_signal_hybrid,
    count_consecutive_bear_days,
)
from schroeder_trader.strategy.feature_engineer import FeaturePipeline
from schroeder_trader.strategy.regime_detector import (
    Regime,
    compute_regime_labels,
    compute_regime_series,
)
from schroeder_trader.strategy.sma_crossover import Signal, generate_signal
from schroeder_trader.strategy.xgboost_classifier import load_model

_ET = ZoneInfo("America/New_York")


def _truncate_to_date(df: pd.DataFrame, date_str: str) -> pd.DataFrame:
    cutoff = pd.Timestamp(date_str)
    # Handle both tz-naive and tz-aware indices
    if df.index.tz is not None:
        cutoff = cutoff.tz_localize(df.index.tz) if cutoff.tz is None else cutoff
    return df[df.index <= cutoff]


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    date_str = sys.argv[1]
    position_qty = int(sys.argv[2]) if len(sys.argv) >= 3 else 0
    portfolio_value = float(sys.argv[3]) if len(sys.argv) >= 4 else 100_000.0

    setup_logging()

    # 1. Fetch SPY bars and truncate to the replay date
    df = fetch_daily_bars(TICKER, days=600)
    df = _truncate_to_date(df, date_str)
    if len(df) == 0:
        print(f"ERROR: no bars on or before {date_str}")
        sys.exit(1)
    last_bar_date = df.index[-1].strftime("%Y-%m-%d")
    close_price = float(df["close"].iloc[-1])

    print(f"\nReplay {date_str} (last bar in fetched range: {last_bar_date})")
    print(f"SPY close: ${close_price:.2f}\n")

    # 2. SMA signal
    sma_signal, sma_50, sma_200 = generate_signal(df)

    # 3. Composite block
    model = load_model(COMPOSITE_MODEL_PATH)
    ext_df = pd.read_csv(str(FEATURES_CSV_PATH), index_col="date", parse_dates=True)
    ext_df = _truncate_to_date(ext_df, date_str)

    pipeline = FeaturePipeline()
    features = pipeline.compute_features_extended(df, ext_df)

    regime_series = compute_regime_series(features)
    features["regime_label"] = compute_regime_labels(features)
    today_regime = regime_series.iloc[-1]
    if not isinstance(today_regime, Regime):
        today_regime = Regime.CHOPPY
    bear_days = count_consecutive_bear_days(regime_series)

    bear_weakening = False
    if today_regime == Regime.BEAR and "log_return_5d" in features.columns:
        lr5 = features["log_return_5d"].iloc[-1]
        bear_weakening = not pd.isna(lr5) and lr5 > 0

    feature_cols = [
        "log_return_5d", "log_return_20d", "volatility_20d",
        "credit_spread", "dollar_momentum", "regime_label",
    ]
    last_row = features[feature_cols].iloc[[-1]]
    class_to_idx = {int(c): i for i, c in enumerate(model.classes_)}
    idx_up = class_to_idx[2]
    idx_down = class_to_idx[0]
    idx_flat = class_to_idx[1]

    proba = model.predict_proba(last_row)[0]
    if int(np.argmax(proba)) == idx_up and proba[idx_up] > XGB_THRESHOLD_LOW:
        xgb_low = Signal.BUY
    elif int(np.argmax(proba)) == idx_down and proba[idx_down] > XGB_THRESHOLD_LOW:
        xgb_low = Signal.SELL
    else:
        xgb_low = Signal.HOLD

    composite_sig, source = composite_signal_hybrid(
        today_regime, sma_signal, xgb_low, bear_weakening=bear_weakening,
    )

    # Effective action given current position
    if composite_sig == Signal.BUY and position_qty > 0:
        action_summary = "HOLD (composite=BUY but already long)"
    elif composite_sig == Signal.SELL and position_qty == 0:
        action_summary = "HOLD (composite=SELL but already flat)"
    elif composite_sig == Signal.BUY:
        action_summary = "BUY (open position)"
    elif composite_sig == Signal.SELL:
        action_summary = "SELL (close position)"
    else:
        action_summary = "HOLD"

    print("=" * 64)
    print(f"SYSTEM RECOMMENDATION for next session (Monday if {date_str} was Fri)")
    print("=" * 64)
    print(f"  Regime: {today_regime.value} (bear_days={bear_days}, weakening={bear_weakening})")
    print(f"  XGB probs: DOWN={proba[idx_down]:.3f}  FLAT={proba[idx_flat]:.3f}  UP={proba[idx_up]:.3f}")
    print(f"  XGB signal: {xgb_low.value}")
    print(f"  SMA signal: {sma_signal.value} (50=${sma_50:.2f}, 200=${sma_200:.2f})")
    print(f"  Composite:  {composite_sig.value} (source={source})")
    print(f"  Given position qty={position_qty}: {action_summary}")
    print()

    # 4. LLM oracles
    recent_closes = df["close"].tail(20).astype(float).tolist()
    inp = OracleInput(
        date_str=date_str,
        current_price=close_price,
        recent_closes=recent_closes,
        position_qty=position_qty,
        portfolio_value=portfolio_value,
    )
    print("Querying Claude + ChatGPT oracles (30-60s with web search)...\n")
    responses = query_all(inp)

    # Log to DB with the replay date's 4:30 PM ET timestamp
    y, m, d = (int(x) for x in date_str.split("-"))
    replay_ts = datetime.combine(datetime(y, m, d), time(16, 30), _ET).astimezone(timezone.utc)
    conn = init_db(DB_PATH)
    try:
        for resp in responses:
            log_llm_signal(
                conn, replay_ts, TICKER, close_price,
                provider=resp.provider, model=resp.model,
                action=resp.action, target_exposure=resp.target_exposure,
                confidence=resp.confidence,
                regime_assessment=resp.regime_assessment,
                key_drivers=resp.key_drivers, reasoning=resp.reasoning,
                raw_response=resp.raw_response, error=resp.error,
            )
    finally:
        conn.close()

    # 5. Print comparison
    current_exposure = (position_qty * close_price) / portfolio_value if portfolio_value > 0 else 0.0
    system_target = 0.98 if composite_sig == Signal.BUY else 0.0 if composite_sig == Signal.SELL else current_exposure

    for resp in responses:
        print(f"=== {resp.provider.upper()} ({resp.model}) ===")
        if resp.error:
            print(f"  ERROR: {resp.error}")
        else:
            print(f"  Action:          {resp.action}")
            print(f"  Target exposure: {resp.target_exposure:.2f} ({resp.target_exposure:.1%})")
            print(f"  Confidence:      {resp.confidence}")
            print(f"  Regime:          {resp.regime_assessment}")
            print(f"  Drivers:         {', '.join(resp.key_drivers) or '(none)'}")
            print(f"  Reasoning:       {resp.reasoning}")
        print()

    print("=" * 64)
    print("HEAD-TO-HEAD")
    print("=" * 64)
    print(f"  Current exposure: {current_exposure:.1%}")
    print()
    print(f"  SYSTEM : {composite_sig.value:4s}  target={system_target:.2f}  (source={source}, regime={today_regime.value})")
    for resp in responses:
        if resp.error:
            print(f"  {resp.provider.upper():7s}: ERROR ({resp.error[:60]})")
        else:
            print(f"  {resp.provider.upper():7s}: {resp.action:4s}  target={resp.target_exposure:.2f}  ({resp.confidence} conf, regime={resp.regime_assessment})")
    print()


if __name__ == "__main__":
    main()
