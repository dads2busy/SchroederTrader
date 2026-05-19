"""Per-ticker signal + trailing-stop loop for the basket pipeline.

For each ticker in the basket, this module computes the composite signal
(via the existing strategy modules), runs the per-ticker TrailingStop, and
returns a decision dict consumed by the rebalancer. Decisions are also
logged to shadow_signals.csv with pipeline='basket'.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd

from schroeder_trader.config import (
    COMPOSITE_MODEL_PATH,
    SHADOW_TICKERS,
    TRAILING_STOP_PCT,
    TRAILING_STOP_COOLDOWN_DAYS,
    XGB_THRESHOLD_LOW,
)
from schroeder_trader.data.market_data import fetch_daily_bars
from schroeder_trader.risk.trailing_stop import TrailingStop
from schroeder_trader.storage.csv_store import CsvStore
from schroeder_trader.storage.trade_log import log_shadow_signal
from schroeder_trader.strategy.composite import (
    Signal, composite_signal_hybrid, count_consecutive_bear_days,
)
from schroeder_trader.strategy.feature_engineer import FeaturePipeline
from schroeder_trader.strategy.regime_detector import (
    Regime, compute_regime_series, compute_regime_labels,
)
from schroeder_trader.strategy.sma_crossover import generate_signal
from schroeder_trader.strategy.xgboost_classifier import load_model

from schroeder_trader.basket.portfolio import (
    is_basket_cold_start,
    prior_exposure,
    read_trading_dates,
)


def _compute_signal_for_ticker(
    ticker: str, model_path: Path, ext_df: pd.DataFrame,
):
    """Run the composite-signal pipeline for one ticker.

    Returns (composite_signal: Signal, source: str, regime: Regime,
             bear_days: int, last_bar: dict). Mirrors the structure of
    _run_shadow_for_ticker in main.py but without the side effects.
    """
    model = load_model(model_path)
    if model is None:
        raise RuntimeError(f"Basket {ticker}: no model at {model_path}")
    if list(model.classes_) != [0, 1, 2]:
        raise RuntimeError(f"Basket {ticker}: unexpected model classes {list(model.classes_)}")
    df = fetch_daily_bars(ticker, days=600)
    pipeline = FeaturePipeline()
    features = pipeline.compute_features_extended(df, ext_df)
    if len(features) == 0:
        raise RuntimeError(f"Shadow {ticker}: no features computed")

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
    if last_row.isna().any().any():
        raise RuntimeError(f"Shadow {ticker}: NaN in feature row")

    class_to_idx = {int(c): i for i, c in enumerate(model.classes_)}
    idx_up = class_to_idx[2]
    idx_down = class_to_idx[0]
    proba = model.predict_proba(last_row)[0]
    pred_class = int(np.argmax(proba))
    proba_dict = {
        "DOWN": float(proba[idx_down]),
        "FLAT": float(proba[class_to_idx[1]]),
        "UP": float(proba[idx_up]),
    }
    if pred_class == idx_up and proba[idx_up] > XGB_THRESHOLD_LOW:
        xgb_low = Signal.BUY
    elif pred_class == idx_down and proba[idx_down] > XGB_THRESHOLD_LOW:
        xgb_low = Signal.SELL
    else:
        xgb_low = Signal.HOLD

    sma_signal, _, _ = generate_signal(df)

    composite_sig, source = composite_signal_hybrid(
        today_regime, sma_signal, xgb_low, bear_weakening=bear_weakening,
    )

    last_bar = {
        "close": float(df["close"].iloc[-1]),
        "pred_class": pred_class,
        "proba_json": json.dumps(proba_dict),
        "sma_signal": sma_signal.value,
    }
    return composite_sig, source, today_regime, bear_days, last_bar


def compute_decisions(
    store: CsvStore,
    weights: dict[str, float],
    ext_df: pd.DataFrame,
    now: datetime,
    portfolio_value: float,
) -> dict[str, dict]:
    """Compute the per-ticker decision dict for one daily basket run.

    Side effect: writes one row per ticker to shadow_signals.csv with
    pipeline='basket' capturing the decision (including HWM and stop state).
    """
    cold_start = is_basket_cold_start(store)
    if cold_start:
        logger.info(
            "Basket cold start detected — force-investing to target weights "
            "regardless of per-ticker signal"
        )

    decisions: dict[str, dict] = {}
    for ticker in weights:
        # Resolve model path: SPY uses the production model, others use SHADOW_TICKERS
        model_path = COMPOSITE_MODEL_PATH if ticker == "SPY" else SHADOW_TICKERS[ticker]

        signal, source, regime, bear_days, last_bar = \
            _compute_signal_for_ticker(ticker, model_path, ext_df)

        # Per-ticker trailing stop. HWM persists via shadow_signals.csv rows.
        ts = _load_or_create_stop(store, ticker)
        trading_dates = read_trading_dates(store, ticker)
        ts_triggered = ts.update(portfolio_value, now.date(), trading_dates=trading_dates)
        in_cooldown = ts.in_cooldown(now.date(), trading_dates)

        if ts_triggered or in_cooldown:
            exposure = 0.0
        elif cold_start:
            # Force-invest on cold start: bootstrap to target weights regardless
            # of per-ticker signal (per user-chosen semantics). Subsequent runs
            # use the standard HOLD-carries-prior logic below.
            exposure = 1.0
            if signal == Signal.SELL:
                logger.warning(
                    "Basket cold start: overriding SELL signal for %s to BUY-to-target. "
                    "If model is bearish on this ticker, consider revisiting cold-start policy.",
                    ticker,
                )
        elif signal == Signal.BUY:
            exposure = 1.0
        elif signal == Signal.SELL:
            exposure = 0.0
        else:  # HOLD
            exposure = prior_exposure(store, ticker)

        decisions[ticker] = {
            "signal": signal.value,
            "exposure": exposure,
            "price": last_bar["close"],
            "regime": regime.value,
            "source": source,
            "stop_state": {
                "triggered_today_or_cooldown": bool(ts_triggered or in_cooldown),
                "high_water_mark": float(ts.high_water_mark),
            },
        }

        log_shadow_signal(
            store, now, ticker, last_bar["close"],
            predicted_class=last_bar["pred_class"],
            predicted_proba=last_bar["proba_json"],
            ml_signal=signal.value,
            sma_signal=last_bar["sma_signal"],
            regime=regime.value,
            signal_source=source,
            bear_day_count=bear_days if regime == Regime.BEAR else None,
            high_water_mark=ts.high_water_mark,
            trailing_stop_triggered=ts_triggered or in_cooldown,
            pipeline="basket",
        )

    return decisions


def _load_or_create_stop(store: CsvStore, ticker: str) -> TrailingStop:
    """Load the per-ticker stop's HWM and stop_date from the latest basket
    shadow_signals row. Restores stop_date when the prior row was triggered
    so the cooldown logic correctly counts days from the trigger."""
    df = store.read("shadow_signals")
    if not df.empty:
        rows = df[(df["pipeline"] == "basket") & (df["ticker"] == ticker)]
        if not rows.empty:
            latest = rows.sort_values("timestamp").iloc[-1]
            hwm = float(latest["high_water_mark"]) if pd.notna(latest["high_water_mark"]) else 0.0
            stop_date = None
            triggered = latest["trailing_stop_triggered"]
            if pd.notna(triggered) and int(triggered) == 1:
                stop_date = pd.Timestamp(latest["timestamp"]).date()
            return TrailingStop(
                drawdown_pct=TRAILING_STOP_PCT,
                cooldown_days=TRAILING_STOP_COOLDOWN_DAYS,
                high_water_mark=hwm,
                stop_date=stop_date,
            )
    return TrailingStop(
        drawdown_pct=TRAILING_STOP_PCT,
        cooldown_days=TRAILING_STOP_COOLDOWN_DAYS,
    )
