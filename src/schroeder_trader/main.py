import json
import logging
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path

import numpy as np
import pandas as pd

from schroeder_trader.config import (
    COMPOSITE_MODEL_PATH,
    DB_PATH,
    FEATURES_CSV_PATH,
    KELLY_MULTIPLIER,
    KELLY_WIN_LOSS_RATIO,
    PROJECT_ROOT,
    TICKER,
    TRAILING_STOP_PCT,
    TRAILING_STOP_COOLDOWN_DAYS,
    XGB_THRESHOLD_LOW,
)
from schroeder_trader.risk.kelly import kelly_fraction as compute_kelly_fraction, kelly_qty as compute_kelly_qty
from schroeder_trader.risk.trailing_stop import TrailingStop
from schroeder_trader.logging_config import setup_logging
from schroeder_trader.data.market_data import fetch_daily_bars, is_market_open_today
from schroeder_trader.strategy.sma_crossover import Signal, generate_signal
from schroeder_trader.strategy.composite import composite_signal_hybrid, count_consecutive_bear_days
from schroeder_trader.strategy.regime_detector import Regime, compute_regime_labels, compute_regime_series
from schroeder_trader.risk.risk_manager import evaluate
from schroeder_trader.execution.broker import (
    submit_order,
    get_order_status,
    get_position,
    get_account,
)
from schroeder_trader.storage.trade_log import (
    init_db,
    log_signal,
    log_order,
    log_portfolio,
    get_signal_by_date,
    get_pending_orders,
    update_order_fill,
    log_shadow_signal,
    get_shadow_signals,
)
from schroeder_trader.strategy.feature_engineer import FeaturePipeline
from schroeder_trader.strategy.xgboost_classifier import load_model
from schroeder_trader.alerts.email_alert import (
    send_trade_alert,
    send_fill_alert,
    send_error_alert,
    send_daily_summary,
)
from schroeder_trader.agents.daily_report import generate_daily_report

logger = logging.getLogger(__name__)


_ET = ZoneInfo("America/New_York")


def run_pipeline(db_path: Path = DB_PATH) -> None:
    """Run the full trading pipeline."""
    conn = init_db(db_path)
    try:
        _run_pipeline_inner(conn)
    finally:
        conn.close()


def _run_pipeline_inner(conn) -> None:
    now = datetime.now(timezone.utc)
    today = datetime.now(_ET).strftime("%Y-%m-%d")

    # Initialize trailing stop from DB state
    ts_row = conn.execute(
        "SELECT high_water_mark, trailing_stop_triggered, timestamp FROM shadow_signals "
        "WHERE high_water_mark IS NOT NULL ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if ts_row and ts_row["trailing_stop_triggered"]:
        stop_date = datetime.fromisoformat(ts_row["timestamp"]).date()
        trailing_stop = TrailingStop(
            TRAILING_STOP_PCT, TRAILING_STOP_COOLDOWN_DAYS,
            high_water_mark=ts_row["high_water_mark"], stop_date=stop_date,
        )
    elif ts_row:
        trailing_stop = TrailingStop(
            TRAILING_STOP_PCT, TRAILING_STOP_COOLDOWN_DAYS,
            high_water_mark=ts_row["high_water_mark"],
        )
    else:
        trailing_stop = TrailingStop(TRAILING_STOP_PCT, TRAILING_STOP_COOLDOWN_DAYS)

    # Step 1: Idempotency check
    existing = get_signal_by_date(conn, today)
    if existing is not None:
        logger.info("Already ran today (%s), exiting", today)
        return

    # Step 2: Fill check for pending orders
    pending = get_pending_orders(conn)
    for order in pending:
        try:
            status = get_order_status(order["alpaca_order_id"])
            if status["status"] == "filled":
                update_order_fill(
                    conn,
                    order["alpaca_order_id"],
                    status["fill_price"],
                    status["fill_timestamp"],
                    "FILLED",
                )
                send_fill_alert(
                    action=order["action"],
                    ticker=order["ticker"],
                    quantity=order["quantity"],
                    fill_price=status["fill_price"],
                )
                logger.info("Order %s filled at $%.2f", order["alpaca_order_id"], status["fill_price"])
            elif status["status"] in ("canceled", "expired", "rejected"):
                update_order_fill(conn, order["alpaca_order_id"], 0.0, now, "REJECTED")
                send_error_alert("Order rejected/canceled", f"Order {order['alpaca_order_id']} status: {status['status']}")
        except Exception:
            logger.exception("Error checking order %s", order["alpaca_order_id"])

    # Step 3: Market calendar check
    if not is_market_open_today(today):
        logger.info("Market closed today (%s), exiting", today)
        return

    # Step 4: Fetch data
    df = fetch_daily_bars(TICKER)
    close_price = float(df["close"].iloc[-1])

    # Step 5: Generate signal
    signal, sma_50, sma_200 = generate_signal(df)
    logger.info("Signal: %s | Close: $%.2f | SMA50: %.2f | SMA200: %.2f", signal.value, close_price, sma_50, sma_200)

    # Step 8 (partial): Log signal
    signal_id = log_signal(conn, now, TICKER, close_price, sma_50, sma_200, signal.value)

    # Step 6: Risk evaluation
    account = get_account()
    position_qty = get_position(TICKER)
    order_request = evaluate(
        signal=signal,
        portfolio_value=account["portfolio_value"],
        close_price=close_price,
        current_position_qty=position_qty,
    )

    # Step 7: Execute order
    if order_request is not None:
        result = submit_order(order_request, TICKER)
        log_order(
            conn, signal_id, result.alpaca_order_id,
            result.timestamp, TICKER, order_request.action,
            order_request.quantity, result.status,
        )
        send_trade_alert(
            action=order_request.action,
            ticker=TICKER,
            quantity=order_request.quantity,
            portfolio_value=account["portfolio_value"],
            cash=account["cash"],
            sma_50=sma_50,
            sma_200=sma_200,
        )

    # Step 8: Log portfolio snapshot
    account = get_account()  # refresh after potential trade
    position_qty = get_position(TICKER)
    position_value = position_qty * close_price
    log_portfolio(conn, now, account["cash"], position_qty, position_value, account["portfolio_value"])

    # Step 9: Daily summary
    send_daily_summary(
        portfolio_value=account["portfolio_value"],
        cash=account["cash"],
        position_qty=position_qty,
        signal=signal.value,
        sma_50=sma_50,
        sma_200=sma_200,
    )

    # Step 10: Composite shadow signal
    try:
        # Fetch/update external features (idempotent, skips if <24h old)
        try:
            result = subprocess.run(
                [sys.executable, str(PROJECT_ROOT / "backtest" / "download_features.py")],
                cwd=str(PROJECT_ROOT), capture_output=True, timeout=120,
            )
            if result.returncode != 0:
                logger.warning("External feature download failed (rc=%d), using cached data", result.returncode)
        except Exception:
            logger.warning("External feature download failed, using cached data")

        # Load composite model
        model = load_model(COMPOSITE_MODEL_PATH)
        if model is None:
            logger.info("No composite model at %s, skipping shadow step", COMPOSITE_MODEL_PATH)
        elif list(model.classes_) != [0, 1, 2]:
            logger.error("Model classes %s don't match expected [0, 1, 2]", list(model.classes_))
        elif not FEATURES_CSV_PATH.exists():
            logger.warning("No external features at %s, skipping shadow step", FEATURES_CSV_PATH)
        else:
            class_to_idx = {int(c): i for i, c in enumerate(model.classes_)}
            idx_up = class_to_idx[2]
            idx_down = class_to_idx[0]

            # Load external features
            ext_df = pd.read_csv(str(FEATURES_CSV_PATH), index_col="date", parse_dates=True)

            # Fetch 600 days of SPY data for feature + regime warmup
            # (regime needs 252-day rolling median + 20-day vol window)
            shadow_df = fetch_daily_bars(TICKER, days=600)

            # Compute extended features
            pipeline = FeaturePipeline()
            features = pipeline.compute_features_extended(shadow_df, ext_df)

            if len(features) > 0:
                # Compute regime labels (backward-looking)
                regime_series = compute_regime_series(features)
                features["regime_label"] = compute_regime_labels(features)

                # Today's data
                today_regime = regime_series.iloc[-1]
                if not isinstance(today_regime, Regime):
                    today_regime = Regime.CHOPPY

                bear_days = count_consecutive_bear_days(regime_series)

                # XGBoost signals at both thresholds
                feature_cols = [
                    "log_return_5d", "log_return_20d", "volatility_20d",
                    "credit_spread", "dollar_momentum", "regime_label",
                ]
                last_row = features[feature_cols].iloc[[-1]]

                if last_row.isna().any().any():
                    logger.warning("NaN in feature row, skipping shadow signal")
                else:
                    proba = model.predict_proba(last_row)[0]
                    pred_class = int(np.argmax(proba))
                    proba_dict = {
                        "DOWN": float(proba[idx_down]),
                        "FLAT": float(proba[class_to_idx[1]]),
                        "UP": float(proba[idx_up]),
                    }

                    # Low threshold for Choppy regime
                    if pred_class == idx_up and proba[idx_up] > XGB_THRESHOLD_LOW:
                        xgb_low = Signal.BUY
                    elif pred_class == idx_down and proba[idx_down] > XGB_THRESHOLD_LOW:
                        xgb_low = Signal.SELL
                    else:
                        xgb_low = Signal.HOLD

                    # Route composite signal
                    composite_sig, source = composite_signal_hybrid(
                        today_regime, signal, xgb_low,
                    )

                    # Always compute Kelly sizing for analysis
                    k_frac = compute_kelly_fraction(
                        p_up=proba[idx_up],
                        p_down=proba[idx_down],
                        win_loss_ratio=KELLY_WIN_LOSS_RATIO,
                        kelly_multiplier=KELLY_MULTIPLIER,
                    )
                    k_qty = compute_kelly_qty(k_frac, account["portfolio_value"], close_price)

                    # Evaluate trailing stop
                    ts_trading_dates = [
                        datetime.fromisoformat(r["timestamp"]).date()
                        for r in conn.execute(
                            "SELECT timestamp FROM shadow_signals ORDER BY id"
                        ).fetchall()
                    ]
                    today_date = datetime.now(_ET).date()
                    ts_triggered = trailing_stop.update(
                        account["portfolio_value"], today_date,
                        trading_dates=ts_trading_dates,
                    )
                    ts_in_cooldown = trailing_stop.in_cooldown(today_date, ts_trading_dates)

                    # Log shadow signal (always log XGB prediction for analysis)
                    log_shadow_signal(
                        conn, now, TICKER, close_price,
                        predicted_class=pred_class,
                        predicted_proba=json.dumps(proba_dict),
                        ml_signal=composite_sig.value,
                        sma_signal=signal.value,
                        regime=today_regime.value,
                        signal_source=source,
                        bear_day_count=bear_days if today_regime == Regime.BEAR else None,
                        kelly_fraction=k_frac,
                        kelly_qty=k_qty,
                        high_water_mark=trailing_stop.high_water_mark,
                        trailing_stop_triggered=ts_triggered or ts_in_cooldown,
                    )
                    logger.info(
                        "Shadow composite: %s (source=%s, regime=%s, bear_days=%d, kelly=%.3f, kelly_qty=%d, hwm=%.0f, ts_stop=%s)",
                        composite_sig.value, source, today_regime.value, bear_days,
                        k_frac or 0.0, k_qty or 0,
                        trailing_stop.high_water_mark, ts_triggered or ts_in_cooldown,
                    )
    except Exception:
        logger.exception("Shadow composite prediction failed (non-fatal)")

    # Step 11: LLM daily intelligence report (non-fatal)
    try:
        recent = get_shadow_signals(conn)[-10:]
        if recent:
            account = get_account()
            report = generate_daily_report(recent[-1], recent, account)
            send_daily_summary(
                portfolio_value=account["portfolio_value"],
                cash=account["cash"],
                position_qty=get_position(TICKER),
                signal=signal.value,
                sma_50=sma_50,
                sma_200=sma_200,
                llm_report=report,
            )
            logger.info("LLM daily report sent")
    except Exception:
        logger.exception("LLM report generation failed (non-fatal)")

    logger.info("Pipeline complete")


def main() -> None:
    setup_logging()
    try:
        run_pipeline()
    except Exception:
        logger.exception("Pipeline failed")
        # Note: error logging to SQLite is skipped here because the error may have
        # occurred before init_db completed. Errors are captured via Python logging
        # (file + console) and the email alert. Phase 4 can add structured error
        # logging to SQLite once the risk layer guarantees DB availability.
        try:
            send_error_alert("Pipeline failure", traceback.format_exc())
        except Exception:
            logger.exception("Failed to send error alert")
        sys.exit(1)


if __name__ == "__main__":
    main()
