import json
import logging
import subprocess
import sys
import traceback
from datetime import datetime, timezone
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
)
from schroeder_trader.risk.kelly import kelly_fraction as compute_kelly_fraction, kelly_qty as compute_kelly_qty
from schroeder_trader.logging_config import setup_logging
from schroeder_trader.data.market_data import fetch_daily_bars, is_market_open_today
from schroeder_trader.strategy.sma_crossover import Signal, generate_signal
from schroeder_trader.strategy.composite import composite_signal_hybrid, count_consecutive_bear_days
from schroeder_trader.strategy.regime_detector import Regime, detect_regime
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
)
from schroeder_trader.strategy.feature_engineer import FeaturePipeline
from schroeder_trader.strategy.xgboost_classifier import load_model
from schroeder_trader.alerts.email_alert import (
    send_trade_alert,
    send_fill_alert,
    send_error_alert,
    send_daily_summary,
)

logger = logging.getLogger(__name__)


def run_pipeline(db_path: Path = DB_PATH) -> None:
    """Run the full trading pipeline."""
    conn = init_db(db_path)
    now = datetime.now(timezone.utc)
    today = now.strftime("%Y-%m-%d")

    # Step 1: Idempotency check
    existing = get_signal_by_date(conn, today)
    if existing is not None:
        logger.info("Already ran today (%s), exiting", today)
        conn.close()
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
        conn.close()
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
            subprocess.run(
                ["uv", "run", "python", str(PROJECT_ROOT / "backtest" / "download_features.py")],
                cwd=str(PROJECT_ROOT), capture_output=True, timeout=120,
            )
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
                log_ret_20d = np.log(features["close"] / features["close"].shift(20))
                vol_20d = features["close"].pct_change().rolling(20).std()
                vol_med = vol_20d.rolling(252).median()

                regime_series = pd.Series(index=features.index, dtype=object)
                for idx in range(len(features)):
                    lr = log_ret_20d.iloc[idx]
                    vol = vol_20d.iloc[idx]
                    vm = vol_med.iloc[idx]
                    if pd.isna(lr) or pd.isna(vol) or pd.isna(vm):
                        regime_series.iloc[idx] = np.nan
                    else:
                        regime_series.iloc[idx] = detect_regime(lr, vol, vm)

                # Add regime_label as integer feature
                regime_map = {Regime.BEAR: 0, Regime.CHOPPY: 1, Regime.BULL: 2}
                features["regime_label"] = regime_series.map(regime_map)

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

                    # Low threshold (0.35) for Choppy
                    if pred_class == idx_up and proba[idx_up] > 0.35:
                        xgb_low = Signal.BUY
                    elif pred_class == idx_down and proba[idx_down] > 0.35:
                        xgb_low = Signal.SELL
                    else:
                        xgb_low = Signal.HOLD

                    # High threshold (0.50) for late Bear
                    if pred_class == idx_up and proba[idx_up] > 0.50:
                        xgb_high = Signal.BUY
                    elif pred_class == idx_down and proba[idx_down] > 0.50:
                        xgb_high = Signal.SELL
                    else:
                        xgb_high = Signal.HOLD

                    # Route composite signal
                    composite_sig, source = composite_signal_hybrid(
                        today_regime, signal, xgb_low, xgb_high, bear_days,
                    )

                    # Compute Kelly sizing (XGB sources only)
                    k_frac = None
                    k_qty = None
                    if source == "XGB":
                        k_frac = compute_kelly_fraction(
                            p_up=proba[idx_up],
                            p_down=proba[idx_down],
                            win_loss_ratio=KELLY_WIN_LOSS_RATIO,
                            kelly_multiplier=KELLY_MULTIPLIER,
                        )
                        k_qty = compute_kelly_qty(k_frac, account["portfolio_value"], close_price)

                    # Log shadow signal
                    log_shadow_signal(
                        conn, now, TICKER, close_price,
                        predicted_class=pred_class if source == "XGB" else None,
                        predicted_proba=json.dumps(proba_dict) if source == "XGB" else None,
                        ml_signal=composite_sig.value,
                        sma_signal=signal.value,
                        regime=today_regime.value,
                        signal_source=source,
                        bear_day_count=bear_days if today_regime == Regime.BEAR else None,
                        kelly_fraction=k_frac,
                        kelly_qty=k_qty,
                    )
                    logger.info(
                        "Shadow composite: %s (source=%s, regime=%s, bear_days=%d, kelly=%.3f, kelly_qty=%d)",
                        composite_sig.value, source, today_regime.value, bear_days,
                        k_frac or 0.0, k_qty or 0,
                    )
    except Exception:
        logger.exception("Shadow composite prediction failed (non-fatal)")

    conn.close()
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
