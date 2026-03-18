import json
import logging
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from schroeder_trader.config import DB_PATH, PROJECT_ROOT, TICKER
from schroeder_trader.logging_config import setup_logging
from schroeder_trader.data.market_data import fetch_daily_bars, is_market_open_today
from schroeder_trader.strategy.sma_crossover import generate_signal
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
from schroeder_trader.strategy.xgboost_classifier import load_model, predict_signal
from schroeder_trader.alerts.email_alert import (
    send_trade_alert,
    send_fill_alert,
    send_error_alert,
    send_daily_summary,
)

logger = logging.getLogger(__name__)

MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_spy.json"


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

    # Step 10: Shadow ML prediction
    try:
        model = load_model(MODEL_PATH)
        if model is not None:
            shadow_df = fetch_daily_bars(TICKER, days=400)
            pipeline = FeaturePipeline()
            features = pipeline.compute_features(shadow_df)
            if len(features) > 0:
                last_row = features.iloc[[-1]]
                feature_cols = [
                    "log_return_5d", "log_return_20d", "volatility_20d",
                    "sma_ratio", "volume_ratio", "rsi_14",
                ]
                ml_signal, proba = predict_signal(model, last_row[feature_cols])
                class_map = {"SELL": 0, "HOLD": 1, "BUY": 2}
                pred_class = class_map.get(ml_signal.value, 1)
                log_shadow_signal(
                    conn, now, TICKER, close_price,
                    predicted_class=pred_class,
                    predicted_proba=json.dumps(proba),
                    ml_signal=ml_signal.value,
                    sma_signal=signal.value,
                )
                logger.info("Shadow ML signal: %s (UP=%.2f, FLAT=%.2f, DOWN=%.2f)",
                           ml_signal.value, proba["UP"], proba["FLAT"], proba["DOWN"])
    except Exception:
        logger.exception("Shadow ML prediction failed (non-fatal)")

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
