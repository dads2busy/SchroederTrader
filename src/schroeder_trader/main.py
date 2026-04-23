import json
import logging
import signal
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path

# Guard against silent network hangs. Alpaca/Anthropic/SMTP clients inherit
# the default socket timeout when they don't set their own.
_SOCKET_TIMEOUT_SECONDS = 60
# Wall-clock ceiling for the full pipeline. If anything blocks past this,
# SIGALRM raises TimeoutError and the main() handler sends an alert.
_PIPELINE_DEADLINE_SECONDS = 600
# DNS backoff schedule (seconds before attempts 1..N). Handles the
# wake-from-sleep case where launchd fires before the network is ready.
_NETWORK_READY_BACKOFFS = [0, 15, 30, 45]
_NETWORK_READY_HOST = "paper-api.alpaca.markets"

import numpy as np
import pandas as pd

from schroeder_trader.config import (
    COMPOSITE_MODEL_PATH,
    DB_PATH,
    FEATURES_CSV_PATH,
    HMM_MODEL_PATH,
    KELLY_MULTIPLIER,
    KELLY_WIN_LOSS_RATIO,
    PROJECT_ROOT,
    TICKER,
    TRAILING_STOP_PCT,
    STALE_CASH_DAYS,
    TRAILING_STOP_COOLDOWN_DAYS,
    XGB_THRESHOLD_LOW,
)
from schroeder_trader.risk.kelly import kelly_fraction as compute_kelly_fraction, kelly_qty as compute_kelly_qty
from schroeder_trader.risk.trailing_stop import TrailingStop
from schroeder_trader.logging_config import setup_logging
from schroeder_trader.data.market_data import fetch_daily_bars, is_market_open_today
from schroeder_trader.strategy.sma_crossover import Signal, generate_signal
from schroeder_trader.strategy.composite import composite_signal_hybrid, composite_signal_blended, count_consecutive_bear_days, stale_cash_override
from schroeder_trader.strategy.regime_detector import Regime, HMMRegimeDetector, compute_regime_labels, compute_regime_series
from schroeder_trader.risk.risk_manager import evaluate
from schroeder_trader.execution.broker import (
    submit_order,
    get_order_status,
    get_position,
    get_account,
)
from schroeder_trader.execution.reconcile import reconcile_orders
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
    log_llm_signal,
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
from schroeder_trader.agents.llm_oracle import OracleInput, query_all

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

    # Step 1: Reconcile orphaned orders (from prior crash between submit and log)
    try:
        orphans = reconcile_orders(conn, TICKER)
        if orphans:
            send_error_alert(
                "Orphaned orders reconciled",
                f"Found {len(orphans)} orders at Alpaca with no DB record: {orphans}",
            )
    except Exception:
        logger.exception("Order reconciliation failed (non-fatal)")

    # Step 2: Fill check for pending orders (runs before idempotency so a crashed
    # run's pending orders still get updated on manual re-run)
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
                slippage = status["fill_price"] - order["signal_close_price"] if order.get("signal_close_price") else None
                if slippage is not None:
                    logger.info("Order %s filled at $%.2f (slippage: $%.2f/share)", order["alpaca_order_id"], status["fill_price"], slippage)
                else:
                    logger.info("Order %s filled at $%.2f", order["alpaca_order_id"], status["fill_price"])
            elif status["status"] in ("canceled", "expired", "rejected"):
                update_order_fill(conn, order["alpaca_order_id"], 0.0, now, "REJECTED")
                send_error_alert("Order rejected/canceled", f"Order {order['alpaca_order_id']} status: {status['status']}")
        except Exception:
            logger.exception("Error checking order %s", order["alpaca_order_id"])

    # Step 3: Idempotency check (after fill + reconcile so those always run)
    existing = get_signal_by_date(conn, today)
    if existing is not None:
        logger.info("Already ran today (%s), exiting", today)
        return

    # Step 4: Market calendar check
    if not is_market_open_today(today):
        logger.info("Market closed today (%s), exiting", today)
        return

    # Step 5: Fetch data
    df = fetch_daily_bars(TICKER)
    close_price = float(df["close"].iloc[-1])

    # Step 6: Generate signal
    signal, sma_50, sma_200 = generate_signal(df)
    logger.info("Signal: %s | Close: $%.2f | SMA50: %.2f | SMA200: %.2f", signal.value, close_price, sma_50, sma_200)

    # Log signal
    signal_id = log_signal(conn, now, TICKER, close_price, sma_50, sma_200, signal.value)

    # Step 7: Get account state
    account = get_account()
    position_qty = get_position(TICKER)
    position_value = position_qty * close_price

    # Step 8: Compute composite signal (HOLD on failure — never trade blind)
    effective_signal = Signal.HOLD
    llm_report = None
    try:
        # Fetch/update external features (idempotent, skips if <24h old)
        try:
            logger.info("Downloading external features...")
            result = subprocess.run(
                [sys.executable, str(PROJECT_ROOT / "backtest" / "download_features.py")],
                cwd=str(PROJECT_ROOT), capture_output=True, timeout=120,
            )
            if result.returncode != 0:
                stderr_snippet = (result.stderr or b"").decode(errors="replace")[-200:]
                logger.warning("External feature download failed (rc=%d): %s", result.returncode, stderr_snippet)
        except subprocess.TimeoutExpired:
            logger.warning("External feature download timed out after 120s, using cached data")
        except Exception:
            logger.warning("External feature download failed, using cached data", exc_info=True)

        # Load composite model
        logger.info("Loading composite model...")
        model = load_model(COMPOSITE_MODEL_PATH)
        if model is None:
            logger.info("No composite model at %s, holding", COMPOSITE_MODEL_PATH)
        elif list(model.classes_) != [0, 1, 2]:
            logger.error("Model classes %s don't match expected [0, 1, 2], holding", list(model.classes_))
        elif not FEATURES_CSV_PATH.exists():
            logger.warning("No external features at %s, holding", FEATURES_CSV_PATH)
        else:
            class_to_idx = {int(c): i for i, c in enumerate(model.classes_)}
            idx_up = class_to_idx[2]
            idx_down = class_to_idx[0]

            # Load HMM regime detector
            hmm_detector = None
            if HMM_MODEL_PATH.exists():
                try:
                    hmm_detector = HMMRegimeDetector.load(HMM_MODEL_PATH)
                    logger.info("HMM regime detector loaded (%d states)", hmm_detector.n_states)
                except Exception:
                    logger.warning("Failed to load HMM detector, falling back to threshold", exc_info=True)

            # Load external features
            ext_df = pd.read_csv(str(FEATURES_CSV_PATH), index_col="date", parse_dates=True)
            logger.info("External features loaded: %d rows, last date %s", len(ext_df), ext_df.index[-1])

            # Fetch 600 days of SPY data for feature + regime warmup
            # (regime needs 252-day rolling median + 20-day vol window)
            shadow_df = fetch_daily_bars(TICKER, days=600)
            logger.info("Bars fetched: %d rows, last date %s", len(shadow_df), shadow_df.index[-1])

            # Compute extended features
            pipeline = FeaturePipeline()
            features = pipeline.compute_features_extended(shadow_df, ext_df)
            logger.info("Extended features computed: %d rows", len(features))

            if len(features) > 0:
                # Compute regime labels (backward-looking)
                regime_series = compute_regime_series(features)
                features["regime_label"] = compute_regime_labels(features)

                # Today's data
                today_regime = regime_series.iloc[-1]
                if not isinstance(today_regime, Regime):
                    today_regime = Regime.CHOPPY

                bear_days = count_consecutive_bear_days(regime_series)

                # Bear weakening: positive 5-day return while in BEAR
                bear_weakening = False
                if today_regime == Regime.BEAR and "log_return_5d" in features.columns:
                    lr5 = features["log_return_5d"].iloc[-1]
                    bear_weakening = not pd.isna(lr5) and lr5 > 0

                # XGBoost signals at low threshold
                feature_cols = [
                    "log_return_5d", "log_return_20d", "volatility_20d",
                    "credit_spread", "dollar_momentum", "regime_label",
                ]
                last_row = features[feature_cols].iloc[[-1]]

                if last_row.isna().any().any():
                    nan_cols = [c for c in feature_cols if last_row[c].isna().any()]
                    logger.warning("NaN in feature row (columns: %s), holding", nan_cols)
                else:
                    proba = model.predict_proba(last_row)[0]
                    pred_class = int(np.argmax(proba))
                    proba_dict = {
                        "DOWN": float(proba[idx_down]),
                        "FLAT": float(proba[class_to_idx[1]]),
                        "UP": float(proba[idx_up]),
                    }

                    # Low threshold for Choppy/Bear-weakening regimes
                    if pred_class == idx_up and proba[idx_up] > XGB_THRESHOLD_LOW:
                        xgb_low = Signal.BUY
                    elif pred_class == idx_down and proba[idx_down] > XGB_THRESHOLD_LOW:
                        xgb_low = Signal.SELL
                    else:
                        xgb_low = Signal.HOLD

                    # Route composite signal
                    composite_sig, source = composite_signal_hybrid(
                        today_regime, signal, xgb_low,
                        bear_weakening=bear_weakening,
                    )

                    # Stale cash override: re-enter BULL if in cash too long
                    recent_shadow = get_shadow_signals(conn)
                    days_in_cash = 0
                    for s in reversed(recent_shadow):
                        if s["ml_signal"] == "SELL" or s["ml_signal"] == "HOLD":
                            days_in_cash += 1
                        else:
                            break
                    if composite_sig != Signal.BUY and stale_cash_override(
                        today_regime, sma_50, sma_200, days_in_cash, STALE_CASH_DAYS,
                    ):
                        composite_sig = Signal.BUY
                        source = "STALE_CASH"
                        logger.info(
                            "Stale cash override: %d days in cash, regime=%s, SMA50=%.2f > SMA200=%.2f",
                            days_in_cash, today_regime.value, sma_50, sma_200,
                        )

                    # Kelly sizing (for analysis/logging)
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

                    # Log shadow signal (always log for analysis)
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
                        "Composite: %s (source=%s, regime=%s, bear_days=%d, kelly=%.3f, kelly_qty=%d, hwm=%.0f, ts_stop=%s)",
                        composite_sig.value, source, today_regime.value, bear_days,
                        k_frac or 0.0, k_qty or 0,
                        trailing_stop.high_water_mark, ts_triggered or ts_in_cooldown,
                    )

                    # Apply trailing stop override
                    if ts_triggered and position_qty > 0:
                        effective_signal = Signal.SELL
                        logger.info("Trailing stop triggered, forcing SELL")
                    elif (ts_triggered or ts_in_cooldown) and composite_sig == Signal.BUY:
                        effective_signal = Signal.HOLD
                        logger.info("Trailing stop cooldown active, blocking BUY")
                    else:
                        effective_signal = composite_sig

                    # HMM blended signal (logged for analysis only)
                    hmm_cols = ["log_return_20d", "volatility_20d", "vix_close", "vix_term_structure"]
                    if hmm_detector is not None and all(c in features.columns for c in hmm_cols):
                        try:
                            hmm_row = features[hmm_cols].iloc[[-1]]
                            if not hmm_row.isna().any().any():
                                regime_probs = hmm_detector.predict_proba(hmm_row)[0]
                                hmm_dominant = hmm_detector.dominant_regime(regime_probs)

                                current_exp = position_value / account["portfolio_value"] if account["portfolio_value"] > 0 else 0.0

                                blended_exposure = composite_signal_blended(
                                    regime_probs=regime_probs,
                                    sma_signal=signal,
                                    xgb_signal=composite_sig,
                                    bear_weakening=bear_weakening,
                                    current_exposure=current_exp,
                                )
                                logger.info(
                                    "HMM blended: exposure=%.3f, dominant=%s, probs=%s",
                                    blended_exposure, hmm_dominant.value,
                                    {k: f"{v:.3f}" for k, v in regime_probs.items()},
                                )
                        except Exception:
                            logger.warning("HMM prediction failed (non-fatal)", exc_info=True)
    except Exception:
        logger.exception("Composite signal failed, holding")
        try:
            send_error_alert("Composite signal failed", traceback.format_exc()[-500:])
        except Exception:
            logger.exception("Failed to send composite failure alert")

    logger.info("Effective signal: %s", effective_signal.value)

    # Step 9: Risk evaluation + execution
    order_request = evaluate(
        signal=effective_signal,
        portfolio_value=account["portfolio_value"],
        close_price=close_price,
        current_position_qty=position_qty,
    )

    if order_request is not None:
        result = submit_order(order_request, TICKER)
        log_order(
            conn, signal_id, result.alpaca_order_id,
            result.timestamp, TICKER, order_request.action,
            order_request.quantity, result.status,
            signal_close_price=close_price,
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

    # Step 10: Log portfolio snapshot
    account = get_account()  # refresh after potential trade
    position_qty = get_position(TICKER)
    position_value = position_qty * close_price
    log_portfolio(conn, now, account["cash"], position_qty, position_value, account["portfolio_value"])

    # Step 11: LLM oracle shadow (Claude + ChatGPT head-to-head, non-fatal)
    oracle_responses = []
    try:
        recent_closes = df["close"].tail(20).astype(float).tolist()
        oracle_input = OracleInput(
            date_str=today,
            current_price=close_price,
            recent_closes=recent_closes,
            position_qty=position_qty,
            portfolio_value=account["portfolio_value"],
        )
        oracle_responses = query_all(oracle_input)
        for resp in oracle_responses:
            log_llm_signal(
                conn, now, TICKER, close_price,
                provider=resp.provider, model=resp.model,
                action=resp.action, target_exposure=resp.target_exposure,
                confidence=resp.confidence,
                regime_assessment=resp.regime_assessment,
                key_drivers=resp.key_drivers, reasoning=resp.reasoning,
                raw_response=resp.raw_response, error=resp.error,
            )
            if resp.error:
                logger.warning("LLM oracle %s errored: %s", resp.provider, resp.error)
            else:
                logger.info(
                    "LLM oracle %s: %s target=%.2f (%s conf, regime=%s)",
                    resp.provider, resp.action, resp.target_exposure,
                    resp.confidence, resp.regime_assessment,
                )
    except Exception:
        logger.exception("LLM oracle block failed (non-fatal)")

    # Step 12: LLM daily intelligence report (non-fatal)
    try:
        recent = get_shadow_signals(conn)[-10:]
        if recent:
            llm_report = generate_daily_report(recent[-1], recent, account)
            logger.info("LLM daily report generated")
    except Exception:
        logger.exception("LLM report generation failed (non-fatal)")
        try:
            send_error_alert("LLM report failed", traceback.format_exc()[-500:])
        except Exception:
            logger.exception("Failed to send LLM failure alert")

    # Send daily summary (with LLM report if available, plain otherwise)
    send_daily_summary(
        portfolio_value=account["portfolio_value"],
        cash=account["cash"],
        position_qty=position_qty,
        signal=signal.value,
        sma_50=sma_50,
        sma_200=sma_200,
        oracle_responses=oracle_responses,
        llm_report=llm_report,
    )

    logger.info("Pipeline complete")


def _deadline_handler(signum, frame):
    raise TimeoutError(
        f"Pipeline exceeded wall-clock deadline of {_PIPELINE_DEADLINE_SECONDS}s"
    )


def _wait_for_network(host: str = _NETWORK_READY_HOST) -> None:
    """Block until DNS resolves for host; handles launchd-after-wake DNS lag."""
    last_error = None
    for i, delay in enumerate(_NETWORK_READY_BACKOFFS):
        if delay > 0:
            logger.info(
                "Network not ready; sleeping %ds before retry %d/%d",
                delay, i + 1, len(_NETWORK_READY_BACKOFFS),
            )
            time.sleep(delay)
        try:
            socket.gethostbyname(host)
            if i > 0:
                logger.info("Network ready after %d retries", i)
            return
        except socket.gaierror as exc:
            last_error = exc
    raise RuntimeError(
        f"DNS resolution failed for {host} after "
        f"{len(_NETWORK_READY_BACKOFFS)} attempts: {last_error}"
    )


def main() -> None:
    setup_logging()
    socket.setdefaulttimeout(_SOCKET_TIMEOUT_SECONDS)
    signal.signal(signal.SIGALRM, _deadline_handler)
    signal.alarm(_PIPELINE_DEADLINE_SECONDS)
    try:
        _wait_for_network()
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
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    main()
