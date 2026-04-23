import json
import logging

import anthropic

from schroeder_trader.config import ANTHROPIC_API_KEY, LLM_MODEL

logger = logging.getLogger(__name__)


def _build_prompt(
    today_signal: dict,
    recent_signals: list[dict],
    portfolio: dict,
) -> str:
    """Build the prompt for Claude from structured signal data."""
    ts = today_signal
    date_str = ts["timestamp"][:10]
    close = ts["close_price"]
    regime = ts.get("regime", "UNKNOWN")
    bear_days = ts.get("bear_day_count") or 0
    ml_signal = ts.get("ml_signal", "N/A")
    source = ts.get("signal_source", "N/A")
    sma_signal = ts.get("sma_signal", "N/A")

    proba_str = "N/A"
    if ts.get("predicted_proba"):
        try:
            proba = json.loads(ts["predicted_proba"])
            proba_str = ", ".join(f"{k}: {v:.1%}" for k, v in proba.items())
        except (json.JSONDecodeError, TypeError):
            pass

    kelly_frac = ts.get("kelly_fraction")
    kelly_qty = ts.get("kelly_qty")
    kelly_str = f"{kelly_frac:.1%} -> {kelly_qty} shares (${kelly_qty * close:,.0f})" if kelly_frac is not None and kelly_qty is not None else "N/A"

    hwm = ts.get("high_water_mark")
    hwm_str = f"${hwm:,.0f}" if hwm is not None else "N/A"
    ts_triggered = "Yes" if ts.get("trailing_stop_triggered") else "No"

    pv = portfolio.get("portfolio_value", 0)
    cash = portfolio.get("cash", 0)

    history_lines = []
    for sig in recent_signals:
        d = sig["timestamp"][:10]
        c = sig["close_price"]
        r = sig.get("regime", "?")
        s = sig.get("ml_signal", "?")
        src = sig.get("signal_source", "?")
        history_lines.append(f"  {d}  SPY ${c:.2f}  {r:8s}  {s:4s} ({src})")

    history = "\n".join(history_lines) if history_lines else "  (no prior data)"

    return f"""You are a trading system analyst for SchroederTrader, an automated SPY trading system.
Summarize today's signals in a concise 3-5 sentence briefing.

Today's data:
- Date: {date_str}
- SPY Close: ${close:.2f}
- Regime: {regime} (bear day {bear_days})
- Composite Signal: {ml_signal} (source: {source})
- XGB Prediction: {proba_str}
- Kelly Sizing: {kelly_str}
- Trailing Stop: HWM {hwm_str}, triggered: {ts_triggered}
- SMA Signal: {sma_signal}
- Portfolio: ${pv:,.0f} ({cash:,.0f} cash)

Recent history (last {len(recent_signals)} trading days):
{history}

Instructions:
- Explain what the system recommends today and why
- Note any changes from recent days (regime shifts, signal changes, confidence moves)
- Keep it factual — describe what the system is doing, not what the market will do
- Do not give investment advice"""


def generate_daily_report(
    today_signal: dict,
    recent_signals: list[dict],
    portfolio: dict,
) -> str:
    """Generate a natural-language daily briefing using Claude."""
    prompt = _build_prompt(today_signal, recent_signals, portfolio)

    logger.info("Calling Claude for daily report (model=%s)", LLM_MODEL)
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY, timeout=60.0)
    response = client.messages.create(
        model=LLM_MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    logger.info("Claude daily report received (%d content blocks)", len(response.content))

    return response.content[0].text
