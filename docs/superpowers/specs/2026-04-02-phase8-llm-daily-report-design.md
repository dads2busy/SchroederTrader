# Phase 8: LLM Daily Intelligence Report — Design Spec

## Overview

Add a Claude-powered daily intelligence report to the pipeline. After all trading and shadow signal steps complete, the system queries the last 10 shadow signals from the DB, sends them to Claude Haiku with today's data, and receives a 3-5 sentence narrative briefing. This replaces the current bare-bones daily email with a narrative + raw data format.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| LLM role | Narrative report (not trading decisions) | LLM explains what the system did, not what it should do |
| Content | Signal summary + recent context | Comparison to recent history makes the report useful vs just reading numbers |
| Email format | LLM narrative + raw data footer | Quick read at top, verifiable data at bottom |
| Pipeline integration | Step 11 (after Step 10 shadow signals) | Has access to all data; non-fatal on failure |
| Model | Claude Haiku (claude-haiku-4-5-20251001) | Cheapest, fast, sufficient for structured-to-narrative |
| API | Anthropic Python SDK | Native Python, matches project ecosystem |

## New Module: `src/schroeder_trader/agents/daily_report.py`

### `generate_daily_report()`

```python
def generate_daily_report(
    today_signal: dict,
    recent_signals: list[dict],
    portfolio: dict,
) -> str:
    """Generate a natural-language daily briefing using Claude.

    Args:
        today_signal: Today's shadow signal row (dict from DB).
        recent_signals: Last 10 shadow signals for context.
        portfolio: Current portfolio state (cash, position_qty, total_value).

    Returns:
        Narrative string (3-5 sentences).
    """
```

### Prompt Template

The prompt provides structured data and asks Claude to summarize it:

```
You are a trading system analyst for SchroederTrader, an automated SPY trading system.
Summarize today's signals in a concise 3-5 sentence briefing.

Today's data:
- Date: {date}
- SPY Close: ${close_price}
- Regime: {regime} (bear day {bear_day_count})
- Composite Signal: {ml_signal} (source: {signal_source})
- XGB Prediction: {predicted_proba}
- Kelly Sizing: {kelly_fraction:.1%} → {kelly_qty} shares (${kelly_value:,.0f})
- Trailing Stop: HWM ${high_water_mark:,.0f}, triggered: {trailing_stop_triggered}
- SMA Signal: {sma_signal}
- Portfolio: ${portfolio_value:,.0f}

Recent history (last {n} trading days):
{recent_signals_table}

Instructions:
- Explain what the system recommends today and why
- Note any changes from recent days (regime shifts, signal changes, confidence moves)
- Keep it factual — describe what the system is doing, not what the market will do
- Do not give investment advice
```

### Error Handling

If the Claude API call fails (network, auth, rate limit), log the error and fall back to the current email format without the LLM narrative. The pipeline never blocks on LLM failure.

## Config Changes

Add to `.env`:
```
ANTHROPIC_API_KEY=sk-ant-...
```

Add to `src/schroeder_trader/config.py`:
```python
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
LLM_MODEL = "claude-haiku-4-5-20251001"
```

## Dependencies

Add `anthropic` to project dependencies:
```
uv add anthropic
```

## Pipeline Integration (Step 11)

Add after Step 10 in `_run_pipeline_inner()`:

```python
    # Step 11: LLM daily intelligence report (non-fatal)
    try:
        recent = get_shadow_signals(conn)[-10:]
        today_shadow = recent[-1] if recent else None
        if today_shadow:
            report = generate_daily_report(today_shadow, recent, account)
            # Send enhanced email with narrative + raw data
    except Exception:
        logger.exception("LLM report generation failed (non-fatal)")
```

## Email Format Change

### Current email (send_daily_summary):
```
Subject: [SchroederTrader] Daily run complete - Portfolio: $100,000
```
(Minimal info, just portfolio value)

### New email format:
```
Subject: [SchroederTrader] Daily Report — 2026-04-02

[3-5 sentence LLM narrative]

---
Raw Data:
SPY: $653.78 | Regime: BEAR (day 14) | Signal: SELL (FLAT)
XGB: UP 57.8% | Kelly: 14.8% (22 shares, $14,450)
SMA: HOLD (50d: 683.89, 200d: 660.36)
Trailing Stop: HWM $100,000 | Triggered: No
Portfolio: $100,000 cash | 0 shares
```

Update `send_daily_summary()` in `alerts/email_alert.py` to accept an optional `llm_report` parameter. When present, use the enhanced format. When absent (LLM failure), fall back to current format.

## What This Does NOT Include

- No trading decisions from the LLM
- No market news or headlines (avoids hallucination)
- No interactive/conversational mode (future extension)
- No long-form analysis (3-5 sentences only)
- No LLM calls outside the daily pipeline
