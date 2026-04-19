import json
import logging
from dataclasses import dataclass, field
from datetime import datetime

import anthropic
from openai import OpenAI

from schroeder_trader.config import (
    ANTHROPIC_API_KEY,
    CLAUDE_ORACLE_MODEL,
    OPENAI_API_KEY,
    OPENAI_ORACLE_MODEL,
)

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a trading advisor deciding today's SPY position: BUY, SELL, or HOLD, and how much to hold.
Use everything you know about markets, regimes, macro conditions, seasonality, and technical analysis.
Use web search to gather current news, macro data, earnings calendar, Fed actions, or geopolitical events that may be relevant to today's decision.
The user will provide today's date, current SPY price, recent price history, portfolio value, and current position.

Respond ONLY with a single JSON object matching this schema (no prose outside the JSON):
{
  "action": "BUY" | "SELL" | "HOLD",
  "target_exposure": 0.0 to 1.0,
  "confidence": "LOW" | "MEDIUM" | "HIGH",
  "regime_assessment": "BULL" | "BEAR" | "CHOPPY",
  "key_drivers": ["driver1", "driver2", "driver3"],
  "reasoning": "short explanation, <=400 chars"
}

Rules:
- "action" is your recommendation for the trade direction from today's close.
- "target_exposure" is the fraction of total portfolio value you want in SPY AFTER this decision (0.0 = all cash, 1.0 = fully invested). This is what drives your hypothetical P&L — be deliberate.
- "action" must be consistent with target_exposure relative to current exposure: BUY if target > current, SELL if target < current, HOLD if target ≈ current.
- "confidence" reflects how strongly you hold this view.
- "regime_assessment" is your independent read of current market regime.
- "key_drivers" lists up to 3 short phrases citing the most important factors.
- If data is ambiguous, prefer HOLD at current exposure with LOW confidence."""


@dataclass
class OracleResponse:
    provider: str
    model: str
    action: str
    target_exposure: float
    confidence: str
    regime_assessment: str
    key_drivers: list[str]
    reasoning: str
    raw_response: str
    error: str | None = None


@dataclass
class OracleInput:
    date_str: str
    current_price: float
    recent_closes: list[float]
    position_qty: int
    portfolio_value: float


def _build_user_prompt(inp: OracleInput) -> str:
    closes_str = ", ".join(f"${c:.2f}" for c in inp.recent_closes)
    position_value = inp.position_qty * inp.current_price
    current_exposure = position_value / inp.portfolio_value if inp.portfolio_value > 0 else 0.0
    if inp.position_qty > 0:
        position_line = (
            f"long {inp.position_qty} shares "
            f"(${position_value:,.0f} = {current_exposure:.1%} of portfolio)"
        )
    else:
        position_line = "flat (0% SPY exposure)"
    return f"""DATE: {inp.date_str}
CURRENT SPY PRICE: ${inp.current_price:.2f}
RECENT CLOSES (20 sessions, oldest → newest): [{closes_str}]
PORTFOLIO VALUE: ${inp.portfolio_value:,.0f}
CURRENT POSITION: {position_line}
CURRENT EXPOSURE: {current_exposure:.2f}

Decide the right action AND target_exposure for the next close-to-close hold.
Search the web for relevant news, macro data, or events as needed.
Respond with the JSON object only."""


def _parse_json_response(text: str) -> dict:
    # Claude/GPT may wrap in markdown fences or add whitespace; locate outermost JSON object.
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON object found in response (first 200 chars): {text[:200]}")
    return json.loads(text[start : end + 1])


def _coerce_response(provider: str, model: str, raw: str) -> OracleResponse:
    data = _parse_json_response(raw)
    try:
        target = float(data.get("target_exposure", 0.0))
    except (TypeError, ValueError):
        target = 0.0
    target = max(0.0, min(1.0, target))
    return OracleResponse(
        provider=provider,
        model=model,
        action=str(data.get("action", "HOLD")).upper(),
        target_exposure=target,
        confidence=str(data.get("confidence", "LOW")).upper(),
        regime_assessment=str(data.get("regime_assessment", "CHOPPY")).upper(),
        key_drivers=list(data.get("key_drivers", []))[:3],
        reasoning=str(data.get("reasoning", ""))[:500],
        raw_response=raw,
    )


def query_claude(inp: OracleInput) -> OracleResponse:
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY, timeout=120.0)
    response = client.messages.create(
        model=CLAUDE_ORACLE_MODEL,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _build_user_prompt(inp)}],
        thinking={"type": "adaptive"},
        tools=[{"type": "web_search_20260209", "name": "web_search"}],
    )
    text_parts = [b.text for b in response.content if getattr(b, "type", None) == "text"]
    raw = "\n".join(text_parts).strip()
    return _coerce_response("claude", CLAUDE_ORACLE_MODEL, raw)


def query_openai(inp: OracleInput) -> OracleResponse:
    client = OpenAI(api_key=OPENAI_API_KEY, timeout=120.0)
    response = client.responses.create(
        model=OPENAI_ORACLE_MODEL,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(inp)},
        ],
        tools=[{"type": "web_search"}],
    )
    raw = (response.output_text or "").strip()
    return _coerce_response("openai", OPENAI_ORACLE_MODEL, raw)


def query_all(inp: OracleInput) -> list[OracleResponse]:
    """Run both oracles. Per-provider failures are captured as error rows, never raised."""
    results: list[OracleResponse] = []
    for provider, fn, model in (
        ("claude", query_claude, CLAUDE_ORACLE_MODEL),
        ("openai", query_openai, OPENAI_ORACLE_MODEL),
    ):
        try:
            results.append(fn(inp))
        except Exception as e:
            logger.exception("LLM oracle %s failed", provider)
            results.append(OracleResponse(
                provider=provider, model=model,
                action="HOLD", target_exposure=0.0,
                confidence="LOW", regime_assessment="CHOPPY",
                key_drivers=[], reasoning="",
                raw_response="", error=f"{type(e).__name__}: {e}",
            ))
    return results
