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


SYSTEM_PROMPT = """You are a trading advisor deciding today's SPY position: BUY, SELL, or HOLD.
Use everything you know about markets, regimes, macro conditions, seasonality, and technical analysis.
Use web search to gather current news, macro data, earnings calendar, Fed actions, or geopolitical events that may be relevant to today's decision.
The user will provide today's date, current SPY price, recent price history, and current position.

Respond ONLY with a single JSON object matching this schema (no prose outside the JSON):
{
  "action": "BUY" | "SELL" | "HOLD",
  "confidence": "LOW" | "MEDIUM" | "HIGH",
  "regime_assessment": "BULL" | "BEAR" | "CHOPPY",
  "key_drivers": ["driver1", "driver2", "driver3"],
  "reasoning": "short explanation, <=400 chars"
}

Rules:
- "action" is your recommendation for a close-to-close hold from today's close.
- "confidence" reflects how strongly you hold this view given the data and any context you gathered.
- "regime_assessment" is your independent read of current market regime.
- "key_drivers" lists up to 3 short phrases citing the most important factors (can reference news you found).
- If data is ambiguous, prefer HOLD with LOW confidence over a forced call."""


@dataclass
class OracleResponse:
    provider: str
    model: str
    action: str
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


def _build_user_prompt(inp: OracleInput) -> str:
    closes_str = ", ".join(f"${c:.2f}" for c in inp.recent_closes)
    position_line = (
        f"long {inp.position_qty} shares"
        if inp.position_qty > 0
        else "flat (no position)"
    )
    return f"""DATE: {inp.date_str}
CURRENT SPY PRICE: ${inp.current_price:.2f}
RECENT CLOSES (20 sessions, oldest → newest): [{closes_str}]
CURRENT POSITION: {position_line}

Decide the right action for the next close-to-close hold.
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
    return OracleResponse(
        provider=provider,
        model=model,
        action=str(data.get("action", "HOLD")).upper(),
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
                action="HOLD", confidence="LOW", regime_assessment="CHOPPY",
                key_drivers=[], reasoning="",
                raw_response="", error=f"{type(e).__name__}: {e}",
            ))
    return results
