import json
import logging
from dataclasses import dataclass
from typing import Literal

import anthropic
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

from schroeder_trader.config import (
    ANTHROPIC_API_KEY,
    CLAUDE_ORACLE_MODEL,
    OPENAI_API_KEY,
    OPENAI_ORACLE_MODEL,
)

logger = logging.getLogger(__name__)


# --- Strict output schema -----------------------------------------------------

class OracleOutput(BaseModel):
    """Strict schema for what we ask the LLMs to produce.

    Each provider's native structured-output mode is given this schema so we
    don't have to parse JSON out of free text or clamp values manually.

    Note: we intentionally do NOT constrain string lengths via Pydantic.
    Anthropic only treats string maxLength as advisory, so a strict cap there
    causes ValidationError on overflows. We truncate downstream in
    _from_parsed instead, which is reliable across providers.
    """
    action: Literal["BUY", "SELL", "HOLD"]
    target_exposure: float = Field(ge=0.0, le=1.0)
    confidence: Literal["LOW", "MEDIUM", "HIGH"]
    regime_assessment: Literal["BULL", "BEAR", "CHOPPY"]
    key_drivers: list[str]
    reasoning: str


SYSTEM_PROMPT = """You are a trading advisor deciding today's SPY position: BUY, SELL, or HOLD, and how much to hold.
Use everything you know about markets, regimes, macro conditions, seasonality, and technical analysis.
Use web search to gather current news, macro data, earnings calendar, Fed actions, or geopolitical events that may be relevant to today's decision.
The user will provide today's date, current SPY price, recent price history, portfolio value, and current position.

Field semantics:
- action is your recommendation for the trade direction from today's close.
- target_exposure is the fraction of total portfolio value you want in SPY AFTER this decision (0.0 = all cash, 1.0 = fully invested). This drives your hypothetical P&L — be deliberate.
- action must be consistent with target_exposure relative to current exposure: BUY if target > current, SELL if target < current, HOLD if target ≈ current.
- confidence reflects how strongly you hold this view.
- regime_assessment is your independent read of current market regime.
- key_drivers lists up to 3 short phrases citing the most important factors.
- reasoning is a short explanation.
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
Search the web for relevant news, macro data, or events as needed."""


# --- Free-text JSON parsing (fallback path) -----------------------------------

def _parse_json_response(text: str) -> dict:
    """Locate the outermost JSON object in free-text content."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON object found in response (first 200 chars): {text[:200]}")
    return json.loads(text[start : end + 1])


def _coerce_from_text(provider: str, model: str, raw: str) -> OracleResponse:
    """Used when the provider's structured-output path is unavailable."""
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


def _from_parsed(provider: str, model: str, parsed: OracleOutput, raw: str) -> OracleResponse:
    """Build an OracleResponse from a validated Pydantic OracleOutput.

    Truncate variable-length string fields here so we never bloat the email
    or CSV regardless of what the model returned.
    """
    return OracleResponse(
        provider=provider,
        model=model,
        action=parsed.action,
        target_exposure=parsed.target_exposure,
        confidence=parsed.confidence,
        regime_assessment=parsed.regime_assessment,
        key_drivers=list(parsed.key_drivers)[:3],
        reasoning=parsed.reasoning[:500],
        raw_response=raw,
    )


def _content_text(content_blocks) -> str:
    """Concatenate all 'text' content blocks; skip tool_use / server_tool_use blocks."""
    return "\n".join(
        b.text for b in content_blocks if getattr(b, "type", None) == "text"
    ).strip()


# --- Provider calls -----------------------------------------------------------

def query_claude(inp: OracleInput) -> OracleResponse:
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY, timeout=120.0)
    user_prompt = _build_user_prompt(inp)
    common_kwargs = dict(
        model=CLAUDE_ORACLE_MODEL,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
        thinking={"type": "adaptive"},
        tools=[{"type": "web_search_20260209", "name": "web_search"}],
    )
    try:
        response = client.messages.parse(output_format=OracleOutput, **common_kwargs)
        raw = _content_text(response.content)
        return _from_parsed("claude", CLAUDE_ORACLE_MODEL, response.parsed_output, raw)
    except (TypeError, AttributeError, NotImplementedError, ValidationError) as exc:
        # Either the SDK doesn't expose messages.parse with this combo, or the
        # model's response failed schema validation — fall back to free-text.
        logger.warning("Claude structured-output failed, falling back to free-text: %s", exc)
        response = client.messages.create(**common_kwargs)
        raw = _content_text(response.content)
        return _coerce_from_text("claude", CLAUDE_ORACLE_MODEL, raw)


def query_openai(inp: OracleInput) -> OracleResponse:
    client = OpenAI(api_key=OPENAI_API_KEY, timeout=120.0)
    user_prompt = _build_user_prompt(inp)
    common_kwargs = dict(
        model=OPENAI_ORACLE_MODEL,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        tools=[{"type": "web_search"}],
    )
    try:
        response = client.responses.parse(text_format=OracleOutput, **common_kwargs)
        raw = (response.output_text or "").strip()
        return _from_parsed("openai", OPENAI_ORACLE_MODEL, response.output_parsed, raw)
    except (TypeError, AttributeError, NotImplementedError, ValidationError) as exc:
        logger.warning("OpenAI structured-output failed, falling back to free-text: %s", exc)
        response = client.responses.create(**common_kwargs)
        raw = (response.output_text or "").strip()
        return _coerce_from_text("openai", OPENAI_ORACLE_MODEL, raw)


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
