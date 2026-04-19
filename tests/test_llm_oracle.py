from unittest.mock import MagicMock, patch

import pytest

from schroeder_trader.agents.llm_oracle import (
    OracleInput,
    _build_user_prompt,
    _parse_json_response,
    query_all,
    query_claude,
    query_openai,
)


@pytest.fixture
def sample_input():
    return OracleInput(
        date_str="2026-04-20",
        current_price=710.14,
        recent_closes=[659.22, 676.01, 679.91, 679.46, 686.10, 694.46, 699.94, 701.66, 710.14] * 3,
        position_qty=141,
        portfolio_value=102_095.0,
    )


def test_user_prompt_includes_all_fields(sample_input):
    prompt = _build_user_prompt(sample_input)
    assert "2026-04-20" in prompt
    assert "$710.14" in prompt
    assert "long 141 shares" in prompt
    assert "$659.22" in prompt
    assert "$102,095" in prompt
    assert "CURRENT EXPOSURE" in prompt


def test_user_prompt_flat_position():
    inp = OracleInput(
        date_str="2026-04-20", current_price=700.0, recent_closes=[700.0],
        position_qty=0, portfolio_value=100_000.0,
    )
    prompt = _build_user_prompt(inp)
    assert "flat" in prompt
    assert "0% SPY exposure" in prompt


def test_parse_json_plain():
    text = '{"action": "BUY", "target_exposure": 0.98, "confidence": "HIGH"}'
    data = _parse_json_response(text)
    assert data["action"] == "BUY"
    assert data["target_exposure"] == 0.98


def test_parse_json_with_markdown_fences():
    text = '```json\n{"action": "SELL", "confidence": "LOW"}\n```'
    data = _parse_json_response(text)
    assert data["action"] == "SELL"


def test_parse_json_with_prose_prefix():
    text = 'Here is my analysis:\n{"action": "HOLD", "confidence": "MEDIUM"}'
    data = _parse_json_response(text)
    assert data["action"] == "HOLD"


def test_parse_json_no_object_raises():
    with pytest.raises(ValueError, match="No JSON object"):
        _parse_json_response("I cannot make a decision.")


@patch("schroeder_trader.agents.llm_oracle.anthropic.Anthropic")
def test_query_claude_parses_response(mock_anthropic_cls, sample_input):
    client = MagicMock()
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = (
        '{"action": "BUY", "target_exposure": 0.95, "confidence": "HIGH", '
        '"regime_assessment": "BULL", "key_drivers": ["momentum", "breadth"], '
        '"reasoning": "uptrend intact"}'
    )
    response = MagicMock()
    response.content = [text_block]
    client.messages.create.return_value = response
    mock_anthropic_cls.return_value = client

    result = query_claude(sample_input)
    assert result.provider == "claude"
    assert result.action == "BUY"
    assert result.target_exposure == 0.95
    assert result.confidence == "HIGH"
    assert result.regime_assessment == "BULL"
    assert result.key_drivers == ["momentum", "breadth"]
    assert result.error is None


@patch("schroeder_trader.agents.llm_oracle.anthropic.Anthropic")
def test_query_claude_clamps_target_exposure(mock_anthropic_cls, sample_input):
    client = MagicMock()
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = '{"action": "BUY", "target_exposure": 1.5, "confidence": "HIGH"}'
    response = MagicMock()
    response.content = [text_block]
    client.messages.create.return_value = response
    mock_anthropic_cls.return_value = client

    result = query_claude(sample_input)
    assert result.target_exposure == 1.0  # clamped


@patch("schroeder_trader.agents.llm_oracle.anthropic.Anthropic")
def test_query_claude_handles_missing_target_exposure(mock_anthropic_cls, sample_input):
    client = MagicMock()
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = '{"action": "HOLD", "confidence": "LOW"}'
    response = MagicMock()
    response.content = [text_block]
    client.messages.create.return_value = response
    mock_anthropic_cls.return_value = client

    result = query_claude(sample_input)
    assert result.target_exposure == 0.0  # default


@patch("schroeder_trader.agents.llm_oracle.anthropic.Anthropic")
def test_query_claude_filters_tool_use_blocks(mock_anthropic_cls, sample_input):
    client = MagicMock()
    tool_block = MagicMock()
    tool_block.type = "server_tool_use"
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = (
        '{"action": "HOLD", "target_exposure": 0.5, "confidence": "LOW", '
        '"regime_assessment": "CHOPPY"}'
    )
    response = MagicMock()
    response.content = [tool_block, text_block]
    client.messages.create.return_value = response
    mock_anthropic_cls.return_value = client

    result = query_claude(sample_input)
    assert result.action == "HOLD"
    assert result.target_exposure == 0.5


@patch("schroeder_trader.agents.llm_oracle.OpenAI")
def test_query_openai_parses_response(mock_openai_cls, sample_input):
    client = MagicMock()
    response = MagicMock()
    response.output_text = (
        '{"action": "SELL", "target_exposure": 0.0, "confidence": "MEDIUM", '
        '"regime_assessment": "BEAR", "key_drivers": ["breadth"], "reasoning": "weakening"}'
    )
    client.responses.create.return_value = response
    mock_openai_cls.return_value = client

    result = query_openai(sample_input)
    assert result.provider == "openai"
    assert result.action == "SELL"
    assert result.target_exposure == 0.0
    assert result.confidence == "MEDIUM"
    assert result.regime_assessment == "BEAR"


@patch("schroeder_trader.agents.llm_oracle.query_openai")
@patch("schroeder_trader.agents.llm_oracle.query_claude")
def test_query_all_returns_both_even_on_failure(mock_claude, mock_openai, sample_input):
    from schroeder_trader.agents.llm_oracle import OracleResponse
    mock_claude.return_value = OracleResponse(
        provider="claude", model="m", action="BUY", target_exposure=0.98,
        confidence="HIGH", regime_assessment="BULL",
        key_drivers=[], reasoning="", raw_response="",
    )
    mock_openai.side_effect = RuntimeError("boom")

    results = query_all(sample_input)
    assert len(results) == 2
    assert results[0].provider == "claude"
    assert results[0].error is None
    assert results[1].provider == "openai"
    assert "RuntimeError" in results[1].error
    assert results[1].action == "HOLD"  # safe default
