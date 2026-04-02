from unittest.mock import patch, MagicMock

from schroeder_trader.agents.daily_report import generate_daily_report, _build_prompt


def _make_signal(
    close=650.0, regime="BEAR", bear_day_count=14, ml_signal="SELL",
    signal_source="FLAT", sma_signal="HOLD",
    predicted_proba='{"DOWN": 0.25, "FLAT": 0.17, "UP": 0.58}',
    kelly_fraction=0.148, kelly_qty=22,
    high_water_mark=100000.0, trailing_stop_triggered=0,
    timestamp="2026-04-02T20:30:00+00:00",
):
    return {
        "timestamp": timestamp,
        "close_price": close,
        "regime": regime,
        "bear_day_count": bear_day_count,
        "ml_signal": ml_signal,
        "signal_source": signal_source,
        "sma_signal": sma_signal,
        "predicted_proba": predicted_proba,
        "kelly_fraction": kelly_fraction,
        "kelly_qty": kelly_qty,
        "high_water_mark": high_water_mark,
        "trailing_stop_triggered": trailing_stop_triggered,
    }


def test_build_prompt_includes_today_data():
    today = _make_signal()
    recent = [_make_signal(timestamp=f"2026-04-0{i}T20:30:00+00:00") for i in range(1, 3)]
    portfolio = {"portfolio_value": 100000.0, "cash": 100000.0}
    prompt = _build_prompt(today, recent, portfolio)
    assert "BEAR" in prompt
    assert "SELL" in prompt
    assert "650.0" in prompt or "650.00" in prompt
    assert "100,000" in prompt


def test_build_prompt_includes_recent_history():
    today = _make_signal()
    recent = [
        _make_signal(timestamp="2026-04-01T20:30:00+00:00", close=655.0),
        _make_signal(timestamp="2026-04-02T20:30:00+00:00", close=650.0),
    ]
    portfolio = {"portfolio_value": 100000.0, "cash": 100000.0}
    prompt = _build_prompt(today, recent, portfolio)
    assert "655.0" in prompt or "655.00" in prompt


def test_build_prompt_handles_missing_fields():
    today = _make_signal(
        predicted_proba=None, kelly_fraction=None, kelly_qty=None,
        high_water_mark=None, trailing_stop_triggered=None, bear_day_count=None,
    )
    portfolio = {"portfolio_value": 100000.0, "cash": 100000.0}
    prompt = _build_prompt(today, [], portfolio)
    assert "SELL" in prompt


@patch("schroeder_trader.agents.daily_report.anthropic.Anthropic")
def test_generate_daily_report_returns_string(mock_anthropic_cls):
    mock_client = MagicMock()
    mock_anthropic_cls.return_value = mock_client
    mock_client.messages.create.return_value = MagicMock(
        content=[MagicMock(text="The system recommends staying flat today.")]
    )

    today = _make_signal()
    recent = [_make_signal()]
    portfolio = {"portfolio_value": 100000.0, "cash": 100000.0}
    result = generate_daily_report(today, recent, portfolio)

    assert isinstance(result, str)
    assert "staying flat" in result
    mock_client.messages.create.assert_called_once()


@patch("schroeder_trader.agents.daily_report.anthropic.Anthropic")
def test_generate_daily_report_uses_configured_model(mock_anthropic_cls):
    mock_client = MagicMock()
    mock_anthropic_cls.return_value = mock_client
    mock_client.messages.create.return_value = MagicMock(
        content=[MagicMock(text="Report text.")]
    )

    today = _make_signal()
    result = generate_daily_report(today, [], {"portfolio_value": 100000.0, "cash": 100000.0})

    call_kwargs = mock_client.messages.create.call_args
    assert "claude-haiku" in call_kwargs.kwargs.get("model", "") or "claude-haiku" in str(call_kwargs)
