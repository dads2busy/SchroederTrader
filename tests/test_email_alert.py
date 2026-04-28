from unittest.mock import patch, MagicMock

from schroeder_trader.alerts.email_alert import send_trade_alert, send_error_alert, send_daily_summary


@patch("schroeder_trader.alerts.email_alert.smtplib.SMTP_SSL")
def test_send_trade_alert(mock_smtp_cls):
    mock_smtp = MagicMock()
    mock_smtp_cls.return_value.__enter__ = MagicMock(return_value=mock_smtp)
    mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)

    send_trade_alert(
        action="BUY",
        ticker="SPY",
        quantity=45,
        portfolio_value=28539.50,
        cash=5000.00,
        sma_50=525.0,
        sma_200=518.0,
    )
    mock_smtp.send_message.assert_called_once()


@patch("schroeder_trader.alerts.email_alert.smtplib.SMTP_SSL")
def test_send_error_alert(mock_smtp_cls):
    mock_smtp = MagicMock()
    mock_smtp_cls.return_value.__enter__ = MagicMock(return_value=mock_smtp)
    mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)

    send_error_alert("Data fetch failed", "ConnectionError: timeout")
    mock_smtp.send_message.assert_called_once()


@patch("schroeder_trader.alerts.email_alert.smtplib.SMTP_SSL")
def test_send_daily_summary(mock_smtp_cls):
    mock_smtp = MagicMock()
    mock_smtp_cls.return_value.__enter__ = MagicMock(return_value=mock_smtp)
    mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)

    send_daily_summary(
        portfolio_value=28539.50,
        cash=5000.00,
        position_qty=45,
        signal="HOLD",
        sma_50=525.0,
        sma_200=518.0,
    )
    mock_smtp.send_message.assert_called_once()


@patch("schroeder_trader.alerts.email_alert.smtplib.SMTP_SSL")
def test_send_fill_alert(mock_smtp_cls):
    mock_smtp = MagicMock()
    mock_smtp_cls.return_value.__enter__ = MagicMock(return_value=mock_smtp)
    mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)

    from schroeder_trader.alerts.email_alert import send_fill_alert
    send_fill_alert(
        action="BUY",
        ticker="SPY",
        quantity=45,
        fill_price=523.15,
    )
    mock_smtp.send_message.assert_called_once()


@patch("schroeder_trader.alerts.email_alert.smtplib.SMTP_SSL")
def test_send_trade_alert_smtp_failure_does_not_raise(mock_smtp_cls):
    mock_smtp_cls.side_effect = Exception("SMTP connection failed")
    # Should not raise — just log the error
    send_trade_alert(
        action="BUY",
        ticker="SPY",
        quantity=45,
        portfolio_value=28539.50,
        cash=5000.00,
        sma_50=525.0,
        sma_200=518.0,
    )


@patch("schroeder_trader.alerts.email_alert.smtplib.SMTP_SSL")
def test_send_daily_summary_uses_provided_body(mock_smtp_cls):
    mock_smtp = MagicMock()
    mock_smtp_cls.return_value.__enter__ = MagicMock(return_value=mock_smtp)
    mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)

    custom_body = "SchroederTrader Daily Report — 2026-04-28\nTODAY\n  SPY: $700"
    send_daily_summary(
        portfolio_value=100000.0,
        cash=100000.0,
        position_qty=0,
        signal="HOLD",
        sma_50=683.89,
        sma_200=660.36,
        email_body=custom_body,
    )
    body = mock_smtp.send_message.call_args[0][0].get_content()
    assert custom_body in body


@patch("schroeder_trader.alerts.email_alert.smtplib.SMTP_SSL")
def test_send_daily_summary_falls_back_when_no_body(mock_smtp_cls):
    mock_smtp = MagicMock()
    mock_smtp_cls.return_value.__enter__ = MagicMock(return_value=mock_smtp)
    mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)

    send_daily_summary(
        portfolio_value=100000.0,
        cash=100000.0,
        position_qty=0,
        signal="HOLD",
        sma_50=683.89,
        sma_200=660.36,
    )
    body = mock_smtp.send_message.call_args[0][0].get_content()
    assert "fallback" in body


@patch("schroeder_trader.alerts.email_alert.smtplib.SMTP_SSL")
def test_send_daily_summary_includes_oracle_block(mock_smtp_cls):
    from schroeder_trader.agents.llm_oracle import OracleResponse

    mock_smtp = MagicMock()
    mock_smtp_cls.return_value.__enter__ = MagicMock(return_value=mock_smtp)
    mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)

    oracles = [
        OracleResponse(
            provider="claude", model="claude-opus-4-7",
            action="BUY", target_exposure=0.95, confidence="HIGH",
            regime_assessment="BULL",
            key_drivers=["momentum", "earnings"], reasoning="trend intact",
            raw_response="",
        ),
        OracleResponse(
            provider="openai", model="gpt-5.4",
            action="HOLD", target_exposure=0.80, confidence="MEDIUM",
            regime_assessment="CHOPPY",
            key_drivers=[], reasoning="",
            raw_response="", error="TimeoutError",
        ),
    ]

    # When no email_body is supplied, the fallback embeds the oracle block
    send_daily_summary(
        portfolio_value=100000.0,
        cash=1965.0,
        position_qty=141,
        signal="HOLD",
        sma_50=683.0,
        sma_200=660.0,
        oracle_responses=oracles,
    )
    body = mock_smtp.send_message.call_args[0][0].get_content()
    assert "LLM Oracle Comparison" in body
    assert "CLAUDE (claude-opus-4-7): BUY" in body
    assert "target=0.95" in body
    assert "momentum" in body
    assert "OPENAI (gpt-5.4): ERROR" in body


@patch("schroeder_trader.alerts.email_alert.smtplib.SMTP_SSL")
def test_send_email_passes_explicit_timeout(mock_smtp_cls):
    mock_smtp = MagicMock()
    mock_smtp_cls.return_value.__enter__ = MagicMock(return_value=mock_smtp)
    mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)

    send_trade_alert(
        action="BUY", ticker="SPY", quantity=1,
        portfolio_value=100.0, cash=0.0, sma_50=0.0, sma_200=0.0,
    )

    # SMTP_SSL(host, port, timeout=30) — timeout must be present
    kwargs = mock_smtp_cls.call_args.kwargs
    assert "timeout" in kwargs
    assert kwargs["timeout"] == 30


@patch("schroeder_trader.alerts.email_alert.smtplib.SMTP_SSL")
def test_send_daily_summary_no_oracle_block_when_empty(mock_smtp_cls):
    mock_smtp = MagicMock()
    mock_smtp_cls.return_value.__enter__ = MagicMock(return_value=mock_smtp)
    mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)

    send_daily_summary(
        portfolio_value=100000.0,
        cash=100000.0,
        position_qty=0,
        signal="HOLD",
        sma_50=683.0,
        sma_200=660.0,
        email_body="some custom body without an oracle section",
    )
    body = mock_smtp.send_message.call_args[0][0].get_content()
    assert "LLM Oracle Comparison" not in body
