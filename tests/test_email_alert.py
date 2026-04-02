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
def test_send_daily_summary_with_llm_report(mock_smtp_cls):
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
        llm_report="The system recommends staying flat in BEAR regime day 14.",
    )
    mock_smtp.send_message.assert_called_once()
    sent_msg = mock_smtp.send_message.call_args[0][0]
    body = sent_msg.get_content()
    assert "staying flat" in body
    assert "Raw Data" in body


@patch("schroeder_trader.alerts.email_alert.smtplib.SMTP_SSL")
def test_send_daily_summary_without_llm_report(mock_smtp_cls):
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
    mock_smtp.send_message.assert_called_once()
    sent_msg = mock_smtp.send_message.call_args[0][0]
    body = sent_msg.get_content()
    assert "Daily Summary" in body
