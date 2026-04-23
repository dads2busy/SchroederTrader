import logging
import smtplib
from email.message import EmailMessage

from schroeder_trader.config import ALERT_EMAIL_FROM, ALERT_EMAIL_TO, ALERT_EMAIL_APP_PASSWORD

logger = logging.getLogger(__name__)

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465
_SMTP_TIMEOUT = 30  # Explicit timeout so a hung Gmail connection can't stall the pipeline.


def _send_email(subject: str, body: str) -> None:
    """Send an email via Gmail SMTP. Logs and swallows errors."""
    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = ALERT_EMAIL_FROM
        msg["To"] = ALERT_EMAIL_TO
        msg.set_content(body)

        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT, timeout=_SMTP_TIMEOUT) as smtp:
            smtp.login(ALERT_EMAIL_FROM, ALERT_EMAIL_APP_PASSWORD)
            smtp.send_message(msg)

        logger.info("Email sent: %s", subject)
    except Exception:
        logger.exception("Failed to send email: %s", subject)


def send_trade_alert(
    action: str,
    ticker: str,
    quantity: int,
    portfolio_value: float,
    cash: float,
    sma_50: float,
    sma_200: float,
) -> None:
    subject = f"[SchroederTrader] SUBMITTED: {action} {quantity} {ticker} (fills at next open)"
    body = (
        f"Trade Submitted\n"
        f"{'=' * 40}\n"
        f"Action: {action}\n"
        f"Ticker: {ticker}\n"
        f"Quantity: {quantity} shares\n\n"
        f"Portfolio: ${portfolio_value:,.2f}\n"
        f"Cash: ${cash:,.2f}\n\n"
        f"SMA 50: {sma_50:.2f}\n"
        f"SMA 200: {sma_200:.2f}\n"
    )
    _send_email(subject, body)


def send_fill_alert(
    action: str,
    ticker: str,
    quantity: int,
    fill_price: float,
) -> None:
    subject = f"[SchroederTrader] FILLED: {action} {quantity} {ticker} @ ${fill_price:.2f}"
    body = (
        f"Order Filled\n"
        f"{'=' * 40}\n"
        f"Action: {action}\n"
        f"Ticker: {ticker}\n"
        f"Quantity: {quantity} shares\n"
        f"Fill Price: ${fill_price:.2f}\n"
    )
    _send_email(subject, body)


def send_error_alert(error_type: str, details: str) -> None:
    subject = f"[SchroederTrader] ERROR: {error_type}"
    body = (
        f"Error Report\n"
        f"{'=' * 40}\n"
        f"Error: {error_type}\n\n"
        f"Details:\n{details}\n"
    )
    _send_email(subject, body)


def _format_oracle_block(oracle_responses: list) -> str:
    """Build a structured comparison block from a list of OracleResponse objects."""
    if not oracle_responses:
        return ""
    lines = [f"{'—' * 40}", "LLM Oracle Comparison:"]
    for resp in oracle_responses:
        label = f"{resp.provider.upper()} ({resp.model})"
        if getattr(resp, "error", None):
            lines.append(f"{label}: ERROR — {resp.error}")
            continue
        lines.append(
            f"{label}: {resp.action}  target={resp.target_exposure:.2f}  "
            f"({resp.confidence} conf, regime={resp.regime_assessment})"
        )
        if resp.key_drivers:
            lines.append(f"  drivers: {', '.join(resp.key_drivers)}")
        if resp.reasoning:
            lines.append(f"  reasoning: {resp.reasoning}")
    return "\n".join(lines) + "\n"


def send_daily_summary(
    portfolio_value: float,
    cash: float,
    position_qty: int,
    signal: str,
    sma_50: float,
    sma_200: float,
    llm_report: str | None = None,
    oracle_responses: list | None = None,
) -> None:
    oracle_block = _format_oracle_block(oracle_responses or [])
    if llm_report:
        subject = f"[SchroederTrader] Daily Report — Portfolio: ${portfolio_value:,.0f}"
        body = (
            f"{llm_report}\n\n"
            f"{'—' * 40}\n"
            f"Raw Data:\n"
            f"Signal: {signal} | Portfolio: ${portfolio_value:,.2f}\n"
            f"Cash: ${cash:,.2f} | Position: {position_qty} shares SPY\n"
            f"SMA 50: {sma_50:.2f} | SMA 200: {sma_200:.2f}\n"
        )
        if oracle_block:
            body += f"\n{oracle_block}"
    else:
        subject = f"[SchroederTrader] Daily run complete - Portfolio: ${portfolio_value:,.0f}"
        body = (
            f"Daily Summary\n"
            f"{'=' * 40}\n"
            f"Signal: {signal}\n"
            f"Portfolio Value: ${portfolio_value:,.2f}\n"
            f"Cash: ${cash:,.2f}\n"
            f"Position: {position_qty} shares SPY\n\n"
            f"SMA 50: {sma_50:.2f}\n"
            f"SMA 200: {sma_200:.2f}\n"
        )
        if oracle_block:
            body += f"\n{oracle_block}"
    _send_email(subject, body)
