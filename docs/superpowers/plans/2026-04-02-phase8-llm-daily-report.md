# LLM Daily Intelligence Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Claude-powered daily intelligence report that transforms structured shadow signal data into a 3-5 sentence narrative briefing, delivered via the existing email pipeline.

**Architecture:** New `agents/daily_report.py` module calls Claude Haiku with today's shadow signal + last 10 days of history. The narrative is passed to an updated `send_daily_summary()` which formats it above a raw data footer. Integrated as Step 11 in the pipeline, wrapped in try/except for non-fatal failure.

**Tech Stack:** Python, Anthropic SDK, pytest

---

### Task 1: Add anthropic dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add anthropic to dependencies**

```bash
uv add anthropic
```

- [ ] **Step 2: Verify import works**

```bash
.venv/bin/python -c "import anthropic; print(anthropic.__version__)"
```

Expected: prints version number without error

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: add anthropic SDK dependency"
```

---

### Task 2: Config — Add LLM settings

**Files:**
- Modify: `src/schroeder_trader/config.py`

- [ ] **Step 1: Add LLM config values**

Add after the email config section at the end of `config.py`:

```python
# LLM
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
LLM_MODEL = "claude-haiku-4-5-20251001"
```

- [ ] **Step 2: Run config tests**

Run: `pytest tests/test_config.py -v`
Expected: all PASS

- [ ] **Step 3: Commit**

```bash
git add src/schroeder_trader/config.py
git commit -m "feat: add ANTHROPIC_API_KEY and LLM_MODEL config"
```

---

### Task 3: Daily Report Module — Tests

**Files:**
- Create: `tests/test_daily_report.py`

- [ ] **Step 1: Write tests**

```python
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
    assert "SELL" in prompt  # should still work with missing optional fields


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_daily_report.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'schroeder_trader.agents'`

- [ ] **Step 3: Commit**

```bash
git add tests/test_daily_report.py
git commit -m "test: add daily report tests (red)"
```

---

### Task 4: Daily Report Module — Implementation

**Files:**
- Create: `src/schroeder_trader/agents/__init__.py`
- Create: `src/schroeder_trader/agents/daily_report.py`

- [ ] **Step 1: Create agents package**

Create empty `__init__.py`:

```bash
mkdir -p src/schroeder_trader/agents
touch src/schroeder_trader/agents/__init__.py
```

- [ ] **Step 2: Implement daily_report.py**

```python
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

    # Parse XGB probabilities
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

    # Build recent history table
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
    """Generate a natural-language daily briefing using Claude.

    Args:
        today_signal: Today's shadow signal row (dict from DB).
        recent_signals: Last 10 shadow signals for context.
        portfolio: Current portfolio state (portfolio_value, cash).

    Returns:
        Narrative string (3-5 sentences).
    """
    prompt = _build_prompt(today_signal, recent_signals, portfolio)

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    response = client.messages.create(
        model=LLM_MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )

    return response.content[0].text
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `pytest tests/test_daily_report.py -v`
Expected: all 5 tests PASS

- [ ] **Step 4: Commit**

```bash
git add src/schroeder_trader/agents/__init__.py src/schroeder_trader/agents/daily_report.py
git commit -m "feat: add LLM daily report generation with Claude Haiku"
```

---

### Task 5: Update Email — Enhanced Daily Summary

**Files:**
- Modify: `src/schroeder_trader/alerts/email_alert.py`
- Modify: `tests/test_email_alert.py`

- [ ] **Step 1: Write test for enhanced email**

Add to `tests/test_email_alert.py`:

```python
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
    # Check the email body contains the LLM report
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
    # Without LLM report, should use original format
    sent_msg = mock_smtp.send_message.call_args[0][0]
    body = sent_msg.get_content()
    assert "Daily Summary" in body
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_email_alert.py::test_send_daily_summary_with_llm_report -v`
Expected: FAIL (unexpected keyword argument `llm_report`)

- [ ] **Step 3: Update send_daily_summary**

Replace the `send_daily_summary` function in `alerts/email_alert.py`:

```python
def send_daily_summary(
    portfolio_value: float,
    cash: float,
    position_qty: int,
    signal: str,
    sma_50: float,
    sma_200: float,
    llm_report: str | None = None,
) -> None:
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
    _send_email(subject, body)
```

- [ ] **Step 4: Run all email tests**

Run: `pytest tests/test_email_alert.py -v`
Expected: all 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/schroeder_trader/alerts/email_alert.py tests/test_email_alert.py
git commit -m "feat: add LLM report support to daily summary email"
```

---

### Task 6: Pipeline Integration — Step 11

**Files:**
- Modify: `src/schroeder_trader/main.py`

- [ ] **Step 1: Add import**

Add near the other imports in `main.py`:

```python
from schroeder_trader.agents.daily_report import generate_daily_report
```

- [ ] **Step 2: Add Step 11 after Step 10**

Add after the Step 10 `except` block (after the line `logger.exception("Shadow composite prediction failed (non-fatal)")`), before `logger.info("Pipeline complete")`:

```python
    # Step 11: LLM daily intelligence report (non-fatal)
    try:
        recent = get_shadow_signals(conn)[-10:]
        if recent:
            account = get_account()
            report = generate_daily_report(recent[-1], recent, account)
            send_daily_summary(
                portfolio_value=account["portfolio_value"],
                cash=account["cash"],
                position_qty=get_position(TICKER),
                signal=signal.value,
                sma_50=sma_50,
                sma_200=sma_200,
                llm_report=report,
            )
            logger.info("LLM daily report sent")
    except Exception:
        logger.exception("LLM report generation failed (non-fatal)")
```

Note: This sends a second daily summary email with the LLM report. The original `send_daily_summary` call in Step 9 still runs (without LLM report). To avoid two emails, remove the Step 9 `send_daily_summary` call and move it into Step 11 as the fallback when LLM fails. Update Step 9 to only call `send_daily_summary` if there are no shadow signals (the Step 11 path handles it when shadow signals exist):

Replace the Step 9 block:

```python
    # Step 9: Daily summary (sent in Step 11 with LLM report if shadow signals exist)
    has_shadow = False
```

Then in Step 11, set `has_shadow = True` if shadow signals exist, and add a fallback after Step 11:

```python
    # Fallback: send plain summary if no shadow signals or LLM failed
    if not has_shadow:
        account = get_account()
        send_daily_summary(
            portfolio_value=account["portfolio_value"],
            cash=account["cash"],
            position_qty=get_position(TICKER),
            signal=signal.value,
            sma_50=sma_50,
            sma_200=sma_200,
        )
```

Actually, this is getting complex. Simpler approach: keep Step 9 as-is for the basic email (it always fires). In Step 11, send a *second* enhanced email only when shadow signals + LLM report are available. Two emails on good days, one on error days. This avoids any changes to the existing Step 9 logic.

- [ ] **Step 3: Run full test suite**

Run: `pytest -v`
Expected: all tests PASS

- [ ] **Step 4: Commit**

```bash
git add src/schroeder_trader/main.py
git commit -m "feat: integrate LLM daily report as pipeline Step 11"
```

---

### Task 7: End-to-End Validation + Push

**Files:** (no new files)

- [ ] **Step 1: Verify the ANTHROPIC_API_KEY is set**

```bash
grep ANTHROPIC_API_KEY .env
```

If not present, add it:
```bash
echo 'ANTHROPIC_API_KEY=sk-ant-...' >> .env
```

- [ ] **Step 2: Test report generation manually**

```bash
.venv/bin/python -c "
from schroeder_trader.agents.daily_report import generate_daily_report

today = {
    'timestamp': '2026-04-02T20:30:00+00:00',
    'close_price': 653.78,
    'regime': 'BEAR',
    'bear_day_count': 14,
    'ml_signal': 'SELL',
    'signal_source': 'FLAT',
    'sma_signal': 'HOLD',
    'predicted_proba': '{\"DOWN\": 0.25, \"FLAT\": 0.17, \"UP\": 0.58}',
    'kelly_fraction': 0.148,
    'kelly_qty': 22,
    'high_water_mark': 100000.0,
    'trailing_stop_triggered': 0,
}
portfolio = {'portfolio_value': 100000.0, 'cash': 100000.0}

report = generate_daily_report(today, [today], portfolio)
print(report)
"
```

Expected: 3-5 sentence narrative about the BEAR regime and SELL signal.

- [ ] **Step 3: Run full test suite**

Run: `pytest -v`
Expected: all tests PASS

- [ ] **Step 4: Push**

```bash
git push
```
