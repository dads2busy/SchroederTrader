from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd

from schroeder_trader.reports.daily_email import (
    build_today_section,
    build_system_section,
    build_oracles_section,
    build_email_body,
)


def _fake_oracle(provider, action, target, conf="MEDIUM", regime="BULL", drivers=None, reasoning=""):
    o = MagicMock()
    o.provider = provider
    o.action = action
    o.target_exposure = target
    o.confidence = conf
    o.regime_assessment = regime
    o.key_drivers = drivers or []
    o.reasoning = reasoning
    o.error = None
    return o


def test_today_section_has_all_fields():
    section = build_today_section(
        date_str="2026-04-28",
        spy_close=715.16,
        spy_prev_close=713.97,
        portfolio_value=102795.0,
        portfolio_prev_value=102633.0,
        cash=1965.0,
        position_qty=141,
    )
    assert "TODAY" in section
    assert "$715.16" in section
    assert "+0.17%" in section  # SPY change
    assert "141 shares" in section
    assert "$102,795" in section


def test_system_section_with_xgb_proba():
    section = build_system_section(
        sma_signal="HOLD",
        sma_50=676.23,
        sma_200=667.76,
        composite_signal="BUY",
        composite_source="XGB",
        regime="CHOPPY",
        bear_days=0,
        xgb_proba_up=0.578,
        xgb_threshold=0.35,
        today_action="HOLD",
    )
    assert "Composite signal:  BUY" in section
    assert "source: XGB" in section
    assert "57.8%" in section
    assert "CHOPPY" in section


def test_oracles_section_renders_each_provider():
    oracles = [
        _fake_oracle("claude", "SELL", 0.85, drivers=["overbought"], reasoning="trim at highs"),
        _fake_oracle("openai", "HOLD", 0.98, conf="LOW", regime="CHOPPY"),
    ]
    section = build_oracles_section(oracles)
    assert "CLAUDE:" in section
    assert "target=0.85" in section
    assert "overbought" in section
    assert "trim at highs" in section
    assert "OPENAI:" in section
    assert "target=0.98" in section


def test_oracles_section_handles_error():
    err = MagicMock()
    err.provider = "claude"
    err.error = "TimeoutError"
    section = build_oracles_section([err])
    assert "CLAUDE: ERROR" in section
    assert "TimeoutError" in section


def test_build_email_body_full(tmp_path):
    # Synthetic SPY history covering the live-start window
    dates = pd.bdate_range("2026-04-15", periods=10).tz_localize(None)
    spy = pd.DataFrame({"close": [700.0 + i for i in range(10)]}, index=dates)

    # Synthetic portfolio.csv (system real P&L)
    pf = pd.DataFrame({
        "id": [1, 2],
        "timestamp": ["2026-04-15T20:30:00+00:00", "2026-04-28T20:30:00+00:00"],
        "cash": [1965.0, 1965.0],
        "position_qty": [141, 141],
        "position_value": [98700.0, 99918.0],
        "total_value": [100665.0, 101883.0],
    })
    pf.to_csv(tmp_path / "portfolio.csv", index=False)

    # Empty llm_shadow_signals.csv (forces sim sections to be N/A)
    pd.DataFrame(columns=[
        "id", "timestamp", "provider", "target_exposure", "error"
    ]).to_csv(tmp_path / "llm_shadow_signals.csv", index=False)

    body = build_email_body(
        date_str="2026-04-28",
        spy_close=715.16,
        spy_prev_close=713.97,
        portfolio_value=101883.0,
        portfolio_prev_value=100665.0,
        cash=1965.0,
        position_qty=141,
        sma_signal="HOLD",
        sma_50=676.0,
        sma_200=667.0,
        composite_signal="BUY",
        composite_source="XGB",
        regime="CHOPPY",
        bear_days=0,
        xgb_proba_up=0.578,
        xgb_threshold=0.35,
        today_action="HOLD",
        oracle_responses=[_fake_oracle("claude", "SELL", 0.85)],
        data_root=tmp_path,
        spy_history=spy,
        live_start_date=date(2026, 4, 15),
    )
    # Top-level structure
    assert "SchroederTrader Daily Report — 2026-04-28" in body
    assert "TODAY" in body
    assert "SYSTEM" in body
    assert "LLM ORACLES" in body
    assert "PERFORMANCE" in body
    # Real P&L row should reflect our synthetic portfolio
    assert "System (real)" in body
