from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd


from schroeder_trader.reports.daily_email import (
    build_today_section,
    build_system_section,
    build_oracles_section,
    build_email_body,
    build_sector_shadow_section,
    build_comparison_section,
    _compute_performance,
    _exposure_from_decisions,
    _compute_ticker_shadow_pnl,
    _compute_basket_pnl,
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

    # Empty shadow_signals.csv (no sector shadow section)
    pd.DataFrame(columns=[
        "id", "timestamp", "ticker", "close_price", "ml_signal",
    ]).to_csv(tmp_path / "shadow_signals.csv", index=False)

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
        sector_close_histories={},
    )
    assert "SchroederTrader Daily Report — 2026-04-28" in body
    assert "TODAY" in body
    assert "SYSTEM" in body
    assert "LLM ORACLES" in body
    assert "PERFORMANCE" in body
    # No sector shadow section when csv is empty
    assert "SECTOR SHADOW" not in body
    assert "System (real)" in body


def test_build_email_body_includes_sector_shadow(tmp_path):
    dates = pd.bdate_range("2026-04-15", periods=10).tz_localize(None)
    spy = pd.DataFrame({"close": [700.0 + i for i in range(10)]}, index=dates)
    pd.DataFrame(columns=["id", "timestamp", "cash", "position_qty", "position_value", "total_value"]) \
      .to_csv(tmp_path / "portfolio.csv", index=False)
    pd.DataFrame(columns=["id", "timestamp", "provider", "target_exposure", "error"]) \
      .to_csv(tmp_path / "llm_shadow_signals.csv", index=False)
    pd.DataFrame({
        "timestamp": [
            "2026-05-12T20:30:00+00:00", "2026-05-13T20:30:00+00:00", "2026-05-14T20:30:00+00:00",
        ],
        "ticker": ["XLK"] * 3,
        "close_price": [100.0, 110.0, 121.0],
        "ml_signal": ["BUY", "HOLD", "HOLD"],
    }).to_csv(tmp_path / "shadow_signals.csv", index=False)

    xlk_closes = pd.Series(
        [100.0, 110.0, 121.0],
        index=pd.to_datetime(["2026-05-12", "2026-05-13", "2026-05-14"]),
    )

    body = build_email_body(
        date_str="2026-05-14",
        spy_close=715.16, spy_prev_close=713.97,
        portfolio_value=101883.0, portfolio_prev_value=100665.0,
        cash=1965.0, position_qty=141,
        sma_signal="HOLD", sma_50=676.0, sma_200=667.0,
        composite_signal="BUY", composite_source="XGB", regime="CHOPPY",
        bear_days=0, xgb_proba_up=0.578, xgb_threshold=0.35,
        today_action="HOLD",
        oracle_responses=[],
        data_root=tmp_path,
        spy_history=spy,
        live_start_date=date(2026, 4, 15),
        sector_close_histories={"XLK": xlk_closes},
    )
    assert "SECTOR SHADOW" in body
    assert "XLK" in body
    assert "+21.00%" in body


def test_exposure_from_decisions_buy_hold_sell_carry_forward():
    decisions = {
        date(2026, 5, 12): "BUY",
        date(2026, 5, 13): "HOLD",
        date(2026, 5, 14): "HOLD",
        date(2026, 5, 15): "SELL",
        date(2026, 5, 18): "HOLD",
    }
    exposure = _exposure_from_decisions(decisions)
    assert exposure[date(2026, 5, 12)] == 1.0
    assert exposure[date(2026, 5, 13)] == 1.0  # HOLD carries BUY forward
    assert exposure[date(2026, 5, 14)] == 1.0
    assert exposure[date(2026, 5, 15)] == 0.0  # SELL flattens
    assert exposure[date(2026, 5, 18)] == 0.0  # HOLD carries SELL forward


def test_exposure_from_decisions_starts_flat_if_first_is_hold():
    decisions = {
        date(2026, 5, 12): "HOLD",
        date(2026, 5, 13): "BUY",
    }
    exposure = _exposure_from_decisions(decisions)
    assert exposure[date(2026, 5, 12)] == 0.0  # nothing to carry forward
    assert exposure[date(2026, 5, 13)] == 1.0


def test_exposure_from_decisions_empty_input():
    assert _exposure_from_decisions({}) == {}


def test_exposure_from_decisions_rejects_unknown_decision():
    import pytest as _pytest
    with _pytest.raises(ValueError, match="Unknown decision"):
        _exposure_from_decisions({date(2026, 5, 12): "SELL "})  # trailing space


def test_compute_ticker_shadow_pnl_basic():
    # Three days for XLK: BUY at 100, HOLD at 110, SELL at 105
    # Expected: exposure 1.0 from day 1 onward.
    #   day 1 → day 2: 1.0 * (110/100 - 1) = +10%
    #   day 2 → day 3: 1.0 * (105/110 - 1) = -4.55%
    #   composite final = 100 * 1.10 * (105/110) = 105.0  →  +5.00%
    # B&H: (105/100 - 1) = +5.00%  →  edge 0.00pp (signals never sold in time)
    shadow_df = pd.DataFrame({
        "timestamp": [
            "2026-05-12T20:30:00+00:00",
            "2026-05-13T20:30:00+00:00",
            "2026-05-14T20:30:00+00:00",
        ],
        "ticker": ["XLK", "XLK", "XLK"],
        "close_price": [100.0, 110.0, 105.0],
        "ml_signal": ["BUY", "HOLD", "SELL"],
    })
    closes = pd.Series(
        [100.0, 110.0, 105.0],
        index=pd.to_datetime(["2026-05-12", "2026-05-13", "2026-05-14"]),
        name="close",
    )
    result = _compute_ticker_shadow_pnl(shadow_df, closes)
    assert result is not None
    assert result["sessions"] == 3
    assert result["inception"] == date(2026, 5, 12)
    assert abs(result["composite_return_pct"] - 5.0) < 1e-6
    assert abs(result["bnh_return_pct"] - 5.0) < 1e-6
    assert abs(result["edge_pp"] - 0.0) < 1e-6


def test_compute_ticker_shadow_pnl_skips_single_session():
    shadow_df = pd.DataFrame({
        "timestamp": ["2026-05-12T20:30:00+00:00"],
        "ticker": ["XLK"],
        "close_price": [100.0],
        "ml_signal": ["BUY"],
    })
    closes = pd.Series(
        [100.0], index=pd.to_datetime(["2026-05-12"]), name="close",
    )
    assert _compute_ticker_shadow_pnl(shadow_df, closes) is None


def test_compute_ticker_shadow_pnl_sell_then_buy_captures_partial_run():
    # Day 1 BUY @100, Day 2 SELL @110 (signal applied next day), Day 3 BUY @105, Day 4 HOLD @120
    # Composite exposures: d1→d2 = 1.0, d2→d3 = 0.0, d3→d4 = 1.0
    #   d1→d2: 1.0 * (110/100 - 1) = +10%   → 110.0
    #   d2→d3: 0.0 * (105/110 - 1) = 0       → 110.0
    #   d3→d4: 1.0 * (120/105 - 1) = +14.29% → 125.71
    # Composite return = +25.71%; B&H = +20.00%; edge ≈ +5.71pp
    shadow_df = pd.DataFrame({
        "timestamp": [
            "2026-05-12T20:30:00+00:00",
            "2026-05-13T20:30:00+00:00",
            "2026-05-14T20:30:00+00:00",
            "2026-05-15T20:30:00+00:00",
        ],
        "ticker": ["XLK"] * 4,
        "close_price": [100.0, 110.0, 105.0, 120.0],
        "ml_signal": ["BUY", "SELL", "BUY", "HOLD"],
    })
    closes = pd.Series(
        [100.0, 110.0, 105.0, 120.0],
        index=pd.to_datetime(["2026-05-12", "2026-05-13", "2026-05-14", "2026-05-15"]),
        name="close",
    )
    result = _compute_ticker_shadow_pnl(shadow_df, closes)
    assert result["sessions"] == 4
    assert abs(result["composite_return_pct"] - 25.7142857) < 1e-4
    assert abs(result["bnh_return_pct"] - 20.0) < 1e-6
    assert abs(result["edge_pp"] - 5.7142857) < 1e-4


def test_sector_shadow_section_two_tickers(tmp_path):
    # XLK: 3 sessions, composite=BUY all → matches B&H
    # XLE: 3 sessions, BUY then SELL then BUY → sits out the middle day
    shadow = pd.DataFrame({
        "timestamp": [
            "2026-05-12T20:30:00+00:00", "2026-05-13T20:30:00+00:00", "2026-05-14T20:30:00+00:00",
            "2026-05-12T20:30:00+00:00", "2026-05-13T20:30:00+00:00", "2026-05-14T20:30:00+00:00",
            "2026-05-12T20:30:00+00:00", "2026-05-13T20:30:00+00:00", "2026-05-14T20:30:00+00:00",
        ],
        "ticker": ["XLK", "XLK", "XLK", "XLE", "XLE", "XLE", "SPY", "SPY", "SPY"],
        "close_price": [
            100.0, 110.0, 121.0,
             50.0,  55.0,  52.0,
            700.0, 710.0, 720.0,
        ],
        "ml_signal": ["BUY"] * 3 + ["BUY", "SELL", "BUY"] + ["HOLD"] * 3,
    })
    shadow.to_csv(tmp_path / "shadow_signals.csv", index=False)

    xlk_closes = pd.Series(
        [100.0, 110.0, 121.0],
        index=pd.to_datetime(["2026-05-12", "2026-05-13", "2026-05-14"]),
    )
    xle_closes = pd.Series(
        [50.0, 55.0, 52.0],
        index=pd.to_datetime(["2026-05-12", "2026-05-13", "2026-05-14"]),
    )

    section = build_sector_shadow_section(
        shadow_signals_path=tmp_path / "shadow_signals.csv",
        ticker_close_histories={"XLK": xlk_closes, "XLE": xle_closes},
    )
    assert "SECTOR SHADOW" in section
    assert "XLK" in section
    assert "XLE" in section
    assert "SPY" not in section  # SPY excluded
    # XLK composite return = B&H = +21.00%
    assert "+21.00%" in section
    # XLE composite: exposures = [1.0, 1.0, 0.0] → value sequence 100, 110, 110 → +10.00%
    assert "+10.00%" in section


def test_sector_shadow_section_empty_when_only_spy(tmp_path):
    shadow = pd.DataFrame({
        "timestamp": ["2026-05-12T20:30:00+00:00"],
        "ticker": ["SPY"],
        "close_price": [700.0],
        "ml_signal": ["HOLD"],
    })
    shadow.to_csv(tmp_path / "shadow_signals.csv", index=False)

    section = build_sector_shadow_section(
        shadow_signals_path=tmp_path / "shadow_signals.csv",
        ticker_close_histories={},
    )
    assert section == ""


def test_sector_shadow_section_missing_file_returns_empty(tmp_path):
    section = build_sector_shadow_section(
        shadow_signals_path=tmp_path / "does-not-exist.csv",
        ticker_close_histories={},
    )
    assert section == ""


def test_sector_shadow_section_ignores_nan_ticker(tmp_path):
    # A blank ticker cell shouldn't crash sorted() — drop NaN before unique.
    shadow = pd.DataFrame({
        "timestamp": [
            "2026-05-12T20:30:00+00:00", "2026-05-13T20:30:00+00:00",
            "2026-05-12T20:30:00+00:00",
        ],
        "ticker": ["XLK", "XLK", None],
        "close_price": [100.0, 110.0, 50.0],
        "ml_signal": ["BUY", "HOLD", "BUY"],
    })
    shadow.to_csv(tmp_path / "shadow_signals.csv", index=False)

    xlk_closes = pd.Series(
        [100.0, 110.0],
        index=pd.to_datetime(["2026-05-12", "2026-05-13"]),
    )

    section = build_sector_shadow_section(
        shadow_signals_path=tmp_path / "shadow_signals.csv",
        ticker_close_histories={"XLK": xlk_closes},
    )
    assert "XLK" in section
    assert "+10.00%" in section  # XLK B&H over 100 → 110


def test_compute_ticker_shadow_pnl_handles_date_indexed_closes():
    # Some upstream callers produce a Series indexed by datetime.date
    # rather than a DatetimeIndex (CSV round-trips, .set_index on a date column).
    # The function should accept both.
    shadow_df = pd.DataFrame({
        "timestamp": [
            "2026-05-12T20:30:00+00:00",
            "2026-05-13T20:30:00+00:00",
            "2026-05-14T20:30:00+00:00",
        ],
        "ticker": ["XLK"] * 3,
        "close_price": [100.0, 110.0, 105.0],
        "ml_signal": ["BUY", "HOLD", "SELL"],
    })
    closes = pd.Series(
        [100.0, 110.0, 105.0],
        index=pd.Index([date(2026, 5, 12), date(2026, 5, 13), date(2026, 5, 14)]),
        name="close",
    )
    result = _compute_ticker_shadow_pnl(shadow_df, closes)
    assert result is not None
    assert result["sessions"] == 3
    assert abs(result["composite_return_pct"] - 5.0) < 1e-6


def test_compute_ticker_shadow_pnl_returns_value_series_for_basket_math():
    shadow_df = pd.DataFrame({
        "timestamp": [
            "2026-05-12T20:30:00+00:00",
            "2026-05-13T20:30:00+00:00",
            "2026-05-14T20:30:00+00:00",
        ],
        "ticker": ["XLK"] * 3,
        "close_price": [100.0, 110.0, 121.0],
        "ml_signal": ["BUY", "HOLD", "HOLD"],
    })
    closes = pd.Series(
        [100.0, 110.0, 121.0],
        index=pd.to_datetime(["2026-05-12", "2026-05-13", "2026-05-14"]),
    )
    result = _compute_ticker_shadow_pnl(shadow_df, closes)
    assert "composite_value_series" in result
    assert "bnh_value_series" in result
    vs = result["composite_value_series"]
    assert abs(float(vs.iloc[0]) - 100.0) < 1e-6
    assert abs(float(vs.iloc[-1]) - 121.0) < 1e-6


def test_compute_basket_pnl_daily_rebalanced():
    # Two tickers, three sessions. SPY value 100→105→110; XLK value 100→120→144.
    # Daily rets: SPY = [+5%, +4.76%]; XLK = [+20%, +20%].
    # Weights 50/50 → basket daily rets = [+12.5%, +12.38%].
    # Basket value: 100 * 1.125 * 1.1238 = 126.4%; total return = +26.4%
    spy_vs = pd.Series([100.0, 105.0, 110.0],
                       index=pd.to_datetime(["2026-05-12", "2026-05-13", "2026-05-14"]))
    xlk_vs = pd.Series([100.0, 120.0, 144.0],
                       index=pd.to_datetime(["2026-05-12", "2026-05-13", "2026-05-14"]))
    per_ticker = {
        "SPY": {"composite_value_series": spy_vs, "bnh_value_series": spy_vs},
        "XLK": {"composite_value_series": xlk_vs, "bnh_value_series": xlk_vs},
    }
    weights = {"SPY": 0.5, "XLK": 0.5}
    result = _compute_basket_pnl(per_ticker, weights)
    assert result is not None
    assert result["sessions"] == 3
    # 1.125 * 1.123809524 = 1.264285... → +26.43%
    expected = (1.125 * (130 / 115.5 + 1 - 1) / 1 - 1) * 100  # paranoid manual check
    # Cleaner: just compute it the way the function would
    expected_basket = (1.125 * 1.123809523809) - 1
    assert abs(result["composite_return_pct"] / 100 - expected_basket) < 1e-6
    # Composite and B&H are the same series here so edge should be 0
    assert abs(result["edge_pp"]) < 1e-9


def test_compute_basket_pnl_returns_none_when_ticker_missing():
    per_ticker = {
        "SPY": {
            "composite_value_series": pd.Series([100.0, 110.0]),
            "bnh_value_series": pd.Series([100.0, 110.0]),
        },
    }
    weights = {"SPY": 0.5, "XLK": 0.5}  # XLK absent
    assert _compute_basket_pnl(per_ticker, weights) is None


def test_sector_shadow_section_renders_basket_row(tmp_path):
    shadow = pd.DataFrame({
        "timestamp": [
            "2026-05-12T20:30:00+00:00", "2026-05-13T20:30:00+00:00", "2026-05-14T20:30:00+00:00",
            "2026-05-12T20:30:00+00:00", "2026-05-13T20:30:00+00:00", "2026-05-14T20:30:00+00:00",
        ],
        "ticker": ["SPY", "SPY", "SPY", "XLK", "XLK", "XLK"],
        "close_price": [700.0, 700.0, 700.0, 100.0, 100.0, 100.0],
        "ml_signal": ["BUY"] * 3 + ["BUY"] * 3,
    })
    shadow.to_csv(tmp_path / "shadow_signals.csv", index=False)

    spy_closes = pd.Series(
        [700.0, 707.0, 714.0],
        index=pd.to_datetime(["2026-05-12", "2026-05-13", "2026-05-14"]),
    )
    xlk_closes = pd.Series(
        [100.0, 105.0, 110.0],
        index=pd.to_datetime(["2026-05-12", "2026-05-13", "2026-05-14"]),
    )

    section = build_sector_shadow_section(
        shadow_signals_path=tmp_path / "shadow_signals.csv",
        ticker_close_histories={"SPY": spy_closes, "XLK": xlk_closes},
        basket_weights={"SPY": 0.5, "XLK": 0.5},
    )
    assert "BASKET" in section
    assert "SECTOR SHADOW" in section
    # XLK row should appear; SPY's per-row should not (excluded from per-row display)
    assert "XLK" in section


def test_sector_shadow_section_omits_basket_when_ticker_missing(tmp_path):
    # XLK has shadow data, but basket weights ask for XLV too — basket should
    # be silently omitted while the XLK row still renders.
    shadow = pd.DataFrame({
        "timestamp": [
            "2026-05-12T20:30:00+00:00", "2026-05-13T20:30:00+00:00",
        ],
        "ticker": ["XLK", "XLK"],
        "close_price": [100.0, 110.0],
        "ml_signal": ["BUY", "HOLD"],
    })
    shadow.to_csv(tmp_path / "shadow_signals.csv", index=False)
    xlk_closes = pd.Series(
        [100.0, 110.0],
        index=pd.to_datetime(["2026-05-12", "2026-05-13"]),
    )

    section = build_sector_shadow_section(
        shadow_signals_path=tmp_path / "shadow_signals.csv",
        ticker_close_histories={"XLK": xlk_closes},
        basket_weights={"SPY": 0.5, "XLK": 0.5},
    )
    assert "XLK" in section
    assert "BASKET" not in section


def test_build_basket_paper_section_renders_all_tickers(tmp_path):
    from schroeder_trader.reports.daily_email import build_basket_paper_section
    from datetime import date

    portfolio_df = pd.DataFrame({
        "timestamp": ["2026-05-21T20:25:00+00:00"] * 4,
        "pipeline": ["basket"] * 4,
        "ticker": ["SPY", "XLK", "XLV", "XLE"],
        "cash": [2140.0] * 4,
        "position_qty": [64, 181, 111, 175],
        "position_value": [47747.0, 32491.0, 16212.0, 10150.0],
        "total_value": [106540.0] * 4,
    })
    shadow_signals_df = pd.DataFrame({
        "timestamp": ["2026-05-21T20:25:00+00:00"] * 4,
        "pipeline": ["basket"] * 4,
        "ticker": ["SPY", "XLK", "XLV", "XLE"],
        "ml_signal": ["HOLD", "HOLD", "HOLD", "BUY"],
        "signal_source": ["SMA", "XGB", "XGB", "XGB"],
        "trailing_stop_triggered": [0, 0, 0, 0],
        "high_water_mark": [47747.0, 32491.0, 16212.0, 10150.0],
    })
    weights = {"SPY": 0.45, "XLK": 0.30, "XLV": 0.15, "XLE": 0.10}
    section = build_basket_paper_section(
        portfolio_df=portfolio_df,
        shadow_signals_df=shadow_signals_df,
        basket_weights=weights,
        launch_date=date(2026, 5, 21),
    )
    assert "BASKET PAPER" in section
    assert "SPY" in section
    assert "XLK" in section
    assert "XLV" in section
    assert "XLE" in section
    assert "45.0%" in section  # SPY target weight
    assert "$106,540" in section or "106,540" in section


def test_build_basket_paper_section_shows_fired_stop():
    from schroeder_trader.reports.daily_email import build_basket_paper_section
    from datetime import date

    portfolio_df = pd.DataFrame({
        "timestamp": ["2026-07-14T20:25:00+00:00"],
        "pipeline": ["basket"],
        "ticker": ["XLE"],
        "cash": [10000.0], "position_qty": [0],
        "position_value": [0.0], "total_value": [100000.0],
    })
    shadow_signals_df = pd.DataFrame({
        "timestamp": ["2026-07-14T20:25:00+00:00"],
        "pipeline": ["basket"], "ticker": ["XLE"],
        "ml_signal": ["SELL"], "signal_source": ["FLAT"],
        "trailing_stop_triggered": [1],
        "high_water_mark": [61.20],
    })
    section = build_basket_paper_section(
        portfolio_df=portfolio_df, shadow_signals_df=shadow_signals_df,
        basket_weights={"XLE": 1.0}, launch_date=date(2026, 5, 21),
    )
    assert "FIRED" in section
    assert "trailing stop fired" in section.lower()


def test_build_basket_paper_section_returns_empty_when_no_basket_rows():
    from schroeder_trader.reports.daily_email import build_basket_paper_section
    from datetime import date

    portfolio_df = pd.DataFrame(columns=[
        "timestamp", "pipeline", "ticker", "cash",
        "position_qty", "position_value", "total_value",
    ])
    shadow_signals_df = pd.DataFrame(columns=[
        "timestamp", "pipeline", "ticker", "ml_signal",
        "signal_source", "trailing_stop_triggered", "high_water_mark",
    ])
    section = build_basket_paper_section(
        portfolio_df=portfolio_df, shadow_signals_df=shadow_signals_df,
        basket_weights={"SPY": 1.0}, launch_date=date(2026, 5, 21),
    )
    assert section == ""


def test_build_basket_paper_section_renders_xgb_bear_weak_source():
    """Long signal_source (XGB_BEAR_WEAK) shouldn't break column alignment."""
    from schroeder_trader.reports.daily_email import build_basket_paper_section
    from datetime import date

    portfolio_df = pd.DataFrame({
        "timestamp": ["2026-05-12T20:30:00+00:00"],
        "pipeline": ["basket"],
        "ticker": ["XLV"],
        "cash": [500.0],
        "position_qty": [107],
        "position_value": [15763.0],
        "total_value": [100000.0],
    })
    shadow_signals_df = pd.DataFrame({
        "timestamp": ["2026-05-12T20:30:00+00:00"],
        "pipeline": ["basket"],
        "ticker": ["XLV"],
        "ml_signal": ["BUY"],
        "signal_source": ["XGB_BEAR_WEAK"],
        "trailing_stop_triggered": [0],
        "high_water_mark": [15763.0],
    })
    section = build_basket_paper_section(
        portfolio_df=portfolio_df,
        shadow_signals_df=shadow_signals_df,
        basket_weights={"XLV": 1.0},
        launch_date=date(2026, 5, 12),
    )
    # The signal+source should appear fully (not truncated)
    assert "BUY (XGB_BEAR_WEAK)" in section
    # All rendered lines should have the same length (no alignment drift)
    lines = [ln for ln in section.split("\n") if ln.strip().startswith(("Ticker", "----", "XLV"))]
    # At least header, separator, and data row
    assert len(lines) >= 3
    lengths = set(len(ln.rstrip()) for ln in lines)
    assert len(lengths) == 1, f"Lines have inconsistent widths: {lengths}"


def test_compute_performance_system_ignores_basket_rows(tmp_path):
    """System (real) must use only spy_only rows, not the larger basket rows."""
    pd.DataFrame({
        "timestamp": [
            "2026-04-15T20:30:00+00:00", "2026-05-18T20:30:00+00:00",  # spy_only
            "2026-06-02T22:11:00+00:00",                                # basket (larger)
        ],
        "pipeline": ["spy_only", "spy_only", "basket"],
        "ticker": ["SPY", "SPY", "XLE"],
        "total_value": [100000.0, 106000.0, 999999.0],  # basket value is a decoy
    }).to_csv(tmp_path / "portfolio.csv", index=False)
    pd.DataFrame(columns=["timestamp", "provider", "target_exposure", "error"]).to_csv(
        tmp_path / "llm_shadow_signals.csv", index=False
    )
    idx = pd.bdate_range("2026-04-15", "2026-05-18")
    spy = pd.DataFrame({"close": [700.0 + i for i in range(len(idx))]}, index=idx)

    perf = _compute_performance(data_root=tmp_path, spy_history=spy, start_date=date(2026, 4, 15))
    # 106000/100000 - 1 = +6.00%, NOT contaminated by the 999999 basket row
    assert perf["system_real"] == 106000.0
    assert round(perf["system_real_return_pct"], 2) == 6.00


def test_build_comparison_section_four_way_table(tmp_path):
    """Unified scorecard renders all four + SPY, sorted, since basket launch."""
    pd.DataFrame({
        "timestamp": [
            "2026-05-18T20:30:00+00:00",                                # spy_only baseline
            "2026-05-19T20:38:00+00:00", "2026-06-02T22:11:00+00:00",   # basket inception->latest
        ],
        "pipeline": ["spy_only", "basket", "basket"],
        "ticker": ["SPY", "SPY", "SPY"],
        "total_value": [100000.0, 100000.0, 105000.0],  # basket +5%
    }).to_csv(tmp_path / "portfolio.csv", index=False)
    # Claude fully invested (= SPY), OpenAI half-invested
    pd.DataFrame({
        "timestamp": ["2026-05-19T20:30:00+00:00"] * 2,
        "provider": ["claude", "openai"],
        "target_exposure": [1.0, 0.5],
        "error": [None, None],
    }).to_csv(tmp_path / "llm_shadow_signals.csv", index=False)
    # SPY +3% over the window
    idx = pd.to_datetime(["2026-05-19", "2026-05-20", "2026-06-02"])
    spy = pd.DataFrame({"close": [600.0, 600.0, 618.0]}, index=idx)

    section = build_comparison_section(
        data_root=tmp_path, spy_history=spy,
        basket_launch_date=date(2026, 5, 19),
        system_current_value=104000.0,  # 104000/100000 = +4%
    )
    assert "SCORECARD" in section
    for label in ("Basket", "System (real)", "Claude (sim)", "OpenAI (sim)", "SPY B&H"):
        assert label in section
    assert "Per $100k" in section and "vs SPY" in section
    assert "+5.00%" in section   # basket
    assert "+4.00%" in section   # system from live value vs pre-launch baseline
    # Sorted by return desc: Basket(+5) before System(+4) before SPY(+3)
    assert section.index("Basket") < section.index("System (real)") < section.index("SPY B&H")


def test_build_basket_paper_section_shows_return_and_vs_spy():
    """With >=2 snapshots and spy_history, render Return-since-launch and vs-SPY."""
    from schroeder_trader.reports.daily_email import build_basket_paper_section
    from datetime import date

    # Inception 100k (5/19) -> latest 105k (6/02) = +5.00%
    portfolio_df = pd.DataFrame({
        "timestamp": ["2026-05-19T20:38:00+00:00", "2026-06-02T22:11:00+00:00"],
        "pipeline": ["basket", "basket"],
        "ticker": ["SPY", "SPY"],
        "cash": [100.0, 100.0],
        "position_qty": [160, 168],
        "position_value": [99900.0, 104900.0],
        "total_value": [100000.0, 105000.0],
    })
    shadow_signals_df = pd.DataFrame({
        "timestamp": ["2026-06-02T22:11:00+00:00"],
        "pipeline": ["basket"], "ticker": ["SPY"],
        "ml_signal": ["HOLD"], "signal_source": ["SMA"],
        "trailing_stop_triggered": [0], "high_water_mark": [104900.0],
    })
    # SPY B&H over same window: 600 -> 606 = +1.00%; basket edge = +4.00 pp
    spy = pd.DataFrame(
        {"close": [600.0, 606.0]},
        index=pd.to_datetime(["2026-05-19", "2026-06-02"]),
    )
    section = build_basket_paper_section(
        portfolio_df=portfolio_df,
        shadow_signals_df=shadow_signals_df,
        basket_weights={"SPY": 1.0},
        launch_date=date(2026, 5, 19),
        spy_history=spy,
    )
    assert "Return:" in section
    assert "+5.00%" in section
    assert "vs SPY B&H" in section
    assert "+4.00 pp" in section


def test_build_basket_paper_section_omits_return_with_single_snapshot():
    """A single basket snapshot has no since-launch return, so omit the line."""
    from schroeder_trader.reports.daily_email import build_basket_paper_section
    from datetime import date

    portfolio_df = pd.DataFrame({
        "timestamp": ["2026-05-19T20:38:00+00:00"],
        "pipeline": ["basket"], "ticker": ["SPY"],
        "cash": [100.0], "position_qty": [160],
        "position_value": [99900.0], "total_value": [100000.0],
    })
    shadow_signals_df = pd.DataFrame({
        "timestamp": ["2026-05-19T20:38:00+00:00"],
        "pipeline": ["basket"], "ticker": ["SPY"],
        "ml_signal": ["HOLD"], "signal_source": ["SMA"],
        "trailing_stop_triggered": [0], "high_water_mark": [99900.0],
    })
    spy = pd.DataFrame({"close": [600.0]}, index=pd.to_datetime(["2026-05-19"]))
    section = build_basket_paper_section(
        portfolio_df=portfolio_df,
        shadow_signals_df=shadow_signals_df,
        basket_weights={"SPY": 1.0},
        launch_date=date(2026, 5, 19),
        spy_history=spy,
    )
    assert "Return:" not in section
    assert "Total value:" in section


def test_build_email_body_omits_basket_section_when_basket_state_is_none(tmp_path):
    from datetime import date
    dates = pd.bdate_range("2026-04-15", periods=10).tz_localize(None)
    spy = pd.DataFrame({"close": [700.0 + i for i in range(10)]}, index=dates)
    pd.DataFrame(columns=["id", "timestamp", "cash", "position_qty", "position_value", "total_value"]) \
      .to_csv(tmp_path / "portfolio.csv", index=False)
    pd.DataFrame(columns=["id", "timestamp", "provider", "target_exposure", "error"]) \
      .to_csv(tmp_path / "llm_shadow_signals.csv", index=False)
    pd.DataFrame(columns=["id", "timestamp", "ticker", "close_price", "ml_signal"]) \
      .to_csv(tmp_path / "shadow_signals.csv", index=False)

    body = build_email_body(
        date_str="2026-05-14",
        spy_close=715.16, spy_prev_close=713.97,
        portfolio_value=101883.0, portfolio_prev_value=100665.0,
        cash=1965.0, position_qty=141,
        sma_signal="HOLD", sma_50=676.0, sma_200=667.0,
        composite_signal="BUY", composite_source="XGB", regime="CHOPPY",
        bear_days=0, xgb_proba_up=0.578, xgb_threshold=0.35,
        today_action="HOLD",
        oracle_responses=[],
        data_root=tmp_path,
        spy_history=spy,
        live_start_date=date(2026, 4, 15),
        sector_close_histories={},
        basket_state=None,
    )
    assert "BASKET PAPER" not in body
