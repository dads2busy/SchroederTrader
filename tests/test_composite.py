import numpy as np
import pandas as pd

from schroeder_trader.strategy.composite import (
    composite_signal_blended,
    composite_signal_hybrid,
    count_consecutive_bear_days,
    stale_cash_override,
)
from schroeder_trader.strategy.regime_detector import Regime
from schroeder_trader.strategy.sma_crossover import Signal


def test_hybrid_bull_routes_to_sma():
    signal, source = composite_signal_hybrid(Regime.BULL, Signal.BUY, Signal.SELL)
    assert signal == Signal.BUY
    assert source == "SMA"


def test_hybrid_bear_flat_by_default():
    signal, source = composite_signal_hybrid(Regime.BEAR, Signal.BUY, Signal.BUY)
    assert signal == Signal.SELL
    assert source == "FLAT"


def test_hybrid_bear_flat_when_not_weakening():
    signal, source = composite_signal_hybrid(Regime.BEAR, Signal.HOLD, Signal.BUY, bear_weakening=False)
    assert signal == Signal.SELL
    assert source == "FLAT"


def test_hybrid_bear_weakening_routes_to_xgb():
    """When bear is weakening (positive 5d return), route to XGB."""
    signal, source = composite_signal_hybrid(Regime.BEAR, Signal.HOLD, Signal.BUY, bear_weakening=True)
    assert signal == Signal.BUY
    assert source == "XGB_BEAR_WEAK"


def test_hybrid_bear_weakening_sell_signal():
    """Bear weakening still passes through XGB SELL signals."""
    signal, source = composite_signal_hybrid(Regime.BEAR, Signal.HOLD, Signal.SELL, bear_weakening=True)
    assert signal == Signal.SELL
    assert source == "XGB_BEAR_WEAK"


def test_hybrid_bear_weakening_hold_signal():
    """Bear weakening with XGB HOLD stays out."""
    signal, source = composite_signal_hybrid(Regime.BEAR, Signal.HOLD, Signal.HOLD, bear_weakening=True)
    assert signal == Signal.HOLD
    assert source == "XGB_BEAR_WEAK"


def test_hybrid_choppy_routes_to_xgb_low():
    signal, source = composite_signal_hybrid(Regime.CHOPPY, Signal.BUY, Signal.SELL)
    assert signal == Signal.SELL
    assert source == "XGB"


def test_hybrid_returns_tuple():
    result = composite_signal_hybrid(Regime.BULL, Signal.HOLD, Signal.HOLD)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], Signal)
    assert isinstance(result[1], str)


def test_bear_count_all_bear():
    regimes = pd.Series([Regime.BEAR] * 10)
    assert count_consecutive_bear_days(regimes) == 10


def test_bear_count_last_not_bear():
    regimes = pd.Series([Regime.BEAR, Regime.BEAR, Regime.CHOPPY])
    assert count_consecutive_bear_days(regimes) == 0


def test_bear_count_mixed_ending_bear():
    regimes = pd.Series([Regime.BULL, Regime.CHOPPY, Regime.BEAR])
    assert count_consecutive_bear_days(regimes) == 1


def test_bear_count_interrupted_streak():
    regimes = pd.Series([Regime.BEAR, Regime.BEAR, Regime.CHOPPY, Regime.BEAR, Regime.BEAR])
    assert count_consecutive_bear_days(regimes) == 2


def test_bear_count_last_nan():
    regimes = pd.Series([Regime.BEAR, Regime.BEAR, np.nan])
    assert count_consecutive_bear_days(regimes) == 0


def test_bear_count_empty():
    regimes = pd.Series([], dtype=object)
    assert count_consecutive_bear_days(regimes) == 0


class TestStaleCashOverride:
    def test_fires_in_bull_sma_above_enough_days(self):
        assert stale_cash_override(Regime.BULL, sma_50=400, sma_200=380, days_in_cash=7) is True

    def test_does_not_fire_before_threshold(self):
        assert stale_cash_override(Regime.BULL, sma_50=400, sma_200=380, days_in_cash=6) is False

    def test_does_not_fire_in_bear(self):
        assert stale_cash_override(Regime.BEAR, sma_50=400, sma_200=380, days_in_cash=10) is False

    def test_does_not_fire_in_choppy(self):
        assert stale_cash_override(Regime.CHOPPY, sma_50=400, sma_200=380, days_in_cash=10) is False

    def test_does_not_fire_when_sma_below(self):
        assert stale_cash_override(Regime.BULL, sma_50=370, sma_200=380, days_in_cash=10) is False

    def test_custom_threshold(self):
        assert stale_cash_override(Regime.BULL, sma_50=400, sma_200=380, days_in_cash=3, stale_cash_threshold=3) is True
        assert stale_cash_override(Regime.BULL, sma_50=400, sma_200=380, days_in_cash=2, stale_cash_threshold=3) is False


class TestCompositeSignalBlended:
    def test_pure_bull_sma_buy(self):
        exposure = composite_signal_blended(
            regime_probs={"BULL": 1.0, "BEAR": 0.0, "CHOPPY": 0.0},
            sma_signal=Signal.BUY, xgb_signal=Signal.SELL,
            bear_weakening=False, current_exposure=0.0,
        )
        assert abs(exposure - 0.98) < 1e-6

    def test_pure_bear_no_weakening(self):
        exposure = composite_signal_blended(
            regime_probs={"BULL": 0.0, "BEAR": 1.0, "CHOPPY": 0.0},
            sma_signal=Signal.BUY, xgb_signal=Signal.BUY,
            bear_weakening=False, current_exposure=0.5,
        )
        assert abs(exposure - 0.0) < 1e-6

    def test_pure_bear_with_weakening_xgb_buy(self):
        exposure = composite_signal_blended(
            regime_probs={"BULL": 0.0, "BEAR": 1.0, "CHOPPY": 0.0},
            sma_signal=Signal.HOLD, xgb_signal=Signal.BUY,
            bear_weakening=True, current_exposure=0.0,
        )
        assert abs(exposure - 0.98) < 1e-6

    def test_pure_choppy_xgb_buy(self):
        exposure = composite_signal_blended(
            regime_probs={"BULL": 0.0, "BEAR": 0.0, "CHOPPY": 1.0},
            sma_signal=Signal.HOLD, xgb_signal=Signal.BUY,
            bear_weakening=False, current_exposure=0.0,
        )
        assert abs(exposure - 0.98) < 1e-6

    def test_blended_60_bull_40_choppy(self):
        exposure = composite_signal_blended(
            regime_probs={"BULL": 0.6, "BEAR": 0.0, "CHOPPY": 0.4},
            sma_signal=Signal.BUY, xgb_signal=Signal.SELL,
            bear_weakening=False, current_exposure=0.0,
        )
        assert abs(exposure - 0.588) < 1e-6

    def test_hold_uses_current_exposure(self):
        exposure = composite_signal_blended(
            regime_probs={"BULL": 1.0, "BEAR": 0.0, "CHOPPY": 0.0},
            sma_signal=Signal.HOLD, xgb_signal=Signal.HOLD,
            bear_weakening=False, current_exposure=0.5,
        )
        assert abs(exposure - 0.5) < 1e-6

    def test_four_state_choppy_variants(self):
        exposure = composite_signal_blended(
            regime_probs={"BULL": 0.0, "BEAR": 0.0, "CHOPPY_0": 0.5, "CHOPPY_1": 0.5},
            sma_signal=Signal.HOLD, xgb_signal=Signal.BUY,
            bear_weakening=False, current_exposure=0.0,
        )
        assert abs(exposure - 0.98) < 1e-6

    def test_output_clamped_to_098(self):
        exposure = composite_signal_blended(
            regime_probs={"BULL": 1.0, "BEAR": 0.0, "CHOPPY": 0.0},
            sma_signal=Signal.BUY, xgb_signal=Signal.BUY,
            bear_weakening=False, current_exposure=0.98,
        )
        assert exposure <= 0.98

    def test_output_non_negative(self):
        exposure = composite_signal_blended(
            regime_probs={"BULL": 0.0, "BEAR": 1.0, "CHOPPY": 0.0},
            sma_signal=Signal.SELL, xgb_signal=Signal.SELL,
            bear_weakening=False, current_exposure=0.0,
        )
        assert exposure >= 0.0
