import numpy as np
import pandas as pd

from schroeder_trader.strategy.composite import (
    composite_signal_hybrid,
    count_consecutive_bear_days,
)
from schroeder_trader.strategy.regime_detector import Regime
from schroeder_trader.strategy.sma_crossover import Signal


def test_hybrid_bull_routes_to_sma():
    signal, source = composite_signal_hybrid(
        Regime.BULL, Signal.BUY, Signal.SELL, Signal.SELL, bear_days=0,
    )
    assert signal == Signal.BUY
    assert source == "SMA"


def test_hybrid_early_bear_routes_to_flat():
    signal, source = composite_signal_hybrid(
        Regime.BEAR, Signal.HOLD, Signal.BUY, Signal.BUY, bear_days=10,
    )
    assert signal == Signal.SELL
    assert source == "FLAT"


def test_hybrid_bear_day_20_still_flat():
    signal, source = composite_signal_hybrid(
        Regime.BEAR, Signal.HOLD, Signal.BUY, Signal.BUY, bear_days=20,
    )
    assert signal == Signal.SELL
    assert source == "FLAT"


def test_hybrid_bear_day_21_routes_to_xgb_high():
    signal, source = composite_signal_hybrid(
        Regime.BEAR, Signal.HOLD, Signal.SELL, Signal.BUY, bear_days=21,
    )
    assert signal == Signal.BUY
    assert source == "XGB"


def test_hybrid_choppy_routes_to_xgb_low():
    signal, source = composite_signal_hybrid(
        Regime.CHOPPY, Signal.BUY, Signal.SELL, Signal.HOLD, bear_days=0,
    )
    assert signal == Signal.SELL
    assert source == "XGB"


def test_hybrid_returns_tuple():
    result = composite_signal_hybrid(
        Regime.BULL, Signal.HOLD, Signal.HOLD, Signal.HOLD, bear_days=0,
    )
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
