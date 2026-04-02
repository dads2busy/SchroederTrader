import numpy as np
import pandas as pd

from schroeder_trader.strategy.regime_detector import Regime
from schroeder_trader.strategy.sma_crossover import Signal


def composite_signal_hybrid(
    regime: Regime,
    sma_signal: Signal,
    xgb_signal_low: Signal,
) -> tuple[Signal, str]:
    """Route signal based on regime.

    Routing:
        BULL → SMA crossover signal
        CHOPPY → XGB at low confidence threshold
        BEAR → always SELL (stay flat)

    Returns:
        Tuple of (Signal, source) where source is "SMA", "FLAT", or "XGB".
    """
    if regime == Regime.BULL:
        return sma_signal, "SMA"
    elif regime == Regime.BEAR:
        return Signal.SELL, "FLAT"
    else:
        return xgb_signal_low, "XGB"


def count_consecutive_bear_days(regimes: pd.Series) -> int:
    """Count consecutive BEAR days ending at the last row.

    Returns 0 if the last row is not BEAR, is NaN, or series is empty.
    """
    if len(regimes) == 0:
        return 0

    last = regimes.iloc[-1]
    if not isinstance(last, Regime) or last != Regime.BEAR:
        return 0

    count = 0
    for i in range(len(regimes) - 1, -1, -1):
        val = regimes.iloc[i]
        if isinstance(val, Regime) and val == Regime.BEAR:
            count += 1
        else:
            break

    return count
