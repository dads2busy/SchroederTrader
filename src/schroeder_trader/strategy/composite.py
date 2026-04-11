import numpy as np
import pandas as pd

from schroeder_trader.strategy.regime_detector import Regime
from schroeder_trader.strategy.sma_crossover import Signal


def composite_signal_hybrid(
    regime: Regime,
    sma_signal: Signal,
    xgb_signal_low: Signal,
    bear_weakening: bool = False,
) -> tuple[Signal, str]:
    """Route signal based on regime.

    Routing:
        BULL → SMA crossover signal
        CHOPPY → XGB at low confidence threshold
        BEAR → SELL (stay flat), unless bear is weakening (positive
               5-day return) in which case route to XGB

    Returns:
        Tuple of (Signal, source) where source is "SMA", "FLAT",
        "XGB", or "XGB_BEAR_WEAK".
    """
    if regime == Regime.BULL:
        return sma_signal, "SMA"
    elif regime == Regime.BEAR:
        if bear_weakening:
            return xgb_signal_low, "XGB_BEAR_WEAK"
        return Signal.SELL, "FLAT"
    else:
        return xgb_signal_low, "XGB"


MAX_EXPOSURE = 0.98


def _signal_to_exposure(signal: Signal, current_exposure: float) -> float:
    """Convert a Signal enum to a target exposure value."""
    if signal == Signal.BUY:
        return MAX_EXPOSURE
    elif signal == Signal.SELL:
        return 0.0
    else:
        return current_exposure


def composite_signal_blended(
    regime_probs: dict[str, float],
    sma_signal: Signal,
    xgb_signal: Signal,
    bear_weakening: bool = False,
    current_exposure: float = 0.0,
) -> float:
    """Compute blended target exposure from regime probabilities.

    Each regime's signal source determines a target exposure, then
    the final exposure is the probability-weighted average.

    Args:
        regime_probs: Dict mapping regime label to probability.
            Keys: "BULL", "BEAR", "CHOPPY" (or "CHOPPY_0", "CHOPPY_1" for 4-state).
        sma_signal: Signal from SMA crossover.
        xgb_signal: Signal from XGB at low confidence threshold.
        bear_weakening: True if 5-day return is positive while dominant regime is BEAR.
        current_exposure: Current portfolio exposure (0.0 to 0.98).

    Returns:
        Target exposure as float in [0.0, 0.98].
    """
    blended = 0.0

    for label, prob in regime_probs.items():
        if prob <= 0:
            continue

        if label == "BULL":
            target = _signal_to_exposure(sma_signal, current_exposure)
        elif label == "BEAR":
            if bear_weakening:
                target = _signal_to_exposure(xgb_signal, current_exposure)
            else:
                target = 0.0
        else:
            # CHOPPY, CHOPPY_0, CHOPPY_1, etc. all route to XGB
            target = _signal_to_exposure(xgb_signal, current_exposure)

        blended += prob * target

    return max(0.0, min(MAX_EXPOSURE, blended))


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
