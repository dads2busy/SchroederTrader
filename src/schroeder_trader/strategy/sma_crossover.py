import enum

import pandas as pd

from schroeder_trader.config import SMA_SHORT_WINDOW, SMA_LONG_WINDOW


class Signal(enum.Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


def generate_signal(
    df: pd.DataFrame,
    short_window: int = SMA_SHORT_WINDOW,
    long_window: int = SMA_LONG_WINDOW,
) -> tuple[Signal, float, float]:
    """Generate a trading signal based on SMA crossover.

    Args:
        df: DataFrame with a 'close' column and at least `long_window + 1` rows.
        short_window: Short SMA window (default 50).
        long_window: Long SMA window (default 200).

    Returns:
        Tuple of (Signal, sma_short_value, sma_long_value).

    Raises:
        ValueError: If DataFrame has fewer than long_window + 1 rows.
    """
    min_rows = long_window + 1
    if len(df) < min_rows:
        raise ValueError(f"Need at least {min_rows} rows, got {len(df)}. Minimum {long_window}+ bars required.")

    sma_short = df["close"].rolling(window=short_window).mean()
    sma_long = df["close"].rolling(window=long_window).mean()

    current_short = float(sma_short.iloc[-1])
    current_long = float(sma_long.iloc[-1])
    prev_short = float(sma_short.iloc[-2])
    prev_long = float(sma_long.iloc[-2])

    if current_short > current_long and prev_short <= prev_long:
        return Signal.BUY, current_short, current_long
    elif current_short < current_long and prev_short >= prev_long:
        return Signal.SELL, current_short, current_long
    else:
        return Signal.HOLD, current_short, current_long
