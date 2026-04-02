import enum

import numpy as np
import pandas as pd


class Regime(enum.Enum):
    BULL = "BULL"
    BEAR = "BEAR"
    CHOPPY = "CHOPPY"


def detect_regime(
    log_return_20d: float,
    volatility_20d: float,
    vol_median: float,
) -> Regime:
    """Classify market regime from backward-looking indicators.

    Args:
        log_return_20d: 20-day log return of the close price.
        volatility_20d: 20-day rolling standard deviation of daily returns.
        vol_median: 252-day rolling median of *volatility_20d*.

    Returns:
        Regime enum value.
    """
    high_vol = volatility_20d > vol_median

    if log_return_20d > 0 and not high_vol:
        return Regime.BULL
    elif log_return_20d < 0 and high_vol:
        return Regime.BEAR
    else:
        return Regime.CHOPPY


def compute_regime_series(features_df: pd.DataFrame) -> pd.Series:
    """Compute a Regime enum series for every row in *features_df*.

    Rows where any required input is NaN receive ``np.nan``.

    Args:
        features_df: DataFrame with at least a ``close`` column.

    Returns:
        pd.Series of :class:`Regime` values (or NaN), indexed like *features_df*.
    """
    log_ret_20d = np.log(features_df["close"] / features_df["close"].shift(20))
    vol_20d = features_df["close"].pct_change().rolling(20).std()
    vol_med = vol_20d.rolling(252).median()

    regime_series = pd.Series(index=features_df.index, dtype=object)
    for idx in range(len(features_df)):
        lr = log_ret_20d.iloc[idx]
        vol = vol_20d.iloc[idx]
        vm = vol_med.iloc[idx]
        if pd.isna(lr) or pd.isna(vol) or pd.isna(vm):
            regime_series.iloc[idx] = np.nan
        else:
            regime_series.iloc[idx] = detect_regime(lr, vol, vm)

    return regime_series


_REGIME_MAP = {Regime.BEAR: 0, Regime.CHOPPY: 1, Regime.BULL: 2}


def compute_regime_labels(features_df: pd.DataFrame) -> pd.Series:
    """Return integer regime labels for every row in *features_df*.

    Label mapping: BEAR=0, CHOPPY=1, BULL=2.  NaN rows stay NaN.

    Args:
        features_df: DataFrame with at least a ``close`` column.

    Returns:
        pd.Series of integers (or NaN), indexed like *features_df*.
    """
    return compute_regime_series(features_df).map(_REGIME_MAP)
