import enum


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
