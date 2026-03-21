import math


def kelly_fraction(
    p_up: float,
    p_down: float,
    win_loss_ratio: float,
    kelly_multiplier: float = 0.5,
) -> float:
    """Compute fractional Kelly position size.

    Args:
        p_up: XGBoost P(UP) — probability of a winning trade.
        p_down: XGBoost P(DOWN) — probability of a losing trade.
        win_loss_ratio: Average win / average loss from backtest. Must be > 0.
        kelly_multiplier: Fractional Kelly safety factor (0.5 = half-Kelly).

    Returns:
        Target position as fraction of available capital [0.0, 1.0].
        Returns 0.0 when Kelly is negative (model predicts net loss).

    Raises:
        ValueError: If win_loss_ratio <= 0.
    """
    if win_loss_ratio <= 0:
        raise ValueError(f"win_loss_ratio must be > 0, got {win_loss_ratio}")

    kelly_pct = (p_up * win_loss_ratio - p_down) / win_loss_ratio
    fractional = kelly_pct * kelly_multiplier
    return min(1.0, max(0.0, fractional))


def kelly_qty(
    kelly_frac: float,
    portfolio_value: float,
    close_price: float,
    cash_buffer_pct: float = 0.02,
) -> int:
    """Convert Kelly fraction to whole share count.

    Args:
        kelly_frac: Kelly fraction [0.0, 1.0] from kelly_fraction().
        portfolio_value: Total portfolio value in dollars.
        close_price: Current price per share.
        cash_buffer_pct: Fraction of portfolio reserved as cash buffer.

    Returns:
        Number of whole shares to hold (always >= 0).
    """
    available = portfolio_value * (1 - cash_buffer_pct)
    target_dollars = available * kelly_frac
    return math.floor(target_dollars / close_price)
