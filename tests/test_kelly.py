import pytest

from schroeder_trader.risk.kelly import kelly_fraction


def test_kelly_balanced_confidence():
    """P(UP)=0.6, P(DOWN)=0.2, b=1.5 → positive Kelly."""
    result = kelly_fraction(p_up=0.6, p_down=0.2, win_loss_ratio=1.5, kelly_multiplier=0.5)
    assert 0.23 < result < 0.24


def test_kelly_high_flat_prob_gives_near_zero():
    """P(UP)=0.05, P(DOWN)=0.05 → Kelly near zero."""
    result = kelly_fraction(p_up=0.05, p_down=0.05, win_loss_ratio=1.5, kelly_multiplier=0.5)
    assert result < 0.02


def test_kelly_negative_clamped_to_zero():
    """P(DOWN) > P(UP)*b → negative Kelly clamped to 0."""
    result = kelly_fraction(p_up=0.1, p_down=0.8, win_loss_ratio=1.0, kelly_multiplier=0.5)
    assert result == 0.0


def test_kelly_perfect_confidence():
    """P(UP)=1.0, P(DOWN)=0.0 → full Kelly multiplier."""
    result = kelly_fraction(p_up=1.0, p_down=0.0, win_loss_ratio=1.5, kelly_multiplier=0.5)
    assert result == 0.5


def test_kelly_zero_probs():
    """P(UP)=0, P(DOWN)=0 → 0."""
    result = kelly_fraction(p_up=0.0, p_down=0.0, win_loss_ratio=1.5, kelly_multiplier=0.5)
    assert result == 0.0


def test_kelly_clamped_to_one():
    """Full Kelly with extreme confidence clamped to 1.0."""
    result = kelly_fraction(p_up=1.0, p_down=0.0, win_loss_ratio=1.5, kelly_multiplier=1.5)
    assert result == 1.0


def test_kelly_invalid_win_loss_ratio():
    """win_loss_ratio <= 0 raises ValueError."""
    with pytest.raises(ValueError):
        kelly_fraction(p_up=0.5, p_down=0.3, win_loss_ratio=0.0, kelly_multiplier=0.5)
    with pytest.raises(ValueError):
        kelly_fraction(p_up=0.5, p_down=0.3, win_loss_ratio=-1.0, kelly_multiplier=0.5)


from schroeder_trader.risk.kelly import kelly_qty


def test_kelly_qty_basic():
    """50% Kelly fraction, $100k portfolio, SPY at $500."""
    qty = kelly_qty(kelly_frac=0.5, portfolio_value=100000, close_price=500, cash_buffer_pct=0.02)
    assert qty == 98


def test_kelly_qty_zero_fraction():
    qty = kelly_qty(kelly_frac=0.0, portfolio_value=100000, close_price=500)
    assert qty == 0


def test_kelly_qty_small_portfolio():
    """Kelly fraction too small to buy even 1 share."""
    qty = kelly_qty(kelly_frac=0.01, portfolio_value=1000, close_price=500)
    assert qty == 0
