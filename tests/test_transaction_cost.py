from schroeder_trader.risk.transaction_cost import estimate_slippage


def test_low_vix():
    """VIX < 15 -> 3 bps."""
    assert estimate_slippage(10.0) == 0.0003
    assert estimate_slippage(14.99) == 0.0003


def test_normal_vix():
    """VIX 15-25 -> 5 bps."""
    assert estimate_slippage(15.0) == 0.0005
    assert estimate_slippage(20.0) == 0.0005
    assert estimate_slippage(24.99) == 0.0005


def test_elevated_vix():
    """VIX 25-35 -> 10 bps."""
    assert estimate_slippage(25.0) == 0.0010
    assert estimate_slippage(30.0) == 0.0010
    assert estimate_slippage(34.99) == 0.0010


def test_crisis_vix():
    """VIX > 35 -> 15 bps."""
    assert estimate_slippage(35.0) == 0.0015
    assert estimate_slippage(80.0) == 0.0015


def test_boundary_values():
    """Exact boundary values fall in the higher tier."""
    assert estimate_slippage(15.0) == 0.0005  # not 0.0003
    assert estimate_slippage(25.0) == 0.0010  # not 0.0005
    assert estimate_slippage(35.0) == 0.0015  # not 0.0010
