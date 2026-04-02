def estimate_slippage(vix: float) -> float:
    """Estimate transaction cost based on VIX level.

    Returns slippage as a fraction of trade value.

    Tiers:
        VIX < 15:  0.0003 (3 bps)  — low vol, tight spreads
        VIX 15-25: 0.0005 (5 bps)  — normal conditions
        VIX 25-35: 0.0010 (10 bps) — elevated vol, wider spreads
        VIX >= 35: 0.0015 (15 bps) — crisis, wide spreads
    """
    if vix < 15:
        return 0.0003
    if vix < 25:
        return 0.0005
    if vix < 35:
        return 0.0010
    return 0.0015
