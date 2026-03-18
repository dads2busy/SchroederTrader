import pandas as pd
import numpy as np

from schroeder_trader.strategy.sma_crossover import Signal, generate_signal


def _make_df(prices: list[float]) -> pd.DataFrame:
    """Create a DataFrame with the given closing prices."""
    return pd.DataFrame({
        "close": prices,
        "open": prices,
        "high": prices,
        "low": prices,
        "volume": [1000000] * len(prices),
    }, index=pd.date_range("2020-01-01", periods=len(prices), freq="B"))


def test_hold_when_sma50_above_sma200_no_crossover():
    prices = [100.0] * 200 + [110.0] * 50
    df = _make_df(prices)
    signal, sma_50, sma_200 = generate_signal(df)
    assert signal == Signal.HOLD


def test_buy_on_golden_cross():
    prices = [100.0] * 200
    for i in range(50):
        prices.append(100.0 + (i + 1) * 1.0)
    df = _make_df(prices)
    signal, sma_50, sma_200 = generate_signal(df)
    assert sma_50 > sma_200
    sma_50_prev = df["close"].iloc[:-1].tail(50).mean()
    sma_200_prev = df["close"].iloc[:-1].tail(200).mean()
    if sma_50_prev <= sma_200_prev:
        assert signal == Signal.BUY
    else:
        assert signal == Signal.HOLD


def test_sell_on_death_cross():
    prices = [120.0] * 200
    for i in range(50):
        prices.append(120.0 - (i + 1) * 1.5)
    df = _make_df(prices)
    signal, sma_50, sma_200 = generate_signal(df)
    assert sma_50 < sma_200
    sma_50_prev = df["close"].iloc[:-1].tail(50).mean()
    sma_200_prev = df["close"].iloc[:-1].tail(200).mean()
    if sma_50_prev >= sma_200_prev:
        assert signal == Signal.SELL
    else:
        assert signal == Signal.HOLD


def test_hold_when_no_crossover_sma50_below():
    prices = [100.0] * 200 + [90.0] * 50
    df = _make_df(prices)
    signal, sma_50, sma_200 = generate_signal(df)
    assert signal == Signal.HOLD


def test_requires_minimum_200_bars():
    prices = [100.0] * 100
    df = _make_df(prices)
    try:
        generate_signal(df)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "200" in str(e)


def test_returns_sma_values():
    prices = [100.0] * 250
    df = _make_df(prices)
    signal, sma_50, sma_200 = generate_signal(df)
    assert isinstance(sma_50, float)
    assert isinstance(sma_200, float)
    assert abs(sma_50 - 100.0) < 0.01
    assert abs(sma_200 - 100.0) < 0.01
