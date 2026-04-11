import numpy as np
import pandas as pd

from schroeder_trader.strategy.feature_engineer import FeaturePipeline

FEATURE_COLUMNS = [
    "log_return_5d",
    "log_return_20d",
    "volatility_20d",
    "sma_ratio",
    "volume_ratio",
    "rsi_14",
]


def _make_ohlcv(n: int = 300) -> pd.DataFrame:
    """Create synthetic OHLCV data with a trend."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "open": close - np.random.uniform(0, 1, n),
        "high": close + np.random.uniform(0, 2, n),
        "low": close - np.random.uniform(0, 2, n),
        "close": close,
        "volume": np.random.randint(1_000_000, 10_000_000, n).astype(float),
    }, index=dates)


def test_compute_features_returns_all_columns():
    df = _make_ohlcv(300)
    pipeline = FeaturePipeline()
    result = pipeline.compute_features(df)
    for col in FEATURE_COLUMNS:
        assert col in result.columns, f"Missing feature: {col}"


def test_compute_features_drops_nan_rows():
    df = _make_ohlcv(300)
    pipeline = FeaturePipeline()
    result = pipeline.compute_features(df)
    assert not result[FEATURE_COLUMNS].isna().any().any()
    # Should lose ~200 rows to SMA_200 warmup
    assert len(result) < len(df)
    assert len(result) > 50


def test_compute_features_no_label_column():
    df = _make_ohlcv(300)
    pipeline = FeaturePipeline()
    result = pipeline.compute_features(df)
    assert "forward_return_5d_class" not in result.columns


def test_compute_features_with_labels_has_label():
    df = _make_ohlcv(300)
    pipeline = FeaturePipeline()
    result = pipeline.compute_features_with_labels(df)
    assert "forward_return_5d_class" in result.columns


def test_label_encoding_values():
    df = _make_ohlcv(300)
    pipeline = FeaturePipeline()
    result = pipeline.compute_features_with_labels(df)
    labels = result["forward_return_5d_class"].unique()
    # All labels should be 0, 1, or 2
    assert set(labels).issubset({0, 1, 2})


def test_labels_drop_trailing_rows():
    df = _make_ohlcv(300)
    pipeline = FeaturePipeline()
    features_only = pipeline.compute_features(df)
    with_labels = pipeline.compute_features_with_labels(df)
    # With-labels should have fewer rows (last 5 dropped)
    assert len(with_labels) <= len(features_only) - 5


def test_sma_ratio_positive():
    df = _make_ohlcv(300)
    pipeline = FeaturePipeline()
    result = pipeline.compute_features(df)
    assert (result["sma_ratio"] > 0).all()


def test_volume_ratio_positive():
    df = _make_ohlcv(300)
    pipeline = FeaturePipeline()
    result = pipeline.compute_features(df)
    assert (result["volume_ratio"] > 0).all()


def test_rsi_between_0_and_100():
    df = _make_ohlcv(300)
    pipeline = FeaturePipeline()
    result = pipeline.compute_features(df)
    assert (result["rsi_14"] >= 0).all()
    assert (result["rsi_14"] <= 100).all()


def test_compute_features_extended_includes_vix_term_structure():
    """VIX term structure (VIX/VIX3M) should be computed when both columns exist."""
    spy_df = pd.DataFrame(
        {"close": np.linspace(100, 110, 50), "open": 100, "high": 110, "low": 90, "volume": 1000},
        index=pd.date_range("2020-01-01", periods=50, freq="B"),
    )
    ext_df = pd.DataFrame(
        {
            "credit_spread": np.random.default_rng(42).normal(0, 0.01, 50),
            "dollar_momentum": np.random.default_rng(42).normal(0, 0.01, 50),
            "vix_close": np.full(50, 20.0),
            "vix3m_close": np.full(50, 22.0),
        },
        index=pd.date_range("2020-01-01", periods=50, freq="B", name="date"),
    )
    pipeline = FeaturePipeline()
    result = pipeline.compute_features_extended(spy_df, ext_df)
    assert "vix_term_structure" in result.columns
    assert abs(result["vix_term_structure"].iloc[-1] - 20.0 / 22.0) < 0.01


def test_compute_features_extended_no_vix_columns():
    """Without VIX columns, vix_term_structure should not be added."""
    spy_df = pd.DataFrame(
        {"close": np.linspace(100, 110, 50), "open": 100, "high": 110, "low": 90, "volume": 1000},
        index=pd.date_range("2020-01-01", periods=50, freq="B"),
    )
    ext_df = pd.DataFrame(
        {
            "credit_spread": np.random.default_rng(42).normal(0, 0.01, 50),
            "dollar_momentum": np.random.default_rng(42).normal(0, 0.01, 50),
        },
        index=pd.date_range("2020-01-01", periods=50, freq="B", name="date"),
    )
    pipeline = FeaturePipeline()
    result = pipeline.compute_features_extended(spy_df, ext_df)
    assert "vix_term_structure" not in result.columns
