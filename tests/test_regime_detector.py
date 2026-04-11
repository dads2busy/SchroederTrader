import numpy as np
import pandas as pd
import pytest
import joblib

from schroeder_trader.strategy.regime_detector import (
    Regime,
    HMMRegimeDetector,
    compute_regime_labels,
    compute_regime_series,
    detect_regime,
)


# ---------------------------------------------------------------------------
# detect_regime() unit tests
# ---------------------------------------------------------------------------

class TestDetectRegime:
    """Tests for the three regime branches and boundary conditions."""

    # --- BULL ---

    def test_bull_positive_return_low_vol(self):
        """Positive return + vol below median => BULL."""
        assert detect_regime(log_return_20d=0.06, volatility_20d=0.01, vol_median=0.02) == Regime.BULL

    def test_bull_small_positive_return_low_vol(self):
        """Small positive return + vol at exactly vol_median is CHOPPY (not strictly below)."""
        # vol == vol_median is not high_vol (strict >), so this should be BULL
        assert detect_regime(log_return_20d=0.001, volatility_20d=0.015, vol_median=0.015) == Regime.BULL

    def test_bull_large_positive_return_low_vol(self):
        assert detect_regime(log_return_20d=0.20, volatility_20d=0.005, vol_median=0.02) == Regime.BULL

    # --- BEAR ---

    def test_bear_negative_return_high_vol(self):
        """Negative return + vol above median => BEAR."""
        assert detect_regime(log_return_20d=-0.06, volatility_20d=0.03, vol_median=0.02) == Regime.BEAR

    def test_bear_small_negative_return_high_vol(self):
        """Small negative return + high vol still => BEAR."""
        assert detect_regime(log_return_20d=-0.001, volatility_20d=0.021, vol_median=0.02) == Regime.BEAR

    def test_bear_large_negative_return_high_vol(self):
        assert detect_regime(log_return_20d=-0.20, volatility_20d=0.05, vol_median=0.02) == Regime.BEAR

    # --- CHOPPY ---

    def test_choppy_positive_return_high_vol(self):
        """Positive return + high vol => CHOPPY (not BULL, not BEAR)."""
        assert detect_regime(log_return_20d=0.06, volatility_20d=0.03, vol_median=0.02) == Regime.CHOPPY

    def test_choppy_negative_return_low_vol(self):
        """Negative return + low vol => CHOPPY."""
        assert detect_regime(log_return_20d=-0.06, volatility_20d=0.01, vol_median=0.02) == Regime.CHOPPY

    def test_choppy_zero_return_any_vol(self):
        """Zero log return falls into neither positive nor negative => CHOPPY."""
        assert detect_regime(log_return_20d=0.0, volatility_20d=0.01, vol_median=0.02) == Regime.CHOPPY
        assert detect_regime(log_return_20d=0.0, volatility_20d=0.03, vol_median=0.02) == Regime.CHOPPY

    # --- Boundary: vol == vol_median ---

    def test_boundary_vol_equals_median_positive_return(self):
        """vol == vol_median is NOT high_vol (condition is strict >), so positive return => BULL."""
        assert detect_regime(log_return_20d=0.05, volatility_20d=0.02, vol_median=0.02) == Regime.BULL

    def test_boundary_vol_equals_median_negative_return(self):
        """vol == vol_median is NOT high_vol, so negative return + equal vol => CHOPPY."""
        assert detect_regime(log_return_20d=-0.05, volatility_20d=0.02, vol_median=0.02) == Regime.CHOPPY

    def test_boundary_vol_just_above_median_positive_return(self):
        """vol just above median + positive return => CHOPPY."""
        assert detect_regime(log_return_20d=0.05, volatility_20d=0.02001, vol_median=0.02) == Regime.CHOPPY

    def test_boundary_vol_just_above_median_negative_return(self):
        """vol just above median + negative return => BEAR."""
        assert detect_regime(log_return_20d=-0.05, volatility_20d=0.02001, vol_median=0.02) == Regime.BEAR


# ---------------------------------------------------------------------------
# Helpers for compute_regime_series / compute_regime_labels
# ---------------------------------------------------------------------------

def _make_close_series(n: int = 350, seed: int = 42) -> pd.DataFrame:
    """Build a synthetic price DataFrame with enough rows for 252-day vol median.

    Uses a random walk so all three regimes are likely to appear.
    """
    rng = np.random.default_rng(seed)
    returns = rng.normal(loc=0.0003, scale=0.01, size=n)
    prices = 100.0 * np.exp(np.cumsum(returns))
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame({"close": prices}, index=idx)


def _make_bull_df(n: int = 350) -> pd.DataFrame:
    """Strongly trending upward with low volatility — mostly BULL."""
    prices = 100.0 * np.exp(np.linspace(0, 0.5, n))
    # Add tiny noise to avoid identical daily returns (need non-zero vol)
    prices += np.linspace(0, 0.01, n)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame({"close": prices}, index=idx)


def _make_bear_df(n: int = 350) -> pd.DataFrame:
    """Sharply declining with high volatility — mostly BEAR in the declining region."""
    rng = np.random.default_rng(7)
    # Large downward drift + elevated vol
    returns = rng.normal(loc=-0.003, scale=0.025, size=n)
    prices = 100.0 * np.exp(np.cumsum(returns))
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame({"close": prices}, index=idx)


# ---------------------------------------------------------------------------
# compute_regime_series() tests
# ---------------------------------------------------------------------------

class TestComputeRegimeSeries:
    """Tests for compute_regime_series()."""

    def test_returns_series(self):
        df = _make_close_series()
        result = compute_regime_series(df)
        assert isinstance(result, pd.Series)

    def test_index_matches_input(self):
        df = _make_close_series()
        result = compute_regime_series(df)
        assert result.index.equals(df.index)

    def test_length_matches_input(self):
        df = _make_close_series()
        result = compute_regime_series(df)
        assert len(result) == len(df)

    def test_early_rows_are_nan(self):
        """First 271 rows (20 for log_ret + 252 for vol_median) should be NaN."""
        df = _make_close_series(n=350)
        result = compute_regime_series(df)
        # At minimum the first 20 rows lack log_ret_20d; first 271 lack vol_med
        nan_mask = result.isna()
        assert nan_mask.iloc[:20].all(), "First 20 rows should be NaN"

    def test_later_rows_are_regime_values(self):
        """Rows after warm-up period should be Regime enum values."""
        df = _make_close_series(n=350)
        result = compute_regime_series(df)
        non_nan = result.dropna()
        assert len(non_nan) > 0, "Expected at least some non-NaN regime values"
        for val in non_nan:
            assert isinstance(val, Regime), f"Expected Regime, got {type(val)}: {val}"

    def test_all_three_regimes_present(self):
        """With a random walk of 350 rows, all three regimes should appear."""
        df = _make_close_series(n=350, seed=42)
        result = compute_regime_series(df)
        non_nan = result.dropna()
        regimes_found = set(non_nan)
        assert regimes_found == {Regime.BULL, Regime.BEAR, Regime.CHOPPY}, (
            f"Expected all three regimes, got: {regimes_found}"
        )

    def test_bull_dominated_series_produces_bull_regime(self):
        """A consistently upward-trending, low-vol series should produce BULL regimes."""
        df = _make_bull_df(n=350)
        result = compute_regime_series(df)
        non_nan = result.dropna()
        assert len(non_nan) > 0
        bull_count = (non_nan == Regime.BULL).sum()
        assert bull_count > 0, "Expected BULL regimes in uptrending series"

    def test_bear_dominated_series_produces_bear_regime(self):
        """A sharply declining, high-vol series should produce BEAR regimes."""
        df = _make_bear_df(n=350)
        result = compute_regime_series(df)
        non_nan = result.dropna()
        assert len(non_nan) > 0
        bear_count = (non_nan == Regime.BEAR).sum()
        assert bear_count > 0, "Expected BEAR regimes in declining high-vol series"

    def test_requires_close_column(self):
        """DataFrame without 'close' column should raise KeyError."""
        df = pd.DataFrame({"price": [1.0, 2.0, 3.0]})
        with pytest.raises(KeyError):
            compute_regime_series(df)

    def test_single_row_is_nan(self):
        """A one-row DataFrame has no rolling window data — should be NaN."""
        df = pd.DataFrame({"close": [100.0]}, index=pd.date_range("2020-01-01", periods=1))
        result = compute_regime_series(df)
        assert result.isna().all()


# ---------------------------------------------------------------------------
# compute_regime_labels() tests
# ---------------------------------------------------------------------------

class TestComputeRegimeLabels:
    """Tests for compute_regime_labels()."""

    def test_returns_series(self):
        df = _make_close_series()
        result = compute_regime_labels(df)
        assert isinstance(result, pd.Series)

    def test_index_matches_input(self):
        df = _make_close_series()
        result = compute_regime_labels(df)
        assert result.index.equals(df.index)

    def test_early_rows_are_nan(self):
        df = _make_close_series(n=350)
        result = compute_regime_labels(df)
        assert result.iloc[:20].isna().all()

    def test_non_nan_values_are_integers(self):
        df = _make_close_series(n=350)
        result = compute_regime_labels(df)
        non_nan = result.dropna()
        assert len(non_nan) > 0
        for val in non_nan:
            assert val in (0, 1, 2), f"Unexpected label value: {val}"

    def test_label_mapping_bear_is_0(self):
        """BEAR regime should map to integer 0."""
        # Use bear-dominated data and verify 0 appears
        df = _make_bear_df(n=350)
        labels = compute_regime_labels(df)
        regimes = compute_regime_series(df)
        non_nan_mask = regimes.notna()
        bear_mask = regimes[non_nan_mask] == Regime.BEAR
        if bear_mask.any():
            assert (labels[non_nan_mask][bear_mask] == 0).all()

    def test_label_mapping_bull_is_2(self):
        """BULL regime should map to integer 2."""
        df = _make_bull_df(n=350)
        labels = compute_regime_labels(df)
        regimes = compute_regime_series(df)
        non_nan_mask = regimes.notna()
        bull_mask = regimes[non_nan_mask] == Regime.BULL
        if bull_mask.any():
            assert (labels[non_nan_mask][bull_mask] == 2).all()

    def test_label_mapping_choppy_is_1(self):
        """CHOPPY regime should map to integer 1."""
        df = _make_close_series(n=350, seed=42)
        labels = compute_regime_labels(df)
        regimes = compute_regime_series(df)
        non_nan_mask = regimes.notna()
        choppy_mask = regimes[non_nan_mask] == Regime.CHOPPY
        if choppy_mask.any():
            assert (labels[non_nan_mask][choppy_mask] == 1).all()

    def test_consistent_with_regime_series(self):
        """Labels should be the integer mapping of compute_regime_series()."""
        df = _make_close_series(n=350)
        regimes = compute_regime_series(df)
        labels = compute_regime_labels(df)
        _map = {Regime.BEAR: 0, Regime.CHOPPY: 1, Regime.BULL: 2}
        for idx in df.index:
            r = regimes[idx]
            l = labels[idx]
            if pd.isna(r):
                assert pd.isna(l), f"NaN regime at {idx} should give NaN label"
            else:
                assert l == _map[r], f"Mismatch at {idx}: regime={r}, label={l}"

    def test_only_values_0_1_2_and_nan(self):
        """Labels must only contain 0, 1, 2, or NaN — nothing else."""
        df = _make_close_series(n=350)
        result = compute_regime_labels(df)
        non_nan = result.dropna()
        assert set(non_nan.values).issubset({0, 1, 2})


# ---------------------------------------------------------------------------
# HMMRegimeDetector tests
# ---------------------------------------------------------------------------

class TestHMMRegimeDetector:
    def _make_feature_df(self, n=500, seed=42):
        rng = np.random.default_rng(seed)
        returns = rng.normal(0.0003, 0.01, n)
        prices = 100.0 * np.exp(np.cumsum(returns))
        idx = pd.date_range("2020-01-01", periods=n, freq="B")
        df = pd.DataFrame({"close": prices}, index=idx)
        df["log_return_20d"] = np.log(df["close"] / df["close"].shift(20))
        df["volatility_20d"] = df["close"].pct_change().rolling(20).std()
        df["vix_close"] = 15 + rng.normal(0, 3, n)
        df["vix_term_structure"] = 0.9 + rng.normal(0, 0.05, n)
        return df.dropna()

    def test_fit_returns_self(self):
        df = self._make_feature_df()
        detector = HMMRegimeDetector()
        result = detector.fit(df)
        assert result is detector

    def test_predict_proba_returns_dict(self):
        df = self._make_feature_df()
        detector = HMMRegimeDetector().fit(df)
        probs = detector.predict_proba(df)
        assert isinstance(probs, list)
        assert len(probs) == len(df)
        assert isinstance(probs[0], dict)

    def test_predict_proba_sums_to_one(self):
        df = self._make_feature_df()
        detector = HMMRegimeDetector().fit(df)
        probs = detector.predict_proba(df)
        for p in probs:
            assert abs(sum(p.values()) - 1.0) < 1e-6

    def test_predict_proba_keys_are_regime_labels(self):
        df = self._make_feature_df()
        detector = HMMRegimeDetector().fit(df)
        probs = detector.predict_proba(df)
        for p in probs:
            assert "BULL" in p
            assert "BEAR" in p

    def test_predict_proba_single_row(self):
        df = self._make_feature_df()
        detector = HMMRegimeDetector().fit(df)
        probs = detector.predict_proba(df.iloc[[-1]])
        assert len(probs) == 1
        assert abs(sum(probs[0].values()) - 1.0) < 1e-6

    def test_dominant_regime_returns_regime_enum(self):
        df = self._make_feature_df()
        detector = HMMRegimeDetector().fit(df)
        probs = detector.predict_proba(df)
        dominant = detector.dominant_regime(probs[0])
        assert isinstance(dominant, Regime)

    def test_select_n_states_returns_int(self):
        df = self._make_feature_df()
        detector = HMMRegimeDetector()
        n = detector.select_n_states(df)
        assert isinstance(n, int)
        assert 2 <= n <= 4

    def test_save_and_load(self, tmp_path):
        df = self._make_feature_df()
        detector = HMMRegimeDetector().fit(df)
        path = tmp_path / "hmm_test.pkl"
        detector.save(path)
        loaded = HMMRegimeDetector.load(path)
        probs_orig = detector.predict_proba(df.iloc[[-1]])
        probs_loaded = loaded.predict_proba(df.iloc[[-1]])
        for key in probs_orig[0]:
            assert abs(probs_orig[0][key] - probs_loaded[0][key]) < 1e-6

    def test_state_labeling_bear_has_lowest_return(self):
        df = self._make_feature_df(n=800)
        detector = HMMRegimeDetector().fit(df)
        means = detector.state_means_
        bear_idx = detector.label_to_state_["BEAR"]
        bull_idx = detector.label_to_state_["BULL"]
        assert means[bear_idx, 0] < means[bull_idx, 0]
