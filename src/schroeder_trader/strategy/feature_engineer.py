import numpy as np
import pandas as pd

from schroeder_trader.config import SMA_SHORT_WINDOW, SMA_LONG_WINDOW

# Label encoding: XGBoost requires 0-indexed contiguous integers
CLASS_DOWN = 0
CLASS_FLAT = 1
CLASS_UP = 2

CLASS_NAMES = {CLASS_DOWN: "DOWN", CLASS_FLAT: "FLAT", CLASS_UP: "UP"}

# Threshold for classifying 5-day forward returns
RETURN_THRESHOLD = 0.005  # 0.5%


class FeaturePipeline:
    """Compute ML features from OHLCV data."""

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute features for live prediction (no label).

        Args:
            df: DataFrame with open, high, low, close, volume columns.

        Returns:
            DataFrame with feature columns added, NaN rows dropped.
        """
        result = df.copy()

        # Momentum
        result["log_return_5d"] = np.log(result["close"] / result["close"].shift(5))
        result["log_return_20d"] = np.log(result["close"] / result["close"].shift(20))

        # Risk regime
        result["volatility_20d"] = result["close"].pct_change().rolling(20).std()

        # Trend strength
        sma_short = result["close"].rolling(SMA_SHORT_WINDOW).mean()
        sma_long = result["close"].rolling(SMA_LONG_WINDOW).mean()
        result["sma_ratio"] = sma_short / sma_long

        # Activity
        result["volume_ratio"] = result["volume"] / result["volume"].rolling(20).mean()

        # Mean-reversion (RSI 14)
        result["rsi_14"] = self._compute_rsi(result["close"], 14)

        # Drop rows with NaN features
        feature_cols = [
            "log_return_5d", "log_return_20d", "volatility_20d",
            "sma_ratio", "volume_ratio", "rsi_14",
        ]
        result = result.dropna(subset=feature_cols)

        return result

    def compute_features_extended(
        self, spy_df: pd.DataFrame, ext_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute the 6-feature set used by the composite model.

        Features: log_return_5d, log_return_20d, volatility_20d,
                  credit_spread, dollar_momentum, regime_label (added later).

        Args:
            spy_df: SPY OHLCV DataFrame (DatetimeIndex).
            ext_df: External features DataFrame with at least
                    ``credit_spread`` and ``dollar_momentum`` columns
                    (DatetimeIndex named "date").

        Returns:
            DataFrame with feature columns; NaN rows dropped
            (except ``regime_label`` which is added by the caller).
        """
        result = spy_df.copy()

        # Normalize index: strip timezone so join with tz-naive ext_df works
        if hasattr(result.index, "tz") and result.index.tz is not None:
            result.index = result.index.tz_localize(None)
        # Normalize to date-only (daily resolution) to match ext_df
        result.index = result.index.normalize()

        # Momentum
        result["log_return_5d"] = np.log(result["close"] / result["close"].shift(5))
        result["log_return_20d"] = np.log(result["close"] / result["close"].shift(20))

        # Risk regime
        result["volatility_20d"] = result["close"].pct_change().rolling(20).std()

        # Merge external features (forward-fill to cover market holidays)
        if ext_df is not None and len(ext_df) > 0:
            ext_cols = [c for c in ("credit_spread", "dollar_momentum", "vix_close", "vix3m_close") if c in ext_df.columns]
            if ext_cols:
                ext = ext_df[ext_cols].copy()
                if hasattr(ext.index, "tz") and ext.index.tz is not None:
                    ext.index = ext.index.tz_localize(None)
                ext.index = ext.index.normalize()
                result = result.join(ext, how="left")
                result[ext_cols] = result[ext_cols].ffill()

        # VIX term structure: VIX / VIX3M (>1 = backwardation/fear, <1 = contango)
        if "vix_close" in result.columns and "vix3m_close" in result.columns:
            result["vix_term_structure"] = result["vix_close"] / result["vix3m_close"]

        # Drop rows where core features are NaN
        core_cols = [
            "log_return_5d", "log_return_20d", "volatility_20d",
            "credit_spread", "dollar_momentum",
        ]
        existing_cols = [c for c in core_cols if c in result.columns]
        result = result.dropna(subset=existing_cols)

        return result

    def compute_features_with_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute features + forward-looking label for training.

        Args:
            df: DataFrame with open, high, low, close, volume columns.

        Returns:
            DataFrame with features and forward_return_5d_class column.
            Trailing rows where the 5-day forward return cannot be computed are dropped.
        """
        result = self.compute_features(df)

        # Forward 5-day return (uses future data — training only!)
        forward_return = result["close"].shift(-5) / result["close"] - 1

        # Drop trailing rows where forward return is NaN (last 5 rows)
        result = result[forward_return.notna()].copy()
        forward_return = forward_return[forward_return.notna()]

        # Classify
        result["forward_return_5d_class"] = CLASS_FLAT  # default
        result.loc[forward_return > RETURN_THRESHOLD, "forward_return_5d_class"] = CLASS_UP
        result.loc[forward_return < -RETURN_THRESHOLD, "forward_return_5d_class"] = CLASS_DOWN
        result["forward_return_5d_class"] = result["forward_return_5d_class"].astype(int)

        return result

    @staticmethod
    def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Compute RSI (Relative Strength Index)."""
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
