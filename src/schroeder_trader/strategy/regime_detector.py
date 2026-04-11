import enum
from pathlib import Path
from typing import Union

import joblib
import numpy as np
import pandas as pd
from hmmlearn import hmm


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


# ---------------------------------------------------------------------------
# HMMRegimeDetector
# ---------------------------------------------------------------------------

HMM_FEATURES = ["log_return_20d", "volatility_20d", "vix_close", "vix_term_structure"]


class HMMRegimeDetector:
    """Gaussian HMM-based regime detector with automatic state-count selection.

    States are labeled by sorting on mean ``log_return_20d``:
    lowest = BEAR, highest = BULL, middle states = CHOPPY (or CHOPPY_0/CHOPPY_1
    for 4-state models).
    """

    def __init__(self, n_states: int | None = None, random_state: int = 42) -> None:
        self._n_states_arg = n_states
        self.random_state = random_state
        self.n_states: int | None = n_states
        self.model_: hmm.GaussianHMM | None = None
        self.state_means_: np.ndarray | None = None
        self.label_to_state_: dict[str, int] = {}
        self.state_to_label_: dict[int, str] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """Return a (n_samples, n_features) float array of HMM_FEATURES columns."""
        return df[HMM_FEATURES].to_numpy(dtype=float)

    def _bic(self, model: hmm.GaussianHMM, X: np.ndarray) -> float:
        n_samples, n_features = X.shape
        n = model.n_components
        log_likelihood = model.score(X)
        # Transition matrix: n*(n-1) free params (rows sum to 1)
        # Means: n * n_features
        # Full covariance: n * n_features*(n_features+1)/2
        n_params = (
            n * n
            + n * n_features
            + n * n_features * (n_features + 1) // 2
        )
        return -2 * log_likelihood * n_samples + n_params * np.log(n_samples)

    def _fit_hmm(self, X: np.ndarray, n: int) -> hmm.GaussianHMM:
        model = hmm.GaussianHMM(
            n_components=n,
            covariance_type="full",
            n_iter=100,
            random_state=self.random_state,
        )
        model.fit(X)
        return model

    def _label_states(self) -> None:
        """Assign string labels to HMM states sorted by mean log_return_20d."""
        # Index 0 in feature array corresponds to log_return_20d
        mean_returns = self.state_means_[:, 0]
        sorted_states = np.argsort(mean_returns)  # ascending: lowest first

        n = self.n_states
        if n == 2:
            labels = ["BEAR", "BULL"]
        elif n == 3:
            labels = ["BEAR", "CHOPPY", "BULL"]
        else:  # 4 states
            labels = ["BEAR", "CHOPPY_0", "CHOPPY_1", "BULL"]

        self.label_to_state_ = {}
        self.state_to_label_ = {}
        for rank, state_idx in enumerate(sorted_states):
            label = labels[rank]
            self.label_to_state_[label] = int(state_idx)
            self.state_to_label_[int(state_idx)] = label

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_n_states(self, df: pd.DataFrame) -> int:
        """Fit HMMs with 2–4 states and return the n with the lowest BIC."""
        X = self._extract_features(df)
        best_n = 2
        best_bic = float("inf")
        for n in (2, 3, 4):
            try:
                model = self._fit_hmm(X, n)
                bic = self._bic(model, X)
                if bic < best_bic:
                    best_bic = bic
                    best_n = n
            except Exception:
                pass
        return best_n

    def fit(self, df: pd.DataFrame) -> "HMMRegimeDetector":
        """Fit the HMM on *df*, label states, and return self."""
        X = self._extract_features(df)

        if self._n_states_arg is None:
            self.n_states = self.select_n_states(df)
        else:
            self.n_states = self._n_states_arg

        self.model_ = self._fit_hmm(X, self.n_states)
        self.state_means_ = self.model_.means_
        self._label_states()
        return self

    def predict_proba(self, df: pd.DataFrame) -> list[dict[str, float]]:
        """Return per-row posterior state probabilities as a list of dicts.

        Each dict maps label string → probability.
        """
        if self.model_ is None:
            raise RuntimeError("Call fit() before predict_proba().")
        X = self._extract_features(df)
        # posteriors shape: (n_samples, n_states)
        _, posteriors = self.model_.score_samples(X)
        result = []
        for row in posteriors:
            result.append({
                self.state_to_label_[i]: float(row[i])
                for i in range(self.n_states)
            })
        return result

    def dominant_regime(self, probs: dict[str, float]) -> Regime:
        """Return the :class:`Regime` enum for the highest-probability label.

        CHOPPY_* variants map to ``Regime.CHOPPY``.
        """
        best_label = max(probs, key=probs.__getitem__)
        if best_label.startswith("CHOPPY"):
            return Regime.CHOPPY
        return Regime[best_label]

    def save(self, path: Union[str, Path]) -> None:
        """Serialize this detector to *path* using joblib."""
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "HMMRegimeDetector":
        """Load and return a serialized :class:`HMMRegimeDetector` from *path*."""
        return joblib.load(path)
