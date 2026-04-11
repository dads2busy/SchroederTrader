# HMM Regime Detector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the threshold-based regime detector with a Gaussian HMM trained on returns, volatility, VIX, and VIX term structure, with probability-based blended signal routing.

**Architecture:** A new `HMMRegimeDetector` class wraps `hmmlearn.GaussianHMM`, fits on 4 features, selects state count via BIC, and outputs per-state probabilities. A new `composite_signal_blended()` function uses those probabilities to produce a continuous target exposure (0.0-0.98) instead of discrete Signal enums. The existing threshold detector and hard-switching router remain unchanged for backward compatibility.

**Tech Stack:** hmmlearn, joblib (for model persistence), existing pandas/numpy/xgboost stack.

**Spec:** `docs/superpowers/specs/2026-04-11-hmm-regime-detector-design.md`

---

### Task 1: Add hmmlearn dependency

**Files:**
- Modify: `pyproject.toml:6-14`

- [ ] **Step 1: Add hmmlearn to dependencies**

In `pyproject.toml`, add `hmmlearn` to the dependencies list:

```toml
dependencies = [
    "alpaca-py>=0.33.0",
    "yfinance>=0.2.40",
    "pandas>=2.2.0",
    "python-dotenv>=1.0.0",
    "xgboost>=2.0.0",
    "scikit-learn>=1.4.0",
    "anthropic>=0.88.0",
    "hmmlearn>=0.3.0",
]
```

- [ ] **Step 2: Install**

Run: `uv pip install -e ".[dev,backtest]"`
Expected: hmmlearn installs successfully.

- [ ] **Step 3: Verify import**

Run: `uv run python -c "from hmmlearn.hmm import GaussianHMM; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "chore: add hmmlearn dependency"
```

---

### Task 2: Add VIX term structure feature

**Files:**
- Modify: `src/schroeder_trader/strategy/feature_engineer.py:58-110`
- Test: `tests/test_feature_engineer.py`

- [ ] **Step 1: Write failing test for vix_term_structure**

Add to `tests/test_feature_engineer.py`:

```python
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
    # VIX=20, VIX3M=22 => term structure = 20/22 ≈ 0.909
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_feature_engineer.py -v -k "vix_term"`
Expected: FAIL — `vix_term_structure` not in columns.

- [ ] **Step 3: Implement vix_term_structure in compute_features_extended**

In `src/schroeder_trader/strategy/feature_engineer.py`, in `compute_features_extended()`, after the external feature merge block (after line 100), add:

```python
        # VIX term structure: VIX / VIX3M (>1 = backwardation/fear, <1 = contango)
        if "vix_close" in result.columns and "vix3m_close" in result.columns:
            result["vix_term_structure"] = result["vix_close"] / result["vix3m_close"]
```

Also update the `ext_cols` list on line 93 to include the VIX columns:

```python
            ext_cols = [c for c in ("credit_spread", "dollar_momentum", "vix_close", "vix3m_close") if c in ext_df.columns]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_feature_engineer.py -v`
Expected: All pass, including new vix_term_structure tests.

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest tests/ -q`
Expected: 149+ passed, no failures.

- [ ] **Step 6: Commit**

```bash
git add src/schroeder_trader/strategy/feature_engineer.py tests/test_feature_engineer.py
git commit -m "feat: add VIX term structure feature to extended pipeline"
```

---

### Task 3: Implement HMMRegimeDetector class

**Files:**
- Modify: `src/schroeder_trader/strategy/regime_detector.py`
- Test: `tests/test_regime_detector.py`

- [ ] **Step 1: Write failing tests for HMMRegimeDetector**

Add to `tests/test_regime_detector.py`:

```python
import joblib
from schroeder_trader.strategy.regime_detector import HMMRegimeDetector


class TestHMMRegimeDetector:
    """Tests for the HMM-based regime detector."""

    def _make_feature_df(self, n=500, seed=42):
        """Build synthetic features with the 4 HMM input columns."""
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
            # CHOPPY may or may not be present depending on BIC state count

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
        # Predict with loaded model should give same results
        probs_orig = detector.predict_proba(df.iloc[[-1]])
        probs_loaded = loaded.predict_proba(df.iloc[[-1]])
        for key in probs_orig[0]:
            assert abs(probs_orig[0][key] - probs_loaded[0][key]) < 1e-6

    def test_state_labeling_bear_has_lowest_return(self):
        """BEAR state should have the lowest mean log_return_20d."""
        df = self._make_feature_df(n=800)
        detector = HMMRegimeDetector().fit(df)
        # The state labeled BEAR should correspond to the lowest mean return
        means = detector.state_means_
        bear_idx = detector.label_to_state_["BEAR"]
        bull_idx = detector.label_to_state_["BULL"]
        # log_return_20d is feature index 0
        assert means[bear_idx, 0] < means[bull_idx, 0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_regime_detector.py -v -k "HMM"`
Expected: FAIL — `HMMRegimeDetector` not importable.

- [ ] **Step 3: Implement HMMRegimeDetector**

Add to `src/schroeder_trader/strategy/regime_detector.py`:

```python
import joblib
from hmmlearn.hmm import GaussianHMM


# Features used by the HMM detector (order matters — matches training)
HMM_FEATURES = ["log_return_20d", "volatility_20d", "vix_close", "vix_term_structure"]


class HMMRegimeDetector:
    """Gaussian HMM-based regime detector with BIC state selection."""

    def __init__(self, n_states: int | None = None, random_state: int = 42):
        self.n_states = n_states
        self.random_state = random_state
        self.model_: GaussianHMM | None = None
        self.state_means_: np.ndarray | None = None
        self.label_to_state_: dict[str, int] = {}
        self.state_to_label_: dict[int, str] = {}

    def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract HMM feature matrix from DataFrame."""
        return df[HMM_FEATURES].values

    def select_n_states(self, df: pd.DataFrame) -> int:
        """Select optimal state count by BIC over 2, 3, 4 states."""
        X = self._extract_features(df)
        best_bic = np.inf
        best_n = 3  # fallback

        for n in [2, 3, 4]:
            try:
                model = GaussianHMM(
                    n_components=n, covariance_type="full",
                    n_iter=100, random_state=self.random_state,
                )
                model.fit(X)
                log_likelihood = model.score(X)
                n_params = n * n + n * len(HMM_FEATURES) + n * len(HMM_FEATURES) * (len(HMM_FEATURES) + 1) // 2
                bic = -2 * log_likelihood * len(X) + n_params * np.log(len(X))
                if bic < best_bic:
                    best_bic = bic
                    best_n = n
            except Exception:
                continue

        return best_n

    def fit(self, df: pd.DataFrame) -> "HMMRegimeDetector":
        """Fit the HMM on feature data."""
        X = self._extract_features(df)

        if self.n_states is None:
            self.n_states = self.select_n_states(df)

        self.model_ = GaussianHMM(
            n_components=self.n_states, covariance_type="full",
            n_iter=100, random_state=self.random_state,
        )
        self.model_.fit(X)
        self.state_means_ = self.model_.means_

        self._label_states()
        return self

    def _label_states(self) -> None:
        """Label states by sorting on mean log_return_20d (feature index 0)."""
        mean_returns = self.state_means_[:, 0]  # log_return_20d is first feature
        sorted_indices = np.argsort(mean_returns)

        n = self.n_states
        self.label_to_state_ = {}
        self.state_to_label_ = {}

        # Lowest return = BEAR, highest = BULL
        self.label_to_state_["BEAR"] = int(sorted_indices[0])
        self.state_to_label_[int(sorted_indices[0])] = "BEAR"

        self.label_to_state_["BULL"] = int(sorted_indices[-1])
        self.state_to_label_[int(sorted_indices[-1])] = "BULL"

        # Middle states are CHOPPY (or CHOPPY_0, CHOPPY_1 for 4 states)
        for i, idx in enumerate(sorted_indices[1:-1]):
            label = "CHOPPY" if n <= 3 else f"CHOPPY_{i}"
            self.label_to_state_[label] = int(idx)
            self.state_to_label_[int(idx)] = label

    def predict_proba(self, df: pd.DataFrame) -> list[dict[str, float]]:
        """Predict regime probabilities for each row.

        Returns:
            List of dicts mapping regime label ("BULL", "BEAR", "CHOPPY", etc.)
            to probability. One dict per row.
        """
        X = self._extract_features(df)
        # posteriors is (n_samples, n_states)
        posteriors = self.model_.predict_proba(X)

        result = []
        for row in posteriors:
            prob_dict = {}
            for state_idx, prob in enumerate(row):
                label = self.state_to_label_[state_idx]
                prob_dict[label] = float(prob)
            result.append(prob_dict)
        return result

    def dominant_regime(self, probs: dict[str, float]) -> Regime:
        """Return the Regime enum for the highest-probability state."""
        top_label = max(probs, key=probs.get)
        # Map CHOPPY_0, CHOPPY_1 etc back to Regime.CHOPPY
        if top_label.startswith("CHOPPY"):
            return Regime.CHOPPY
        return Regime[top_label]

    def save(self, path) -> None:
        """Persist the fitted detector to disk."""
        state = {
            "model": self.model_,
            "n_states": self.n_states,
            "state_means": self.state_means_,
            "label_to_state": self.label_to_state_,
            "state_to_label": self.state_to_label_,
            "random_state": self.random_state,
        }
        joblib.dump(state, path)

    @classmethod
    def load(cls, path) -> "HMMRegimeDetector":
        """Load a fitted detector from disk."""
        state = joblib.load(path)
        detector = cls(
            n_states=state["n_states"],
            random_state=state["random_state"],
        )
        detector.model_ = state["model"]
        detector.state_means_ = state["state_means"]
        detector.label_to_state_ = state["label_to_state"]
        detector.state_to_label_ = state["state_to_label"]
        return detector
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_regime_detector.py -v`
Expected: All existing threshold tests pass + all new HMM tests pass.

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest tests/ -q`
Expected: All pass.

- [ ] **Step 6: Commit**

```bash
git add src/schroeder_trader/strategy/regime_detector.py tests/test_regime_detector.py
git commit -m "feat: add HMMRegimeDetector with BIC state selection"
```

---

### Task 4: Implement blended composite router

**Files:**
- Modify: `src/schroeder_trader/strategy/composite.py`
- Test: `tests/test_composite.py`

- [ ] **Step 1: Write failing tests for composite_signal_blended**

Add to `tests/test_composite.py`:

```python
from schroeder_trader.strategy.composite import composite_signal_blended


class TestCompositeSignalBlended:
    """Tests for probability-weighted blended routing."""

    def test_pure_bull_sma_buy(self):
        """100% BULL with SMA BUY => 0.98 exposure."""
        exposure = composite_signal_blended(
            regime_probs={"BULL": 1.0, "BEAR": 0.0, "CHOPPY": 0.0},
            sma_signal=Signal.BUY,
            xgb_signal=Signal.SELL,
            bear_weakening=False,
            current_exposure=0.0,
        )
        assert abs(exposure - 0.98) < 1e-6

    def test_pure_bear_no_weakening(self):
        """100% BEAR without weakening => 0.0 exposure."""
        exposure = composite_signal_blended(
            regime_probs={"BULL": 0.0, "BEAR": 1.0, "CHOPPY": 0.0},
            sma_signal=Signal.BUY,
            xgb_signal=Signal.BUY,
            bear_weakening=False,
            current_exposure=0.5,
        )
        assert abs(exposure - 0.0) < 1e-6

    def test_pure_bear_with_weakening_xgb_buy(self):
        """100% BEAR with weakening + XGB BUY => 0.98."""
        exposure = composite_signal_blended(
            regime_probs={"BULL": 0.0, "BEAR": 1.0, "CHOPPY": 0.0},
            sma_signal=Signal.HOLD,
            xgb_signal=Signal.BUY,
            bear_weakening=True,
            current_exposure=0.0,
        )
        assert abs(exposure - 0.98) < 1e-6

    def test_pure_choppy_xgb_buy(self):
        """100% CHOPPY with XGB BUY => 0.98."""
        exposure = composite_signal_blended(
            regime_probs={"BULL": 0.0, "BEAR": 0.0, "CHOPPY": 1.0},
            sma_signal=Signal.HOLD,
            xgb_signal=Signal.BUY,
            bear_weakening=False,
            current_exposure=0.0,
        )
        assert abs(exposure - 0.98) < 1e-6

    def test_blended_60_bull_40_choppy(self):
        """60% BULL (SMA BUY) + 40% CHOPPY (XGB SELL) => 0.60*0.98 + 0.40*0.0 = 0.588."""
        exposure = composite_signal_blended(
            regime_probs={"BULL": 0.6, "BEAR": 0.0, "CHOPPY": 0.4},
            sma_signal=Signal.BUY,
            xgb_signal=Signal.SELL,
            bear_weakening=False,
            current_exposure=0.0,
        )
        assert abs(exposure - 0.588) < 1e-6

    def test_hold_uses_current_exposure(self):
        """BULL (SMA HOLD) should use current_exposure as its target."""
        exposure = composite_signal_blended(
            regime_probs={"BULL": 1.0, "BEAR": 0.0, "CHOPPY": 0.0},
            sma_signal=Signal.HOLD,
            xgb_signal=Signal.HOLD,
            bear_weakening=False,
            current_exposure=0.5,
        )
        assert abs(exposure - 0.5) < 1e-6

    def test_four_state_choppy_variants(self):
        """CHOPPY_0 and CHOPPY_1 both route to XGB."""
        exposure = composite_signal_blended(
            regime_probs={"BULL": 0.0, "BEAR": 0.0, "CHOPPY_0": 0.5, "CHOPPY_1": 0.5},
            sma_signal=Signal.HOLD,
            xgb_signal=Signal.BUY,
            bear_weakening=False,
            current_exposure=0.0,
        )
        assert abs(exposure - 0.98) < 1e-6

    def test_output_clamped_to_098(self):
        """Output should never exceed 0.98."""
        exposure = composite_signal_blended(
            regime_probs={"BULL": 1.0, "BEAR": 0.0, "CHOPPY": 0.0},
            sma_signal=Signal.BUY,
            xgb_signal=Signal.BUY,
            bear_weakening=False,
            current_exposure=0.98,
        )
        assert exposure <= 0.98

    def test_output_non_negative(self):
        """Output should never be negative."""
        exposure = composite_signal_blended(
            regime_probs={"BULL": 0.0, "BEAR": 1.0, "CHOPPY": 0.0},
            sma_signal=Signal.SELL,
            xgb_signal=Signal.SELL,
            bear_weakening=False,
            current_exposure=0.0,
        )
        assert exposure >= 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_composite.py -v -k "Blended"`
Expected: FAIL — `composite_signal_blended` not importable.

- [ ] **Step 3: Implement composite_signal_blended**

Add to `src/schroeder_trader/strategy/composite.py`:

```python
MAX_EXPOSURE = 0.98


def _signal_to_exposure(signal: Signal, current_exposure: float) -> float:
    """Convert a Signal enum to a target exposure value."""
    if signal == Signal.BUY:
        return MAX_EXPOSURE
    elif signal == Signal.SELL:
        return 0.0
    else:
        return current_exposure


def composite_signal_blended(
    regime_probs: dict[str, float],
    sma_signal: Signal,
    xgb_signal: Signal,
    bear_weakening: bool = False,
    current_exposure: float = 0.0,
) -> float:
    """Compute blended target exposure from regime probabilities.

    Each regime's signal source determines a target exposure, then
    the final exposure is the probability-weighted average.

    Args:
        regime_probs: Dict mapping regime label to probability.
            Keys: "BULL", "BEAR", "CHOPPY" (or "CHOPPY_0", "CHOPPY_1" for 4-state).
        sma_signal: Signal from SMA crossover.
        xgb_signal: Signal from XGB at low confidence threshold.
        bear_weakening: True if 5-day return is positive while dominant regime is BEAR.
        current_exposure: Current portfolio exposure (0.0 to 0.98).

    Returns:
        Target exposure as float in [0.0, 0.98].
    """
    blended = 0.0

    for label, prob in regime_probs.items():
        if prob <= 0:
            continue

        if label == "BULL":
            target = _signal_to_exposure(sma_signal, current_exposure)
        elif label == "BEAR":
            if bear_weakening:
                target = _signal_to_exposure(xgb_signal, current_exposure)
            else:
                target = 0.0
        else:
            # CHOPPY, CHOPPY_0, CHOPPY_1, etc. all route to XGB
            target = _signal_to_exposure(xgb_signal, current_exposure)

        blended += prob * target

    return max(0.0, min(MAX_EXPOSURE, blended))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_composite.py -v`
Expected: All existing tests pass + all new blended tests pass.

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest tests/ -q`
Expected: All pass.

- [ ] **Step 6: Commit**

```bash
git add src/schroeder_trader/strategy/composite.py tests/test_composite.py
git commit -m "feat: add composite_signal_blended for probability-weighted routing"
```

---

### Task 5: Add HMM config and training script

**Files:**
- Modify: `src/schroeder_trader/config.py:14-26`
- Modify: `backtest/train_final_composite.py`

- [ ] **Step 1: Add HMM config**

In `src/schroeder_trader/config.py`, after the `COMPOSITE_MODEL_PATH` line (line 14), add:

```python
HMM_MODEL_PATH = PROJECT_ROOT / "models" / "hmm_regime.pkl"
HMM_RETRAIN_DAYS = 30
```

- [ ] **Step 2: Add HMM training to train_final_composite.py**

Add HMM training at the end of `train_and_save()` in `backtest/train_final_composite.py`. Add import at top:

```python
from schroeder_trader.config import HMM_MODEL_PATH
from schroeder_trader.strategy.regime_detector import HMMRegimeDetector
```

Add at the end of `train_and_save()`, before the last print:

```python
    # Train HMM regime detector
    print("\nTraining HMM regime detector...")
    # Need VIX features for HMM
    ext_path = DATA_DIR / "features_daily.csv"
    ext_df = pd.read_csv(ext_path, index_col="date", parse_dates=True)

    # Add VIX columns to features_df
    for col in ["vix_close", "vix3m_close"]:
        if col in ext_df.columns:
            vix_series = ext_df[col].copy()
            if hasattr(vix_series.index, "tz") and vix_series.index.tz is not None:
                vix_series.index = vix_series.index.tz_localize(None)
            vix_series.index = vix_series.index.normalize()
            features_df[col] = vix_series.reindex(features_df.index).ffill()

    if "vix_close" in features_df.columns and "vix3m_close" in features_df.columns:
        features_df["vix_term_structure"] = features_df["vix_close"] / features_df["vix3m_close"]

    hmm_features = ["log_return_20d", "volatility_20d", "vix_close", "vix_term_structure"]
    hmm_df = features_df.dropna(subset=hmm_features)

    if len(hmm_df) > 100:
        detector = HMMRegimeDetector()
        detector.fit(hmm_df)
        HMM_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        detector.save(HMM_MODEL_PATH)
        print(f"HMM saved to {HMM_MODEL_PATH}")
        print(f"  States: {detector.n_states}, Labels: {detector.state_to_label_}")
    else:
        print("WARNING: Not enough data for HMM training")
```

- [ ] **Step 3: Run training script**

Run: `PYTHONPATH=. uv run python backtest/train_final_composite.py`
Expected: XGBoost trains as before, then HMM trains and saves to `models/hmm_regime.pkl`.

- [ ] **Step 4: Verify HMM model loads**

Run:
```bash
uv run python -c "
from schroeder_trader.strategy.regime_detector import HMMRegimeDetector
from schroeder_trader.config import HMM_MODEL_PATH
d = HMMRegimeDetector.load(HMM_MODEL_PATH)
print(f'States: {d.n_states}, Labels: {d.state_to_label_}')
"
```
Expected: Prints state count and labels.

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest tests/ -q`
Expected: All pass.

- [ ] **Step 6: Commit**

```bash
git add src/schroeder_trader/config.py backtest/train_final_composite.py models/hmm_regime.pkl
git commit -m "feat: add HMM training to composite training pipeline"
```

---

### Task 6: Integrate HMM into shadow pipeline

**Files:**
- Modify: `src/schroeder_trader/main.py:13-24,183-322`
- Test: `tests/test_main.py`

- [ ] **Step 1: Add HMM imports to main.py**

Add to the imports at the top of `main.py`:

```python
from schroeder_trader.config import HMM_MODEL_PATH
from schroeder_trader.strategy.regime_detector import HMMRegimeDetector
from schroeder_trader.strategy.composite import composite_signal_blended
```

- [ ] **Step 2: Load HMM model in the shadow pipeline**

In `_run_pipeline_inner()`, inside the shadow signal block (after the composite model is loaded and validated, around line 209), add HMM loading:

```python
            # Load HMM regime detector
            hmm_detector = None
            if HMM_MODEL_PATH.exists():
                try:
                    hmm_detector = HMMRegimeDetector.load(HMM_MODEL_PATH)
                    logger.info("HMM regime detector loaded (%d states)", hmm_detector.n_states)
                except Exception:
                    logger.warning("Failed to load HMM detector, falling back to threshold", exc_info=True)
```

- [ ] **Step 3: Add HMM prediction after XGB prediction**

After the existing composite signal logging block (after line 322), add the HMM blended prediction. This runs alongside the existing threshold-based shadow signal:

```python
                    # HMM blended signal (logged alongside threshold-based signal)
                    if hmm_detector is not None:
                        try:
                            hmm_row = features[["log_return_20d", "volatility_20d", "vix_close", "vix_term_structure"]].iloc[[-1]]
                            if not hmm_row.isna().any().any():
                                regime_probs = hmm_detector.predict_proba(hmm_row)[0]
                                hmm_dominant = hmm_detector.dominant_regime(regime_probs)

                                # Current exposure: position_value / portfolio_value
                                current_exp = position_value / account["portfolio_value"] if account["portfolio_value"] > 0 else 0.0

                                blended_exposure = composite_signal_blended(
                                    regime_probs=regime_probs,
                                    sma_signal=signal,
                                    xgb_signal=composite_sig,
                                    bear_weakening=bear_weakening,
                                    current_exposure=current_exp,
                                )
                                logger.info(
                                    "HMM blended: exposure=%.3f, dominant=%s, probs=%s",
                                    blended_exposure, hmm_dominant.value,
                                    {k: f"{v:.3f}" for k, v in regime_probs.items()},
                                )
                        except Exception:
                            logger.warning("HMM prediction failed (non-fatal)", exc_info=True)
```

- [ ] **Step 4: Ensure VIX features are available in the shadow pipeline**

The `compute_features_extended` call at line 225 already joins external features. With the Task 2 change, `vix_close` and `vix3m_close` will be joined and `vix_term_structure` will be computed. Verify this works by checking the log output after running the pipeline.

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest tests/ -q`
Expected: All pass (existing main.py tests mock external calls so the new code paths aren't hit in unit tests, but they don't break anything either).

- [ ] **Step 6: Commit**

```bash
git add src/schroeder_trader/main.py
git commit -m "feat: integrate HMM blended signal into shadow pipeline"
```

---

### Task 7: Walk-forward backtest comparison

**Files:**
- Create: `backtest/backtest_hmm_comparison.py`

- [ ] **Step 1: Create the comparison backtest script**

Create `backtest/backtest_hmm_comparison.py`:

```python
"""Walk-forward backtest comparing threshold vs HMM regime detection."""
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from backtest.train_final_composite import (
    prepare_features, XGB_FEATURES, XGB_PARAMS, TRAIN_YEARS, TEST_MONTHS,
)
from schroeder_trader.strategy.composite import (
    composite_signal_hybrid,
    composite_signal_blended,
)
from schroeder_trader.strategy.feature_engineer import CLASS_DOWN, CLASS_UP
from schroeder_trader.strategy.regime_detector import (
    Regime,
    HMMRegimeDetector,
    HMM_FEATURES,
)
from schroeder_trader.strategy.sma_crossover import Signal

CASH_BUFFER = 0.02
INITIAL = 100_000.0
TRAILING_STOP_PCT = 0.10


def _regime_from_label(label):
    return {0: Regime.BEAR, 1: Regime.CHOPPY, 2: Regime.BULL}.get(int(label), Regime.CHOPPY)


def _xgb_signal(proba, class_order, threshold):
    idx_up = list(class_order).index(CLASS_UP)
    idx_down = list(class_order).index(CLASS_DOWN)
    pred = int(np.argmax(proba))
    if pred == idx_up and proba[idx_up] > threshold:
        return Signal.BUY
    elif pred == idx_down and proba[idx_down] > threshold:
        return Signal.SELL
    return Signal.HOLD


def _sma_signal(close_series):
    if len(close_series) < 201:
        return Signal.HOLD
    sma50 = close_series.rolling(50).mean()
    sma200 = close_series.rolling(200).mean()
    if sma50.iloc[-1] > sma200.iloc[-1] and sma50.iloc[-2] <= sma200.iloc[-2]:
        return Signal.BUY
    elif sma50.iloc[-1] < sma200.iloc[-1] and sma50.iloc[-2] >= sma200.iloc[-2]:
        return Signal.SELL
    return Signal.HOLD


def compute_metrics(values):
    daily_rets = np.diff(values) / values[:-1]
    sharpe = np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(252) if np.std(daily_rets) > 0 else 0
    peak = np.maximum.accumulate(values)
    max_dd = float(np.min((values - peak) / peak))
    total_ret = (values[-1] / values[0]) - 1.0
    return sharpe, max_dd, total_ret


def run():
    print("Preparing features...")
    features_df = prepare_features()
    close = features_df["close"]
    daily_returns = close.pct_change()

    # Load external features for VIX
    ext_path = "backtest/data/features_daily.csv"
    ext_df = pd.read_csv(ext_path, index_col="date", parse_dates=True)

    # Add VIX features
    for col in ["vix_close", "vix3m_close"]:
        if col in ext_df.columns:
            vix = ext_df[col].copy()
            if hasattr(vix.index, "tz") and vix.index.tz is not None:
                vix.index = vix.index.tz_localize(None)
            vix.index = vix.index.normalize()
            features_df[col] = vix.reindex(features_df.index).ffill()

    if "vix_close" in features_df.columns and "vix3m_close" in features_df.columns:
        features_df["vix_term_structure"] = features_df["vix_close"] / features_df["vix3m_close"]

    print(f"Feature matrix: {len(features_df)} rows")

    # Walk-forward
    start_date = features_df.index.min()
    end_date = features_df.index.max()
    train_start = start_date
    records = []
    window_num = 0

    while True:
        train_end = train_start + pd.DateOffset(years=TRAIN_YEARS)
        test_end = train_end + pd.DateOffset(months=TEST_MONTHS)
        if train_end >= end_date:
            break

        train_data = features_df[
            (features_df.index >= train_start) & (features_df.index < train_end)
        ].dropna(subset=XGB_FEATURES)
        test_data = features_df[
            (features_df.index >= train_end) & (features_df.index < test_end)
        ].dropna(subset=XGB_FEATURES)

        if len(train_data) < 100 or len(train_data["label"].unique()) < 3:
            train_start += pd.DateOffset(months=TEST_MONTHS)
            continue
        if len(test_data) == 0:
            train_start += pd.DateOffset(months=TEST_MONTHS)
            continue

        window_num += 1

        # Train XGB
        X_train = train_data[XGB_FEATURES]
        y_train = train_data["label"]
        val_split = int(len(X_train) * 0.8)
        xgb_model = XGBClassifier(**XGB_PARAMS, n_estimators=500, early_stopping_rounds=20)
        xgb_model.fit(X_train[:val_split], y_train[:val_split],
                      eval_set=[(X_train[val_split:], y_train[val_split:])], verbose=False)

        # Train HMM on training data
        hmm_train = train_data.dropna(subset=HMM_FEATURES)
        hmm_detector = None
        if len(hmm_train) > 100:
            try:
                hmm_detector = HMMRegimeDetector()
                hmm_detector.fit(hmm_train)
            except Exception:
                hmm_detector = None

        if window_num % 8 == 0:
            hmm_info = f", HMM {hmm_detector.n_states} states" if hmm_detector else ", HMM failed"
            print(f"  Window {window_num}: test {test_data.index.min().date()}-{test_data.index.max().date()}{hmm_info}")

        class_order = xgb_model.classes_
        idx_up = list(class_order).index(CLASS_UP)
        idx_down = list(class_order).index(CLASS_DOWN)

        for dt, row in test_data.iterrows():
            X_row = row[XGB_FEATURES].values.reshape(1, -1)
            proba = xgb_model.predict_proba(pd.DataFrame(X_row, columns=XGB_FEATURES))[0]
            xgb_low = _xgb_signal(proba, class_order, 0.35)
            regime = _regime_from_label(row["regime_label"])
            hist_end = features_df.index.get_loc(dt)
            close_up_to = close.iloc[:hist_end + 1]
            sma_sig = _sma_signal(close_up_to)

            bear_weakening = False
            if regime == Regime.BEAR:
                lr5 = row.get("log_return_5d")
                if lr5 is not None and not np.isnan(lr5) and lr5 > 0:
                    bear_weakening = True

            # Threshold-based signal
            threshold_sig, threshold_src = composite_signal_hybrid(
                regime, sma_sig, xgb_low, bear_weakening=bear_weakening)

            # HMM signals
            hmm_hard_sig = threshold_sig  # fallback
            hmm_hard_src = threshold_src
            hmm_regime_probs = None

            if hmm_detector is not None:
                hmm_row = features_df[HMM_FEATURES].loc[[dt]]
                if not hmm_row.isna().any().any():
                    hmm_regime_probs = hmm_detector.predict_proba(hmm_row)[0]
                    hmm_dominant = hmm_detector.dominant_regime(hmm_regime_probs)

                    # HMM hard routing uses dominant regime
                    hmm_bear_weak = bear_weakening and hmm_dominant == Regime.BEAR
                    hmm_hard_sig, hmm_hard_src = composite_signal_hybrid(
                        hmm_dominant, sma_sig, xgb_low, bear_weakening=hmm_bear_weak)

            records.append({
                "date": dt,
                "threshold_sig": threshold_sig,
                "hmm_hard_sig": hmm_hard_sig,
                "hmm_regime_probs": hmm_regime_probs,
                "sma_sig": sma_sig,
                "xgb_sig": xgb_low,
                "bear_weakening": bear_weakening,
            })

        train_start += pd.DateOffset(months=TEST_MONTHS)

    signals_df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    print(f"\n{window_num} windows, {len(signals_df)} test days")

    # Simulate strategies
    strategies = {}

    # 1. Baseline: threshold + binary
    values = []
    invested = 0.0
    v = INITIAL
    for _, row in signals_df.iterrows():
        dr = daily_returns.get(row["date"], 0.0)
        if np.isnan(dr): dr = 0.0
        sig = row["threshold_sig"]
        if sig == Signal.BUY: invested = 1.0 - CASH_BUFFER
        elif sig == Signal.SELL: invested = 0.0
        v *= (1 + invested * dr)
        values.append(v)
    strategies["Threshold (current)"] = np.array(values)

    # 2. HMM hard routing + binary
    values = []
    invested = 0.0
    v = INITIAL
    for _, row in signals_df.iterrows():
        dr = daily_returns.get(row["date"], 0.0)
        if np.isnan(dr): dr = 0.0
        sig = row["hmm_hard_sig"]
        if sig == Signal.BUY: invested = 1.0 - CASH_BUFFER
        elif sig == Signal.SELL: invested = 0.0
        v *= (1 + invested * dr)
        values.append(v)
    strategies["HMM hard"] = np.array(values)

    # 3. HMM blended
    values = []
    invested = 0.0
    v = INITIAL
    for _, row in signals_df.iterrows():
        dr = daily_returns.get(row["date"], 0.0)
        if np.isnan(dr): dr = 0.0
        if row["hmm_regime_probs"] is not None:
            invested = composite_signal_blended(
                regime_probs=row["hmm_regime_probs"],
                sma_signal=row["sma_sig"],
                xgb_signal=row["xgb_sig"],
                bear_weakening=row["bear_weakening"],
                current_exposure=invested,
            )
        else:
            sig = row["threshold_sig"]
            if sig == Signal.BUY: invested = 1.0 - CASH_BUFFER
            elif sig == Signal.SELL: invested = 0.0
        v *= (1 + invested * dr)
        values.append(v)
    strategies["HMM blended"] = np.array(values)

    # SPY benchmark
    first_dt = signals_df["date"].iloc[0]
    last_dt = signals_df["date"].iloc[-1]
    spy_ret = (close.loc[last_dt] / close.loc[first_dt]) - 1

    # Print results
    years = (last_dt - first_dt).days / 365.25
    print(f"\n{'='*75}")
    print(f"REGIME DETECTION COMPARISON ({first_dt.date()} to {last_dt.date()}, {years:.1f} years)")
    print(f"{'='*75}")
    print(f"\n{'Strategy':<24} {'Sharpe':>8} {'Max DD':>8} {'Return':>10} {'Annual':>8} {'Final':>14}")
    print(f"{'-'*75}")

    spy_ann = (1 + spy_ret) ** (1/years) - 1
    print(f"{'SPY Buy & Hold':<24} {'':>8} {'':>8} {spy_ret:>9.1%} {spy_ann:>7.1%}")
    print(f"{'-'*75}")

    for label, vals in strategies.items():
        sharpe, max_dd, total_ret = compute_metrics(vals)
        annual = (1 + total_ret) ** (1/years) - 1
        print(f"{label:<24} {sharpe:>8.3f} {max_dd:>7.1%} {total_ret:>9.1%} {annual:>7.1%} ${vals[-1]:>13,.0f}")

    # Per-year breakdown
    print(f"\n{'Year':<6}", end="")
    for label in strategies:
        print(f" {label:>16}", end="")
    print(f" {'SPY':>10}")
    print("-" * (6 + 16 * len(strategies) + 10))

    dates = signals_df["date"].values
    for yr in range(int(first_dt.year), int(last_dt.year) + 1):
        yr_mask = pd.Series(dates).apply(lambda d: pd.Timestamp(d).year == yr)
        if yr_mask.sum() == 0:
            continue
        first_idx = yr_mask.idxmax()
        last_idx = yr_mask[::-1].idxmax()
        print(f"{yr:<6}", end="")
        for label, vals in strategies.items():
            if first_idx > 0:
                yr_ret = (vals[last_idx] / vals[first_idx - 1] - 1) * 100
            else:
                yr_ret = (vals[last_idx] / INITIAL - 1) * 100
            print(f" {yr_ret:>15.1f}%", end="")
        yr_close = close[close.index.year == yr]
        if len(yr_close) > 1:
            spy_yr = (yr_close.iloc[-1] / yr_close.iloc[0] - 1) * 100
            print(f" {spy_yr:>9.1f}%")
        else:
            print()


if __name__ == "__main__":
    run()
```

- [ ] **Step 2: Run the comparison**

Run: `PYTHONPATH=. uv run python backtest/backtest_hmm_comparison.py`
Expected: Prints comparison table with Sharpe, max DD, annual return for all three strategies plus per-year breakdown.

- [ ] **Step 3: Commit**

```bash
git add backtest/backtest_hmm_comparison.py
git commit -m "feat: add HMM vs threshold comparison backtest"
```

---

### Task 8: Evaluate results and decide

This is a manual decision point. After running the Task 7 backtest:

- [ ] **Step 1: Review comparison table**

Compare Sharpe, max DD, and annualized return across all three strategies. Pay special attention to:
- 2023 (the known weak spot — does HMM improve the 4.4% return?)
- Max drawdown (does it stay under 10%?)
- Overall Sharpe (does blending help or hurt?)

- [ ] **Step 2: Decide which strategy to adopt**

If HMM blended outperforms or matches threshold on key metrics, proceed to make it the default in the shadow pipeline. If it underperforms, keep threshold as default and HMM as experimental.

- [ ] **Step 3: Commit any config changes**

Based on the decision, update config.py if needed and commit.
