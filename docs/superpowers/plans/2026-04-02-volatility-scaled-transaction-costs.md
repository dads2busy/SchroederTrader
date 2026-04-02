# Volatility-Scaled Transaction Costs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the fixed 5 bps slippage estimate with a VIX-tiered cost model in all backtest scripts, making backtest results more realistic.

**Architecture:** New `estimate_slippage(vix)` function in `risk/transaction_cost.py` returns a slippage fraction based on VIX tier. Three backtest scripts updated to use it instead of the fixed `SLIPPAGE_ESTIMATE` config value, which is removed.

**Tech Stack:** Python, pytest

---

### Task 1: Transaction Cost Module — Tests

**Files:**
- Create: `tests/test_transaction_cost.py`

- [ ] **Step 1: Write failing tests**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_transaction_cost.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Commit test file**

```bash
git add tests/test_transaction_cost.py
git commit -m "test: add transaction cost tests (red)"
```

---

### Task 2: Transaction Cost Module — Implementation

**Files:**
- Create: `src/schroeder_trader/risk/transaction_cost.py`

- [ ] **Step 1: Implement estimate_slippage**

```python
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
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `pytest tests/test_transaction_cost.py -v`
Expected: all 5 tests PASS

- [ ] **Step 3: Commit**

```bash
git add src/schroeder_trader/risk/transaction_cost.py
git commit -m "feat: add VIX-tiered transaction cost estimator"
```

---

### Task 3: Remove SLIPPAGE_ESTIMATE from Config

**Files:**
- Modify: `src/schroeder_trader/config.py`

- [ ] **Step 1: Remove the SLIPPAGE_ESTIMATE line**

Remove this line from `config.py`:
```python
SLIPPAGE_ESTIMATE = 0.0005
```

- [ ] **Step 2: Run config tests**

Run: `pytest tests/test_config.py -v`
Expected: all PASS

- [ ] **Step 3: Commit**

```bash
git add src/schroeder_trader/config.py
git commit -m "chore: remove fixed SLIPPAGE_ESTIMATE config"
```

---

### Task 4: Update train_model.py

**Files:**
- Modify: `backtest/train_model.py`

- [ ] **Step 1: Update imports**

Replace:
```python
from schroeder_trader.config import SMA_SHORT_WINDOW, SMA_LONG_WINDOW, SLIPPAGE_ESTIMATE
```
With:
```python
from schroeder_trader.config import SMA_SHORT_WINDOW, SMA_LONG_WINDOW
from schroeder_trader.risk.transaction_cost import estimate_slippage
```

- [ ] **Step 2: Load VIX data**

Add at the top of `walk_forward_evaluate()`, after `df = load_data()`:

```python
    # Load VIX for volatility-scaled transaction costs
    features_csv = DATA_DIR / "features_daily.csv"
    vix_df = pd.read_csv(str(features_csv), index_col="date", parse_dates=True)["vix_close"]
```

- [ ] **Step 3: Replace fixed slippage with VIX-based cost**

In the strategy return simulation loop (around line 139-141), replace:
```python
            strategy_return -= SLIPPAGE_ESTIMATE
```
With:
```python
            trade_date = all_dates[i]
            vix_val = vix_df.get(trade_date, 20.0)  # default to normal VIX if missing
            strategy_return -= estimate_slippage(vix_val)
```

Note: `all_dates` is a `pd.DatetimeIndex` built during the walk-forward loop. The `vix_df` index may need normalization — use `.get()` with a default to handle missing dates gracefully.

- [ ] **Step 4: Verify script runs**

Run: `.venv/bin/python backtest/train_model.py`
Expected: completes without error, prints results

- [ ] **Step 5: Commit**

```bash
git add backtest/train_model.py
git commit -m "feat: use VIX-scaled slippage in walk-forward backtest"
```

---

### Task 5: Update compare_models.py

**Files:**
- Modify: `backtest/compare_models.py`

- [ ] **Step 1: Update imports**

Replace:
```python
from schroeder_trader.config import DB_PATH, SLIPPAGE_ESTIMATE
```
With:
```python
from schroeder_trader.config import DB_PATH
from schroeder_trader.risk.transaction_cost import estimate_slippage
```

- [ ] **Step 2: Pass VIX data to _simulate_strategy**

Update the `_simulate_strategy` function signature to accept a VIX series:

```python
def _simulate_strategy(signals: np.ndarray, daily_returns: np.ndarray, vix_values: np.ndarray) -> dict:
```

Replace the fixed slippage line:
```python
        if position != prev_position:
            strat_ret -= SLIPPAGE_ESTIMATE
```
With:
```python
        if position != prev_position:
            vix = vix_values[i] if i < len(vix_values) else 20.0
            strat_ret -= estimate_slippage(vix)
```

Note: add `for i, (sig, ret) in enumerate(zip(signals, daily_returns)):` to get the index.

- [ ] **Step 3: Load VIX and pass to simulate calls**

In `compare()`, after building `merged`, load VIX data and pass it:

```python
    # Load VIX for transaction costs
    features_csv = Path(__file__).parent / "data" / "features_daily.csv"
    if features_csv.exists():
        vix_df = pd.read_csv(str(features_csv), index_col="date", parse_dates=True)["vix_close"]
        vix_values = merged["date"].apply(lambda d: vix_df.get(pd.Timestamp(d), 20.0)).values
    else:
        vix_values = np.full(len(merged), 20.0)
```

Update the simulation calls to pass `vix_values[:-1]` (since `daily_returns` has length `len(merged) - 1`):

```python
    for name, signal_col in [("sma", "signal"), ("ml", "ml_signal")]:
        metrics = _simulate_strategy(merged[signal_col].values[:-1], daily_returns, vix_values[:-1])
```

- [ ] **Step 4: Commit**

```bash
git add backtest/compare_models.py
git commit -m "feat: use VIX-scaled slippage in model comparison"
```

---

### Task 6: Update run_backtest.py

**Files:**
- Modify: `backtest/run_backtest.py`

- [ ] **Step 1: Update imports**

Replace:
```python
from schroeder_trader.config import SMA_SHORT_WINDOW, SMA_LONG_WINDOW, SLIPPAGE_ESTIMATE
```
With:
```python
from schroeder_trader.config import SMA_SHORT_WINDOW, SMA_LONG_WINDOW
from schroeder_trader.risk.transaction_cost import estimate_slippage
```

- [ ] **Step 2: Build per-day fee array from VIX**

In `run_backtest()`, after `close = df["Close"]`, add:

```python
    # Load VIX for volatility-scaled fees
    features_csv = DATA_DIR / "features_daily.csv"
    vix_df = pd.read_csv(str(features_csv), index_col="date", parse_dates=True)["vix_close"]
    # Build per-day fee series aligned to close's index
    fees = close.index.to_series().apply(
        lambda d: estimate_slippage(vix_df.get(d.normalize(), 20.0))
    )
```

- [ ] **Step 3: Replace fixed fees with per-day array**

Replace all three occurrences of `fees=SLIPPAGE_ESTIMATE` with `fees=fees` (for full period) or the appropriate slice (for test period):

For the full portfolio:
```python
    portfolio = vbt.Portfolio.from_signals(
        close,
        entries=entries,
        exits=exits,
        init_cash=10000,
        fees=fees,
        freq="1D",
    )
```

For the test portfolio:
```python
    test_fees = fees[fees.index >= split_date]
    test_portfolio = vbt.Portfolio.from_signals(
        test_close,
        entries=test_entries,
        exits=test_exits,
        init_cash=10000,
        fees=test_fees,
        freq="1D",
    )
```

- [ ] **Step 4: Commit**

```bash
git add backtest/run_backtest.py
git commit -m "feat: use VIX-scaled slippage in SMA backtest"
```

---

### Task 7: Full Test Suite + Push

**Files:** (no new files)

- [ ] **Step 1: Run full test suite**

Run: `pytest -v`
Expected: all tests PASS (no test imports SLIPPAGE_ESTIMATE)

- [ ] **Step 2: Verify no remaining references to SLIPPAGE_ESTIMATE**

Run: `grep -r "SLIPPAGE_ESTIMATE" src/ backtest/ tests/`
Expected: no matches

- [ ] **Step 3: Push**

```bash
git push
```
