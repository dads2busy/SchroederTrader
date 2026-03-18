# SchroederTrader System Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        DAILY PIPELINE (main.py)                             │
│                     Runs Mon-Fri at 4:30 PM ET                              │
│                     via macOS launchd agent                                  │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                    STEPS 1-9: LIVE SMA PIPELINE                      │  │
│  │                    (Phase 1 — Production)                             │  │
│  │                                                                       │  │
│  │  Step 1: Idempotency ──► Already ran today? → EXIT                   │  │
│  │      │                                                                │  │
│  │      ▼                                                                │  │
│  │  Step 2: Fill Check ──► Check pending Alpaca orders                  │  │
│  │      │                  Update filled/rejected status                 │  │
│  │      │                  Send fill alerts                              │  │
│  │      ▼                                                                │  │
│  │  Step 3: Market Check ──► Market closed today? → EXIT                │  │
│  │      │                                                                │  │
│  │      ▼                                                                │  │
│  │  Step 4: Fetch Data ──► yfinance: SPY 365 days OHLCV                 │  │
│  │      │                                                                │  │
│  │      ▼                                                                │  │
│  │  Step 5: SMA Signal ──► SMA 50/200 crossover detection               │  │
│  │      │                  Golden cross → BUY                            │  │
│  │      │                  Death cross → SELL                            │  │
│  │      │                  No crossover → HOLD                           │  │
│  │      ▼                                                                │  │
│  │  Step 6: Risk Eval ──► Position sizing (cash buffer, whole shares)   │  │
│  │      │                                                                │  │
│  │      ▼                                                                │  │
│  │  Step 7: Execute ──► Alpaca paper trading API                        │  │
│  │      │                Submit market order                             │  │
│  │      ▼                                                                │  │
│  │  Step 8: Log ──► SQLite: signals, orders, portfolio tables           │  │
│  │      │                                                                │  │
│  │      ▼                                                                │  │
│  │  Step 9: Summary ──► Email daily report                              │  │
│  │                                                                       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│      │                                                                      │
│      │  signal, close_price, sma_50, sma_200 passed by variable scope       │
│      ▼                                                                      │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │              STEP 10: COMPOSITE SHADOW SIGNAL                        │  │
│  │              (Phase 4 — Shadow Mode, No Orders)                      │  │
│  │              Entire block wrapped in try/except                       │  │
│  │                                                                       │  │
│  │  10a. Download External Features                                      │  │
│  │       └─► download_features.py (idempotent, skips if <24h old)       │  │
│  │           ├── yfinance: ^VIX, ^VIX3M, HYG, LQD, GLD, UUP, TLT, EEM │  │
│  │           └── FRED CSV: DGS10, DGS2 (Treasury yields)               │  │
│  │           → backtest/data/features_daily.csv                          │  │
│  │                                                                       │  │
│  │  10b. Load Model                                                      │  │
│  │       └─► models/xgboost_spy_20d.json                                │  │
│  │           If missing → skip entire shadow step                        │  │
│  │           Validate model.classes_ == [0, 1, 2]                        │  │
│  │                                                                       │  │
│  │  10c. Fetch SPY 400 Days                                              │  │
│  │       └─► yfinance: SPY 400 days OHLCV (for SMA200 + feature warmup)│  │
│  │                                                                       │  │
│  │  10d. Compute Extended Features                                       │  │
│  │       └─► FeaturePipeline.compute_features_extended(spy_df, ext_df)  │  │
│  │           ├── log_return_5d     (SPY momentum)                        │  │
│  │           ├── log_return_20d    (SPY trend)                           │  │
│  │           ├── volatility_20d    (SPY risk regime)                     │  │
│  │           ├── credit_spread     (HYG/LQD 20d change)                 │  │
│  │           └── dollar_momentum   (UUP 20d log return)                  │  │
│  │                                                                       │  │
│  │  10e. Compute Regime Labels (backward-looking, no look-ahead)         │  │
│  │       └─► detect_regime() for each day in 400-day history             │  │
│  │           ├── BULL:   return > 0 AND vol < 252d median vol            │  │
│  │           ├── BEAR:   return < 0 AND vol > 252d median vol            │  │
│  │           └── CHOPPY: everything else                                 │  │
│  │           + regime_label as integer feature (BEAR=0, CHOPPY=1, BULL=2)│  │
│  │                                                                       │  │
│  │  10f. Extract Today's State                                           │  │
│  │       ├── today_regime = last row of regime series                    │  │
│  │       └── bear_days = count_consecutive_bear_days(regime_series)      │  │
│  │                                                                       │  │
│  │  10g. XGBoost Predictions (both thresholds, unconditional)            │  │
│  │       └─► model.predict_proba(last_row[6 features])                  │  │
│  │           ├── xgb_low:  argmax-gated, threshold 0.35 (for Choppy)    │  │
│  │           └── xgb_high: argmax-gated, threshold 0.50 (for late Bear) │  │
│  │                                                                       │  │
│  │  10h. Composite Signal Routing                                        │  │
│  │       └─► composite_signal_hybrid()                                   │  │
│  │                                                                       │  │
│  │           ┌─────────────┬────────────────────────────────────────┐    │  │
│  │           │ Regime      │ Signal                                 │    │  │
│  │           ├─────────────┼────────────────────────────────────────┤    │  │
│  │           │ BULL        │ SMA crossover (from Step 5)            │    │  │
│  │           │ BEAR ≤ 20d  │ SELL (go flat)                         │    │  │
│  │           │ BEAR > 20d  │ XGBoost @ 0.50 threshold               │    │  │
│  │           │ CHOPPY      │ XGBoost @ 0.35 threshold               │    │  │
│  │           └─────────────┴────────────────────────────────────────┘    │  │
│  │                                                                       │  │
│  │  10i. Log Shadow Signal                                               │  │
│  │       └─► SQLite shadow_signals table                                 │  │
│  │           ├── ml_signal (composite signal)                            │  │
│  │           ├── sma_signal (from Step 5)                                │  │
│  │           ├── regime (BULL/BEAR/CHOPPY)                               │  │
│  │           ├── signal_source (SMA/FLAT/XGB)                            │  │
│  │           ├── bear_day_count (if BEAR)                                │  │
│  │           ├── predicted_class (if XGB, else NULL)                     │  │
│  │           └── predicted_proba (if XGB, else NULL)                     │  │
│  │                                                                       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  conn.close()                                                               │
│  Pipeline complete.                                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                        OFFLINE SCRIPTS (manual)                             │
│                                                                             │
│  backtest/train_final_composite.py                                          │
│  └─► Train XGBoost model for shadow deployment                             │
│      1. Load SPY + external features                                        │
│      2. Compute 6 features + regime_label + 20-day labels                   │
│      3. Walk-forward to find median n_estimators                            │
│      4. Retrain on all data with fixed n_estimators                         │
│      5. Save to models/xgboost_spy_20d.json                                │
│                                                                             │
│  backtest/download_features.py                                              │
│  └─► Download/cache external feature data                                  │
│      yfinance (8 tickers) + FRED (2 yield series)                          │
│      → backtest/data/features_daily.csv                                     │
│                                                                             │
│  backtest/download_data.py                                                  │
│  └─► Download/cache SPY historical OHLCV                                   │
│      → backtest/data/spy_daily.csv                                          │
│                                                                             │
│  backtest/compare_models.py                                                 │
│  └─► Compare SMA vs composite signals from shadow_signals table            │
│      Run after accumulating shadow data                                     │
│                                                                             │
│  backtest/feature_selection.py                                              │
│  └─► Forward feature selection using walk-forward Sharpe                   │
│      Used during Phase 2.1 feature iteration (research tool)                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                          DATA STORAGE                                       │
│                                                                             │
│  data/trades.db (SQLite)                                                    │
│  ├── signals          — SMA signal log (1 row/day)                          │
│  ├── orders           — Alpaca order log                                    │
│  ├── portfolio         — Daily portfolio snapshot                            │
│  └── shadow_signals   — Composite shadow predictions                        │
│       ├── timestamp, ticker, close_price                                    │
│       ├── ml_signal (composite), sma_signal                                 │
│       ├── regime, signal_source, bear_day_count                             │
│       └── predicted_class, predicted_proba (NULL if not XGB)                │
│                                                                             │
│  models/xgboost_spy_20d.json — Trained XGBoost model (20-day horizon)       │
│                                                                             │
│  backtest/data/spy_daily.csv — Cached SPY OHLCV (30+ years)                │
│  backtest/data/features_daily.csv — Cached external features                │
│  backtest/results/*.json — Walk-forward and selection results               │
│                                                                             │
│  logs/                                                                      │
│  ├── schroeder_trader.log — Python logging output                           │
│  ├── launchd_stdout.log   — launchd stdout                                  │
│  └── launchd_stderr.log   — launchd stderr                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                       EXTERNAL SERVICES                                     │
│                                                                             │
│  Alpaca Paper Trading API ──► Order execution, account info, positions      │
│  yfinance ──► SPY OHLCV, VIX, ETF prices                                   │
│  FRED public CSV ──► Treasury yields (DGS10, DGS2)                          │
│  Gmail SMTP ──► Trade alerts, fill alerts, error alerts, daily summary      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                    MODULE DEPENDENCY MAP                                     │
│                                                                             │
│  main.py                                                                    │
│  ├── config.py (paths, parameters, API keys)                                │
│  ├── logging_config.py                                                      │
│  ├── data/market_data.py (fetch_daily_bars, is_market_open_today)           │
│  ├── strategy/                                                              │
│  │   ├── sma_crossover.py (Signal enum, generate_signal)                   │
│  │   ├── feature_engineer.py (FeaturePipeline, class constants)            │
│  │   ├── xgboost_classifier.py (load_model, predict_signal)               │
│  │   ├── regime_detector.py (Regime enum, detect_regime)                   │
│  │   └── composite.py (composite_signal_hybrid, count_consecutive_bear_days)│
│  ├── risk/risk_manager.py (evaluate → OrderRequest)                         │
│  ├── execution/broker.py (submit_order, get_position, get_account)          │
│  ├── storage/trade_log.py (init_db, log_*, get_*)                           │
│  └── alerts/email_alert.py (send_*_alert, send_daily_summary)               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Future Development (Not Yet Implemented)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PHASE 4 ROADMAP — FUTURE TRACKS                          │
│                                                                             │
│  TRACK 1: SHADOW VALIDATION (current — collecting data)                     │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  3-month forward test before real capital                           │    │
│  │  ├── Monitor regime detection stability (oscillation at boundaries?)│    │
│  │  ├── Measure signal-to-execution gap (backtest vs live fills)       │    │
│  │  ├── Validate feature pipeline reliability (data gaps, delays)      │    │
│  │  └── Run compare_models.py weekly to track composite vs SMA         │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  TRACK 2: STRATEGY REFINEMENT                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Regime Persistence Filter                            [~0.5 days]   │    │
│  │  └── Require 3+ consecutive days before transitioning to BEAR       │    │
│  │      Reduces spurious bear entries from one-day vol spikes          │    │
│  │                                                                     │    │
│  │  Fourth "Recovery" Regime                             [~2-3 days]   │    │
│  │  └── RECOVERY: 20d return trending positive AND prior regime = BEAR │    │
│  │      Routes to XGBoost to capture early snapback rallies            │    │
│  │                                                                     │    │
│  │  Expanded XGBoost Features                            [~3-5 days]   │    │
│  │  └── VIX level, VIX 5d change                                      │    │
│  │      SPY vs equal-weight RSP ratio (market breadth)                 │    │
│  │      Lagged signals (prior day regime, prior day signal)            │    │
│  │      Calendar effects (day of week, month-end)                      │    │
│  │      Interaction features (momentum × volatility)                   │    │
│  │                                                                     │    │
│  │  Optuna Hyperparameter Tuning                         [~1-2 days]   │    │
│  │  └── Systematic search over max_depth, learning_rate, subsample     │    │
│  │      Walk-forward Sharpe as optimization objective                  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  TRACK 3: RISK MANAGEMENT                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Fractional Kelly Position Sizing                     [~2-3 days]   │    │
│  │  └── Scale position by XGBoost confidence (P(UP) probability)       │    │
│  │      Quarter-Kelly or half-Kelly in practice                        │    │
│  │      Currently binary: 100% long or 0% flat                         │    │
│  │                                                                     │    │
│  │  Volatility-Scaled Transaction Cost Model             [~1 day]      │    │
│  │  └── cost = base_cost × (current_vol / median_vol)                  │    │
│  │      More realistic during high-vol periods                         │    │
│  │                                                                     │    │
│  │  Portfolio-Level Trailing Stop                         [~1 day]      │    │
│  │  └── If drawdown from peak > 8% → force flat for N days            │    │
│  │      Meta-regime detector for failure modes                         │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  TRACK 4: ARCHITECTURE EXTENSIONS                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Probabilistic Regime Classifier (HMM/Jump Model)     [~5-7 days]   │    │
│  │  └── P(BULL)/P(BEAR)/P(CHOPPY) per day instead of hard labels      │    │
│  │      Enables soft signal blending (70% SMA + 30% XGB)              │    │
│  │      Regime confidence as an XGBoost input feature                  │    │
│  │      Better transition detection                                    │    │
│  │                                                                     │    │
│  │  Automated Retraining Pipeline (Phase 7)              [~3-5 days]   │    │
│  │  └── Drift detection triggers model retraining                     │    │
│  │      Scheduled weekly/monthly walk-forward evaluation               │    │
│  │      Model versioning with performance tracking                     │    │
│  │                                                                     │    │
│  │  Live Trading Promotion                               [~2-3 days]   │    │
│  │  └── Replace shadow Step 10 with live order execution              │    │
│  │      composite_signal → risk_manager → broker                       │    │
│  │      Requires: shadow validation passed, manual approval            │    │
│  │                                                                     │    │
│  │  Gate Criteria Revision                                              │    │
│  │  └── Add Sortino ratio (captures asymmetric downside protection)    │    │
│  │      Add Calmar ratio (return / max drawdown)                       │    │
│  │      Better metrics for a strategy whose edge is risk management    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                   PROMOTION PATH: SHADOW → LIVE                             │
│                                                                             │
│   Shadow Mode (current)                                                     │
│   ├── Composite signals logged daily                                        │
│   ├── No orders placed                                                      │
│   ├── SMA pipeline trades independently                                     │
│   │                                                                         │
│   ▼  After 3 months of shadow data:                                         │
│   │                                                                         │
│   Validation Gate                                                           │
│   ├── Shadow composite Sharpe ≥ 0.88?                                       │
│   ├── Shadow max drawdown ≤ 25%?                                            │
│   ├── Regime detection stable? (no rapid oscillation)                       │
│   ├── Feature pipeline reliable? (no data gaps)                             │
│   │                                                                         │
│   ▼  If all pass + manual approval:                                         │
│   │                                                                         │
│   Live Promotion                                                            │
│   ├── Step 5: generate_signal() replaced by composite routing              │
│   ├── Steps 6-7: risk_manager + broker use composite signal                │
│   ├── Step 10: becomes audit log (both SMA and composite recorded)         │
│   └── Shadow mode retained as comparison baseline                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Walk-Forward Validation Results (Verified)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    COMPOSITE STRATEGY PERFORMANCE                           │
│                    (2007-2026, 34 walk-forward windows)                      │
│                                                                             │
│  Configuration:                                                             │
│    BULL (47% of days)   → SMA crossover                                     │
│    BEAR ≤20d (21%)      → Go flat                                           │
│    BEAR >20d            → XGBoost @ 0.50 threshold                          │
│    CHOPPY (32%)         → XGBoost @ 0.35 threshold                          │
│                                                                             │
│  ┌──────────────────┬──────────┬──────────┬────────┐                       │
│  │ Metric           │ Result   │ Target   │ Status │                       │
│  ├──────────────────┼──────────┼──────────┼────────┤                       │
│  │ Full Sharpe      │ 0.94     │ ≥ 0.88   │ PASS   │                       │
│  │ Post-2020 Sharpe │ 1.31     │ ≥ 0.80   │ PASS   │                       │
│  │ Max Drawdown     │ 16.1%    │ ≤ 25%    │ PASS   │                       │
│  │ Total Trades     │ 286      │ ≥ 30     │ PASS   │                       │
│  └──────────────────┴──────────┴──────────┴────────┘                       │
│                                                                             │
│  Total Return: 512%                                                         │
│  XGBoost Trades: 168 (59% of total)                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
