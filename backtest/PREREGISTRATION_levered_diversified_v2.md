# Pre-Registration: Levered Diversified Portfolios (v2)

**Date frozen:** 2026-06-10 (registration = the git commit introducing this file).
**Relationship to v1:** fresh hypothesis space (static multi-asset allocation; no
timing signals of any kind). v1's kill criterion ended SPY-timing research; nothing
here was derived from v1's results beyond the shared cost model and harness
conventions. Objective unchanged: more terminal wealth than SPY B&H, user maxDD
tolerance −25%, max leverage 1.3x.

## 1. Hypothesis under test

A diversified stock/bond/gold mix has a structurally higher Sharpe and shallower
drawdowns than equities alone; 1.3x leverage converts that into terminal wealth.
Diversification addresses the drawdown cap; leverage addresses "beat B&H." No
forecasting anywhere.

## 2. Candidates (frozen; static target weights; zero fitted parameters)

| ID | Portfolio (asset: target weight) | Leverage |
|----|----------------------------------|----------|
| S0 | SPY 100% (benchmark; exempt from DD gate) | 1.0 |
| S1 | SPY 60 / IEF 40 (the null; exempt from gate 3 — it is the gate's reference) | 1.0 |
| S2 | SPY 60 / IEF 40 | 1.3x |
| S3 | SPY 55 / IEF 35 / GLD 10 | 1.3x |
| S4 | SPY 60 / TLT 40 | 1.3x |
| S5 | SPY 55 / TLT 35 / GLD 10 | 1.3x |

Diagnostic shadow (not a candidate, printed for reference): SPY 100% at 1.3x —
shows what leverage alone does, isolating diversification's contribution.

## 3. Engine, execution, costs

- **Rebalancing:** to exact target notional weights at the close of the first
  trading day of each month, plus an emergency rebalance at any close where gross
  leverage (sum of notional weights) has drifted to ≥ 1.40. No other trading.
- **Timing convention (honest, as v1):** a rebalance decided at close[t] earns the
  new weights' returns from close[t]→close[t+1]. Day-t returns accrue to the
  weights held coming into day t. Between rebalances weights drift with returns.
- **Slippage:** `estimate_slippage(VIX[t]) × turnover` per rebalance, turnover =
  Σ|target_i − drifted_i| in notional terms (same 4-tier VIX function as v1).
- **Financing:** daily `(DTB3[t−1] + 0.40%)/252 × max(0, gross − 1)`; idle cash
  (gross < 1) earns `DTB3[t−1]/252 × (1 − gross)`. Rates lagged one day (knowable).
- **Dividends:** all assets via yfinance adjusted closes.
- **Data:** SPY/IEF/TLT/GLD daily adjusted closes (cached per `load_history`
  format), VIX from `features_daily.csv`, DTB3 from `data/dtb3.csv` (v1 AMENDMENT 1
  source). **Window:** first date all inputs exist (~GLD inception 2004-11-18)
  through the latest common date. Sub-period diagnostics: 2004–2013, 2014–end,
  2015–end.

## 4. Pass gates (ALL binding, full sample, net of all costs — identical to v1)

1. **maxDD ≥ −20%** on daily closes (buffer below the −25% tolerance).
2. **CAGR > S0 (SPY B&H).**
3. **CAGR > S1 (unlevered 60/40 null).** S1 itself is exempt (cannot beat itself).

**Winner:** highest CAGR among passers; tie-break within 0.25pp → fewer rebalances.
**Fragility flag (binding):** winner's 2015→end CAGR < S0's same window → the
paper-trade gate extends from 6 to 12 months.
**Kill criterion:** if no candidate passes, the registered conclusion is that the
constraint pair (beat SPY terminal wealth, maxDD ≤ −25%, leverage ≤ 1.3x) is
infeasible by static allocation with these assets — the follow-up conversation is
about the constraints, not about more strategies. No tweaked variant of this
hypothesis space may be evaluated without a fresh pre-registration.

## 5. Validity checks (must pass BEFORE results are read)

- **V1 (engine identity):** S0 run through the engine must match the raw SPY
  adjusted-close total return within 0.1% (zero turnover ⇒ zero costs ⇒ identity).
- **V2 (leverage realism):** the 1.3x SPY diagnostic must show (a) average gross
  leverage within 1.30 ± 0.05 and (b) a deeper maxDD than S0 — confirming
  financing and leverage are actually being modeled.

## 6. Anti-snooping rules (identical to v1)

One evaluation run; results stand. Only bug-class amendments (code ≠ spec),
documented with diffs. Sub-periods are diagnostics; gates are full-sample only; no
post-hoc window or gate changes. Any new idea = new pre-registration, fresh sample.

## 7. Paper-trade gate

Survivor runs ≥ 6 months (12 if fragility-flagged) simulated alongside the live
System control before any real dollar moves; judged on spec fidelity and cost
realism, not on raw outperformance over the window.

## 8. Registered predictions (calibration on record, 2026-06-10)

Made after back-of-envelope 2008/2022 arithmetic and stated as a downward revision
of the 55–60% quoted in conversation before that arithmetic:

- P(at least one candidate passes all gates): **30%**
- Most likely passer: S3 (IEF + gold).
- P(either TLT variant passes): **10%** (2022 at 1.3x is their kill scenario).
- P(kill criterion — constraint pair infeasible here): **70%**

The known hard cases: 2008 peak-to-trough for unlevered 60/40 was ≈ −31%; at 1.3x
the DD gate likely fails unless the bond/gold rally and monthly rebalancing carry
enough of the path. That is precisely what the run decides.
