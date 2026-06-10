# Investment Policy Statement — 55/40/5 with Staged Entry

**Adopted:** ____________ (sign and date; the git commit introducing this file is
the drafting record)
**Owner:** Aaron Schroeder
**Status:** This document is the "written risk policy" called for by the registered
conclusion of `backtest/PREREGISTRATION_levered_brake_v1.md` and confirmed by v2:
no tested timing or leverage scheme beat buy-and-hold within the owner's risk
constraints. The strategy below is therefore deliberately static. Drafted with an
AI assistant; not licensed financial or tax advice.

---

## 1. Objective

Long-term growth of a $100,000 investment with drawdowns materially shallower than
an all-equity portfolio, implemented with liquid index ETFs, no forecasting, no
discretion.

## 2. Target allocation

| Sleeve | Weight | Primary instrument | Acceptable equivalent |
|---|---|---|---|
| US equities | **55%** | VOO or VTI (≤0.05% ER) | SPY (higher ER) |
| Intermediate US Treasuries | **40%** | VGIT (0.04% ER) | IEF |
| Gold | **5%** | GLDM (0.10% ER) | IAUM |

Rationale on record: bonds hedge deflationary crashes (2008, 2020), gold hedges
the inflationary kind (2022); the 2026-06-10 v2 evaluation measured a 10% gold
sleeve improving Sharpe (0.82 vs 0.74) and drawdown in every mix tested over
2004–2026. Sized at 5% per the owner's choice, funded toward the bond side per the
owner's −25% drawdown preference. If a tax-advantaged account is available, place
bonds and gold there first (gold ETFs are taxed as collectibles in taxable
accounts; bond income is ordinary).

## 3. Entry schedule (fixed; no discretion)

- **At adoption:** invest **$50,000** at target weights — $27,500 equities /
  $20,000 Treasuries / $2,500 gold.
- **Then six monthly tranches of $8,333.33** ($4,583 / $3,333 / $417), executed on
  the **first trading day of each month**, starting the month after adoption.
- Undeployed cash sits in a Treasury money-market fund or T-bills until its
  scheduled date.
- **No acceleration, no delay, no skipping.** A tranche date that "feels wrong" is
  executed anyway. The 2026-06-10 evaluations measured what discretionary timing
  is worth: nothing, at best.

## 4. Rebalancing rule

- **Check** on the first trading day of each calendar quarter (and at no other
  time).
- **Act** only if equities or Treasuries are off target by more than **5
  percentage points**, or gold by more than **2** — then rebalance all three back
  to 55/40/5.
- Dividends and any future contributions buy the most-underweight sleeve first.

## 5. Expected behavior — agreed to in advance

Planning numbers, from the measured 2004–2026 record (net of costs), not promises:

- Long-run return: roughly **6.5–8.5%/yr nominal** — *expected to slightly trail
  100% SPY over long bull runs. That is the accepted price of the drawdown
  profile, not a malfunction.*
- Routine year: −10 to −15% peak-to-trough at least once every few years.
- Severe crisis (2008 analog): **approximately −30%** peak-to-trough.
  **Acknowledged residual risk:** this exceeds the owner's stated −25% tolerance.
  Staying within −25% through a 2008-type event would require ~45–50% equities at
  a real cost to long-run return; the owner consciously declined that trade on
  2026-06-10. If −25% is in fact a hard limit, the correct action is to amend
  §2 *now*, not during the crisis.

## 6. Conduct rules during drawdowns

At −15%, at −25%, and at −35%: **the policy is to do nothing except execute §3
and §4 on their fixed dates.** No sleeve is sold, no allocation changed, no
"temporary" cash raised. Rebalancing into the fallen sleeve on schedule is the
only buying-low this policy performs, and it is mechanical.

## 7. Amendment rules (the lesson of 2026-06-10, encoded)

This policy may be amended only when **all** of the following hold:

1. The portfolio is within **10% of its high-water mark** (no rewriting risk
   tolerance mid-drawdown — that is when the worst decisions are made), **except**
   for amendments that *reduce* equity weight following §5's acknowledged-risk
   clause;
2. A written amendment note (date, change, reason) is added to §9 **at least 30
   days before** the change executes;
3. The reason is a change in the owner's circumstances (income, horizon,
   liquidity needs) — **never** recent market performance, a forecast, or a
   backtest that has not survived a pre-registered, look-ahead-free evaluation.

## 8. What this policy is not

No market timing, no regime detection, no leverage, no tactical tilts, no new
strategies adopted from flattering backtests. Two pre-registered evaluations
(v1: timing; v2: levered diversification) ended in their kill criteria on
2026-06-10. Any future candidate strategy must pass a fresh pre-registration and
a ≥6-month paper trade before a dollar of this portfolio moves.

## 9. Amendment log

| Date written | Effective | Change | Reason |
|---|---|---|---|
| | | | |
