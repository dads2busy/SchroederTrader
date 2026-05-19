"""One-off: BUY 141 SPY to restore the position the basket pipeline sold
during its broken pre-SimulatedBroker live run on 2026-05-19.

Invoked by .github/workflows/restore-spy-position.yml on 2026-05-20.
Exits cleanly (success) if:
  - Today isn't 2026-05-20 ET (so the cron never fires destructively in
    a future year)
  - SPY position is already ≥ 141 (idempotent)
  - Alpaca rejects the order due to insufficient buying power (the
    workflow's cron retries every 5 minutes for ~20 minutes total)

Run:
    uv run python scripts/restore_spy_position.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from zoneinfo import ZoneInfo

from schroeder_trader.execution.broker import (
    get_account, get_position, submit_order,
)
from schroeder_trader.risk.risk_manager import OrderRequest

TARGET_DATE = "2026-05-20"
TARGET_QTY = 141
TICKER = "SPY"


def main() -> int:
    today_et = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
    if today_et != TARGET_DATE:
        print(f"Today ({today_et}) is not the target date ({TARGET_DATE}). Skipping.")
        return 0

    current_qty = get_position(TICKER)
    if current_qty >= TARGET_QTY:
        print(f"{TICKER} position already {current_qty} ≥ target {TARGET_QTY}. Nothing to do.")
        return 0

    account = get_account()
    cash = float(account.get("cash", 0) or 0)
    buying_power = float(account.get("buying_power", 0) or 0)
    needed_qty = TARGET_QTY - current_qty
    print(f"Pre-order: {TICKER}={current_qty}, cash=${cash:,.2f}, buying_power=${buying_power:,.2f}")
    print(f"Attempting BUY {needed_qty} {TICKER}...")

    req = OrderRequest(action="BUY", quantity=needed_qty)
    try:
        result = submit_order(req, TICKER)
        print(f"SUCCESS: alpaca_order_id={result.alpaca_order_id}, status={result.status}")
    except Exception as exc:
        # Exit 0 so cron keeps retrying the next scheduled tick.
        print(f"REJECTED (will retry on next cron tick if scheduled): {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
