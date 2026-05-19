"""Basket paper-trading pipeline entry point.

Invoked by .github/workflows/daily-basket.yml at 21:20 UTC.
Computes per-ticker decisions, rebalances to target weights, writes per-
ticker portfolio snapshots. Does NOT send email — the SPY-only pipeline
(running at 21:30 UTC) reads basket state and renders the unified email.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from schroeder_trader.config import (
    DB_PATH, FEATURES_CSV_PATH, SHADOW_BASKET_WEIGHTS,
)
from schroeder_trader.execution.broker import get_position, get_account
from schroeder_trader.execution.reconcile import reconcile_orders
from schroeder_trader.storage.csv_store import CsvStore

from schroeder_trader.basket.orchestrator import compute_decisions
from schroeder_trader.basket.portfolio import (
    bootstrap_starting_value,
    write_basket_portfolio_snapshot,
)
from schroeder_trader.basket.rebalance import rebalance_to_targets

logger = logging.getLogger(__name__)


def run_basket_pipeline(
    data_dir: Path | None = None,
    weights: dict[str, float] | None = None,
    now: datetime | None = None,
) -> None:
    """Run one daily basket pipeline iteration.

    Arguments are exposed for testing; production calls use defaults.
    """
    if data_dir is None:
        data_dir = Path(DB_PATH).parent
    if weights is None:
        weights = SHADOW_BASKET_WEIGHTS
    if now is None:
        now = datetime.now(timezone.utc)

    store = CsvStore(data_dir)

    # 1. Reconcile orphaned orders per ticker (paper account).
    for ticker in weights:
        try:
            reconcile_orders(store, ticker)
        except Exception:
            logger.exception("Reconcile failed for %s (non-fatal)", ticker)

    # 2. Bootstrap portfolio value.
    portfolio_value = bootstrap_starting_value(store)

    # 3. Load extended features once.
    ext_df = pd.DataFrame()
    if FEATURES_CSV_PATH.exists():
        ext_df = pd.read_csv(str(FEATURES_CSV_PATH), index_col="date", parse_dates=True)

    # 4. Compute per-ticker decisions (logs shadow_signals rows with pipeline='basket').
    decisions = compute_decisions(store, weights, ext_df, now, portfolio_value)

    # 5. Read current positions from broker.
    current_positions = {t: int(get_position(t)) for t in weights}

    # 6. Rebalance to targets — submit orders, log to orders.csv with pipeline='basket'.
    rebalance_to_targets(
        store, portfolio_value, weights, decisions, current_positions, now,
    )

    # 7. Read updated positions from broker after fills.
    prices = {t: decisions[t]["price"] for t in weights}

    class _BrokerAdapter:
        @staticmethod
        def get_position(t):
            return get_position(t)
        @staticmethod
        def get_account():
            return get_account()

    write_basket_portfolio_snapshot(store, _BrokerAdapter, list(weights), prices, now)

    logger.info("Basket pipeline complete. Wrote %d ticker rows.", len(weights))


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    run_basket_pipeline()


if __name__ == "__main__":
    main()
