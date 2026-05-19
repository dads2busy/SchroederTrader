"""Basket paper-trading pipeline entry point.

Software-simulated: maintains its own portfolio state in basket rows of
portfolio.csv. Does NOT call Alpaca. The SPY-only pipeline at 21:30 UTC
reads basket state and renders the unified email.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from schroeder_trader.config import (
    DB_PATH, FEATURES_CSV_PATH, PROJECT_ROOT, SHADOW_BASKET_WEIGHTS,
)
from schroeder_trader.storage.csv_store import CsvStore

from schroeder_trader.basket.orchestrator import compute_decisions
from schroeder_trader.basket.portfolio import (
    bootstrap_starting_value,
    load_basket_broker,
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

    # 1. Download external features (idempotent, skips if <24h old), then load.
    try:
        logger.info("Downloading external features...")
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "backtest" / "download_features.py")],
            cwd=str(PROJECT_ROOT), capture_output=True, timeout=120,
        )
        if result.returncode != 0:
            stdout_snippet = (result.stdout or b"").decode(errors="replace")[-500:]
            stderr_snippet = (result.stderr or b"").decode(errors="replace")[-500:]
            logger.warning(
                "External feature download failed (rc=%d)\nstderr: %s\nstdout: %s",
                result.returncode, stderr_snippet, stdout_snippet,
            )
    except subprocess.TimeoutExpired:
        logger.warning("External feature download timed out after 120s, using cached data")
    except Exception:
        logger.warning("External feature download failed, using cached data", exc_info=True)

    ext_df = pd.DataFrame()
    if FEATURES_CSV_PATH.exists():
        ext_df = pd.read_csv(str(FEATURES_CSV_PATH), index_col="date", parse_dates=True)

    # 2. Get current portfolio value for orchestrator (used by trailing stop).
    #    Use the previously-saved total_value before prices are loaded.
    portfolio_value = bootstrap_starting_value(store)

    # 3. Compute per-ticker decisions (logs shadow_signals rows with pipeline='basket').
    decisions = compute_decisions(store, weights, ext_df, now, portfolio_value)

    # 4. Build prices dict from decisions and load the simulated broker.
    prices = {t: decisions[t]["price"] for t in weights}
    broker = load_basket_broker(store, prices)

    # 5. Rebalance to targets — updates broker state, logs orders.
    rebalance_to_targets(store, broker, weights, decisions, now)

    # 6. Write per-ticker portfolio snapshot from broker's final state.
    write_basket_portfolio_snapshot(store, broker, list(weights), now)

    logger.info("Basket pipeline complete. Wrote %d ticker rows.", len(weights))


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    run_basket_pipeline()


if __name__ == "__main__":
    main()
