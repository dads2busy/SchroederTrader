"""Basket paper-trading portfolio state — bootstrap, snapshots, prior reads.

Reads and writes rows in shared CSVs (portfolio.csv, shadow_signals.csv)
filtered to pipeline='basket'. SPY-only rows are ignored except as a
fallback during first-ever basket-pipeline run (see bootstrap_starting_value
and load_basket_broker).
"""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd

from schroeder_trader.storage.csv_store import CsvStore
from schroeder_trader.storage.trade_log import log_portfolio


class SimulatedBroker:
    """Pure-software broker for the basket's paper-trading simulation.

    Tracks cash and per-ticker positions in memory during a single
    pipeline run. State is loaded from portfolio.csv basket rows at the
    start, mutated by submit_order during rebalancing, and persisted
    back to portfolio.csv via write_basket_portfolio_snapshot.

    Does NOT call Alpaca. Switching to live trading later (Phase E)
    would replace this with the real broker module.
    """

    def __init__(self, cash: float, positions: dict[str, int], prices: dict[str, float]):
        self.cash = float(cash)
        self.positions: dict[str, int] = dict(positions)
        self.prices: dict[str, float] = dict(prices)

    def get_position(self, ticker: str) -> int:
        return int(self.positions.get(ticker, 0))

    def get_account(self) -> dict:
        equity = sum(qty * self.prices.get(t, 0.0) for t, qty in self.positions.items())
        return {"cash": self.cash, "portfolio_value": self.cash + equity}

    def submit_order(self, ticker: str, action: str, qty: int) -> dict:
        signed_qty = qty if action == "BUY" else -qty
        price = self.prices[ticker]
        self.positions[ticker] = self.get_position(ticker) + signed_qty
        self.cash -= signed_qty * price
        return {"alpaca_order_id": f"sim-{ticker}-{signed_qty}", "status": "FILLED"}


def load_basket_broker(store: CsvStore, prices: dict[str, float]) -> SimulatedBroker:
    """Load the basket's simulated broker state from portfolio.csv.

    Warm start: if pipeline='basket' rows exist, read cash and per-ticker
    positions from the latest snapshot (one row per ticker, all sharing
    the same timestamp).

    Cold start: no basket rows. Use SPY-only's latest total_value as
    starting cash and empty positions.
    """
    df = store.read("portfolio")
    if df.empty:
        raise RuntimeError("no portfolio history to bootstrap from")

    basket = df[df["pipeline"] == "basket"]
    if not basket.empty:
        latest_ts = basket["timestamp"].max()
        latest_rows = basket[basket["timestamp"] == latest_ts]
        cash = float(latest_rows.iloc[0]["cash"])
        positions = {row["ticker"]: int(row["position_qty"])
                     for _, row in latest_rows.iterrows()}
        return SimulatedBroker(cash=cash, positions=positions, prices=prices)

    spy_only = df[df["pipeline"] == "spy_only"]
    if spy_only.empty:
        raise RuntimeError("no portfolio history to bootstrap from")
    latest = spy_only.sort_values("timestamp").iloc[-1]
    return SimulatedBroker(
        cash=float(latest["total_value"]),  # cold start: all cash, no positions
        positions={},
        prices=prices,
    )


def bootstrap_starting_value(store: CsvStore) -> float:
    """Return the basket pipeline's starting portfolio value.

    If any pipeline='basket' rows exist, return the latest basket total_value.
    Otherwise, fall back to the latest pipeline='spy_only' total_value
    (this is the first-ever basket run — mirror real portfolio size).
    Raise RuntimeError if neither has any data.
    """
    df = store.read("portfolio")
    if df.empty:
        raise RuntimeError("no portfolio history to bootstrap from")
    basket = df[df["pipeline"] == "basket"]
    if not basket.empty:
        latest = basket.sort_values("timestamp").iloc[-1]
        return float(latest["total_value"])
    spy_only = df[df["pipeline"] == "spy_only"]
    if not spy_only.empty:
        latest = spy_only.sort_values("timestamp").iloc[-1]
        return float(latest["total_value"])
    raise RuntimeError("no portfolio history to bootstrap from")


def read_position_qty(store: CsvStore, ticker: str) -> int:
    """Latest basket-pipeline position quantity for `ticker`, or 0 if none."""
    df = store.read("portfolio")
    if df.empty:
        return 0
    rows = df[(df["pipeline"] == "basket") & (df["ticker"] == ticker)]
    if rows.empty:
        return 0
    latest = rows.sort_values("timestamp").iloc[-1]
    return int(latest["position_qty"])


def prior_exposure(store: CsvStore, ticker: str) -> float:
    """Return 1.0 if the basket currently holds `ticker` (latest basket
    portfolio row has position_qty > 0), else 0.0.

    Position-based semantics: "invested iff holding shares." Used by the
    orchestrator's HOLD branch to carry forward existing positions. This
    is intentionally NOT signal-based — the cold-start force-invest
    establishes positions on HOLD signals, and the position is what we
    want to carry forward, not the signal that happened to fire that day.
    """
    df = store.read("portfolio")
    if df.empty:
        return 0.0
    rows = df[(df["pipeline"] == "basket") & (df["ticker"] == ticker)]
    if rows.empty:
        return 0.0
    latest = rows.sort_values("timestamp").iloc[-1]
    qty = int(latest["position_qty"]) if pd.notna(latest["position_qty"]) else 0
    return 1.0 if qty > 0 else 0.0


def is_basket_cold_start(store: CsvStore) -> bool:
    """Return True iff there are no pipeline='basket' rows in portfolio.csv.

    Used by the orchestrator to detect the basket's first-ever run and
    force-invest to target weights regardless of signal. Once any basket
    row exists, the standard HOLD-carries-prior semantics take over.
    """
    df = store.read("portfolio")
    if df.empty:
        return True
    return bool((df["pipeline"] == "basket").sum() == 0)


def read_trading_dates(store: CsvStore, ticker: str) -> list[date]:
    """ET-local dates for which a basket-pipeline shadow signal exists for
    `ticker`, in ascending order. Used by the per-ticker TrailingStop's
    cooldown logic."""
    df = store.read("shadow_signals")
    if df.empty:
        return []
    rows = df[(df["pipeline"] == "basket") & (df["ticker"] == ticker)].copy()
    if rows.empty:
        return []
    rows["date"] = pd.to_datetime(rows["timestamp"], utc=True, format="ISO8601") \
        .dt.tz_convert("America/New_York").dt.date
    return sorted(set(rows["date"]))


def write_basket_portfolio_snapshot(
    store: CsvStore,
    broker: SimulatedBroker,
    tickers: list[str],
    now: datetime,
) -> None:
    """Write one portfolio.csv row per ticker with pipeline='basket'.

    Reads final cash, positions, and total_value from the SimulatedBroker
    after rebalancing. Each row is a self-contained snapshot.
    """
    account = broker.get_account()
    cash = float(account["cash"])
    total_value = float(account["portfolio_value"])
    for ticker in tickers:
        qty = broker.get_position(ticker)
        position_value = qty * broker.prices.get(ticker, 0.0)
        log_portfolio(
            store, now, cash, qty, position_value, total_value,
            pipeline="basket", ticker=ticker,
        )
