"""Basket paper-trading portfolio state — bootstrap, snapshots, prior reads.

Reads and writes rows in shared CSVs (portfolio.csv, shadow_signals.csv)
filtered to pipeline='basket'. SPY-only rows are ignored except as a
fallback during first-ever basket-pipeline run (see bootstrap_starting_value).
"""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd

from schroeder_trader.storage.csv_store import CsvStore
from schroeder_trader.storage.trade_log import log_portfolio


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
    """Latest decided exposure (0.0 or 1.0) for `ticker` from basket shadow
    signals. Returns 0.0 when no prior basket signal exists (cold start).

    Walks back through HOLD rows until it finds the last BUY or SELL,
    matching the HOLD-carries-forward semantics of _exposure_from_decisions.
    """
    df = store.read("shadow_signals")
    if df.empty:
        return 0.0
    rows = df[(df["pipeline"] == "basket") & (df["ticker"] == ticker)]
    if rows.empty:
        return 0.0
    chain = rows.sort_values("timestamp")
    for _, r in chain[::-1].iterrows():
        s = str(r["ml_signal"])
        if s == "BUY":
            return 1.0
        if s == "SELL":
            return 0.0
    return 0.0


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
    return list(rows.sort_values("date")["date"])


def write_basket_portfolio_snapshot(
    store: CsvStore,
    broker,
    tickers: list[str],
    prices: dict[str, float],
    now: datetime,
) -> None:
    """Write one portfolio.csv row per ticker with pipeline='basket'.

    `prices` is the closing price per ticker used to compute position_value.
    The shared basket cash and total portfolio value are repeated on every
    row (each row is a self-contained snapshot for that ticker).
    """
    account = broker.get_account()
    cash = float(account["cash"])
    total_value = float(account["portfolio_value"])
    for ticker in tickers:
        qty = int(broker.get_position(ticker))
        position_value = qty * prices[ticker]
        log_portfolio(
            store, now, cash, qty, position_value, total_value,
            pipeline="basket", ticker=ticker,
        )
