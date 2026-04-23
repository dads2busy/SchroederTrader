"""CSV-backed replacement for the old SQLite trade log.

Public function signatures match the original SQLite version: callers pass a
store object as the first argument (previously `sqlite3.Connection`, now
`CsvStore`) and the rest of the code path is unchanged.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from schroeder_trader.storage.csv_store import CsvStore


def init_db(db_path: Path) -> CsvStore:
    """Open the CSV-backed store. Accepts the legacy db_path and uses its
    parent directory as the data root so existing callers don't change."""
    root = Path(db_path).parent
    return CsvStore(root)


def log_signal(
    store: CsvStore,
    timestamp: datetime,
    ticker: str,
    close_price: float,
    sma_50: float,
    sma_200: float,
    signal: str,
) -> int:
    return store.append("signals", {
        "timestamp": timestamp.isoformat(),
        "ticker": ticker,
        "close_price": close_price,
        "sma_50": sma_50,
        "sma_200": sma_200,
        "signal": signal,
    })


def log_order(
    store: CsvStore,
    signal_id: int,
    alpaca_order_id: str,
    timestamp: datetime,
    ticker: str,
    action: str,
    quantity: int,
    status: str,
    signal_close_price: float | None = None,
) -> int:
    return store.append("orders", {
        "signal_id": signal_id,
        "alpaca_order_id": alpaca_order_id,
        "timestamp": timestamp.isoformat(),
        "ticker": ticker,
        "action": action,
        "quantity": quantity,
        "fill_price": None,
        "fill_timestamp": None,
        "status": status,
        "signal_close_price": signal_close_price,
        "slippage": None,
    })


def log_portfolio(
    store: CsvStore,
    timestamp: datetime,
    cash: float,
    position_qty: int,
    position_value: float,
    total_value: float,
) -> int:
    return store.append("portfolio", {
        "timestamp": timestamp.isoformat(),
        "cash": cash,
        "position_qty": position_qty,
        "position_value": position_value,
        "total_value": total_value,
    })


def get_signal_by_date(store: CsvStore, date_str: str) -> dict | None:
    df = store.read("signals")
    if df.empty:
        return None
    matches = df[df["timestamp"].astype(str).str.startswith(date_str)]
    if matches.empty:
        return None
    return matches.iloc[0].to_dict()


def get_portfolio_by_date(store: CsvStore, date_str: str) -> dict | None:
    df = store.read("portfolio")
    if df.empty:
        return None
    matches = df[df["timestamp"].astype(str).str.startswith(date_str)]
    if matches.empty:
        return None
    return matches.iloc[0].to_dict()


def get_pending_orders(store: CsvStore) -> list[dict]:
    df = store.read("orders")
    if df.empty:
        return []
    pending = df[df["status"] == "SUBMITTED"]
    return [_row_to_dict(r) for _, r in pending.iterrows()]


def get_order_by_alpaca_id(store: CsvStore, alpaca_order_id: str) -> dict | None:
    df = store.read("orders")
    if df.empty:
        return None
    matches = df[df["alpaca_order_id"] == alpaca_order_id]
    if matches.empty:
        return None
    return _row_to_dict(matches.iloc[0])


def insert_reconciled_order(
    store: CsvStore,
    alpaca_order_id: str,
    timestamp: datetime,
    ticker: str,
    action: str,
    quantity: int,
    status: str,
    fill_price: float | None = None,
    fill_timestamp: datetime | None = None,
) -> int:
    # signal_id=0 marks reconciled-orphan, same convention as the SQLite version
    return store.append("orders", {
        "signal_id": 0,
        "alpaca_order_id": alpaca_order_id,
        "timestamp": timestamp.isoformat(),
        "ticker": ticker,
        "action": action,
        "quantity": quantity,
        "fill_price": fill_price,
        "fill_timestamp": fill_timestamp.isoformat() if fill_timestamp else None,
        "status": status,
        "signal_close_price": None,
        "slippage": None,
    })


def update_order_fill(
    store: CsvStore,
    alpaca_order_id: str,
    fill_price: float,
    fill_timestamp: datetime,
    status: str,
) -> None:
    existing = get_order_by_alpaca_id(store, alpaca_order_id)
    slippage = None
    if existing and existing.get("signal_close_price") is not None and fill_price > 0:
        try:
            signal_close = float(existing["signal_close_price"])
            if existing["action"] == "BUY":
                slippage = fill_price - signal_close
            else:
                slippage = signal_close - fill_price
        except (TypeError, ValueError):
            slippage = None
    store.update_where(
        "orders",
        where={"alpaca_order_id": alpaca_order_id},
        set_={
            "fill_price": fill_price,
            "fill_timestamp": fill_timestamp.isoformat(),
            "status": status,
            "slippage": slippage,
        },
    )


def log_shadow_signal(
    store: CsvStore,
    timestamp: datetime,
    ticker: str,
    close_price: float,
    predicted_class: int | None,
    predicted_proba: str | None,
    ml_signal: str,
    sma_signal: str,
    regime: str | None = None,
    signal_source: str | None = None,
    bear_day_count: int | None = None,
    kelly_fraction: float | None = None,
    kelly_qty: int | None = None,
    high_water_mark: float | None = None,
    trailing_stop_triggered: bool | None = None,
) -> int:
    ts_int = int(trailing_stop_triggered) if trailing_stop_triggered is not None else None
    return store.append("shadow_signals", {
        "timestamp": timestamp.isoformat(),
        "ticker": ticker,
        "close_price": close_price,
        "predicted_class": predicted_class,
        "predicted_proba": predicted_proba,
        "ml_signal": ml_signal,
        "sma_signal": sma_signal,
        "regime": regime,
        "signal_source": signal_source,
        "bear_day_count": bear_day_count,
        "kelly_fraction": kelly_fraction,
        "kelly_qty": kelly_qty,
        "high_water_mark": high_water_mark,
        "trailing_stop_triggered": ts_int,
    })


def get_shadow_signals(store: CsvStore) -> list[dict]:
    df = store.read("shadow_signals")
    if df.empty:
        return []
    df = df.sort_values("id")
    return [_row_to_dict(r) for _, r in df.iterrows()]


def log_llm_signal(
    store: CsvStore,
    timestamp: datetime,
    ticker: str,
    close_price: float,
    provider: str,
    model: str,
    action: str,
    target_exposure: float | None,
    confidence: str | None,
    regime_assessment: str | None,
    key_drivers: list[str] | None,
    reasoning: str | None,
    raw_response: str | None,
    error: str | None = None,
) -> int:
    return store.append("llm_shadow_signals", {
        "timestamp": timestamp.isoformat(),
        "ticker": ticker,
        "close_price": close_price,
        "provider": provider,
        "model": model,
        "action": action,
        "target_exposure": target_exposure,
        "confidence": confidence,
        "regime_assessment": regime_assessment,
        "key_drivers": json.dumps(key_drivers) if key_drivers is not None else None,
        "reasoning": reasoning,
        "raw_response": raw_response,
        "error": error,
    })


def get_llm_signals(store: CsvStore, limit: int = 60) -> list[dict]:
    df = store.read("llm_shadow_signals")
    if df.empty:
        return []
    df = df.sort_values("id", ascending=False).head(limit)
    return [_row_to_dict(r) for _, r in df.iterrows()]


# Direct-read helpers used by main.py (replacing raw conn.execute calls).

def get_latest_trailing_stop_state(store: CsvStore) -> dict | None:
    """Most recent shadow_signals row with a non-null high_water_mark."""
    df = store.read("shadow_signals")
    if df.empty or "high_water_mark" not in df.columns:
        return None
    df = df[df["high_water_mark"].notna()]
    if df.empty:
        return None
    row = df.sort_values("id", ascending=False).iloc[0]
    return _row_to_dict(row)


def get_shadow_signal_timestamps(store: CsvStore) -> list[str]:
    """All timestamps in shadow_signals, ordered by id ascending.
    Used by trailing stop for trading-date history."""
    df = store.read("shadow_signals")
    if df.empty:
        return []
    df = df.sort_values("id")
    return df["timestamp"].astype(str).tolist()


def _row_to_dict(row: pd.Series) -> dict:
    """Convert a pandas row to a plain dict, normalizing NaN to None."""
    out = {}
    for k, v in row.to_dict().items():
        if isinstance(v, float) and pd.isna(v):
            out[k] = None
        elif v is pd.NA:
            out[k] = None
        else:
            out[k] = v
    return out
