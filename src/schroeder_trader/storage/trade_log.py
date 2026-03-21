import sqlite3
from datetime import datetime
from pathlib import Path


def init_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            ticker TEXT NOT NULL,
            close_price REAL NOT NULL,
            sma_50 REAL NOT NULL,
            sma_200 REAL NOT NULL,
            signal TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_id INTEGER NOT NULL,
            alpaca_order_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            ticker TEXT NOT NULL,
            action TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            fill_price REAL,
            fill_timestamp TEXT,
            status TEXT NOT NULL,
            FOREIGN KEY (signal_id) REFERENCES signals(id)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS portfolio (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            cash REAL NOT NULL,
            position_qty INTEGER NOT NULL,
            position_value REAL NOT NULL,
            total_value REAL NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS shadow_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            ticker TEXT NOT NULL,
            close_price REAL NOT NULL,
            predicted_class INTEGER,
            predicted_proba TEXT,
            ml_signal TEXT NOT NULL,
            sma_signal TEXT NOT NULL,
            regime TEXT,
            signal_source TEXT,
            bear_day_count INTEGER,
            kelly_fraction REAL,
            kelly_qty INTEGER
        )
    """)
    # Defensive migration for existing databases missing new columns
    for col, col_type in [
        ("regime", "TEXT"), ("signal_source", "TEXT"), ("bear_day_count", "INTEGER"),
        ("kelly_fraction", "REAL"), ("kelly_qty", "INTEGER"),
    ]:
        try:
            conn.execute(f"ALTER TABLE shadow_signals ADD COLUMN {col} {col_type}")
        except sqlite3.OperationalError:
            pass  # column already exists
    conn.commit()
    return conn


def log_signal(
    conn: sqlite3.Connection,
    timestamp: datetime,
    ticker: str,
    close_price: float,
    sma_50: float,
    sma_200: float,
    signal: str,
) -> int:
    cursor = conn.execute(
        "INSERT INTO signals (timestamp, ticker, close_price, sma_50, sma_200, signal) VALUES (?, ?, ?, ?, ?, ?)",
        (timestamp.isoformat(), ticker, close_price, sma_50, sma_200, signal),
    )
    conn.commit()
    return cursor.lastrowid


def log_order(
    conn: sqlite3.Connection,
    signal_id: int,
    alpaca_order_id: str,
    timestamp: datetime,
    ticker: str,
    action: str,
    quantity: int,
    status: str,
) -> int:
    cursor = conn.execute(
        "INSERT INTO orders (signal_id, alpaca_order_id, timestamp, ticker, action, quantity, status) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (signal_id, alpaca_order_id, timestamp.isoformat(), ticker, action, quantity, status),
    )
    conn.commit()
    return cursor.lastrowid


def log_portfolio(
    conn: sqlite3.Connection,
    timestamp: datetime,
    cash: float,
    position_qty: int,
    position_value: float,
    total_value: float,
) -> int:
    cursor = conn.execute(
        "INSERT INTO portfolio (timestamp, cash, position_qty, position_value, total_value) VALUES (?, ?, ?, ?, ?)",
        (timestamp.isoformat(), cash, position_qty, position_value, total_value),
    )
    conn.commit()
    return cursor.lastrowid


def get_signal_by_date(conn: sqlite3.Connection, date_str: str) -> dict | None:
    row = conn.execute(
        "SELECT * FROM signals WHERE timestamp LIKE ?", (f"{date_str}%",)
    ).fetchone()
    if row is None:
        return None
    return dict(row)


def get_pending_orders(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM orders WHERE status = 'SUBMITTED'"
    ).fetchall()
    return [dict(row) for row in rows]


def update_order_fill(
    conn: sqlite3.Connection,
    alpaca_order_id: str,
    fill_price: float,
    fill_timestamp: datetime,
    status: str,
) -> None:
    conn.execute(
        "UPDATE orders SET fill_price = ?, fill_timestamp = ?, status = ? WHERE alpaca_order_id = ?",
        (fill_price, fill_timestamp.isoformat(), status, alpaca_order_id),
    )
    conn.commit()


def log_shadow_signal(
    conn: sqlite3.Connection,
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
) -> int:
    cursor = conn.execute(
        "INSERT INTO shadow_signals (timestamp, ticker, close_price, predicted_class, predicted_proba, ml_signal, sma_signal, regime, signal_source, bear_day_count, kelly_fraction, kelly_qty) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (timestamp.isoformat(), ticker, close_price, predicted_class, predicted_proba, ml_signal, sma_signal, regime, signal_source, bear_day_count, kelly_fraction, kelly_qty),
    )
    conn.commit()
    return cursor.lastrowid


def get_shadow_signals(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute("SELECT * FROM shadow_signals ORDER BY id").fetchall()
    return [dict(row) for row in rows]
