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
        CREATE TABLE IF NOT EXISTS llm_shadow_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            ticker TEXT NOT NULL,
            close_price REAL NOT NULL,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            action TEXT NOT NULL,
            target_exposure REAL,
            confidence TEXT,
            regime_assessment TEXT,
            key_drivers TEXT,
            reasoning TEXT,
            raw_response TEXT,
            error TEXT
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
            kelly_qty INTEGER,
            high_water_mark REAL,
            trailing_stop_triggered INTEGER
        )
    """)
    # Defensive migration for existing databases missing new columns
    for col, col_type in [
        ("regime", "TEXT"), ("signal_source", "TEXT"), ("bear_day_count", "INTEGER"),
        ("kelly_fraction", "REAL"), ("kelly_qty", "INTEGER"),
        ("high_water_mark", "REAL"), ("trailing_stop_triggered", "INTEGER"),
    ]:
        try:
            conn.execute(f"ALTER TABLE shadow_signals ADD COLUMN {col} {col_type}")
        except sqlite3.OperationalError:
            pass  # column already exists
    for col, col_type in [
        ("signal_close_price", "REAL"), ("slippage", "REAL"),
    ]:
        try:
            conn.execute(f"ALTER TABLE orders ADD COLUMN {col} {col_type}")
        except sqlite3.OperationalError:
            pass  # column already exists
    for col, col_type in [("target_exposure", "REAL")]:
        try:
            conn.execute(f"ALTER TABLE llm_shadow_signals ADD COLUMN {col} {col_type}")
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
    signal_close_price: float | None = None,
) -> int:
    cursor = conn.execute(
        "INSERT INTO orders (signal_id, alpaca_order_id, timestamp, ticker, action, quantity, status, signal_close_price) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (signal_id, alpaca_order_id, timestamp.isoformat(), ticker, action, quantity, status, signal_close_price),
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


def get_portfolio_by_date(conn: sqlite3.Connection, date_str: str) -> dict | None:
    """Used for run-level idempotency: a portfolio snapshot only exists after
    a full pipeline run reached step 10, so its presence means today is done.
    A partial run that crashed earlier (e.g. between log_signal and log_order)
    leaves no portfolio row, which lets the next invocation re-run cleanly."""
    row = conn.execute(
        "SELECT * FROM portfolio WHERE timestamp LIKE ?", (f"{date_str}%",)
    ).fetchone()
    if row is None:
        return None
    return dict(row)


def get_pending_orders(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM orders WHERE status = 'SUBMITTED'"
    ).fetchall()
    return [dict(row) for row in rows]


def get_order_by_alpaca_id(conn: sqlite3.Connection, alpaca_order_id: str) -> dict | None:
    row = conn.execute(
        "SELECT * FROM orders WHERE alpaca_order_id = ?", (alpaca_order_id,)
    ).fetchone()
    return dict(row) if row else None


def insert_reconciled_order(
    conn: sqlite3.Connection,
    alpaca_order_id: str,
    timestamp: datetime,
    ticker: str,
    action: str,
    quantity: int,
    status: str,
    fill_price: float | None = None,
    fill_timestamp: datetime | None = None,
) -> int:
    # signal_id=0 marks the row as reconciled from Alpaca rather than originated by us.
    cursor = conn.execute(
        "INSERT INTO orders (signal_id, alpaca_order_id, timestamp, ticker, action, "
        "quantity, status, fill_price, fill_timestamp) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            0, alpaca_order_id, timestamp.isoformat(), ticker, action, quantity, status,
            fill_price, fill_timestamp.isoformat() if fill_timestamp else None,
        ),
    )
    conn.commit()
    return cursor.lastrowid


def update_order_fill(
    conn: sqlite3.Connection,
    alpaca_order_id: str,
    fill_price: float,
    fill_timestamp: datetime,
    status: str,
) -> None:
    # Compute slippage if signal_close_price was recorded
    row = conn.execute(
        "SELECT signal_close_price, action FROM orders WHERE alpaca_order_id = ?",
        (alpaca_order_id,),
    ).fetchone()
    slippage = None
    if row and row["signal_close_price"] is not None and fill_price > 0:
        # Positive slippage = cost (paid more on BUY, received less on SELL)
        if row["action"] == "BUY":
            slippage = fill_price - row["signal_close_price"]
        else:
            slippage = row["signal_close_price"] - fill_price
    conn.execute(
        "UPDATE orders SET fill_price = ?, fill_timestamp = ?, status = ?, slippage = ? WHERE alpaca_order_id = ?",
        (fill_price, fill_timestamp.isoformat(), status, slippage, alpaca_order_id),
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
    high_water_mark: float | None = None,
    trailing_stop_triggered: bool | None = None,
) -> int:
    trailing_stop_int = int(trailing_stop_triggered) if trailing_stop_triggered is not None else None
    cursor = conn.execute(
        "INSERT INTO shadow_signals (timestamp, ticker, close_price, predicted_class, predicted_proba, ml_signal, sma_signal, regime, signal_source, bear_day_count, kelly_fraction, kelly_qty, high_water_mark, trailing_stop_triggered) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (timestamp.isoformat(), ticker, close_price, predicted_class, predicted_proba, ml_signal, sma_signal, regime, signal_source, bear_day_count, kelly_fraction, kelly_qty, high_water_mark, trailing_stop_int),
    )
    conn.commit()
    return cursor.lastrowid


def get_shadow_signals(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute("SELECT * FROM shadow_signals ORDER BY id").fetchall()
    return [dict(row) for row in rows]


def log_llm_signal(
    conn: sqlite3.Connection,
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
    import json as _json
    cursor = conn.execute(
        "INSERT INTO llm_shadow_signals (timestamp, ticker, close_price, provider, model, "
        "action, target_exposure, confidence, regime_assessment, key_drivers, reasoning, raw_response, error) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            timestamp.isoformat(), ticker, close_price, provider, model,
            action, target_exposure, confidence, regime_assessment,
            _json.dumps(key_drivers) if key_drivers is not None else None,
            reasoning, raw_response, error,
        ),
    )
    conn.commit()
    return cursor.lastrowid


def get_llm_signals(conn: sqlite3.Connection, limit: int = 60) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM llm_shadow_signals ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    return [dict(row) for row in rows]
