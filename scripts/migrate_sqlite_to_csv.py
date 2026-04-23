"""One-off migration: dump each SQLite table in data/trades.db to data/<table>.csv.

Run once to carry existing state (orders, portfolio, shadow + llm signals) into
the new CSV-backed storage. Safe to re-run: it overwrites the CSVs atomically.

Usage:
    uv run python scripts/migrate_sqlite_to_csv.py
"""

import sqlite3
import sys
from pathlib import Path

import pandas as pd

from schroeder_trader.config import DB_PATH


TABLES = ["signals", "orders", "portfolio", "shadow_signals", "llm_shadow_signals"]


def main():
    db = Path(DB_PATH)
    if not db.exists():
        print(f"No SQLite DB at {db}; nothing to migrate.")
        sys.exit(0)

    root = db.parent
    root.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db))
    try:
        for table in TABLES:
            try:
                df = pd.read_sql_query(f"SELECT * FROM {table}", con)
            except pd.io.sql.DatabaseError:
                print(f"  {table}: (no such table, skipping)")
                continue
            dest = root / f"{table}.csv"
            tmp = dest.with_suffix(".csv.tmp")
            df.to_csv(tmp, index=False)
            tmp.replace(dest)
            print(f"  {table}: {len(df)} rows → {dest}")
    finally:
        con.close()

    print("\nMigration complete. The SQLite file is untouched; delete it once you")
    print("have verified a few runs against the CSVs.")


if __name__ == "__main__":
    main()
