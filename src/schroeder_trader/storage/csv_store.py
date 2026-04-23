"""File-backed CSV storage used in place of SQLite.

Each logical "table" is one CSV file under a root directory. Callers pass a
CsvStore instance where they previously passed a sqlite3.Connection — the
trade_log helpers encapsulate the read/append/update pattern on top.

This design is deliberate: CSVs + git commits make the state auditable and
diffable in a way SQLite can't, and eliminate the file-lock concerns that
would otherwise come with a workflow environment.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class CsvStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, table: str) -> Path:
        return self.root / f"{table}.csv"

    def read(self, table: str) -> pd.DataFrame:
        """Read a table. Returns an empty DataFrame if the file doesn't exist."""
        p = self.path(table)
        if not p.exists() or p.stat().st_size == 0:
            return pd.DataFrame()
        return pd.read_csv(p)

    def write(self, table: str, df: pd.DataFrame) -> None:
        """Atomically replace a table (temp file + rename)."""
        p = self.path(table)
        tmp = p.with_suffix(".csv.tmp")
        df.to_csv(tmp, index=False)
        tmp.replace(p)

    def append(self, table: str, row: dict, *, id_col: str = "id") -> int:
        """Append a row, auto-assigning an integer id. Returns the new id."""
        df = self.read(table)
        if id_col in df.columns and len(df) > 0:
            next_id = int(df[id_col].max()) + 1
        else:
            next_id = 1
        row = {id_col: next_id, **row}
        new_df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        self.write(table, new_df)
        return next_id

    def update_where(self, table: str, *, where: dict, set_: dict) -> int:
        """Update rows matching `where` with fields from `set_`. Returns row count updated."""
        df = self.read(table)
        if df.empty:
            return 0
        mask = pd.Series([True] * len(df))
        for k, v in where.items():
            mask &= df[k] == v
        if not mask.any():
            return 0
        for k, v in set_.items():
            if k not in df.columns:
                df[k] = None
            # If assigning a non-numeric value to a numeric column (e.g. ISO
            # timestamp into a previously all-null float column), coerce the
            # column to object first so pandas doesn't reject the assignment.
            if v is not None and not isinstance(v, (int, float, bool)):
                df[k] = df[k].astype(object)
            df.loc[mask, k] = v
        self.write(table, df)
        return int(mask.sum())

    def close(self) -> None:
        """No-op; kept for sqlite3.Connection API compatibility."""
        return None
