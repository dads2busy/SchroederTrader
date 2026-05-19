"""One-time migration: add a `pipeline` column (default 'spy_only') and a
`ticker` column (default 'SPY' where missing) to portfolio.csv, orders.csv,
and shadow_signals.csv. Idempotent — re-running is a no-op.

Run:
    uv run python -m scripts.migrate_portfolio_to_pipeline_column
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

_FILES = [
    ("portfolio.csv", {"pipeline": "spy_only", "ticker": "SPY"}),
    ("orders.csv", {"pipeline": "spy_only"}),
    ("shadow_signals.csv", {"pipeline": "spy_only"}),
]


def migrate(data_dir: Path = DEFAULT_DATA_DIR) -> None:
    for filename, defaults in _FILES:
        path = data_dir / filename
        if not path.exists():
            continue
        df = pd.read_csv(path)
        changed = False
        for col, default in defaults.items():
            if col not in df.columns:
                if col == "pipeline":
                    insert_at = 2 if "timestamp" in df.columns else 1
                else:
                    insert_at = min(3, len(df.columns))
                df.insert(insert_at, col, default)
                changed = True
        if changed:
            df.to_csv(path, index=False)


if __name__ == "__main__":
    migrate()
    print("Migration complete.")
