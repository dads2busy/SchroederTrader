"""Refresh basket-pipeline rows in the local data CSVs from origin/main.

The basket pipeline runs ONLY in CI (`daily-basket` workflow) and commits its
state to origin/main. The local launchd pipeline reads basket rows from the
local working tree but never pulls, so the daily email shows a stale basket
(frozen at the last-pulled snapshot). This script pulls just the basket rows
from origin/main into the local CSVs, leaving every other row (the uncommitted
local `spy_only` writes) untouched.

Why not `git pull`: the local working tree is perpetually dirty (the pipeline
appends spy_only rows without committing), so a merge/rebase would conflict.
This row-level refresh sidesteps git's line-merge entirely.

Safe + idempotent. Non-fatal by design: on any error it leaves local data as-is
and exits 0, so it can never block the daily trading run.
"""
from __future__ import annotations

import io
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "data"
FILES = ["portfolio.csv", "shadow_signals.csv"]
REF = "origin/main"


def merge_basket_rows(local: pd.DataFrame, origin: pd.DataFrame) -> pd.DataFrame:
    """Return `local` with its basket rows replaced by `origin`'s basket rows.

    Non-basket local rows (spy_only, etc.) are preserved untouched and kept
    first, matching the original file ordering (spy_only rows, then basket).
    If origin has no basket rows or either frame lacks a pipeline column,
    returns local unchanged.
    """
    if "pipeline" not in origin.columns or "pipeline" not in local.columns:
        return local
    origin_basket = origin[origin["pipeline"] == "basket"]
    if origin_basket.empty:
        return local
    local_non_basket = local[local["pipeline"] != "basket"]
    return pd.concat([local_non_basket, origin_basket], ignore_index=True)


def _read_origin(rel: str) -> pd.DataFrame | None:
    try:
        out = subprocess.run(
            ["git", "show", f"{REF}:data/{rel}"],
            capture_output=True, text=True, check=True, cwd=REPO_ROOT,
        ).stdout
        return pd.read_csv(io.StringIO(out))
    except (subprocess.CalledProcessError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return None


def sync_file(rel: str) -> str:
    local_path = DATA / rel
    if not local_path.exists():
        return f"{rel}: local missing, skip"
    origin = _read_origin(rel)
    if origin is None or "pipeline" not in origin.columns:
        return f"{rel}: no origin basket data, skip"
    local = pd.read_csv(local_path)
    merged = merge_basket_rows(local, origin)
    if merged.equals(local):
        return f"{rel}: already current, no change"
    merged.to_csv(local_path, index=False)
    latest = merged.loc[merged["pipeline"] == "basket", "timestamp"].max()
    return f"{rel}: refreshed basket rows (latest {latest})"


def main() -> int:
    subprocess.run(["git", "fetch", "origin", "main"], capture_output=True, cwd=REPO_ROOT)
    for rel in FILES:
        print(sync_file(rel))
    return 0


if __name__ == "__main__":
    sys.exit(main())
