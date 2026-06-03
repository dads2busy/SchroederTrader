#!/bin/bash
# Daily launchd entrypoint for the local SPY-only pipeline.
#
# 1. Refresh basket-pipeline rows from origin/main so the email's BASKET PAPER
#    section is current (the basket runs only in CI; the local repo never pulls).
#    Non-fatal: a sync failure (offline, etc.) must NEVER block the trading run.
# 2. Run the pipeline, which builds and sends the daily email.
#
# Installed via the com.schroedertrader.daily launchd job.
set -uo pipefail

REPO="/Users/ads7fg/git/SchroederTrader"
PYTHON="$REPO/.venv/bin/python"
LOG="$REPO/logs/run_daily.log"

cd "$REPO" || exit 1
mkdir -p "$REPO/logs"

{
  echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) run_daily start ==="
  if "$PYTHON" "$REPO/scripts/sync_basket_state.py"; then
    echo "basket sync ok"
  else
    echo "basket sync FAILED (non-fatal, proceeding with local data)"
  fi
} >>"$LOG" 2>&1

# Hand off to the pipeline. exec replaces the shell so launchd tracks it directly.
exec "$PYTHON" -m schroeder_trader.main
