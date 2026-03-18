#!/bin/bash
# Install the SchroederTrader cron job
# Runs at 4:30 PM ET on weekdays
#
# NOTE: macOS cron does not support CRON_TZ. We use UTC times instead.
# 4:30 PM ET = 21:30 UTC (EST, Nov-Mar) or 20:30 UTC (EDT, Mar-Nov)
# We schedule at 21:30 UTC (conservative — during EDT this runs at 5:30 PM ET,
# which is still after market close). Adjust if needed.

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PROJECT_DIR}/.venv/bin/python"
MAIN="${PROJECT_DIR}/src/schroeder_trader/main.py"
LOG_DIR="${PROJECT_DIR}/logs"

mkdir -p "${LOG_DIR}"

# Check that python and main.py exist
if [ ! -f "$PYTHON" ]; then
    echo "Error: Python not found at $PYTHON"
    echo "Run 'uv venv && uv pip install -e .[dev,backtest]' first"
    exit 1
fi

# 21:30 UTC = 4:30 PM EST / 5:30 PM EDT (both after market close)
CRON_LINE="30 21 * * 1-5 ${PYTHON} ${MAIN} >> ${LOG_DIR}/cron.log 2>&1"

# Check if already installed
if crontab -l 2>/dev/null | grep -q "schroeder_trader/main.py"; then
    echo "Cron job already installed. Current entry:"
    crontab -l | grep "schroeder_trader"
    exit 0
fi

# Add to crontab
(crontab -l 2>/dev/null; echo "$CRON_LINE") | crontab -
echo "Cron job installed:"
echo "$CRON_LINE"
