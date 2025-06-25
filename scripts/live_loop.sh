#!/bin/bash

# Live trading loop script for cron execution
# Run every minute during market hours

set -e  # Exit on any error

# Set environment
export PYTHONPATH="/path/to/algua:$PYTHONPATH"
export TRADING_ENABLED="false"  # Safety: ensure paper trading by default

# Change to project directory
cd /path/to/algua

# Activate conda environment
source /path/to/conda/bin/activate algua

# Log current time
echo "$(date): Starting live trading loop"

# Check if markets are open (basic check)
HOUR=$(date +%H)
DAY=$(date +%u)  # 1=Monday, 7=Sunday

if [ $DAY -ge 6 ]; then
    echo "$(date): Markets closed (weekend)"
    exit 0
fi

if [ $HOUR -lt 9 ] || [ $HOUR -gt 16 ]; then
    echo "$(date): Markets closed (outside trading hours)"
    exit 0
fi

# Run live trading script
python scripts/run_live_trading.py

echo "$(date): Live trading loop completed" 