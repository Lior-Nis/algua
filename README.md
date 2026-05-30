# Algua

Agent-first algorithmic-trading research and lifecycle platform.
See `docs/superpowers/specs/` for the architecture and `docs/agent/` for operating docs.

## Quickstart
```
uv sync
uv run algua doctor
```

## Data snapshots
```
# Register a local file snapshot.
uv run algua data ingest daily-bars \
  --provider local \
  --symbols AAPL,MSFT \
  --start 2026-01-02 \
  --end 2026-01-31 \
  --as-of 2026-02-01T00:00:00+00:00 \
  --source local-csv \
  --from-file data/raw/daily-bars.csv

# Fetch provider bars into a parquet point-in-time snapshot.
uv run algua data ingest-bars \
  --provider yfinance \
  --symbols AAPL,MSFT \
  --start 2026-01-02 \
  --end 2026-01-31 \
  --timeframe 1d \
  --adjustment auto

# Record universe membership as of a date.
uv run algua data ingest-universe core \
  --symbols AAPL,MSFT \
  --effective-date 2026-01-02

uv run algua data inspect
uv run algua data inspect --summary
```
