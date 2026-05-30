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
uv run algua data ingest daily-bars \
  --provider local \
  --symbols AAPL,MSFT \
  --start 2026-01-02 \
  --end 2026-01-31 \
  --as-of 2026-02-01T00:00:00+00:00 \
  --source local-csv \
  --from-file data/raw/daily-bars.csv

uv run algua data inspect
```
