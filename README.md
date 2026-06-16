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
  --adjustment none

# Record universe membership as of a date.
uv run algua data ingest-universe core \
  --symbols AAPL,MSFT \
  --effective-date 2026-01-02

uv run algua data inspect
uv run algua data inspect --summary
```

## Containerized research-loop runs (local-first)

Reproducible, disposable runs that don't pollute the host. One image runs both the loop and the
gate. Prerequisite: `cp .env.example .env` (compose loads it; paper creds only — never live).

```bash
# Build the image.
docker compose build

# A disposable run gets its own DB + mlruns under ./runs/<RUN_ID>/; snapshots + kb are shared :ro.
scripts/run.sh alpha doctor
scripts/run.sh alpha backtest ...        # explore: backtest / walk-forward / sweep

# Re-use an existing run dir on purpose:
ALGUA_REUSE=1 scripts/run.sh alpha ...

# Run the same quality gate the host runs, inside the freshly built image:
docker compose build && docker compose run --rm gate
```

**Exploration vs promotion (important):** disposable per-run DBs are for *exploration* only
(backtest / walk-forward / sweep). `research promote` burns the single-use holdout, which must be
tracked in the **one shared authoritative DB** — running it against a throwaway per-run DB would
silently reuse the holdout across runs. The container refuses `research promote` on a per-run DB
unless you explicitly point `ALGUA_DB_PATH` at the authoritative DB (or set `ALGUA_ALLOW_PROMOTE=1`).

**Assumptions / limits:** linux/amd64; a local filesystem (SQLite WAL is not safe on NFS); data
*ingest* (which writes snapshots) is a separate step, not part of the read-only disposable-run path.
