from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from algua.data.files import (
    BARS_STREAMED_HASH_ALGO,
    compose_bars_symbol_hash,
    logical_bars_hash,
    write_partitioned_bars,
)
from algua.data.models import Dataset, Kind, SnapshotRecord
from algua.data.schema import to_bar_schema
from algua.data.store.bars import BarsStoreMixin
from algua.data.store.identity import build_metadata, compute_snapshot_id
from algua.data.timeframes import validate_timeframe


class BarsStreamedStoreMixin(BarsStoreMixin):
    def ingest_bars_streamed(
        self,
        *,
        provider: str,
        symbols: list[str],
        as_of: str,
        source: str,
        chunks: Iterable[pd.DataFrame],
        timeframe: str = "1d",
        adjustment: str = "split_div",
        start: str | None = None,
        end: str | None = None,
        source_metadata: dict[str, str] | None = None,
    ) -> SnapshotRecord:
        """Stream per-symbol bar chunks into one hive-partitioned-by-symbol bars snapshot.

        Crash-safe: each chunk is normalized via `to_bar_schema` (so output is schema-valid) and
        written as its own `symbol=<SYM>/` partition under a UUID staging dir (one chunk in memory
        at a time -> bounded RAM). The per-symbol logical leaf hashes are composed (sorted by
        symbol, so order-independent) into the snapshot `content_hash`; the snapshot_id is content-
        addressed. Commit goes through the shared `_commit_bars_dir` protocol (#158): dedup on
        snapshot_id (idempotent re-ingest), then `os.replace` the staging dir onto the immutable
        snapshot dir — if that target dir already exists (an orphan from a crash between rename and
        manifest-append, or a concurrent winner) re-check the manifest, otherwise VALIDATE the
        existing dir (fail closed on a partial/foreign dir) and adopt it. The manifest record is
        committed last via `append_if_absent`, with `storage_format="parquet_dataset"` so
        `read_bars` serves it with pushdown.

        Cross-chunk integrity: each symbol must appear in exactly one chunk — the method rejects a
        symbol that recurs in a later chunk (so each `symbol=<SYM>/` partition is written once and
        the snapshot is globally unique on (timestamp, symbol) given each chunk is internally
        schema-valid). The FirstRate importer satisfies this by yielding one chunk per symbol.

        Note: when `start`/`end` are given, the coverage check is span-only (observed range covers
        the requested endpoints); it does not detect interior gaps.
        """
        validate_timeframe(timeframe)
        # Lease the staging dir for the whole import so a concurrent clear_staging can't rmtree it
        # mid-write — the staging-root mtime is set once at mkdir and never refreshes (#255).
        staging_dir, lock_fd, lock_path = self._staging.new_leased_staging()
        row_count = 0
        observed_min: pd.Timestamp | None = None
        observed_max: pd.Timestamp | None = None
        seen_symbols_set: set[str] = set()
        leaves: list[tuple[str, int, str]] = []
        try:
            for chunk in chunks:
                chunk_canon = (
                    to_bar_schema(chunk, timeframe=timeframe)
                    .reset_index()
                    .rename(columns={"timestamp": "ts"})
                )
                chunk_symbols = set(chunk_canon["symbol"].unique())
                clash = chunk_symbols & seen_symbols_set
                if clash:
                    raise ValueError(
                        f"symbol(s) {sorted(clash)} appear in more than one chunk; streamed "
                        "ingest requires each symbol's bars in a single contiguous chunk"
                    )
                seen_symbols_set |= chunk_symbols
                write_partitioned_bars(chunk_canon, staging_dir)
                row_count += len(chunk_canon)
                cmin = chunk_canon["ts"].min()
                cmax = chunk_canon["ts"].max()
                observed_min = cmin if observed_min is None else min(observed_min, cmin)
                observed_max = cmax if observed_max is None else max(observed_max, cmax)
                for sym, group in chunk_canon.groupby("symbol"):
                    leaves.append((str(sym), len(group), logical_bars_hash(group)))
            if not leaves or row_count == 0:
                raise ValueError("no bars to ingest (empty chunk stream)")

            if observed_min is None or observed_max is None:  # unreachable: leaves => loop ran
                raise ValueError("no bars to ingest (empty chunk stream)")
            observed_start = observed_min.date().isoformat()
            observed_end = observed_max.date().isoformat()
            if start is not None or end is not None:
                if (start is not None and observed_start > start) or (
                    end is not None and observed_end < end
                ):
                    raise ValueError(
                        f"observed coverage [{observed_start}, {observed_end}] does not cover "
                        f"requested [{start}, {end}]"
                    )

            meta_extra = dict(source_metadata or {})
            if start is not None:
                meta_extra["requested_start"] = start
            if end is not None:
                meta_extra["requested_end"] = end
            meta_extra["observed_start"] = observed_start
            meta_extra["observed_end"] = observed_end
            meta_extra["content_hash_algorithm"] = BARS_STREAMED_HASH_ALGO

            metadata = build_metadata(
                dataset=Dataset.BARS,
                provider=provider,
                symbols=symbols,
                start=observed_start,
                end=observed_end,
                as_of=as_of,
                source=source,
                kind=Kind.BARS,
                timeframe=timeframe,
                adjustment=adjustment,
                source_metadata=meta_extra,
            )
            content_hash = compose_bars_symbol_hash(leaves)
            snapshot_id = compute_snapshot_id(metadata, content_hash)

            relative_path = Path("snapshots") / metadata.dataset / snapshot_id  # a DIR
            existing = self.manifest.find(snapshot_id)
            if existing is not None:
                return existing
            rec = SnapshotRecord(
                snapshot_id=snapshot_id,
                metadata=metadata,
                row_count=row_count,
                content_hash=content_hash,
                data_path=relative_path,
                created_at=datetime.now(UTC).isoformat(),
                storage_format="parquet_dataset",
            )
            return self._commit_bars_dir(rec, staging_dir, expected_symbols=seen_symbols_set)
        finally:
            self._staging.release_leased_staging(staging_dir, lock_fd, lock_path)
