from __future__ import annotations

from datetime import date

import pandas as pd

from algua.data.models import Dataset, Kind, SnapshotRecord, UniverseSnapshot
from algua.data.store.identity import IdentityMixin, build_metadata, normalize_symbols


class UniverseStoreMixin(IdentityMixin):
    def ingest_universe(
        self,
        *,
        universe: str,
        symbols: list[str],
        effective_date: str,
        as_of: str,
        source: str,
        provider: str = "local",
        source_metadata: dict[str, str] | None = None,
    ) -> SnapshotRecord:
        clean_symbols = normalize_symbols(symbols)
        frame = pd.DataFrame(
            {"effective_date": effective_date, "universe": universe, "symbol": clean_symbols}
        )
        metadata = build_metadata(
            dataset=Dataset.UNIVERSES,
            provider=provider,
            symbols=clean_symbols,
            start=effective_date,
            end=effective_date,
            as_of=as_of,
            source=source,
            kind=Kind.UNIVERSE,
            universe=universe,
            source_metadata=source_metadata,
        )

        # Universes are immutable on EVERY ingest path (#263): a same-(universe, effective_date)
        # change with different membership aborts at the manifest commit (append_if_absent, under
        # the manifest lock so it is race-safe), so no conflicting record is ever committed — not
        # just caught later at read time. (A rejected ingest may leave an inert orphan parquet that
        # the manifest never references — the shared _ingest_parquet publish-then-commit behavior;
        # it feeds no read.) Corrections require a new universe name.
        def conflict_check(committed, rec):
            for other in committed:
                if (
                    other.dataset == Dataset.UNIVERSES
                    and other.metadata.universe == universe
                    and other.metadata.start == effective_date
                    and other.content_hash != rec.content_hash
                ):
                    raise ValueError(
                        f"universe {universe!r} already has a DIFFERENT membership on "
                        f"{effective_date} (immutable; corrections require a new name)"
                    )

        return self._ingest_parquet(
            metadata=metadata, frame=frame, filename="universe.parquet",
            conflict_check=conflict_check,
        )

    def read_universe(self, universe: str) -> list[UniverseSnapshot]:
        """Read a named universe's point-in-time membership timeline.

        A time-varying universe is recorded as one membership snapshot per `effective_date`,
        all sharing the universe NAME (see `ingest_universe`). This reads every snapshot tagged
        with `universe`, normalizes its symbols, and returns the timeline sorted ascending by
        `effective_date`. The as-of-date-t membership is the snapshot with the greatest
        `effective_date <= t` (empty before the earliest effective date) — that resolution is the
        consumer's, but the timeline this returns is what makes it leak-free.

        Raises ``ValueError`` if two snapshots share an `effective_date` but disagree on
        membership: the as-of answer for that date would be ambiguous, so we refuse rather than
        silently pick one.
        """
        records = [
            rec
            for rec in self.manifest.list_records(Dataset.UNIVERSES)
            if rec.metadata.universe == universe
        ]
        by_date: dict[date, UniverseSnapshot] = {}
        for rec in records:
            frame = pd.read_parquet(self.data_dir / rec.data_path)
            eff = date.fromisoformat(str(frame["effective_date"].iloc[0]))
            symbols = frozenset(normalize_symbols([str(s) for s in frame["symbol"]]))
            existing = by_date.get(eff)
            if existing is not None and existing.symbols != symbols:
                raise ValueError(
                    f"ambiguous as-of membership for universe {universe!r} on {eff.isoformat()}: "
                    f"two snapshots disagree ({sorted(existing.symbols)} vs {sorted(symbols)})"
                )
            by_date[eff] = UniverseSnapshot(
                snapshot_id=rec.snapshot_id, effective_date=eff, symbols=symbols
            )
        return [by_date[eff] for eff in sorted(by_date)]
