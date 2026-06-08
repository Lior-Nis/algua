from __future__ import annotations

import json
from pathlib import Path

from algua.data.models import SnapshotRecord


class SnapshotManifest:
    def __init__(self, path: Path) -> None:
        self.path = path

    def list_records(self, dataset: str | None = None) -> list[SnapshotRecord]:
        records = self._read_all()
        if dataset is not None:
            records = [r for r in records if r.dataset == dataset]
        return records

    def find(self, snapshot_id: str) -> SnapshotRecord | None:
        for rec in self._read_all():
            if rec.snapshot_id == snapshot_id:
                return rec
        return None

    def append(self, rec: SnapshotRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec.to_dict(), sort_keys=True) + "\n")

    def _read_all(self) -> list[SnapshotRecord]:
        if not self.path.exists():
            return []
        raw = self.path.read_text(encoding="utf-8")
        lines = [ln for ln in raw.splitlines() if ln.strip()]
        ends_clean = raw.endswith("\n")
        records: list[SnapshotRecord] = []
        for index, line in enumerate(lines):
            is_last = index == len(lines) - 1
            try:
                records.append(SnapshotRecord.from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError, ValueError):
                # A crash mid-append can leave a torn final line with no trailing newline.
                # Tolerate ONLY that case; any other parse failure is real corruption.
                if is_last and not ends_clean:
                    break
                raise
        return records
