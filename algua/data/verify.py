from __future__ import annotations

from pathlib import Path

from algua.data.files import (
    parquet_dataset_row_count,
    parquet_file_row_count,
    sha256_file,
)
from algua.data.models import SnapshotRecord


class SnapshotVerifier:
    """Power-loss read-back backstop for a single snapshot payload (#184), extracted from
    ``DataStore`` (#384) as the cohesive verification collaborator.

    Read-only and stateless beyond ``data_dir``: it owns no writes, no lease, and no manifest
    mutation. Record resolution + the per-snapshot aggregation loop stay on ``DataStore`` so its
    dynamic ``self.verify_snapshot``/``get_snapshot``/``list_snapshots`` dispatch (and the
    ``SnapshotNotFound`` contract) is preserved byte-for-byte and this module needs no import back
    into ``store``.
    """

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir

    def verify_snapshot(self, rec: SnapshotRecord) -> None:
        """Power-loss read-back of one snapshot's payload (#184). Reads the bytes back to prove
        they are durable and decompressible, and checks the row count against the record. Raises
        on any damage (the caller decides how to surface it). Dispatch by `storage_format`:

        - ``parquet_dataset`` (bars): full read of every partition; summed rows == ``row_count``.
        - ``parquet`` (universe/fundamentals/news, or a ``.parquet`` via ``ingest_file``): full
          read of the single file; ``num_rows == row_count``. Readability check, NOT a
          content-hash recompute. For a ``.parquet`` ingested via ``ingest_file`` (byte-hash
          ``content_hash``) this is a strictly weaker check than the ``else`` branch's
          ``sha256_file`` comparison — by design, since verify targets power-loss readability,
          not tampering.
        - anything else (``ingest_file`` csv/generic): ``sha256_file == content_hash`` (a full
          read). Fails closed: a record whose ``content_hash`` is not a byte hash would report a
          (false) failure rather than a false pass — that signals the dispatch needs extending.
        """
        target = self.data_dir / rec.data_path
        fmt = rec.storage_format
        if fmt == "parquet_dataset":
            if not target.is_dir():
                raise ValueError(f"snapshot {rec.snapshot_id}: payload dir missing at {target}")
            rows = parquet_dataset_row_count(target)
            if rec.row_count is not None and rows != rec.row_count:
                raise ValueError(
                    f"snapshot {rec.snapshot_id}: read {rows} rows, expected {rec.row_count}"
                )
        elif fmt == "parquet":
            if not target.is_file():
                raise ValueError(f"snapshot {rec.snapshot_id}: payload file missing at {target}")
            rows = parquet_file_row_count(target)
            if rec.row_count is not None and rows != rec.row_count:
                raise ValueError(
                    f"snapshot {rec.snapshot_id}: read {rows} rows, expected {rec.row_count}"
                )
        else:
            if not target.is_file():
                raise ValueError(f"snapshot {rec.snapshot_id}: payload file missing at {target}")
            actual = sha256_file(target)
            if actual != rec.content_hash:
                raise ValueError(
                    f"snapshot {rec.snapshot_id}: content hash {actual} != {rec.content_hash}"
                )
