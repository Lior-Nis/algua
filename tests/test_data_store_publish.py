"""Staged/atomic payload-publish tests (#158)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from algua.data.files import (
    validate_partitioned_bars_dir,
    write_bytes_snapshot,
    write_partitioned_bars,
)


def _bars_canon(symbols: list[str], n: int = 2) -> pd.DataFrame:
    rows = []
    for sym in symbols:
        for i in range(n):
            rows.append({
                "ts": pd.Timestamp(f"2024-07-0{i + 1}T00:00:00+00:00"), "symbol": sym,
                "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
                "adj_close": 1.0, "volume": 10.0,
            })
    return pd.DataFrame(rows)


def test_write_bytes_snapshot_publishes_atomically_no_temp_residue(tmp_path):
    write_bytes_snapshot(b"payload", tmp_path, Path("snapshots/x/id1/file.bin"))
    target_dir = tmp_path / "snapshots" / "x" / "id1"
    assert (target_dir / "file.bin").read_bytes() == b"payload"
    assert [p.name for p in target_dir.iterdir()] == ["file.bin"]  # no temp left behind


def test_write_bytes_snapshot_replaces_existing_identical_file(tmp_path):
    rel = Path("snapshots/x/id1/file.bin")
    write_bytes_snapshot(b"payload", tmp_path, rel)
    write_bytes_snapshot(b"payload", tmp_path, rel)  # same id => identical bytes; benign
    assert (tmp_path / rel).read_bytes() == b"payload"


def test_validate_partitioned_bars_dir_accepts_complete_dataset(tmp_path):
    canon = _bars_canon(["AAA", "BBB"])
    write_partitioned_bars(canon, tmp_path / "ds")
    validate_partitioned_bars_dir(
        tmp_path / "ds", expected_row_count=len(canon), expected_symbols={"AAA", "BBB"}
    )


def test_validate_partitioned_bars_dir_rejects_missing_partition(tmp_path):
    canon = _bars_canon(["AAA"])
    write_partitioned_bars(canon, tmp_path / "ds")
    with pytest.raises(ValueError, match="adoption"):
        validate_partitioned_bars_dir(
            tmp_path / "ds", expected_row_count=4, expected_symbols={"AAA", "BBB"}
        )


def test_validate_partitioned_bars_dir_rejects_wrong_row_count(tmp_path):
    canon = _bars_canon(["AAA"])
    write_partitioned_bars(canon, tmp_path / "ds")
    with pytest.raises(ValueError, match="adoption"):
        validate_partitioned_bars_dir(
            tmp_path / "ds", expected_row_count=len(canon) + 1, expected_symbols={"AAA"}
        )


def test_validate_partitioned_bars_dir_rejects_torn_part_file(tmp_path):
    canon = _bars_canon(["AAA"])
    write_partitioned_bars(canon, tmp_path / "ds")
    part = next((tmp_path / "ds").rglob("part-*.parquet"))
    part.write_bytes(part.read_bytes()[: part.stat().st_size // 2])  # truncate the footer
    with pytest.raises(Exception):  # noqa: B017 — pyarrow raises its own invalid-file error; type is not our API
        validate_partitioned_bars_dir(
            tmp_path / "ds", expected_row_count=len(canon), expected_symbols={"AAA"}
        )


def test_validate_partitioned_bars_dir_handles_dotted_symbols(tmp_path):
    canon = _bars_canon(["BRK.B"])
    write_partitioned_bars(canon, tmp_path / "ds")
    validate_partitioned_bars_dir(
        tmp_path / "ds", expected_row_count=len(canon), expected_symbols={"BRK.B"}
    )
