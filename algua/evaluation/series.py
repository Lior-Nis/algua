"""``emit_series_file`` -- writes a backtest's daily return series to a provenance-stamped parquet.

Moved out of ``algua.cli.backtest_cmd`` alongside ``run_backtest_task`` (its only non-test caller)
so the task body can resolve it without reaching back into ``algua.cli`` (#181, #349 series work).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa

from algua.backtest.errors import BacktestError
from algua.backtest.result import BacktestResult, series_frame
from algua.data.files import frame_to_parquet_bytes
from algua.primitives.atomic_io import write_bytes_atomic


def emit_series_file(result: BacktestResult, path: Path) -> dict:
    """Write the backtest's daily return series to a deterministic, provenance-stamped parquet at
    `path` and return the stdout `series` descriptor. Fail closed (#181): a `None`, empty, or
    non-finite series raises BacktestError — never a partial/empty file."""
    if (
        result.returns is None
        or len(result.returns) == 0
        or not np.isfinite(result.returns.to_numpy(dtype=float)).all()
    ):
        raise BacktestError("backtest produced no finite return series; nothing to emit")
    frame, metadata = series_frame(result)
    try:
        write_bytes_atomic(frame_to_parquet_bytes(frame, metadata), path)
    except (OSError, pa.ArrowInvalid, Exception) as exc:
        if isinstance(exc, BacktestError):
            raise
        raise BacktestError(f"failed to write series to {path}: {exc}") from exc
    return {
        "path": str(path), "n": int(len(frame)),
        "code_hash": result.code_hash, "dependency_hash": result.dependency_hash,
        "config_hash": result.config_hash, "snapshot_id": result.snapshot_id,
        "seed": result.seed, "data_source": result.data_source,
        "start": result.period["start"], "end": result.period["end"],
        "timeframe": result.timeframe,
        "universe_name": result.universe_name,
        "fundamentals_snapshot": result.fundamentals_snapshot,
        "news_snapshot": result.news_snapshot,
        "delisting_snapshot": result.delisting_snapshot,
    }
