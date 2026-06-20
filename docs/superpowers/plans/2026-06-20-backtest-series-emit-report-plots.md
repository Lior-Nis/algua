# Backtest series-emit + report series-plots Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Emit a backtest's daily portfolio return series through the CLI (file + MLflow artifact) and teach the `report-experiments` skill to plot equity / drawdown / rolling-Sharpe / return-distribution from it.

**Architecture:** A pure `series_frame()` on `BacktestResult` builds a `[date, ret]` frame + a fully-stamped provenance metadata dict; `frame_to_parquet_bytes` (extended with optional metadata) + `write_bytes_atomic` serialize it deterministically. `backtest run --emit-series PATH` writes the file; `log_backtest` (on `--track`) logs the same bytes as a `series.parquet` MLflow artifact. The skill *reads* that artifact (no re-run) and renders four SVGs.

**Tech Stack:** Python, typer (CLI), pandas, pyarrow/parquet, mlflow, matplotlib (skill).

**Spec:** `docs/superpowers/specs/2026-06-20-backtest-series-emit-report-plots-design.md` (GATE-1 reviewed).

## Global Constraints

- Quality gate at EVERY commit: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports` (all green; import-linter must stay **18 kept, 0 broken**).
- Import boundaries (enforced): `algua.backtest` MUST NOT import `algua.data`/`algua.cli`/`algua.registry`/`algua.tracking` → therefore `series_frame` (in `algua/backtest/result.py`) stays pure (frame + dict only; NO serialization, NO I/O). `algua.tracking → algua.data` IS allowed (no contract forbids it) — the tracker may import `frame_to_parquet_bytes`.
- Determinism: same seed+snapshot ⇒ byte-identical parquet (sorted metadata keys, `-0.0 → +0.0` canonicalization) and byte-identical SVG (existing `svg.hashsalt`/`MPLCONFIGDIR` pinning).
- Holdout: only `log_backtest` logs a series artifact — `log_sweep`/`log_walk_forward` MUST NOT (their return vectors contain the reserved holdout tail).
- Fail closed: `--emit-series` with `returns is None` OR empty series ⇒ `BacktestError` (JSON error envelope, non-zero exit), no file written.
- Demo backtest invocation used throughout tests: strategy `cross_sectional_momentum`, `--demo --start 2023-01-01 --end 2023-12-31` (yields a multi-bar finite return series).
- Commit message trailer on every commit: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

## File Structure

- `algua/data/files.py` — MODIFY: extend `frame_to_parquet_bytes(frame, metadata=None)`; add `write_bytes_atomic(data, dest)`. (Task 1)
- `algua/backtest/result.py` — MODIFY: add pure `series_frame(result)`. (Task 2)
- `algua/cli/backtest_cmd.py` — MODIFY: add `emit_series_file(result, path)` helper + `--emit-series` flag wiring. (Task 3)
- `algua/tracking/mlflow_tracker.py` — MODIFY: log `series.parquet` artifact in `log_backtest`. (Task 4)
- `pyproject.toml` — MODIFY: declare `matplotlib` direct dep. (Task 5)
- `.codex/skills/report-experiments/SKILL.md` — MODIFY: series plots + prose. (Task 6)
- Tests (new): `tests/test_data_files_series.py`, `tests/test_backtest_series_frame.py`, `tests/test_cli_backtest_series.py`, `tests/test_tracking_series_artifact.py`.

---

### Task 1: Parquet serialization with metadata + atomic write

**Files:**
- Modify: `algua/data/files.py:105` (`frame_to_parquet_bytes`); add `write_bytes_atomic` near it.
- Test: `tests/test_data_files_series.py` (create)

**Interfaces:**
- Produces: `frame_to_parquet_bytes(frame: pd.DataFrame, metadata: dict[str, str] | None = None) -> bytes` (metadata attached to arrow schema with SORTED keys; `None` = strip, as today). `write_bytes_atomic(data: bytes, dest: Path) -> None` (same-dir temp + `os.replace`, no fsync).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_data_files_series.py
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from algua.data.files import frame_to_parquet_bytes, write_bytes_atomic


def _frame():
    return pd.DataFrame({"date": ["2023-01-01T00:00:00", "2023-01-02T00:00:00"], "ret": [0.01, -0.02]})


def test_metadata_attached_and_sorted_key_order_independent():
    b1 = frame_to_parquet_bytes(_frame(), {"b": "2", "a": "1"})
    b2 = frame_to_parquet_bytes(_frame(), {"a": "1", "b": "2"})
    assert b1 == b2  # sorted keys -> insertion order irrelevant -> deterministic
    meta = pq.read_schema(pa.BufferReader(b1)).metadata
    assert meta == {b"a": b"1", b"b": b"2"}


def test_no_metadata_strips_schema_metadata():
    b = frame_to_parquet_bytes(_frame())
    assert pq.read_schema(pa.BufferReader(b)).metadata is None


def test_write_bytes_atomic_roundtrip_no_temp_left(tmp_path: Path):
    dest = tmp_path / "sub" / "series.parquet"
    write_bytes_atomic(b"hello-bytes", dest)
    assert dest.read_bytes() == b"hello-bytes"
    leftovers = [p.name for p in (tmp_path / "sub").iterdir() if p.name.startswith(".emit-")]
    assert leftovers == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_data_files_series.py -q`
Expected: FAIL (`write_bytes_atomic` not defined / `frame_to_parquet_bytes` has no `metadata` arg).

- [ ] **Step 3: Implement**

Replace `frame_to_parquet_bytes` in `algua/data/files.py` and add `write_bytes_atomic`:

```python
def frame_to_parquet_bytes(
    frame: pd.DataFrame, metadata: dict[str, str] | None = None
) -> bytes:
    """Serialize `frame` to canonical, reproducible parquet bytes.

    Determinism is what makes the content hash a stable provenance key (issue #55). We pin the arrow
    table directly (no pandas index sidecar) and fix the writer/compression so two equal frames hash
    identically. `metadata` (#181) is OPTIONAL self-describing schema key/value metadata; keys are
    SORTED so the byte output is independent of dict insertion order. `None` (the default) strips all
    schema metadata exactly as before, so existing content-addressed callers are unchanged.
    """
    table = pa.Table.from_pandas(frame, preserve_index=False).replace_schema_metadata(
        None if metadata is None else {k: metadata[k] for k in sorted(metadata)}
    )
    buffer = pa.BufferOutputStream()
    pq.write_table(table, buffer, compression="snappy", version="2.6")
    return buffer.getvalue().to_pybytes()


def write_bytes_atomic(data: bytes, dest: Path) -> None:
    """Write `data` to `dest` atomically via a same-dir temp + `os.replace` (#181): a reader never
    sees a partially written file even if the process dies mid-write. No fsync — this is an ephemeral
    export (a plotting input), not a durable snapshot (cf. `write_bytes_snapshot`)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix=".emit-")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, dest)
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
```

(`os`, `tempfile`, `Path`, `pa`, `pq` are already imported at the top of `files.py`.)

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_data_files_series.py tests/test_data_store_publish.py -q`
Expected: PASS (new tests pass; existing `frame_to_parquet_bytes` callers unaffected).

- [ ] **Step 5: Gate + commit**

```bash
uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports
git add algua/data/files.py tests/test_data_files_series.py
git commit -m "feat(181): parquet metadata + write_bytes_atomic in data/files

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: `series_frame` on BacktestResult (pure)

**Files:**
- Modify: `algua/backtest/result.py` (add `series_frame`; add `import json`)
- Test: `tests/test_backtest_series_frame.py` (create)

**Interfaces:**
- Consumes: `BacktestResult` (`result.returns: pd.Series`, `result.to_dict()`).
- Produces: `series_frame(result: BacktestResult) -> tuple[pd.DataFrame, dict[str, str]]` — frame columns `["date", "ret"]` (`date` = ISO-8601 strings; `ret` float64 with `-0.0 → +0.0`); metadata `{"algua.result_json": json.dumps(result.to_dict(), sort_keys=True, default=str)}`. Caller MUST guard `returns is not None and len > 0`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_backtest_series_frame.py
import json

import numpy as np
import pandas as pd

from algua.backtest.result import BacktestResult, series_frame


def _result(returns):
    return BacktestResult(
        strategy="mom", metrics={"sharpe": 1.0}, config_hash="cfg",
        data_source="SyntheticProvider", timeframe="1d",
        period={"start": "2023-01-01", "end": "2023-01-03"},
        seed=0, snapshot_id=None, code_hash="abc", dependency_hash="dep", returns=returns,
    )


def test_series_frame_columns_and_iso_dates():
    idx = pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"])
    frame, meta = series_frame(_result(pd.Series([0.01, -0.0, 0.02], index=idx)))
    assert list(frame.columns) == ["date", "ret"]
    assert frame["date"].tolist() == [
        "2023-01-01T00:00:00", "2023-01-02T00:00:00", "2023-01-03T00:00:00"]
    assert frame["ret"].tolist() == [0.01, 0.0, 0.02]


def test_series_frame_canonicalizes_negative_zero():
    idx = pd.to_datetime(["2023-01-01"])
    frame, _ = series_frame(_result(pd.Series([-0.0], index=idx)))
    assert not np.signbit(frame["ret"].to_numpy()[0])  # +0.0, not -0.0


def test_series_frame_metadata_carries_full_identity_minus_returns():
    idx = pd.to_datetime(["2023-01-01", "2023-01-02"])
    _, meta = series_frame(_result(pd.Series([0.01, 0.02], index=idx)))
    payload = json.loads(meta["algua.result_json"])
    assert payload["strategy"] == "mom"
    assert payload["config_hash"] == "cfg"
    assert "returns" not in payload  # to_dict already excludes it
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_backtest_series_frame.py -q`
Expected: FAIL (`cannot import name 'series_frame'`).

- [ ] **Step 3: Implement**

Add `import json` to the imports of `algua/backtest/result.py`, then append:

```python
def series_frame(result: BacktestResult) -> tuple[pd.DataFrame, dict[str, str]]:
    """Pure projection of a backtest's daily return series to a `[date, ret]` frame + a fully-stamped
    provenance metadata dict (#181). NO serialization, NO I/O — keeps `algua.backtest` off the data
    lane. Caller guards `returns is not None and len(returns) > 0`.

    `ret` is canonicalized `-0.0 -> +0.0` (a flat day must not perturb the parquet bytes — mirrors
    `logical_bars_hash`). Metadata embeds the WHOLE run identity as one sorted-key JSON blob so the
    standalone file is self-describing (carries config/code/dependency hashes, snapshot, seed,
    timeframe, period, universe/fundamentals/news/delisting provenance, metrics — everything in
    `to_dict()` except the series itself)."""
    assert result.returns is not None
    r = result.returns
    frame = pd.DataFrame(
        {
            "date": [pd.Timestamp(ts).isoformat() for ts in r.index],
            "ret": r.to_numpy(dtype=float) + 0.0,  # -0.0 -> +0.0
        }
    )
    metadata = {"algua.result_json": json.dumps(result.to_dict(), sort_keys=True, default=str)}
    return frame, metadata
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_backtest_series_frame.py -q`
Expected: PASS.

- [ ] **Step 5: Gate + commit**

```bash
uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports
git add algua/backtest/result.py tests/test_backtest_series_frame.py
git commit -m "feat(181): pure series_frame() projecting returns to [date, ret] + provenance

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: `backtest run --emit-series PATH`

**Files:**
- Modify: `algua/cli/backtest_cmd.py` (add `emit_series_file` helper near `_track`; add `--emit-series` option + wiring in `run`)
- Test: `tests/test_cli_backtest_series.py` (create)

**Interfaces:**
- Consumes: `series_frame` (Task 2), `frame_to_parquet_bytes`/`write_bytes_atomic` (Task 1), `BacktestError`.
- Produces: `emit_series_file(result: BacktestResult, path: Path) -> dict` — raises `BacktestError` if `returns is None` or empty; else writes parquet atomically and returns the stdout `series` descriptor dict. `run` gains `emit_series: str = typer.Option(None, "--emit-series", ...)`; when set, `payload["series"] = emit_series_file(result, Path(emit_series))`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli_backtest_series.py
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from algua.backtest.engine import BacktestError
from algua.backtest.result import BacktestResult
from algua.cli.backtest_cmd import emit_series_file
from algua.cli.main import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def _tmp_db(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))


def _result(returns):
    return BacktestResult(
        strategy="mom", metrics={"sharpe": 1.0}, config_hash="cfg",
        data_source="SyntheticProvider", timeframe="1d",
        period={"start": "2023-01-01", "end": "2023-01-02"},
        seed=0, code_hash="abc", dependency_hash="dep", returns=returns,
    )


def test_emit_series_file_fails_closed_on_none(tmp_path):
    with pytest.raises(BacktestError):
        emit_series_file(_result(None), tmp_path / "s.parquet")


def test_emit_series_file_fails_closed_on_empty(tmp_path):
    empty = pd.Series([], dtype=float, index=pd.to_datetime([]))
    with pytest.raises(BacktestError):
        emit_series_file(_result(empty), tmp_path / "s.parquet")
    assert not (tmp_path / "s.parquet").exists()


def test_emit_series_file_writes_and_descriptor(tmp_path):
    idx = pd.to_datetime(["2023-01-01", "2023-01-02"])
    desc = emit_series_file(_result(pd.Series([0.01, 0.02], index=idx)), tmp_path / "s.parquet")
    assert desc["n"] == 2
    assert desc["config_hash"] == "cfg" and desc["start"] == "2023-01-01"
    meta = pq.read_schema(pa.BufferReader((tmp_path / "s.parquet").read_bytes())).metadata
    assert b"algua.result_json" in meta


def test_cli_emit_series_demo(tmp_path):
    out = tmp_path / "series.parquet"
    res = runner.invoke(app, ["backtest", "run", "cross_sectional_momentum", "--demo",
                              "--start", "2023-01-01", "--end", "2023-12-31",
                              "--emit-series", str(out)])
    assert res.exit_code == 0, res.stdout
    payload = json.loads(res.stdout)
    assert payload["series"]["path"] == str(out)
    df = pd.read_parquet(out)
    assert list(df.columns) == ["date", "ret"]
    assert payload["series"]["n"] == len(df) > 0


def test_cli_emit_series_deterministic(tmp_path):
    a, b = tmp_path / "a.parquet", tmp_path / "b.parquet"
    for out in (a, b):
        r = runner.invoke(app, ["backtest", "run", "cross_sectional_momentum", "--demo",
                                "--start", "2023-01-01", "--end", "2023-12-31",
                                "--emit-series", str(out)])
        assert r.exit_code == 0, r.stdout
    assert a.read_bytes() == b.read_bytes()


def test_cli_no_emit_series_has_no_series_key(tmp_path):
    res = runner.invoke(app, ["backtest", "run", "cross_sectional_momentum", "--demo",
                              "--start", "2023-01-01", "--end", "2023-12-31"])
    assert res.exit_code == 0, res.stdout
    assert "series" not in json.loads(res.stdout)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli_backtest_series.py -q`
Expected: FAIL (`cannot import name 'emit_series_file'`).

- [ ] **Step 3: Implement**

In `algua/cli/backtest_cmd.py`, add imports near the top:

```python
from pathlib import Path

from algua.backtest.result import series_frame
from algua.data.files import frame_to_parquet_bytes, write_bytes_atomic
```

Add the helper (after `_track`):

```python
def emit_series_file(result, path: Path) -> dict:
    """Write the backtest's daily return series to a deterministic, provenance-stamped parquet at
    `path` and return the stdout `series` descriptor. Fail closed (#181): a `None` (non-finite) or
    empty series raises BacktestError — never a partial/empty file."""
    if result.returns is None or len(result.returns) == 0:
        raise BacktestError("backtest produced no finite return series; nothing to emit")
    frame, metadata = series_frame(result)
    write_bytes_atomic(frame_to_parquet_bytes(frame, metadata), path)
    return {
        "path": str(path), "n": int(len(frame)),
        "code_hash": result.code_hash, "dependency_hash": result.dependency_hash,
        "config_hash": result.config_hash, "snapshot_id": result.snapshot_id,
        "seed": result.seed, "data_source": result.data_source,
        "start": result.period["start"], "end": result.period["end"],
        "timeframe": result.timeframe,
    }
```

Add the option to `run`'s signature (next to `track`):

```python
    emit_series: str = typer.Option(
        None, "--emit-series",
        help="write the daily return series to a parquet at PATH (for series plots)"),
```

Wire it in `run`, immediately before `payload = result.to_dict()` is emitted (after the existing `persist_backtest_returns` block, before `if track:`):

```python
    payload = result.to_dict()
    if emit_series:
        payload["series"] = emit_series_file(result, Path(emit_series))
    if track:
        payload["mlflow_run_id"] = _track(
            lambda: log_backtest(
                result, strategy.config.params, tracking_uri=get_settings().mlflow_tracking_uri
            )
        )
    emit(ok(payload))
```

(Delete the old `payload = result.to_dict()` line that preceded `if track:` so it isn't duplicated.)

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_cli_backtest_series.py tests/test_cli_backtest.py -q`
Expected: PASS.

- [ ] **Step 5: Gate + commit**

```bash
uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports
git add algua/cli/backtest_cmd.py tests/test_cli_backtest_series.py
git commit -m "feat(181): backtest run --emit-series writes provenance-stamped series parquet

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: `series.parquet` artifact in `log_backtest`

**Files:**
- Modify: `algua/tracking/mlflow_tracker.py` (`log_backtest`; add `_log_series_artifact` helper + imports)
- Test: `tests/test_tracking_series_artifact.py` (create)

**Interfaces:**
- Consumes: `series_frame` (Task 2), `frame_to_parquet_bytes` (Task 1).
- Produces: inside `log_backtest`, after `log_dict(result.to_dict(), "result.json")`, call `_log_series_artifact(result)` which logs `series.parquet` iff `returns` is non-empty. `log_sweep`/`log_walk_forward` unchanged.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tracking_series_artifact.py
import pandas as pd
from mlflow.tracking import MlflowClient

from algua.backtest.result import BacktestResult
from algua.tracking.mlflow_tracker import log_backtest


def _result(returns):
    return BacktestResult(
        strategy="mom_series", metrics={"sharpe": 1.0}, config_hash="cfg",
        data_source="SyntheticProvider", timeframe="1d",
        period={"start": "2023-01-01", "end": "2023-01-03"},
        seed=0, code_hash="abc", dependency_hash="dep", returns=returns,
    )


def _artifact_names(uri, run_id):
    return {a.path for a in MlflowClient(tracking_uri=uri).list_artifacts(run_id)}


def test_log_backtest_logs_series_parquet(tmp_path):
    uri = (tmp_path / "mlruns").as_uri()
    idx = pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"])
    run_id = log_backtest(_result(pd.Series([0.01, -0.02, 0.0], index=idx)), {}, tracking_uri=uri)
    names = _artifact_names(uri, run_id)
    assert "series.parquet" in names
    local = MlflowClient(tracking_uri=uri).download_artifacts(run_id, "series.parquet")
    df = pd.read_parquet(local)
    assert list(df.columns) == ["date", "ret"] and len(df) == 3


def test_log_backtest_skips_series_when_returns_none(tmp_path):
    uri = (tmp_path / "mlruns").as_uri()
    run_id = log_backtest(_result(None), {}, tracking_uri=uri)
    assert "series.parquet" not in _artifact_names(uri, run_id)
```

Add to `tests/test_tracking_series_artifact.py` a negative for sweep/walk-forward — they must never log a series. (Build minimal `SweepResult`/`WalkForwardResult` is heavier; instead assert via the public CLI in the next test using `--track`. If constructing those results is non-trivial, cover the invariant with this lighter assertion that only `log_backtest` references `series.parquet`:)

```python
import inspect

from algua.tracking import mlflow_tracker


def test_only_log_backtest_emits_series_artifact():
    assert "series.parquet" in inspect.getsource(mlflow_tracker.log_backtest)
    assert "series.parquet" not in inspect.getsource(mlflow_tracker.log_sweep)
    assert "series.parquet" not in inspect.getsource(mlflow_tracker.log_walk_forward)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tracking_series_artifact.py -q`
Expected: FAIL (`series.parquet` not in artifacts).

- [ ] **Step 3: Implement**

In `algua/tracking/mlflow_tracker.py` add imports (top):

```python
import tempfile
from pathlib import Path

from algua.backtest.result import BacktestResult, series_frame
from algua.data.files import frame_to_parquet_bytes
```

(Adjust the existing `from algua.backtest.result import BacktestResult` line to include `series_frame`.)

Add the helper:

```python
def _log_series_artifact(result: BacktestResult) -> None:
    """Log the backtest's daily return series as a `series.parquet` MLflow artifact (#181), so the
    report-experiments skill plots the LOGGED run's own series (no re-run, no code/input drift). Only
    `log_backtest` calls this — `log_sweep`/`log_walk_forward` must NOT, because their return vectors
    contain the reserved single-use holdout tail. Best-effort: an absent/empty series is skipped."""
    import mlflow

    if result.returns is None or len(result.returns) == 0:
        return
    frame, metadata = series_frame(result)
    data = frame_to_parquet_bytes(frame, metadata)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "series.parquet"  # mlflow logs the artifact under its basename
        p.write_bytes(data)
        mlflow.log_artifact(str(p))
```

In `log_backtest`, after the existing `mlflow.log_dict(result.to_dict(), "result.json")` line and before `return run.info.run_id`:

```python
        _log_series_artifact(result)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_tracking_series_artifact.py tests/test_tracking_backtest.py -q`
Expected: PASS.

- [ ] **Step 5: Gate + commit**

```bash
uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports
git add algua/tracking/mlflow_tracker.py tests/test_tracking_series_artifact.py
git commit -m "feat(181): log_backtest logs series.parquet artifact (sweep/wf never do)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Declare matplotlib as a direct dependency

**Files:**
- Modify: `pyproject.toml` (`[project].dependencies`)

**Interfaces:** none (packaging only).

- [ ] **Step 1: Add the dependency**

In `pyproject.toml`, add to the `dependencies` list (after the `mlflow>=3.1` line):

```toml
    "matplotlib>=3.8",  # #181 series plots (report-experiments); was transitive via mlflow
```

- [ ] **Step 2: Sync + verify import**

Run: `uv sync && uv run python -c "import matplotlib; print(matplotlib.__version__)"`
Expected: prints a version ≥ 3.8, no error.

- [ ] **Step 3: Gate + commit**

```bash
uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports
git add pyproject.toml uv.lock
git commit -m "build(181): declare matplotlib as a direct dependency

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: report-experiments skill — series plots

**Files:**
- Modify: `.codex/skills/report-experiments/SKILL.md` (reference script + prose)

**Interfaces:**
- Consumes (at runtime, not import): MLflow `kind="backtest"` runs with a `series.parquet` artifact + `result.json`.

This task edits a skill doc (embedded python). Verification is a real dry-run, not unit tests.

- [ ] **Step 1: Add the four plot functions** to the reference script (after `plot_cross_run`, before `def main()`):

```python
def _series_provenance(uri: str, run, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": run.info.run_id, "config_hash": result.get("config_hash"),
        "dependency_hash": result.get("dependency_hash"), "snapshot_id": result.get("snapshot_id"),
        "code_hash": result.get("code_hash"), "seed": result.get("seed"),
        "period": result.get("period"),
    }


def _load_series(uri: str, run_id: str):
    """Download + read the series.parquet artifact -> a [date, ret] DataFrame, or None on any
    failure (missing/corrupt artifact => series plots are simply omitted)."""
    import pandas as pd
    try:
        local = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path="series.parquet", tracking_uri=uri)
        df = pd.read_parquet(local)
        return df if {"date", "ret"}.issubset(df.columns) and len(df) else None
    except Exception:
        return None


def _equity(ret):
    return (1.0 + ret).cumprod()


def plot_equity(df, strategy: str, out: Path) -> str | None:
    import numpy as np  # noqa: F401
    eq = _equity(df["ret"])
    fig, ax = plt.subplots()
    ax.plot(range(len(eq)), eq.to_numpy())
    ax.set_xlabel("bar")
    ax.set_ylabel("equity (start=1.0)")
    ax.set_title(f"{strategy} — equity curve")
    ax.grid(True, alpha=0.3)
    _savefig(fig, out)
    return out.name


def plot_drawdown(df, strategy: str, out: Path) -> str | None:
    eq = _equity(df["ret"])
    # Peak floored at starting capital (1.0) — matches the canonical max-drawdown in metrics.py.
    dd = eq / eq.cummax().clip(lower=1.0) - 1.0
    fig, ax = plt.subplots()
    ax.fill_between(range(len(dd)), dd.to_numpy(), 0.0, color="#c44e52", alpha=0.6)
    ax.set_xlabel("bar")
    ax.set_ylabel("drawdown")
    ax.set_title(f"{strategy} — drawdown")
    ax.grid(True, alpha=0.3)
    _savefig(fig, out)
    return out.name


def plot_rolling_sharpe(df, strategy: str, out: Path, window: int = 63) -> str | None:
    import numpy as np
    ret = df["ret"]
    if len(ret) < window:
        return None  # not enough observations for a single full window
    mean = ret.rolling(window).mean()
    std = ret.rolling(window).std()
    # std == 0 (a flat window) -> Sharpe undefined: NaN (a gap), never a fake 0 or inf.
    rs = (mean / std.replace(0.0, np.nan)) * np.sqrt(252.0)
    fig, ax = plt.subplots()
    ax.plot(range(len(rs)), rs.to_numpy())
    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_xlabel("bar")
    ax.set_ylabel(f"rolling Sharpe ({window}d, ann.)")
    ax.set_title(f"{strategy} — rolling Sharpe (window={window})")
    ax.grid(True, alpha=0.3)
    _savefig(fig, out)
    return out.name


def plot_return_dist(df, strategy: str, out: Path) -> str | None:
    fig, ax = plt.subplots()
    ax.hist(df["ret"].to_numpy(), bins=40, color="#4c72b0")
    ax.set_xlabel("daily return")
    ax.set_ylabel("frequency")
    ax.set_title(f"{strategy} — return distribution")
    ax.grid(True, axis="y", alpha=0.3)
    _savefig(fig, out)
    return out.name
```

- [ ] **Step 2: Wire series selection + plots into `main()`.** Add a `--backtest-run-id` arg and, after the `cross_run` block (before building `lines`), insert:

```python
    # --- series plots (#181): read the LOGGED backtest's series.parquet artifact (no re-run) ---
    bt_runs = [r for r in runs if r.data.tags.get("kind") == "backtest"]
    chosen = None
    if args.backtest_run_id:
        chosen = next((r for r in bt_runs if r.info.run_id == args.backtest_run_id), None)
    # Prefer a backtest whose identity matches the report's walk-forward (then sweep) run.
    ref = provenance.get("walk_forward") or provenance.get("sweep") or {}
    if chosen is None:
        for r in bt_runs:
            res = _artifact_json(r.info.run_id, "result.json", uri)
            if res is None or _load_series(uri, r.info.run_id) is None:
                continue
            if ref and (res.get("config_hash"), res.get("snapshot_id")) == (
                    ref.get("config_hash"), ref.get("snapshot_id")):
                chosen, chosen_res = r, res
                break
    if chosen is None:  # fall back to newest backtest with a series artifact (label its identity)
        for r in bt_runs:
            res = _artifact_json(r.info.run_id, "result.json", uri)
            if res is not None and _load_series(uri, r.info.run_id) is not None:
                chosen, chosen_res = r, res
                break
    if chosen is not None:
        df = _load_series(uri, chosen.info.run_id)
        provenance["series"] = _series_provenance(uri, chosen, chosen_res)
        for fn, name in (
            (plot_equity(df, args.strategy, rdir / "series_equity.svg"), "Equity curve"),
            (plot_drawdown(df, args.strategy, rdir / "series_drawdown.svg"), "Drawdown"),
            (plot_rolling_sharpe(df, args.strategy, rdir / "series_rolling_sharpe.svg"),
             "Rolling Sharpe (63d)"),
            (plot_return_dist(df, args.strategy, rdir / "series_return_dist.svg"),
             "Return distribution"),
        ):
            if fn:
                figs.append((name, fn))
```

Add the argparse line in `main()` next to the others:

```python
    ap.add_argument("--backtest-run-id", default=None,
                    help="MLflow run_id of the backtest whose series.parquet to plot (#181)")
```

- [ ] **Step 3: Update the prose.**
  - In `## v1 scope`, move the "Deferred (out of scope): equity curve, drawdown, rolling Sharpe, return distribution…" paragraph into a new `## v2 scope (series plots, #181)` section describing the four plots and that they read the `series.parquet` artifact logged by a `--track`ed `backtest run` (no re-run; logged-run identity; never sweep/walk-forward — no holdout tail).
  - In the Playbook **Preflight** step, change the matplotlib note: it is now a **direct** dependency (drop the `uv add matplotlib` fallback; keep `uv run python -c "import matplotlib"` as a friendly check).
  - In `## Notes`, document the rolling-Sharpe **window=63** (≈ one quarter), the NaN warm-up/flat-window behavior, and that series plots are omitted when no `--track`ed backtest run has a `series.parquet`.

- [ ] **Step 4: Dry-run validation.**

```bash
uv run algua backtest run cross_sectional_momentum --demo --start 2023-01-01 --end 2023-12-31 --track
# extract the embedded script and run it:
python - <<'PY'
import re, pathlib
src = pathlib.Path(".codex/skills/report-experiments/SKILL.md").read_text()
code = re.search(r"```python\n(.*?)\n```", src, re.S).group(1)
pathlib.Path("/tmp/report_experiments.py").write_text(code)
print("extracted", len(code), "chars")
PY
uv run python /tmp/report_experiments.py cross_sectional_momentum --family demo
ls kb/strategies/cross_sectional_momentum/reports/*/series_*.svg
```
Expected: four `series_*.svg` files exist; the generated `report.md` has a `series` provenance block. (Clean up the throwaway `kb/strategies/cross_sectional_momentum/reports/...` dir and any `mlruns/` created for the dry-run — do NOT commit them.)

- [ ] **Step 5: Gate + commit** (skill doc only)

```bash
uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports
git add .codex/skills/report-experiments/SKILL.md
git commit -m "feat(181): report-experiments renders equity/drawdown/rolling-Sharpe/return-dist

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- Part A (`--emit-series`, fail-closed, descriptor, atomic write) → Tasks 1, 3 ✓
- Part B (series artifact at `--track`, sweep/wf excluded) → Task 4 ✓
- Part C (skill reads artifact, 4 plots, selection, degrade) → Task 6 ✓
- Part D (matplotlib direct dep) → Task 5 ✓
- Shared plumbing (`series_frame` pure, `frame_to_parquet_bytes(metadata)`, `write_bytes_atomic`) → Tasks 1, 2 ✓
- Determinism (sorted keys, `-0.0`, deterministic CLI emit) → Tasks 1, 2, 3 (`test_cli_emit_series_deterministic`) ✓
- Degenerate series (None/empty fail-closed; rolling-Sharpe NaN; n<window skip) → Tasks 3, 6 ✓
- Drawdown peak floored at 1.0 → Task 6 ✓
- Holdout: only `log_backtest` emits a series → Task 4 (`test_only_log_backtest_emits_series_artifact`) ✓
- Deferred global holdout guard → out of scope (spec Decisions); not a task ✓

**Type consistency:** `series_frame(result) -> (DataFrame, dict[str,str])` used identically in Tasks 3 & 4; `frame_to_parquet_bytes(frame, metadata)` signature consistent across Tasks 1, 3, 4; `emit_series_file(result, Path) -> dict` matches its call site in `run`.

**Placeholders:** none — every code step is complete; the only non-code verification is the Task 6 dry-run (inherent to a skill doc).
