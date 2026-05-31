# MLflow Experiment Tracking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add opt-in `--track` to `backtest run` / `walk-forward` / `sweep` so each evaluation is logged to MLflow (per-eval run; sweep = parent + nested child-per-combo), queryable via the MLflow UI/API.

**Architecture:** A new `algua/tracking/mlflow_tracker.py` (lazy-imports `mlflow`) logs params/metrics/tags + a JSON artifact per run, grouping by experiment = strategy name. The CLI calls it only when `--track` is passed, using a configurable local file backend (`mlruns/`). The backtest engine stays pure of tracking (enforced by import-linter).

**Tech Stack:** Python 3.12, MLflow (local file backend), Typer, pytest. Builds on `walkforward.py` + `sweep.py`.

**Key existing types:**
- `BacktestResult` (`algua/backtest/result.py`): `.strategy`, `.metrics` (dict, all numeric), `.config_hash`, `.data_source`, `.timeframe`, `.period` ({start,end}), `.seed`, `.snapshot_id`, `.to_dict()`.
- `WalkForwardResult` (`algua/backtest/walkforward.py`): `.strategy`, `.config_hash`, `.data_source`, `.snapshot_id`, `.timeframe`, `.seed`, `.period`, `.windows`, `.holdout_frac`, `.stability` (flat numeric dict), `.holdout_metrics` (dict: `n_bars`/`sharpe`/`total_return`/`max_drawdown` numeric + `start`/`end` str), `.to_dict()`.
- `SweepResult` (`algua/backtest/sweep.py`): `.strategy`, `.data_source`, `.snapshot_id`, `.timeframe`, `.seed`, `.period`, `.windows`, `.holdout_frac`, `.grid` (dict[str,list]), `.n_combos`, `.rank_by`, `.ranked` (list of `{params, config_hash, n_windows, stability, holdout, score}`), `.best` ({params,score}), `.to_dict()`.
- CLI (`algua/cli/backtest_cmd.py`): commands `run`, `walk_forward_cmd`, `sweep_cmd`; helpers `_utc`, `_select_provider`; `emit`; `load_strategy`; `get_settings`; `@json_errors(ValueError, LookupError, BacktestError)`.

---

### Task 1: Add MLflow dependency + tracking-URI setting

**Files:**
- Modify: `pyproject.toml` (dependency)
- Modify: `algua/config/settings.py`
- Test: `tests/test_config.py` (append)

- [ ] **Step 1: Add the dependency**

Run: `uv add mlflow`
This appends `mlflow>=...` to `[project].dependencies` and updates `uv.lock`. (Heavy install — ~80+ transitive packages; verified to co-resolve with pinned pandas 2.3.3 / numpy 2.4.6.)

- [ ] **Step 2: Confirm it imports**

Run: `uv run python -c "import mlflow; print(mlflow.__version__)"`
Expected: prints a 3.x version, no error.

- [ ] **Step 3: Write the failing settings test (append to `tests/test_config.py`)**

```python
def test_mlflow_tracking_uri_default_and_override(monkeypatch):
    from algua.config.settings import get_settings
    assert get_settings().mlflow_tracking_uri == "mlruns"
    monkeypatch.setenv("ALGUA_MLFLOW_TRACKING_URI", "/tmp/x/mlruns")
    assert get_settings().mlflow_tracking_uri == "/tmp/x/mlruns"
```

- [ ] **Step 4: Run it, verify it FAILS**

Run: `uv run pytest tests/test_config.py::test_mlflow_tracking_uri_default_and_override -v`
Expected: FAIL (`Settings` has no `mlflow_tracking_uri`).

- [ ] **Step 5: Add the field to `algua/config/settings.py`**

Add this field to the `Settings` model (alongside `exchange`, `timezone`, etc.):

```python
    mlflow_tracking_uri: str = "mlruns"
```

- [ ] **Step 6: Run test, verify PASS**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock algua/config/settings.py tests/test_config.py
git commit -m "build: add mlflow dependency and tracking-uri setting"
```

---

### Task 2: Tracker module — `_flatten`, `_numeric_metrics`, `log_backtest`

**Files:**
- Create: `algua/tracking/__init__.py` (empty), `algua/tracking/mlflow_tracker.py`
- Test: `tests/test_tracking_backtest.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tracking_backtest.py
from algua.backtest.result import BacktestResult
from algua.tracking.mlflow_tracker import _flatten, log_backtest


def test_flatten_nested():
    assert _flatten({"a": 1, "b": {"c": 2, "d": 3}}) == {"a": 1, "b.c": 2, "b.d": 3}


def _result():
    return BacktestResult(
        strategy="ew", metrics={"sharpe": 1.25, "cagr": 0.2, "n_rebalances": 7},
        config_hash="abc123", data_source="SyntheticProvider", timeframe="1d",
        period={"start": "2022-01-01", "end": "2023-12-31"}, seed=0, snapshot_id=None,
    )


def test_log_backtest_records_run(tmp_path):
    from mlflow.tracking import MlflowClient

    uri = str(tmp_path / "mlruns")
    run_id = log_backtest(_result(), {"lookback": 60, "top_k": 3}, tracking_uri=uri)

    client = MlflowClient(tracking_uri=uri)
    exp = client.get_experiment_by_name("ew")
    assert exp is not None
    runs = client.search_runs([exp.experiment_id])
    assert len(runs) == 1
    run = runs[0]
    assert run.info.run_id == run_id
    assert abs(run.data.metrics["sharpe"] - 1.25) < 1e-9
    assert run.data.params["config_hash"] == "abc123"
    assert run.data.params["param.lookback"] == "60"
    assert run.data.tags["kind"] == "backtest"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tracking_backtest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'algua.tracking.mlflow_tracker'`.

- [ ] **Step 3: Write the implementation**

```python
# algua/tracking/mlflow_tracker.py
from __future__ import annotations

from typing import Any

from algua.backtest.result import BacktestResult


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten a nested dict into `prefix.key` entries."""
    out: dict[str, Any] = {}
    for key, value in d.items():
        full = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(_flatten(value, f"{full}."))
        else:
            out[full] = value
    return out


def _numeric_metrics(d: dict[str, Any]) -> dict[str, float]:
    """Keep only real numeric values (drops None/str/bool), as floats — safe for log_metrics."""
    return {
        k: float(v)
        for k, v in d.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }


def _stamp_params(result: Any) -> dict[str, Any]:
    return {
        "config_hash": result.config_hash,
        "snapshot_id": result.snapshot_id,
        "seed": result.seed,
        "period_start": result.period["start"],
        "period_end": result.period["end"],
        "timeframe": result.timeframe,
    }


def log_backtest(result: BacktestResult, params: dict[str, Any], *, tracking_uri: str) -> str:
    """Log a single backtest as an MLflow run (experiment = strategy name). Returns run_id."""
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(result.strategy)
    with mlflow.start_run() as run:
        mlflow.log_params({
            **{f"param.{k}": v for k, v in params.items()},
            **_stamp_params(result),
        })
        mlflow.log_metrics(_numeric_metrics(result.metrics))
        mlflow.set_tags({
            "kind": "backtest", "strategy": result.strategy,
            "config_hash": result.config_hash, "snapshot_id": str(result.snapshot_id),
            "timeframe": result.timeframe,
        })
        mlflow.log_dict(result.to_dict(), "result.json")
        return run.info.run_id
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_tracking_backtest.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add algua/tracking/__init__.py algua/tracking/mlflow_tracker.py tests/test_tracking_backtest.py
git commit -m "feat: add mlflow tracker with log_backtest"
```

---

### Task 3: `log_walk_forward`

**Files:**
- Modify: `algua/tracking/mlflow_tracker.py`
- Test: `tests/test_tracking_walkforward.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tracking_walkforward.py
from algua.backtest.walkforward import WalkForwardResult
from algua.tracking.mlflow_tracker import log_walk_forward


def _wf():
    return WalkForwardResult(
        strategy="ew", config_hash="abc", data_source="SyntheticProvider", snapshot_id=None,
        timeframe="1d", seed=0, period={"start": "2022-01-01", "end": "2023-12-31"},
        windows=4, holdout_frac=0.2,
        window_metrics=[{"index": 0, "start": "2022-01-03", "end": "2022-06-01", "n_bars": 100,
                         "total_return": 0.1, "ann_return": 0.2, "ann_volatility": 0.15,
                         "sharpe": 1.3, "max_drawdown": -0.05}],
        holdout_metrics={"start": "2023-06-01", "end": "2023-12-31", "n_bars": 120,
                         "total_return": 0.05, "ann_return": 0.1, "ann_volatility": 0.12,
                         "sharpe": 0.8, "max_drawdown": -0.07},
        stability={"mean_sharpe": 1.1, "std_sharpe": 0.3, "min_sharpe": 0.7,
                   "pct_positive_windows": 0.75},
    )


def test_log_walk_forward_records_metrics(tmp_path):
    from mlflow.tracking import MlflowClient

    uri = str(tmp_path / "mlruns")
    log_walk_forward(_wf(), {"lookback": 60}, tracking_uri=uri)

    client = MlflowClient(tracking_uri=uri)
    exp = client.get_experiment_by_name("ew")
    runs = client.search_runs([exp.experiment_id])
    assert len(runs) == 1
    m = runs[0].data.metrics
    assert abs(m["mean_sharpe"] - 1.1) < 1e-9          # stability flattened
    assert abs(m["holdout.sharpe"] - 0.8) < 1e-9       # holdout flattened, numeric only
    assert "holdout.start" not in m                    # non-numeric excluded from metrics
    assert runs[0].data.tags["kind"] == "walk_forward"
    assert runs[0].data.params["windows"] == "4"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tracking_walkforward.py -v`
Expected: FAIL with `ImportError: cannot import name 'log_walk_forward'`.

- [ ] **Step 3: Add to `algua/tracking/mlflow_tracker.py`**

Add the import and function:

```python
from algua.backtest.walkforward import WalkForwardResult


def log_walk_forward(
    result: WalkForwardResult, params: dict[str, Any], *, tracking_uri: str
) -> str:
    """Log a walk-forward evaluation as one MLflow run. Returns run_id."""
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(result.strategy)
    with mlflow.start_run() as run:
        mlflow.log_params({
            **{f"param.{k}": v for k, v in params.items()},
            "windows": result.windows, "holdout_frac": result.holdout_frac,
            **_stamp_params(result),
        })
        mlflow.log_metrics({
            **_numeric_metrics(result.stability),
            **_numeric_metrics(_flatten(result.holdout_metrics, "holdout.")),
        })
        mlflow.set_tags({
            "kind": "walk_forward", "strategy": result.strategy,
            "config_hash": result.config_hash, "snapshot_id": str(result.snapshot_id),
            "timeframe": result.timeframe,
        })
        mlflow.log_dict(result.to_dict(), "result.json")
        return run.info.run_id
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_tracking_walkforward.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add algua/tracking/mlflow_tracker.py tests/test_tracking_walkforward.py
git commit -m "feat: add log_walk_forward to mlflow tracker"
```

---

### Task 4: `log_sweep` (parent + nested child per combo)

**Files:**
- Modify: `algua/tracking/mlflow_tracker.py`
- Test: `tests/test_tracking_sweep.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tracking_sweep.py
from algua.backtest.sweep import SweepResult
from algua.tracking.mlflow_tracker import log_sweep


def _combo(lookback, score):
    return {
        "params": {"lookback": lookback, "top_k": 1}, "config_hash": f"h{lookback}",
        "n_windows": 4,
        "stability": {"mean_sharpe": score, "std_sharpe": 0.2, "min_sharpe": score - 0.3,
                      "pct_positive_windows": 0.75},
        "holdout": {"n_bars": 100, "sharpe": 0.5, "total_return": 0.04, "max_drawdown": -0.06},
        "score": score,
    }


def _sweep():
    return SweepResult(
        strategy="ew", data_source="SyntheticProvider", snapshot_id=None, timeframe="1d", seed=0,
        period={"start": "2022-01-01", "end": "2023-12-31"}, windows=4, holdout_frac=0.2,
        grid={"lookback": [20, 40], "top_k": [1]}, n_combos=2, rank_by="mean_sharpe",
        ranked=[_combo(20, 1.4), _combo(40, 1.1)],
        best={"params": {"lookback": 20, "top_k": 1}, "score": 1.4},
    )


def test_log_sweep_parent_and_children(tmp_path):
    from mlflow.tracking import MlflowClient

    uri = str(tmp_path / "mlruns")
    parent_id = log_sweep(_sweep(), tracking_uri=uri)

    client = MlflowClient(tracking_uri=uri)
    exp = client.get_experiment_by_name("ew")
    runs = client.search_runs([exp.experiment_id])
    parents = [r for r in runs if r.data.tags.get("kind") == "sweep"]
    children = [r for r in runs if r.data.tags.get("kind") == "sweep_combo"]
    assert len(parents) == 1 and parents[0].info.run_id == parent_id
    assert parents[0].data.params["n_combos"] == "2"
    assert abs(parents[0].data.metrics["best_score"] - 1.4) < 1e-9
    assert len(children) == 2
    for child in children:
        assert child.data.tags["mlflow.parentRunId"] == parent_id
        assert "score" in child.data.metrics
        assert "param.lookback" in child.data.params
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tracking_sweep.py -v`
Expected: FAIL with `ImportError: cannot import name 'log_sweep'`.

- [ ] **Step 3: Add to `algua/tracking/mlflow_tracker.py`**

Add the import and function:

```python
from algua.backtest.sweep import SweepResult


def log_sweep(result: SweepResult, *, tracking_uri: str) -> str:
    """Log a sweep as a parent run with a nested child run per ranked combo. Returns parent run_id."""
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(result.strategy)
    with mlflow.start_run() as parent:
        mlflow.log_params({
            **{f"grid.{k}": ",".join(str(x) for x in vals) for k, vals in result.grid.items()},
            "n_combos": result.n_combos, "rank_by": result.rank_by,
            "windows": result.windows, "holdout_frac": result.holdout_frac,
            "snapshot_id": result.snapshot_id, "seed": result.seed,
            "period_start": result.period["start"], "period_end": result.period["end"],
            "timeframe": result.timeframe,
        })
        if result.best is not None:
            mlflow.log_metric("best_score", float(result.best["score"]))
        mlflow.set_tags({"kind": "sweep", "strategy": result.strategy})
        mlflow.log_dict(result.to_dict(), "sweep.json")

        for entry in result.ranked:
            with mlflow.start_run(nested=True):
                mlflow.log_params({
                    **{f"param.{k}": v for k, v in entry["params"].items()},
                    "config_hash": entry["config_hash"],
                })
                mlflow.log_metrics({
                    "score": float(entry["score"]),
                    **_numeric_metrics(_flatten(entry["stability"])),
                    **_numeric_metrics(_flatten(entry["holdout"], "holdout.")),
                })
                mlflow.set_tags({"kind": "sweep_combo", "strategy": result.strategy})
        return parent.info.run_id
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_tracking_sweep.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Gate + commit**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: all green (import-linter still 4 kept here — the new boundary contract is added in Task 5).
```bash
git add algua/tracking/mlflow_tracker.py tests/test_tracking_sweep.py
git commit -m "feat: add log_sweep with nested per-combo runs"
```

---

### Task 5: CLI `--track` + import-linter boundary

**Files:**
- Modify: `algua/cli/backtest_cmd.py`
- Modify: `pyproject.toml` (import-linter contract)
- Test: `tests/test_cli_track.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli_track.py
import json
import pytest
from typer.testing import CliRunner
from algua.cli.main import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def _tmp(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ALGUA_MLFLOW_TRACKING_URI", str(tmp_path / "mlruns"))


def _runs(tmp_path, experiment="cross_sectional_momentum"):
    from mlflow.tracking import MlflowClient
    client = MlflowClient(tracking_uri=str(tmp_path / "mlruns"))
    exp = client.get_experiment_by_name(experiment)
    return [] if exp is None else client.search_runs([exp.experiment_id])


def test_run_track_logs_a_run(tmp_path):
    result = runner.invoke(app, ["backtest", "run", "cross_sectional_momentum", "--demo",
                                 "--start", "2022-01-01", "--end", "2023-12-31", "--track"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["mlflow_run_id"]
    assert len(_runs(tmp_path)) == 1


def test_run_without_track_logs_nothing(tmp_path):
    result = runner.invoke(app, ["backtest", "run", "cross_sectional_momentum", "--demo",
                                 "--start", "2022-01-01", "--end", "2023-12-31"])
    assert result.exit_code == 0, result.stdout
    assert "mlflow_run_id" not in json.loads(result.stdout)
    assert _runs(tmp_path) == []


def test_sweep_track_logs_parent_and_children(tmp_path):
    result = runner.invoke(app, ["backtest", "sweep", "cross_sectional_momentum", "--demo",
                                 "--start", "2022-01-01", "--end", "2023-12-31",
                                 "--param", "lookback=20,40", "--track"])
    assert result.exit_code == 0, result.stdout
    runs = _runs(tmp_path)
    assert sum(1 for r in runs if r.data.tags.get("kind") == "sweep") == 1
    assert sum(1 for r in runs if r.data.tags.get("kind") == "sweep_combo") == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli_track.py -v`
Expected: FAIL — no `--track` option / `mlflow_run_id` absent.

- [ ] **Step 3: Edit `algua/cli/backtest_cmd.py`**

Add the tracker imports (with the other imports):
```python
from algua.tracking.mlflow_tracker import log_backtest, log_sweep, log_walk_forward
```
Add `track: bool = typer.Option(False, "--track", help="log this run to MLflow")` to the option list of **`run`**, **`walk_forward_cmd`**, and **`sweep_cmd`**.

In `run`, after `result = run_backtest(...)` and the `--register` block, replace the final `emit(result.to_dict())` with:
```python
    payload = result.to_dict()
    if track:
        payload["mlflow_run_id"] = log_backtest(
            result, strategy.config.params, tracking_uri=get_settings().mlflow_tracking_uri
        )
    emit(payload)
```

In `walk_forward_cmd`, replace `emit(result.to_dict())` with:
```python
    payload = result.to_dict()
    if track:
        payload["mlflow_run_id"] = log_walk_forward(
            result, strategy.config.params, tracking_uri=get_settings().mlflow_tracking_uri
        )
    emit(payload)
```

In `sweep_cmd`, the body currently builds `payload = result.to_dict()` then truncates `ranked`.
Log the FULL result before truncation:
```python
    run_id = None
    if track:
        run_id = log_sweep(result, tracking_uri=get_settings().mlflow_tracking_uri)
    payload = result.to_dict()
    payload["ranked"] = payload["ranked"][:top]
    if run_id is not None:
        payload["mlflow_run_id"] = run_id
    emit(payload)
```

- [ ] **Step 4: Add the import-linter boundary contract to `pyproject.toml`**

Append a new contract under `[tool.importlinter]`:
```toml
[[tool.importlinter.contracts]]
name = "backtest engine stays off the tracking layer"
type = "forbidden"
source_modules = ["algua.backtest"]
forbidden_modules = ["algua.tracking"]
```

- [ ] **Step 5: Run tests + full gate**

Run:
```bash
uv run pytest tests/test_cli_track.py tests/test_cli_backtest.py tests/test_cli_walkforward.py tests/test_cli_sweep.py -q
uv run pytest -q
uv run ruff check .
uv run mypy algua
uv run lint-imports
```
Expected: all pass; ruff/mypy clean; import-linter now `5 kept, 0 broken` (the new "backtest off tracking" contract holds — `tracking` imports `backtest`, never the reverse).

- [ ] **Step 6: Commit**

```bash
git add algua/cli/backtest_cmd.py pyproject.toml tests/test_cli_track.py
git commit -m "feat: add --track to backtest commands; enforce backtest-off-tracking boundary"
```

---

### Task 6: Full verification & smoke

**Files:** none (verification only)

- [ ] **Step 1: Full quality gate**

Run:
```bash
uv run pytest -q
uv run ruff check .
uv run mypy algua
uv run lint-imports
```
Expected: all pass; ruff clean; mypy `Success`; import-linter `5 kept, 0 broken`.

- [ ] **Step 2: CLI smoke (synthetic, with tracking to a temp dir)**

Run:
```bash
export ALGUA_MLFLOW_TRACKING_URI="$(mktemp -d)/mlruns"
uv run algua backtest sweep cross_sectional_momentum --demo --start 2021-01-01 --end 2023-12-31 \
    --param lookback=20,40,60 --param top_k=1,2 --track --top 3
uv run python -c "
from mlflow.tracking import MlflowClient
import os
c = MlflowClient(tracking_uri=os.environ['ALGUA_MLFLOW_TRACKING_URI'])
e = c.get_experiment_by_name('cross_sectional_momentum')
runs = c.search_runs([e.experiment_id])
print('sweep parents:', sum(1 for r in runs if r.data.tags.get('kind')=='sweep'))
print('sweep_combo children:', sum(1 for r in runs if r.data.tags.get('kind')=='sweep_combo'))
"
```
Expected: sweep JSON includes `mlflow_run_id`; the query prints `sweep parents: 1` and `sweep_combo children: 6` (3×2 grid).

- [ ] **Step 3: Final commit (if any verification fixes were needed)**

```bash
git add -A
git commit -m "test: verify mlflow tracking end to end" --allow-empty
```

---

## Self-Review Notes

- **Spec coverage:** dependency + `mlflow_tracking_uri` setting (Task 1), tracker `_flatten`/
  `_numeric_metrics`/`log_backtest` (Task 2), `log_walk_forward` (Task 3), `log_sweep` parent+
  nested children (Task 4), CLI `--track` on all three commands + `mlflow_run_id` in output +
  import boundary (Task 5), verification (Task 6). Out-of-scope (promotion gates, server backend,
  default-on tracking) intentionally absent.
- **Boundary:** `tracking/` imports `mlflow` + `backtest` result types; the new import-linter
  contract forbids `algua.backtest` from importing `algua.tracking`, keeping the engine pure. CLI
  → tracking is allowed. `mlflow` is lazy-imported inside tracker functions.
- **Type consistency:** `_flatten(d, prefix="")`, `_numeric_metrics(d) -> dict[str,float]`,
  `_stamp_params(result)`, `log_backtest(result, params, *, tracking_uri) -> str`,
  `log_walk_forward(result, params, *, tracking_uri) -> str`, `log_sweep(result, *, tracking_uri)
  -> str`, and the CLI's `mlflow_run_id` output key are consistent across tasks/tests. Non-numeric
  values (holdout `start`/`end`, `snapshot_id`) go to params/tags, never `log_metrics`.
