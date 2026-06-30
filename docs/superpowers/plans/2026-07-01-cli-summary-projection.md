# CLI `--summary` field projection (#349) Implementation Plan

> **For agentic workers:** implement task-by-task; run the quality gate between tasks.

**Goal:** Add an opt-in `--summary` flag to `backtest walk-forward`, `backtest sweep`, and
`research promote` that projects the JSON to decision-relevant scalars (context-rot defense).

**Architecture:** One shared `project(payload, keep)` helper in `cli/_common.py` (preserves
`ok`, stamps `summary: true`, keeps only listed keys). Each command defines a `_SUMMARY_KEYS`
tuple and applies the helper as the last step before `emit(ok(...))`. Non-breaking: default
output unchanged.

**Tech Stack:** Python, Typer, pytest (`typer.testing.CliRunner`).

## Global Constraints

- Quality gate (run between tasks + before PR): `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`.
- Projection applies ONLY to success payloads; the `@json_errors` error envelope is never projected.
- Keep-lists, not drop-lists.
- Scoped `git add` (never `-A`).

---

### Task 1: `project()` helper in `cli/_common.py`

**Files:**
- Modify: `algua/cli/_common.py`
- Test: `tests/test_cli_summary.py` (new)

- [ ] Add to `algua/cli/_common.py` (note `Collection` is already imported):

```python
def project(payload: dict, keep: Collection[str]) -> dict:
    """Project a success payload to its decision-relevant subset for ``--summary``
    (context-rot defense, #349). Always preserves the ``ok`` discriminator, stamps
    ``summary: True`` so a consumer can distinguish a projected payload from a full one,
    and keeps only the listed keys that are present. Success payloads only — the
    ``@json_errors`` failure envelope never reaches this."""
    return {k: v for k, v in payload.items() if k == "ok" or k in keep} | {"summary": True}
```

- [ ] Test (new file `tests/test_cli_summary.py`):

```python
from algua.cli._common import project


def test_project_keeps_listed_and_ok_adds_marker():
    out = project({"ok": True, "a": 1, "b": 2, "c": 3}, keep=("a", "c"))
    assert out == {"ok": True, "a": 1, "c": 3, "summary": True}


def test_project_ignores_absent_keys():
    out = project({"ok": True, "a": 1}, keep=("a", "missing"))
    assert out == {"ok": True, "a": 1, "summary": True}
```

- [ ] Run: `uv run pytest tests/test_cli_summary.py -q` → PASS. Commit.

---

### Task 2: `--summary` on `backtest walk-forward` and `backtest sweep`

**Files:**
- Modify: `algua/cli/backtest_cmd.py`
- Test: `tests/test_cli_summary.py`

- [ ] In `backtest_cmd.py` import `project` from `algua.cli._common`; define module-level tuples:

```python
_WF_SUMMARY_KEYS = (
    "strategy", "data_source", "snapshot_id", "timeframe", "seed", "period", "windows",
    "holdout_frac", "stability", "code_hash", "dependency_hash", "config_hash",
    "universe_name", "universe_snapshots", "fundamentals_snapshot", "news_snapshot",
    "mlflow_run_id",
)
_SWEEP_SUMMARY_KEYS = (
    "strategy", "n_combos", "rank_by", "best", "trial_sharpe_count", "trial_sharpe_mean",
    "trial_sharpe_var_ann", "recorded_breadth", "code_hash", "dependency_hash",
    "universe_name", "universe_snapshots", "fundamentals_snapshot", "news_snapshot",
    "mlflow_run_id",
)
```

- [ ] Add `summary: bool = typer.Option(False, "--summary", help="emit only decision-relevant scalars (context-rot defense)")` to both `walk_forward_cmd` and `sweep_cmd`.
- [ ] In `walk_forward_cmd`, change the final emit to:

```python
    out = ok(payload)
    emit(project(out, _WF_SUMMARY_KEYS) if summary else out)
```

(`payload` already has `holdout_metrics` popped and `mlflow_run_id` added when `--track`.)

- [ ] In `sweep_cmd`, change the final emit to:

```python
    out = ok(payload)
    emit(project(out, _SWEEP_SUMMARY_KEYS) if summary else out)
```

(`payload` already has `ranked` capped to `--top`, `recorded_breadth`, and `mlflow_run_id`.)

- [ ] Tests (append to `tests/test_cli_summary.py`, demo provider):

```python
import json
from typer.testing import CliRunner
from algua.cli.main import app

runner = CliRunner()


def _run(args):
    res = runner.invoke(app, args)
    assert res.exit_code == 0, res.output
    return json.loads(res.output)


def test_walk_forward_summary_projects():
    full = _run(["backtest", "walk-forward", "demo_sma", "--demo"])
    summ = _run(["backtest", "walk-forward", "demo_sma", "--demo", "--summary"])
    assert "window_metrics" in full and "summary" not in full
    assert "window_metrics" not in summ
    assert summ["ok"] is True and summ["summary"] is True
    assert summ["stability"] == full["stability"]


def test_sweep_summary_projects():
    base = ["backtest", "sweep", "demo_sma", "--demo", "--param", "fast=5,10"]
    full = _run(base)
    summ = _run(base + ["--summary"])
    assert "ranked" in full and "grid" in full and "summary" not in full
    assert "ranked" not in summ and "grid" not in summ
    assert summ["summary"] is True and summ["best"] == full["best"]
```

(Use the actual demo strategy name + a real param key present in the repo — verify with
`uv run algua registry list` / the strategy module before finalizing the test args.)

- [ ] Run `uv run pytest tests/test_cli_summary.py -q` → PASS. Run full gate. Commit.

---

### Task 3: `--summary` on `research promote`

**Files:**
- Modify: `algua/cli/research_cmd.py`
- Test: `tests/test_cli_summary.py`

- [ ] In `research_cmd.py` import `project`; define:

```python
_PROMOTE_SUMMARY_KEYS = (
    "promoted", "strategy", "passed", "checks", "n_combos", "n_funnel",
    "breadth_provenance", "base_min_holdout_sharpe", "effective_min_holdout_sharpe",
    "pit_ok", "pit_override", "dsr_binding", "fdr_binding", "regime_robustness_binding",
    "holdout", "stability", "config_hash", "snapshot_id", "universe_name",
    "universe_snapshots", "fundamentals_snapshot", "news_snapshot", "holdout_reuse",
)
```

- [ ] Add `summary: bool = typer.Option(False, "--summary", help="emit only decision-relevant scalars (context-rot defense)")` to `promote`.
- [ ] Change the final emit to:

```python
    out = ok(payload)
    emit(project(out, _PROMOTE_SUMMARY_KEYS) if summary else out)
```

- [ ] Test (append; reuse the existing `test_cli_research.py` promote-setup pattern — register a
strategy, get it to `backtested`, ensure a family, record breadth — or import those helpers):

```python
def test_promote_summary_drops_deep_diagnostics():
    # Build a backtested strategy + breadth the same way test_cli_research.py does, then:
    full = _run([...promote args...])
    summ = _run([...promote args..., "--summary"])
    assert "dsr_confidence" in full and "dsr_n_eff" in full and "summary" not in full
    assert "dsr_confidence" not in summ and "haircut_would_have_blocked" not in summ
    assert summ["summary"] is True
    assert summ["promoted"] == full["promoted"] and "checks" in summ
```

(A holdout is single-use — a second full+summary promote of the same window will hit the
burn guard. Drive the two runs against DISTINCT windows/strategies, or assert projection on
ONE promote run plus a unit-level `project(_PROMOTE_SUMMARY_KEYS)` check on a synthetic
decision dict to avoid double-burning. Prefer the unit-level projection test for the
keep/drop assertion + one real promote run asserting `summary` marker presence.)

- [ ] Run `uv run pytest tests/test_cli_summary.py tests/test_cli_research.py -q` → PASS. Full gate. Commit.

---

### Task 4: Docs

**Files:**
- Modify: `CLAUDE.md`
- Modify: `.codex/skills/operating-algua/SKILL.md`

- [ ] Add a short note to the `paper promote`/`research promote`/`backtest` lines (or a single
sentence in the command-surface section) that the heavy commands accept `--summary` for
decision-relevant scalars (context-rot defense, #349).
- [ ] Add a line to `operating-algua` SKILL.md telling the operator to pass `--summary` on
`backtest walk-forward`, `backtest sweep`, and `research promote`.
- [ ] Commit.
