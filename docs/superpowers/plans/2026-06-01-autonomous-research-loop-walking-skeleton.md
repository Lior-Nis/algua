# Autonomous Research Loop — Walking Skeleton Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an unattended research loop that authors momentum parameter-variants, drives each `idea→backtested→shortlisted` through the algua CLI, and stops on a budget/guardrail — runnable in CI with no LLM.

**Architecture:** A pure controller (`run_loop`) depends on injected `AgentRunner` + `Workspace` protocols and a `registered_names` callable. Production wires a `CodexRunner` (subprocess `codex exec`), a `GitWorkspace` (git-status diff), and a CLI-backed names reader; tests wire deterministic fakes. The operator package talks to algua **only through the CLI** (no `cli`/`registry`/`backtest` imports), enforced by an import-linter contract. An additions-only diff-gate protects curated strategies; a consecutive-failure circuit breaker and max-iterations bound the run.

**Tech Stack:** Python 3.12, pydantic v2, Typer, pytest, ruff, mypy, import-linter. Strategies are modules in `algua/strategies/examples/` exposing `CONFIG` + `target_weights`.

**Spec:** `docs/superpowers/specs/2026-06-01-autonomous-research-loop-walking-skeleton-design.md`

---

## File structure

| File | Responsibility |
|---|---|
| `algua/operator/__init__.py` | Package marker. |
| `algua/operator/result.py` | `IterationResult` + `RunSummary` pydantic models (the result contract). |
| `algua/operator/variants.py` | `Variant` + `variant_list()` — the fixed ideation stand-in. |
| `algua/operator/gate.py` | Pure: `Change`, `diff_gate(...)`, `untried_variants(...)`. |
| `algua/operator/loop.py` | Pure: `AgentRunner`/`Workspace` protocols + `run_loop(...)` controller. |
| `algua/operator/adapters.py` | I/O edge: `CodexRunner`, `GitWorkspace`, `registered_names_via_cli`. |
| `algua/cli/operator_cmd.py` | `algua operator run` — wires adapters + `run_loop`, emits JSON. |
| `algua/cli/main.py` | Add `operator_cmd` to the subcommand imports. |
| `pyproject.toml` | New import-linter contract for `algua.operator`. |
| `tests/test_operator_result.py` | Model + schema-generation tests. |
| `tests/test_operator_variants.py` | Variant naming/determinism. |
| `tests/test_operator_gate.py` | Gate + dedup pure-function tests. |
| `tests/test_operator_loop.py` | Controller tests with fakes (happy/dedup/stop/gate/breaker). |
| `tests/test_operator_adapters.py` | `CodexRunner` parse (mocked subprocess) + `GitWorkspace` (tmp git). |
| `tests/test_cli_operator.py` | e2e with `FakeRunner`: positive + negative (gate violation). |

---

### Task 1: Result contract (`IterationResult` + `RunSummary`)

**Files:**
- Create: `algua/operator/__init__.py`
- Create: `algua/operator/result.py`
- Test: `tests/test_operator_result.py`

- [ ] **Step 1: Create the package marker**

Create `algua/operator/__init__.py` (empty file).

- [ ] **Step 2: Write the failing test**

Create `tests/test_operator_result.py`:

```python
import json

from algua.operator.result import IterationResult, RunSummary


def test_iteration_result_roundtrips_and_defaults_error_none():
    r = IterationResult(
        iteration=1, strategy_name="momentum_lb40",
        stage_before=None, stage_after="shortlisted",
        gate_passed=True, promoted=True, config_hash="abc",
        status="ok", reason="gate pass",
    )
    assert r.error is None
    assert IterationResult.model_validate_json(r.model_dump_json()) == r


def test_iteration_result_json_schema_is_object_with_known_fields():
    schema = IterationResult.model_json_schema()
    assert schema["type"] == "object"
    assert "strategy_name" in schema["properties"]
    assert "status" in schema["properties"]
    # The schema must be JSON-serializable (codex --output-schema consumes a file).
    json.dumps(schema)


def test_run_summary_holds_iterations_and_counts():
    s = RunSummary(iterations=[], stopped_reason="max_iterations", n_shortlisted=0)
    assert s.stopped_reason == "max_iterations"
    assert s.iterations == []
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_operator_result.py -q`
Expected: FAIL with `ModuleNotFoundError: algua.operator.result`.

- [ ] **Step 4: Implement the models**

Create `algua/operator/result.py`:

```python
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class IterationResult(BaseModel):
    """The typed outcome of one loop iteration. Serialized to codex --output-schema."""

    iteration: int
    strategy_name: str
    stage_before: str | None
    stage_after: str | None
    gate_passed: bool
    promoted: bool
    config_hash: str | None
    status: Literal["ok", "failed"]
    reason: str
    error: str | None = None


class RunSummary(BaseModel):
    """The result of a whole loop run, emitted as JSON by `algua operator run`."""

    iterations: list[IterationResult]
    stopped_reason: str
    n_shortlisted: int
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_operator_result.py -q`
Expected: PASS (3 passed).

- [ ] **Step 6: Commit**

```bash
git add algua/operator/__init__.py algua/operator/result.py tests/test_operator_result.py
git commit -m "feat(operator): IterationResult + RunSummary result contract"
```

---

### Task 2: Variant source (`variants.py`)

**Files:**
- Create: `algua/operator/variants.py`
- Test: `tests/test_operator_variants.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_operator_variants.py`:

```python
from algua.operator.variants import Variant, variant_list


def test_variant_list_is_deterministic_and_named_by_lookback():
    first = variant_list()
    second = variant_list()
    assert first == second  # deterministic order + values
    assert [v.name for v in first] == [
        "momentum_lb20", "momentum_lb40", "momentum_lb60", "momentum_lb80"
    ]


def test_variants_carry_params():
    v = variant_list()[1]
    assert v == Variant(name="momentum_lb40", lookback=40, top_k=3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_operator_variants.py -q`
Expected: FAIL with `ModuleNotFoundError: algua.operator.variants`.

- [ ] **Step 3: Implement variants**

Create `algua/operator/variants.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

# Fixed ideation stand-in for the walking skeleton: parameter variants of the bundled
# cross_sectional_momentum strategy. Replaced by a real ideation engine in a later slice.
_BASE = "momentum"
_LOOKBACKS = (20, 40, 60, 80)
_TOP_K = 3


@dataclass(frozen=True)
class Variant:
    name: str
    lookback: int
    top_k: int


def variant_list() -> list[Variant]:
    return [
        Variant(name=f"{_BASE}_lb{lb}", lookback=lb, top_k=_TOP_K) for lb in _LOOKBACKS
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_operator_variants.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add algua/operator/variants.py tests/test_operator_variants.py
git commit -m "feat(operator): fixed momentum variant list (skeleton ideation)"
```

---

### Task 3: Gate + dedup pure functions (`gate.py`)

**Files:**
- Create: `algua/operator/gate.py`
- Test: `tests/test_operator_gate.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_operator_gate.py`:

```python
from algua.operator.gate import Change, diff_gate, untried_variants

PREFIX = "algua/strategies/examples/"


def test_diff_gate_allows_new_file_under_prefix():
    changes = [Change(path=PREFIX + "momentum_lb20.py", kind="added")]
    assert diff_gate(changes, PREFIX) == []


def test_diff_gate_flags_modified_existing_file_under_prefix():
    changes = [Change(path=PREFIX + "cross_sectional_momentum.py", kind="modified")]
    assert diff_gate(changes, PREFIX) == changes  # additions-only: a modify is a violation


def test_diff_gate_flags_path_outside_prefix():
    changes = [Change(path="algua/backtest/engine.py", kind="added")]
    assert diff_gate(changes, PREFIX) == changes


def test_diff_gate_flags_deleted_file():
    changes = [Change(path=PREFIX + "cross_sectional_momentum.py", kind="deleted")]
    assert diff_gate(changes, PREFIX) == changes


def test_untried_variants_skips_registered_names():
    assert untried_variants(
        ["momentum_lb20", "momentum_lb40"], registered=["momentum_lb20"]
    ) == ["momentum_lb40"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_operator_gate.py -q`
Expected: FAIL with `ModuleNotFoundError: algua.operator.gate`.

- [ ] **Step 3: Implement the gate**

Create `algua/operator/gate.py`:

```python
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class Change:
    path: str
    kind: str  # "added" | "modified" | "deleted"


def diff_gate(changes: Iterable[Change], allowed_prefix: str) -> list[Change]:
    """Additions-only gate. A change is allowed ONLY if it adds a new file under
    allowed_prefix. Modifying/deleting an existing file, or any path outside the prefix,
    is a violation. Returns the list of violating changes (empty == ok)."""
    return [
        c for c in changes if c.kind != "added" or not c.path.startswith(allowed_prefix)
    ]


def untried_variants(names: Iterable[str], registered: Iterable[str]) -> list[str]:
    """Names not yet present in the registry (cross-run dedup)."""
    reg = set(registered)
    return [n for n in names if n not in reg]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_operator_gate.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add algua/operator/gate.py tests/test_operator_gate.py
git commit -m "feat(operator): additions-only diff-gate + registry dedup"
```

---

### Task 4: Controller (`loop.py`)

**Files:**
- Create: `algua/operator/loop.py`
- Test: `tests/test_operator_loop.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_operator_loop.py`:

```python
from algua.operator.gate import Change
from algua.operator.loop import run_loop
from algua.operator.result import IterationResult
from algua.operator.variants import Variant

PREFIX = "algua/strategies/examples/"
VARIANTS = [Variant(f"momentum_lb{lb}", lb, 3) for lb in (20, 40, 60)]


class ProgrammableRunner:
    """Returns canned IterationResults and authors canned Changes per turn."""

    def __init__(self, results, changes_per_turn):
        self._results = list(results)
        self._changes = list(changes_per_turn)
        self.calls = 0

    def run(self, iteration, variant):
        self.calls += 1
        return self._results.pop(0)

    def changes(self):
        return self._changes.pop(0)


class FakeWorkspace:
    def __init__(self, runner):
        self._runner = runner
        self._current = []

    def begin_turn(self):
        self._current = self._runner.changes()

    def turn_changes(self):
        return self._current


def _ok(i, name, promoted=True):
    return IterationResult(
        iteration=i, strategy_name=name, stage_before=None,
        stage_after="shortlisted" if promoted else "backtested",
        gate_passed=promoted, promoted=promoted, config_hash="h",
        status="ok", reason="fake",
    )


def _added(name):
    return [Change(PREFIX + f"{name}.py", "added")]


def test_happy_path_runs_until_max_iterations():
    runner = ProgrammableRunner(
        results=[_ok(1, "momentum_lb20"), _ok(2, "momentum_lb40")],
        changes_per_turn=[_added("momentum_lb20"), _added("momentum_lb40")],
    )
    summary = run_loop(
        runner=runner, workspace=FakeWorkspace(runner),
        registered_names=lambda: [], all_variants=VARIANTS,
        max_iterations=2, allowed_prefix=PREFIX,
    )
    assert summary.stopped_reason == "max_iterations"
    assert summary.n_shortlisted == 2
    assert [r.strategy_name for r in summary.iterations] == ["momentum_lb20", "momentum_lb40"]


def test_stops_when_no_untried_variants_left():
    runner = ProgrammableRunner(results=[], changes_per_turn=[])
    summary = run_loop(
        runner=runner, workspace=FakeWorkspace(runner),
        registered_names=lambda: [v.name for v in VARIANTS], all_variants=VARIANTS,
        max_iterations=5, allowed_prefix=PREFIX,
    )
    assert summary.stopped_reason == "no_untried_variants"
    assert runner.calls == 0


def test_gate_violation_fails_closed_and_stops():
    runner = ProgrammableRunner(
        results=[_ok(1, "momentum_lb20")],
        changes_per_turn=[[Change("algua/backtest/engine.py", "added")]],
    )
    summary = run_loop(
        runner=runner, workspace=FakeWorkspace(runner),
        registered_names=lambda: [], all_variants=VARIANTS,
        max_iterations=5, allowed_prefix=PREFIX,
    )
    assert summary.stopped_reason == "gate_violation"
    assert summary.n_shortlisted == 0
    assert summary.iterations[-1].status == "failed"
    assert "diff-gate" in summary.iterations[-1].reason


def test_circuit_breaker_stops_after_consecutive_failures():
    fails = [
        IterationResult(iteration=i, strategy_name=VARIANTS[i - 1].name, stage_before=None,
                        stage_after=None, gate_passed=False, promoted=False, config_hash=None,
                        status="failed", reason="boom", error="x")
        for i in range(1, 4)
    ]
    runner = ProgrammableRunner(
        results=fails,
        changes_per_turn=[_added(v.name) for v in VARIANTS],
    )
    summary = run_loop(
        runner=runner, workspace=FakeWorkspace(runner),
        registered_names=lambda: [], all_variants=VARIANTS,
        max_iterations=5, allowed_prefix=PREFIX, max_consecutive_failures=3,
    )
    assert summary.stopped_reason == "circuit_breaker"
    assert runner.calls == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_operator_loop.py -q`
Expected: FAIL with `ModuleNotFoundError: algua.operator.loop`.

- [ ] **Step 3: Implement the controller**

Create `algua/operator/loop.py`:

```python
from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from algua.operator.gate import diff_gate, untried_variants
from algua.operator.result import IterationResult, RunSummary
from algua.operator.variants import Variant


class AgentRunner(Protocol):
    def run(self, iteration: int, variant: Variant) -> IterationResult: ...


class Workspace(Protocol):
    def begin_turn(self) -> None: ...
    def turn_changes(self) -> list: ...  # list[Change]


def run_loop(
    *,
    runner: AgentRunner,
    workspace: Workspace,
    registered_names: Callable[[], list[str]],
    all_variants: list[Variant],
    max_iterations: int,
    allowed_prefix: str,
    max_consecutive_failures: int = 3,
) -> RunSummary:
    """Drive the bounded research loop. Pure orchestration over injected collaborators."""
    results: list[IterationResult] = []
    attempted: set[str] = set()
    consecutive_failures = 0
    stopped_reason = "max_iterations"

    for i in range(1, max_iterations + 1):
        names = [v.name for v in all_variants]
        remaining = [
            v for v in all_variants
            if v.name in set(untried_variants(names, registered_names()))
            and v.name not in attempted
        ]
        if not remaining:
            stopped_reason = "no_untried_variants"
            break

        variant = remaining[0]
        attempted.add(variant.name)

        workspace.begin_turn()
        result = runner.run(i, variant)

        violations = diff_gate(workspace.turn_changes(), allowed_prefix)
        if violations:
            paths = [c.path for c in violations]
            results.append(result.model_copy(update={
                "status": "failed", "promoted": False,
                "reason": f"diff-gate violation: {paths}",
            }))
            stopped_reason = "gate_violation"
            break

        results.append(result)
        if result.status == "failed":
            consecutive_failures += 1
            if consecutive_failures >= max_consecutive_failures:
                stopped_reason = "circuit_breaker"
                break
        else:
            consecutive_failures = 0

    return RunSummary(
        iterations=results,
        stopped_reason=stopped_reason,
        n_shortlisted=sum(1 for r in results if r.promoted),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_operator_loop.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add algua/operator/loop.py tests/test_operator_loop.py
git commit -m "feat(operator): bounded loop controller with gate + circuit breaker"
```

---

### Task 5: I/O adapters (`adapters.py`)

**Files:**
- Create: `algua/operator/adapters.py`
- Test: `tests/test_operator_adapters.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_operator_adapters.py`:

```python
import subprocess

from algua.operator.adapters import CodexRunner, GitWorkspace, registered_names_via_cli
from algua.operator.gate import Change
from algua.operator.variants import Variant

VARIANT = Variant("momentum_lb40", 40, 3)


def test_codex_runner_parses_schema_output(tmp_path, monkeypatch):
    out_payload = (
        '{"iteration":1,"strategy_name":"momentum_lb40","stage_before":null,'
        '"stage_after":"shortlisted","gate_passed":true,"promoted":true,'
        '"config_hash":"h","status":"ok","reason":"ok","error":null}'
    )

    def fake_run(cmd, capture_output, text):
        # codex writes the schema-validated result to the path after `-o`.
        out_path = cmd[cmd.index("-o") + 1]
        with open(out_path, "w") as fh:
            fh.write(out_payload)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = CodexRunner(repo=tmp_path).run(1, VARIANT)
    assert result.promoted is True
    assert result.strategy_name == "momentum_lb40"


def test_codex_runner_returns_failed_on_nonzero_exit(tmp_path, monkeypatch):
    def fake_run(cmd, capture_output, text):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="boom")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = CodexRunner(repo=tmp_path).run(1, VARIANT)
    assert result.status == "failed"
    assert "boom" in (result.error or "")


def test_registered_names_via_cli_parses_list(monkeypatch):
    def fake_run(cmd, capture_output, text, cwd):
        return subprocess.CompletedProcess(
            cmd, 0, stdout='[{"name":"a","stage":"idea"},{"name":"b","stage":"paper"}]',
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert registered_names_via_cli(repo=".") == ["a", "b"]


def test_git_workspace_detects_new_file(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    examples = tmp_path / "algua" / "strategies" / "examples"
    examples.mkdir(parents=True)
    ws = GitWorkspace(repo=tmp_path)
    ws.begin_turn()
    (examples / "momentum_lb40.py").write_text("x = 1\n")
    changes = ws.turn_changes()
    assert Change("algua/strategies/examples/momentum_lb40.py", "added") in changes
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_operator_adapters.py -q`
Expected: FAIL with `ModuleNotFoundError: algua.operator.adapters`.

- [ ] **Step 3: Implement the adapters**

Create `algua/operator/adapters.py`:

```python
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from algua.operator.gate import Change
from algua.operator.result import IterationResult
from algua.operator.variants import Variant

EXAMPLES_DIR = "algua/strategies/examples"

_PORCELAIN_KIND = {"A": "added", "?": "added", "M": "modified", "D": "deleted", "R": "modified"}


def _build_prompt(iteration: int, variant: Variant) -> str:
    return (
        f"You are operating the algua research loop, iteration {iteration}. "
        f"Author a new strategy module at {EXAMPLES_DIR}/{variant.name}.py that re-uses the "
        f"cross_sectional_momentum target_weights with params lookback={variant.lookback}, "
        f"top_k={variant.top_k}, and a CONFIG named '{variant.name}'. Then run, via the algua "
        f"CLI only: `uv run algua backtest run {variant.name} --demo --register` followed by "
        f"`uv run algua research promote {variant.name} --demo`. Do not edit any existing file. "
        f"Return the IterationResult matching the output schema."
    )


class CodexRunner:
    """Production AgentRunner: runs one iteration via `codex exec`, sandboxed to EXAMPLES_DIR."""

    def __init__(self, repo: Path, examples_dir: str = EXAMPLES_DIR) -> None:
        self.repo = repo
        self.examples_dir = examples_dir

    def run(self, iteration: int, variant: Variant) -> IterationResult:
        with tempfile.TemporaryDirectory() as td:
            schema_path = Path(td) / "schema.json"
            out_path = Path(td) / "out.json"
            schema_path.write_text(json.dumps(IterationResult.model_json_schema()))
            cmd = [
                "codex", "exec", "--json",
                "--output-schema", str(schema_path),
                "-o", str(out_path),
                "--sandbox", "workspace-write",
                "--add-dir", self.examples_dir,
                "--ask-for-approval", "never",
                "-C", str(self.repo),
                _build_prompt(iteration, variant),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0 or not out_path.exists():
                return IterationResult(
                    iteration=iteration, strategy_name=variant.name,
                    stage_before=None, stage_after=None,
                    gate_passed=False, promoted=False, config_hash=None,
                    status="failed", reason="codex exec failed",
                    error=(proc.stderr or "no output")[:500],
                )
            return IterationResult.model_validate_json(out_path.read_text())


class GitWorkspace:
    """Detects working-tree changes per turn via `git status --porcelain` snapshot/diff."""

    def __init__(self, repo: Path) -> None:
        self.repo = repo
        self._baseline: set[tuple[str, str]] = set()

    def _status(self) -> set[tuple[str, str]]:
        proc = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=self.repo, capture_output=True, text=True, check=True,
        )
        out: set[tuple[str, str]] = set()
        for line in proc.stdout.splitlines():
            if not line.strip():
                continue
            code, path = line[:2], line[3:]
            kind = _PORCELAIN_KIND.get(code.strip()[:1] or "?", "modified")
            out.add((path, kind))
        return out

    def begin_turn(self) -> None:
        self._baseline = self._status()

    def turn_changes(self) -> list[Change]:
        new = self._status() - self._baseline
        return [Change(path=p, kind=k) for p, k in sorted(new)]


def registered_names_via_cli(repo: str) -> list[str]:
    """Read registered strategy names through the CLI (golden rule: no module imports)."""
    proc = subprocess.run(
        ["uv", "run", "algua", "registry", "list"],
        cwd=repo, capture_output=True, text=True, check=True,
    )
    return [row["name"] for row in json.loads(proc.stdout)]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_operator_adapters.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Verify `registry list` JSON shape matches the adapter**

Run: `uv run algua registry list`
Expected: a JSON array of objects each containing a `"name"` field (may be `[]`). If the shape differs, fix `registered_names_via_cli` to match before committing.

- [ ] **Step 6: Commit**

```bash
git add algua/operator/adapters.py tests/test_operator_adapters.py
git commit -m "feat(operator): codex/git/cli I/O adapters"
```

---

### Task 6: CLI command + end-to-end loop (`operator_cmd.py`)

**Files:**
- Create: `algua/cli/operator_cmd.py`
- Modify: `algua/cli/main.py` (add `operator_cmd` to the subcommand imports)
- Test: `tests/test_cli_operator.py`

- [ ] **Step 1: Write the failing e2e test**

Create `tests/test_cli_operator.py`:

```python
import importlib
import json
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from algua.cli.main import app
from algua.operator.adapters import GitWorkspace
from algua.operator.loop import run_loop
from algua.operator.result import IterationResult
from algua.operator.variants import variant_list

runner = CliRunner()
EXAMPLES = Path("algua/strategies/examples")
PREFIX = "algua/strategies/examples/"


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "op.db"))
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path))


@pytest.fixture
def _cleanup_authored():
    """Remove any files the loop authored under examples/ and purge their modules."""
    before = {p.name for p in EXAMPLES.glob("*.py")}
    intruder = EXAMPLES.parent.parent.parent / "INTRUDER.py"  # repo-root sentinel for negatives
    yield
    for p in list(EXAMPLES.glob("*.py")):
        if p.name not in before:
            sys.modules.pop(f"algua.strategies.examples.{p.stem}", None)
            p.unlink()
    if intruder.exists():
        intruder.unlink()
    importlib.invalidate_caches()


def _variant_source(name: str, lookback: int, top_k: int) -> str:
    return (
        'GENERATED_BY = "agent"\n'
        "# Generated by the autonomous research loop (walking skeleton).\n"
        "from __future__ import annotations\n\n"
        "from algua.contracts.types import ExecutionContract\n"
        "from algua.strategies.base import StrategyConfig\n"
        "from algua.strategies.examples.cross_sectional_momentum import target_weights\n\n"
        "CONFIG = StrategyConfig(\n"
        f'    name="{name}",\n'
        '    universe=["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"],\n'
        '    execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1),\n'
        f'    params={{"lookback": {lookback}, "top_k": {top_k}}},\n'
        ")\n"
    )


class FakeRunner:
    """Deterministic AgentRunner double: authors the variant file and drives the CLI."""

    def __init__(self, *, write_outside: bool = False):
        self.write_outside = write_outside

    def run(self, iteration, variant):
        if self.write_outside:
            (EXAMPLES.parent.parent.parent / "INTRUDER.py").write_text("# oops\n")
            return IterationResult(
                iteration=iteration, strategy_name=variant.name, stage_before=None,
                stage_after=None, gate_passed=False, promoted=False, config_hash=None,
                status="ok", reason="authored outside", error=None,
            )
        (EXAMPLES / f"{variant.name}.py").write_text(
            _variant_source(variant.name, variant.lookback, variant.top_k)
        )
        importlib.invalidate_caches()
        runner.invoke(app, ["backtest", "run", variant.name, "--demo", "--register",
                            "--start", "2022-01-01", "--end", "2023-12-31"])
        promo = runner.invoke(app, ["research", "promote", variant.name, "--demo",
                                    "--start", "2022-01-01", "--end", "2023-12-31",
                                    "--min-holdout-sharpe", "-100", "--min-holdout-return", "-100",
                                    "--min-pct-positive", "0", "--min-window-sharpe", "-100"])
        payload = json.loads(promo.stdout)
        show = json.loads(runner.invoke(app, ["registry", "show", variant.name]).stdout)
        return IterationResult(
            iteration=iteration, strategy_name=variant.name, stage_before="idea",
            stage_after=show["stage"], gate_passed=payload["passed"],
            promoted=payload["promoted"], config_hash=payload.get("config_hash"),
            status="ok", reason="fake", error=None,
        )


def _names():
    out = runner.invoke(app, ["registry", "list"])
    return [row["name"] for row in json.loads(out.stdout)]


def test_loop_authors_two_strategies_and_shortlists_them(_cleanup_authored):
    summary = run_loop(
        runner=FakeRunner(), workspace=GitWorkspace(repo=Path.cwd()),
        registered_names=_names, all_variants=variant_list(),
        max_iterations=2, allowed_prefix=PREFIX,
    )
    assert summary.stopped_reason == "max_iterations"
    assert summary.n_shortlisted == 2
    authored = {p.stem for p in EXAMPLES.glob("momentum_lb*.py")}
    assert {"momentum_lb20", "momentum_lb40"} <= authored
    assert _names().count("momentum_lb20") == 1  # registered exactly once


def test_loop_fails_closed_on_write_outside_examples(_cleanup_authored):
    summary = run_loop(
        runner=FakeRunner(write_outside=True), workspace=GitWorkspace(repo=Path.cwd()),
        registered_names=_names, all_variants=variant_list(),
        max_iterations=3, allowed_prefix=PREFIX,
    )
    assert summary.stopped_reason == "gate_violation"
    assert summary.n_shortlisted == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli_operator.py -q`
Expected: FAIL (the `operator` CLI command does not exist yet; and these tests exercise `run_loop` directly, which is fine — they should pass once imports resolve. Run now to confirm the *current* failure is only the missing `operator_cmd` import wiring in Step 4, not a logic error.)

Note: Steps 1–2 test the loop wiring directly. The CLI command is added in Step 3 and covered in Step 5.

- [ ] **Step 3: Implement the CLI command**

Create `algua/cli/operator_cmd.py`:

```python
from __future__ import annotations

from pathlib import Path

import typer

from algua.cli.app import app, emit
from algua.cli.errors import json_errors
from algua.operator.adapters import EXAMPLES_DIR, CodexRunner, GitWorkspace, registered_names_via_cli
from algua.operator.loop import run_loop
from algua.operator.variants import variant_list

operator_app = typer.Typer(help="Autonomous research loop", no_args_is_help=True)
app.add_typer(operator_app, name="operator")


@operator_app.command("run")
@json_errors(ValueError)
def run(
    max_iterations: int = typer.Option(4, "--max-iterations", help="hard iteration budget"),
) -> None:
    """Run the bounded research loop and emit the run summary as JSON."""
    if max_iterations < 1:
        raise ValueError("--max-iterations must be >= 1")
    repo = Path.cwd()
    summary = run_loop(
        runner=CodexRunner(repo=repo),
        workspace=GitWorkspace(repo=repo),
        registered_names=lambda: registered_names_via_cli(str(repo)),
        all_variants=variant_list(),
        max_iterations=max_iterations,
        allowed_prefix=f"{EXAMPLES_DIR}/",
    )
    emit(summary.model_dump())
```

- [ ] **Step 4: Wire the command into the CLI**

Modify `algua/cli/main.py` — add `operator_cmd` to the registration import block:

```python
from algua.cli import (  # noqa: F401 - imports register subcommands
    backtest_cmd,
    data_cmd,
    operator_cmd,
    registry_cmd,
    research_cmd,
    strategy_cmd,
)
```

- [ ] **Step 5: Verify the command registers and validates input**

Run: `uv run algua operator run --max-iterations 0`
Expected: `{"ok": false, "error": "--max-iterations must be >= 1"}` and exit code 1.

- [ ] **Step 6: Run the e2e tests to verify they pass**

Run: `uv run pytest tests/test_cli_operator.py -q`
Expected: PASS (2 passed). If a `momentum_lb*.py` file lingers after a failure, delete it manually before re-running.

- [ ] **Step 7: Commit**

```bash
git add algua/cli/operator_cmd.py algua/cli/main.py tests/test_cli_operator.py
git commit -m "feat(operator): `algua operator run` CLI + end-to-end loop e2e"
```

---

### Task 7: Import boundary contract + full quality gate

**Files:**
- Modify: `pyproject.toml` (add an import-linter contract)

- [ ] **Step 1: Add the import-linter contract**

Append to `pyproject.toml` after the existing contracts:

```toml
[[tool.importlinter.contracts]]
name = "operator drives algua only through the CLI (no module imports)"
type = "forbidden"
source_modules = ["algua.operator"]
forbidden_modules = [
    "algua.cli",
    "algua.registry",
    "algua.backtest",
    "algua.research",
    "algua.data",
]
```

- [ ] **Step 2: Verify the contract holds**

Run: `uv run lint-imports`
Expected: `Contracts: 7 kept, 0 broken.` (If `algua.operator` broke it, an adapter is importing an algua module instead of shelling the CLI — fix the adapter.)

- [ ] **Step 3: Run the full quality gate**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: all pass (pytest green, ruff "All checks passed!", mypy "Success", import-linter "7 kept, 0 broken"). Fix any ruff import-sorting or mypy typing nits inline.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "test(operator): enforce CLI-only boundary for algua.operator"
```

---

## Self-review notes

- **Spec coverage:** variants (§2, Task 2); `IterationResult`/schema (§6, Task 1); additions-only diff-gate + dedup (§5/§4, Task 3); controller with stop conditions + circuit breaker + fail-closed gate (§4/§5, Task 4); `CodexRunner` + `GitWorkspace` + CLI names reader (§3/§4, Task 5); CLI command + positive/negative e2e with `FakeRunner` (§3/§7, Task 6); CLI-only import boundary + full gate (§3/§7, Task 7). Live-wall unreachability (§5) holds because the loop only issues `backtest run --register` + `research promote` (tops out at `shortlisted`).
- **Refinement vs spec:** the `IterationResult` JSON schema is **generated at runtime** by `CodexRunner` from `model_json_schema()` rather than committed to `docs/contracts/`. This removes a drift risk and a static artifact; the pydantic model is the single source of truth. Functionally identical for `--output-schema`.
- **Type consistency:** `IterationResult`, `RunSummary`, `Variant`, `Change` signatures are identical across all tasks; `run_loop` keyword args match between `loop.py`, the unit tests, and `operator_cmd.py`/e2e.
