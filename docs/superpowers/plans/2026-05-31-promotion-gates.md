# Promotion Gates Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `algua research promote <strategy>` — run a walk-forward, evaluate it against quantitative gate criteria, and advance the registry `backtested→shortlisted` only on pass (recording the evidence).

**Architecture:** A pure `algua/research/gates.py` (`GateCriteria`, `GateDecision`, `evaluate_gate`) judges a `WalkForwardResult` on holdout + stability thresholds; the holdout (never used in selection) is the search-breadth defense and `n_combos` is recorded. A new `research` CLI sub-app orchestrates run→evaluate→(transition on pass). Shortlisting is a research checkpoint, not a safety wall (paper→live stays the hard wall).

**Tech Stack:** Python 3.12, Typer, pytest. Builds on `walkforward.py`, the registry, and the lifecycle state machine.

**Key existing code:**
- `algua/backtest/walkforward.py`: `walk_forward(strategy, provider, start, end, *, windows=4, holdout_frac=0.2) -> WalkForwardResult`; `WalkForwardResult` has `.config_hash`, `.snapshot_id`, `.holdout_metrics` (dict incl. numeric `sharpe`/`total_return`), `.stability` (dict: `mean_sharpe`/`std_sharpe`/`min_sharpe`/`pct_positive_windows`).
- `algua/cli/backtest_cmd.py`: helpers `_select_provider(demo, snapshot)`, `_utc(date_str)`; pattern `@json_errors(ValueError, LookupError, BacktestError)`; `emit`.
- `algua/backtest/engine.py::BacktestError`.
- `algua/registry/db.py` (`connect`, `migrate`), `algua/registry/store.py` (`transition`), `algua/contracts/lifecycle.py` (`Stage`, `Actor`). Lifecycle allows `backtested → shortlisted`; `idea → shortlisted` is illegal.
- `algua/strategies/loader.py::load_strategy`; `algua/cli/main.py` registers sub-apps via side-effect imports.

---

### Task 1: `algua/research/gates.py` — criteria, decision, evaluate

**Files:**
- Create: `algua/research/__init__.py` (empty), `algua/research/gates.py`
- Test: `tests/test_research_gates.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_research_gates.py
from algua.backtest.walkforward import WalkForwardResult
from algua.research.gates import GateCriteria, GateDecision, evaluate_gate


def _wf(holdout_sharpe=0.8, holdout_return=0.05, pct_positive=0.75, min_sharpe=0.1):
    return WalkForwardResult(
        strategy="ew", config_hash="abc", data_source="SyntheticProvider", snapshot_id=None,
        timeframe="1d", seed=0, period={"start": "2022-01-01", "end": "2023-12-31"},
        windows=4, holdout_frac=0.2, window_metrics=[],
        holdout_metrics={"start": "2023-06-01", "end": "2023-12-31", "n_bars": 100,
                         "total_return": holdout_return, "ann_return": 0.1, "ann_volatility": 0.12,
                         "sharpe": holdout_sharpe, "max_drawdown": -0.07},
        stability={"mean_sharpe": 1.0, "std_sharpe": 0.3, "min_sharpe": min_sharpe,
                   "pct_positive_windows": pct_positive},
    )


def test_all_thresholds_met_passes():
    d = evaluate_gate(_wf(), GateCriteria(), n_combos=9)
    assert isinstance(d, GateDecision)
    assert d.passed is True
    assert {c["name"] for c in d.checks} == {
        "holdout_sharpe", "holdout_return", "pct_positive_windows", "min_window_sharpe"}
    assert all(c["passed"] for c in d.checks)
    assert d.n_combos == 9


def test_low_holdout_sharpe_fails_that_check():
    d = evaluate_gate(_wf(holdout_sharpe=0.1), GateCriteria(min_holdout_sharpe=0.5))
    assert d.passed is False
    failed = [c["name"] for c in d.checks if not c["passed"]]
    assert failed == ["holdout_sharpe"]


def test_zero_holdout_return_fails_strict_gt():
    d = evaluate_gate(_wf(holdout_return=0.0), GateCriteria())
    assert d.passed is False
    assert [c["name"] for c in d.checks if not c["passed"]] == ["holdout_return"]


def test_low_pct_positive_and_negative_window_fail():
    d = evaluate_gate(_wf(pct_positive=0.4, min_sharpe=-0.5), GateCriteria())
    assert d.passed is False
    failed = {c["name"] for c in d.checks if not c["passed"]}
    assert failed == {"pct_positive_windows", "min_window_sharpe"}


def test_to_dict_serializable():
    import json
    json.dumps(evaluate_gate(_wf(), GateCriteria()).to_dict())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_research_gates.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'algua.research.gates'`.

- [ ] **Step 3: Write the implementation**

```python
# algua/research/gates.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from algua.backtest.walkforward import WalkForwardResult


@dataclass
class GateCriteria:
    """Thresholds for promoting backtested -> shortlisted. Holdout checks are the search-breadth
    defense (the holdout was never used during selection)."""

    min_holdout_sharpe: float = 0.5
    min_holdout_return: float = 0.0       # strict > 0
    min_pct_positive_windows: float = 0.6
    min_window_sharpe: float = 0.0        # the worst window's Sharpe must be >= this


@dataclass
class GateDecision:
    passed: bool
    checks: list[dict[str, Any]]
    n_combos: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "checks": self.checks, "n_combos": self.n_combos}


def _check(name: str, value: float, threshold: float, op: str) -> dict[str, Any]:
    if op == ">=":
        ok = value >= threshold
    elif op == ">":
        ok = value > threshold
    else:  # pragma: no cover - guarded by the fixed call sites below
        raise ValueError(f"unknown op {op!r}")
    return {"name": name, "value": float(value), "threshold": float(threshold),
            "op": op, "passed": bool(ok)}


def evaluate_gate(
    wf: WalkForwardResult, criteria: GateCriteria, *, n_combos: int | None = None
) -> GateDecision:
    """Judge a walk-forward result against the gate criteria. Pure; no side effects."""
    h = wf.holdout_metrics
    s = wf.stability
    checks = [
        _check("holdout_sharpe", h["sharpe"], criteria.min_holdout_sharpe, ">="),
        _check("holdout_return", h["total_return"], criteria.min_holdout_return, ">"),
        _check("pct_positive_windows", s["pct_positive_windows"],
               criteria.min_pct_positive_windows, ">="),
        _check("min_window_sharpe", s["min_sharpe"], criteria.min_window_sharpe, ">="),
    ]
    return GateDecision(passed=all(c["passed"] for c in checks), checks=checks, n_combos=n_combos)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_research_gates.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add algua/research/__init__.py algua/research/gates.py tests/test_research_gates.py
git commit -m "feat: add promotion-gate criteria and evaluator"
```

---

### Task 2: CLI `research promote` + wiring + import boundary

**Files:**
- Create: `algua/cli/research_cmd.py`
- Modify: `algua/cli/main.py` (register the `research` sub-app)
- Modify: `pyproject.toml` (import-linter contract)
- Test: `tests/test_cli_research.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli_research.py
import json
import pytest
from typer.testing import CliRunner
from algua.cli.main import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def _tmp(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path))


def _stage(name="cross_sectional_momentum"):
    show = runner.invoke(app, ["registry", "show", name])
    return json.loads(show.stdout)["stage"]


def _backtest_to_backtested():
    # idea -> backtested via the registering backtest path
    return runner.invoke(app, ["backtest", "run", "cross_sectional_momentum", "--demo",
                               "--start", "2022-01-01", "--end", "2023-12-31", "--register"])


def test_promote_passes_and_shortlists():
    assert _backtest_to_backtested().exit_code == 0
    result = runner.invoke(app, ["research", "promote", "cross_sectional_momentum", "--demo",
                                 "--start", "2022-01-01", "--end", "2023-12-31",
                                 "--min-holdout-sharpe", "-100", "--min-holdout-return", "-100",
                                 "--min-pct-positive", "0", "--min-window-sharpe", "-100",
                                 "--n-combos", "9"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["passed"] is True
    assert payload["promoted"] is True
    assert payload["n_combos"] == 9
    assert _stage() == "shortlisted"


def test_promote_fails_does_not_transition():
    assert _backtest_to_backtested().exit_code == 0
    result = runner.invoke(app, ["research", "promote", "cross_sectional_momentum", "--demo",
                                 "--start", "2022-01-01", "--end", "2023-12-31",
                                 "--min-holdout-sharpe", "999"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["passed"] is False
    assert payload["promoted"] is False
    assert _stage() == "backtested"   # unchanged


def test_promote_from_idea_is_json_error():
    # registered but never backtested -> still 'idea'; backtested->shortlisted precondition unmet
    runner.invoke(app, ["registry", "add", "cross_sectional_momentum"])
    result = runner.invoke(app, ["research", "promote", "cross_sectional_momentum", "--demo",
                                 "--start", "2022-01-01", "--end", "2023-12-31",
                                 "--min-holdout-sharpe", "-100", "--min-holdout-return", "-100",
                                 "--min-pct-positive", "0", "--min-window-sharpe", "-100"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["ok"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli_research.py -v`
Expected: FAIL — no `research` command group.

- [ ] **Step 3: Write `algua/cli/research_cmd.py`**

```python
# algua/cli/research_cmd.py
from __future__ import annotations

from contextlib import closing
from typing import Any

import typer

from algua.backtest.engine import BacktestError
from algua.backtest.walkforward import walk_forward
from algua.cli.app import app, emit
from algua.cli.backtest_cmd import _select_provider, _utc
from algua.cli.errors import json_errors
from algua.config.settings import get_settings
from algua.contracts.lifecycle import Actor, Stage
from algua.registry import store
from algua.registry.db import connect, migrate
from algua.research.gates import GateCriteria, GateDecision, evaluate_gate
from algua.strategies.loader import load_strategy

research_app = typer.Typer(help="Research workflow: gates and promotion", no_args_is_help=True)
app.add_typer(research_app, name="research")


def _gate_reason(decision: GateDecision) -> str:
    parts = [f"{c['name']}={c['value']:.4g}{c['op']}{c['threshold']:.4g}" for c in decision.checks]
    extra = f"; n_combos={decision.n_combos}" if decision.n_combos is not None else ""
    return "gate pass: " + ", ".join(parts) + extra


@research_app.command("promote")
@json_errors(ValueError, LookupError, BacktestError)
def promote(
    name: str,
    start: str = typer.Option("2023-01-01", "--start"),
    end: str = typer.Option("2023-12-31", "--end"),
    demo: bool = typer.Option(False, "--demo", help="use the synthetic data provider"),
    snapshot: str = typer.Option(None, "--snapshot", help="backtest an ingested bars snapshot id"),
    windows: int = typer.Option(4, "--windows", help="walk-forward windows"),
    holdout_frac: float = typer.Option(0.2, "--holdout-frac", help="fraction reserved as holdout"),
    min_holdout_sharpe: float = typer.Option(0.5, "--min-holdout-sharpe"),
    min_holdout_return: float = typer.Option(0.0, "--min-holdout-return"),
    min_pct_positive: float = typer.Option(0.6, "--min-pct-positive"),
    min_window_sharpe: float = typer.Option(0.0, "--min-window-sharpe"),
    n_combos: int = typer.Option(None, "--n-combos", help="combos searched (recorded as evidence)"),
    actor: str = typer.Option("agent", "--actor", help="human | agent | system"),
) -> None:
    """Gate backtested->shortlisted on walk-forward holdout + stability; promote only on pass."""
    strategy = load_strategy(name)
    provider = _select_provider(demo, snapshot)
    wf = walk_forward(strategy, provider, _utc(start), _utc(end),
                      windows=windows, holdout_frac=holdout_frac)
    criteria = GateCriteria(
        min_holdout_sharpe=min_holdout_sharpe, min_holdout_return=min_holdout_return,
        min_pct_positive_windows=min_pct_positive, min_window_sharpe=min_window_sharpe,
    )
    decision = evaluate_gate(wf, criteria, n_combos=n_combos)

    promoted = False
    if decision.passed:
        with closing(connect(get_settings().db_path)) as conn:
            migrate(conn)
            store.transition(conn, name, Stage.SHORTLISTED, Actor(actor), _gate_reason(decision),
                             code_hash=wf.config_hash, config_hash=wf.config_hash)
        promoted = True

    payload: dict[str, Any] = {
        **decision.to_dict(),
        "strategy": name,
        "promoted": promoted,
        "config_hash": wf.config_hash,
        "snapshot_id": wf.snapshot_id,
        "holdout": wf.holdout_metrics,
        "stability": wf.stability,
    }
    emit(payload)
```

- [ ] **Step 4: Register the sub-app in `algua/cli/main.py`**

Add `research_cmd` to the side-effect import line (keep it sorted with the others), e.g.:
```python
from algua.cli import backtest_cmd, data_cmd, registry_cmd, research_cmd, strategy_cmd  # noqa: F401
```
(Match the existing exact import statement in `main.py`; just add `research_cmd` in alphabetical position.)

- [ ] **Step 5: Add the import-linter contract to `pyproject.toml`**

Append under `[tool.importlinter]`:
```toml
[[tool.importlinter.contracts]]
name = "backtest engine stays off the research layer"
type = "forbidden"
source_modules = ["algua.backtest"]
forbidden_modules = ["algua.research"]
```

- [ ] **Step 6: Run tests + full gate**

Run:
```bash
uv run pytest tests/test_cli_research.py -q
uv run pytest -q
uv run ruff check .
uv run mypy algua
uv run lint-imports
```
Expected: all pass; ruff/mypy clean; import-linter **6 kept, 0 broken** (`research`/`research_cmd` import `backtest` result types; `backtest` imports neither `research` nor `cli`). If ruff flags the `n_combos`/`snapshot` `typer.Option(None)` defaults, mirror the existing `# noqa: B008` pattern used elsewhere in the CLI.

- [ ] **Step 7: Commit**

```bash
git add algua/cli/research_cmd.py algua/cli/main.py pyproject.toml tests/test_cli_research.py
git commit -m "feat: add 'research promote' gate command; enforce backtest-off-research boundary"
```

---

### Task 3: Full verification & end-to-end smoke

**Files:** none (verification only)

- [ ] **Step 1: Full quality gate**

Run:
```bash
uv run pytest -q
uv run ruff check .
uv run mypy algua
uv run lint-imports
```
Expected: all pass; ruff clean; mypy `Success`; import-linter `6 kept, 0 broken`.

- [ ] **Step 2: End-to-end smoke (synthetic): idea → backtested → gated shortlist**

Run:
```bash
export ALGUA_DB_PATH="$(mktemp -d)/r.db"
uv run algua registry add cross_sectional_momentum >/dev/null
uv run algua backtest run cross_sectional_momentum --demo --start 2021-01-01 --end 2023-12-31 --register \
  | python3 -c "import sys,json; print('after backtest:', json.load(sys.stdin)['stage'])"
uv run algua research promote cross_sectional_momentum --demo --start 2021-01-01 --end 2023-12-31 \
  --min-holdout-sharpe -100 --min-holdout-return -100 --min-pct-positive 0 --min-window-sharpe -100 --n-combos 9 \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print('passed:', d['passed'], '| promoted:', d['promoted'])"
uv run algua registry show cross_sectional_momentum \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print('stage:', d['stage'], '| last reason:', d['transitions'][-1]['reason'])"
```
Expected: `after backtest: backtested`; `passed: True | promoted: True`; `stage: shortlisted` with a `gate pass: ...` reason recording the thresholds + `n_combos=9`.

(The lenient thresholds force a pass for the smoke; real promotion uses the defaults. `backtest run --register` advances `idea→backtested` first because `research promote` gates only the `backtested→shortlisted` step.)

- [ ] **Step 3: Final commit (if any verification fixes were needed)**

```bash
git add -A
git commit -m "test: verify promotion gate end to end" --allow-empty
```

---

## Self-Review Notes

- **Spec coverage:** `GateCriteria`/`GateDecision`/`evaluate_gate` with holdout+stability checks and
  recorded `n_combos` (Task 1); `research promote` CLI orchestration (run→evaluate→transition on
  pass, JSON decision, exit-0-on-evaluate) + main wiring + import contract (Task 2); verification +
  e2e (Task 3). Out-of-scope (explicit breadth penalty, registry hard-enforcement, multi-strategy
  ranking, the agent loop) intentionally absent.
- **Boundary:** gate logic is pure (`research/gates.py` imports only the WF result type + stdlib);
  the new import contract forbids `algua.backtest` from importing `algua.research`; the CLI wires
  research+walkforward+registry. lint-imports goes 5 → 6.
- **Type consistency:** `GateCriteria(min_holdout_sharpe, min_holdout_return,
  min_pct_positive_windows, min_window_sharpe)`, `GateDecision(passed, checks, n_combos).to_dict()`,
  `evaluate_gate(wf, criteria, *, n_combos=None)`, and the CLI's `promoted`/`passed` output keys are
  used consistently across tasks/tests. The gate reads `wf.holdout_metrics["sharpe"|"total_return"]`
  and `wf.stability["pct_positive_windows"|"min_sharpe"]` — names that match the walk-forward slice.
