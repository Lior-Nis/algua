# Stage 5a — Broker-Neutral Error Leaf + Experiment-Tracker Seam Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The two zero-production-behavior-change seams of Stage 5. (1) Extract `BrokerError` into a broker-neutral leaf `algua/execution/errors.py` so `execution/tick_clock.py` and `cli/errors.py` stop reaching into the Alpaca adapter for it — closing a debt the code documents in its own comment. (2) Wire the existing-but-dead `ExperimentTracker` Protocol behind `algua/tracking/factory.py`, so `cli/backtest_cmd.py` depends on the Protocol rather than importing three concrete `log_*` functions — closing the PR#110 tracker-DI deferral.

**Architecture:** Both follow the spec's stated seam pattern — *small Protocol + name→factory registry + config field* — and both mirror shapes this codebase already proves: `algua/data/providers/errors.py` is the precedent for the error leaf, and `algua/data/providers/__init__.py`'s `_REGISTRY`/`register_provider`/`get_provider` is the precedent for the tracker registry.

**Tech Stack:** Python 3.12, uv, pytest, ruff, mypy, import-linter.

**Spec:** `docs/superpowers/specs/2026-08-18-system-simplification-design.md` §5 items 1 (the error-leaf clause) and 3 (experiment tracker).

**Ground truth this plan is written from:** a research pass against `main`@`8d7abc2` that read every seam in Stage 5 and **corrected most of the spec's counts** (the spec's "~5 direct `log_*` call sites" is actually **3**; its "both self-documented as debt" is half right — `tick_clock.py` carries the comment, `cli/errors.py` does not). Its findings are folded in below.

**Stage 5 is split into three slices; this is the first.** The research measured the stage and found its risk is concentrated and heterogeneous: the broker factory alone carries ~120 of 172 test wiring-pins plus the paper/live safety boundary, while the calendar and knowledge-sink seams each require a behavior-change ruling. One plan would either over-ceremony the free parts or under-scrutinize the broker. Slices:

- **5a (this plan)** — error leaf + tracker. Zero production behavior change; mechanically provable.
- **5b** — broker factory alone.
- **5c** — calendar + knowledge sink (both carry deliberate, ruled behavior changes).

Ordering is forced at one point: **5a's error leaf must land before 5b**, because the broker factory should not be built while `tick_clock`/`cli/errors` still reach into `alpaca_broker`.

### Decisions the plan's author made (recorded for a reviewer to check, not to re-derive)

1. **No re-export shim.** Every importer of `BrokerError` is re-pointed at the new leaf and `alpaca_broker.py` does **not** re-export it. Leaving a re-export would create exactly the dual import path this program exists to remove; there are only 6 importers (4 production, 2 test), so re-pointing is cheap and total.
2. **The no-op tracker gets its own explicit JSON state.** `_record_tracking`'s docstring pins a **three-state contract** — not-requested (no keys) / succeeded (`mlflow_run_id` set) / failed (`mlflow_run_id` null + `mlflow_tracking_error`). A no-op returning a fake run id would fabricate "succeeded"; returning `None` silently would create an ambiguous fourth state indistinguishable from a failure that forgot its error key. So the no-op records `mlflow_run_id: null` **plus a distinct `mlflow_tracking_skipped` key**. This key can only ever appear when a non-default backend is selected, so all three existing states and every existing test are untouched.
3. **`tracking_uri` stays a per-call keyword.** The Protocol's three methods each take `*, tracking_uri: str`. Binding it at construction instead would be tidier, but it changes the Protocol surface *and* all three call sites for no behavior gain, and this slice's whole value is being provably inert. Left as-is; revisit only if a real second backend wants it.

## Global Constraints

- Quality gate on EVERY task before commit: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`. All four must pass.
- **Zero production behavior change.** The default path must be byte-for-byte equivalent in observable behavior: `BrokerError` keeps its identity and `except` semantics; with the default MLflow backend, `--track` produces exactly the JSON it does today. The new `mlflow_tracking_skipped` key appears only under an explicitly-selected non-default backend.
- Test count must not fall. It may rise (new tests for the new seams are expected and welcome); it must never drop, since this slice deletes no behavior.
- **`algua/cli/paper_cmd.py` is CODEOWNERS-protected** and is one of the files re-pointed in Task 1. That is unavoidable (it imports `BrokerError`) and expected — note it in the PR so the human review is anticipated rather than a surprise. This is the first program stage where most slices touch review-gated files.
- **Run the full test suite in the FOREGROUND** with the Bash tool's `timeout: 600000` (`uv run pytest -q 2>&1 | tail -20`). Do NOT background it. Seven implementer subagents across Stages 4a-4c stalled by backgrounding a long command and then ending their turn awaiting a notification that never arrives; running it in the foreground removes the failure mode rather than warning about it, and worked first try in Stage 4c.
- If any step involves mutating a file to prove a test bites, clear `__pycache__` (or set `PYTHONDONTWRITEBYTECODE=1`) around it. Statement-level edits can leave file size unchanged, and `.pyc` mtime granularity is one second, so a same-second restore can silently reuse mutated bytecode — this misled a reviewer in Stage 4c in both directions.
- `git add` scoped to named files — never `git add -A`.
- Commits end with: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- Known worktree hazard: some test writes a demo strategy file into `algua/strategies/momentum/`. If `git status` shows an untracked file there after a run, delete it before staging.
- **Import-linter:** no contract change expected. The binding contract for Stage 5 is `pyproject.toml`'s *"contracts layer is pure (imports no other algua module)"* — this slice adds nothing to `algua/contracts`, so it is unaffected. Confirm `lint-imports` still reports 23 kept / 0 broken.

---

### Task 1: Extract the broker-neutral error leaf

**Files:**
- Create: `algua/execution/errors.py`
- Modify: `algua/execution/alpaca_broker.py`, `algua/execution/tick_clock.py`, `algua/cli/errors.py`, `algua/cli/live_cmd.py`, `algua/cli/paper_cmd.py`, `tests/test_cli_live.py`, `tests/test_alpaca_broker.py`

**Interfaces:**
- Produces: `algua.execution.errors.BrokerError` — the same class, same base (`RuntimeError`), same docstring, importable from a module that depends on nothing.
- Consumes: nothing.

**Why this is first:** it is the lowest-risk, highest-clarity change in Stage 5 — one 4-line exception class moves to a new leaf and six importers are re-pointed — and 5b's broker factory depends on it.

- [ ] **Step 1: Confirm the current shape**

Read `algua/execution/alpaca_broker.py` around the `BrokerError` definition (currently ~line 61) and `algua/data/providers/errors.py` in full — the latter is the precedent this mirrors (a bare exception subclass in a dependency-free module, imported by `cli/errors.py` exactly as this one will be).

Then enumerate every importer yourself rather than trusting this plan's list:

```bash
grep -rn "BrokerError" algua/ tests/ --include='*.py'
```
As of writing: 4 production importers (`execution/tick_clock.py`, `cli/errors.py`, `cli/live_cmd.py`, `cli/paper_cmd.py`) and 2 test importers (`tests/test_cli_live.py`, `tests/test_alpaca_broker.py`). Some hits are prose mentions in comments/docstrings (`execution/flatten.py`, `primitives/retry.py`) — those reference the *concept* and need no change; do not edit them.

- [ ] **Step 2: Create `algua/execution/errors.py`**

Move the class verbatim — same name, same base, same docstring text:

```python
"""Broker-neutral error leaf.

``BrokerError`` is the domain error for *any* broker adapter, not just Alpaca. It lives here rather
than in ``alpaca_broker`` so broker-agnostic consumers (``execution/tick_clock``, ``cli/errors``)
can catch it without importing a concrete adapter. Mirrors ``algua/data/providers/errors.py``,
which plays the same role for the data-provider seam.

Imports nothing — keeping this a true leaf is what makes it safe for any layer to depend on.
"""

from __future__ import annotations


class BrokerError(RuntimeError):
    """Any failure talking to the Alpaca trading API — network error, non-2xx status,
    or a malformed/unexpected response. Callers (the CLI, the future loop) catch this so a
    broker hiccup never escapes as a raw traceback."""
```

Re-verify that docstring against the live source before committing to it; move the text exactly rather than retyping from this plan.

Note the docstring still says "the Alpaca trading API". That is accurate today (one adapter exists) and generalizing it is 5b's business, once a second construction path exists. Leave it; do not pre-emptively reword.

- [ ] **Step 3: Remove the definition from `alpaca_broker.py` and import it instead**

Delete the `class BrokerError(RuntimeError):` block from `algua/execution/alpaca_broker.py` and add `from algua.execution.errors import BrokerError` to its imports — the module raises `BrokerError` throughout, so it consumes the leaf like everyone else.

**Do not re-export it** (no `__all__` entry, no `BrokerError = BrokerError` alias). A re-export would leave two valid import paths for one class, which is the dual-path cruft this program removes; Step 4 re-points every importer instead.

- [ ] **Step 4: Re-point all six importers**

For each, change the import source from `algua.execution.alpaca_broker` to `algua.execution.errors`, leaving everything else untouched:

- `algua/execution/tick_clock.py` — and **delete the now-false debt comment** in `tick_clock`'s docstring, currently: *"Coupling note: imports ``BrokerError`` from ``alpaca_broker`` — not broker-agnostic today (only one broker exists; extracting a shared exceptions leaf is deferred — YAGNI)."* That deferral is exactly what this task closes, so the comment must go rather than linger as a false statement. Delete those two lines; keep the rest of the docstring verbatim.
- `algua/cli/errors.py` — a function-local import alongside ~9 other error types; change only the `BrokerError` source. Note this file already imports `ProviderError` from `algua/data/providers/errors.py`, so the new import sits naturally beside its precedent.
- `algua/cli/live_cmd.py` — imports `AlpacaLiveBroker, BrokerError` together; split so `BrokerError` comes from the leaf and `AlpacaLiveBroker` still comes from the adapter.
- `algua/cli/paper_cmd.py` — **CODEOWNERS-protected**; `BrokerError` is one name in a multi-name import block. Same treatment.
- `tests/test_cli_live.py`, `tests/test_alpaca_broker.py` — re-point too. In `test_alpaca_broker.py` the import is `AlpacaPaperBroker, BrokerError`; split it as above.

- [ ] **Step 5: Prove nothing still reaches into the adapter for the error**

```bash
grep -rn "from algua.execution.alpaca_broker import" algua/ tests/ --include='*.py' | grep BrokerError
```
Expected: **no hits**. Any hit is a missed importer.

Then confirm the class identity is genuinely unchanged (a moved exception that is not the same object would break every `except` clause elsewhere):

```bash
uv run python -c "
from algua.execution.errors import BrokerError as A
from algua.execution.alpaca_broker import AlpacaPaperBroker  # adapter still imports fine
import algua.execution.tick_clock as tc, algua.cli.errors as ce
print('is RuntimeError subclass:', issubclass(A, RuntimeError))
print('tick_clock sees same class:', tc.BrokerError is A)
print('OK')
"
```
Expected: both `True`, exit 0.

- [ ] **Step 6: Full quality gate**

Run in the foreground with `timeout: 600000`:
`uv run pytest -q 2>&1 | tail -20`
then `uv run ruff check . && uv run mypy algua && uv run lint-imports`.

Expected: all four pass, test count unchanged (this task adds no tests and removes none), `lint-imports` 23 kept / 0 broken. Check `git status` for the momentum-strategy hazard.

- [ ] **Step 7: Commit**

```bash
git add algua/execution/errors.py algua/execution/alpaca_broker.py algua/execution/tick_clock.py algua/cli/errors.py algua/cli/live_cmd.py algua/cli/paper_cmd.py tests/test_cli_live.py tests/test_alpaca_broker.py
```

```bash
git commit -m "$(cat <<'EOF'
refactor: extract the broker-neutral error leaf (stage 5a)

BrokerError lived in the Alpaca adapter, so broker-agnostic consumers had to import a concrete
adapter to catch it. execution/tick_clock.py said so in its own docstring ("extracting a shared
exceptions leaf is deferred -- YAGNI"); this closes that deferral, and the comment is deleted
rather than left as a false statement.

Moves the class verbatim to algua/execution/errors.py -- a true leaf importing nothing -- mirroring
algua/data/providers/errors.py, which already plays this role for the data-provider seam. All six
importers (4 production, 2 test) are re-pointed; alpaca_broker deliberately does NOT re-export it,
so exactly one import path exists rather than two.

Zero behavior change: same class, same RuntimeError base, same docstring, same identity (verified:
tick_clock resolves the same object), so every existing `except BrokerError` is unaffected.

Lands before the stage-5b broker factory, which should not be built while tick_clock and cli/errors
still reach into the adapter.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Wire the experiment-tracker seam

**Files:**
- Create: `algua/tracking/factory.py`
- Modify: `algua/tracking/mlflow_tracker.py`, `algua/config/settings.py`, `algua/cli/backtest_cmd.py`
- Test: `tests/test_tracking_backtest.py` (extend)

**Interfaces:**
- Produces: `algua.tracking.factory.get_tracker(name: str | None = None) -> ExperimentTracker` — returns the configured tracker; `register_tracker(name, factory)` for extension; the `MLFLOW` backend is the default and the `NOOP` backend is the second registered impl.
- Consumes: `algua.tracking.mlflow_tracker.ExperimentTracker` (the existing Protocol, unchanged).

**Why this closes a real deferral:** the `ExperimentTracker` Protocol has existed since #45 and is **genuinely dead** — the only references anywhere are three `hasattr` assertions in `tests/test_tracking_backtest.py`. No production code accepts it as a parameter; `cli/backtest_cmd.py` imports the three concrete `log_*` functions directly. This is the PR#110 tracker-DI deferral.

- [ ] **Step 1: Read the contract you must not break**

Read `algua/cli/backtest_cmd.py`'s `_record_tracking` (currently ~lines 59-72) in full. Its docstring pins a **three-state JSON contract**, and it is the thing most at risk in this task:

- **not-requested** — `--track` absent, so `_record_tracking` is never called and *no* tracking keys appear;
- **succeeded** — `mlflow_run_id` set, no error key;
- **failed** — `mlflow_run_id` null **plus** `mlflow_tracking_error`, warned to stderr, **never raised** (a tracker failure must not discard a completed evaluation, #341).

Read the three call sites (`~:229` `log_backtest`, `~:308` `log_walk_forward`, `~:417` `log_sweep`) and note all three are gated by `if track:` and all three pass `tracking_uri=get_settings().mlflow_tracking_uri`.

- [ ] **Step 2: Give the MLflow functions a Protocol-shaped implementation**

In `algua/tracking/mlflow_tracker.py`, below the existing `ExperimentTracker` Protocol and the existing `log_backtest`/`log_sweep`/`log_walk_forward` functions, add a thin class that satisfies the Protocol by delegating to them. Do **not** move or alter the functions themselves — they stay the implementation, and other callers/tests that import them directly keep working:

```python
class MlflowTracker:
    """The MLflow-backed :class:`ExperimentTracker`. A thin adapter over the module-level
    ``log_*`` functions, which remain the implementation — this class exists so callers can depend
    on the Protocol instead of on three concrete function imports."""

    def log_backtest(
        self, result: BacktestResult, params: dict[str, Any], *, tracking_uri: str
    ) -> str:
        return log_backtest(result, params, tracking_uri=tracking_uri)

    def log_sweep(self, result: SweepResult, *, tracking_uri: str) -> str:
        return log_sweep(result, tracking_uri=tracking_uri)

    def log_walk_forward(
        self, result: WalkForwardResult, params: dict[str, Any], *, tracking_uri: str
    ) -> str:
        return log_walk_forward(result, params, tracking_uri=tracking_uri)
```

Match the Protocol's signatures exactly (verify against the live Protocol — each method takes `*, tracking_uri: str`). mypy is the proof that this satisfies the Protocol; Step 5 asserts it explicitly.

- [ ] **Step 3: Add the no-op backend and the sentinel it returns**

Still in `algua/tracking/mlflow_tracker.py` (or a sibling module if you prefer — but keep it beside the Protocol so both impls are found together), add:

```python
#: Returned by :class:`NoopTracker` in place of a run id. ``_record_tracking`` translates this into
#: an explicit ``mlflow_tracking_skipped`` key rather than letting a null run id masquerade as a
#: failure — the JSON contract distinguishes "backend disabled" from "backend errored".
TRACKING_SKIPPED = "__tracking_skipped__"


class NoopTracker:
    """An :class:`ExperimentTracker` that logs nothing and never raises.

    Selected with ``ALGUA_TRACKING_BACKEND=noop`` — for environments without an MLflow store, and
    as the honest second implementation that proves the seam. It deliberately does NOT invent a run
    id: fabricating one would make the payload claim a run succeeded when nothing was logged.
    """

    def log_backtest(
        self, result: BacktestResult, params: dict[str, Any], *, tracking_uri: str
    ) -> str:
        return TRACKING_SKIPPED

    def log_sweep(self, result: SweepResult, *, tracking_uri: str) -> str:
        return TRACKING_SKIPPED

    def log_walk_forward(
        self, result: WalkForwardResult, params: dict[str, Any], *, tracking_uri: str
    ) -> str:
        return TRACKING_SKIPPED
```

- [ ] **Step 4: Create `algua/tracking/factory.py`**

Mirror `algua/data/providers/__init__.py`'s registry shape (read it first — `_REGISTRY`, `register_provider`, `get_provider`) so this seam looks like the one the spec cites as precedent:

```python
"""Name→factory registry for experiment trackers (spec §5 item 3).

Adding a backend is one module plus one ``register_tracker`` entry — the extension point the
``ExperimentTracker`` Protocol has had since #45 without ever being wired to anything.

Selection comes from ``settings.tracking_backend``; ``mlflow`` is the default, so the behaviour of
every existing ``--track`` run is unchanged.
"""

from __future__ import annotations

from collections.abc import Callable

from algua.config.settings import get_settings
from algua.tracking.mlflow_tracker import ExperimentTracker, MlflowTracker, NoopTracker

_REGISTRY: dict[str, Callable[[], ExperimentTracker]] = {
    "mlflow": MlflowTracker,
    "noop": NoopTracker,
}


def register_tracker(name: str, factory: Callable[[], ExperimentTracker]) -> None:
    """Register a tracker backend under ``name`` (last registration wins)."""
    _REGISTRY[name] = factory


def get_tracker(name: str | None = None) -> ExperimentTracker:
    """The configured tracker. ``name`` overrides ``settings.tracking_backend`` (for tests).

    Unknown names fail closed with the valid set named, rather than silently falling back to a
    default — a typo'd backend must not quietly log somewhere unintended.
    """
    key = name if name is not None else get_settings().tracking_backend
    try:
        return _REGISTRY[key]()
    except KeyError:
        valid = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"unknown tracking backend {key!r}; valid: {valid}") from None
```

Note the deliberate fail-closed lookup with no default — the same discipline Stage 5b will need for the broker registry, where a defaulted lookup would be a safety problem.

- [ ] **Step 5: Add the settings field**

In `algua/config/settings.py`, beside `mlflow_tracking_uri`, add:

```python
    # Which ExperimentTracker backend `--track` uses: "mlflow" (default) or "noop" (log nothing).
    # See algua/tracking/factory.py. Default preserves existing behaviour exactly.
    tracking_backend: str = "mlflow"
```

`env_prefix="ALGUA_"` makes this `ALGUA_TRACKING_BACKEND`. Do not add a validator — the factory's fail-closed lookup already rejects unknown values with a better message, and duplicating the check in two places is the kind of redundancy this program removes.

- [ ] **Step 6: Route `backtest_cmd` through the Protocol**

In `algua/cli/backtest_cmd.py`:

1. Replace `from algua.tracking.mlflow_tracker import log_backtest, log_sweep, log_walk_forward` with an import of `get_tracker` (and `TRACKING_SKIPPED`).
2. At each of the three `if track:` sites, obtain the tracker and call the method instead of the free function — e.g. `_record_tracking(payload, lambda: get_tracker().log_backtest(result, strategy.config.params, tracking_uri=get_settings().mlflow_tracking_uri))`. Keep the `lambda` and keep `_record_tracking` wrapping it: that wrapper is what guarantees a tracker failure never discards a completed evaluation.
3. Teach `_record_tracking` the fourth state, without disturbing the three that exist:

```python
    try:
        run_id = call()
    except Exception as exc:  # noqa: BLE001 - tracking is a best-effort side effect
        detail = f"{type(exc).__name__}: {exc}"
        payload["mlflow_run_id"] = None
        payload["mlflow_tracking_error"] = detail
        typer.echo(f"warning: mlflow tracking failed (result preserved): {detail}", err=True)
        return
    if run_id == TRACKING_SKIPPED:
        # A no-op backend was selected. Report that honestly rather than as a null run id, which
        # would be indistinguishable from a failure whose error key went missing.
        payload["mlflow_run_id"] = None
        payload["mlflow_tracking_skipped"] = "tracking backend logs nothing"
        return
    payload["mlflow_run_id"] = run_id
```

Extend `_record_tracking`'s docstring to document the fourth state.

4. **Add `"mlflow_tracking_skipped"` to BOTH `--summary` projection tuples** — `_WF_SUMMARY_KEYS` (~line 44) and `_SWEEP_SUMMARY_KEYS` (~line 50), which already list `mlflow_run_id` and `mlflow_tracking_error`. They are applied via `project(out, ...)` at the walk-forward (~:312) and sweep (~:357) emit sites (the `--summary` context-rot defense, #349).

This is **not optional polish**: `--summary` drops any key not in its tuple, so without it a `--summary --track` run on the no-op backend would emit `mlflow_run_id: null` with the explanation stripped — recreating precisely the ambiguous "null run id that looks like a failure" state this whole design exists to prevent, and only under `--summary`, which is the mode the CLI guide recommends for unattended operation. (`backtest run` has no summary projection, so it needs no change.)

**The default path must be untouched:** with `tracking_backend="mlflow"`, `run_id` is a real MLflow id, so neither new branch fires and the payload is exactly what it is today.

- [ ] **Step 7: Extend the tests**

`tests/test_tracking_backtest.py` currently only asserts the Protocol *has* three methods via `hasattr` — assertions that pass whether or not anything implements it. Add real coverage:

```python
def test_registered_trackers_satisfy_the_protocol():
    """Both backends structurally satisfy ExperimentTracker (the Protocol was dead until #5a)."""
    from algua.tracking.factory import get_tracker
    from algua.tracking.mlflow_tracker import ExperimentTracker
    for name in ("mlflow", "noop"):
        assert isinstance(get_tracker(name), ExperimentTracker)


def test_unknown_tracking_backend_fails_closed():
    from algua.tracking.factory import get_tracker
    with pytest.raises(ValueError, match="unknown tracking backend"):
        get_tracker("nope")


def test_noop_backend_reports_skipped_not_success(tmp_path, monkeypatch):
    """A no-op backend must not fabricate a run id -- the payload says skipped, not succeeded."""
    monkeypatch.setenv("ALGUA_TRACKING_BACKEND", "noop")
    ...  # invoke `backtest run --demo ... --track` via the CliRunner used elsewhere in this file
    assert payload["mlflow_run_id"] is None
    assert "mlflow_tracking_skipped" in payload
    assert "mlflow_tracking_error" not in payload
```

`ExperimentTracker` is **not** currently `@runtime_checkable` (verified: `mlflow_tracker.py` imports only `Protocol` from `typing`), so the `isinstance` check above will fail until you add the decorator. Add it — it is purely additive, changes neither the Protocol's shape nor any static behaviour, and is what lets a test assert the seam is real rather than assert three method *names* exist. Import `runtime_checkable` alongside `Protocol`.

For the third test, follow whatever invocation pattern the file's existing tests use for `backtest run` — read them first and match, rather than inventing a new harness. Assert the **default** backend still produces the old shape too (no `mlflow_tracking_skipped` key), so the fourth state is proven inert by default.

- [ ] **Step 8: Full quality gate**

Foreground, `timeout: 600000`: `uv run pytest -q 2>&1 | tail -20`, then ruff, mypy, lint-imports.

Expected: all four pass; test count **up** by the number of tests you added (this task removes none). Confirm `SCHEMA_VERSION`-style invariants are irrelevant here (this slice touches no DB), and check `git status` for the momentum hazard.

- [ ] **Step 9: Commit**

```bash
git add algua/tracking/factory.py algua/tracking/mlflow_tracker.py algua/config/settings.py algua/cli/backtest_cmd.py tests/test_tracking_backtest.py
```

```bash
git commit -m "$(cat <<'EOF'
refactor: wire the experiment-tracker seam behind a factory (stage 5a)

The ExperimentTracker Protocol has existed since #45 and was genuinely dead: its only references
anywhere were three hasattr assertions in a test. cli/backtest_cmd.py imported the three concrete
log_* functions directly. This closes the PR#110 tracker-DI deferral.

Adds tracking/factory.py (name->factory registry mirroring data/providers), an MlflowTracker
adapter over the existing log_* functions (which stay the implementation), a NoopTracker second
backend that proves the seam, and a tracking_backend setting defaulting to "mlflow".

The delicate part is _record_tracking's three-state JSON contract (not-requested / succeeded /
failed). A no-op returning a fake run id would fabricate "succeeded"; returning null silently would
be indistinguishable from a failure that lost its error key. So the no-op returns a sentinel and
the payload gets a distinct mlflow_tracking_skipped key -- a fourth state that can only appear when
a non-default backend is selected, leaving all three existing states and every existing test
untouched.

Unknown backends fail closed naming the valid set, rather than defaulting -- the same discipline
stage 5b's broker registry will need, where a defaulted lookup would be a safety problem.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Close-out verification

**Files:** none expected (verification only; fix anything found)

- [ ] **Step 1: The error leaf is a true leaf**

```bash
uv run python -c "
import ast, pathlib
tree = ast.parse(pathlib.Path('algua/execution/errors.py').read_text())
imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
algua = [n for n in imports if isinstance(n, ast.ImportFrom) and (n.module or '').startswith('algua')]
assert not algua, f'errors.py must import nothing from algua: {[n.module for n in algua]}'
print('OK: error leaf imports nothing from algua')
"
```

- [ ] **Step 2: Exactly one import path for `BrokerError`**

```bash
grep -rn "import.*BrokerError" algua/ tests/ --include='*.py'
```
Every hit must name `algua.execution.errors`. A hit naming `algua.execution.alpaca_broker` means a re-export or a missed importer survived — either is a defect (see Decision 1).

- [ ] **Step 3: The tracker default path is genuinely inert**

Prove the seam changed nothing for the default backend, by comparing a `--track` payload's key set against `main` before this branch. Simplest sufficient check: assert the default backend never produces the new key.

```bash
uv run python -c "
from algua.tracking.factory import get_tracker
from algua.tracking.mlflow_tracker import MlflowTracker, NoopTracker, TRACKING_SKIPPED
assert isinstance(get_tracker(), MlflowTracker), 'default backend must be mlflow'
assert isinstance(get_tracker('noop'), NoopTracker)
print('default backend:', type(get_tracker()).__name__)
print('noop sentinel  :', TRACKING_SKIPPED)
"
```
Expected: default is `MlflowTracker`, exit 0.

- [ ] **Step 4: Full quality gate**

Foreground, `timeout: 600000`: `uv run pytest -q 2>&1 | tail -20`, then ruff, mypy, lint-imports. Expected all four green; `lint-imports` 23 kept / 0 broken; test count equal to Task 2's ending count.

- [ ] **Step 5: CLI smoke test**

```bash
uv run algua doctor
uv run algua backtest run momentum --demo 2>&1 | tail -5
```
(`backtest run` takes the strategy as a positional `name`, not a `--strategy` option.) Both should behave exactly as on `main` — in particular the backtest payload must carry **no** tracking keys without `--track`. Anything mentioning a missing import, `BrokerError`, or a tracking key on an untracked run is a real regression.

- [ ] **Step 6: Commit any fixes**

If steps 1-5 forced fixes, commit them (scoped `git add`, correct trailer). If nothing needed fixing, this task makes no commit — expected, and consistent with how Stages 3, 4a, 4b, and 4c close-outs landed.
