# Dormant Lifecycle Stage (Slice A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a non-terminal `dormant` lifecycle stage (entered from `live`/`paper`, recoverable to `paper`) so a validated-but-resting strategy has a home other than the terminal `retired` tombstone — with a fail-closed wind-down wall on the `live -> dormant` edge.

**Architecture:** `dormant` is added to the `Stage` enum and `_LIVE_TRANSITIONS`; the auto-derived `-> RETIRED` edge gives `dormant -> retired` for free. All new edges are open to any actor. Entering `dormant` requires a non-empty reason (enforced in `transition_strategy`, the single policy path). Benching from `live` additionally requires the strategy be flat and atomically revokes its capital allocation — the revoke is threaded through `apply_transition` like the existing `consume_gate_id`, so it shares one transaction with the stage change and stays behind the same policy guards.

**Tech Stack:** Python 3.12, SQLite (registry DB), Typer CLI, pytest. Quality gate: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`.

**Spec:** `docs/superpowers/specs/2026-06-13-dormant-lifecycle-stage-issue-125-design.md`

---

## File Structure

- `algua/contracts/lifecycle.py` (CODEOWNERS-protected) — add `Stage.DORMANT` + edges. *Task 1.*
- `algua/registry/allocations.py` — add non-committing `revoke_active_locked`; refactor `deallocate` to use it. *Task 3.*
- `algua/registry/repository.py` — add `revoke_allocation` param to the `apply_transition` Protocol. *Task 3.*
- `algua/registry/store.py` — implement `revoke_allocation` in `apply_transition`/`_apply_transition_locked`. *Task 3.*
- `algua/registry/transitions.py` — bench-reason guard (Task 2) + `live -> dormant` flat check & revoke wiring (Task 4).
- `algua/cli/live_cmd.py` — `allocate` rejects a `dormant` strategy. *Task 5.*
- Tests: `tests/test_lifecycle.py` (Task 1), `tests/test_transitions.py` (Tasks 2, 4), `tests/test_allocations.py` (Task 3), `tests/test_registry_store.py` (Task 3), `tests/test_cli_live.py` (Task 5), `tests/test_cli_registry.py` or `tests/test_e2e_lifecycle.py` (Task 6).

> Before writing a new test file, check whether a sibling test module already exists for the target (e.g. `tests/test_transitions.py`, `tests/test_cli_live.py`); if so, append to it rather than creating a duplicate. Use `rg -l "transition_strategy" tests/` to confirm.

---

## Task 1: Add `dormant` to the lifecycle contract

**Files:**
- Modify: `algua/contracts/lifecycle.py`
- Test: `tests/test_lifecycle.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_lifecycle.py`:

```python
def test_dormant_entry_only_from_live_and_paper():
    assert can_transition(Stage.LIVE, Stage.DORMANT)
    assert can_transition(Stage.PAPER, Stage.DORMANT)
    # never entered from below paper — nothing pre-validation can "rest"
    for frm in (Stage.IDEA, Stage.BACKTESTED, Stage.CANDIDATE,
                Stage.FORWARD_TESTED, Stage.RETIRED):
        assert not can_transition(frm, Stage.DORMANT)


def test_dormant_is_non_terminal_and_recovers_to_paper():
    assert can_transition(Stage.DORMANT, Stage.PAPER)
    assert can_transition(Stage.DORMANT, Stage.RETIRED)  # derived give-up edge
    assert ALLOWED_TRANSITIONS[Stage.DORMANT]  # non-empty => non-terminal


def test_dormant_cannot_jump_to_live_or_forward():
    assert not can_transition(Stage.DORMANT, Stage.LIVE)
    assert not can_transition(Stage.DORMANT, Stage.FORWARD_TESTED)
    assert not can_transition(Stage.DORMANT, Stage.CANDIDATE)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_lifecycle.py -q`
Expected: FAIL with `AttributeError: DORMANT` (the enum member does not exist yet).

- [ ] **Step 3: Add the enum member and edges**

In `algua/contracts/lifecycle.py`, add `DORMANT` to the `Stage` enum between `LIVE` and `RETIRED`:

```python
class Stage(StrEnum):
    IDEA = "idea"
    BACKTESTED = "backtested"
    CANDIDATE = "candidate"
    PAPER = "paper"
    FORWARD_TESTED = "forward_tested"
    LIVE = "live"
    DORMANT = "dormant"
    RETIRED = "retired"
```

Then wire the edges in `_LIVE_TRANSITIONS` (the `-> RETIRED` edge is auto-derived for every non-retired stage, so do NOT add it by hand):

```python
_LIVE_TRANSITIONS: dict[Stage, set[Stage]] = {
    Stage.IDEA: {Stage.BACKTESTED},
    Stage.BACKTESTED: {Stage.CANDIDATE, Stage.IDEA},
    Stage.CANDIDATE: {Stage.PAPER, Stage.BACKTESTED},
    Stage.PAPER: {Stage.FORWARD_TESTED, Stage.CANDIDATE, Stage.DORMANT},
    Stage.FORWARD_TESTED: {Stage.LIVE, Stage.PAPER},
    Stage.LIVE: {Stage.PAPER, Stage.DORMANT},
    Stage.DORMANT: {Stage.PAPER},
}
```

- [ ] **Step 4: Run the full lifecycle test module**

Run: `uv run pytest tests/test_lifecycle.py -q`
Expected: PASS (new tests + the existing `test_transition_table_is_total` / `test_every_non_retired_stage_can_retire`, which automatically cover `dormant` because they iterate `Stage`).

- [ ] **Step 5: Commit**

```bash
git add algua/contracts/lifecycle.py tests/test_lifecycle.py
git commit -m "feat(125): add non-terminal dormant lifecycle stage + edges"
```

---

## Task 2: Bench-reason guard (entering `dormant` requires a reason)

**Files:**
- Modify: `algua/registry/transitions.py` (`transition_strategy`, near line 27-30, right after `validate_transition(rec.stage, target)`)
- Test: `tests/test_transitions.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_transitions.py` (mirror the fixture style already used in that file to build a `SqliteStrategyRepository` and move a strategy to `paper`; if no helper exists, use `algua.registry.db.connect`/`migrate` + `SqliteStrategyRepository`, `repo.add`, then `repo.apply_transition` through the legal chain idea→backtested→candidate→paper):

```python
import pytest
from algua.contracts.lifecycle import Actor, Stage, TransitionError
from algua.registry.db import connect, migrate
from algua.registry.store import SqliteStrategyRepository
from algua.registry.transitions import transition_strategy


def _paper_strategy(tmp_path):
    conn = connect(str(tmp_path / "reg.db"))
    migrate(conn)
    repo = SqliteStrategyRepository(conn)
    repo.add(name="s1")
    rec = repo.get("s1")
    for to in (Stage.BACKTESTED, Stage.CANDIDATE, Stage.PAPER):
        rec = repo.apply_transition(rec, to, Actor.HUMAN, reason="setup")
    return repo


def test_bench_to_dormant_requires_reason(tmp_path):
    repo = _paper_strategy(tmp_path)
    with pytest.raises(TransitionError, match="reason"):
        transition_strategy(repo, "s1", Stage.DORMANT, Actor.AGENT, reason="")


def test_bench_to_dormant_with_reason_succeeds(tmp_path):
    repo = _paper_strategy(tmp_path)
    rec = transition_strategy(repo, "s1", Stage.DORMANT, Actor.AGENT,
                              reason="seasonal signal degradation")
    assert rec.stage is Stage.DORMANT
    assert repo.list_transitions("s1")[-1]["reason"] == "seasonal signal degradation"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_transitions.py -k dormant -q`
Expected: FAIL — `test_bench_to_dormant_requires_reason` does not raise (no guard yet).

- [ ] **Step 3: Add the guard**

In `algua/registry/transitions.py`, inside `transition_strategy`, immediately after `validate_transition(rec.stage, target)` (around line 30) and before the `code_hash` locals, add:

```python
    if target is Stage.DORMANT and not (reason and reason.strip()):
        raise TransitionError("transition to dormant requires a non-empty reason")
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_transitions.py -k dormant -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add algua/registry/transitions.py tests/test_transitions.py
git commit -m "feat(125): require a reason when benching to dormant"
```

---

## Task 3: Atomic allocation revoke primitive + `apply_transition` parameter

**Files:**
- Modify: `algua/registry/allocations.py` (add `revoke_active_locked`; refactor `deallocate`)
- Modify: `algua/registry/repository.py` (`apply_transition` Protocol signature, lines 139-150)
- Modify: `algua/registry/store.py` (`apply_transition` ~254-273, `_apply_transition_locked` ~275-348)
- Test: `tests/test_allocations.py`, `tests/test_registry_store.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_allocations.py`:

```python
def test_revoke_active_locked_no_commit_is_idempotent(tmp_path):
    conn = _conn(tmp_path)
    repo = SqliteStrategyRepository(conn)
    repo.add(name="s1")
    sid = repo.get("s1").id
    allocations.allocate(conn, sid, capital=10_000.0, actor="human", account_equity=50_000.0)
    # revoke (no commit) then verify the active row is gone in this connection
    allocations.revoke_active_locked(conn, sid)
    assert allocations.active_allocation(conn, sid) is None
    # idempotent: a second revoke with nothing active is a no-op (no error)
    allocations.revoke_active_locked(conn, sid)
    assert allocations.active_allocation(conn, sid) is None
```

Append to `tests/test_registry_store.py` (use that module's existing `_repo`/`_conn` helper if present; otherwise build via `connect`/`migrate`/`SqliteStrategyRepository`):

```python
def test_apply_transition_revokes_allocation_atomically(tmp_path):
    from algua.registry import allocations
    from algua.contracts.lifecycle import Actor, Stage
    conn = connect(str(tmp_path / "reg.db")); migrate(conn)
    repo = SqliteStrategyRepository(conn)
    repo.add(name="s1")
    rec = repo.get("s1")
    for to in (Stage.BACKTESTED, Stage.CANDIDATE, Stage.PAPER,
               Stage.FORWARD_TESTED, Stage.LIVE):
        rec = repo.apply_transition(rec, to, Actor.HUMAN, reason="setup")
    allocations.allocate(conn, rec.id, capital=10_000.0, actor="human", account_equity=50_000.0)
    assert allocations.active_allocation(conn, rec.id) is not None
    rec = repo.apply_transition(rec, Stage.DORMANT, Actor.AGENT, reason="bench",
                                revoke_allocation=True)
    assert rec.stage is Stage.DORMANT
    assert allocations.active_allocation(conn, rec.id) is None


def test_apply_transition_revoke_rolls_back_with_stage_on_cas_failure(tmp_path):
    from algua.registry import allocations
    from algua.contracts.lifecycle import Actor, Stage
    from algua.registry.repository import StrategyRecord
    from algua.registry.transitions import TransitionError
    import pytest
    conn = connect(str(tmp_path / "reg.db")); migrate(conn)
    repo = SqliteStrategyRepository(conn)
    repo.add(name="s1")
    rec = repo.get("s1")
    for to in (Stage.BACKTESTED, Stage.CANDIDATE, Stage.PAPER,
               Stage.FORWARD_TESTED, Stage.LIVE):
        rec = repo.apply_transition(rec, to, Actor.HUMAN, reason="setup")
    allocations.allocate(conn, rec.id, capital=10_000.0, actor="human", account_equity=50_000.0)
    # Force the stage CAS to fail by passing a stale from-stage (rec read as PAPER, but DB is LIVE)
    stale = StrategyRecord(id=rec.id, name=rec.name, stage=Stage.PAPER,
                           created_at=rec.created_at, updated_at=rec.updated_at)
    with pytest.raises(TransitionError):
        repo.apply_transition(stale, Stage.DORMANT, Actor.AGENT, reason="bench",
                              revoke_allocation=True)
    # rollback: allocation still active AND stage still live
    assert allocations.active_allocation(conn, rec.id) is not None
    assert repo.get("s1").stage is Stage.LIVE
```

> Note: `StrategyRecord` field names — confirm them against `algua/registry/repository.py:31` and adjust the `stale = StrategyRecord(...)` construction to match the actual dataclass fields (id, name, stage, plus whatever timestamps it carries).

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_allocations.py -k revoke tests/test_registry_store.py -k revoke -q`
Expected: FAIL — `revoke_active_locked` and the `revoke_allocation` kwarg don't exist yet.

- [ ] **Step 3a: Add `revoke_active_locked` and refactor `deallocate` in `allocations.py`**

Add (no commit — caller owns the transaction):

```python
def revoke_active_locked(conn: sqlite3.Connection, strategy_id: int) -> None:
    """Revoke a strategy's active allocation WITHOUT committing — the caller owns the transaction
    (e.g. the stage-change txn in `_apply_transition_locked`). No-op if nothing is active."""
    existing = active_allocation(conn, strategy_id)
    if existing is None:
        return
    conn.execute("UPDATE strategy_allocations SET revoked_ts = ? WHERE id = ?",
                 (datetime.now(UTC).isoformat(), existing["id"]))
```

Refactor `deallocate` to delegate, preserving its `is_flat` guard + commit:

```python
def deallocate(conn: sqlite3.Connection, strategy_id: int, actor: str, is_flat: bool) -> None:
    """Revoke a strategy's active allocation. Requires the strategy flat with no open orders
    (the caller computes `is_flat` from the ledger + broker)."""
    if not is_flat:
        raise AllocationError("cannot deallocate a strategy that is not flat / has open orders")
    revoke_active_locked(conn, strategy_id)
    conn.commit()
```

- [ ] **Step 3b: Add the `revoke_allocation` parameter to the `apply_transition` Protocol**

In `algua/registry/repository.py`, add the parameter to the `apply_transition` signature (after `consume_forward_gate_id`):

```python
    def apply_transition(
        self,
        rec: StrategyRecord,
        to: Stage,
        actor: Actor,
        reason: str | None = None,
        code_hash: str | None = None,
        config_hash: str | None = None,
        dependency_hash: str | None = None,
        consume_gate_id: int | None = None,
        consume_forward_gate_id: int | None = None,
        revoke_allocation: bool = False,
    ) -> StrategyRecord:
```

Extend its docstring with one line: `When ``revoke_allocation`` is set, the strategy's active live allocation is revoked in the SAME transaction as the stage change (used by the live -> dormant bench edge).`

- [ ] **Step 3c: Implement it in `store.py`**

In `apply_transition` (store.py ~254), thread the new param through both the signature and the `_apply_transition_locked` call:

```python
    def apply_transition(
        self,
        rec: StrategyRecord,
        to: Stage,
        actor: Actor,
        reason: str | None = None,
        code_hash: str | None = None,
        config_hash: str | None = None,
        dependency_hash: str | None = None,
        consume_gate_id: int | None = None,
        consume_forward_gate_id: int | None = None,
        revoke_allocation: bool = False,
    ) -> StrategyRecord:
        if consume_gate_id is not None and consume_forward_gate_id is not None:
            raise ValueError(
                "at most one of consume_gate_id/consume_forward_gate_id may be set — a single"
                " transition spends a single token")
        with self._conn:  # consume + revoke + UPDATE + INSERT commit together or not at all
            return self._apply_transition_locked(
                rec, to, actor, reason, code_hash, config_hash, dependency_hash,
                consume_gate_id, consume_forward_gate_id, _now(),
                revoke_allocation=revoke_allocation)
```

Add the matching keyword-only param to `_apply_transition_locked`'s signature (after `now: str`):

```python
        now: str,
        revoke_allocation: bool = False,
    ) -> StrategyRecord:
```

Inside `_apply_transition_locked`, BEFORE the stage compare-and-swap `UPDATE strategies ...` block, add the revoke (so a CAS failure rolls it back too — they share the caller's `with self._conn:`):

```python
        if revoke_allocation:
            # Bench wind-down (#125): revoke the live capital reservation in the SAME transaction
            # as the stage CAS below, so a raced/failed transition leaves the allocation intact —
            # never a dormant strategy still holding capital, nor a live one with none.
            from algua.registry import allocations
            allocations.revoke_active_locked(self._conn, rec.id)
```

> The other call site of `_apply_transition_locked` (`record_forward_pass_and_promote`, store.py:~683) passes positional args up to `now=_now()`; since `revoke_allocation` defaults to `False`, that call needs no change. Verify it still type-checks.

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_allocations.py tests/test_registry_store.py -q`
Expected: PASS (including the rollback test).

- [ ] **Step 5: Commit**

```bash
git add algua/registry/allocations.py algua/registry/repository.py algua/registry/store.py tests/test_allocations.py tests/test_registry_store.py
git commit -m "feat(125): atomic allocation-revoke threaded through apply_transition"
```

---

## Task 4: `live -> dormant` flat precondition + revoke wiring in `transition_strategy`

**Files:**
- Modify: `algua/registry/transitions.py` (`transition_strategy`)
- Test: `tests/test_transitions.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_transitions.py` (reuse the `connect`/`migrate` setup; drive a strategy to `live` via the legal chain, then seed a live fill so `believed_positions` is non-empty). Confirm the `live_fills` insert columns against `algua/execution/live_ledger.py:believed_positions` (it sums `qty` grouped by `symbol` from `live_fills WHERE strategy = ?`):

```python
from algua.registry import allocations


def _live_strategy(tmp_path):
    conn = connect(str(tmp_path / "reg.db")); migrate(conn)
    repo = SqliteStrategyRepository(conn)
    repo.add(name="s1")
    rec = repo.get("s1")
    for to in (Stage.BACKTESTED, Stage.CANDIDATE, Stage.PAPER,
               Stage.FORWARD_TESTED, Stage.LIVE):
        rec = repo.apply_transition(rec, to, Actor.HUMAN, reason="setup")
    return repo, conn


def test_live_to_dormant_rejected_when_not_flat(tmp_path):
    repo, conn = _live_strategy(tmp_path)
    conn.execute("INSERT INTO live_fills(strategy, symbol, qty) VALUES (?,?,?)",
                 ("s1", "AAPL", 5.0))
    conn.commit()
    with pytest.raises(TransitionError, match="flat"):
        transition_strategy(repo, "s1", Stage.DORMANT, Actor.AGENT, reason="bench")


def test_live_to_dormant_flat_succeeds_and_revokes_allocation(tmp_path):
    repo, conn = _live_strategy(tmp_path)
    allocations.allocate(conn, repo.get("s1").id, capital=10_000.0, actor="human",
                         account_equity=50_000.0)
    rec = transition_strategy(repo, "s1", Stage.DORMANT, Actor.AGENT, reason="bench")
    assert rec.stage is Stage.DORMANT
    assert allocations.active_allocation(conn, rec.id) is None
```

> `live_fills` column names: verify with `rg "CREATE TABLE.*live_fills" -A6 algua/` and adjust the INSERT. If the table requires more NOT NULL columns, include them.

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_transitions.py -k "live_to_dormant" -q`
Expected: FAIL — no flat check yet; the not-flat case wrongly succeeds.

- [ ] **Step 3: Add the live -> dormant branch**

In `algua/registry/transitions.py` `transition_strategy`, add a local `revoke_allocation = False` near the other locals (after the `consume_*` locals), then add an `elif` branch to the existing target/stage chain (alongside the `Stage.LIVE` / `backtested->candidate` / `paper->forward_tested` branches):

```python
    elif rec.stage is Stage.LIVE and target is Stage.DORMANT:
        # Bench wind-down wall (#125): a live strategy must be flat before resting, else its open
        # positions are orphaned (run-all only iterates Stage.LIVE). The check is on this single
        # policy path so it cannot be bypassed; the revoke happens atomically in apply_transition.
        # believed_positions is imported lazily — same registry->execution pattern the live
        # certificate verifier already uses in this module.
        from algua.execution.live_ledger import believed_positions
        conn = getattr(repo, "connection", None)
        if conn is None:
            raise TransitionError(
                "benching a live strategy needs a sqlite-backed repository")
        if believed_positions(conn, name):
            raise TransitionError(
                f"{name} is not flat (open live positions); flatten before benching to dormant")
        revoke_allocation = True
```

Then pass it to the final `apply_transition` call:

```python
    return repo.apply_transition(
        rec=rec,
        to=target,
        actor=transition_actor,
        reason=reason,
        code_hash=code_hash,
        config_hash=config_hash,
        dependency_hash=dependency_hash,
        consume_gate_id=consume_gate_id,
        consume_forward_gate_id=consume_forward_gate_id,
        revoke_allocation=revoke_allocation,
    )
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_transitions.py -q`
Expected: PASS (the flat-reject, flat-success-revokes, and the Task-2 reason tests).

- [ ] **Step 5: Commit**

```bash
git add algua/registry/transitions.py tests/test_transitions.py
git commit -m "feat(125): live->dormant flat precondition + atomic allocation release"
```

---

## Task 5: `live allocate` rejects a dormant strategy

**Files:**
- Modify: `algua/cli/live_cmd.py` (`allocate`, ~line 75-92)
- Test: `tests/test_cli_live.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_cli_live.py` (follow that module's existing CLI-invocation pattern — it likely uses the typer runner or the `algua.cli.main:main` helper with `ALGUA_DB_PATH` pointed at a tmp registry; reuse whatever fixture creates a strategy). The assertion: allocating to a `dormant` strategy fails with a clear message. Build the dormant strategy through the legal chain (live->dormant after flat), or for a pure-CLI unit just transition a `paper` strategy to `dormant`:

```python
def test_live_allocate_rejects_dormant(tmp_path, monkeypatch):
    # ... set ALGUA_DB_PATH to tmp registry; add strategy "s1"; move it to dormant
    #     via the legal chain (paper -> dormant with a reason) using the registry CLI/repo ...
    # Then:
    result = run_cli(["live", "allocate", "s1", "--capital", "10000"])
    assert result.exit_code != 0
    assert "dormant" in result.output.lower()
```

> Match `run_cli`/fixtures to the existing helpers in `tests/test_cli_live.py`. If `allocate` calls Alpaca for equity (`_live_account_equity`), the guard must fire BEFORE that network call — so this test needs no broker mock (the stage check rejects first).

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_cli_live.py -k allocate_rejects_dormant -q`
Expected: FAIL — allocate proceeds (or errors for the wrong reason, e.g. missing Alpaca creds).

- [ ] **Step 3: Add the guard**

In `algua/cli/live_cmd.py` `allocate`, fetch the record (instead of just `.id`) and reject dormant before the equity call:

```python
    with registry_conn() as conn:
        rec = SqliteStrategyRepository(conn).get(name)
        if rec.stage is Stage.DORMANT:
            raise ValueError(
                f"cannot allocate live capital to dormant strategy {name!r}; a recovered "
                "strategy re-allocates only after re-climbing paper -> ... -> live")
        allocations.allocate(conn, rec.id, capital=capital, actor="human",
                             account_equity=_live_account_equity())
```

(Confirm `Stage` is already imported in `live_cmd.py` — it is, at line 14.)

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_cli_live.py -k allocate_rejects_dormant -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add algua/cli/live_cmd.py tests/test_cli_live.py
git commit -m "feat(125): reject live allocate for a dormant strategy"
```

---

## Task 6: End-to-end CLI lifecycle + trade-lane rejection

**Files:**
- Test: `tests/test_e2e_lifecycle.py` (append; if absent, add to `tests/test_cli_registry.py`)

- [ ] **Step 1: Write the failing/again-green e2e tests**

Append CLI-level tests driving the real `registry transition` command (reuse the module's existing CLI runner + DB fixture). Cover:

```python
def test_e2e_paper_to_dormant_to_retired(...):
    # add s1; drive to paper via legal chain (CLI transitions, actor=agent where allowed)
    # bench: registry transition s1 --to dormant --actor agent --reason "seasonal"
    #   -> exit 0, stage == "dormant"
    # give up: registry transition s1 --to retired --actor agent --reason "done"
    #   -> exit 0, stage == "retired"
    ...

def test_e2e_paper_to_dormant_requires_reason(...):
    # registry transition s1 --to dormant --actor agent   (no --reason)
    #   -> exit != 0, message mentions reason
    ...

def test_e2e_dormant_recovers_to_paper(...):
    # from dormant: registry transition s1 --to paper --actor agent
    #   -> exit 0, stage == "paper"
    ...

def test_dormant_strategy_not_run_by_paper_lane(...):
    # a dormant strategy is rejected by the paper run command
    #   (paper_cmd guards stage in (PAPER, FORWARD_TESTED)) -> exit != 0
    ...
```

> Fill each `...` with the module's real CLI-invocation helper and assertions on `exit_code` / parsed JSON `stage`. The `paper -> dormant` edge needs no flat/allocation handling, so these are pure CLI-flow assertions. For the paper-lane rejection, call the same paper run subcommand the existing paper tests use, with the strategy parked at `dormant`.

- [ ] **Step 2: Run to verify behavior**

Run: `uv run pytest tests/test_e2e_lifecycle.py -q` (or the chosen module)
Expected: PASS — these exercise already-implemented behavior (Tasks 1-2) end-to-end through the CLI; if any fail, fix the test wiring, not the production code.

- [ ] **Step 3: Commit**

```bash
git add tests/test_e2e_lifecycle.py
git commit -m "test(125): e2e dormant lifecycle + paper-lane rejection"
```

---

## Task 7: Full quality gate

- [ ] **Step 1: Run the complete gate**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: all green. In particular `lint-imports` must pass — the `believed_positions` and `allocations` imports added in Tasks 3-4 are function-level (lazy), matching the existing `registry -> execution` pattern in `transitions.py`, so no contract should break.

- [ ] **Step 2: Fix any failures**

If `mypy` flags the new `revoke_allocation` param at any `apply_transition` call site, add the arg or rely on the default. If `ruff` flags unused imports or line length, fix inline. Re-run the gate until green.

- [ ] **Step 3: Final commit (if Step 2 changed anything)**

```bash
git add -A
git commit -m "chore(125): satisfy quality gate for dormant stage"
```

---

## Self-Review notes (already reconciled against the spec)

- **State model** (spec §"The state model") → Task 1.
- **Bench-reason guard** (spec §"The bench-reason guard") → Task 2.
- **Flat precondition + atomic allocation revoke, single policy path** (spec §"Winding down a live strategy") → Tasks 3 + 4.
- **`live allocate` rejects dormant** (spec §"What does NOT change", capital-guard edit) → Task 5.
- **No DB migration** — confirmed: no schema change in any task (`stage` is free TEXT; `revoked_ts` already exists).
- **Tests** (spec §"Tests": flat-reject, flat-success+revoke, atomic rollback, blank-reason, illegal-source, live-allocate-rejects-dormant, e2e flows, paper-lane rejection) → distributed across Tasks 1-6. Illegal-source into dormant is covered by Task 1's `test_dormant_cannot_jump_*` + `validate_transition` (already called first in `transition_strategy`).
- **Deferred (NOT in this plan):** the re-eval sweep (Slice B), a structured `dormancy_reason` column, and the pre-existing `live -> paper` wind-down gap.
