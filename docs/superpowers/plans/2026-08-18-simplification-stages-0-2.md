# System Simplification — Stages 0–2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze operations, execute the approved kill-list (~4,300 LOC), and build `algua/primitives/` + `registry/challenges.py`, migrating every duplicate flock/atomic-write/retry/challenge implementation onto them.

**Architecture:** Strangler stages 0–2 of the program spec. Deletions first (they shrink the primitive migration surface), then a stdlib-only leaf package `algua/primitives/` (flock, atomic_io, retry) that everything may import, then the registry-internal challenge-lifecycle unification. Every task is behavior-preserving except the deletions, and each task leaves the full quality gate green.

**Tech Stack:** Python 3.12, uv, pytest, ruff, mypy, import-linter, sqlite3, fcntl.

**Spec:** `docs/superpowers/specs/2026-08-18-system-simplification-design.md`

## Global Constraints

- Quality gate on EVERY task before commit: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`. All four must pass.
- No behavior change outside the explicit deletions. No compat shims, no re-exports kept "just in case" (user rule: no backwards-compat cruft).
- Safety-critical ordering comments move VERBATIM with their code (audit-before-mutate, fail-closed probe semantics, fresh-fd flock rationale, #394 redirect defenses, #329 namespace scoping).
- `git add` is always scoped to the named files — never `git add -A` (concurrent-session rule).
- Commits end with: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`
- DB migrations: `migrate()` stays idempotent-bootstrap (no user_version gating); table drops use `DROP TABLE IF EXISTS` inside `migrate()`; `SCHEMA_VERSION` bumps are generation markers only.
- Two audit corrections are BINDING amendments to the spec kill-list (Task 1 records them in the spec):
  1. `algua/registry/lineage.py` is CORE — `registry/promotion.py:16` and `registry/store.py:2152` import `factors_used_by` for family classification. It is NOT deleted. `tests/test_registry_lineage.py` and `tests/test_cross_sectional_momentum_rewire.py` stay.
  2. `append_if_absent` stays a `SnapshotManifest` method in `algua/data/manifest.py` (it is manifest-append-specific, not a free function); only its internal `_acquire_lock` migrates to the flock primitive.

---

### Task 1: Stage 0 — freeze operations + spec amendments

**Files:**
- Modify: `docs/superpowers/specs/2026-08-18-system-simplification-design.md` (kill-list §3, primitives §4)

**Interfaces:**
- Produces: a halted system (no timers firing) and a corrected spec for all later tasks.

- [ ] **Step 1: Stop and disable the three operator timers (keep algua-web.service running — it is read-only)**

Run:
```bash
systemctl --user disable --now algua-paper.timer algua-mergeback-drain.timer algua-research.timer
```

- [ ] **Step 2: Verify nothing algua-related fires anymore**

Run: `systemctl --user list-timers --all | grep algua || echo "no algua timers"`
Expected: `no algua timers`. Also `systemctl --user is-active algua-web.service` → `active` (deliberately left running).

- [ ] **Step 3: Amend the spec kill-list**

In `docs/superpowers/specs/2026-08-18-system-simplification-design.md`:
- In §3, factor row: change "…`registry/lineage.py`, `research/factor_fdr.py`…" so `registry/lineage.py` is removed from the DELETE row and add to its Notes: "KEEPS `registry/lineage.py` — `factors_used_by` is imported by `registry/promotion.py` and `registry/store.py` (family classification factor axis)."
- In §4 item 2, change "`append_if_absent` promoted here" to: "`append_if_absent` stays a `SnapshotManifest` method (manifest-specific); only its `_acquire_lock` migrates to the flock primitive."

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/specs/2026-08-18-system-simplification-design.md
git commit -m "docs: amend spec kill-list (lineage.py is core; append_if_absent stays in manifest)"
```

---

### Task 2: Delete the shadow lane

**Files:**
- Delete: `algua/shadow/` (`__init__.py`, `evaluate.py`, `store.py`), `algua/cli/shadow_cmd.py`, `tests/test_shadow_cmd.py`, `tests/test_shadow_evaluate.py`
- Modify: `algua/cli/main.py` (import tuple), `pyproject.toml` (3 places), `algua/registry/db.py` (schema + migrate + version), `tests/test_registry_db.py` (shadow-table assertions)

**Interfaces:**
- Produces: `SCHEMA_VERSION = 40`; `migrate()` drops `shadow_evaluations` on existing DBs.

- [ ] **Step 1: Delete the package and CLI module**

```bash
git rm -r algua/shadow algua/cli/shadow_cmd.py tests/test_shadow_cmd.py tests/test_shadow_evaluate.py
```

- [ ] **Step 2: Remove registration and lint references**

- `algua/cli/main.py`: delete the `shadow_cmd,` line from the `from algua.cli import (…)` tuple.
- `pyproject.toml`: delete `"algua.cli.shadow_cmd",` from the CLI independence contract module list (~line 314); delete BOTH shadow contracts entirely — the `[[tool.importlinter.contracts]]` block named "shadow lane cannot reach real orders, allocations, or the risk machinery" (~319–343) and the block named "trading and research lanes stay off the advisory shadow lane" (~344–351), including their comment headers.

- [ ] **Step 3: Drop the table from schema + migrate**

In `algua/registry/db.py`:
- Delete the `shadow_evaluations` `CREATE TABLE` block from `_SCHEMA` (the block around lines 730–756, including its comment header).
- Change `SCHEMA_VERSION = 39` → `SCHEMA_VERSION = 40` and add above it:
```python
# v40 (simplification stage 1): the advisory shadow lane is deleted; migrate() drops its table.
```
- In `migrate()`, immediately after `conn.executescript(_SCHEMA)`, add:
```python
    # v40: advisory shadow lane deleted (simplification stage 1). Idempotent drop; the rows were
    # advisory-only (never gate evidence), so no export is taken.
    conn.execute("DROP TABLE IF EXISTS shadow_evaluations")
```

- [ ] **Step 4: Fix tests that assert the table exists**

In `tests/test_registry_db.py`, find assertions listing `shadow_evaluations` (e.g. an expected-tables set) and remove the entry. If a test asserts the table is created, invert nothing — just delete the entry/test line.

- [ ] **Step 5: Run the quality gate**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: all pass. If a straggler import of `algua.shadow` surfaces, delete that reference too (it is dead by definition — nothing in Inventory 3 imports it).

- [ ] **Step 6: Commit**

```bash
git add -u algua tests pyproject.toml
git commit -m "refactor: delete advisory shadow lane (spec §3, v40 drops shadow_evaluations)"
```

---

### Task 3: Delete the monitoring lane (drift + decay)

**Files:**
- Delete: `algua/monitoring/` (`__init__.py`, `drift.py`, `decay.py`), `algua/cli/monitoring_cmd.py`, `tests/test_cli_monitoring.py`, `tests/test_cli_monitoring_decay.py`, `tests/test_monitoring_drift.py`, `tests/test_monitoring_decay.py`
- Modify: `algua/cli/main.py`, `pyproject.toml`

**Interfaces:**
- Consumes: nothing. Produces: nothing (pure removal; no DB table involved).

- [ ] **Step 1: Delete**

```bash
git rm -r algua/monitoring algua/cli/monitoring_cmd.py tests/test_cli_monitoring.py tests/test_cli_monitoring_decay.py tests/test_monitoring_drift.py tests/test_monitoring_decay.py
```

- [ ] **Step 2: Remove registration and lint references**

- `algua/cli/main.py`: delete the `monitoring_cmd,` line.
- `pyproject.toml`: delete `"algua.cli.monitoring_cmd",` from the CLI independence list (~line 308). Search the whole file for any other `monitoring` contract mention (e.g. purity/layer contracts listing `algua.monitoring`) and delete those entries.

- [ ] **Step 3: Quality gate**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: all pass. `monitoring/drift.py` imported `backtest.factor_eval.factor_ic` — that edge dies with this deletion (factor_eval itself dies in Task 5).

- [ ] **Step 4: Commit**

```bash
git add -u algua tests pyproject.toml
git commit -m "refactor: delete advisory monitoring lane (drift/decay, spec §3)"
```

---

### Task 4: Delete research advisory commands (gc, family-audit, dormant-sweep) + dead family_budget

**Files:**
- Delete: `algua/research/lifecycle_gc.py`, `algua/research/family_audit.py`, `algua/registry/family_budget.py`, `tests/test_research_gc.py`, `tests/test_cli_research_gc.py`, `tests/test_cli_family_audit.py`, `tests/research/test_family_audit.py`, `tests/registry/test_family_budget.py`
- Modify: `algua/cli/research_cmd.py` (three command bodies + gc fs layer), `tests/test_cli_research.py` (dormant-sweep tests), `CLAUDE.md` (three command doc bullets)

**Interfaces:**
- Consumes: nothing from other tasks. Produces: `research_cmd.py` shrinks by ~750 lines; `research/clustering.py` is untouched (it is core).

- [ ] **Step 1: Delete the standalone modules and their tests**

```bash
git rm algua/research/lifecycle_gc.py algua/research/family_audit.py algua/registry/family_budget.py \
  tests/test_research_gc.py tests/test_cli_research_gc.py tests/test_cli_family_audit.py \
  tests/research/test_family_audit.py tests/registry/test_family_budget.py
```

- [ ] **Step 2: Excise the three commands from `algua/cli/research_cmd.py`**

Delete these regions (locate by function name; approximate current lines given):
- `dormant_sweep` command + helpers (~482–595).
- `family_audit_cmd` + its 5-step pipeline (~766–843).
- The ENTIRE gc block: `_gc_scan_roots`, `_gc_inventory`, `_archive_run_id`, `_open_archive_parent_dir`, `_gc_archive`, `_read_fd_all`, and the `gc` command (~844–1312).
- Then remove the now-unused imports at the top: `from algua.research.family_audit import …`, `from algua.research.lifecycle_gc import …`, and any imports used ONLY by the deleted bodies (run ruff to find them — `uv run ruff check algua/cli/research_cmd.py` reports unused imports).

- [ ] **Step 3: Remove dormant-sweep tests from `tests/test_cli_research.py`**

Delete every test function whose name contains `dormant_sweep` (grep: `grep -n "dormant" tests/test_cli_research.py`). Leave all other tests untouched.

- [ ] **Step 4: Update CLAUDE.md**

Delete the three command-surface bullets: `uv run algua research dormant-sweep …`, `uv run algua research family-audit`, `uv run algua research gc …` (each is one full bullet paragraph).

- [ ] **Step 5: Quality gate**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add -u algua tests CLAUDE.md
git commit -m "refactor: delete advisory research commands gc/family-audit/dormant-sweep + dead family_budget (spec §3)"
```

---

### Task 5: Delete pbo + factor-eval layer; collapse sweep_with_matrix

**Files:**
- Delete: `algua/research/cscv.py`, `algua/backtest/factor_eval.py`, `algua/cli/factor_cmd.py`, `algua/research/factor_fdr.py`, `tests/test_cli_research_pbo.py`, `tests/research/test_cscv.py`, `tests/test_sweep_pbo_matrix.py`, `tests/test_sweep_pbo_matrix_withheld.py`, `tests/test_cli_factor.py`, `tests/test_factor_eval_adapter.py`, `tests/test_factor_fdr.py`, `tests/test_factor_ledger.py`, `tests/test_factor_ic.py`, `tests/test_factor_eval_panel.py`, `tests/test_factor_eval_run.py`
- Modify: `algua/cli/research_cmd.py` (pbo command), `algua/backtest/sweep.py` (collapse), `algua/cli/main.py`, `pyproject.toml`, `algua/registry/db.py` (drop table, v41), `algua/registry/store.py` (factor ledger methods), `algua/registry/repository.py` (FactorLedger protocol), `tests/test_db_migrations.py`, `tests/test_registry_db.py` (factor-table refs), `tests/test_result_to_dict.py` (comment), `CLAUDE.md`
- KEEP UNTOUCHED: `algua/registry/lineage.py`, `algua/features/catalogue.py`, `algua/features/alphas.py`, `algua/features/indicators.py`, `tests/test_registry_lineage.py`, `tests/test_cross_sectional_momentum_rewire.py`

**Interfaces:**
- Produces: `SCHEMA_VERSION = 41`; `sweep()` is the only sweep entry point; `StrategyRepository` union no longer includes `FactorLedger`.

- [ ] **Step 1: Delete modules and tests**

```bash
git rm algua/research/cscv.py algua/backtest/factor_eval.py algua/cli/factor_cmd.py algua/research/factor_fdr.py \
  tests/test_cli_research_pbo.py tests/research/test_cscv.py tests/test_sweep_pbo_matrix.py \
  tests/test_sweep_pbo_matrix_withheld.py tests/test_cli_factor.py tests/test_factor_eval_adapter.py \
  tests/test_factor_fdr.py tests/test_factor_ledger.py tests/test_factor_ic.py \
  tests/test_factor_eval_panel.py tests/test_factor_eval_run.py
```

- [ ] **Step 2: Excise `pbo_cmd` from `algua/cli/research_cmd.py`**

Delete the `pbo_cmd` command body (~596–765) and its now-unused imports (`from algua.research.cscv import …`, sweep-matrix imports). Run `uv run ruff check algua/cli/research_cmd.py` to catch stragglers.

- [ ] **Step 3: Collapse `sweep_with_matrix` in `algua/backtest/sweep.py`**

- Delete the `sweep_with_matrix` function (~line 372 onward) and any module-level helper used only by it (the OOS-matrix ride-along assembly, ~line 230 comment region).
- `sweep()`'s public signature is UNCHANGED (`compute_holdout` stays — it is `walk_forward`'s documented seam). If `sweep()` currently delegates to `sweep_with_matrix`, inline the matrix-free body back into `sweep()` so `sweep()` is self-contained.
- In `tests/test_result_to_dict.py:87`, rewrite the comment that references `sweep_with_matrix()`'s tuple to describe `to_dict()` directly (e.g. "the OOS matrix ride-along was removed with research pbo; to_dict() is a plain scalar projection").

- [ ] **Step 4: Remove registration and lint references**

- `algua/cli/main.py`: delete the `factor_cmd,` line.
- `pyproject.toml`: delete `"algua.cli.factor_cmd",` from the CLI independence list (~line 303). Grep `pyproject.toml` for `factor_eval`/`factor_fdr` contract mentions and delete those entries (the `features/catalogue` purity contract at ~157–160 STAYS).

- [ ] **Step 5: Remove the factor ledger from registry**

- `algua/registry/store.py`: delete `record_factor_evaluation`, `factor_hypothesis_breadth`, `windowed_factor_irs` (and any `finalize_factor_*` sibling in the same region, ~1790–1905). Do NOT touch line ~2152 (`from algua.registry.lineage import factors_used_by` — core).
- `algua/registry/repository.py`: delete the `class FactorLedger(Protocol)` block (~778–837) and remove `FactorLedger` from the `StrategyRepository` union bases (~973–996).
- `algua/registry/db.py`: delete the `factor_evaluations` `CREATE TABLE` block (~562–595); bump `SCHEMA_VERSION = 40` → `41` with comment `# v41 (simplification stage 1): standalone factor-eval layer deleted; migrate() drops its table.`; in `migrate()` next to the v40 drop add:
```python
    conn.execute("DROP TABLE IF EXISTS factor_evaluations")
```

- [ ] **Step 6: Fix schema tests + CLAUDE.md**

- `tests/test_db_migrations.py` and `tests/test_registry_db.py`: remove `factor_evaluations` entries/assertions.
- `CLAUDE.md`: delete the bullets for `research pbo`, `factor list`, `factor show`, `factor eval`, `factor dependents`, `factor uses` (the whole factor block), and the "Factors are NEVER gate-tokened" sentence with them.

- [ ] **Step 7: Quality gate**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: all pass. `research/clustering.py` and `registry/promotion.py` must be untouched by this task.

- [ ] **Step 8: Commit**

```bash
git add -u algua tests pyproject.toml CLAUDE.md
git commit -m "refactor: delete pbo + factor-eval layer, collapse sweep_with_matrix (spec §3, v41)"
```

---

### Task 6: Create `algua/primitives/flock.py` (TDD)

**Files:**
- Create: `algua/primitives/__init__.py`, `algua/primitives/flock.py`
- Test: `tests/primitives/__init__.py`, `tests/primitives/test_flock.py`
- Modify: `pyproject.toml` (leaf contract)

**Interfaces:**
- Produces (consumed by Tasks 7–8):
  - `acquire(path: Path, *, blocking: bool = True, verify_inode: bool = False, retries: int = 5) -> int` — fresh fd + `LOCK_EX`; raises `LockHeld` (non-blocking contention) / `LockReplaced` (inode churn).
  - `release(fd: int) -> None`
  - `file_lock(path: Path, *, blocking: bool = True, metadata: dict | None = None, on_oserror: Literal["raise", "proceed"] = "raise")` — context manager; `LockHeld.holder` carries parsed lock-body JSON.
  - `probe_held(path: Path) -> bool` — fail-closed non-blocking probe.
  - `read_holder(path: Path) -> dict | None`
  - Exceptions: `LockHeld(Exception)` with `.holder: dict | None`; `LockReplaced(Exception)`.

- [ ] **Step 1: Write the failing tests**

`tests/primitives/test_flock.py`:
```python
"""algua.primitives.flock — the ONE cross-process flock primitive (spec §4.1)."""
from __future__ import annotations

import json
import multiprocessing
import os
from pathlib import Path

import pytest

from algua.primitives.flock import (
    LockHeld,
    LockReplaced,
    acquire,
    file_lock,
    probe_held,
    read_holder,
    release,
)


def _hold_lock(path: str, acquired_evt, release_evt) -> None:
    fd = acquire(Path(path))
    acquired_evt.set()
    release_evt.wait(timeout=30)
    release(fd)


@pytest.fixture()
def holder(tmp_path):
    """A live child process holding LOCK_EX on tmp_path/'x.lock'."""
    lock = tmp_path / "x.lock"
    acquired = multiprocessing.Event()
    released = multiprocessing.Event()
    proc = multiprocessing.Process(target=_hold_lock, args=(str(lock), acquired, released))
    proc.start()
    assert acquired.wait(timeout=10)
    yield lock
    released.set()
    proc.join(timeout=10)


def test_acquire_release_roundtrip(tmp_path):
    lock = tmp_path / "a.lock"
    fd = acquire(lock)
    release(fd)
    fd2 = acquire(lock, blocking=False)  # re-acquirable after release
    release(fd2)


def test_nonblocking_contention_raises_lockheld(holder):
    with pytest.raises(LockHeld):
        acquire(holder, blocking=False)


def test_lockheld_carries_holder_metadata(tmp_path):
    lock = tmp_path / "m.lock"
    with file_lock(lock, metadata={"pid": 123, "job": "sweep"}):
        assert read_holder(lock) == {"pid": 123, "job": "sweep"}
        with pytest.raises(LockHeld) as exc_info:
            with file_lock(lock, blocking=False):
                pass
        assert exc_info.value.holder == {"pid": 123, "job": "sweep"}
    # body truncated on release
    assert read_holder(lock) is None


def test_probe_held_fail_closed(tmp_path, holder):
    assert probe_held(holder) is True          # live holder -> held
    assert probe_held(tmp_path / "no.lock") is False  # absent marker -> not held
    free = tmp_path / "free.lock"
    free.touch()
    assert probe_held(free) is False           # present but unlocked -> not held


def test_verify_inode_detects_replacement(tmp_path, monkeypatch):
    lock = tmp_path / "v.lock"

    real_stat = os.stat

    def replaced_stat(p, *a, **kw):
        if Path(p) == lock:
            lock.unlink(missing_ok=True)
            lock.write_text("")  # new inode every check
        return real_stat(p, *a, **kw)

    monkeypatch.setattr(os, "stat", replaced_stat)
    with pytest.raises(LockReplaced):
        acquire(lock, verify_inode=True, retries=3)


def test_file_lock_on_oserror_proceed(tmp_path, monkeypatch):
    lock = tmp_path / "e.lock"

    def boom(*a, **kw):
        raise OSError("ENOLCK")

    monkeypatch.setattr("algua.primitives.flock.acquire", boom)
    entered = False
    with file_lock(lock, on_oserror="proceed"):
        entered = True
    assert entered  # degraded to unlocked, did not raise


def test_read_holder_garbled_body_is_none(tmp_path):
    lock = tmp_path / "g.lock"
    lock.write_text("not json")
    assert read_holder(lock) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/primitives/test_flock.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'algua.primitives'`.

- [ ] **Step 3: Implement `algua/primitives/flock.py`**

`algua/primitives/__init__.py`:
```python
"""Stdlib-only leaf primitives (spec §4). Imports NOTHING from algua — everything, including
other leaf modules (models/registry), may import this package. Enforced by import-linter."""
```

`algua/primitives/flock.py`:
```python
"""THE cross-process flock primitive (spec §4.1) — replaces 8 hand-rolled implementations.

Parameterized on exactly the axes those sites differed on: blocking vs LOCK_NB, OSError
policy (fail-closed default; explicit degrade-to-unlocked opt-in for best-effort curation),
optional JSON holder metadata in the lock body, and inode re-verification for lock files
that must never be replaced externally. flock is advisory and per-open-file-description:
`acquire` always opens a FRESH fd — a cached/shared fd would silently self-grant. The kernel
releases a flock on holder death (even a hard kill), so a crashed holder never wedges the
next acquire.
"""
from __future__ import annotations

import contextlib
import fcntl
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal


class LockHeld(Exception):
    """A non-blocking acquire found a live holder. `holder` carries the parsed lock-body
    metadata (None when the body is absent/garbled) so the caller can report who holds it."""

    def __init__(self, holder: dict | None = None) -> None:
        super().__init__("lock is held by another process")
        self.holder = holder


class LockReplaced(Exception):
    """The lock file was replaced externally while acquiring (verify_inode=True): the locked
    fd's inode no longer matches the path. The lock file contract says it must never be
    deleted; callers fail distinctly rather than proceed on a phantom lock."""


def acquire(
    path: Path, *, blocking: bool = True, verify_inode: bool = False, retries: int = 5
) -> int:
    """Open a FRESH fd on `path` (creating it 0o644 if absent) and take LOCK_EX. Returns the
    fd; the caller MUST `release(fd)` in a finally. `blocking=False` raises LockHeld on
    contention. `verify_inode=True` re-checks that the path still names the locked inode —
    a mismatch means something replaced the lock file externally; retry bounded, then raise
    LockReplaced."""
    attempts = retries if verify_inode else 1
    for _ in range(attempts):
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError as exc:
            os.close(fd)
            raise LockHeld(read_holder(path)) from exc
        except BaseException:
            os.close(fd)
            raise
        if not verify_inode:
            return fd
        fd_stat = os.fstat(fd)
        try:
            path_stat = os.stat(path)
        except FileNotFoundError:
            path_stat = None
        if path_stat is not None and (
            (path_stat.st_dev, path_stat.st_ino) == (fd_stat.st_dev, fd_stat.st_ino)
        ):
            return fd
        os.close(fd)
    raise LockReplaced(f"lock file {path} was replaced externally while acquiring")


def release(fd: int) -> None:
    """Unlock and close. Closing releases the flock even if LOCK_UN raises."""
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def read_holder(path: Path) -> dict | None:
    """Recover holder metadata from the lock-file body without taking the lock (flock is
    advisory, so a read needs no lock). None on a missing/empty/garbled body."""
    try:
        raw = path.read_text().strip()
    except OSError:
        return None
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


@contextmanager
def file_lock(
    path: Path,
    *,
    blocking: bool = True,
    metadata: dict | None = None,
    on_oserror: Literal["raise", "proceed"] = "raise",
) -> Iterator[None]:
    """Scoped LOCK_EX on `path`. With `metadata`, the holder identity is written into the
    lock body (fsync'd) on entry and truncated on exit, so a wedged holder is recoverable on
    contention via LockHeld.holder. `on_oserror="proceed"` degrades to UNLOCKED when the
    acquire itself fails with a non-contention OSError (exotic FS / ENOLCK) — ONLY for
    best-effort paths whose writes are individually atomic (the kb-sync curation case);
    everything else keeps the fail-closed default."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = acquire(path, blocking=blocking)
    except LockHeld:
        raise
    except OSError:
        if on_oserror == "proceed":
            yield
            return
        raise
    try:
        if metadata is not None:
            body = json.dumps(metadata).encode()
            os.ftruncate(fd, 0)
            os.pwrite(fd, body, 0)
            os.fsync(fd)
        yield
    finally:
        if metadata is not None:
            with contextlib.suppress(OSError):
                os.ftruncate(fd, 0)
        release(fd)


def probe_held(path: Path) -> bool:
    """True iff a live process holds the exclusive flock on `path`. Non-blocking. FAIL
    CLOSED: only a genuinely absent marker (FileNotFoundError) counts as not-held; any other
    open/lock error (ENOLCK, permission, unsupported flock, transient I/O) is treated as
    held, so a cleanup caller never deletes what it cannot prove is abandoned — leftover
    residue is recoverable, a deleted live write is not."""
    try:
        fd = os.open(path, os.O_RDWR | os.O_CLOEXEC)
    except FileNotFoundError:
        return False
    except OSError:
        return True
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return True
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)
```

- [ ] **Step 4: Add the leaf import-linter contract**

In `pyproject.toml`, next to the other contracts:
```toml
[[tool.importlinter.contracts]]
# primitives is the stdlib-only leaf (spec §4): anything may import it; it imports nothing
# from algua. This is what lets true leaves (models.registry) stop hand-duplicating helpers.
name = "primitives is a stdlib-only leaf"
type = "forbidden"
source_modules = ["algua.primitives"]
forbidden_modules = [
    "algua.audit", "algua.backtest", "algua.calendar", "algua.cli", "algua.config",
    "algua.contracts", "algua.data", "algua.execution", "algua.features", "algua.knowledge",
    "algua.live", "algua.models", "algua.observability", "algua.operator", "algua.portfolio",
    "algua.provenance", "algua.registry", "algua.research", "algua.risk", "algua.strategies",
    "algua.tracking",
]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/primitives/test_flock.py -v && uv run lint-imports`
Expected: all PASS.

- [ ] **Step 6: Full quality gate, then commit**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`

```bash
git add algua/primitives tests/primitives pyproject.toml
git commit -m "feat: algua/primitives/flock.py — the one cross-process lock primitive (spec §4.1)"
```

---

### Task 7: Migrate data-layer + models flock sites onto the primitive

**Files:**
- Modify: `algua/data/manifest.py` (`_acquire_lock`, `append_if_absent` release path), `algua/data/staging.py` (`_lock_held`, `new_leased_staging`, `release_leased_staging`), `algua/models/registry.py` (`_name_lease`)

**Interfaces:**
- Consumes: `acquire`, `release`, `probe_held`, `file_lock` from Task 6 (exact signatures above).
- Produces: no API change — all public functions keep their signatures and error types.

- [ ] **Step 1: `algua/data/manifest.py`**

Replace the body of `_acquire_lock` (keeping its docstring's fresh-fd + inode-verify rationale, now referencing the primitive):
```python
    def _acquire_lock(self) -> int:
        """Blocking LOCK_EX on the sidecar lock file via primitives.flock — a FRESH fd per
        call with inode re-verification (the lock file must never be replaced externally;
        see SnapshotManifest contract)."""
        lock_path = self.path.with_name(self.path.name + ".lock")
        try:
            return flock.acquire(lock_path, verify_inode=True, retries=_LOCK_ACQUIRE_RETRIES)
        except flock.LockReplaced as exc:
            raise ManifestLockReplacedError(
                f"lock file {lock_path} was replaced externally while acquiring; it must never "
                "be deleted (see SnapshotManifest contract)"
            ) from exc
```
Add `from algua.primitives import flock` to imports. In `append_if_absent`'s `finally` (currently `fcntl.flock(lock_fd, fcntl.LOCK_UN); os.close(lock_fd)`), replace with `flock.release(lock_fd)`. Remove the now-unused `import fcntl` if nothing else uses it.

- [ ] **Step 2: `algua/data/staging.py`**

- Replace `_lock_held`'s body with `return flock.probe_held(lock_path)`, KEEPING the full fail-closed docstring (move it onto the one-line wrapper — or delete the wrapper and call `flock.probe_held` directly at both call sites with the docstring's #255 reference as an inline comment; prefer the direct call, no wrapper cruft).
- In `new_leased_staging`, replace the `os.open` + `fcntl.flock` pair with `lock_fd = flock.acquire(lock_path)` (the `try/except BaseException` cleanup block stays, minus the manual `os.close` → `flock` already closed on raise inside `acquire`; the except block still unlinks the marker and rmtrees the dir but must NOT double-close: change it to not call `os.close(lock_fd)` — `lock_fd` is unbound if `acquire` raised).
- In `release_leased_staging`, replace the unlock/close pair with `flock.release(lock_fd)`.
- Remove `import fcntl` if now unused.

- [ ] **Step 3: `algua/models/registry.py`**

Replace `_name_lease` with:
```python
@contextmanager
def _name_lease(name_dir: Path) -> Iterator[None]:
    """Exclusive per-name flock on a sibling `<name>.lock`, serializing register()."""
    name_dir.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(name_dir.parent / f"{name_dir.name}.lock"):
        yield
```
Add `from algua.primitives.flock import file_lock`. Update the module docstring's leaf claim (lines ~24–26): the module remains a leaf; it now imports `algua.primitives` (itself a stdlib-only leaf) instead of inlining flock helpers. Keep `_fsync_file`/`_fsync_dir` for now (Task 10 removes them). Remove `import fcntl` if now unused.

- [ ] **Step 4: Quality gate (the existing data/models test suites are the behavior net)**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: all pass. If the `models.registry` leaf is enforced by an import-linter contract listing its allowed imports, add `algua.primitives` to that allowance.

- [ ] **Step 5: Commit**

```bash
git add -u algua
git commit -m "refactor: data manifest/staging + models registry locks onto primitives.flock"
```

---

### Task 8: Migrate operator, knowledge, and core-budget flock sites

**Files:**
- Modify: `algua/operator/gitops.py` (`merge_back_lock`), `algua/operator/schedule.py` (`operator_run_lock`, `_read_lock_holder`, `_marker_lock`), `algua/knowledge/sync.py` (`kb_sync_lock`), `algua/backtest/core_budget.py` (`admit`'s lease + admission acquires)

**Interfaces:**
- Consumes: Task 6 primitives. Produces: no public API change (`OperatorLockHeld`, `merge_back_lock`, `kb_sync_lock` signatures unchanged).

- [ ] **Step 1: `algua/operator/gitops.py`**

Replace `merge_back_lock`'s body (docstring stays, with the "mirrors staging" sentence replaced by "uses primitives.flock — the shared lock discipline"):
```python
@contextlib.contextmanager
def merge_back_lock(lock_path: Path) -> Iterator[None]:
    try:
        with file_lock(lock_path, blocking=False):
            yield
    except LockHeld as exc:
        raise RuntimeError("another merge-back cycle is in progress") from exc
```
Import `from algua.primitives.flock import LockHeld, file_lock`. Remove `import fcntl` if unused.

- [ ] **Step 2: `algua/operator/schedule.py`**

- `operator_run_lock` becomes:
```python
@contextmanager
def operator_run_lock(
    lock_path: Path, *, job: str, host: str, pid: int
) -> Iterator[None]:
    # (keep the existing docstring verbatim)
    metadata = {"pid": pid, "job": job, "started_at": datetime.now(UTC).isoformat(),
                "host": host}
    try:
        with file_lock(lock_path, blocking=False, metadata=metadata):
            yield
    except LockHeld as exc:
        raise OperatorLockHeld(exc.holder) from exc
```
- Delete `_read_lock_holder` and repoint its callers to `flock.read_holder` (grep: `grep -n "_read_lock_holder" algua`).
- `_marker_lock` becomes:
```python
    @contextmanager
    def _marker_lock(self) -> Iterator[None]:
        self._dir.mkdir(parents=True, exist_ok=True)
        with file_lock(self._lock_path):  # blocking — inner, on a distinct file/fd
            yield
```
- Remove `import fcntl` if unused.

- [ ] **Step 3: `algua/knowledge/sync.py`**

`kb_sync_lock` becomes (docstring stays verbatim — it documents WHY fail-open is safe here):
```python
@contextmanager
def kb_sync_lock(settings: Settings) -> Iterator[None]:
    settings.knowledge_dir.mkdir(parents=True, exist_ok=True)
    with file_lock(settings.knowledge_dir / ".sync.lock", on_oserror="proceed"):
        yield
```
Remove `import fcntl` if unused.

- [ ] **Step 4: `algua/backtest/core_budget.py` (partial adoption — the publish-under-lock protocol stays local)**

- In `admit`: the lease-fd acquire keeps its create-tmp-then-rename protocol but takes the lock via the primitive: replace `lease_fd = os.open(tmp_path, …)` + `fcntl.flock(lease_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)` with `lease_fd = flock.acquire(tmp_path, blocking=False)` (same O_CLOEXEC + fresh-fd semantics; the rename-preserves-fd comment stays).
- The admission lock: replace the `os.open` + blocking `fcntl.flock(admission_fd, LOCK_EX)` + `os.close` pattern with `admission_fd = flock.acquire(lease_dir / _ADMISSION_LOCK_NAME)` … `finally: flock.release(admission_fd)` (keep the "releases the admission flock" comment).
- `_sum_live_grants` keeps its bespoke three-way probe UNCHANGED (it reads the grant when held and reclaims orphans when free — richer than `probe_held`; its fail-closed comments stay).
- Keep `import fcntl` (still used by `_sum_live_grants`).

- [ ] **Step 5: Full quality gate**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: all pass — the operator/mergeback/kb-sync/core-budget suites are the behavior net.

- [ ] **Step 6: Commit**

```bash
git add -u algua
git commit -m "refactor: operator/knowledge/core-budget locks onto primitives.flock"
```

---

### Task 9: Create `algua/primitives/atomic_io.py` and move the `data/files.py` helpers (TDD)

**Files:**
- Create: `algua/primitives/atomic_io.py`
- Test: `tests/primitives/test_atomic_io.py`
- Modify: `algua/data/files.py` (remove moved helpers), `algua/data/manifest.py`, `algua/data/store.py`, `algua/data/verify.py`, `algua/cli/backtest_cmd.py`, `algua/tracking/mlflow_tracker.py`, and the test importers: `tests/test_data_store_publish.py`, `tests/test_data_durability.py`, `tests/test_data_files_series.py`, `tests/test_data_files_partitioned.py`, `tests/test_data_store.py`, `tests/test_determinism.py`

**Interfaces:**
- Produces (consumed by Task 10):
  - `fsync_file(path: Path) -> None`, `fsync_dir(path: Path) -> None`,
    `fsync_parents(path: Path, *, stop_at: Path) -> None`, `fsync_tree(root: Path) -> None`
  - `write_bytes_atomic(data: bytes, dest: Path) -> None` (atomic, NOT durable)
  - `write_text_atomic(text: str, dest: Path) -> None` (atomic, NOT durable — new, for kb-sync)
  - `write_bytes_durable(data: bytes, dest: Path, *, durable_root: Path | None = None) -> None`
    (atomic + fsync'd; with `durable_root` the parent chain up to it is fsync'd — this is
    `data/files.write_bytes_snapshot` generalized to any destination)

- [ ] **Step 1: Write the failing tests**

`tests/primitives/test_atomic_io.py`:
```python
"""algua.primitives.atomic_io — one atomic/durable write implementation (spec §4.2)."""
from __future__ import annotations

from pathlib import Path

import pytest

from algua.primitives.atomic_io import (
    fsync_dir,
    fsync_file,
    fsync_parents,
    fsync_tree,
    write_bytes_atomic,
    write_bytes_durable,
    write_text_atomic,
)


def test_write_bytes_atomic_creates_parent_and_no_temp_residue(tmp_path):
    dest = tmp_path / "sub" / "out.bin"
    write_bytes_atomic(b"payload", dest)
    assert dest.read_bytes() == b"payload"
    assert [p.name for p in dest.parent.iterdir()] == ["out.bin"]  # temp cleaned up


def test_write_text_atomic_roundtrip(tmp_path):
    dest = tmp_path / "note.md"
    write_text_atomic("hello", dest)
    write_text_atomic("world", dest)  # overwrite is atomic too
    assert dest.read_text() == "world"


def test_write_bytes_durable_with_root_chain(tmp_path):
    root = tmp_path / "store"
    dest = root / "a" / "b" / "x.bin"
    write_bytes_durable(b"d", dest, durable_root=root)
    assert dest.read_bytes() == b"d"


def test_fsync_parents_rejects_path_outside_root(tmp_path):
    inside = tmp_path / "root"
    inside.mkdir()
    outside = tmp_path / "elsewhere" / "f"
    outside.parent.mkdir()
    outside.touch()
    with pytest.raises(ValueError):
        fsync_parents(outside, stop_at=inside)


def test_fsync_helpers_run_on_real_objects(tmp_path):
    f = tmp_path / "f.txt"
    f.write_text("x")
    fsync_file(f)
    fsync_dir(tmp_path)
    fsync_tree(tmp_path)
    with pytest.raises(OSError):
        fsync_dir(f)  # O_DIRECTORY on a file fails loudly
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/primitives/test_atomic_io.py -v`
Expected: FAIL — no module `algua.primitives.atomic_io`.

- [ ] **Step 3: Implement by MOVING from `data/files.py`**

Create `algua/primitives/atomic_io.py` with a module docstring ("One atomic/durable write implementation (spec §4.2). Linux-only threat model: single local filesystem — see fsync notes on each helper.") and MOVE these functions verbatim from `algua/data/files.py` WITH their docstrings: `fsync_file`, `fsync_dir`, `fsync_parents`, `fsync_tree`, `write_bytes_atomic`. Then:
- Rename `write_bytes_snapshot(data, data_dir, relative_path)` → `write_bytes_durable(data, dest, *, durable_root=None)`: body identical except `target_path = dest`, and the final fsync is `fsync_parents(dest, stop_at=durable_root)` when `durable_root is not None` else `fsync_dir(dest.parent)`. Keep the #158/#184 docstring, adapted to the new signature.
- Add `write_text_atomic(text: str, dest: Path)` — same shape as `write_bytes_atomic` with `"w"` mode (this replaces `knowledge/sync._write_text_atomic` in Task 10).
- In `algua/data/files.py`: DELETE the moved functions (no re-exports). Keep `sha256_file`, `sha256_bytes`, `count_tabular_rows`, `frame_to_parquet_bytes`, `validate_partitioned_bars_dir`, and everything parquet/csv-specific.

- [ ] **Step 4: Update all importers of the moved names**

For each of `algua/data/manifest.py`, `algua/data/store.py`, `algua/data/verify.py`, `algua/cli/backtest_cmd.py`, `algua/tracking/mlflow_tracker.py` and the six test files listed above: change `from algua.data.files import fsync_dir, write_bytes_atomic, …` to import the moved names from `algua.primitives.atomic_io` (leaving genuinely-data-specific names imported from `algua.data.files`). Call sites of `write_bytes_snapshot(data, data_dir, rel)` become `write_bytes_durable(data, data_dir / rel, durable_root=data_dir)`. Find every call: `grep -rn "write_bytes_snapshot" algua tests`.

- [ ] **Step 5: Run tests to verify they pass, then full gate**

Run: `uv run pytest tests/primitives/test_atomic_io.py -v`
Then: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add algua/primitives/atomic_io.py tests/primitives/test_atomic_io.py
git add -u algua tests
git commit -m "feat: primitives/atomic_io — move fsync/atomic-write helpers out of data/files (spec §4.2)"
```

---

### Task 10: Replace the duplicate atomic-write implementations

**Files:**
- Modify: `algua/knowledge/sync.py` (`_write_text_atomic` + callers), `algua/operator/schedule.py` (`SessionMarker.record` write path), `algua/models/registry.py` (`_fsync_file`/`_fsync_dir` + publish path), `algua/data/manifest.py` (`_repair` fsync usage)

**Interfaces:**
- Consumes: Task 9's `write_text_atomic`, `write_bytes_durable`, `fsync_file`, `fsync_dir`.
- Produces: zero duplicate atomic-write implementations remain outside `primitives`.

- [ ] **Step 1: `algua/knowledge/sync.py`**

Delete `_write_text_atomic` entirely. Repoint every caller (grep `_write_text_atomic` in the file) to `write_text_atomic(text, path)` from `algua.primitives.atomic_io` — NOTE the argument order is `(text, dest)` on the primitive vs `(path, text)` on the deleted local; swap at each call site. The deleted function's docstring rationale (atomic so Obsidian/doctor/concurrent sync never see a half-written doc; NOT power-loss durable — regenerable curation) moves to a comment at the import or first call site.

- [ ] **Step 2: `algua/operator/schedule.py`**

In `SessionMarker.record`, replace the hand-rolled tmp→fsync→replace→fsync-dir block (from `payload = json.dumps(...)` through the final dir fsync) with:
```python
            payload = json.dumps(data, indent=2, sort_keys=True).encode("utf-8")
            write_bytes_durable(payload, self._path)
```
(`write_bytes_durable` without `durable_root` fsyncs the temp bytes and the parent dir — exactly what the deleted block did; the §D3 atomic+durable docstring on `record` stays.)

- [ ] **Step 3: `algua/models/registry.py`**

Delete `_fsync_file` and `_fsync_dir`; import `fsync_file, fsync_dir` from `algua.primitives.atomic_io` and repoint the publish path's calls (grep `_fsync_` in the file). Update the module docstring: the durability-primitives-inlined paragraph now reads that durability + lock primitives come from `algua.primitives` (a stdlib-only leaf, so the model layer remains a clean leaf under import-linter). Delete the `# durability primitives (inlined …)` section header if the section is now empty.

- [ ] **Step 4: `algua/data/manifest.py` `_repair`**

`_repair` KEEPS its temp-naming protocol (`_REPAIR_TEMP_SUFFIX` is scanned by `_clean_stale_repair_temps` — do not change the prefix/suffix scheme) and its explicit `tempfile.mkstemp` + `os.replace`; only ensure its fsync calls use the primitives' `fsync_dir` import (already repointed in Task 9). No further change — this step is a verification: `grep -n "fsync" algua/data/manifest.py` shows only `algua.primitives.atomic_io` imports.

- [ ] **Step 5: Full quality gate**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: all pass (kb-sync, operator schedule, model-registry suites are the net).

- [ ] **Step 6: Commit**

```bash
git add -u algua
git commit -m "refactor: replace duplicate atomic-write impls with primitives.atomic_io"
```

---

### Task 11: Create `algua/primitives/retry.py` and unify the two HTTP retry loops (TDD)

**Files:**
- Create: `algua/primitives/retry.py`
- Test: `tests/primitives/test_retry.py`
- Modify: `algua/execution/alpaca_broker.py` (`_AlpacaBroker._request`), `algua/data/providers/alpaca.py` (`_fetch_bars`)

**Interfaces:**
- Produces:
  - `call_with_backoff(send: Callable[[], T], *, attempts: int, backoff_base: float, retryable_exceptions: tuple[type[BaseException], ...], retry_result: Callable[[T], bool] = lambda _r: False, sleep: Callable[[float], None] = time.sleep) -> T`
  - `RetriesExhausted(Exception)` with `.attempts: int`, `.last_exception: BaseException`
  - Semantics (both sites' existing behavior): sleep `backoff_base * 2**i` between attempts; a retryable exception on the LAST attempt raises `RetriesExhausted`; a retryable RESULT on the last attempt is RETURNED for caller inspection.

- [ ] **Step 1: Write the failing tests**

`tests/primitives/test_retry.py`:
```python
"""algua.primitives.retry — one bounded-exponential-backoff loop (spec §4.3)."""
from __future__ import annotations

import pytest

from algua.primitives.retry import RetriesExhausted, call_with_backoff


class _Boom(Exception):
    pass


def test_returns_first_success_no_sleep():
    sleeps: list[float] = []
    result = call_with_backoff(
        lambda: "ok", attempts=3, backoff_base=0.5,
        retryable_exceptions=(_Boom,), sleep=sleeps.append,
    )
    assert result == "ok"
    assert sleeps == []


def test_retries_exceptions_with_exponential_schedule():
    sleeps: list[float] = []
    calls = {"n": 0}

    def send():
        calls["n"] += 1
        if calls["n"] < 3:
            raise _Boom(f"attempt {calls['n']}")
        return "ok"

    assert call_with_backoff(
        send, attempts=3, backoff_base=0.5,
        retryable_exceptions=(_Boom,), sleep=sleeps.append,
    ) == "ok"
    assert sleeps == [0.5, 1.0]  # base * 2**0, base * 2**1


def test_exhausted_exceptions_raise_with_last_exception():
    with pytest.raises(RetriesExhausted) as exc_info:
        call_with_backoff(
            lambda: (_ for _ in ()).throw(_Boom("always")), attempts=2, backoff_base=0.0,
            retryable_exceptions=(_Boom,), sleep=lambda _s: None,
        )
    assert exc_info.value.attempts == 2
    assert isinstance(exc_info.value.last_exception, _Boom)


def test_retryable_result_returned_on_last_attempt():
    results = iter([503, 503, 503])
    out = call_with_backoff(
        lambda: next(results), attempts=3, backoff_base=0.0,
        retryable_exceptions=(_Boom,), retry_result=lambda r: r == 503,
        sleep=lambda _s: None,
    )
    assert out == 503  # final response returned for caller inspection, not raised


def test_non_retryable_exception_propagates_immediately():
    with pytest.raises(ValueError):
        call_with_backoff(
            lambda: (_ for _ in ()).throw(ValueError("no")), attempts=3, backoff_base=0.0,
            retryable_exceptions=(_Boom,), sleep=lambda _s: None,
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/primitives/test_retry.py -v`
Expected: FAIL — no module.

- [ ] **Step 3: Implement `algua/primitives/retry.py`**

```python
"""One bounded-exponential-backoff retry loop (spec §4.3), unifying the two HTTP clones
(execution/alpaca_broker, data/providers/alpaca). The caller supplies error mapping and
retryability; safety-specific handling (e.g. the #394 redirect refusal) stays at the call
site — this module knows nothing about HTTP."""
from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


class RetriesExhausted(Exception):
    """Every attempt failed with a retryable exception. Carries the last one so the caller
    can wrap it in its domain error (BrokerError / ProviderError) with full context."""

    def __init__(self, attempts: int, last_exception: BaseException) -> None:
        super().__init__(f"failed after {attempts} attempts: {last_exception}")
        self.attempts = attempts
        self.last_exception = last_exception


def call_with_backoff(
    send: Callable[[], T],
    *,
    attempts: int,
    backoff_base: float,
    retryable_exceptions: tuple[type[BaseException], ...],
    retry_result: Callable[[T], bool] = lambda _r: False,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call `send` up to `attempts` times, sleeping `backoff_base * 2**i` between attempts.

    A retryable exception on the final attempt raises RetriesExhausted. A retryable RESULT
    (e.g. an HTTP 429/5xx response) on the final attempt is RETURNED — the caller inspects
    the final response and decides (both Alpaca sites' documented semantics). A
    non-retryable exception propagates immediately."""
    last_exc: BaseException | None = None
    for attempt in range(attempts):
        try:
            result = send()
        except retryable_exceptions as exc:
            last_exc = exc
        else:
            if not retry_result(result) or attempt == attempts - 1:
                return result
        if attempt < attempts - 1:
            sleep(backoff_base * (2**attempt))
    assert last_exc is not None  # result path always returned above on the final attempt
    raise RetriesExhausted(attempts, last_exc)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/primitives/test_retry.py -v`
Expected: PASS.

- [ ] **Step 5: Migrate `algua/execution/alpaca_broker.py` `_request`**

Keep `_TIMEOUT`, `_RETRYABLE_STATUS`, `_MAX_RETRIES`, `_BACKOFF_BASE` and the full #24/#394 docstrings/comments. The body becomes:
```python
        url = f"{self.base_url}{path}"

        def _send() -> requests.Response:
            # allow_redirects=False on every verb: requests re-sends the APCA credential
            # headers on a cross-host 3xx, which would leak them to the redirect target. A
            # 3xx is not retryable, so it returns below and the caller's non-2xx handling
            # (a BrokerError) rejects it — the credentials never reach the redirect host (#394).
            if method == "GET":
                return requests.get(url, headers=self._headers(), timeout=_TIMEOUT,
                                    allow_redirects=False)
            if method == "POST":
                return requests.post(url, headers=self._headers(), json=body, timeout=_TIMEOUT,
                                     allow_redirects=False)
            return requests.delete(url, headers=self._headers(), timeout=_TIMEOUT,
                                   allow_redirects=False)

        try:
            return call_with_backoff(
                _send, attempts=_MAX_RETRIES, backoff_base=_BACKOFF_BASE,
                retryable_exceptions=(RequestException,),
                retry_result=lambda resp: resp.status_code in retryable_status,
            )
        except RetriesExhausted as exc:
            raise BrokerError(
                f"alpaca {method} {path} failed after {_MAX_RETRIES} attempts: "
                f"{exc.last_exception}"
            ) from exc
```
Add `from algua.primitives.retry import RetriesExhausted, call_with_backoff`; drop `import time` if now unused.

- [ ] **Step 6: Migrate `algua/data/providers/alpaca.py` `_fetch_bars`**

Same pattern: extract the `requests.get(...)` call (with its #394 `allow_redirects=False` comment) into a local `_send`, then:
```python
        try:
            response = call_with_backoff(
                _send, attempts=MAX_ATTEMPTS, backoff_base=BACKOFF_BASE_SECONDS,
                retryable_exceptions=(requests.RequestException,),
                retry_result=lambda r: r.status_code in RETRYABLE_STATUS,
            )
        except RetriesExhausted as exc:
            raise ProviderError(
                f"alpaca request failed after {MAX_ATTEMPTS} attempts: {exc.last_exception}"
            ) from exc
```
The post-loop handling (redirect refusal, `raise_for_status` wrap, JSON parse/shape checks) runs ONCE on the returned final response — it keeps its exact current code and comments, de-indented out of the loop. Delete the manual `for attempt…` loop and the trailing `raise AssertionError("unreachable…")`. Drop `import time` if now unused.

- [ ] **Step 7: Full quality gate**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: all pass — the broker/provider retry tests are the behavior net (backoff schedule is identical: base × 2^i).

- [ ] **Step 8: Commit**

```bash
git add algua/primitives/retry.py tests/primitives/test_retry.py
git add -u algua
git commit -m "feat: primitives/retry — one backoff loop for both Alpaca HTTP sites (spec §4.3)"
```

---

### Task 12: Create `registry/challenges.py` and collapse the twin challenge stacks (TDD)

**Files:**
- Create: `algua/registry/challenges.py`
- Test: `tests/registry/test_challenges.py`
- Modify: `algua/registry/live_gate.py` (issue/find/consume/build become wrappers), `algua/registry/human_actor.py` (same)

**Interfaces:**
- Produces:
  - `ChallengeSpec(table: str, namespace: str, payload_fields: tuple[str, ...], column_fields: tuple[str, ...], ttl: timedelta)` — frozen dataclass. `payload_fields` is the ordered `k=v` line set of the signed payload; `column_fields` are the stored+matched DB columns (a subset of payload keys, in table column order).
  - `build_payload(spec, values: dict[str, object], nonce: str, expires_at: str) -> str` — `namespace\nk=v…\nnonce=…\nexpires_at=…` (byte-identical to both current builders).
  - `issue(conn, spec, values, *, now=None) -> dict[str, str]` (`{nonce, expires_at, challenge}`)
  - `find_pending(conn, spec, values, *, now=None) -> sqlite3.Row | None`
  - `consume(conn, spec, nonce, *, now=None) -> bool`
- Consumers: `live_gate.py` and `human_actor.py` keep their EXACT public signatures (`issue_challenge`, `find_pending_challenge`, `consume_challenge`, `build_challenge`, `issue_actor_challenge`, `find_pending_actor_challenge`, `consume_actor_challenge`, `build_actor_challenge`) as thin wrappers — zero call-site churn elsewhere, existing tests stay green.

- [ ] **Step 1: Write the failing tests**

`tests/registry/test_challenges.py`:
```python
"""registry.challenges — one challenge lifecycle for both signing namespaces (spec §4.4)."""
from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from algua.registry.challenges import ChallengeSpec, build_payload, consume, find_pending, issue

SPEC = ChallengeSpec(
    table="test_challenges",
    namespace="algua-test",
    payload_fields=("strategy", "strategy_id", "code_hash"),
    column_fields=("strategy_id", "code_hash"),
    ttl=timedelta(minutes=10),
)

VALUES = {"strategy": "momo", "strategy_id": 7, "code_hash": "abc"}


@pytest.fixture()
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.execute(
        "CREATE TABLE test_challenges(nonce TEXT PRIMARY KEY, strategy_id INTEGER,"
        " code_hash TEXT, issued_at TEXT, expires_at TEXT, consumed_at TEXT)"
    )
    return c


def test_payload_format_is_namespace_kv_nonce_expiry():
    payload = build_payload(SPEC, VALUES, "n0nce", "2026-01-01T00:00:00+00:00")
    assert payload == (
        "algua-test\nstrategy=momo\nstrategy_id=7\ncode_hash=abc\n"
        "nonce=n0nce\nexpires_at=2026-01-01T00:00:00+00:00"
    )


def test_issue_then_find_then_consume_single_use(conn):
    issued = issue(conn, SPEC, VALUES)
    assert set(issued) == {"nonce", "expires_at", "challenge"}
    row = find_pending(conn, SPEC, VALUES)
    assert row is not None and row["nonce"] == issued["nonce"]
    assert consume(conn, SPEC, issued["nonce"]) is True
    assert consume(conn, SPEC, issued["nonce"]) is False  # single-use
    assert find_pending(conn, SPEC, VALUES) is None       # consumed -> not pending


def test_find_pending_respects_expiry(conn):
    old = datetime.now(UTC) - timedelta(hours=1)
    issue(conn, SPEC, VALUES, now=old)
    assert find_pending(conn, SPEC, VALUES) is None  # expired


def test_find_pending_matches_null_column_with_is(conn):
    spec = ChallengeSpec(
        table="test_challenges", namespace="algua-test",
        payload_fields=("strategy", "strategy_id", "code_hash"),
        column_fields=("strategy_id", "code_hash"), ttl=timedelta(minutes=10),
    )
    values = {"strategy": "momo", "strategy_id": 7, "code_hash": None}
    issue(conn, spec, values)
    row = find_pending(conn, spec, values)
    assert row is not None  # NULL-valued bound column matches via IS
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/registry/test_challenges.py -v`
Expected: FAIL — no module.

- [ ] **Step 3: Implement `algua/registry/challenges.py`**

```python
"""One single-use signed-challenge lifecycle (spec §4.4) behind both signing namespaces.

`live_gate` (algua-go-live) and `human_actor` (algua-human-actor, #329) were line-for-line
parallel issue/find/consume stacks over their own tables; each is now a `ChallengeSpec` +
thin wrappers. The payload format (`namespace\\nk=v…\\nnonce=…\\nexpires_at=…`) is
byte-identical to the previous per-module builders, so existing enrolled keys and any
in-flight signed challenges verify unchanged. Signature verification itself stays in
`live_gate.verify_signature` — this module owns only nonce issuance, matching, and
single-use consumption. Column matching uses SQLite `IS` uniformly: identical to `=` for
non-NULL values, and NULL-correct for nullable identity columns (dependency_hash)."""
from __future__ import annotations

import secrets
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta


def _now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class ChallengeSpec:
    """One namespace's challenge shape: its table, the ordered payload lines the human
    signs, and the stored/matched DB columns (a subset of the payload keys)."""

    table: str
    namespace: str
    payload_fields: tuple[str, ...]
    column_fields: tuple[str, ...]
    ttl: timedelta = timedelta(minutes=10)


def build_payload(
    spec: ChallengeSpec, values: dict[str, object], nonce: str, expires_at: str
) -> str:
    """The exact bytes the operator signs. ONE definition used to both issue and verify so
    the two can never drift (each wrapper module passes the same spec+values to both)."""
    lines = [spec.namespace] + [f"{k}={values[k]}" for k in spec.payload_fields]
    lines += [f"nonce={nonce}", f"expires_at={expires_at}"]
    return "\n".join(lines)


def issue(
    conn: sqlite3.Connection,
    spec: ChallengeSpec,
    values: dict[str, object],
    *,
    now: datetime | None = None,
) -> dict[str, str]:
    """Create + persist a pending challenge; return {nonce, expires_at, challenge}."""
    now = now or _now()
    nonce = secrets.token_hex(32)
    expires_at = (now + spec.ttl).isoformat()
    cols = ", ".join(spec.column_fields)
    marks = ", ".join("?" for _ in spec.column_fields)
    conn.execute(
        f"INSERT INTO {spec.table}(nonce, {cols}, issued_at, expires_at, consumed_at)"
        f" VALUES (?, {marks}, ?, ?, NULL)",
        (nonce, *[values[k] for k in spec.column_fields], now.isoformat(), expires_at),
    )
    conn.commit()
    return {
        "nonce": nonce,
        "expires_at": expires_at,
        "challenge": build_payload(spec, values, nonce, expires_at),
    }


def find_pending(
    conn: sqlite3.Connection,
    spec: ChallengeSpec,
    values: dict[str, object],
    *,
    now: datetime | None = None,
) -> sqlite3.Row | None:
    """Newest unconsumed, unexpired challenge matching EVERY bound column."""
    now = now or _now()
    where = " AND ".join(f"{c} IS ?" for c in spec.column_fields)
    return conn.execute(
        f"SELECT * FROM {spec.table} WHERE {where} AND consumed_at IS NULL"
        f" AND expires_at > ? ORDER BY issued_at DESC LIMIT 1",
        (*[values[k] for k in spec.column_fields], now.isoformat()),
    ).fetchone()


def consume(
    conn: sqlite3.Connection, spec: ChallengeSpec, nonce: str, *, now: datetime | None = None
) -> bool:
    """Mark a challenge consumed (single-use). False if already consumed / missing — a lost
    consume race fails closed at the caller."""
    now = now or _now()
    cur = conn.execute(
        f"UPDATE {spec.table} SET consumed_at=? WHERE nonce=? AND consumed_at IS NULL",
        (now.isoformat(), nonce),
    )
    conn.commit()
    return cur.rowcount > 0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/registry/test_challenges.py -v`
Expected: PASS.

- [ ] **Step 5: Rewire `algua/registry/live_gate.py`**

Add at module level (below `_NAMESPACE`/`_TTL`, which it replaces):
```python
_SPEC = ChallengeSpec(
    table="live_challenges",
    namespace="algua-go-live",
    payload_fields=("strategy", "strategy_id", "code_hash", "config_hash", "dependency_hash"),
    column_fields=("strategy_id", "code_hash", "config_hash", "dependency_hash"),
)
```
Then make the four functions thin wrappers, keeping signatures + docstrings:
- `build_challenge(...)` → `return build_payload(_SPEC, {"strategy": strategy, "strategy_id": strategy_id, "code_hash": code_hash, "config_hash": config_hash, "dependency_hash": dependency_hash}, nonce, expires_at)`
- `issue_challenge(...)` → `return issue(conn, _SPEC, {…same dict…}, now=now)`
- `find_pending_challenge(...)` → `return find_pending(conn, _SPEC, {…the four column fields…}, now=now)`
- `consume_challenge(conn, nonce, *, now=None)` → `return consume(conn, _SPEC, nonce, now=now)`
Keep `_NAMESPACE = "algua-go-live"` (still referenced by `verify_signature`'s default) or repoint that default to `_SPEC.namespace`. `verify_signature`, `_assert_all_lines_namespace_scoped`, `verify_pending`, `ALLOWED_SIGNERS_PATH` are UNCHANGED. Delete the now-unused `secrets` import and `_TTL`.

- [ ] **Step 6: Rewire `algua/registry/human_actor.py`**

Same pattern:
```python
_SPEC = ChallengeSpec(
    table="actor_challenges",
    namespace="algua-human-actor",
    payload_fields=("command", "strategy", "strategy_id", "stage_from", "stage_to",
                    "code_hash", "config_hash", "dependency_hash", "run_context"),
    column_fields=("command", "strategy_id", "stage_from", "stage_to",
                   "code_hash", "config_hash", "dependency_hash", "run_context"),
)
```
`build_actor_challenge` / `issue_actor_challenge` / `find_pending_actor_challenge` / `consume_actor_challenge` become wrappers over `build_payload`/`issue`/`find_pending`/`consume` with the values dict assembled from their parameters. The module docstring (#329 threat model), `canonical_run_context`, `verify_actor_assertion`, `resolve_effective_actor`, `HumanActorChallengeRequired` are UNCHANGED. NOTE the column order of `actor_challenges` INSERT is `(nonce, command, strategy_id, stage_from, stage_to, code_hash, config_hash, dependency_hash, run_context, issued_at, expires_at, consumed_at)` — `column_fields` above matches it. Delete now-unused `secrets` import and `_TTL`.

- [ ] **Step 7: Byte-identity check on the payloads (regression tripwire)**

The existing live-gate and human-actor test suites sign/verify real challenges — run them explicitly:
`uv run pytest tests -k "live_gate or human_actor or go_live or actor" -q`
Expected: PASS with zero edits to those test files (payload bytes unchanged is the point).

- [ ] **Step 8: Full quality gate + commit**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`

```bash
git add algua/registry/challenges.py tests/registry/test_challenges.py
git add -u algua
git commit -m "feat: registry/challenges — one challenge lifecycle for go-live + human-actor (spec §4.4)"
```

---

### Task 13: Stage 0–2 close-out verification

**Files:**
- Modify: none expected (verification only; fix anything found)

- [ ] **Step 1: Grep-proof the unifications**

```bash
grep -rn "fcntl.flock" algua --include='*.py'
```
Expected: hits ONLY in `algua/primitives/flock.py` and `algua/backtest/core_budget.py::_sum_live_grants` (the documented bespoke probe).
```bash
grep -rn "tempfile.mkstemp\|os.replace" algua --include='*.py' | grep -v primitives
```
Expected: only `data/manifest.py::_repair` (documented protocol), `data/files.py` parquet-specific writers if any remain, and non-write-path uses. Anything else is a missed duplicate — migrate it.
```bash
grep -rn "2 \*\* attempt\|2\*\*attempt\|2 \*\* (attempt" algua --include='*.py' | grep -v primitives
```
Expected: no hits.

- [ ] **Step 2: Confirm the deletions left no dangling references**

```bash
grep -rn "algua.shadow\|algua.monitoring\|factor_eval\|factor_fdr\|lifecycle_gc\|family_audit\|cscv\|sweep_with_matrix\|family_budget" algua tests --include='*.py'
```
Expected: the only hits are `registry/db.py`'s v40/v41 `DROP TABLE IF EXISTS` lines + their comments (and historical comments elsewhere); any IMPORT of a deleted module is a bug — fix it.

- [ ] **Step 3: Full quality gate one last time**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: all pass.

- [ ] **Step 4: Smoke the CLI surface**

Run: `uv run algua version && uv run algua doctor; uv run algua fleet status`
Expected: `version`/`fleet status` exit 0 with JSON; `doctor` may exit non-zero on environment checks unrelated to this work (broker creds etc.) — acceptable; anything referencing a deleted command/table is not.

- [ ] **Step 5: Commit any fixes and record completion**

If steps 1–4 forced fixes, commit them (`git add -u … && git commit -m "fix: stage 0-2 close-out stragglers"`). Stage 3+ (data/store carve) is a separate plan, written after this one lands.
