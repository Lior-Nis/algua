# Stage 4c — `registry/db.py` Bounded-Context Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the 1088-line `algua/registry/db.py` — which holds `SCHEMA_VERSION`, a 660-line `_SCHEMA` DDL string (44 tables, 23 indexes, 9 triggers), `connect()`, a 197-line `migrate()`, and 5 migration helpers — into `algua/registry/db/`: one module per bounded context (DDL fragment + that context's migration helpers), plus a facade that keeps every existing import working unchanged. `migrate()` stays **one unsplit function**. Zero schema change: the assembled DDL must produce a byte-identical `sqlite_master`.

**Architecture:** Mirrors the packages Stages 3 and 4b already established (`algua/data/store/`, `algua/registry/store/`): a flat package, a facade `__init__.py` re-exporting the public surface at the same import path, and leaf modules that never import the facade back. Each context module exports a `SCHEMA` string fragment and owns the migration helpers that touch its own tables; `schema.py` concatenates the fragments; `migrate.py` holds the single ordered `migrate()` sequence verbatim. Fragmentation is safe because DDL order is provably irrelevant here (see Global Constraints) — the only rule is that a table's indexes and triggers stay in the same fragment as the table.

**Tech Stack:** Python 3.12, uv, pytest, ruff, mypy, import-linter, sqlite3 (3.45.1).

**Spec:** `docs/superpowers/specs/2026-08-18-system-simplification-design.md` §8 (`registry/db.py` bounded-context split).

**Ground truth this plan is written from:** a research pass against `main`@`7626f29` that read `db.py` in full and — critically — **verified its claims empirically rather than by inspection**: it executed 200 randomly-shuffled DDL-fragment orderings (all 200 produced an identical schema — 97 objects for the fragments alone; a full `migrate()` yields 100, and this plan's verification steps use that larger number), and it mutation-tested six reorderings of `migrate()` against the test suite. That mutation testing produced the single most important finding in this plan (see Task 1). The research also corrected several starting numbers this plan's author had assumed: 44 tables not 56 (12 `grep` hits were comment mentions), only 14 of 44 tables are registry-Protocol-owned (not "about half"), and `FdrGateOutcome` has 18 fields not 16.

**Three decisions the plan's author (not the implementer) made, recorded here so a reviewer can check the reasoning rather than guess at it:**

1. **The ordering guards are added FIRST, in their own task, before `migrate()` is touched.** Justification in Task 1.
2. **The program's stated rationale for "`migrate()` must stay one function" is factually wrong and is corrected by this plan.** Justification in Global Constraints.
3. **CODEOWNERS is deliberately OUT of scope**, and `FdrGateOutcome`'s dead fields stay deferred. Justification in "Deliberate scope exclusions" below.

## Global Constraints

- Quality gate on EVERY task before commit: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`. All four must pass.
- **`SCHEMA_VERSION` must NOT change (stays 41), and no DDL may be added, removed, or altered.** This is pure code motion. The objective proof is mandated in Task 2: a `sqlite_master` dump of a freshly-migrated DB must be **byte-identical** (including each object's stored `sql` text) before and after the carve. Because SQLite stores the original statement text, preserving each statement's exact characters and indentation when moving it into a fragment is what makes that check pass — so move statements verbatim, do not re-indent or reflow them.
- **`migrate()` stays ONE function, with its 28-step sequence and every inline rationale comment verbatim and in order.** The justification, corrected against the code:
  - The program's prior framing — *"`migrate()` can't be split because its ordering constraints cross bounded contexts"* — **is not supported by the code**. All four hard ordering constraints are *intra*-context: holdout ALTER→backfill (holdout), the gate DROP-INDEX→backfill→CREATE-INDEX→relabel chain (gate), the `attempt_token` ALTER→index (gate), and the family_members ALTER→trigger (family). The only genuinely cross-context coupling is the single `conn.executescript(_SCHEMA)` barrier, and that couples by *existence*, not by ordering.
  - The conclusion (keep it whole) is still correct, for a **stronger** reason: `migrate()` is the single auditable place where the whole ordered sequence — and the `executescript` barrier every later step depends on — is visible at once. The one historical production bug (the GATE-2 DROP-INDEX-before-backfill finding, whose rationale comment survives in the code) was an *intra*-gate-context reordering that a per-context split would **not** have prevented; splitting the orchestration would scatter the sequence without removing a single real constraint, and would make the global barrier implicit.
  - Do not restate the old cross-context claim anywhere in the new code or commit messages. A reviewer checking it against the code will find it inaccurate.
- **Intra-package import rule (prevents a real circular-import trap):** modules inside `algua/registry/db/` must import each other by **full submodule path** (`from algua.registry.db.core import SCHEMA as CORE_SCHEMA`), and must **never** import from `algua.registry.db` itself (the facade). The facade imports submodules; if a submodule imported the facade back, the package would deadlock on partial initialization at import time. The dependency graph is strictly one-way: `constants`/`_util`/`connection`/the 12 context modules are leaves → `schema` → `migrate` → `__init__`.
- **One deliberate text change is permitted and required** (everything else moves verbatim): two comments assert that *"SQLite validates a trigger's column references at CREATE time."* **This is false** on SQLite 3.45.1 — the research verified that `CREATE TRIGGER … WHEN NEW.nosuchcol IS NULL` succeeds and only errors when the trigger fires. The defensive ordering it justifies is harmless and stays; the stated reason must be corrected rather than propagated verbatim into new files. Exact replacement text is given in Task 2.
- `git add`/`git rm` is always scoped to the named files — never `git add -A`.
- Commits end with: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- Known pre-existing worktree hazard: some test writes a demo strategy file into `algua/strategies/momentum/`. If `git status` shows an untracked file there after running tests, delete it before staging — don't commit it.
- **Process rule — read this before running the test suite.** The full suite takes ~7-8 minutes (3447 tests). If you start it in the background you MUST actively poll its output in a loop with real tool calls. There is no notification that wakes a dispatched subagent when a background command finishes; ending your turn to "wait" for one stalls indefinitely until manually resumed. **Every implementer subagent across Stages 4a and 4b stalled this way despite this exact warning appearing in their prompt.** Actually poll.
- **No import-linter contract change is needed.** No contract in `pyproject.toml` names `algua.registry.db`; every registry contract uses the broad `algua.registry` prefix, and `db.py` → `db/` stays inside it. Same conclusion as Stages 3, 4a, and 4b.

### Deliberate scope exclusions (considered and rejected for this stage — do not add them)

- **CODEOWNERS.** Unlike Stage 4b (where an exact-file glob on `store.py` would have silently stopped matching), `algua/registry/db.py` **is not listed in `/CODEOWNERS` at all**, so the 4b failure mode cannot recur here and no CODEOWNERS edit is required for correctness. The research did surface that `db.py` owns integrity-critical surface (the `gate_evaluations` single-use-token DDL, the #524 append-only triggers, `FDR_COHORT_SIZE`, `SCHEMA_VERSION`) with no protection — a real gap. It is deliberately **not** closed here. `algua/operator/diff_policy.py`'s `_is_allowed` is a strict **allowlist** (only `kb/**` and `algua/strategies/<family>/**.py` ever pass), consulted *before* the CODEOWNERS-derived denylist is ever reached; `algua/registry/db/**` cannot match that allowlist, so it is already unmergeable by the autonomous merge-back loop with or without a CODEOWNERS entry — adding one is a **no-op** for the diff policy. Its only real effect would be GitHub PR review protection for human/agent pull requests, which is exactly the gap being discussed. The exclusion from *this stage* stands on narrower grounds: keeping 4c a pure structural carve with zero operational change, consistent with how Stages 4a and 4b were scoped. It should be closed in its own small operational PR.
- **`FdrGateOutcome`'s dead fields.** 15 of its 18 fields are constructed as literal `None`/`False` and never read (production reads only `.final_passed`; tests additionally read `.updated_rec` and `.gate_id`). Collapsing it requires editing `algua/registry/repository.py` and is an API change, not code motion. Deferred for the third consecutive stage, on the same reasoning Stage 4b's final reviewer explicitly endorsed repeating here.

---

### Task 1: Add the two missing `migrate()` ordering guards — BEFORE anything moves

**Files:**
- Modify: `tests/test_registry_db.py`

**Interfaces:**
- Produces: two new tests that fail if `migrate()`'s hard ordering constraints are violated. Nothing else in the codebase consumes them.
- Consumes: nothing (first task).

**Why this task exists and why it must run first.** The research mutation-tested six reorderings of `migrate()` against the existing 32-test suite. Only one — the historically-bitten GATE-2 reordering — was caught. In particular, moving `_backfill_holdout_intervals` before the `ALTER` that adds the column it reads is a **real, reproducible breakage** (`OperationalError: no such column: holdout_start` on a legacy-shaped DB) that **all 32 tests pass through silently**. The reason is precise: the only legacy-holdout test builds a DB where `holdout_evaluations` is *entirely absent*, so `executescript(_SCHEMA)` creates it complete with the column already present, and the ALTER→backfill ordering is never exercised. Task 2 relocates every one of these steps. Doing that with a net that catches one of four constraints is unacceptable, so the net is extended first, in its own commit, so it is provably green *before* the code moves.

- [ ] **Step 1: Read the existing guard for reference**

Read `tests/test_registry_db.py` in full, paying attention to `test_v33_backfills_legacy_binding_rows_into_cohorts` (~line 394). That test is the model: it *reconstructs the legacy shape* (drops the new composite index and restores the old global-unique one) so the ordering constraint is genuinely exercised rather than incidentally satisfied. Both new tests use the same technique.

Also read `test_migrate_adds_holdout_evaluations_to_legacy_db` (~line 274) to see the gap concretely: it asserts `"holdout_evaluations" not in ...` — i.e. the table is absent — which is exactly why it cannot catch this constraint.

- [ ] **Step 2: Add the holdout ALTER→backfill guard**

Add to `tests/test_registry_db.py` (place it directly after `test_migrate_adds_holdout_evaluations_to_legacy_db`, since it is the complementary case):

```python
def test_v23_backfill_runs_after_the_holdout_interval_alter(tmp_path):
    """ORDERING GUARD: the holdout_evaluations ALTER must precede _backfill_holdout_intervals.

    Reconstructs a genuinely pre-v22/v23-shaped holdout_evaluations — committed_at/holdout_start/
    holdout_end absent — so `CREATE TABLE IF NOT EXISTS` in _SCHEMA leaves the old shape intact and
    the ALTER in migrate() is the only thing that adds them. If the backfill were ever reordered
    before that ALTER, migrate() raises OperationalError('no such column: holdout_start') here.

    Contrast test_migrate_adds_holdout_evaluations_to_legacy_db, which omits the table entirely:
    _SCHEMA then creates it complete and the ordering is never exercised. That is why this guard
    exists separately.
    """
    conn = connect(tmp_path / "r.db")
    conn.executescript(
        """
        CREATE TABLE strategies (
            id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL UNIQUE,
            stage TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        CREATE TABLE holdout_evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_id INTEGER NOT NULL REFERENCES strategies(id),
            data_source TEXT NOT NULL,
            snapshot_id TEXT,
            period_start TEXT NOT NULL,
            period_end TEXT NOT NULL,
            holdout_frac REAL NOT NULL,
            config_hash TEXT NOT NULL,
            reused INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        );
        INSERT INTO strategies(name, stage, created_at, updated_at)
            VALUES ('s', 'idea', '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00');
        INSERT INTO holdout_evaluations(strategy_id, data_source, snapshot_id, period_start,
                                        period_end, holdout_frac, config_hash, reused, created_at)
            VALUES (1, 'SyntheticProvider', NULL, '2022-01-01', '2023-12-31', 0.2, 'cfg', 0,
                    '2026-01-02T00:00:00+00:00');
        """
    )
    conn.commit()
    # The precondition this guard depends on: the legacy row really is missing the v23 columns.
    assert "holdout_start" not in {
        r["name"] for r in conn.execute("PRAGMA table_info(holdout_evaluations)")
    }

    migrate(conn)

    row = conn.execute(
        "SELECT holdout_start, holdout_end FROM holdout_evaluations"
    ).fetchone()
    assert (row["holdout_start"], row["holdout_end"]) == ("2022-01-01", "2023-12-31")
```

- [ ] **Step 3: Add the `attempt_token` ALTER→index guard**

Add immediately after the previous test:

```python
def test_v36_attempt_token_index_created_after_its_column(tmp_path):
    """ORDERING GUARD: the gate_evaluations attempt_token ALTER must precede the
    ux_gate_evaluations_attempt_token index creation.

    Reconstructs a pre-v36-shaped gate_evaluations by removing the column (and its index) from a
    fully-migrated DB — the same "undo the new state to re-expose the old one" technique the GATE-2
    guard uses. If the CREATE UNIQUE INDEX were ever reordered before the ALTER, migrate() raises
    OperationalError('no such column: attempt_token') here.
    """
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    # DROP COLUMN refuses while the column is indexed, so the index goes first.
    conn.execute("DROP INDEX IF EXISTS ux_gate_evaluations_attempt_token")
    conn.execute("ALTER TABLE gate_evaluations DROP COLUMN attempt_token")
    conn.commit()
    assert "attempt_token" not in {
        r["name"] for r in conn.execute("PRAGMA table_info(gate_evaluations)")
    }

    migrate(conn)

    assert "attempt_token" in {
        r["name"] for r in conn.execute("PRAGMA table_info(gate_evaluations)")
    }
    assert conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='index'"
        " AND name='ux_gate_evaluations_attempt_token'"
    ).fetchone() is not None
```

- [ ] **Step 4: Prove both guards actually guard (mutation check)**

A test that passes whether or not the bug is present is worthless. Verify each new test genuinely fails when its constraint is violated, then restore. Do this **one mutation at a time**, and confirm `git diff algua/registry/db.py` is empty after each restore.

Mutation A — in `algua/registry/db.py`'s `migrate()`, move the `_backfill_holdout_intervals(conn)` call to immediately *before* the `_add_missing_columns(conn, "holdout_evaluations", {...})` call that precedes it. Then run:

```bash
uv run pytest tests/test_registry_db.py::test_v23_backfill_runs_after_the_holdout_interval_alter -q
```
Expected: **FAILS** with `sqlite3.OperationalError: no such column: holdout_start`. Restore `db.py` (`git checkout -- algua/registry/db.py`), confirm the test passes again.

Mutation B — move the `CREATE UNIQUE INDEX IF NOT EXISTS ux_gate_evaluations_attempt_token` statement to immediately *before* the `_add_missing_columns(conn, "gate_evaluations", {"attempt_token": "TEXT"})` call. Then run:

```bash
uv run pytest tests/test_registry_db.py::test_v36_attempt_token_index_created_after_its_column -q
```
Expected: **FAILS** with `sqlite3.OperationalError: no such column: attempt_token`. Restore `db.py`, confirm the test passes again.

Record both failure outputs verbatim in your report — they are this task's real deliverable, more than the passing run is. If either mutation does **not** produce a failure, stop and report BLOCKED: the guard does not guard, and Task 2 must not proceed on a false net.

- [ ] **Step 5: Full quality gate**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`

Expected: all four pass, with the test count up by exactly **2** from the pre-task baseline (3447 → 3449). Confirm `git diff algua/registry/db.py` is empty — this task must leave production code untouched. Check `git status` for the momentum-strategy hazard.

- [ ] **Step 6: Commit**

```bash
git add tests/test_registry_db.py
```

```bash
git commit -m "$(cat <<'EOF'
test: guard the two uncovered migrate() ordering constraints (stage 4c.1)

Mutation-testing migrate() against the existing suite showed only one of its four hard ordering
constraints is actually covered: the historically-bitten GATE-2 DROP-INDEX-before-backfill. In
particular, moving _backfill_holdout_intervals before the ALTER that adds the column it reads is a
reproducible OperationalError on a legacy-shaped DB that all 32 tests passed through silently --
the existing legacy-holdout test omits the table entirely, so _SCHEMA creates it complete and the
ordering is never exercised.

Adds two behavioural guards that reconstruct the legacy shape (the technique the GATE-2 guard
already uses): holdout ALTER->backfill, and the gate_evaluations attempt_token ALTER->unique-index.
Both were verified to FAIL under the corresponding reordering before being committed.

Lands before the stage-4c carve so the relocation of these steps happens over a net that covers
them, not one that silently does not.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Split `algua/registry/db.py` into `algua/registry/db/`

**Files:**
- Create: `algua/registry/db/__init__.py`, `constants.py`, `_util.py`, `connection.py`, `schema.py`, `migrate.py`, and 12 context modules: `core.py`, `breadth.py`, `holdout.py`, `gate.py`, `forward_gate.py`, `family.py`, `backtest_returns.py`, `authz.py`, `ideas.py`, `mergeback.py`, `knowledge.py`, `execution.py`
- Delete: `algua/registry/db.py`

**Interfaces:**
- Produces (all at the unchanged import path `algua.registry.db`): `connect`, `migrate`, `SCHEMA_VERSION`, `MAX_N_COMBOS`, `FDR_COHORT_SIZE`, `fdr_cohort_position`, and — **required despite being private** — `_add_missing_columns`, which `tests/test_db_migrations.py` imports directly. Omitting it from the facade breaks that file.
- Consumes: Task 1's guards (they must stay green throughout).

**Consumer safety, already verified:** production imports only 4 names from this module across 4 sites (`algua/cli/_common.py`, `store/family.py`, `store/search_breadth.py`, `store/gate.py`); tests import 6 names across 159 sites. `_SCHEMA` and `fdr_cohort_position` are never imported outside the module. There are **no** string-based `mock.patch("algua.registry.db…")` targets anywhere, so there is no import-path fragility. A facade re-exporting the 7 names above keeps all 163 call sites working unmodified.

- [ ] **Step 1: Capture the schema baseline — do this before editing anything**

This is the objective proof of zero schema change; it must be taken from the pre-carve tree.

```bash
uv run python -c "
import sqlite3, tempfile, pathlib, json
from algua.registry.db import connect, migrate
d = pathlib.Path(tempfile.mkdtemp())
conn = connect(d / 'baseline.db'); migrate(conn)
rows = conn.execute('SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY type, name').fetchall()
out = [dict(type=r['type'], name=r['name'], tbl_name=r['tbl_name'], sql=r['sql']) for r in rows]
pathlib.Path('/tmp/db_schema_baseline.json').write_text(json.dumps(out, indent=1))
print('objects:', len(out))
"
```
Expected: prints `objects: 100` (45 tables, 45 indexes, 10 triggers — i.e. `_SCHEMA`'s 44 tables + 23 indexes + 9 triggers, plus `sqlite_sequence`, plus the ~20 implicit `sqlite_autoindex_*` entries SQLite creates for PRIMARY KEY/UNIQUE constraints, plus the one trigger and two indexes that `migrate()` itself creates after the `executescript`). If it prints a different number, stop and report it — the plan's ground truth has drifted and the comparison in Step 11 would be meaningless.

- [ ] **Step 2: Read the source in full**

Read `algua/registry/db.py` (1088 lines) completely. Also read `algua/registry/store/__init__.py` and one of its leaf modules (e.g. `algua/registry/store/approvals.py`) — that package is the convention this one mirrors (module docstring naming the submodules, `X as X` re-exports, explicit `__all__`, facade owns composition).

**Move statements verbatim.** As in Stage 4b, prefer slicing the exact text out of the source programmatically over retyping it; the Step 11 schema comparison will catch any character that changes, so verbatim fidelity is machine-checked rather than trusted.

- [ ] **Step 3: Create `algua/registry/db/constants.py` (leaf)**

Holds the two cross-context constants. They live here rather than in the facade specifically so `migrate.py` can import them **without** importing the facade (which would be circular).

Move verbatim from `db.py`: the `SCHEMA_VERSION = 41` line together with its full preceding comment block (the "marker stamped into the DB's user_version, NOT a migration cursor" / "never bump this number without the migration that earns it" block, currently around lines 6-18), and `MAX_N_COMBOS = 1_000_000_000` with its comment (around line 25).

Add a short module docstring explaining these are schema-wide, not context-scoped. Keep the existing note that `MAX_N_COMBOS`'s value is **duplicated as a literal** inside the `search_trials` CHECK constraint (the DDL hard-codes `1000000000` rather than interpolating) — if that note does not already exist, add one line saying so, because after the split the two live in different files and the duplication becomes easier to miss.

- [ ] **Step 4: Create `algua/registry/db/_util.py` (leaf)**

Move `_add_missing_columns` verbatim (currently lines 1069-1088), with its docstring. This is the one genuinely context-agnostic utility — it is parameterised by table name and is called 13 times from `migrate()` across five different contexts, which is exactly why it has no context of its own.

- [ ] **Step 5: Create `algua/registry/db/connection.py` (leaf)**

Move `connect()` verbatim (currently lines 711-727), **including its full 9-line `recursive_triggers` comment block**. That comment documents a safety invariant with a deliberately narrowed scope — that `PRAGMA recursive_triggers=ON` is per-connection, so the #524 append-only guarantee holds only for connections opened through this helper and a raw `sqlite3.connect()` bypasses it. It must survive the move word-for-word.

Note for your own understanding (do not change the behaviour): `connect()` does **not** call `migrate()`; callers pair them.

- [ ] **Step 6: Create the 12 context modules**

Each module gets: a module docstring naming the context and its owning code, and a module-level `SCHEMA` string containing that context's DDL statements moved **verbatim** (same text, same indentation, including the comment lines interleaved between statements). Where a context has migration helpers, they live in the same module, below the `SCHEMA` string.

**The one fragmentation rule** (everything else is provably order-free — see Step 7's docstring): *a table's `CREATE INDEX` and `CREATE TRIGGER` statements must be in the same fragment as its `CREATE TABLE`, and after it.* Cross-context `REFERENCES` are **not** a constraint: SQLite resolves FK targets at DML time, not DDL time.

Table-to-module assignment (assign by table **name**; move each table's DDL together with its adjacent comment block, its indexes, and its triggers):

| Module | Tables | Helpers moved here |
|---|---|---|
| `core.py` | `strategies`, `stage_transitions`, `approvals` | `_migrate_shortlisted_to_candidate` (currently 929-964) |
| `breadth.py` | `search_trials` | — |
| `holdout.py` | `holdout_evaluations`, `holdout_returns` | `_backfill_holdout_intervals` (967-985) |
| `gate.py` | `gate_evaluations` | `FDR_COHORT_SIZE`, `fdr_cohort_position` (34-47), `_backfill_fdr_cohorts` (988-1028), `_relabel_fdr_cohorts_for_current_size` (1031-1066) |
| `forward_gate.py` | `forward_gate_evaluations` | — |
| `family.py` | `families`, `family_members`, `family_parents`, `family_events` + their append-only triggers | — |
| `backtest_returns.py` | `backtest_returns` + its two append-only triggers | — |
| `authz.py` | `live_challenges`, `actor_challenges`, `live_authorizations`, `strategy_allocations` | — |
| `ideas.py` | `ideas` | — |
| `mergeback.py` | `mergeback_evidence` | — |
| `knowledge.py` | `audit_log`, `negative_results` | — |
| `execution.py` | the 23 operational-lane tables: `paper_orders`, `paper_fills`, `kill_switches`, `strategy_peaks`, `book_equity_peak`, `tick_snapshots`, `global_halt`, `live_orders`, `live_fills`, `live_activities`, `live_fill_cursor`, `live_activity_quarantine`, `paper_venue_orders`, `paper_venue_fills`, `paper_venue_activities`, `paper_venue_fill_cursor`, `paper_venue_activity_quarantine`, `live_reconcile_state`, `live_cycle`, `paper_reconcile_state`, `paper_cycle`, `live_nav_peaks`, `live_reservations` | — |

This assignment was verified programmatically against the live `_SCHEMA` when this plan was written: 44 tables total, every named table exists in the DDL, no table is assigned twice, and the 23 unnamed remainder are exactly `execution.py`'s. Re-run that check yourself before writing any files — it catches a drifted assignment in seconds, whereas Step 11's schema comparison catches it only after all 12 modules exist:

```bash
uv run python -c "
import re, pathlib
body = re.search(r'_SCHEMA = \"\"\"(.*?)\"\"\"', pathlib.Path('algua/registry/db.py').read_text(), re.S).group(1)
tables = re.findall(r'CREATE TABLE IF NOT EXISTS (\w+)', body)
named = '''strategies stage_transitions approvals search_trials holdout_evaluations holdout_returns
gate_evaluations forward_gate_evaluations families family_members family_parents family_events
backtest_returns live_challenges actor_challenges live_authorizations strategy_allocations ideas
mergeback_evidence audit_log negative_results'''.split()
assert len(tables) == 44, len(tables)
assert not [t for t in named if t not in tables], 'assigned table missing from DDL'
assert len(set(named)) == len(named), 'table assigned twice'
print('OK: 21 explicitly assigned +', len([t for t in tables if t not in named]), 'to execution.py')
"
```
Expected: `OK: 21 explicitly assigned + 23 to execution.py`.

Notes the implementer must honour:

- `gate.py` and `forward_gate.py` are split (rather than combined) to give a 1:1 correspondence with `algua/registry/store/gate.py` and `store/forward_gate.py` from Stage 4b — navigating "the DDL for what `store/gate.py` operates on" should not require knowing they were merged.
- `families.founder_gate_id` has `REFERENCES gate_evaluations(id)` — a genuine cross-module logical coupling. Add a one-line comment in **both** `family.py` and `gate.py` noting it, so a future maintainer splitting further does not separate them carelessly. This is an addition, but a comment-only one.
- `_backfill_holdout_intervals` **raises `RuntimeError`** if any NULL-interval row survives (it refuses to stamp the version). It is the only helper that can hard-fail a bootstrap. Preserve that behaviour and its comment exactly.
- Each helper that carries an ordering precondition must state it in its docstring, since after the split the ALTER it depends on lives in a different file. `_relabel_fdr_cohorts_for_current_size` already does this; make it uniform. Specifically, add to `_backfill_holdout_intervals`' docstring a line reading: `MUST run after migrate()'s holdout_evaluations ALTER (it reads holdout_start); see the ordering guard in tests/test_registry_db.py.` and to `_backfill_fdr_cohorts`: `MUST run after the legacy global-unique fdr index is dropped in migrate() (GATE-2); see migrate() for the rationale.`
- **`family.py` carries the one required text correction.** The `_SCHEMA` comment immediately above the family triggers currently reads:

```
-- NB: trg_family_members_append_only_upd references the v37 member_code_hash/member_factors_json
-- columns, which do NOT exist on a legacy family_members table at executescript(_SCHEMA) time
-- (they are added by ALTER in migrate() afterwards). SQLite validates a trigger's column
-- references at CREATE time, so that trigger is created in migrate() AFTER the ALTER, not here.
```
Replace the last sentence, keeping the rest:
```
-- NB: trg_family_members_append_only_upd references the v37 member_code_hash/member_factors_json
-- columns, which do NOT exist on a legacy family_members table at executescript(_SCHEMA) time
-- (they are added by ALTER in migrate() afterwards). That trigger is therefore created in
-- migrate() AFTER the ALTER, not here -- defensively: SQLite actually resolves a trigger's column
-- references when the trigger FIRES, not at CREATE time (verified on 3.45.1), so creating it early
-- would not raise here, but it would leave a window in which the trigger is unfireable. Keeping
-- creation after the ALTER keeps the guarantee unconditional.
```

- [ ] **Step 7: Create `algua/registry/db/schema.py`**

Concatenates the 12 fragments into the `SCHEMA` string `migrate()` executes. Import each fragment by **full submodule path** (never `from algua.registry.db import core`):

```python
"""Assembles the full registry DDL from the per-context fragments.

The concatenation order below is chosen for readability only -- it is provably irrelevant to the
resulting schema. Verified empirically: 200 randomly-shuffled fragment orders all produced an
identical sqlite_master (97 objects for the fragments alone, before the rest of migrate() adds its
own trigger and two indexes). SQLite resolves ``REFERENCES`` targets at DML time, not DDL
time, so cross-context foreign keys impose no ordering; the only real rule is that a table's
indexes and triggers live in the same fragment as the table, which each fragment satisfies.

Caveat for future changes: that independence holds because every fragment is executed inside one
``executescript`` before any DML. If a future change ever interleaves DML between fragments, the
foreign-key deferral no longer saves you.
"""
from __future__ import annotations

from algua.registry.db.authz import SCHEMA as AUTHZ_SCHEMA
from algua.registry.db.backtest_returns import SCHEMA as BACKTEST_RETURNS_SCHEMA
from algua.registry.db.breadth import SCHEMA as BREADTH_SCHEMA
from algua.registry.db.core import SCHEMA as CORE_SCHEMA
from algua.registry.db.execution import SCHEMA as EXECUTION_SCHEMA
from algua.registry.db.family import SCHEMA as FAMILY_SCHEMA
from algua.registry.db.forward_gate import SCHEMA as FORWARD_GATE_SCHEMA
from algua.registry.db.gate import SCHEMA as GATE_SCHEMA
from algua.registry.db.holdout import SCHEMA as HOLDOUT_SCHEMA
from algua.registry.db.ideas import SCHEMA as IDEAS_SCHEMA
from algua.registry.db.knowledge import SCHEMA as KNOWLEDGE_SCHEMA
from algua.registry.db.mergeback import SCHEMA as MERGEBACK_SCHEMA

SCHEMA = "\n".join([
    CORE_SCHEMA,
    BREADTH_SCHEMA,
    HOLDOUT_SCHEMA,
    GATE_SCHEMA,
    FORWARD_GATE_SCHEMA,
    FAMILY_SCHEMA,
    BACKTEST_RETURNS_SCHEMA,
    AUTHZ_SCHEMA,
    IDEAS_SCHEMA,
    MERGEBACK_SCHEMA,
    KNOWLEDGE_SCHEMA,
    EXECUTION_SCHEMA,
])
```

- [ ] **Step 8: Create `algua/registry/db/migrate.py`**

Holds `migrate()` — **one function, unsplit**, with its full 28-step sequence, its docstring, and every inline rationale comment verbatim and in the original order. Move it as a single block (currently lines 730-926). The only changes are its imports (it now pulls `SCHEMA` from `schema.py`, `SCHEMA_VERSION` from `constants.py`, `_add_missing_columns` from `_util.py`, and the four context helpers from their new modules) and the substitution of `_SCHEMA` → `SCHEMA` at the `executescript` call.

Give the module a docstring stating why the function is not split, in the corrected terms from Global Constraints — single-point auditability, the `executescript` barrier every later step depends on, and the fact that the one historical production ordering bug was intra-context so a per-context split would not have prevented it. Do **not** repeat the inaccurate cross-context claim.

Preserve verbatim, in particular:
- the docstring's statement that `migrate()` deliberately does **not** gate on `user_version` ("doing so would falsely imply migration history and could skip needed table creation on a pre-stamped DB; we only stamp it afterward as a schema-generation marker");
- the GATE-2 rationale block above the `DROP INDEX` (the "DROP the old global-unique index BEFORE the backfill … the 65th row's fdr_test_index … would collide … (GATE-2 finding)" comment) — this is the load-bearing record of a real production bug;
- the two `DROP TABLE IF EXISTS` statements for `shadow_evaluations` and `factor_evaluations` (retired in an earlier stage of this program);
- the `attempt_token` comment explaining why its index is created in `migrate()` rather than `_SCHEMA`;
- the per-version narration comments (v21…v41) throughout.

The family-trigger comment inside `migrate()` makes the same false SQLite claim corrected in Step 6. Apply the equivalent correction here: keep the "created HERE (after the ALTER), not in _SCHEMA" instruction and its #524 explanation, but replace the parenthetical reason *"(where a legacy family_members table would not yet have those columns and SQLite would reject the trigger's column references)"* with *"(where a legacy family_members table would not yet have those columns — SQLite would not reject the CREATE, since it resolves trigger column references at fire time, but the trigger would be unfireable until the ALTER landed)"*.

- [ ] **Step 9: Create `algua/registry/db/__init__.py` (the facade)**

Re-exports the public surface so all 163 existing call sites keep working unchanged. Match `algua/registry/store/__init__.py`'s conventions (module docstring naming the submodules, `X as X` re-export form, explicit `__all__`).

```python
"""sqlite registry schema + bootstrap, split by bounded context (spec §8).

Each context owns its DDL fragment and its own migration helpers (core, breadth, holdout, gate,
forward_gate, family, backtest_returns, authz, ideas, mergeback, knowledge, execution);
``schema.py`` concatenates the fragments; ``migrate.py`` holds the single ordered bootstrap
sequence; ``connection.py`` owns ``connect()`` and its pragmas; ``constants.py`` holds the
schema-wide constants; ``_util.py`` holds the one context-agnostic ALTER helper.

Submodules import each other by full path and never import this facade -- the dependency graph is
one-way (leaves -> schema -> migrate -> here), which is what keeps the package importable.
"""
from __future__ import annotations

from algua.registry.db._util import _add_missing_columns as _add_missing_columns
from algua.registry.db.connection import connect as connect
from algua.registry.db.constants import (
    MAX_N_COMBOS as MAX_N_COMBOS,
    SCHEMA_VERSION as SCHEMA_VERSION,
)
from algua.registry.db.gate import (
    FDR_COHORT_SIZE as FDR_COHORT_SIZE,
    fdr_cohort_position as fdr_cohort_position,
)
from algua.registry.db.migrate import migrate as migrate

__all__ = [
    "FDR_COHORT_SIZE",
    "MAX_N_COMBOS",
    "SCHEMA_VERSION",
    "connect",
    "fdr_cohort_position",
    "migrate",
]
```

`_add_missing_columns` is deliberately re-exported despite being private (`tests/test_db_migrations.py` imports it directly) and is deliberately **absent** from `__all__` — it is not public API, it is a compatibility re-export. Add a one-line comment at its import saying exactly that, so a future cleanup does not delete it as unused.

Let ruff/isort settle the final import formatting; do not fight it.

- [ ] **Step 10: Delete the old module**

```bash
git rm algua/registry/db.py
```

- [ ] **Step 11: Verify the assembled schema is byte-identical**

The objective zero-change proof:

```bash
uv run python -c "
import pathlib, tempfile, json
from algua.registry.db import connect, migrate
d = pathlib.Path(tempfile.mkdtemp())
conn = connect(d / 'after.db'); migrate(conn)
rows = conn.execute('SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY type, name').fetchall()
after = [dict(type=r['type'], name=r['name'], tbl_name=r['tbl_name'], sql=r['sql']) for r in rows]
before = json.loads(pathlib.Path('/tmp/db_schema_baseline.json').read_text())
assert len(after) == len(before), f'object count {len(before)} -> {len(after)}'
diffs = [(b, a) for b, a in zip(before, after) if b != a]
assert not diffs, f'{len(diffs)} object(s) differ; first: {diffs[0]}'
print('schema identical:', len(after), 'objects')
"
```
Expected: `schema identical: 100 objects` (the same count Step 1 printed). Any difference — an object missing, renamed, or whose stored `sql` text changed — means a statement was dropped, duplicated, or re-indented during the move. Fix it rather than adjusting the check.

Also confirm the migration path (not just the fresh-bootstrap path) still works: `uv run pytest tests/test_registry_db.py tests/test_db_migrations.py -q` must pass, **including the two guards from Task 1**.

- [ ] **Step 12: Full quality gate**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`

Expected: all four pass with the **same** test count as Task 1's ending count (3449) — this task moves code, it adds and removes no tests. Check `git status` for the momentum-strategy hazard.

- [ ] **Step 13: Commit**

```bash
git add algua/registry/db/
```

```bash
git commit -m "$(cat <<'EOF'
refactor: split registry/db.py into registry/db/ by bounded context (stage 4c)

Splits the 1088-line module into a package: one module per bounded context (core, breadth,
holdout, gate, forward_gate, family, backtest_returns, authz, ideas, mergeback, knowledge,
execution), each owning its DDL fragment and its own migration helpers; schema.py concatenates the
fragments; connection.py owns connect(); constants.py holds SCHEMA_VERSION/MAX_N_COMBOS; _util.py
holds the one context-agnostic ALTER helper; __init__.py re-exports the public surface at the
unchanged import path (plus _add_missing_columns, which a test imports directly).

migrate() stays ONE unsplit function with its ordered sequence and rationale comments verbatim.
Note the justification is not the one the program previously assumed: all four hard ordering
constraints are INTRA-context, and the only cross-context coupling is the single
executescript(SCHEMA) barrier. Keeping it whole is still right, for a stronger reason -- it is the
single auditable place the whole sequence is visible, and the one historical production ordering
bug (GATE-2) was intra-context, so a per-context split would not have prevented it.

Zero schema change, proved rather than asserted: a sqlite_master dump of a freshly-migrated DB is
byte-identical (97 objects, including each object's stored sql text) before and after. Fragment
concatenation order is provably irrelevant -- 200 shuffled orders all yield the identical schema.

Also corrects a factually wrong comment the move would otherwise have propagated verbatim: SQLite
resolves a trigger's column references when the trigger fires, not at CREATE time (verified on
3.45.1). The defensive ordering it justified is kept; only the stated reason is fixed.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Converge the `data/store` stub idiom

**Files:**
- Modify: `algua/data/store/bars.py`, `algua/data/store/fundamentals.py`, `algua/data/store/news.py`

**Interfaces:**
- Produces / Consumes: nothing. Purely an idiom change; no behaviour, no public surface.

**Why this is here.** Stage 4b established `if TYPE_CHECKING:`-guarded stubs as the codebase's pattern for a mixin method whose real implementation lives in another module, after runtime `raise NotImplementedError` stubs caused a real MRO-shadowing bug that broke 98 tests. Three runtime stubs remain in `algua/data/store/` — all three the same method (`get_snapshot`), all three pointing at the `DataStore` facade. **They are currently safe**, because a facade's own class body always wins the MRO; this is idiom convergence, not a bug fix. Stage 4b's final reviewer recommended folding it in here precisely because the divergence is undocumented and a future maintainer copying the older pattern into a sibling-mixin context is exactly how the 4b break happened. It is a separate task with its own commit so the carve's diff stays pure.

- [ ] **Step 1: Read the in-package precedent**

Read `algua/data/store/delistings.py` — it already uses `from typing import TYPE_CHECKING` with an `if TYPE_CHECKING:` block. Match its exact style rather than inventing one.

- [ ] **Step 2: Convert the three stubs**

In each of `bars.py` (~lines 30-32), `fundamentals.py` (~24-26), and `news.py` (~25-27), the stub currently reads (modulo the comment wording, which differs slightly per file — preserve each file's own):

```python
    def get_snapshot(self, snapshot_id: str) -> SnapshotRecord:
        # provided by the DataStore facade (store/__init__.py); stub for mypy only
        raise NotImplementedError
```

Convert to a `TYPE_CHECKING`-guarded declaration, adding the `TYPE_CHECKING` import where the file lacks it:

```python
    if TYPE_CHECKING:  # provided by the DataStore facade (store/__init__.py); mypy-only declaration
        def get_snapshot(self, snapshot_id: str) -> SnapshotRecord: ...
```

Keep each file's existing comment wording; only its placement moves. Verify the real signature still matches `DataStore.get_snapshot` in `algua/data/store/__init__.py` — mypy will catch a mismatch, but check rather than assume.

- [ ] **Step 3: Verify no runtime leakage**

```bash
uv run python -c "
from algua.data.store import DataStore
from algua.data.store.bars import BarsStoreMixin
from algua.data.store.fundamentals import FundamentalsStoreMixin
from algua.data.store.news import NewsStoreMixin
for cls in (BarsStoreMixin, FundamentalsStoreMixin, NewsStoreMixin):
    assert 'get_snapshot' not in vars(cls), cls
owner = next(c for c in DataStore.__mro__ if 'get_snapshot' in vars(c))
print('OK: no stub leaks at runtime; get_snapshot resolves to', owner.__name__)
"
```
Expected: `OK: no stub leaks at runtime; get_snapshot resolves to DataStore`.

- [ ] **Step 4: Full quality gate**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`

Expected: all four pass, test count unchanged from Task 2 (3449). mypy passing is the meaningful signal here — it proves the declarations still type-check the cross-module calls.

- [ ] **Step 5: Commit**

```bash
git add algua/data/store/bars.py algua/data/store/fundamentals.py algua/data/store/news.py
```

```bash
git commit -m "$(cat <<'EOF'
refactor: converge data/store on the TYPE_CHECKING stub idiom (stage 4c)

Stage 4b established if TYPE_CHECKING:-guarded stubs as the pattern for a mixin method implemented
in another module, after runtime `raise NotImplementedError` stubs shadowed a sibling mixin's real
implementations and broke 98 tests. These three (all the same get_snapshot, all pointing at the
DataStore facade) were safe as written -- a facade's own class body always wins the MRO -- so this
is idiom convergence, not a bug fix. Converting them leaves one pattern in the codebase instead of
a live, undocumented divergence that a future maintainer could copy into a context where it is not
safe, which is exactly how the 4b break happened.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Close-out verification

**Files:** none expected (verification only; fix anything found)

- [ ] **Step 1: No stale references to the old module layout**

```bash
grep -rn "registry/db\.py\|_SCHEMA" algua/ tests/ docs/agent/ AGENTS.md CLAUDE.md .codex/ --include='*.py' --include='*.md' --include='*.toml' 2>/dev/null
```
Read every hit. `_SCHEMA` should now appear only as the per-fragment `SCHEMA` name inside the new package (the old private name should be gone). Any prose reference to `algua/registry/db.py` as a file path is now stale and should be updated to `algua/registry/db/` — this is the same class of drift Stage 4b had to fix in four files after the fact, so catch it here instead. Anything under `docs/superpowers/` is historical plan text and must be left alone.

- [ ] **Step 2: The public surface resolves**

```bash
uv run python -c "
import algua.registry.db as db
from algua.registry.db import (
    SCHEMA_VERSION, MAX_N_COMBOS, FDR_COHORT_SIZE, connect, migrate, fdr_cohort_position,
)
from algua.registry.db import _add_missing_columns
assert SCHEMA_VERSION == 41, SCHEMA_VERSION
assert MAX_N_COMBOS == 1_000_000_000
assert FDR_COHORT_SIZE == 8
assert fdr_cohort_position(9) == (1, 1)
print('OK', sorted(db.__all__))
"
```
Expected: exits 0 and prints the six public names. `SCHEMA_VERSION` must still be 41 — a changed value means DDL was altered, which this stage forbids.

- [ ] **Step 3: No import cycles, and no submodule imports the facade**

```bash
grep -rn "^from algua\.registry\.db import\|^import algua\.registry\.db$" algua/registry/db/*.py
```
Expected: **no hits**. Every intra-package import must use a full submodule path (`from algua.registry.db.core import ...`). A hit here is the circular-import trap described in Global Constraints; fix it before proceeding.

Then confirm the package imports cleanly from a cold interpreter in both orders (facade-first and submodule-first), since a cycle can hide behind import order:

```bash
uv run python -c "import algua.registry.db; print('facade-first OK')"
uv run python -c "import algua.registry.db.migrate; import algua.registry.db; print('submodule-first OK')"
```

- [ ] **Step 4: The ordering guards still guard after the move**

The whole point of Task 1 was to protect Task 2's relocation, so re-prove it on the moved code. Re-run **Mutation A** from Task 1 Step 4 — but now against the relocated `migrate()` in `algua/registry/db/migrate.py`: move the `_backfill_holdout_intervals(conn)` call before its `_add_missing_columns` call, run

```bash
uv run pytest tests/test_registry_db.py::test_v23_backfill_runs_after_the_holdout_interval_alter -q
```

confirm it **FAILS**, then restore (`git checkout -- algua/registry/db/migrate.py`) and confirm it passes. Record the output. If it now passes under mutation, the guard was silently defeated by the move and that is a Critical finding — stop and report it.

- [ ] **Step 5: Full quality gate**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`

Expected: all four pass. Test count is 3449 — exactly +2 versus this plan's pre-Task-1 baseline of 3447, all of it Task 1's two guards. Confirm `git status` is clean (watch for the momentum-strategy hazard and for a stray `/tmp/db_schema_baseline.json`, which lives outside the repo and should not appear).

- [ ] **Step 6: CLI smoke test**

```bash
uv run algua doctor
uv run algua registry list
```
Both exercise the real `connect()` + `migrate()` bootstrap path against a real DB. Expected: both exit 0 (or a clean unrelated non-zero on an empty worktree). Anything mentioning a missing import, `ModuleNotFoundError: algua.registry.db`, an `OperationalError`, or a missing table is a real regression.

- [ ] **Step 7: Commit any fixes**

If steps 1-6 forced fixes, commit them (scoped `git add`, correct trailer). If nothing needed fixing, this task makes no commit — expected, and consistent with how Stages 3, 4a, and 4b's close-out tasks landed.
