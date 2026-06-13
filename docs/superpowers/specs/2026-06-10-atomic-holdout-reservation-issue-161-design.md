# Atomic holdout reservation (#161)

**Status:** GATE-1 CLOSED (2 rounds; panel approves) + user sign-off 2026-06-13. Two drift
corrections applied vs the 2026-06-10 draft: (a) `SCHEMA_VERSION` is now **21**, so the bump is
**21 → 22** (not 19 → 20); (b) `PRAGMA busy_timeout=5000` is **already present** in `connect`
(db.py:390, landed with #164) — §2 below is already satisfied and is a no-op for implementation.
**Issue:** #161 — *Holdout burn is not atomic: concurrent `research promote` runs can both consume the same out-of-sample window* (P1, `bug`, `severity:high`)
**Date:** 2026-06-10

## Problem

In `algua/cli/research_cmd.py` the holdout-reuse guard is a read-then-write split across the
entire walk-forward:

1. `repo.overlapping_holdout_evaluations(...)` — read, **no transaction** (sqlite3 opens none on a
   bare SELECT).
2. `walk_forward(...)` — the expensive part, no lock held.
3. `repo.record_holdout_evaluation(...)` — the burn write.

Two concurrent `research promote` calls for the same strategy + data window both see "no overlap"
at step 1, both run, and both record at step 3 — violating the single-use-holdout guarantee the
promotion gate's statistical validity (the #137 multiple-testing defense) rests on. Concurrent
agent sessions against one `ALGUA_DB_PATH` are the documented operating mode, and the planned VPS
scale-out multiplies the pressure.

This is a classic TOCTOU. The codebase already has the correct primitive: `allocations.py:41`
takes `BEGIN IMMEDIATE` so its read-check-write (Σ allocations ≤ equity) is atomic. We apply the
same pattern here as a **reservation**, without serializing the expensive `walk_forward`.

## Approach (chosen)

A holdout window is **reserved** atomically (under the write lock — a fast check + one INSERT),
then `walk_forward` runs with **no lock held**, then the reservation is **finalized** into a
committed burn on success or **released** on failure.

The reservation lives as a lifecycle on the **existing** `holdout_evaluations` table (one table,
one index, single source of truth for "this window is taken"), distinguished by a new nullable
`committed_at` column:

- `committed_at IS NULL`  → an **in-flight reservation** (or, on a row that predates this column,
  a legacy committed burn — see Schema below; both block fail-closed, so the distinction is moot
  for the overlap wall)
- `committed_at` set      → a **committed burn**

The integrity-critical overlap query matches **any** row (pending or committed), so a pending
reservation blocks a concurrent run exactly like a committed burn does.

### Rejected alternatives

- **Separate `holdout_reservations` table.** Cleaner row-level isolation, but duplicates the
  interval+identity match SQL across two tables; interval overlap can't be a SQLite unique
  constraint anyway, so the single-table lifecycle is no less safe. Rejected for the
  single-source-of-truth model above. *(All three GATE-1 reviewers agreed the single-table model
  is sound once migration + tx hygiene are fixed.)*
- **TTL / auto-expiry of orphaned reservations.** Adds a wall-clock dependency, a tunable timeout,
  and a residual race. Rejected in favor of fail-closed (below).
- **Backfilling legacy rows' `committed_at` (the originally-approved design).** GATE-1 showed the
  backfill `UPDATE` *introduces* a migration race (see Schema): under concurrent first-upgrade it
  can convert a genuinely in-flight reservation into a committed burn. Dropped — see below.

## Design

### 1. Schema — `algua/registry/db.py` (`SCHEMA_VERSION` 21 → 22)

Add one nullable column to `holdout_evaluations`:

```sql
committed_at TEXT   -- NULL = in-flight reservation (or a legacy burn predating this column);
                    -- non-NULL = committed burn. Either way an overlapping row blocks fail-closed.
```

Added two ways, **with NO data backfill**:
- to the `CREATE TABLE` in `_SCHEMA` (fresh DBs get it from the start), and
- to the existing `_add_missing_columns(conn, "holdout_evaluations", {"committed_at": "TEXT"})`
  call for existing DBs (a plain `ALTER TABLE ADD COLUMN`, NULL on existing rows — the same
  pattern every other migrated column already uses).

**Why no backfill (changed from the approved design).** GATE-1 (all three reviewers, CRITICAL)
showed a backfill `UPDATE holdout_evaluations SET committed_at = created_at` is *not* race-safe:
in WAL with concurrent processes, once one process's `ALTER` commits, a second process whose
`migrate()` already ran can `reserve` a pending row (`committed_at=NULL`) *before* the first
process's `UPDATE` runs — and the unconditional `UPDATE` then clobbers that genuine reservation
into a "committed burn." Dropping the backfill removes the `UPDATE` entirely, so there is nothing
to clobber. A legacy row that predates the column simply has `committed_at=NULL` and is treated as
a permanent reservation: the overlap query blocks on it exactly like a burn (fail-closed, correct),
and it is never released/finalized because those operate only on a reservation id this process just
minted. This is documented in the schema comment. *(There are few/no such rows in practice; the
table is recent.)*

**Concurrent-`ALTER` race (all three reviewers).** Two processes calling `migrate()` on a fresh
upgrade can both see the column absent and both `ALTER`, the second raising
`sqlite3.OperationalError: duplicate column name`. This is a pre-existing latent property of every
`_add_missing_columns` column, not new to this change — but since cross-process safety is the whole
point here, harden `_add_missing_columns` to make the add idempotent: catch the
`OperationalError` whose message contains `duplicate column name` and treat it as success (the
column now exists). This fixes the race for **all** migrated columns. `SCHEMA_VERSION` bumps to 22
per the db.py contract that a version bump carries its migration.

### 2. Connection — `algua/registry/db.py:connect`  *(ALREADY DONE — #164, db.py:390)*

`conn.execute("PRAGMA busy_timeout=5000;")` is already present in `connect`. No code change here;
the rationale below explains why it matters for this fix. Without it, a `BEGIN IMMEDIATE` loser gets a raw
`sqlite3.OperationalError: database is locked`. With the timeout, the loser waits for the winner's
sub-millisecond critical section (a SELECT + one INSERT), then re-checks and fails closed with the
clean "holdout already consumed" message.

**Blast radius (OpenCode MEDIUM):** this is system-wide — it changes *every* `BEGIN IMMEDIATE`
lock-contention outcome (allocations, the migration) from an immediate error to a wait of up to 5s.
That is the desired direction (wait-then-proceed beats immediate failure), and no current call path
assumes an immediate `OperationalError` on contention (`allocations.allocate` and `migrate` both
just want to proceed once the lock frees). Kept fixed (not settings-configurable) — YAGNI.

**busy_timeout is UX, not correctness (Codex/OpenCode MEDIUM).** Correctness comes from
`BEGIN IMMEDIATE` + re-check-under-lock + insert-before-releasing-lock. The timeout only makes the
loser's failure orderly. Tests assert the *invariant* (no second burn), not merely the message.

### 3. Store + Protocol — `algua/registry/store.py`, `algua/registry/repository.py`

**Remove** (promote-path-only; replaced):

- `overlapping_holdout_evaluations`
- `record_holdout_evaluation`

**Add** three methods (Protocol in `repository.py`, sqlite impl in `store.py`):

```python
def reserve_holdout(
    self, strategy_id: int, *, data_source: str, snapshot_id: str | None,
    period_start: str, period_end: str, holdout_frac: float, allow_reuse: bool,
) -> tuple[int, bool]:
    """Atomically claim the holdout window; return (reservation_id, reused).

    Raises ValueError (fail closed) if an overlapping row exists and not allow_reuse.
    TOP-LEVEL ONLY: must not be called inside an open transaction / `with self._conn:` block."""
```

**Transaction hygiene (all three reviewers, HIGH):**
- Guard first: `if self._conn.in_transaction: raise RuntimeError(...)` — a manual `BEGIN IMMEDIATE`
  inside an already-open transaction would raise "cannot start a transaction within a transaction",
  and a blanket rollback could roll back a caller's surrounding tx. Fail loudly instead. (In the
  promote flow this never triggers — only reads precede it — but the guard makes the contract
  enforced, not assumed.)
- Then `self._conn.execute("BEGIN IMMEDIATE")` (acquires the write lock so the overlap SELECT +
  INSERT are one atomic critical section). The SELECT runs **after** `BEGIN IMMEDIATE`.
- Wrap the body in `try: ... except BaseException: self._conn.rollback(); raise` — `BaseException`
  (not `Exception`) so a `KeyboardInterrupt`/`SystemExit` in the critical section still releases
  the lock. (allocations.py uses `except Exception`; same latent gap — note it, optionally fix
  there too.)

**Overlap predicate (under the lock):** port the EXACT existing `overlapping_holdout_evaluations`
SQL (`store.py:381-396`), now matching **all** rows (pending or committed — i.e. no `committed_at`
filter). Precise rule, against `holdout_evaluations` for this `strategy_id`:
- **Data identity** depends on the *probe*: if the probe has a `snapshot_id`, match rows
  `WHERE snapshot_id = <probe>`; if the probe has **no** snapshot, match rows
  `WHERE snapshot_id IS NULL AND data_source = <probe>` (a snapshot-backed row is a *distinct*
  identity from a non-snapshot probe — they must NOT collide).
- **AND** period overlap: `period_start <= <probe period_end> AND <probe period_start> <= period_end`.
- **AND** same `holdout_frac`.

Match is on the window, never config. This is the integrity core — port it verbatim from the
removed method; do not re-derive it.

**Decision:**
- overlap **and not** `allow_reuse` → raise `ValueError("holdout already consumed: an overlapping
  out-of-sample window was already evaluated. Use fresh out-of-sample data, or --allow-holdout-reuse
  (--actor human) to override and accept the statistical cost.")` inside the `try` → rollback (no
  row written, lock released) → propagates as a JSON error. **Fail closed.** (Message drops the
  `{name!r}`: the store works in `strategy_id`; the promote invocation already identifies the
  strategy.)
- overlap **and** `allow_reuse` → INSERT a pending row with `reused=1` (the human's audited
  override; also the unstick path past an orphaned reservation).
- no overlap → INSERT a pending row with `reused=0`.

The INSERT writes `committed_at=NULL`, `config_hash=''` (placeholder, filled at finalize),
`created_at=now`. `commit`, return `(lastrowid, reused)`.

```python
def finalize_holdout_reservation(self, reservation_id: int, *, config_hash: str) -> None:
    """Commit a reservation into a burn: set committed_at + the real (evidentiary) config_hash."""
```

- Inside `with self._conn:`: `UPDATE holdout_evaluations SET committed_at = ?, config_hash = ?
  WHERE id = ? AND committed_at IS NULL`; if `rowcount != 1`, **raise inside the `with`** (so the
  mismatch rolls back, mirroring `apply_transition`'s gate-consume guard at `store.py:272-275`).
  Guards a double-finalize or a vanished/released row. (Raise, not `assert` — asserts strip under
  `python -O`.)

```python
def release_holdout_reservation(self, reservation_id: int) -> None:
    """Free a still-pending reservation (clean walk_forward failure). Never touches a burn."""
```

- `with self._conn:` → `DELETE FROM holdout_evaluations WHERE id = ? AND committed_at IS NULL`.
  (No guard: a release after a finalize/crash is a harmless no-op.)

**Schema-comment documentation (Codex/Gemini/OpenCode LOW):** in `holdout_evaluations`, document:
(a) `committed_at IS NULL` = in-flight reservation (or legacy pre-column burn); non-NULL = burn;
(b) `config_hash=''` is the in-flight placeholder, replaced with the real hash at finalize — never
a real empty hash; (c) orphaned reservations are listable via `SELECT … WHERE committed_at IS NULL`
(the human-inspection path for the orphan policy).

### 4. Orchestration — `algua/cli/research_cmd.py`

Replace the check→record pair (lines ~103–122) with reserve → run → finalize/release:

```python
breadth = promotion_preflight(...)
reservation_id, reused = repo.reserve_holdout(
    repo.get(name).id, data_source=data_source, snapshot_id=snapshot_id,
    period_start=period_start, period_end=period_end, holdout_frac=holdout_frac,
    allow_reuse=allow_holdout_reuse)            # raises here = fail closed (overlap, no reuse)
try:
    wf = walk_forward(strategy, provider, start_dt, end_dt, windows=windows,
                      holdout_frac=holdout_frac, universe_by_date=universe_by_date,
                      universe_name=universe, universe_snapshots=universe_prov)
except Exception:
    repo.release_holdout_reservation(reservation_id)   # clean failure frees the window
    raise
repo.finalize_holdout_reservation(reservation_id, config_hash=wf.config_hash)
outcome = run_gate(...)
```

`reused` flows downstream exactly as before. Orchestration stays in the CLI — moving it into
`promotion.py` is #165's scope, out of scope here.

**finalize ordering (Gemini Q4 — declined, with rationale).** `finalize` sits after a successful
`walk_forward` and *before* `run_gate`, matching today's burn-before-gate ordering (today
`record_holdout_evaluation` precedes `run_gate`). The holdout is "looked at" the moment
`walk_forward` computes its metrics, so burning there is correct per the burn-on-peek invariant; a
`run_gate` exception leaving the holdout burned is exactly today's behavior. Moving `finalize` after
`run_gate` would risk a transitioned-but-unfinalized state (run_gate both records the gate row and
performs the stage transition) — strictly worse. Declined.

**walk_forward all-or-nothing (Codex MEDIUM — verify in implementation).** Release-on-failure is
safe iff `walk_forward` raises only *before* it exposes usable holdout metrics (so a released
window was never really peeked). This is already true of today's code (today's `record` is after
`walk_forward`, so a raising `walk_forward` never burns). Plan task: confirm `walkforward.py` raises
only in early validation (insufficient data, bad inputs) and never computes holdout metrics then
raises. If that assumption is false, it is a pre-existing issue to flag, not introduced here.

A clean `walk_forward` failure (`BacktestError`, a `ValueError`, etc.) self-releases via the
`except`. Only a hard kill (SIGKILL/OOM) between reserve and finalize/release orphans a pending row.

### 5. Orphan policy (chosen: fail-closed, human unstick)

An orphaned pending row keeps blocking its window — the safe direction (never silently re-use OOS
data). Recovery is a deliberate human action: `--allow-holdout-reuse` (already human-only and
audited) proceeds past **any** overlapping row, pending included. No TTL, no new command, no
wall-clock dependency. Documented in the `holdout_evaluations` schema comment; orphans are listable
via `WHERE committed_at IS NULL`.

## Testing

### Store unit (`tests/test_registry_store.py` — rewrite the 3 holdout tests + add lifecycle)

- `reserve_holdout` blocks an overlapping window (same identity/frac); allows disjoint period,
  different `holdout_frac`, different `data_source`; snapshot-identity precedence. *(Ports the
  existing overlap cases onto the new method.)*
- Lifecycle: `reserve → release → reserve` same window **OK**; `reserve → finalize → reserve`
  **blocked**; under `allow_reuse=True` the second reserve **succeeds** with `reused=True` (past a
  committed burn AND a pending orphan); orphan (`reserve`, never finalize) **blocks**.
- Guards: `finalize` on an already-finalized/released id raises (`rowcount != 1`); `release` after
  finalize is a no-op; `reserve_holdout` called inside an open transaction raises `RuntimeError`.

### Concurrency — the race proof (all three reviewers: the subprocess form alone is insufficient)

Two complementary tests in a new `tests/test_holdout_concurrency.py`:

1. **Deterministic regression guard (sequential, store-level):** reserve a window (commits a pending
   row), then a second `reserve_holdout` on an overlapping window raises the consumed `ValueError`.
   Proves the committed/pending row blocks a second claim — fast, deterministic, always meaningful.

2. **Barriered true-concurrency proof:** N=2 workers (a `multiprocessing.Barrier`, or threads with
   a `threading.Barrier`), each opening its **own** connection to one shared DB, aligned at the
   barrier, then both calling `reserve_holdout` on the **same** window. To force the critical
   sections to actually overlap (not serialize by luck — the failure mode all three flagged),
   widen the winner's critical section. **Prefer a monkeypatched sleep** between the overlap SELECT
   and the INSERT for the in-process (threads) variant. Where monkeypatch can't reach (the
   subprocess variant), use a **private test-only env hook** — `ALGUA_TEST_RESERVE_DELAY_MS`,
   default `0`, read once inside `reserve_holdout` between the SELECT and the INSERT, NOT surfaced in
   CLI help/docs. Assert: exactly **one** worker returns success and the DB holds exactly
   **one** committed/pending non-reuse row; the other raises the consumed error (a `busy_timeout`
   wait then a clean fail-closed — NOT a second burn). Asserting "exactly one row" (the invariant),
   not just the message, is what makes this a real proof rather than a serialization coincidence.

3. **Real-process end-to-end (the issue's literal ask):** two `subprocess` `algua research promote`
   runs (`--actor human --demo --n-combos N --allow-non-pit`, **no** `--allow-holdout-reuse`) on the
   same strategy/window via `Popen`, launched together. Assert exactly one exits ok+promoted and the
   other exits fail-closed with the holdout-consumed error. Setup mirrors `tests/test_e2e_lifecycle.py`
   / `tests/test_operator_layer.py`; pick a `--demo` window/`--windows` that yields a valid
   `walk_forward` so the winner finalizes. (Honest framing per Codex: this proves the end-to-end path;
   test 2 is the tight-race proof.)

*Note (#164):* tests 2–3 are the first real multi-process/concurrency tests in the suite.

**JSON-contract guard (Codex MEDIUM):** confirm a sqlite lock error on the promote path still emits
parseable JSON (invariant #5), not a raw traceback — either the CLI top-level handler already
covers `sqlite3.OperationalError`, or add it to the promote `json_errors`. With `busy_timeout=5000`
a true lock-timeout is near-impossible (sub-ms critical section), but the envelope must not leak.

## Invariants & boundaries

- **Strengthens** the #137 single-use-holdout wall; weakens nothing. No relaxation path is widened
  (`--allow-holdout-reuse` / `--allow-non-pit` / `--n-combos` remain human-only via
  `promotion_preflight` / `guard_agent_relaxations`).
- Match remains on the **window**, never config; `config_hash` stays evidence-only (written at
  finalize rather than at record).
- `algua/contracts` / `algua/features` purity untouched; no new cross-module imports
  (`lint-imports` stays "0 broken"). All SQL stays in `store.py`.
- Gate green after every commit: `uv run pytest -q && uv run ruff check . && uv run mypy algua &&
  uv run lint-imports`.

## GATE-1 review outcome (multi-model, round 1)

Panel: Codex (GPT-5.x), Gemini 2.5 Flash, OpenCode/GLM-5.1 — all read-only, design lens.

**Accepted → folded in above:** race-unsafe migration (drop the backfill; idempotent `ALTER`) ·
`reserve_holdout` tx hygiene (`in_transaction` guard, `BaseException` rollback, top-level-only) ·
finalize rowcount guard inside the `with` · busy_timeout blast-radius doc · barriered
true-concurrency test + "assert the invariant, not the message" · `walk_forward` all-or-nothing
verification · schema-comment docs (`config_hash=''`, orphan-listing) · JSON-contract guard on lock
errors.

**Declined (rationale above):** move finalize after run_gate · separate reservations table ·
soft-delete / release-audit table · settings-configurable timeout · `__pending__` config_hash
sentinel (the `committed_at` NULL is the authoritative discriminator).

**Noted, out of scope:** OpenCode found a real pre-existing TOCTOU in `allocations.deallocate`
(read-then-UPDATE without `BEGIN IMMEDIATE`). Not fixed here — candidate for a follow-up issue.

## Out of scope

- Moving promote orchestration out of the CLI into `promotion.py` (#165).
- A separate sweep/breadth reservation ledger (the issue's "discharges the #137 sweep ledger" note
  is read as **holdout** reservation only; a breadth ledger is a distinct, larger change).
- TTL/auto-expiry of orphaned reservations.
- `allocations.deallocate` TOCTOU (noted above; follow-up).
