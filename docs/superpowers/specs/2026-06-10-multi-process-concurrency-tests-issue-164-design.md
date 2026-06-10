# Multi-process concurrency tests for the shared registry DB (#164)

**Status:** design · **Date:** 2026-06-10 · **Issue:** #164

## Problem

All 111 test files run single-process against per-test `tmp_path` DBs. Not one
spawns two processes against a shared `ALGUA_DB_PATH` — yet **multiple concurrent
agent sessions on one DB is the documented operating mode** (and the VPS scale-out
will add more). The concurrency primitives the platform relies on are therefore
entirely unproven by CI:

- `connect()` (`algua/registry/db.py:345`) opens WAL but never sets `busy_timeout`
  explicitly. The cap that the cross-process tests depend on is supplied implicitly
  by Python's `sqlite3.connect(timeout=5.0)` default (→ `busy_timeout=5000`).
- `apply_transition` (`algua/registry/store.py:258`) wraps gate-token consume +
  stage UPDATE + transition INSERT in one `with self._conn:` transaction; the
  **rollback path is untested** (`tests/test_registry_store.py` covers success only).
- `allocate` (`algua/registry/allocations.py:38`) uses `BEGIN IMMEDIATE` to make the
  read-check-write capital cap atomic; **untested under real concurrency**.

## Decision

Add **real-subprocess concurrency tests.** The one production change is a single
explicit line in `connect()` (see "busy_timeout" below); everything else is tests.
Empirically the primitives are already correct: WAL readers never block, a second
writer is serialized by the 5s `busy_timeout` cross-process, and `BEGIN IMMEDIATE`
serializes the cap. This issue proves those invariants so the *next* race trips CI
instead of review. (The concrete races already found by inspection — #161 holdout
burn, #158 manifest append — carry their own targeted tests; this is the baseline
counterpart.)

### Why real separate processes are mandatory

An in-process two-connection probe is **misleading**: when both connections live in
one thread, SQLite detects that honoring `busy_timeout` would deadlock and skips the
busy handler, returning `SQLITE_BUSY` ("database is locked") immediately. Only
genuine OS processes exercise the real cross-process serialization path. Hence
`subprocess` workers, not threads.

### Forcing real contention (not vacuous passes)

A `ready`/`go` barrier only aligns process *start*; the critical sections are tiny,
so two workers can still serialize past each other without ever contending — making
the serialization tests pass vacuously. To make contention **observable and
deterministic**, the contended tests use a **lock-holder** worker:

1. a holder worker opens its connection, runs `BEGIN IMMEDIATE` (takes the write
   lock), mutates a row, emits a `lock-held` sentinel, then waits for a `release`
   sentinel before committing;
2. the parent waits for `lock-held`, then releases the real contender worker(s);
3. each contender emits `attempting` immediately before its critical op, so the
   parent can confirm the op is in flight *while the lock is held*;
4. the holder releases **well under** the 5s `busy_timeout`, so contenders are
   guaranteed to have queued on the lock and then proceed — real serialization,
   no dependence on scheduler timing, no `SQLITE_BUSY` from timeout exhaustion.

### busy_timeout: make it explicit

Add `conn.execute("PRAGMA busy_timeout=5000;")` to `connect()` (one line, beside the
WAL pragma). Relying on Python's `sqlite3.connect(timeout=5.0)` default is a
load-bearing contract hidden *outside* the codebase — a latent risk if the driver
default ever changes or someone passes `timeout=0`. Making it explicit reads WAL +
busy_timeout as one deliberate concurrency posture. A test then asserts the value is
`5000` (not merely `> 0`), pinning the intended behavior the cross-process tests
depend on.

## Architecture

### `tests/_concurrency_worker.py` — the worker entrypoint

A runnable module (leading underscore so pytest does not collect it), invoked as
`subprocess.run([sys.executable, "-u", "-m", "tests._concurrency_worker", <op>, ...])`
(`-u` unbuffered so the JSON line is never lost in a buffer on early exit). `tests/`
is already a package, so `-m` resolves from the repo root. The worker:

1. opens its **own** `connect(db_path)` — a real, separate sqlite connection in a
   real OS process;
2. does any per-op prep, then writes a `ready-<i>` sentinel and (for contended ops)
   an `attempting` sentinel immediately before the critical section;
3. spin-waits (small poll sleep, bounded by a timeout) for the `go` / `release`
   sentinel — a **deterministic barrier**;
4. runs the critical op as fast as possible;
5. prints a **single line of JSON** to stdout describing the outcome
   (`{"ok": true, ...}` or `{"error": "AllocationError", "msg": "..."}`) with
   `flush=True`, and exits 0.

`op` dispatches to one small function each — `transition`, `read-poll`, `allocate`,
`lock-hold` — keeping the worker focused. Structured stdout outcomes make the
parent's assertions exact rather than timing-dependent.

### `tests/test_concurrency.py` — orchestrator + tests

A small `run_workers(...)` helper spawns N workers sharing a barrier dir (under
`tmp_path`) and `ALGUA_DB_PATH`, waits for the relevant sentinels (bounded poll),
fires the release sentinel, and joins every worker with a **generous ~30s timeout**.
For each worker it collects **stdout, stderr, return code, and timeout status**, then
parses the stdout JSON; on timeout it terminates → waits → kills, and **always
surfaces stderr in the assertion message** so a worker that dies before printing is a
clear failure, not a mysterious hang or parse error. The parent creates and
`migrate()`s the DB and seeds all required state **before** spawning workers (workers
only `connect()`). Service-level tests drive `transition_strategy()` (the validating
seam), not `apply_transition()` directly.

## Tests

1. **`test_concurrent_writer_and_reader_no_lock_errors`** — a **holder** worker takes
   `BEGIN IMMEDIATE`, mutates a row, emits `lock-held`, and waits; a reader worker
   polls ~200 reads *while the write is uncommitted*; then the holder commits. Each
   read is a **multi-statement snapshot** — within one explicit `BEGIN`/read
   transaction the reader reads the `strategies` row **and** that strategy's
   `stage_transitions` rows, then asserts the two **agree** (the row's `stage` matches
   its latest transition's `to_stage`). Asserts: the reader never raises "database is
   locked", and every snapshot sees either the wholly-old or the wholly-new committed
   state — **never a torn cross-table state** (e.g. a new stage without its transition
   row, or vice-versa). Proves WAL gives readers a consistent snapshot across tables
   and never blocks on an in-flight writer — the cross-process scenario the in-process
   probe cannot show.

2. **`test_concurrent_writers_serialize`** — a holder worker holds the write lock and
   emits `lock-held`; two writer workers each run a valid `transition_strategy()` on
   **distinct** strategies, emitting `attempting` before queuing on the lock; the
   holder releases under busy_timeout. Asserts both writers succeed (the 5s
   busy_timeout serialized them behind the holder — no lock error escaped) and the DB
   is consistent afterward (each strategy at its expected stage, one transition row
   each).

3. **`test_concurrent_allocate_respects_cap`** — `account_equity=50_000`; a holder
   forces contention, then two workers each `allocate(30_000)` to **different**
   strategies, racing. Asserts the **invariant first** — final
   `total_allocated(conn) <= 50_000`, no strategy has two active allocations, no
   partial rows — and then, since the holder releases well under busy_timeout, the
   outcome: **exactly one** succeeds and the other raises `AllocationError` (a domain
   rejection, *not* a lock error, *not* a silent double-commit). Proves `BEGIN
   IMMEDIATE` keeps the read-check-write cap atomic cross-process.

4. **`test_concurrent_candidate_gate_single_use`** — the load-bearing token race. Seed
   one strategy at `backtested` with **one** passing **agent** gate token. Two workers
   both attempt `transition_strategy(.., CANDIDATE, AGENT)` against that same
   strategy/token. Asserts **exactly one** succeeds and the other raises
   `TransitionError`; final token `consumed=1`; final stage `candidate`; **exactly
   one** new `stage_transitions` row to `candidate`. Proves single-use token
   consumption holds across two processes (the `UPDATE ... WHERE consumed=0` +
   `rowcount==1` guard).

5. **`test_apply_transition_rolls_back_on_failure`** — single-process fault injection
   (transaction atomicity; no subprocess). Seed a strategy at `backtested` and mint a
   passing agent gate token. Inject the fault with a **`sqlite3.Connection` subclass /
   proxy** whose `execute` raises only on the `INSERT INTO stage_transitions` of the
   tested call (monkeypatching the C-extension method directly is unreliable; a
   subclass is robust and fires only after the token-consume + stage UPDATE). Install
   the fault **after** all seeding. Call `apply_transition(.., to=CANDIDATE,
   actor=AGENT, consume_gate_id=tok)`, expect the exception, then on a **fresh**
   connection assert the token is still `consumed=0`, the stage is still `backtested`,
   and the `stage_transitions` count is unchanged. Proves the `with self._conn:`
   rollback reverts the token-consume and the stage UPDATE together — all-or-nothing.

6. **`test_connect_sets_busy_timeout`** — asserts `connect()` yields
   `PRAGMA busy_timeout == 5000`, pinning the now-explicit invariant the cross-process
   tests rely on.

## Anti-flake measures

- Deterministic **lock-holder** barrier (not `sleep`-based timing) so contended
  workers genuinely queue on the write lock; holder releases well under busy_timeout
  so a slow CI box never turns serialization into a `SQLITE_BUSY` timeout.
- Generous ~30s join timeout and small poll intervals; a worker exceeding the timeout
  is terminated and fails the test loudly (stderr surfaced) rather than hanging it.
- Workers run `-u` / `flush=True` and emit structured JSON outcomes — assertions check
  outcomes and return codes, never wall-clock; barrier/sentinel files live under
  `tmp_path` and are cleaned with the test.

## Module / file plan

- **New** `tests/_concurrency_worker.py` — worker entrypoint (`op` dispatch).
- **New** `tests/test_concurrency.py` — `run_workers` orchestrator + the 6 tests.
- **Change** `algua/registry/db.py` — one line: explicit `PRAGMA busy_timeout=5000` in
  `connect()`.

## Out of scope (deferred)

- Fixing the concrete races #161 (holdout burn) and #158 (manifest append) — each has
  its own targeted issue/test.
- Data-manifest / parquet-snapshot concurrency (`data import-bars`, `ingest-bars`) —
  this slice covers the registry DB (`strategies`, `stage_transitions`,
  `gate_evaluations`, `strategy_allocations`). Manifest concurrency is #158's lane.
- WAL checkpoint / `-shm`/`-wal` side-file lifecycle and soak/stress testing — covered
  in real systems by stress/soak suites and ops procedures, not tiny deterministic
  unit tests; acceptable to omit here.
- Full CLI-subprocess coverage (`uv run algua ...` as workers) — the races live at the
  store/DB layer; driving the store API directly is faster and less flaky.
- A dedicated two-resource **deadlock** test — precluded by SQLite's model: a single
  database-level write lock means `BEGIN IMMEDIATE`/WAL writers *queue* (serialize via
  `busy_timeout`), they never hold-and-wait across each other, so a classic A↔B
  deadlock between two registry writers cannot arise. Such a test would pass trivially.
- **Multiple connections per worker** (one process, separate read + write connections)
  — not the operating mode (one agent session = one connection); explicitly out of
  scope.

## Gate

`uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
