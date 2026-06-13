# Atomic Holdout Reservation (#161) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the TOCTOU hole where two concurrent `research promote` runs both consume the same out-of-sample holdout window, by replacing the read-then-write holdout guard with an atomic reserve → run → finalize/release lifecycle.

**Architecture:** A holdout window is *reserved* atomically under `BEGIN IMMEDIATE` (a fast overlap SELECT + one INSERT of a pending row), then `walk_forward` runs with no lock held, then the reservation is *finalized* into a committed burn on success or *released* on a clean failure. The reservation lives on the existing `holdout_evaluations` table, distinguished by a new nullable `committed_at` column (NULL = in-flight reservation or legacy pre-column burn; non-NULL = committed burn). The integrity-critical overlap query matches **any** row (pending or committed), so a pending reservation blocks a concurrent run exactly like a burn — fail closed.

**Tech Stack:** Python 3, `sqlite3` (WAL + `BEGIN IMMEDIATE` + `busy_timeout`), Typer CLI, pytest (incl. `multiprocessing`/`threading` + `subprocess` for the concurrency proofs).

**Source of truth:** `docs/superpowers/specs/2026-06-10-atomic-holdout-reservation-issue-161-design.md` (GATE-1 approved, user signed off 2026-06-13).

**Quality gate (run at every commit):** `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`

---

## File Structure

| File | Change | Responsibility |
|---|---|---|
| `algua/registry/db.py` | Modify | Add `committed_at` column to `holdout_evaluations` (CREATE + migration); bump `SCHEMA_VERSION` 21→22; harden `_add_missing_columns` against the concurrent-`ALTER` duplicate-column race. |
| `algua/registry/repository.py` | Modify | Protocol: remove `record_holdout_evaluation` + `overlapping_holdout_evaluations`; add `reserve_holdout` / `finalize_holdout_reservation` / `release_holdout_reservation`. |
| `algua/registry/store.py` | Modify | Sqlite impl of the three new methods (reserve under `BEGIN IMMEDIATE`); remove the two old methods. |
| `algua/cli/research_cmd.py` | Modify | Orchestrate reserve → run → finalize/release; add `sqlite3.OperationalError` to the promote `@json_errors`. |
| `tests/test_registry_store.py` | Modify | Rewrite the 3 holdout tests onto the new methods + add lifecycle/guard tests. |
| `tests/test_db_migrations.py` (or existing migration test file) | Modify/Create | Migration tests: fresh DB has `committed_at`; v21 DB gains it + stamps 22; `_add_missing_columns` tolerates a lost ALTER race. |
| `tests/test_holdout_concurrency.py` | Create | The race proof: sequential regression guard + barriered true-concurrency + real-process subprocess e2e. |

**Conventions to follow** (already in the codebase):
- All SQL lives in `store.py` (or `db.py` for schema/migration). `contracts`/`features` stay pure.
- Migration pattern: a new column is added to the `_SCHEMA` `CREATE TABLE` *and* via `_add_missing_columns(...)` in `migrate()` (no data backfill).
- The `apply_transition` rowcount guard (`store.py` `_apply_transition_locked`, raises inside `with self._conn:` so a mismatch rolls back) is the model for `finalize_holdout_reservation`.
- `allocations.allocate` (`algua/registry/allocations.py:38-62`) is the model for the `BEGIN IMMEDIATE` reserve, but we upgrade `except Exception` → `except BaseException` and add an `in_transaction` guard.

---

## Task 1: Harden `_add_missing_columns` against the concurrent-ALTER duplicate-column race

**Files:**
- Modify: `algua/registry/db.py:522-533` (`_add_missing_columns`)
- Test: `tests/test_db_migrations.py` (create if it does not exist; otherwise add to the existing migration test module — check `ls tests/ | grep -i migrat` first)

Two processes calling `migrate()` on a fresh upgrade can both introspect the column as absent and both `ALTER`; the loser raises `sqlite3.OperationalError: duplicate column name`. This is a pre-existing latent property of every migrated column, but cross-process safety is the whole point of #161, so make the add idempotent for **all** columns.

- [ ] **Step 1: Write the failing test**

In `tests/test_db_migrations.py` (add imports `import sqlite3` and `from algua.registry.db import _add_missing_columns` at top if creating the file):

```python
def test_add_missing_columns_tolerates_lost_alter_race(tmp_path):
    """A process that loses the concurrent-ALTER race holds a stale 'column absent' snapshot,
    then its own ALTER raises 'duplicate column name'. _add_missing_columns must swallow that
    and treat the column as present (idempotent), not propagate the error."""
    conn = sqlite3.connect(tmp_path / "r.db")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE t (id INTEGER)")
    conn.execute("ALTER TABLE t ADD COLUMN c TEXT")  # the 'winner' already added it
    conn.commit()

    class _StaleColumnsConn:
        """Wraps a real connection but hides column ``hidden`` from PRAGMA table_info,
        reproducing the stale schema snapshot a race-loser holds."""
        def __init__(self, real, table, hidden):
            self._real, self._table, self._hidden = real, table, hidden

        def execute(self, sql, *args):
            cur = self._real.execute(sql, *args)
            if sql.startswith(f"PRAGMA table_info({self._table})"):
                return [r for r in cur.fetchall() if r["name"] != self._hidden]
            return cur

        def __getattr__(self, name):
            return getattr(self._real, name)

    stale = _StaleColumnsConn(conn, "t", "c")
    # Column 'c' really exists, but `stale` reports it absent -> _add_missing_columns will ALTER
    # and hit 'duplicate column name'. It must NOT raise.
    _add_missing_columns(stale, "t", {"c": "TEXT"})  # no exception == pass
    assert {r["name"] for r in conn.execute("PRAGMA table_info(t)")} == {"id", "c"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_db_migrations.py::test_add_missing_columns_tolerates_lost_alter_race -v`
Expected: FAIL with `sqlite3.OperationalError: duplicate column name: c`

- [ ] **Step 3: Implement the hardening**

In `algua/registry/db.py`, replace `_add_missing_columns` (currently lines ~522-533) with:

```python
def _add_missing_columns(
    conn: sqlite3.Connection, table: str, columns: dict[str, str]
) -> None:
    """Add any of ``columns`` (name -> column type) missing from ``table`` via ``ALTER TABLE``.

    Idempotent and cross-process safe: existing columns are skipped via introspection, and a
    concurrent process that adds the same column between our introspection and our ALTER (the
    lost-ALTER race) makes our ALTER raise ``duplicate column name`` — which we swallow, since the
    column now exists either way. New columns are added without a default, so on a populated table
    the existing rows get NULL — which the live/forward gates treat as fail-closed (a NULL
    ``dependency_hash`` can never match a recomputed concrete hash)."""
    existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
    for name, col_type in columns.items():
        if name not in existing:
            try:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {col_type}")
            except sqlite3.OperationalError as exc:
                # Lost the concurrent-ALTER race: another process added it first. Idempotent.
                if "duplicate column name" not in str(exc):
                    raise
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_db_migrations.py::test_add_missing_columns_tolerates_lost_alter_race -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add algua/registry/db.py tests/test_db_migrations.py
git commit -m "fix(registry): make _add_missing_columns idempotent under concurrent ALTER (#161)"
```

---

## Task 2: Schema migration — add `committed_at` + bump SCHEMA_VERSION 21→22

**Files:**
- Modify: `algua/registry/db.py` — `_SCHEMA` `holdout_evaluations` CREATE (lines ~133-157), `migrate()` (lines ~396-447), `SCHEMA_VERSION` (line 16)
- Test: `tests/test_db_migrations.py`

- [ ] **Step 1: Write the failing tests**

```python
from algua.registry.db import SCHEMA_VERSION, connect, migrate


def test_fresh_db_has_committed_at_column(tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(holdout_evaluations)")}
    assert "committed_at" in cols
    assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION


def test_v21_db_gains_committed_at_on_migrate(tmp_path):
    """A holdout_evaluations table created WITHOUT committed_at (pre-v22) gains it via migrate,
    legacy rows keep committed_at NULL, and user_version stamps to the new version."""
    conn = connect(tmp_path / "r.db")
    # Minimal legacy table shape (no committed_at), plus a legacy burn row.
    conn.execute(
        "CREATE TABLE holdout_evaluations ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT, strategy_id INTEGER NOT NULL,"
        " data_source TEXT NOT NULL, snapshot_id TEXT, period_start TEXT NOT NULL,"
        " period_end TEXT NOT NULL, holdout_frac REAL NOT NULL, config_hash TEXT NOT NULL,"
        " reused INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL)"
    )
    conn.execute("PRAGMA user_version=21")
    conn.execute(
        "INSERT INTO holdout_evaluations"
        "(strategy_id, data_source, snapshot_id, period_start, period_end, holdout_frac,"
        " config_hash, reused, created_at) VALUES (1,'demo',NULL,'2022-01-01','2022-12-31',"
        " 0.2,'abc',0,'2022-01-01')"
    )
    conn.commit()
    migrate(conn)
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(holdout_evaluations)")}
    assert "committed_at" in cols
    # Legacy row: committed_at is NULL (treated as a permanent reservation -> blocks fail-closed).
    row = conn.execute("SELECT committed_at FROM holdout_evaluations").fetchone()
    assert row["committed_at"] is None
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 22
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_db_migrations.py -k committed_at -v`
Expected: FAIL — `committed_at` absent / `user_version` is 21 not 22.

- [ ] **Step 3: Implement the schema + migration + version bump**

(a) In `db.py`, bump the version (line 16):

```python
SCHEMA_VERSION = 22
```

(b) In `_SCHEMA`, replace the `holdout_evaluations` CREATE TABLE (lines ~144-155) — add the `committed_at` column and extend the doc comment. The new column line and a doc addition:

```sql
CREATE TABLE IF NOT EXISTS holdout_evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id INTEGER NOT NULL REFERENCES strategies(id),
    data_source TEXT NOT NULL,
    snapshot_id TEXT,
    period_start TEXT NOT NULL,
    period_end TEXT NOT NULL,
    holdout_frac REAL NOT NULL,
    config_hash TEXT NOT NULL,   -- '' while a reservation is in-flight (placeholder); the real
                                 -- evidentiary hash is written at finalize. Never a real empty hash.
    reused INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    committed_at TEXT            -- NULL = in-flight reservation (or a legacy burn predating this
                                 -- column); non-NULL = committed burn. Either way an overlapping
                                 -- row blocks fail-closed. Orphaned reservations (pending rows from
                                 -- a crashed run) are listable via WHERE committed_at IS NULL and
                                 -- are cleared only by a deliberate human --allow-holdout-reuse.
);
```

(c) In `migrate()`, after the existing `_add_missing_columns(conn, "paper_orders", {"strategy_id": "INTEGER"})` call (~line 445) and before the `PRAGMA user_version` stamp, add:

```python
    # v22 (#161): committed_at distinguishes an in-flight holdout reservation (NULL) from a
    # committed burn (non-NULL). NO backfill: a legacy row that predates this column keeps
    # committed_at=NULL and is treated as a permanent reservation (blocks fail-closed). Backfilling
    # would introduce a migration race that could clobber a genuine concurrent reservation.
    _add_missing_columns(conn, "holdout_evaluations", {"committed_at": "TEXT"})
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_db_migrations.py -k committed_at -v`
Expected: PASS (both tests)

- [ ] **Step 5: Commit**

```bash
git add algua/registry/db.py tests/test_db_migrations.py
git commit -m "feat(registry): add holdout_evaluations.committed_at; SCHEMA_VERSION 22 (#161)"
```

---

## Task 3: Store reservation methods — reserve / finalize / release (remove old pair)

**Files:**
- Modify: `algua/registry/repository.py:196-231` (Protocol)
- Modify: `algua/registry/store.py:408-459` (sqlite impl)
- Test: `tests/test_registry_store.py:126-...` (rewrite the 3 holdout tests; add lifecycle/guard tests)

This is the integrity core. The overlap predicate is ported **verbatim** from the removed `overlapping_holdout_evaluations`, now matching all rows (no `committed_at` filter).

### 3a. Tests first

- [ ] **Step 1: Rewrite the holdout tests + add lifecycle/guard tests**

Replace the existing holdout tests in `tests/test_registry_store.py` (the block starting at the `# --- holdout_evaluations ---` marker, ~line 126, through the last `test_holdout_*`) with:

```python
# --- holdout reservation lifecycle ------------------------------------------

def _strategy_id(repo):
    return repo.add("h").id


def test_reserve_blocks_overlapping_window(repo):
    sid = _strategy_id(repo)
    repo.reserve_holdout(
        sid, data_source="demo", snapshot_id=None,
        period_start="2022-01-01", period_end="2023-12-31", holdout_frac=0.2, allow_reuse=False)
    # Overlapping period, same data identity + holdout_frac -> fail closed.
    with pytest.raises(ValueError, match="holdout already consumed"):
        repo.reserve_holdout(
            sid, data_source="demo", snapshot_id=None,
            period_start="2023-06-01", period_end="2024-06-01", holdout_frac=0.2,
            allow_reuse=False)


def test_reserve_allows_disjoint_frac_and_source(repo):
    sid = _strategy_id(repo)
    repo.reserve_holdout(
        sid, data_source="demo", snapshot_id=None,
        period_start="2022-01-01", period_end="2022-12-31", holdout_frac=0.2, allow_reuse=False)
    # Disjoint period -> ok.
    rid, reused = repo.reserve_holdout(
        sid, data_source="demo", snapshot_id=None,
        period_start="2023-01-01", period_end="2023-12-31", holdout_frac=0.2, allow_reuse=False)
    assert reused is False and isinstance(rid, int)
    # Different holdout_frac on an overlapping window -> ok (distinct identity).
    repo.reserve_holdout(
        sid, data_source="demo", snapshot_id=None,
        period_start="2022-01-01", period_end="2022-12-31", holdout_frac=0.3, allow_reuse=False)
    # Different data_source on an overlapping window -> ok.
    repo.reserve_holdout(
        sid, data_source="other", snapshot_id=None,
        period_start="2022-01-01", period_end="2022-12-31", holdout_frac=0.2, allow_reuse=False)


def test_reserve_snapshot_identity_precedence(repo):
    sid = _strategy_id(repo)
    # A snapshot-backed reservation is a DISTINCT identity from a non-snapshot probe on the same
    # window: neither blocks the other.
    repo.reserve_holdout(
        sid, data_source="demo", snapshot_id="snapA",
        period_start="2022-01-01", period_end="2022-12-31", holdout_frac=0.2, allow_reuse=False)
    repo.reserve_holdout(
        sid, data_source="demo", snapshot_id=None,
        period_start="2022-01-01", period_end="2022-12-31", holdout_frac=0.2, allow_reuse=False)
    # A second snapshotA reservation on the same window IS blocked.
    with pytest.raises(ValueError, match="holdout already consumed"):
        repo.reserve_holdout(
            sid, data_source="demo", snapshot_id="snapA",
            period_start="2022-06-01", period_end="2022-09-30", holdout_frac=0.2,
            allow_reuse=False)


def test_lifecycle_release_then_reserve_ok(repo):
    sid = _strategy_id(repo)
    rid, _ = repo.reserve_holdout(
        sid, data_source="demo", snapshot_id=None,
        period_start="2022-01-01", period_end="2022-12-31", holdout_frac=0.2, allow_reuse=False)
    repo.release_holdout_reservation(rid)
    # Window freed -> re-reserve succeeds.
    rid2, reused = repo.reserve_holdout(
        sid, data_source="demo", snapshot_id=None,
        period_start="2022-01-01", period_end="2022-12-31", holdout_frac=0.2, allow_reuse=False)
    assert reused is False and rid2 != rid


def test_lifecycle_finalize_then_reserve_blocked(repo):
    sid = _strategy_id(repo)
    rid, _ = repo.reserve_holdout(
        sid, data_source="demo", snapshot_id=None,
        period_start="2022-01-01", period_end="2022-12-31", holdout_frac=0.2, allow_reuse=False)
    repo.finalize_holdout_reservation(rid, config_hash="real-hash")
    with pytest.raises(ValueError, match="holdout already consumed"):
        repo.reserve_holdout(
            sid, data_source="demo", snapshot_id=None,
            period_start="2022-01-01", period_end="2022-12-31", holdout_frac=0.2,
            allow_reuse=False)


def test_orphaned_reservation_blocks(repo):
    sid = _strategy_id(repo)
    repo.reserve_holdout(  # never finalized -> orphan
        sid, data_source="demo", snapshot_id=None,
        period_start="2022-01-01", period_end="2022-12-31", holdout_frac=0.2, allow_reuse=False)
    with pytest.raises(ValueError, match="holdout already consumed"):
        repo.reserve_holdout(
            sid, data_source="demo", snapshot_id=None,
            period_start="2022-01-01", period_end="2022-12-31", holdout_frac=0.2,
            allow_reuse=False)


def test_allow_reuse_proceeds_past_burn_and_orphan(repo):
    sid = _strategy_id(repo)
    rid, _ = repo.reserve_holdout(
        sid, data_source="demo", snapshot_id=None,
        period_start="2022-01-01", period_end="2022-12-31", holdout_frac=0.2, allow_reuse=False)
    repo.finalize_holdout_reservation(rid, config_hash="real-hash")
    # Past a committed burn, the human override succeeds and flags reused.
    rid2, reused = repo.reserve_holdout(
        sid, data_source="demo", snapshot_id=None,
        period_start="2022-01-01", period_end="2022-12-31", holdout_frac=0.2, allow_reuse=True)
    assert reused is True and rid2 != rid
    # Past an orphan (the rid2 pending row), override still succeeds.
    rid3, reused3 = repo.reserve_holdout(
        sid, data_source="demo", snapshot_id=None,
        period_start="2022-01-01", period_end="2022-12-31", holdout_frac=0.2, allow_reuse=True)
    assert reused3 is True and rid3 != rid2


def test_finalize_twice_raises(repo):
    sid = _strategy_id(repo)
    rid, _ = repo.reserve_holdout(
        sid, data_source="demo", snapshot_id=None,
        period_start="2022-01-01", period_end="2022-12-31", holdout_frac=0.2, allow_reuse=False)
    repo.finalize_holdout_reservation(rid, config_hash="h1")
    with pytest.raises(ValueError):
        repo.finalize_holdout_reservation(rid, config_hash="h2")


def test_release_after_finalize_is_noop(repo):
    sid = _strategy_id(repo)
    rid, _ = repo.reserve_holdout(
        sid, data_source="demo", snapshot_id=None,
        period_start="2022-01-01", period_end="2022-12-31", holdout_frac=0.2, allow_reuse=False)
    repo.finalize_holdout_reservation(rid, config_hash="h1")
    repo.release_holdout_reservation(rid)  # no-op, no raise
    # The burn survives: a fresh reserve is still blocked.
    with pytest.raises(ValueError, match="holdout already consumed"):
        repo.reserve_holdout(
            sid, data_source="demo", snapshot_id=None,
            period_start="2022-01-01", period_end="2022-12-31", holdout_frac=0.2,
            allow_reuse=False)


def test_reserve_inside_open_transaction_raises(repo, tmp_path):
    sid = _strategy_id(repo)
    repo._conn.execute("BEGIN")  # simulate a caller holding an open transaction
    try:
        with pytest.raises(RuntimeError, match="open transaction"):
            repo.reserve_holdout(
                sid, data_source="demo", snapshot_id=None,
                period_start="2022-01-01", period_end="2022-12-31", holdout_frac=0.2,
                allow_reuse=False)
    finally:
        repo._conn.rollback()
```

(Check the existing `repo` fixture at the top of `tests/test_registry_store.py`; if there is none, mirror the inline `connect`/`migrate`/`SqliteStrategyRepository` setup used by `test_record_exposes_metadata_defaults` and add a small `@pytest.fixture def repo(tmp_path): conn = connect(tmp_path/"r.db"); migrate(conn); return SqliteStrategyRepository(conn)`.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_registry_store.py -k "reserve or lifecycle or orphan or finalize or release_after" -v`
Expected: FAIL — `AttributeError: 'SqliteStrategyRepository' object has no attribute 'reserve_holdout'`.

### 3b. Implementation

- [ ] **Step 3: Update the Protocol in `repository.py`**

Remove `record_holdout_evaluation` (lines ~196-213) and `overlapping_holdout_evaluations` (lines ~215-231). In their place add:

```python
    def reserve_holdout(
        self,
        strategy_id: int,
        *,
        data_source: str,
        snapshot_id: str | None,
        period_start: str,
        period_end: str,
        holdout_frac: float,
        allow_reuse: bool,
    ) -> tuple[int, bool]:
        """Atomically claim the holdout window; return ``(reservation_id, reused)``.

        Under ``BEGIN IMMEDIATE`` (write lock held): re-check overlap against ALL rows (pending
        reservation OR committed burn) for this strategy + data identity + overlapping period +
        same ``holdout_frac``, then INSERT a pending row (``committed_at=NULL``, placeholder
        ``config_hash=''``). Match is on the WINDOW, never config.

        Raises ``ValueError`` (fail closed) if an overlapping row exists and not ``allow_reuse``.
        ``reused`` is True iff an overlapping row existed and the human override let it proceed.

        TOP-LEVEL ONLY: must not be called inside an open transaction / ``with self._conn:`` block
        (raises ``RuntimeError`` if ``self._conn.in_transaction``)."""
        ...

    def finalize_holdout_reservation(self, reservation_id: int, *, config_hash: str) -> None:
        """Commit a reservation into a burn: set ``committed_at`` + the real evidentiary
        ``config_hash``. Raises if the row is missing or already committed (guards double-finalize).
        """
        ...

    def release_holdout_reservation(self, reservation_id: int) -> None:
        """Free a still-pending reservation (clean walk_forward failure). Never touches a committed
        burn; a release after finalize/crash is a harmless no-op."""
        ...
```

- [ ] **Step 4: Implement in `store.py`**

Add `import os` to the top of `store.py` (after `import sqlite3`). Remove `record_holdout_evaluation` (lines ~408-430) and `overlapping_holdout_evaluations` (lines ~432-459). In their place add:

```python
    def reserve_holdout(
        self,
        strategy_id: int,
        *,
        data_source: str,
        snapshot_id: str | None,
        period_start: str,
        period_end: str,
        holdout_frac: float,
        allow_reuse: bool,
    ) -> tuple[int, bool]:
        # TOP-LEVEL ONLY. A manual BEGIN IMMEDIATE inside an already-open transaction raises
        # "cannot start a transaction within a transaction", and a blanket rollback below could
        # roll back a caller's surrounding tx. Fail loudly so the contract is enforced, not assumed.
        if self._conn.in_transaction:
            raise RuntimeError(
                "reserve_holdout must be called at top level, not inside an open transaction")
        # Data identity: snapshot_id when the probe has one (a snapshot-backed row is a DISTINCT
        # identity from a non-snapshot probe), else data_source among rows that also lack a snapshot.
        # Period overlap is the standard interval test. Match is on the WINDOW, never config. Ported
        # verbatim from the removed overlapping_holdout_evaluations; now matches ALL rows (pending
        # reservation OR committed burn) — no committed_at filter — so a pending row blocks too.
        if snapshot_id is not None:
            data_match = "snapshot_id = ?"
            data_param: str = snapshot_id
        else:
            data_match = "snapshot_id IS NULL AND data_source = ?"
            data_param = data_source
        # BEGIN IMMEDIATE takes the write lock up front so the overlap SELECT + INSERT are one
        # atomic critical section: two concurrent reserves can't both see "no overlap" and both
        # insert. BaseException (not Exception) so a KeyboardInterrupt/SystemExit still releases the
        # lock via rollback.
        try:
            self._conn.execute("BEGIN IMMEDIATE")
            row = self._conn.execute(
                f"SELECT 1 FROM holdout_evaluations WHERE strategy_id = ? AND holdout_frac = ?"
                f" AND {data_match}"
                f" AND period_start <= ? AND ? <= period_end LIMIT 1",
                (strategy_id, holdout_frac, data_param, period_end, period_start),
            ).fetchone()
            overlap = row is not None
            if overlap and not allow_reuse:
                raise ValueError(
                    "holdout already consumed: an overlapping out-of-sample window was already "
                    "evaluated. Use fresh out-of-sample data, or --allow-holdout-reuse "
                    "(--actor human) to override and accept the statistical cost.")
            reused = bool(overlap)  # only reachable here with allow_reuse when overlap is True
            # Test-only hook to widen the critical section so the barriered concurrency test can
            # force two reserves to actually overlap. Default 0; NOT surfaced in CLI help/docs.
            delay_ms = int(os.environ.get("ALGUA_TEST_RESERVE_DELAY_MS", "0"))
            if delay_ms:
                import time
                time.sleep(delay_ms / 1000.0)
            cur = self._conn.execute(
                "INSERT INTO holdout_evaluations"
                "(strategy_id, data_source, snapshot_id, period_start, period_end, holdout_frac,"
                " config_hash, reused, created_at, committed_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,NULL)",
                (strategy_id, data_source, snapshot_id, period_start, period_end, holdout_frac,
                 "", int(reused), _now()),
            )
            self._conn.commit()
        except BaseException:
            self._conn.rollback()
            raise
        rowid = cur.lastrowid
        assert rowid is not None  # a successful INSERT always sets lastrowid
        return rowid, reused

    def finalize_holdout_reservation(self, reservation_id: int, *, config_hash: str) -> None:
        with self._conn:  # UPDATE + guard commit together or roll back
            cur = self._conn.execute(
                "UPDATE holdout_evaluations SET committed_at = ?, config_hash = ?"
                " WHERE id = ? AND committed_at IS NULL",
                (_now(), config_hash, reservation_id),
            )
            if cur.rowcount != 1:
                # Raise INSIDE the with so the mismatch rolls back (mirrors apply_transition's
                # gate-consume guard). Guards a double-finalize or a vanished/released row. Raise,
                # not assert — asserts strip under python -O.
                raise ValueError(
                    f"holdout reservation {reservation_id} is missing or already committed")

    def release_holdout_reservation(self, reservation_id: int) -> None:
        with self._conn:
            # No guard: a release after a finalize/crash is a harmless no-op (rowcount 0). Never
            # touches a committed burn (committed_at IS NULL filter).
            self._conn.execute(
                "DELETE FROM holdout_evaluations WHERE id = ? AND committed_at IS NULL",
                (reservation_id,),
            )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_registry_store.py -k "reserve or lifecycle or orphan or finalize or release_after" -v`
Expected: PASS (all)

- [ ] **Step 6: Run the full store + migration suite + gate**

Run: `uv run pytest tests/test_registry_store.py tests/test_db_migrations.py -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: PASS. (`mypy` confirms the Protocol/impl signatures match. `lint-imports` confirms no new cross-module imports.)

- [ ] **Step 7: Commit**

```bash
git add algua/registry/repository.py algua/registry/store.py tests/test_registry_store.py
git commit -m "feat(registry): atomic reserve/finalize/release holdout reservation (#161)"
```

---

## Task 4: CLI orchestration — reserve → run → finalize/release + JSON-contract guard

**Files:**
- Modify: `algua/cli/research_cmd.py` — promote decorator (line ~29) and orchestration body (lines ~103-122)
- Test: covered end-to-end by Task 5's subprocess test; this task verifies the wiring compiles + the unit suite stays green.

The `walk_forward` all-or-nothing assumption is **verified**: every raise site in `algua/backtest/walkforward.py` (lines 31, 33, 36, 43) plus `build_portfolio`/`_segment_bounds` precede the `holdout_metrics` computation (line 120); the function never computes holdout metrics and then raises. Release-on-failure therefore never frees a genuinely-peeked window.

- [ ] **Step 1: Add `sqlite3.OperationalError` to the promote JSON-error guard**

`@json_errors(...)` (`errors.py`) only converts the listed exception types into the `{"ok": false, "error": ...}` envelope; anything else escapes as a raw traceback, violating the JSON-stdout contract. A sqlite lock error on the promote path must stay parseable JSON. Add the import and extend the decorator.

In `research_cmd.py`, add to the imports:

```python
import sqlite3
```

Change the promote decorator (line ~29) from:

```python
@json_errors(ValueError, LookupError, BacktestError)
```

to:

```python
@json_errors(ValueError, LookupError, BacktestError, sqlite3.OperationalError)
```

- [ ] **Step 2: Replace the check→record orchestration**

Replace the block at `research_cmd.py:103-122` (from the `# Holdout-reuse pre-check` comment through the `repo.record_holdout_evaluation(...)` call) with:

```python
        # Atomic holdout reservation (#161): claim the window under the write lock (fast SELECT +
        # INSERT a pending row), run walk_forward with NO lock held, then finalize on success /
        # release on a clean failure. The match identity is the data window and deliberately
        # EXCLUDES the universe (the same OOS window is burned regardless of universe). A pending
        # reservation blocks a concurrent run exactly like a committed burn (fail closed).
        reservation_id, reused = repo.reserve_holdout(
            repo.get(name).id, data_source=data_source, snapshot_id=snapshot_id,
            period_start=period_start, period_end=period_end, holdout_frac=holdout_frac,
            allow_reuse=allow_holdout_reuse)  # raises here = fail closed (overlap, no reuse)
        try:
            wf = walk_forward(strategy, provider, start_dt, end_dt, windows=windows,
                              holdout_frac=holdout_frac, universe_by_date=universe_by_date,
                              universe_name=universe, universe_snapshots=universe_prov)
        except BaseException:
            repo.release_holdout_reservation(reservation_id)  # clean failure frees the window
            raise
        # Burn-on-peek: walk_forward has now computed holdout metrics, so commit the reservation
        # into a burn BEFORE the gate (mirrors today's record-before-gate ordering).
        repo.finalize_holdout_reservation(reservation_id, config_hash=wf.config_hash)
        outcome = run_gate(
            repo, wf, name=name, actor=actor_enum, criteria=criteria, breadth=breadth,
            universe_name=universe, universe_snapshots=universe_prov,
            period_start=start_dt.date(), period_end=end_dt.date(), holdout_frac=holdout_frac,
            data_source=data_source, snapshot_id=snapshot_id, allow_non_pit=allow_non_pit,
            reason_suffix=("; holdout_reuse=" + _HOLDOUT_REUSE_OVERRIDE) if reused else "")
        decision, promoted = outcome.decision, outcome.promoted
```

(Note: `reserve_holdout` is called at top level — only reads, `promotion_preflight`, precede it — so its `in_transaction` guard never trips here. The old explicit `if overlap and not allow_holdout_reuse: raise ValueError(...)` and the `reused = ...` line are now folded into `reserve_holdout`.)

- [ ] **Step 3: Run the gate to confirm the wiring is sound**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: PASS. (Any test that previously called `repo.record_holdout_evaluation` / `repo.overlapping_holdout_evaluations` outside `test_registry_store.py` will surface here — grep and migrate it: `grep -rn "record_holdout_evaluation\|overlapping_holdout_evaluations" tests/ algua/` should return nothing after this task.)

- [ ] **Step 4: Commit**

```bash
git add algua/cli/research_cmd.py
git commit -m "feat(cli): orchestrate atomic holdout reserve/finalize/release on promote (#161)"
```

---

## Task 5: Concurrency tests — the race proof

**Files:**
- Create: `tests/test_holdout_concurrency.py`

Three complementary tests (the subprocess form alone is insufficient — it can serialize by luck). Mirror the DB/strategy setup in `tests/test_e2e_lifecycle.py` and `tests/test_operator_layer.py` (check those for the `ALGUA_DB_PATH` env wiring and the `--demo` promote invocation).

- [ ] **Step 1: Write the sequential regression guard (deterministic, always meaningful)**

```python
import sqlite3
import threading

import pytest

from algua.registry.db import connect, migrate
from algua.registry.store import SqliteStrategyRepository


def _setup(tmp_path):
    db = tmp_path / "r.db"
    conn = connect(db)
    migrate(conn)
    sid = SqliteStrategyRepository(conn).add("race").id
    conn.close()
    return db, sid


def test_sequential_reserve_blocks_second_claim(tmp_path):
    """A committed pending row blocks a second overlapping reserve — fast, deterministic."""
    db, sid = _setup(tmp_path)
    c1 = connect(db)
    SqliteStrategyRepository(c1).reserve_holdout(
        sid, data_source="demo", snapshot_id=None,
        period_start="2022-01-01", period_end="2022-12-31", holdout_frac=0.2, allow_reuse=False)
    c2 = connect(db)
    with pytest.raises(ValueError, match="holdout already consumed"):
        SqliteStrategyRepository(c2).reserve_holdout(
            sid, data_source="demo", snapshot_id=None,
            period_start="2022-06-01", period_end="2023-06-01", holdout_frac=0.2,
            allow_reuse=False)
```

- [ ] **Step 2: Run it**

Run: `uv run pytest tests/test_holdout_concurrency.py::test_sequential_reserve_blocks_second_claim -v`
Expected: PASS

- [ ] **Step 3: Write the barriered true-concurrency proof**

```python
def test_barriered_concurrent_reserve_exactly_one_wins(tmp_path, monkeypatch):
    """Two threads aligned at a barrier both reserve the SAME window. The critical section is
    widened (ALGUA_TEST_RESERVE_DELAY_MS) so the sections actually overlap rather than serialize by
    luck. Assert the INVARIANT: exactly one non-reuse row exists and exactly one worker fails
    closed — proving BEGIN IMMEDIATE serialized the check+insert, not coincidence."""
    db, sid = _setup(tmp_path)
    monkeypatch.setenv("ALGUA_TEST_RESERVE_DELAY_MS", "50")
    barrier = threading.Barrier(2)
    results: dict[int, object] = {}

    def worker(i):
        conn = connect(db)
        repo = SqliteStrategyRepository(conn)
        barrier.wait()
        try:
            repo.reserve_holdout(
                sid, data_source="demo", snapshot_id=None,
                period_start="2022-01-01", period_end="2022-12-31", holdout_frac=0.2,
                allow_reuse=False)
            results[i] = "ok"
        except ValueError as exc:
            results[i] = exc
        finally:
            conn.close()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    oks = [v for v in results.values() if v == "ok"]
    fails = [v for v in results.values() if isinstance(v, ValueError)]
    assert len(oks) == 1, results
    assert len(fails) == 1 and "holdout already consumed" in str(fails[0])
    # The invariant: exactly one committed/pending non-reuse row for this window.
    conn = connect(db)
    n = conn.execute(
        "SELECT COUNT(*) AS n FROM holdout_evaluations WHERE strategy_id=? AND reused=0", (sid,),
    ).fetchone()["n"]
    assert n == 1
```

- [ ] **Step 4: Run it**

Run: `uv run pytest tests/test_holdout_concurrency.py::test_barriered_concurrent_reserve_exactly_one_wins -v`
Expected: PASS. (If it flakes toward "both ok", the delay is too short to force overlap — raise `ALGUA_TEST_RESERVE_DELAY_MS`; if it errors with `database is locked`, the `busy_timeout=5000` in `connect` should absorb the wait — confirm `connect` sets it.)

- [ ] **Step 5: Write the real-process subprocess e2e (the issue's literal ask)**

Mirror the promote invocation in `tests/test_e2e_lifecycle.py` (registry add → backtest/sweep to satisfy breadth → promote). Launch two `algua research promote` processes together on the same strategy/window, **no** `--allow-holdout-reuse`. Use the `--demo` path so no external data is needed; pick a `--windows`/window that yields a valid `walk_forward` so the winner finalizes.

```python
import os
import subprocess
import sys


def _run_promote(db_path, name, env_extra=None):
    env = {**os.environ, "ALGUA_DB_PATH": str(db_path)}
    if env_extra:
        env.update(env_extra)
    return subprocess.Popen(
        [sys.executable, "-m", "algua", "research", "promote", name,
         "--actor", "human", "--demo", "--n-combos", "8", "--allow-non-pit",
         "--universe", "demo", "--start", "2020-01-01", "--end", "2022-12-31"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, text=True)


def test_two_real_promote_processes_one_burns(tmp_path):
    """Two concurrent `algua research promote` runs on the same strategy/window: exactly one
    promotes, the other fails closed with the holdout-consumed error (parseable JSON, not a raw
    traceback). The issue's literal end-to-end ask."""
    # Setup: build the DB + a backtested strategy with enough recorded breadth to clear preflight.
    # (Reuse the helper sequence from tests/test_e2e_lifecycle.py — registry add, backtest sweep,
    # transition to `backtested`. Inline the exact commands that test uses for the --demo path.)
    db = tmp_path / "r.db"
    # ... (mirror test_e2e_lifecycle.py setup; left explicit there) ...
    name = "racer"
    p1 = _run_promote(db, name)
    p2 = _run_promote(db, name)
    out1, err1 = p1.communicate(timeout=120)
    out2, err2 = p2.communicate(timeout=120)
    codes = sorted([p1.returncode, p2.returncode])
    # Exactly one success (exit 0), one fail-closed (exit 1).
    assert codes == [0, 1], (out1, err1, out2, err2)
    combined = out1 + out2
    assert "holdout already consumed" in combined
    # The loser emitted parseable JSON, not a traceback.
    loser_out = out1 if p1.returncode == 1 else out2
    import json
    payload = json.loads(loser_out.strip().splitlines()[-1])
    assert payload["ok"] is False
```

> **Implementation note for the worker:** open `tests/test_e2e_lifecycle.py` and copy its exact DB-bootstrap + `backtested`-stage setup into the `# ...` block above so the strategy clears `promotion_preflight` (stage legality + breadth). The two `Popen`s must share one `ALGUA_DB_PATH`. If `-m algua` is not the module entrypoint, use the console-script form the other e2e tests use (`uv run algua ...` via `subprocess`).

- [ ] **Step 6: Run it**

Run: `uv run pytest tests/test_holdout_concurrency.py::test_two_real_promote_processes_one_burns -v`
Expected: PASS. (If both succeed, the setup serialized — confirm both `Popen`s truly launch before either blocks; the barriered test above is the tight-race proof, this is the end-to-end proof.)

- [ ] **Step 7: Run the whole concurrency file + full gate**

Run: `uv run pytest tests/test_holdout_concurrency.py -q && uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add tests/test_holdout_concurrency.py
git commit -m "test(registry): concurrency proofs for atomic holdout reservation (#161)"
```

---

## Task 6: Final verification + PR

- [ ] **Step 1: Full gate, clean run**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: all green; `lint-imports` "0 broken".

- [ ] **Step 2: Confirm no stale references to the removed methods**

Run: `grep -rn "record_holdout_evaluation\|overlapping_holdout_evaluations" algua/ tests/`
Expected: no output (both methods fully removed).

- [ ] **Step 3: Push + open the PR** (handled by the review-gated-development flow — GATE 2 review then `posting-review-to-pr` then `finishing-a-development-branch`).

---

## Self-Review (against the spec)

**Spec coverage:**
- §1 Schema (committed_at, no backfill, idempotent ALTER, version bump) → Tasks 1 + 2. ✓
- §2 busy_timeout → already present (drift note); no task needed. ✓
- §3 Store + Protocol (reserve/finalize/release, tx hygiene, overlap predicate verbatim, decision rules, schema-comment docs) → Task 3. ✓
- §4 Orchestration (reserve→run→finalize/release, finalize ordering, all-or-nothing verification, JSON-contract guard) → Task 4. ✓
- §5 Orphan policy (fail-closed, human unstick, listable) → enforced by the overlap-matches-all-rows predicate (Task 3) + documented in the schema comment (Task 2); covered by `test_orphaned_reservation_blocks`. ✓
- Testing §: store unit + lifecycle + guards (Task 3); sequential + barriered + subprocess concurrency (Task 5); JSON-contract guard (Task 4 + Task 5 subprocess assertion). ✓

**Type consistency:** `reserve_holdout` returns `tuple[int, bool]` (used as `reservation_id, reused`) in Protocol (3a), impl (3b), and CLI (Task 4) — consistent. `finalize_holdout_reservation(reservation_id, *, config_hash)` and `release_holdout_reservation(reservation_id)` signatures match across Protocol/impl/CLI. ✓

**Placeholder scan:** the only intentional `# ...` is in Task 5 Step 5, with an explicit instruction to copy the concrete setup from `tests/test_e2e_lifecycle.py` (the exact commands live there; duplicating them blind risks drift from that file's current `--demo` contract). Flagged, not silent. ✓
