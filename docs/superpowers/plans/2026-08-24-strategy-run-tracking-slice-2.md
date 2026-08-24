# Strategy Run Tracking — Slice 2 (read surface) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the run ledger readable — three pure CLI reads (`runs list` / `runs show` / `runs series`) and the three backend endpoints that serve them — so slice 3 can draw charts against real data.

**Architecture:** A new `algua/cli/runs_cmd.py` holds thin command bodies only; the projection logic lives in a new `algua/registry/run_views.py` so it is unit-testable without a CLI (the #165 domain-extraction convention that `ops_cmd.py` documents). Gate detail reuses the existing allowlist projection in `algua/registry/gate_history.py` — `decision_json` is never emitted raw. Backend endpoints follow the existing `run_cli(..., ttl_s=)` seam unchanged.

**Tech Stack:** Python 3.12, uv, typer, sqlite3, FastAPI (web backend), pytest, ruff, mypy, import-linter.

**Spec:** `docs/superpowers/specs/2026-08-23-strategy-run-tracking-design.md` (§5 read surface, §6 views 4–5)

**Base:** `origin/main` @ `2b9e89f` — slice 1 (#593) and the security fix (#594) are both merged.

## Global Constraints

- Quality gate on EVERY task before commit: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`. All four must pass.
- **When running the full suite, pass `timeout: 600000` explicitly to the Bash tool.** Without it the 120s default fires and the harness auto-backgrounds the command. This stalled agents repeatedly during slice 1.
- When a task touches `web/`, also run `uv run --project web pytest web/backend/tests -q`. **Never add web deps to the root project** — the root `uv.lock` is `dependency_hash` identity.
- `git add` scoped to named files — **never `git add -A`.** Other sessions share this repo.
- Commits end with: `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`
- **All three commands are PURE READS.** No broker call, no writes, no locks. The monitor stays strictly read-only (inherited from the 2026-08-15 spec).
- **`decision_json` is NEVER emitted raw.** `algua/registry/gate_history.py` defines `GATE_DECISION_ALLOWLIST` / `FORWARD_DECISION_ALLOWLIST` and the projection helpers; reuse them. Emitting the raw blob would leak fields the allowlist deliberately withholds.
- **`sort` is interpolated into SQL** by `list_runs`, which allow-lists it against `METRIC_COLUMNS`. The CLI must not add a second, looser path to that parameter.
- **Import contract, new since slice 1:** `cli command modules are independent of one another (no cli->cli sibling imports)`. `runs_cmd.py` may import `algua.cli._common`, `algua.cli.app`, `algua.cli.errors` — never a sibling `*_cmd.py`.
- **Payload size is a first-class concern.** `runs list` returns scalars only and must never include a return series; `runs series` is the only command that returns one. This is the `--summary` (#349) lesson: the CLI seam is a subprocess whose stdout is JSON-parsed.
- CODEOWNERS-protected paths this slice touches: **`/algua/registry/store/`** (Task 1 and Task 2). Expected; note it in the PR.
- Known hazard: some test writes a demo strategy file into `algua/strategies/momentum/`. If `git status` shows an untracked file there, delete it before staging.

## Design decisions made before this plan (do not relitigate)

**D1 — the series pointer lives on the run row, not a `run_id` FK on the series tables.**
`runs series <run_id>` needs a run→series link, and slice 1 shipped without one (the final whole-branch review flagged it; only the `gate_id` half was fixed). The obvious fix — adding `run_id` to `backtest_returns` and `holdout_returns` — is the wrong shape here for two reasons: `record_holdout_returns` is **idempotent on exact content** (`strategy_id, holdout_start, holdout_end, n_bars, returns_blob, bar_dates_blob`) and sits on the promote path, so widening its identity key is delicate; and it would mean two migrations on two tables instead of one on the table this work already owns. Instead: nullable `series_backtest_id` / `series_holdout_id` on `runs`, set by the writers, which already hold both ids at record time.

**D2 — `registry gates` is kept, not deleted.**
Spec §5 says `runs show` "Absorbs what `registry gates` does today". That is about the monitor's read surface, not about removing a CLI command: `registry gates <name>` is a *strategy-scoped* history across two ledgers, while `runs show <id>` is *one run's* detail. They answer different questions. Deleting a working command whose shape is not reproduced would be a regression, not a de-crufting.

**D3 — no backfill for the series pointers.** Consistent with spec Q8. Runs already written carry NULL and `runs series` reports them honestly as having no series, rather than guessing by provenance match (which is non-unique across re-runs — the same reason the final review rejected the fallback join).

---

## File Structure

| File | Responsibility |
|---|---|
| `algua/registry/db/runs.py` | **Modify.** Add `series_backtest_id` / `series_holdout_id` columns. |
| `algua/registry/db/constants.py` | **Modify.** `SCHEMA_VERSION` 43 → 44. |
| `algua/registry/db/migrate.py` | **Modify.** `_add_missing_columns` for the two new columns + a `# v44` comment. |
| `algua/registry/store/runs.py` | **Modify.** Accept the two ids through `record_run` / `_insert_run_locked`. No new read method — `run_views` reads the pointer columns off the row `get_run` already returns. |
| `algua/evaluation/backtest_run.py` | **Modify.** Persist the series BEFORE recording the run, pass its id. |
| `algua/registry/promotion.py` | **Modify.** Pass the holdout-returns id into the gate `run_row`. |
| `algua/registry/run_views.py` | **Create.** The three projections: `run_list_payload`, `run_detail_payload`, `run_series_payload`. Reads only through the passed repository (`repo.connection`); no writes, no locks, no subprocess. |
| `algua/cli/runs_cmd.py` | **Create.** Three thin typer commands. |
| `algua/cli/main.py` | **Modify.** Mount `runs_app` at the composition root. |
| `web/backend/main.py` | **Modify.** Three endpoints. |
| `tests/registry/test_run_views.py` | **Create.** Task 3. |
| `tests/test_cli_runs.py` | **Create.** Tasks 4–6. |
| `web/backend/tests/test_api.py` | **Modify.** Task 7. |

---

## Task 1: v44 — series pointers on the run row

**Files:**
- Modify: `algua/registry/db/runs.py`, `algua/registry/db/constants.py`, `algua/registry/db/migrate.py`
- Test: `tests/registry/test_runs_schema.py` (extend)

**Interfaces:**
- Consumes: the v43 `runs` table.
- Produces: `runs.series_backtest_id`, `runs.series_holdout_id` (both nullable INTEGER); `SCHEMA_VERSION == 44`.

- [ ] **Step 1: Write the failing test**

Append to `tests/registry/test_runs_schema.py`:

```python
def test_runs_has_series_pointer_columns() -> None:
    conn = _fresh()
    cols = {r[1] for r in conn.execute("PRAGMA table_info(runs)")}
    assert {"series_backtest_id", "series_holdout_id"} <= cols


def test_schema_version_is_44() -> None:
    conn = _fresh()
    assert SCHEMA_VERSION == 44
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 44


def test_v43_db_gains_series_pointer_columns(tmp_path) -> None:  # noqa: ANN001
    """A populated v43 DB must ALTER cleanly — the bootstrap cannot add a column."""
    import sqlite3 as _sq

    conn = _sq.connect(tmp_path / "legacy.db")
    conn.row_factory = _sq.Row
    migrate(conn)
    conn.execute("ALTER TABLE runs DROP COLUMN series_backtest_id")
    conn.execute("ALTER TABLE runs DROP COLUMN series_holdout_id")
    conn.execute(
        "INSERT INTO runs(kind, strategy_name, created_at, metric_schema_version,"
        " derived_from, components, config_json) VALUES ('backtest','a','t',1,'[]','[]','{}')")
    conn.execute("PRAGMA user_version=43")
    conn.commit()
    migrate(conn)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(runs)")}
    assert {"series_backtest_id", "series_holdout_id"} <= cols
    assert conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 1
```

Rename the existing `test_schema_version_is_43` rather than adding a duplicate — and while you are there, rename the three stragglers still called `test_schema_version_is_42` while asserting a newer value (`tests/test_family_registry.py`, `tests/registry/test_novel_family_seed_524.py`, `tests/registry/test_holdout_returns.py`). That naming drift was a deferred finding from slice 1's final review; a future bump greps for the current name and misses them.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/registry/test_runs_schema.py -q`
Expected: FAIL — the two columns do not exist and `SCHEMA_VERSION` is 43.

- [ ] **Step 3: Add the columns**

In `algua/registry/db/runs.py`, inside the `runs` DDL, after `gate_id INTEGER,`:

```sql
    -- Series pointers (slice 2, D1). The run row points AT its series rather than the series
    -- tables carrying a run_id: `record_holdout_returns` is idempotent on exact content and sits
    -- on the promote path, so widening its identity key is delicate, and this keeps the migration
    -- on the one table this feature owns. NULL is honest and common: an unregistered strategy's
    -- backtest records a run but no `backtest_returns` row, and rows written before v44 have no
    -- pointer at all (spec Q8 — no backfill; `runs series` reports "no series" rather than
    -- guessing by provenance, which is non-unique across re-runs).
    series_backtest_id INTEGER,
    series_holdout_id INTEGER,
```

- [ ] **Step 4: Bump the version and add the migration**

`algua/registry/db/constants.py`:

```python
# v44 (run tracking slice 2): series pointers on `runs` so `runs series <id>` can resolve.
SCHEMA_VERSION = 44
```

`algua/registry/db/migrate.py`, beside the v43 line:

```python
    # v44 (slice 2): series pointers on runs. Additive nullable; every pre-v44 row stays NULL
    # (no backfill — provenance matching is non-unique across re-runs).
    _add_missing_columns(
        conn, "runs", {"series_backtest_id": "INTEGER", "series_holdout_id": "INTEGER"})
```

- [ ] **Step 5: Recompute the pinned schema fingerprint**

`tests/test_registry_db.py` pins `_SCHEMA_OBJECT_COUNT` and `_SCHEMA_DIGEST`. Adding columns does **not** change the object count; the digest does change. **Recompute it by running the fingerprint routine — never edit it to match a failing assertion.** Update every `SCHEMA_VERSION == 43` assertion in the suite.

Also add the ALTER-migration test this file's own convention calls for (it has three: `test_migrate_adds_dependency_hash_column_to_legacy_approvals`, `..._to_legacy_stage_transitions`, `test_migrate_adds_search_trials_to_legacy_db`). Slice 1's `gate_id` ALTER shipped without one — a deferred finding. Cover both `gate_id` and the two new columns now, so deleting a `_add_missing_columns` line turns something red.

- [ ] **Step 6: Run the tests, then the full gate**

Run: `uv run pytest tests/registry/ tests/test_registry_db.py -q`, then the full four-command gate (Bash `timeout: 600000`).
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add algua/registry/db/runs.py algua/registry/db/constants.py algua/registry/db/migrate.py tests/registry/test_runs_schema.py tests/test_registry_db.py tests/test_family_registry.py tests/registry/test_novel_family_seed_524.py tests/registry/test_holdout_returns.py
git commit -m "feat(registry): v44 — series pointers on the run row"
```

---

## Task 2: Wire the series pointers at both write sites

**Files:**
- Modify: `algua/registry/store/runs.py`, `algua/evaluation/backtest_run.py`, `algua/registry/promotion.py`
- Test: `tests/test_runs_write_path.py` (extend)

**Interfaces:**
- Consumes: Task 1's columns.
- Produces: `record_run(..., series_backtest_id=None, series_holdout_id=None)`; both accepted through `_insert_run_locked` too; `RunLedger` Protocol updated to match.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_runs_write_path.py`:

```python
def test_backtest_run_points_at_its_series() -> None:
    """A REGISTERED strategy's backtest run resolves to the backtest_returns row it wrote."""
    from algua.contracts.lifecycle import Actor, Stage
    from algua.registry.transitions import transition_strategy

    with registry_conn() as conn:
        repo = SqliteStrategyRepository(conn)
        repo.add(STRATEGY)
        transition_strategy(repo, STRATEGY, Stage.BACKTESTED, Actor.AGENT, "seed")
    run_backtest_task(STRATEGY, demo=True)
    with registry_conn() as conn:
        row = SqliteStrategyRepository(conn).list_runs(kind="backtest")[0]
        assert row["series_backtest_id"] is not None
        series = conn.execute(
            "SELECT strategy_name FROM backtest_returns WHERE id=?",
            (row["series_backtest_id"],)).fetchone()
    assert series["strategy_name"] == STRATEGY


def test_unregistered_backtest_run_has_no_series_pointer() -> None:
    """NULL is the honest answer: no backtest_returns row is written for an unregistered
    strategy, and the run is still recorded."""
    run_backtest_task(STRATEGY, demo=True, register=False)
    with registry_conn() as conn:
        row = SqliteStrategyRepository(conn).list_runs(kind="backtest")[0]
    assert row["series_backtest_id"] is None


def test_gate_run_points_at_its_holdout_series() -> None:
    _promote(expect_pass=False)
    with registry_conn() as conn:
        row = SqliteStrategyRepository(conn).list_runs(kind="gate")[0]
        assert row["series_holdout_id"] is not None
        n = conn.execute(
            "SELECT n_bars FROM holdout_returns WHERE id=?",
            (row["series_holdout_id"],)).fetchone()["n_bars"]
    assert n > 0
```

Reuse the module's existing `STRATEGY` constant, autouse `_isolated_db` fixture, and `_promote` helper. Confirm the registration idiom against the helpers already in this file before writing the first test.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_runs_write_path.py -q`
Expected: FAIL — `series_backtest_id` / `series_holdout_id` are None.

- [ ] **Step 3: Accept the ids in the store**

In `algua/registry/store/runs.py`, add `series_backtest_id: int | None = None` and `series_holdout_id: int | None = None` to `record_run`'s keyword-only signature, to the `allowed` kwarg set that `_insert_run_locked` validates against, and to the column/value assembly. Mirror the change in the `RunLedger` Protocol in `algua/registry/repository.py` — slice 1's review confirmed the two are character-for-character identical today and that is worth preserving.

- [ ] **Step 4: Reorder the backtest write path**

`algua/evaluation/backtest_run.py` currently records the run and *then* persists the series. Invert that: persist first (still only for a registered strategy, unchanged), keep the returned row id, then pass it into `record_backtest_run`. Move the existing `if result.returns is not None:` block ABOVE the run-recording block, capture `series_id = repo.persist_backtest_returns(...)` (initialise `series_id = None` before the block), and pass `series_backtest_id=series_id`.

**Do not make the run row conditional on the series.** The run is recorded unconditionally — that is the pre-registration-evidence rule slice 1 established and two existing tests assert it.

Add `series_backtest_id` to `record_backtest_run`'s signature in `algua/registry/runs.py` and forward it.

- [ ] **Step 5: Pass the holdout id in promotion**

In `algua/registry/promotion.py`, `repo.record_holdout_returns(...)` already returns a row id in the `returns_available` branch — capture it (`holdout_returns_id`, initialised `None`) and add `"series_holdout_id": holdout_returns_id` to the `run_row` dict.

- [ ] **Step 6: Run the tests, then the full gate**

Run: `uv run pytest tests/test_runs_write_path.py tests/research -q`, then the full four-command gate (Bash `timeout: 600000`).
Expected: all pass, including the five pre-existing tests in the write-path file.

- [ ] **Step 7: Commit**

```bash
git add algua/registry/store/runs.py algua/registry/repository.py algua/registry/runs.py algua/evaluation/backtest_run.py algua/registry/promotion.py tests/test_runs_write_path.py
git commit -m "feat(registry): stamp series pointers on backtest and gate runs"
```

---

## Task 3: `run_views` — the pure projections

**Files:**
- Create: `algua/registry/run_views.py`
- Test: `tests/registry/test_run_views.py`

**Interfaces:**
- Consumes: `SqliteStrategyRepository.list_runs` / `get_run`; `algua/registry/gate_history.py`'s `GATE_DECISION_ALLOWLIST` and its projection helper.
- Produces:
  - `run_list_payload(repo, *, kind, strategy, family, sort, limit) -> dict`
  - `run_detail_payload(repo, run_id) -> dict`
  - `run_series_payload(repo, run_ids) -> dict`

All three take the repository ALONE and reach the connection through `repo.connection` (the
read-only handle `SqliteStrategyRepository` already exposes for exactly this). Do not add a
separate `conn` parameter — passing both invites a caller handing in a connection that is not the
repo's.

The logic lives here, not in the CLI, so it is unit-testable without a subprocess — the #165 convention `ops_cmd.py`'s docstring states.

- [ ] **Step 1: Write the failing test**

Create `tests/registry/test_run_views.py` covering, at minimum:

```python
"""run_views: shaping, allow-listing, and the payload-size contract."""
from __future__ import annotations

import sqlite3

import pytest

from algua.registry.db.migrate import migrate
from algua.registry.run_views import (
    run_detail_payload,
    run_list_payload,
    run_series_payload,
)
from algua.registry.store import SqliteStrategyRepository


@pytest.fixture()
def repo() -> SqliteStrategyRepository:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    migrate(conn)
    return SqliteStrategyRepository(conn)


def test_list_payload_never_contains_a_series(repo: SqliteStrategyRepository) -> None:
    """The payload-size contract: `runs list` returns scalars only."""
    repo.record_run("backtest", "alpha", metrics={"sharpe_is": 1.0})
    payload = run_list_payload(repo, kind=None, strategy=None, family=None, sort=None, limit=10)
    (row,) = payload["runs"]
    # Asserted structurally, not by substring: a repr() grep passes on the wrong thing the moment
    # a column is renamed. series_backtest_id is an INT POINTER and is allowed here; a series
    # PAYLOAD is not.
    assert "returns" not in row
    assert "returns_json" not in row
    assert isinstance(row.get("series_backtest_id"), (int, type(None)))
    for key, value in row.items():
        if isinstance(value, list):
            assert key in {"derived_from", "components"}, f"{key} leaked a list into runs list"


def test_list_payload_rejects_a_non_vocabulary_sort(repo: SqliteStrategyRepository) -> None:
    with pytest.raises(ValueError, match="not a sortable metric"):
        run_list_payload(
            repo, kind=None, strategy=None, family=None, sort="1; DROP TABLE runs", limit=10)


def test_detail_payload_of_a_missing_run_is_an_error(repo: SqliteStrategyRepository) -> None:
    with pytest.raises(ValueError, match="no run"):
        run_detail_payload(repo, 9999)


def test_detail_payload_parses_lineage_as_lists(repo: SqliteStrategyRepository) -> None:
    parent = repo.record_run("walk_forward", "alpha")
    child = repo.record_run("gate", "alpha", derived_from=[parent])
    payload = run_detail_payload(repo, child)
    assert payload["derived_from"] == [parent]
    assert payload["components"] == []


def test_series_payload_reports_a_run_with_no_series_honestly(
    repo: SqliteStrategyRepository,
) -> None:
    run_id = repo.record_run("backtest", "alpha")
    payload = run_series_payload(repo, [run_id])
    assert payload["series"][str(run_id)] is None
```

Add a test asserting a `gate` run's detail carries its checks and that **no key outside `GATE_DECISION_ALLOWLIST` appears** — read `algua/registry/gate_history.py` first and assert against the real allowlist rather than a hand-copied list.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/registry/test_run_views.py -q`
Expected: FAIL — `ModuleNotFoundError: algua.registry.run_views`.

- [ ] **Step 3: Implement the module**

Write `algua/registry/run_views.py`. Requirements, not a transcription — implement them against the real helpers you find:

- `run_list_payload` returns `{"runs": [...], "count": n}` where each row is the run's scalar columns with `derived_from` / `components` / `config_json` parsed from JSON. **No series, ever.** Pass `sort` straight to `list_runs` so its allow-list is the single gate. `family` filters by joining the strategy's family (find how `registry list` resolves family and reuse it; if there is no cheap join, filter in Python over the returned rows and say so in a comment).
- `run_detail_payload` returns one run plus: its `run_metrics` overflow rows as a dict; its parsed lineage; and, when `kind == "gate"` and `gate_id` is not NULL, the **allow-list-projected** gate decision from `gate_history.py`. Raise `ValueError(f"no run {run_id}")` when absent — the CLI's `@json_errors` turns that into the standard envelope.
- `run_series_payload` takes run ids, resolves `series_backtest_id` / `series_holdout_id`, and returns `{"series": {run_id: {...} | None}}`. A run with no pointer maps to `None`, not an empty list — the two mean different things.

- [ ] **Step 4: Run the tests, then the full gate**

Run: `uv run pytest tests/registry/test_run_views.py -q`, then the full four-command gate (Bash `timeout: 600000`).

- [ ] **Step 5: Commit**

```bash
git add algua/registry/run_views.py tests/registry/test_run_views.py
git commit -m "feat(registry): run_views — pure projections for the read surface"
```

---

## Task 4: `algua runs list`

**Files:**
- Create: `algua/cli/runs_cmd.py`
- Modify: `algua/cli/main.py`
- Test: `tests/test_cli_runs.py`

**Interfaces:**
- Consumes: `run_list_payload`.
- Produces: `runs_app` typer group mounted at the composition root; `algua runs list` emitting the standard `ok()` envelope.

- [ ] **Step 1: Write the failing test**

Create `tests/test_cli_runs.py` with the established idioms: an autouse fixture setting `ALGUA_DB_PATH` to a tmp path, and `CliRunner().invoke(app, [...])` with `from algua.cli.main import app`. Cover: an empty ledger returns `ok` with zero rows (not an error); `--kind` filters; `--sort sharpe_oos` orders best-first with NULLs last; `--limit` caps; a bad `--sort` exits non-zero with the JSON error envelope.

- [ ] **Step 2: Run test to verify it fails**

Expected: FAIL — no `runs` command group.

- [ ] **Step 3: Implement**

Create `algua/cli/runs_cmd.py` following `algua/cli/ops_cmd.py`'s shape exactly: module docstring stating the commands are pure reads and that the logic lives in `run_views`; `runs_app = typer.Typer(help=..., no_args_is_help=True)`; `@runs_app.command("list")` + `@json_errors`; body opens `registry_conn()`, builds the repo, calls the view, `emit(ok(payload))`.

Mount it in `algua/cli/main.py` the way the other groups are mounted. **`runs_cmd.py` must not import any sibling `*_cmd.py`** — the `cli command modules are independent of one another` contract is enforced by `lint-imports`.

Flags: `--kind`, `--strategy`, `--family`, `--sort`, `--limit` (default 100, `min=1`).

- [ ] **Step 4: Run the tests, then the full gate**

- [ ] **Step 5: Commit**

```bash
git add algua/cli/runs_cmd.py algua/cli/main.py tests/test_cli_runs.py
git commit -m "feat(cli): algua runs list"
```

---

## Task 5: `algua runs show`

**Files:** Modify `algua/cli/runs_cmd.py`; extend `tests/test_cli_runs.py`.

- [ ] **Step 1: Write the failing test** — a `gate` run's detail includes its checks; a missing run id exits non-zero with the error envelope; **no key outside the gate allowlist appears in the output**.
- [ ] **Step 2: Run test to verify it fails**
- [ ] **Step 3: Implement** `@runs_app.command("show")` taking a positional `run_id: int`, calling `run_detail_payload`.
- [ ] **Step 4: Run the tests, then the full gate**
- [ ] **Step 5: Commit** — `feat(cli): algua runs show`

---

## Task 6: `algua runs series`

**Files:** Modify `algua/cli/runs_cmd.py`; extend `tests/test_cli_runs.py`.

- [ ] **Step 1: Write the failing test** — one run id returns its series; several return several; a run with no pointer returns `null` for that id rather than being omitted (an omitted key is indistinguishable from a typo'd id); `runs list` output still contains no series.
- [ ] **Step 2: Run test to verify it fails**
- [ ] **Step 3: Implement** `@runs_app.command("series")` taking a positional `run_id: int` plus a repeatable `--run-id` option, de-duplicated, calling `run_series_payload`. Cap the number of ids accepted (**16**) and fail closed above it with a message naming the cap — this command is the one that can return megabytes through a JSON-parsed subprocess pipe.
- [ ] **Step 4: Run the tests, then the full gate**
- [ ] **Step 5: Commit** — `feat(cli): algua runs series`

---

## Task 7: Backend endpoints

**Files:** Modify `web/backend/main.py`; extend `web/backend/tests/test_api.py`.

- [ ] **Step 1: Write the failing test** — `/api/runs`, `/api/runs/{id}`, `/api/runs/series?ids=` each return the CLI payload through the existing `run_cli` seam. Follow the existing endpoint tests' patterns for faking the CLI.
- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --project web pytest web/backend/tests -q`

- [ ] **Step 3: Implement** the three endpoints exactly as the existing ones are written — `run_cli("runs", "list", ..., ttl_s=60.0)`, `run_cli("runs", "show", str(run_id), ttl_s=30.0)`, `run_cli("runs", "series", ..., ttl_s=60.0)`. No new plumbing; the `run_cli` adapter consumes these unchanged because they emit the standard envelope.

Query params on `/api/runs` mirror the CLI flags. Validate `ids` on the series endpoint before shelling out, and enforce the same 16-id cap.

- [ ] **Step 4: Run both suites**

Run: `uv run --project web pytest web/backend/tests -q`, then the root four-command gate.

- [ ] **Step 5: Commit**

```bash
git add web/backend/main.py web/backend/tests/test_api.py
git commit -m "feat(web): /api/runs, /api/runs/{id}, /api/runs/series"
```

---

## Task 8: Verify the slice end to end

**Files:** none modified.

- [ ] **Step 1: Full gate from clean** (Bash `timeout: 600000`), plus `uv run --project web pytest web/backend/tests -q`.
- [ ] **Step 2: Drive the real CLI against a scratch DB.** **Every command must set `ALGUA_DB_PATH=/tmp/runs2-check.db`** — the default is the operator's real registry.

```bash
ALGUA_DB_PATH=/tmp/runs2-check.db uv run algua backtest run cross_sectional_momentum --demo
ALGUA_DB_PATH=/tmp/runs2-check.db uv run algua backtest sweep cross_sectional_momentum --demo --param lookback=20,40,60
ALGUA_DB_PATH=/tmp/runs2-check.db uv run algua runs list --kind sweep_trial
ALGUA_DB_PATH=/tmp/runs2-check.db uv run algua runs list --sort sharpe_is --limit 5
```

- [ ] **Step 3: Confirm** each emits valid JSON under the `ok()` envelope, that `runs list` output contains no return series, and that `runs show` on the backtest run resolves its `series_backtest_id`. Report the observed counts.
- [ ] **Step 4: Clean up** (`rm -f /tmp/runs2-check.db`) and report `git log --oneline origin/main..HEAD`.

---

## Out of scope

- **Slice 3** — the five views and the Research-screen rebuild. Still deliberately unplanned; it also owes an answer to the categorical-palette problem (Electric is reserved as the rare signal, green/red/amber/violet for status, so a 4-series overlay has no legal colours).
- **Backfilling series pointers** onto pre-v44 runs (D3).
- **Deleting `registry gates`** (D2).
- **The MLflow FileStore deprecation** — mlflow 3.15 puts the `mlruns` backend in maintenance mode; that decision belongs with the component-tracking layer, not here.
