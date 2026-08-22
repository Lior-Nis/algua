# Strategy Run Tracking — Slice 1 (v42 schema + write path) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every strategy evaluation — backtest, walk-forward, sweep, each sweep trial, and each gate decision — a first-class queryable row with a fixed metric vocabulary, so the funnel stops discarding its own evidence.

**Architecture:** Three layers mirroring the existing search-breadth stack exactly: a DDL fragment (`algua/registry/db/runs.py`, like `db/breadth.py`), a repository mixin (`algua/registry/store/runs.py`, like `store/search_breadth.py`), and a recorder seam (`algua/registry/runs.py`, like `registry/search_breadth.py`) that the CLI and promotion call. Nothing new is invented; the pattern is copied.

**Tech Stack:** Python 3.12, uv, sqlite3 (WAL, `busy_timeout=5000`), pytest, ruff, mypy, import-linter.

**Spec:** `docs/superpowers/specs/2026-08-23-strategy-run-tracking-design.md`

## Global Constraints

- Quality gate on EVERY task before commit: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`. All four must pass.
- **When running the full suite, pass `timeout: 600000` explicitly to the Bash tool.** Without it the 120s default fires and the harness auto-backgrounds the command.
- `git add` scoped to named files — **never `git add -A`**. Concurrent sessions share this repo and will have unrelated untracked work.
- Commits end with:
  `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`
- **Runs are keyed by strategy NAME, with a nullable `strategy_id`.** This is the same rationale `db/breadth.py` documents for `search_trials`: exploration precedes registration, and a run of an unregistered strategy must still be recorded. Never make `strategy_name` a FK.
- **A sentinel metric is NULL, never a number.** `metrics_from_returns` returns `0.0` for undefined Sharpe/Sortino/Calmar. Recording those as `0.0` would sort a degenerate run above a genuine negative one. See Task 3 for the exact detection rule.
- **No metric column is named bare `sharpe`.** Every metric carries its sample class (`_is` / `_oos` / `_realized`). This is a correctness rule (spec §3.1), enforced by a test in Task 1.
- **`derived_from` and `components` are JSON lists, always** — never a scalar, never NULL. Empty is `'[]'`. The singular `model_ref` gap (spec §2.1) must not be baked in here.
- CODEOWNERS-protected paths this slice touches: **`/algua/registry/store/`** (Tasks 2, 3, 7 — "the paper→live wall + approvals") and **`/algua/registry/promotion.py`** (Task 7 — "promotion policy"). Both are on the promotion wall. Expected; note it in the PR so human review is anticipated.
- The canonical demo strategy in CLI tests is **`cross_sectional_momentum`** with `--demo`; the DB-isolation idiom is `monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))` (see `tests/test_cli_backtest.py`, `tests/test_cli_sweep.py`). Use both; do not invent new fixtures.
- `sharpe_realized` is in the vocabulary but **nothing writes it in this slice** — it needs broker-clocked forward data, and `forward_gate_evaluations` is empty. The column exists so the vocabulary is complete; its writer arrives with the forward lane. This is not a missing task.
- Known hazard: some test writes a demo strategy file into `algua/strategies/momentum/`. If `git status` shows an untracked file there, delete it before staging.

---

## File Structure

| File | Responsibility |
|---|---|
| `algua/registry/db/runs.py` | **Create.** DDL fragment: `runs` + `run_metrics` tables and their indexes. No Python logic. |
| `algua/registry/db/schema.py` | **Modify.** Import and concatenate `RUNS_SCHEMA`. |
| `algua/registry/db/constants.py` | **Modify.** `SCHEMA_VERSION` 41 → 42; add `METRIC_SCHEMA_VERSION`, `MAX_PERSISTED_TRIALS`. |
| `algua/registry/store/runs.py` | **Create.** `RunLedgerMixin` — `record_run`, `record_run_metrics`, `record_sweep_trials`, `get_run`, `list_runs`. The only module with `runs` SQL. |
| `algua/registry/store/__init__.py` | **Modify.** Compose `RunLedgerMixin` into `SqliteStrategyRepository`. |
| `algua/registry/runs.py` | **Create.** Pure metric mapping (`backtest_metrics`, `walk_forward_metrics`, `trial_metrics`) + the `record_*_run` recorder seams the CLI calls. Caller-owned transaction, exactly like `registry/search_breadth.py`. |
| `algua/cli/backtest_cmd.py` | **Modify.** Call the recorder from `run_backtest_task`, the walk-forward command, and the sweep command. |
| `algua/registry/promotion.py` | **Modify.** Record the gate run inside the existing atomic transaction, linked to the walk-forward run. |
| `tests/registry/test_runs_schema.py` | **Create.** Task 1. |
| `tests/registry/test_runs_store.py` | **Create.** Task 2. |
| `tests/registry/test_runs_metrics.py` | **Create.** Task 3. |
| `tests/test_runs_write_path.py` | **Create.** Tasks 4–7 (CLI-level, alongside the other `tests/test_backtest_*.py` files). |

---

## Task 1: Schema — `runs` + `run_metrics` at v42

**Files:**
- Create: `algua/registry/db/runs.py`
- Modify: `algua/registry/db/schema.py`
- Modify: `algua/registry/db/constants.py`
- Test: `tests/registry/test_runs_schema.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `algua.registry.db.runs.SCHEMA` (str); `algua.registry.db.constants.SCHEMA_VERSION == 42`, `METRIC_SCHEMA_VERSION == 1`, `MAX_PERSISTED_TRIALS == 10_000`.

- [ ] **Step 1: Write the failing test**

Create `tests/registry/test_runs_schema.py`:

```python
"""v42 schema: the runs ledger tables exist, are correctly shaped, and migrate idempotently."""
from __future__ import annotations

import sqlite3

from algua.registry.db import SCHEMA_VERSION
from algua.registry.db.migrate import migrate

# Every metric column must name its sample class. Spec §3.1: a bare `sharpe` column would let the
# UI sort the most overfit number in the system to the top.
_SAMPLE_SUFFIXES = ("_is", "_oos", "_realized", "_window_sharpe", "_positive_windows")
_METRIC_PREFIXES = ("sharpe", "sortino", "total_return", "max_drawdown", "ann_vol",
                    "cagr", "calmar", "n_obs")


def _fresh() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    migrate(conn)
    return conn


def test_runs_tables_exist() -> None:
    conn = _fresh()
    names = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    assert "runs" in names
    assert "run_metrics" in names


def test_schema_version_is_42() -> None:
    conn = _fresh()
    assert SCHEMA_VERSION == 42
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 42


def test_no_bare_sharpe_column() -> None:
    """No metric column may omit its sample class (spec §3.1)."""
    conn = _fresh()
    cols = [r[1] for r in conn.execute("PRAGMA table_info(runs)")]
    for col in cols:
        if any(col.startswith(p) for p in _METRIC_PREFIXES):
            assert col.endswith(_SAMPLE_SUFFIXES), f"{col} does not name its sample class"


def test_kind_check_rejects_unknown() -> None:
    conn = _fresh()
    try:
        conn.execute(
            "INSERT INTO runs(kind, strategy_name, created_at, metric_schema_version,"
            " derived_from, components, config_json) VALUES ('nonsense','s','t',1,'[]','[]','{}')")
    except sqlite3.IntegrityError:
        return
    raise AssertionError("kind CHECK did not reject an unknown kind")


def test_migrate_is_idempotent() -> None:
    conn = _fresh()
    migrate(conn)
    migrate(conn)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 42


def test_migrates_a_legacy_db_lacking_runs() -> None:
    """A DB stamped at v41 with no runs table gains one without losing data."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    migrate(conn)
    conn.execute("DROP TABLE runs")
    conn.execute("DROP TABLE run_metrics")
    conn.execute("PRAGMA user_version=41")
    migrate(conn)
    names = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"runs", "run_metrics"} <= names
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/registry/test_runs_schema.py -q`
Expected: FAIL — `assert "runs" in names`, and `SCHEMA_VERSION == 42` is False (it is 41).

- [ ] **Step 3: Create the DDL fragment**

Create `algua/registry/db/runs.py`:

```python
"""Run-ledger context: ``runs`` + ``run_metrics``.

Every evaluation of a strategy — backtest, walk-forward, sweep, each sweep trial, and each gate
decision — is one ``runs`` row with a FIXED metric vocabulary in real columns (operated by
``algua/registry/store/runs.py``). This is the economic layer: governance-bearing and
audit-bearing, as distinct from the component (model-training) layer, which lives in MLflow and is
best-effort. See the 2026-08-23 spec §2.

NB: the ``metric_schema_version`` column pins which vocabulary generation a row was written under,
so the vocabulary can evolve without silently changing what an existing chart means. It is
``algua.registry.db.constants.METRIC_SCHEMA_VERSION``, duplicated into every row rather than
assumed globally.
"""
from __future__ import annotations

SCHEMA = """
-- runs is the economic-layer evaluation ledger. One row per evaluation.
-- KEYED BY strategy NAME (free text) with a NULLABLE strategy_id, NOT a strategies(id) FK, ON
-- PURPOSE — the same rationale search_trials documents: exploration precedes registration, and a
-- backtest or sweep of a not-yet-registered strategy must still be recorded. Keying by id would
-- silently drop pre-registration evidence, which is exactly the evidence the breadth tax counts.
-- METRIC NAMING IS A CORRECTNESS RULE, NOT A STYLE: every metric column names its sample class
-- (_is / _oos / _realized). There are four different Sharpes in this system and they are not
-- comparable; a bare `sharpe` column would let a UI sort the most overfit number to the top.
-- A metric that is UNDEFINED for a run is NULL, never 0.0 — metrics_from_returns returns 0.0
-- sentinels for degenerate series, and recording those as zero would rank them above a genuine
-- negative. The writer maps sentinels to NULL; see algua/registry/runs.py.
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL CHECK (kind IN
        ('backtest', 'walk_forward', 'sweep', 'sweep_trial', 'gate')),
    strategy_name TEXT NOT NULL,
    strategy_id INTEGER,
    created_at TEXT NOT NULL,
    metric_schema_version INTEGER NOT NULL,

    -- Lineage. BOTH ARE JSON LISTS, ALWAYS — '[]' when empty, never NULL, never a scalar.
    -- derived_from: parent run ids in the economic layer (a gate run points at the walk-forward
    --   run it evaluated; a sweep_trial points at its parent sweep).
    -- components: component refs in the MODEL layer (name/version/digest/training_as_of). A list
    --   from day one: a strategy composed of several models must be expressible without a
    --   migration, even though strategies/base.py currently permits only one (spec §2.1).
    derived_from TEXT NOT NULL DEFAULT '[]',
    components TEXT NOT NULL DEFAULT '[]',

    -- Provenance, mirroring BacktestResult's fields one-for-one.
    code_hash TEXT,
    config_hash TEXT,
    dependency_hash TEXT,
    data_source TEXT,
    snapshot_id TEXT,
    universe_name TEXT,
    fundamentals_snapshot TEXT,
    news_snapshot TEXT,
    delisting_snapshot TEXT,
    seed INTEGER,
    timeframe TEXT,
    period_start TEXT,
    period_end TEXT,

    -- Free-form: grid points are heterogeneous by nature and freezing them is pointless.
    config_json TEXT NOT NULL DEFAULT '{}',

    -- Fixed metric vocabulary v1.
    sharpe_is REAL,
    sharpe_oos REAL,
    sharpe_realized REAL,
    sortino_is REAL,
    sortino_oos REAL,
    total_return_is REAL,
    total_return_oos REAL,
    max_drawdown_is REAL,
    max_drawdown_oos REAL,
    ann_vol_is REAL,
    ann_vol_oos REAL,
    cagr_is REAL,
    calmar_is REAL,
    n_obs_is INTEGER,
    n_obs_oos INTEGER,
    mean_window_sharpe REAL,
    std_window_sharpe REAL,
    min_window_sharpe REAL,
    pct_positive_windows REAL,

    -- Gate outcome (NULL for non-gate kinds).
    passed INTEGER,
    -- Set on a `sweep` parent when its trial rows were capped at MAX_PERSISTED_TRIALS. NULL means
    -- the trial set is COMPLETE. A silently truncated set would make the funnel-wide distribution
    -- lie about the breadth it depicts, so a reader must be able to tell.
    trials_truncated_at INTEGER
);
CREATE INDEX IF NOT EXISTS ix_runs_strategy ON runs(strategy_name);
CREATE INDEX IF NOT EXISTS ix_runs_kind_created ON runs(kind, created_at);
CREATE INDEX IF NOT EXISTS ix_runs_sharpe_oos ON runs(sharpe_oos);

-- run_metrics is the OVERFLOW key-value tail: the ~40 DSR/IR/regime diagnostics and whatever a
-- future model-bearing run wants. Queryable, but deliberately NOT part of the fixed vocabulary
-- and NOT offered as a default chart axis — a metric that matters enough to sort by should earn
-- a real column and a metric_schema_version bump.
CREATE TABLE IF NOT EXISTS run_metrics (
    run_id INTEGER NOT NULL REFERENCES runs(id),
    key TEXT NOT NULL,
    value REAL,
    PRIMARY KEY (run_id, key)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS ix_run_metrics_key ON run_metrics(key);
"""
```

- [ ] **Step 4: Wire the fragment into the assembled schema**

In `algua/registry/db/schema.py`, add the import next to the others (alphabetical among them):

```python
from algua.registry.db.runs import SCHEMA as RUNS_SCHEMA
```

and add `RUNS_SCHEMA,` to the `SCHEMA = "\n".join([...])` list, after `BACKTEST_RETURNS_SCHEMA,`.

- [ ] **Step 5: Bump the version and add the two new constants**

In `algua/registry/db/constants.py`, change `SCHEMA_VERSION = 41` to:

```python
# v42 (strategy run tracking): runs + run_metrics — the economic-layer evaluation ledger.
SCHEMA_VERSION = 42
```

and append at the end of the file:

```python
# The generation of the FIXED metric vocabulary in `runs`. Stamped into every row so the
# vocabulary can evolve without silently changing what an existing chart means: a chart that
# needs v1 semantics filters on it rather than assuming. Bumping this means adding or
# re-defining a metric COLUMN, and therefore a SCHEMA_VERSION bump too.
METRIC_SCHEMA_VERSION = 1

# Per-sweep upper bound on PERSISTED sweep_trial rows. MAX_N_COMBOS above is a 1e9 overflow guard,
# not a realistic grid size; harmless while a trial was a scalar, but once each trial is a row it
# becomes a row-count bomb. Beyond this cap the writer keeps the search_trials aggregate (which
# still governs breadth) and stamps `trials_truncated_at` on the parent sweep run, so a reader can
# never mistake a truncated trial set for a complete one.
MAX_PERSISTED_TRIALS = 10_000
```

Then re-export both from `algua/registry/db/__init__.py`, alongside the existing `SCHEMA_VERSION as SCHEMA_VERSION` line:

```python
from algua.registry.db.constants import (
    MAX_PERSISTED_TRIALS as MAX_PERSISTED_TRIALS,
)
from algua.registry.db.constants import (
    METRIC_SCHEMA_VERSION as METRIC_SCHEMA_VERSION,
)
```

and add `"MAX_PERSISTED_TRIALS"` and `"METRIC_SCHEMA_VERSION"` to that module's `__all__`.

- [ ] **Step 6: Run the test to verify it passes**

Run: `uv run pytest tests/registry/test_runs_schema.py -q`
Expected: PASS, 6 tests.

- [ ] **Step 7: Run the full quality gate**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports` (Bash `timeout: 600000`)
Expected: all four pass. If a pre-existing test asserts `SCHEMA_VERSION == 41`, update it — that is the version bump landing, not a regression.

- [ ] **Step 8: Commit**

```bash
git add algua/registry/db/runs.py algua/registry/db/schema.py algua/registry/db/constants.py algua/registry/db/__init__.py tests/registry/test_runs_schema.py
git commit -m "feat(registry): v42 runs + run_metrics ledger schema"
```

---

## Task 2: `RunLedgerMixin` — the only module with `runs` SQL

**Files:**
- Create: `algua/registry/store/runs.py`
- Modify: `algua/registry/store/__init__.py`
- Test: `tests/registry/test_runs_store.py`

**Interfaces:**
- Consumes: Task 1's tables; `algua.registry.db.METRIC_SCHEMA_VERSION`, `MAX_PERSISTED_TRIALS`; `algua.registry.store._util._now`.
- Produces:
  - `RunLedgerMixin.record_run(kind, strategy_name, *, strategy_id=None, derived_from=None, components=None, provenance=None, config=None, metrics=None, extra_metrics=None, passed=None, trials_truncated_at=None) -> int`
  - `RunLedgerMixin.record_sweep_trials(parent_run_id, strategy_name, trials) -> tuple[int, int | None]` returning `(n_written, truncated_at)`
  - `RunLedgerMixin.get_run(run_id) -> sqlite3.Row | None`
  - `RunLedgerMixin.list_runs(*, kind=None, strategy_name=None, sort=None, limit=100) -> list[sqlite3.Row]`

Every method name is unique across the composed mixins — `tests/test_registry_store.py::test_no_mixin_shadows_another` enforces this.

- [ ] **Step 1: Write the failing test**

Create `tests/registry/test_runs_store.py`:

```python
"""RunLedgerMixin: insert + read back, JSON-list lineage, overflow metrics, trial capping."""
from __future__ import annotations

import json
import sqlite3

import pytest

from algua.registry.db.migrate import migrate
from algua.registry.store import SqliteStrategyRepository


@pytest.fixture()
def repo() -> SqliteStrategyRepository:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    migrate(conn)
    return SqliteStrategyRepository(conn)


def test_record_run_round_trips(repo: SqliteStrategyRepository) -> None:
    run_id = repo.record_run(
        "backtest", "alpha",
        provenance={"code_hash": "abc", "period_start": "2020-01-01"},
        metrics={"sharpe_is": 1.25, "n_obs_is": 500},
        config={"lookback": 60},
    )
    row = repo.get_run(run_id)
    assert row is not None
    assert row["kind"] == "backtest"
    assert row["strategy_name"] == "alpha"
    assert row["code_hash"] == "abc"
    assert row["sharpe_is"] == pytest.approx(1.25)
    assert row["n_obs_is"] == 500
    assert json.loads(row["config_json"]) == {"lookback": 60}
    assert row["metric_schema_version"] == 1


def test_lineage_defaults_to_empty_lists(repo: SqliteStrategyRepository) -> None:
    row = repo.get_run(repo.record_run("backtest", "alpha"))
    assert row is not None
    assert json.loads(row["derived_from"]) == []
    assert json.loads(row["components"]) == []


def test_lineage_round_trips_as_lists(repo: SqliteStrategyRepository) -> None:
    parent = repo.record_run("walk_forward", "alpha")
    child = repo.record_run(
        "gate", "alpha",
        derived_from=[parent],
        components=[{"name": "sentiment", "version": 3, "digest": "d0"}],
    )
    row = repo.get_run(child)
    assert row is not None
    assert json.loads(row["derived_from"]) == [parent]
    assert json.loads(row["components"])[0]["name"] == "sentiment"


def test_unknown_metric_key_is_rejected(repo: SqliteStrategyRepository) -> None:
    """A typo'd metric must not vanish silently into nowhere."""
    with pytest.raises(ValueError, match="not in the fixed metric vocabulary"):
        repo.record_run("backtest", "alpha", metrics={"sharpe": 1.0})


def test_extra_metrics_land_in_the_overflow_table(repo: SqliteStrategyRepository) -> None:
    run_id = repo.record_run(
        "gate", "alpha", extra_metrics={"dsr_confidence": 0.096, "market_beta": 0.21})
    rows = {r["key"]: r["value"] for r in repo.connection.execute(
        "SELECT key, value FROM run_metrics WHERE run_id=?", (run_id,))}
    assert rows["dsr_confidence"] == pytest.approx(0.096)
    assert rows["market_beta"] == pytest.approx(0.21)


def test_non_finite_extra_metric_is_stored_as_null(repo: SqliteStrategyRepository) -> None:
    run_id = repo.record_run("gate", "alpha", extra_metrics={"dsr_n_eff": float("nan")})
    row = repo.connection.execute(
        "SELECT value FROM run_metrics WHERE run_id=? AND key='dsr_n_eff'", (run_id,)).fetchone()
    assert row["value"] is None


def test_record_sweep_trials_writes_children(repo: SqliteStrategyRepository) -> None:
    parent = repo.record_run("sweep", "alpha")
    trials = [
        {"config": {"lookback": lb}, "metrics": {"mean_window_sharpe": float(i)}}
        for i, lb in enumerate([60, 90, 120])
    ]
    n, truncated = repo.record_sweep_trials(parent, "alpha", trials)
    assert (n, truncated) == (3, None)
    kids = repo.list_runs(kind="sweep_trial", strategy_name="alpha")
    assert len(kids) == 3
    assert all(json.loads(k["derived_from"]) == [parent] for k in kids)


def test_record_sweep_trials_caps_and_reports(
    repo: SqliteStrategyRepository, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Beyond the cap: stop writing rows and REPORT the truncation, never silently."""
    monkeypatch.setattr("algua.registry.store.runs.MAX_PERSISTED_TRIALS", 2)
    parent = repo.record_run("sweep", "alpha")
    trials = [{"config": {"i": i}, "metrics": {}} for i in range(5)]
    n, truncated = repo.record_sweep_trials(parent, "alpha", trials)
    assert (n, truncated) == (2, 2)
    assert len(repo.list_runs(kind="sweep_trial", strategy_name="alpha")) == 2


def test_list_runs_sorts_by_metric_descending_nulls_last(
    repo: SqliteStrategyRepository,
) -> None:
    repo.record_run("gate", "a", metrics={"sharpe_oos": 0.1})
    repo.record_run("gate", "b", metrics={"sharpe_oos": 2.0})
    repo.record_run("gate", "c")  # NULL sharpe_oos
    names = [r["strategy_name"] for r in repo.list_runs(kind="gate", sort="sharpe_oos")]
    assert names == ["b", "a", "c"]


def test_list_runs_rejects_a_non_vocabulary_sort_key(repo: SqliteStrategyRepository) -> None:
    """The sort key is interpolated into SQL, so it MUST be allow-listed."""
    with pytest.raises(ValueError, match="not a sortable metric"):
        repo.list_runs(sort="1; DROP TABLE runs")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/registry/test_runs_store.py -q`
Expected: FAIL with `AttributeError: 'SqliteStrategyRepository' object has no attribute 'record_run'`.

- [ ] **Step 3: Write the mixin**

Create `algua/registry/store/runs.py`:

```python
"""``RunLedger`` — the economic-layer evaluation ledger (``runs`` + ``run_metrics``).

The only module that embeds ``runs`` SQL. Mirrors ``search_breadth.py``: plain mixin, caller-owned
connection, ``with self._conn:`` per write.
"""
from __future__ import annotations

import json
import math
import sqlite3
from typing import Any

from algua.registry.db import MAX_PERSISTED_TRIALS, METRIC_SCHEMA_VERSION
from algua.registry.store._util import _now

#: The fixed metric vocabulary v1 (spec §4.2). A key outside this set is a caller bug, not a
#: reason to widen the table: it either belongs in `extra_metrics` (the overflow tail) or it earns
#: a column and a METRIC_SCHEMA_VERSION bump.
METRIC_COLUMNS: tuple[str, ...] = (
    "sharpe_is", "sharpe_oos", "sharpe_realized",
    "sortino_is", "sortino_oos",
    "total_return_is", "total_return_oos",
    "max_drawdown_is", "max_drawdown_oos",
    "ann_vol_is", "ann_vol_oos",
    "cagr_is", "calmar_is",
    "n_obs_is", "n_obs_oos",
    "mean_window_sharpe", "std_window_sharpe", "min_window_sharpe", "pct_positive_windows",
)

#: Provenance columns a caller may set. Kept explicit (not `**kwargs` into SQL) so a typo is a
#: ValueError rather than a silently-dropped field.
PROVENANCE_COLUMNS: tuple[str, ...] = (
    "code_hash", "config_hash", "dependency_hash", "data_source", "snapshot_id",
    "universe_name", "fundamentals_snapshot", "news_snapshot", "delisting_snapshot",
    "seed", "timeframe", "period_start", "period_end",
)

_KINDS = frozenset({"backtest", "walk_forward", "sweep", "sweep_trial", "gate"})


def _finite_or_none(value: Any) -> float | None:
    """NaN/inf are not measurements. Store NULL rather than a value that poisons every
    aggregate downstream."""
    if value is None:
        return None
    v = float(value)
    return v if math.isfinite(v) else None


class RunLedgerMixin:
    _conn: sqlite3.Connection

    def record_run(  # noqa: PLR0913 — a ledger row genuinely has this many independent parts
        self,
        kind: str,
        strategy_name: str,
        *,
        strategy_id: int | None = None,
        derived_from: list[int] | None = None,
        components: list[dict[str, Any]] | None = None,
        provenance: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
        metrics: dict[str, float | int | None] | None = None,
        extra_metrics: dict[str, float | None] | None = None,
        passed: bool | None = None,
        trials_truncated_at: int | None = None,
    ) -> int:
        """Insert one run row (and its overflow metrics) and return its id.

        `metrics` keys MUST come from `METRIC_COLUMNS`; anything else raises. `extra_metrics` is
        the free-form overflow tail and accepts any key.
        """
        if kind not in _KINDS:
            raise ValueError(f"unknown run kind {kind!r}; expected one of {sorted(_KINDS)}")
        prov = dict(provenance or {})
        for key in prov:
            if key not in PROVENANCE_COLUMNS:
                raise ValueError(f"{key!r} is not a provenance column")
        mets = dict(metrics or {})
        for key in mets:
            if key not in METRIC_COLUMNS:
                raise ValueError(
                    f"{key!r} is not in the fixed metric vocabulary; "
                    f"pass it via extra_metrics or give it a column")

        columns = ["kind", "strategy_name", "strategy_id", "created_at",
                   "metric_schema_version", "derived_from", "components", "config_json",
                   "passed", "trials_truncated_at"]
        values: list[Any] = [
            kind, strategy_name, strategy_id, _now(), METRIC_SCHEMA_VERSION,
            json.dumps(list(derived_from or [])),
            json.dumps(list(components or [])),
            json.dumps(config or {}, sort_keys=True, default=str),
            None if passed is None else int(passed),
            trials_truncated_at,
        ]
        for key in PROVENANCE_COLUMNS:
            if key in prov:
                columns.append(key)
                values.append(prov[key])
        for key in METRIC_COLUMNS:
            if key in mets:
                columns.append(key)
                # n_obs_* are integer counts; everything else is a float that must be finite.
                raw = mets[key]
                values.append(
                    (None if raw is None else int(raw)) if key.startswith("n_obs")
                    else _finite_or_none(raw)
                )
        placeholders = ",".join("?" for _ in columns)
        with self._conn:
            cur = self._conn.execute(
                f"INSERT INTO runs({','.join(columns)}) VALUES ({placeholders})", values)
            rowid = cur.lastrowid
            assert rowid is not None  # a successful INSERT always sets lastrowid
            if extra_metrics:
                self._conn.executemany(
                    "INSERT OR REPLACE INTO run_metrics(run_id, key, value) VALUES (?,?,?)",
                    [(rowid, k, _finite_or_none(v)) for k, v in extra_metrics.items()],
                )
        return rowid

    def record_sweep_trials(
        self, parent_run_id: int, strategy_name: str, trials: list[dict[str, Any]],
    ) -> tuple[int, int | None]:
        """Write up to ``MAX_PERSISTED_TRIALS`` child runs for a sweep, in ONE transaction.

        Returns ``(n_written, truncated_at)``; ``truncated_at`` is None when the whole trial set
        was persisted. The caller stamps ``truncated_at`` onto the parent run — a silently
        truncated set would make the funnel-wide distribution lie about the breadth it depicts.

        One writer, one transaction, at the END of the sweep: ``sweep()`` fans combos out across
        processes, and 70 concurrent writers against the governance DB is the wrong shape.
        """
        cap = MAX_PERSISTED_TRIALS
        kept = trials[:cap]
        truncated_at = cap if len(trials) > cap else None
        now = _now()
        lineage = json.dumps([parent_run_id])
        rows = []
        for trial in kept:
            mets = dict(trial.get("metrics") or {})
            for key in mets:
                if key not in METRIC_COLUMNS:
                    raise ValueError(
                        f"{key!r} is not in the fixed metric vocabulary (sweep trial)")
            rows.append((
                strategy_name, now, lineage,
                json.dumps(trial.get("config") or {}, sort_keys=True, default=str),
                trial.get("config_hash"),
                _finite_or_none(mets.get("mean_window_sharpe")),
                _finite_or_none(mets.get("std_window_sharpe")),
                _finite_or_none(mets.get("min_window_sharpe")),
                _finite_or_none(mets.get("pct_positive_windows")),
            ))
        with self._conn:
            self._conn.executemany(
                "INSERT INTO runs(kind, strategy_name, created_at, metric_schema_version,"
                " derived_from, components, config_json, config_hash,"
                " mean_window_sharpe, std_window_sharpe, min_window_sharpe, pct_positive_windows)"
                f" VALUES ('sweep_trial',?,?,{METRIC_SCHEMA_VERSION},?,'[]',?,?,?,?,?,?)",
                rows,
            )
        return len(kept), truncated_at

    def get_run(self, run_id: int) -> sqlite3.Row | None:
        return self._conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()

    def list_runs(
        self, *, kind: str | None = None, strategy_name: str | None = None,
        sort: str | None = None, limit: int = 100,
    ) -> list[sqlite3.Row]:
        """Scalar run rows, newest first (or best-first when ``sort`` names a metric).

        ``sort`` is interpolated into the SQL — it MUST be allow-listed against the fixed
        vocabulary, never taken from caller input unchecked.
        """
        if sort is not None and sort not in METRIC_COLUMNS:
            raise ValueError(f"{sort!r} is not a sortable metric")
        clauses: list[str] = []
        params: list[Any] = []
        if kind is not None:
            clauses.append("kind=?")
            params.append(kind)
        if strategy_name is not None:
            clauses.append("strategy_name=?")
            params.append(strategy_name)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        # NULLS LAST: a run with no measurement must never outrank one with a real number.
        order = (f" ORDER BY {sort} IS NULL, {sort} DESC, id DESC"
                 if sort else " ORDER BY id DESC")
        params.append(int(limit))
        return list(self._conn.execute(
            f"SELECT * FROM runs{where}{order} LIMIT ?", params).fetchall())
```

- [ ] **Step 4: Compose the mixin into the repository**

In `algua/registry/store/__init__.py`, add the import alongside the others:

```python
from algua.registry.store.runs import RunLedgerMixin
```

and add `RunLedgerMixin,` to the `SqliteStrategyRepository(...)` base list, after `BacktestReturnsLedgerMixin,`. Also add `runs.py` to the module docstring's list of carved modules.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/registry/test_runs_store.py tests/test_registry_store.py -q`
Expected: PASS. `test_no_mixin_shadows_another` must still pass — if it fails, a method name collides and must be renamed.

- [ ] **Step 6: Run the full quality gate**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports` (Bash `timeout: 600000`)
Expected: all four pass.

- [ ] **Step 7: Commit**

```bash
git add algua/registry/store/runs.py algua/registry/store/__init__.py tests/registry/test_runs_store.py
git commit -m "feat(registry): RunLedgerMixin — record_run, sweep trials, list_runs"
```

---

## Task 3: The pure metric mapper

**Files:**
- Create: `algua/registry/runs.py`
- Test: `tests/registry/test_runs_metrics.py`

**Interfaces:**
- Consumes: `RunLedgerMixin` (Task 2); `algua.backtest.result.BacktestResult`; `algua.backtest.walkforward.WalkForwardResult`; `algua.backtest.sweep.SweepResult`.
- Produces:
  - `backtest_metrics(result: BacktestResult) -> dict[str, float | int | None]`
  - `walk_forward_metrics(result: WalkForwardResult) -> dict[str, float | int | None]`
  - `provenance_of(result: Any) -> dict[str, Any]`

**Why a sentinel rule, not "0.0 means NULL":** `metrics_from_returns` returns literal `0.0` for an undefined Sharpe (`ann_volatility == 0`), an undefined Sortino (downside deviation `0`), and an undefined Calmar (`max_drawdown == 0`). A blanket "0.0 → NULL" would also nullify a genuinely-zero measurement. The rule below keys off the *detectable* degeneracy conditions instead. One accepted imprecision, documented in code: a constant NEGATIVE return series has `ann_volatility == 0` but a computable Sortino; we record NULL there. NULL is honest; a sentinel zero is not.

- [ ] **Step 1: Write the failing test**

Create `tests/registry/test_runs_metrics.py`:

```python
"""The fixed-vocabulary metric mapper: sentinel detection and sample-class naming."""
from __future__ import annotations

import pytest

from algua.registry.runs import backtest_metrics, walk_forward_metrics


class _FakeBacktest:
    def __init__(self, metrics: dict[str, float]) -> None:
        self.metrics = metrics
        self.returns = None


def _full_metrics(**overrides: float) -> dict[str, float]:
    base = {
        "sharpe": 1.2, "sortino": 1.4, "total_return": 0.30, "max_drawdown": -0.12,
        "ann_volatility": 0.18, "cagr": 0.15, "calmar": 1.25,
    }
    return base | overrides


def test_maps_to_sample_suffixed_keys() -> None:
    out = backtest_metrics(_FakeBacktest(_full_metrics()))
    assert out["sharpe_is"] == pytest.approx(1.2)
    assert out["total_return_is"] == pytest.approx(0.30)
    assert out["ann_vol_is"] == pytest.approx(0.18)
    assert "sharpe" not in out


def test_zero_ann_vol_nulls_sharpe_and_sortino() -> None:
    """metrics_from_returns returns a 0.0 SENTINEL when ann_volatility == 0."""
    out = backtest_metrics(_FakeBacktest(_full_metrics(ann_volatility=0.0, sharpe=0.0,
                                                       sortino=0.0)))
    assert out["sharpe_is"] is None
    assert out["sortino_is"] is None
    assert out["ann_vol_is"] == pytest.approx(0.0)


def test_zero_max_drawdown_nulls_calmar() -> None:
    out = backtest_metrics(_FakeBacktest(_full_metrics(max_drawdown=0.0, calmar=0.0)))
    assert out["calmar_is"] is None
    assert out["max_drawdown_is"] == pytest.approx(0.0)


def test_a_genuine_zero_sharpe_survives() -> None:
    """A real 0.0 Sharpe with non-zero volatility is a MEASUREMENT, not a sentinel."""
    out = backtest_metrics(_FakeBacktest(_full_metrics(sharpe=0.0)))
    assert out["sharpe_is"] == pytest.approx(0.0)


class _FakeWalkForward:
    def __init__(self) -> None:
        self.stability = {
            "mean_sharpe": 0.8, "std_sharpe": 0.3,
            "min_sharpe": -0.2, "pct_positive_windows": 0.75,
        }
        self.holdout_metrics = {
            "start": "2024-01-01", "end": "2024-06-30", "n_bars": 120,
            "sharpe": 0.4, "sortino": 0.5, "total_return": 0.05,
            "max_drawdown": -0.08, "ann_volatility": 0.14,
        }


def test_walk_forward_maps_holdout_to_oos_and_windows_to_their_own_names() -> None:
    out = walk_forward_metrics(_FakeWalkForward())
    assert out["sharpe_oos"] == pytest.approx(0.4)
    assert out["total_return_oos"] == pytest.approx(0.05)
    assert out["n_obs_oos"] == 120
    assert out["mean_window_sharpe"] == pytest.approx(0.8)
    assert out["min_window_sharpe"] == pytest.approx(-0.2)
    assert out["pct_positive_windows"] == pytest.approx(0.75)
    # A walk-forward measures no in-sample full-period figure — it must not invent one.
    assert "sharpe_is" not in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/registry/test_runs_metrics.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'algua.registry.runs'`.

- [ ] **Step 3: Write the mapper**

Create `algua/registry/runs.py`:

```python
"""Recorder seam for the economic-layer run ledger.

Mirrors ``algua/registry/search_breadth.py``: pure mapping helpers plus thin recorders whose
TRANSACTION IS CALLER-OWNED — the CLI wraps them in ``with registry_conn() as conn:`` and passes
``SqliteStrategyRepository(conn)``.

The mapping helpers are PURE (no I/O, no DB) so they are testable without a connection.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from algua.backtest.result import BacktestResult
    from algua.backtest.sweep import SweepResult
    from algua.backtest.walkforward import WalkForwardResult
    from algua.registry.store import SqliteStrategyRepository

# Provenance attributes shared by BacktestResult / WalkForwardResult / SweepResult. Read with
# getattr because the three dataclasses carry overlapping but not identical field sets.
_PROVENANCE_ATTRS = (
    "code_hash", "config_hash", "dependency_hash", "data_source", "snapshot_id",
    "universe_name", "fundamentals_snapshot", "news_snapshot", "delisting_snapshot",
    "seed", "timeframe",
)


def provenance_of(result: Any) -> dict[str, Any]:
    """Provenance columns present on a backtest-family result. Absent attributes are omitted, not
    written as NULL, so a later widening of a result type needs no change here."""
    out: dict[str, Any] = {
        k: getattr(result, k) for k in _PROVENANCE_ATTRS if getattr(result, k, None) is not None
    }
    period = getattr(result, "period", None)
    if isinstance(period, dict):
        if period.get("start") is not None:
            out["period_start"] = period["start"]
        if period.get("end") is not None:
            out["period_end"] = period["end"]
    return out


def _sample(metrics: dict[str, float], suffix: str) -> dict[str, float | int | None]:
    """Map one `metrics_from_returns` dict onto sample-suffixed vocabulary keys.

    SENTINEL RULE: `metrics_from_returns` returns a literal 0.0 for an UNDEFINED Sharpe
    (ann_volatility == 0), Sortino (zero downside deviation) and Calmar (max_drawdown == 0).
    Recording those as 0.0 would rank a degenerate run above a genuinely negative one, so they
    become NULL. We key off the DEGENERACY CONDITION, not the value — a genuine 0.0 Sharpe on a
    volatile series is a measurement and survives.

    Accepted imprecision: a constant NEGATIVE return series has ann_volatility == 0 but a
    computable Sortino; it is nulled here. NULL is honest; a sentinel zero is not.
    """
    ann_vol = metrics.get("ann_volatility")
    mdd = metrics.get("max_drawdown")
    degenerate_ratio = ann_vol is None or ann_vol == 0.0
    out: dict[str, float | int | None] = {
        f"sharpe{suffix}": None if degenerate_ratio else metrics.get("sharpe"),
        f"sortino{suffix}": None if degenerate_ratio else metrics.get("sortino"),
        f"total_return{suffix}": metrics.get("total_return"),
        f"max_drawdown{suffix}": mdd,
        f"ann_vol{suffix}": ann_vol,
    }
    if suffix == "_is":
        out["cagr_is"] = metrics.get("cagr")
        out["calmar_is"] = None if (mdd is None or mdd == 0.0) else metrics.get("calmar")
    return out


def backtest_metrics(result: BacktestResult) -> dict[str, float | int | None]:
    """Fixed-vocabulary metrics for a `backtest` run. In-sample by construction."""
    out = _sample(dict(result.metrics), "_is")
    returns = getattr(result, "returns", None)
    if returns is not None:
        out["n_obs_is"] = int(len(returns))
    return out


def walk_forward_metrics(result: WalkForwardResult) -> dict[str, float | int | None]:
    """Fixed-vocabulary metrics for a `walk_forward` run.

    The holdout segment is the OUT-OF-SAMPLE measurement; the per-window stability figures keep
    their own names (they are neither IS nor OOS — they are a dispersion across folds). No
    in-sample full-period figure is emitted: a walk-forward does not measure one, and inventing
    it would put a fabricated number on the scatter's x-axis.
    """
    holdout = dict(result.holdout_metrics)
    out = _sample(holdout, "_oos")
    if holdout.get("n_bars") is not None:
        out["n_obs_oos"] = int(holdout["n_bars"])
    stability = dict(result.stability)
    out["mean_window_sharpe"] = stability.get("mean_sharpe")
    out["std_window_sharpe"] = stability.get("std_sharpe")
    out["min_window_sharpe"] = stability.get("min_sharpe")
    out["pct_positive_windows"] = stability.get("pct_positive_windows")
    return out


def record_backtest_run(
    repo: SqliteStrategyRepository,
    name: str,
    result: BacktestResult,
    *,
    params: dict[str, Any] | None = None,
) -> int:
    """Record one `backtest` run. Recorded UNCONDITIONALLY — even for a not-yet-registered
    strategy, for the same reason `record_search_breadth` is: exploration precedes registration
    and that evidence must not be discarded.

    `params` is the strategy's config params, passed EXPLICITLY: `BacktestResult` carries
    `config_hash` but not the config itself, so there is nothing to read off the result.
    """
    return repo.record_run(
        "backtest", name,
        provenance=provenance_of(result),
        config=dict(params or {}),
        metrics=backtest_metrics(result),
        components=list(_components_of(result)),
    )


def record_walk_forward_run(
    repo: SqliteStrategyRepository, name: str, result: WalkForwardResult,
) -> int:
    """Record one `walk_forward` run."""
    return repo.record_run(
        "walk_forward", name,
        provenance=provenance_of(result),
        metrics=walk_forward_metrics(result),
        components=list(_components_of(result)),
    )


def record_sweep_run(
    repo: SqliteStrategyRepository, name: str, result: SweepResult,
) -> dict[str, Any]:
    """Record one `sweep` parent run plus a child `sweep_trial` per ranked combo.

    `SweepResult.ranked` already carries every combo's `{params, config_hash, stability, score}`,
    so no re-computation is needed. The children are written in ONE batched transaction (see
    `record_sweep_trials`), and a truncated trial set is stamped back onto the parent.
    """
    parent = repo.record_run(
        "sweep", name,
        provenance=provenance_of(result),
        config={"grid": result.grid, "rank_by": result.rank_by,
                "windows": result.windows, "holdout_frac": result.holdout_frac},
        metrics={"mean_window_sharpe": result.trial_sharpe_mean},
    )
    trials = [
        {
            "config": record["params"],
            "config_hash": record.get("config_hash"),
            "metrics": {
                "mean_window_sharpe": record["stability"].get("mean_sharpe"),
                "std_window_sharpe": record["stability"].get("std_sharpe"),
                "min_window_sharpe": record["stability"].get("min_sharpe"),
                "pct_positive_windows": record["stability"].get("pct_positive_windows"),
            },
        }
        for record in result.ranked
    ]
    n_written, truncated_at = repo.record_sweep_trials(parent, name, trials)
    if truncated_at is not None:
        repo.stamp_trials_truncated(parent, truncated_at)
    return {"run_id": parent, "trials_written": n_written, "trials_truncated_at": truncated_at}


def _components_of(result: Any) -> list[dict[str, Any]]:
    """Model-layer lineage as a LIST, even though `model_ref` is singular today (spec §2.1)."""
    ref = getattr(result, "model_ref", None)
    return [ref] if isinstance(ref, dict) else []
```

- [ ] **Step 4: Add the parent-stamp method to the mixin**

`record_sweep_run` calls `stamp_trials_truncated`, which Task 2 did not define. Add it to `algua/registry/store/runs.py`, inside `RunLedgerMixin`:

```python
    def stamp_trials_truncated(self, run_id: int, truncated_at: int) -> None:
        """Mark a `sweep` parent whose trial set was capped. Never inferred by a reader — a
        truncated distribution must announce itself."""
        with self._conn:
            self._conn.execute(
                "UPDATE runs SET trials_truncated_at=? WHERE id=?", (truncated_at, run_id))
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/registry/test_runs_metrics.py -q`
Expected: PASS, 6 tests.

- [ ] **Step 6: Run the full quality gate**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports` (Bash `timeout: 600000`)
Expected: all four pass. `lint-imports` matters here: `algua/registry/runs.py` imports backtest result types only under `TYPE_CHECKING`, so no runtime registry→backtest edge is created.

- [ ] **Step 7: Commit**

```bash
git add algua/registry/runs.py algua/registry/store/runs.py tests/registry/test_runs_metrics.py
git commit -m "feat(registry): fixed-vocabulary metric mapper + run recorder seam"
```

---

## Task 4: Write path — `backtest run`

**Files:**
- Modify: `algua/cli/backtest_cmd.py` (in `run_backtest_task`)
- Test: `tests/test_runs_write_path.py`

**Interfaces:**
- Consumes: `algua.registry.runs.record_backtest_run` (Task 3).
- Produces: a `backtest` run row per `backtest run` invocation.

- [ ] **Step 1: Write the failing test**

Create `tests/test_runs_write_path.py`:

```python
"""Every evaluation lands a run row — including for an UNREGISTERED strategy."""
from __future__ import annotations

import pytest

from algua.cli._common import registry_conn
from algua.cli.backtest_cmd import run_backtest_task
from algua.registry.store import SqliteStrategyRepository

STRATEGY = "cross_sectional_momentum"


@pytest.fixture(autouse=True)
def _isolated_db(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:  # noqa: ANN001
    """The established DB-isolation idiom (see tests/test_cli_backtest.py)."""
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))


def test_backtest_records_a_run() -> None:
    run_backtest_task(STRATEGY, demo=True)
    with registry_conn() as conn:
        rows = SqliteStrategyRepository(conn).list_runs(kind="backtest")
    assert len(rows) == 1
    assert rows[0]["strategy_name"] == STRATEGY
    assert rows[0]["sharpe_is"] is not None
    assert rows[0]["code_hash"] is not None


def test_backtest_records_a_run_for_an_unregistered_strategy() -> None:
    """Exploration precedes registration — that evidence must not be discarded."""
    run_backtest_task(STRATEGY, demo=True, register=False)
    with registry_conn() as conn:
        repo = SqliteStrategyRepository(conn)
        assert STRATEGY not in {s.name for s in repo.list_strategies()}
        assert len(repo.list_runs(kind="backtest")) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_runs_write_path.py -q`
Expected: FAIL — `assert len(rows) == 1` gets 0.

- [ ] **Step 3: Add the recorder call**

In `algua/cli/backtest_cmd.py`, add to the imports:

```python
from algua.registry.runs import record_backtest_run
```

In `run_backtest_task`, immediately after the `result = run_backtest(...)` assignment and BEFORE
the `if register:` block, insert:

```python
    # Record the evaluation as a first-class run row. UNCONDITIONAL — including for a
    # not-yet-registered strategy, the same rationale record_search_breadth documents: keying by
    # name means pre-registration evidence still counts. Own transaction, like the sibling writes.
    with registry_conn() as conn:
        record_backtest_run(
            SqliteStrategyRepository(conn), name, result, params=strategy.config.params)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_runs_write_path.py -q`
Expected: PASS, 2 tests.

- [ ] **Step 5: Run the full quality gate**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports` (Bash `timeout: 600000`)
Expected: all four pass.

- [ ] **Step 6: Commit**

```bash
git add algua/cli/backtest_cmd.py tests/test_runs_write_path.py
git commit -m "feat(cli): record a run row for every backtest"
```

---

## Task 5: Write path — `backtest walk-forward`

**Files:**
- Modify: `algua/cli/backtest_cmd.py` (in the `walk-forward` command body)
- Test: `tests/test_runs_write_path.py` (extend)

**Interfaces:**
- Consumes: `algua.registry.runs.record_walk_forward_run` (Task 3).
- Produces: a `walk_forward` run row carrying `sharpe_oos` and the window-dispersion metrics — the y-axis and x-axis of slice 3's scatter.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_runs_write_path.py`:

```python
def test_walk_forward_records_oos_and_window_metrics() -> None:
    from typer.testing import CliRunner

    from algua.cli.main import app

    res = CliRunner().invoke(app, ["backtest", "walk-forward", STRATEGY, "--demo"])
    assert res.exit_code == 0, res.output
    with registry_conn() as conn:
        rows = SqliteStrategyRepository(conn).list_runs(kind="walk_forward")
    assert len(rows) == 1
    row = rows[0]
    assert row["sharpe_oos"] is not None
    assert row["n_obs_oos"] is not None
    assert row["mean_window_sharpe"] is not None
    # A walk-forward measures no full-period in-sample figure; it must not invent one.
    assert row["sharpe_is"] is None
```

The `CliRunner().invoke(app, [...])` form is the idiom the existing CLI tests use (see
`tests/test_cli_backtest.py:17`); match their exact import path for `app`.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_runs_write_path.py -q`
Expected: FAIL — `assert len(rows) == 1` gets 0.

- [ ] **Step 3: Add the recorder call**

In `algua/cli/backtest_cmd.py`, extend the import added in Task 4:

```python
from algua.registry.runs import record_backtest_run, record_walk_forward_run
```

In the walk-forward command body, immediately after the `result = walk_forward(...)` call and
before the payload is built, insert:

```python
    with registry_conn() as conn:
        record_walk_forward_run(SqliteStrategyRepository(conn), name, result)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_runs_write_path.py -q`
Expected: PASS, 3 tests.

- [ ] **Step 5: Run the full quality gate**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports` (Bash `timeout: 600000`)
Expected: all four pass.

- [ ] **Step 6: Commit**

```bash
git add algua/cli/backtest_cmd.py tests/test_runs_write_path.py
git commit -m "feat(cli): record a run row for every walk-forward"
```

---

## Task 6: Write path — `backtest sweep` parent + batched trials

**Files:**
- Modify: `algua/cli/backtest_cmd.py` (in the `sweep` command body, at the existing `record_search_breadth` call site, ~line 419)
- Test: `tests/test_runs_write_path.py` (extend)

**Interfaces:**
- Consumes: `algua.registry.runs.record_sweep_run` (Task 3).
- Produces: one `sweep` parent run + one `sweep_trial` child per combo; `payload["recorded_runs"]` in the sweep JSON.

**This is the task the whole slice exists for.** `search_trials` currently keeps `n_combos` and
mean/variance and discards the per-combo results, so the breadth number the gate deflates against
is an unverifiable assertion. After this, it is a count you can `SELECT`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_runs_write_path.py`:

```python
def _run_sweep() -> None:
    from typer.testing import CliRunner

    from algua.cli.main import app

    res = CliRunner().invoke(
        app, ["backtest", "sweep", STRATEGY, "--demo", "--param", "lookback=20,40,60"])
    assert res.exit_code == 0, res.output


def test_sweep_records_a_parent_and_one_child_per_combo() -> None:
    import json

    _run_sweep()
    with registry_conn() as conn:
        repo = SqliteStrategyRepository(conn)
        parents = repo.list_runs(kind="sweep")
        children = repo.list_runs(kind="sweep_trial")
    assert len(parents) == 1
    assert len(children) == 3
    assert parents[0]["trials_truncated_at"] is None
    assert all(json.loads(c["derived_from"]) == [parents[0]["id"]] for c in children)
    assert all(c["mean_window_sharpe"] is not None for c in children)


def test_sweep_trial_count_matches_the_recorded_breadth() -> None:
    """The point of the slice: n_combos stops being an assertion and becomes a countable set."""
    _run_sweep()
    with registry_conn() as conn:
        repo = SqliteStrategyRepository(conn)
        n_children = len(repo.list_runs(kind="sweep_trial"))
        declared = repo.total_search_combos(STRATEGY)
    assert n_children == declared
```

Match the exact `--param` flag spelling against `tests/test_cli_sweep.py` before running.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_runs_write_path.py -q`
Expected: FAIL — `assert len(parents) == 1` gets 0.

- [ ] **Step 3: Add the recorder call**

In `algua/cli/backtest_cmd.py`, extend the import:

```python
from algua.registry.runs import record_backtest_run, record_sweep_run, record_walk_forward_run
```

Replace the existing breadth-recording block:

```python
    with registry_conn() as conn:
        recorded = record_search_breadth(SqliteStrategyRepository(conn), name, result)
```

with:

```python
    with registry_conn() as conn:
        repo = SqliteStrategyRepository(conn)
        recorded = record_search_breadth(repo, name, result)
        # The per-combo rows behind `recorded["n_combos"]`. Written in ONE batched transaction at
        # the END of the sweep — sweep() fans combos out across processes, so concurrent
        # per-combo writers against the governance DB would be the wrong shape.
        recorded_runs = record_sweep_run(repo, name, result)
```

and, next to the existing `payload["recorded_breadth"] = recorded` line, add:

```python
    # Surface the run rows too, so a truncated trial set is visible in the command's own output
    # rather than only discoverable by querying.
    payload["recorded_runs"] = recorded_runs
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_runs_write_path.py -q`
Expected: PASS, 5 tests.

- [ ] **Step 5: Run the full quality gate**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports` (Bash `timeout: 600000`)
Expected: all four pass. Watch for a JSON-contract test asserting the exact key set of the sweep
payload — `recorded_runs` is a new key and such a test must be updated.

- [ ] **Step 6: Commit**

```bash
git add algua/cli/backtest_cmd.py tests/test_runs_write_path.py
git commit -m "feat(cli): persist sweep trials as child runs — breadth becomes countable"
```

---

## Task 7: Write path — the gate run, inside the atomic transaction

**Files:**
- Modify: `algua/registry/promotion.py` (**CODEOWNERS-protected**)
- Test: `tests/test_runs_write_path.py` (extend)

**Interfaces:**
- Consumes: `RunLedgerMixin.record_run` (Task 2); `algua.registry.runs.walk_forward_metrics` (Task 3).
- Produces: a `gate` run row per `research promote` attempt — **pass AND fail** — whose `derived_from` names the walk-forward run the decision was computed on.

**Files (corrected — this task also touches the gate store):**
- Modify: `algua/registry/store/runs.py` (add `_insert_run_locked`)
- Modify: `algua/registry/store/gate.py` (**CODEOWNERS-protected**)
- Modify: `algua/registry/promotion.py` (**CODEOWNERS-protected**)
- Test: `tests/test_runs_write_path.py` (extend)

**Three constraints specific to this task:**

1. **A failing gate must record a run.** The failures are the point: the IS-vs-OOS scatter and the
   trial distribution are both mostly made of rejected strategies. `record_gate_evaluation`
   already writes on pass and fail; the run row must follow the same rule.
2. **Same transaction as the gate row.** The run row is derived evidence for that exact decision;
   a run row surviving a rolled-back gate would be a phantom evaluation.
3. **THE CONSTRAINT THAT SHAPES THIS TASK.** `record_gate_with_fdr_and_maybe_promote`
   (`algua/registry/store/gate.py:286`) owns its own `BEGIN IMMEDIATE` and **fails loudly** if
   entered with a transaction already open:

   ```python
   if self._conn.in_transaction:
       raise RuntimeError(
           "record_gate_with_fdr_and_maybe_promote must be called at top level,"
           " not inside an open transaction")
   ```

   So `promotion.py` **cannot** wrap it, and the gate run row cannot be written from there. It
   must be inserted *inside* that store method. The run payload is therefore passed in as a new
   `run_row` kwarg and inserted with a lock-free helper that does NOT open its own transaction.

   The **walk-forward** run recorded earlier in `promotion.py` is fine as a normal top-level
   `repo.record_run(...)`: its `with self._conn:` commits and closes, leaving `in_transaction`
   False before the gate call.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_runs_write_path.py`:

```python
def test_failing_gate_still_records_a_run() -> None:
    """The rejections ARE the dataset — a gate that records nothing on failure is useless."""
    _promote(expect_pass=False)
    with registry_conn() as conn:
        rows = SqliteStrategyRepository(conn).list_runs(kind="gate")
    assert len(rows) == 1
    assert rows[0]["passed"] == 0
    assert rows[0]["sharpe_oos"] is not None


def test_gate_run_links_to_its_walk_forward_run() -> None:
    """derived_from is what makes the IS-vs-OOS scatter joinable at all."""
    import json

    _promote(expect_pass=False)
    with registry_conn() as conn:
        repo = SqliteStrategyRepository(conn)
        gate = repo.list_runs(kind="gate")[0]
        wf = repo.list_runs(kind="walk_forward")[0]
    assert json.loads(gate["derived_from"]) == [wf["id"]]


def test_a_rolled_back_gate_leaves_no_phantom_run() -> None:
    """The run row shares the gate's transaction; if the gate rolls back, so must the run."""
    import sqlite3
    from unittest.mock import patch

    from algua.registry.store.gate import GateLedgerMixin

    # Force the commit path to fail AFTER the inserts, inside the same transaction.
    with patch.object(GateLedgerMixin, "_commit_gate_hook",
                      side_effect=sqlite3.OperationalError("boom"), create=True):
        with pytest.raises(sqlite3.OperationalError):
            _promote(expect_pass=False)
    with registry_conn() as conn:
        assert SqliteStrategyRepository(conn).list_runs(kind="gate") == []
```

**Before writing these**, read `tests/_gate_row_helpers.py` and the promote tests under
`tests/research/` and reuse their promote-driving helper to implement `_promote(expect_pass=...)`
rather than building a new one. Every current strategy fails the gate, so `expect_pass=False` is
the easy path.

The third test asserts constraint 2 and is the one that would catch a regression to a separate
`registry_conn()`. If `GateLedgerMixin` has no seam to fail at, drop the `patch.object` and instead
assert the property directly: monkeypatch `_insert_run_locked` to raise, call `_promote`, and
assert that **`gate_evaluations` also has no row** — proving the two writes share a transaction in
the other direction. Implement whichever of the two is honest against the real code; do not ship a
test that passes without exercising the shared transaction.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_runs_write_path.py -q`
Expected: FAIL — `assert len(rows) == 1` gets 0.

- [ ] **Step 3: Record the walk-forward run inside the promote flow**

In `algua/registry/promotion.py`, find where the walk-forward result `wf` becomes available and
`gate_row` is assembled (~line 641). Before the `gate_row = {...}` literal, insert:

```python
    # The walk-forward this decision was computed on, as its own run row. Recorded here rather
    # than by the CLI because `research promote` runs the walk-forward internally — this is the
    # only place its result exists.
    wf_run_id = repo.record_run(
        "walk_forward", rec.name,
        strategy_id=rec.id,
        provenance=provenance_of(wf),
        metrics=walk_forward_metrics(wf),
    )
```

with the import at the top of the module:

```python
from algua.registry.runs import provenance_of, walk_forward_metrics
```

- [ ] **Step 4: Add the lock-free insert helper to the mixin**

`record_run` opens `with self._conn:`. That cannot be used inside the gate's `BEGIN IMMEDIATE`
block. Refactor `algua/registry/store/runs.py` so the SQL lives in one place and the transaction
is the caller's choice:

```python
    def _insert_run_locked(self, kind: str, strategy_name: str, **kwargs: Any) -> int:
        """The `record_run` INSERT with NO transaction of its own — for callers that already
        hold one (the gate's BEGIN IMMEDIATE block). `record_run` is this plus `with self._conn:`.
        """
        # ... body: everything currently inside record_run EXCEPT the `with self._conn:` wrapper
```

Then reduce `record_run` to:

```python
    def record_run(self, kind: str, strategy_name: str, **kwargs: Any) -> int:
        with self._conn:
            return self._insert_run_locked(kind, strategy_name, **kwargs)
```

Keep `record_run`'s explicit keyword-only signature as the public one (it is what Tasks 3–6 call);
`_insert_run_locked` may take `**kwargs` since only the store calls it. Validation (unknown kind,
unknown provenance key, unknown metric key) stays in the locked helper so both paths enforce it.

- [ ] **Step 5: Accept and insert the run row inside the gate transaction**

In `algua/registry/store/gate.py`, add a keyword-only parameter to
`record_gate_with_fdr_and_maybe_promote`:

```python
        run_row: dict[str, Any] | None = None,
```

and document it in the method:

```python
        ``run_row`` is the economic-layer run payload for THIS decision (spec 2026-08-23 §4.1),
        inserted inside this method's BEGIN IMMEDIATE so it shares the gate row's fate. It cannot
        be written by the caller: this method refuses to run inside an open transaction, so a
        caller-side write would either land in a separate transaction (a run row that survives a
        rolled-back gate — a phantom evaluation) or trip the top-level guard above.
```

Inside the transaction, **immediately after the `gate_evaluations` insert**, add:

```python
            if run_row is not None:
                self._insert_run_locked("gate", rec.name, **run_row)
```

Read the method body first to find the exact gate-insert site — it must be after the insert and
before any early return within the block.

- [ ] **Step 6: Build and pass the run row from `promotion.py`**

At the `repo.record_gate_with_fdr_and_maybe_promote(...)` call (~line 703), add the kwarg:

```python
    fdr_outcome = repo.record_gate_with_fdr_and_maybe_promote(
        rec, gate_row=gate_row, funnel=funnel, actor=actor,
        reason=(_gate_reason(decision) + reason_suffix) if decision.passed else None,
        pending_novel_family=breadth.pending_novel_family,  # #524: minted only on pass, in-tx
        # The same decision as an economic-layer run row. PASS AND FAIL — the rejections are the
        # dataset the IS-vs-OOS scatter is mostly made of.
        run_row={
            "strategy_id": rec.id,
            "derived_from": [wf_run_id],
            "passed": decision.passed,
            "provenance": {
                "code_hash": identity.code_hash,
                "config_hash": identity.config_hash,
                "dependency_hash": identity.dependency_hash,
                "data_source": data_source,
                "snapshot_id": snapshot_id,
                "universe_name": universe_name,
                "fundamentals_snapshot": wf.fundamentals_snapshot,
                "news_snapshot": wf.news_snapshot,
                "period_start": period_start.isoformat(),
                "period_end": period_end.isoformat(),
            },
            "metrics": walk_forward_metrics(wf),
            # The ~40 DSR / IR / regime diagnostics: queryable, but deliberately outside the fixed
            # vocabulary. Finite scalars only — decision.to_dict() also carries strings, lists and
            # bools, and `bool` is an `int` subclass so it must be excluded explicitly.
            "extra_metrics": {
                k: float(v) for k, v in decision.to_dict().items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            },
        },
    )
```

Note `decision.passed` here is the PROVISIONAL flag, matching what `gate_row["passed"]` carries.
The two must agree — if the store rewrites `passed` on the gate row before committing, the run row
must be rewritten the same way. Check `record_gate_with_fdr_and_maybe_promote` for that and mirror
it if so; a run row whose `passed` disagrees with its gate row is a corrupt audit trail.

- [ ] **Step 7: Run the tests to verify they pass**

Run: `uv run pytest tests/test_runs_write_path.py tests/research tests/registry -q`
Expected: PASS. The existing promote/gate tests must all still pass — if a promote test now sees
extra rows, that is the feature, and its assertions on `runs` (if any) get updated, but **no
assertion about `gate_evaluations` may change**. If one does, the transaction wiring is wrong.

- [ ] **Step 8: Run the full quality gate**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports` (Bash `timeout: 600000`)
Expected: all four pass.

- [ ] **Step 9: Commit**

```bash
git add algua/registry/store/runs.py algua/registry/store/gate.py algua/registry/promotion.py tests/test_runs_write_path.py
git commit -m "feat(registry): record gate + walk-forward runs inside the promote transaction"
```

---

## Task 8: Verify the slice end to end

**Files:** none modified — this task is verification only.

- [ ] **Step 1: Run the full quality gate one more time from clean**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports` (Bash `timeout: 600000`)
Expected: all four pass.

- [ ] **Step 2: Exercise the real write path against a scratch DB**

```bash
ALGUA_DB_PATH=/tmp/runs-check.db uv run algua backtest run cross_sectional_momentum --demo
ALGUA_DB_PATH=/tmp/runs-check.db uv run algua backtest sweep cross_sectional_momentum --demo --param lookback=20,40,60
```

`ALGUA_DB_PATH` is correct (`Settings.db_path` under `env_prefix="ALGUA_"`). **Do NOT omit it** —
the default is the operator's real `data/algua.db`.

- [ ] **Step 3: Confirm the rows landed**

```bash
python3 -c "
import sqlite3
c = sqlite3.connect('file:/tmp/runs-check.db?mode=ro', uri=True)
c.row_factory = sqlite3.Row
for r in c.execute('SELECT kind, COUNT(*) n FROM runs GROUP BY kind'):
    print(r['kind'], r['n'])
print('trials == declared breadth:',
      c.execute(\"SELECT COUNT(*) FROM runs WHERE kind='sweep_trial'\").fetchone()[0]
      == c.execute('SELECT COALESCE(SUM(n_combos),0) FROM search_trials').fetchone()[0])
"
```

Expected: a `backtest` row, a `sweep` row, three `sweep_trial` rows, and `True`.

- [ ] **Step 4: Clean up and report**

```bash
rm -f /tmp/runs-check.db
git log --oneline origin/main..HEAD
```

Report the commit list. Slice 1 is complete when the write path is landed and accumulating; slices
2 (CLI reads) and 3 (views) get their own plans, deliberately written later so they are built
against real accumulated data rather than an empty store.
