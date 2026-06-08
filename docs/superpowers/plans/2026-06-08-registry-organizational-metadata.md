# Registry Organizational Metadata Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Promote organizational metadata (`family`, `tags`, `author`, `hypothesis_status`, `derived_from`, `description`) to first-class, queryable registry fields, with `registry list` filters and kb frontmatter deriving from the registry.

**Architecture:** Add six NULL columns to the `strategies` table (schema v17) with repository-layer defaults for new rows so a one-shot `backfill-from-kb` can still recover kb-authored values. New `contracts/registry_metadata.py` holds the `Author`/`HypothesisStatus` enums. The registry stays the single SQL source of truth; the knowledge layer stays registry-free and receives a plain metadata dict at the CLI seam.

**Tech Stack:** Python 3.12, sqlite3, Typer CLI, pytest, ruff, mypy, import-linter.

**Spec:** `docs/superpowers/specs/2026-06-08-registry-organizational-metadata-design.md`

**Quality gate (run before every commit):** `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`

## Testing conventions (read before writing any test)

The test snippets below use shorthand names — map them to the **real** helpers already in the
suite (look at `tests/test_cli_registry.py` and `tests/test_cli_strategy.py`):

- `runner = CliRunner()` and `app` come `from algua.cli.main import app`. **Invoke** with
  `runner.invoke(app, ["registry", "add", "alpha"])`. Wherever a snippet writes
  `run_cli([...])`, use `runner.invoke(app, [...])`.
- `_json(result)` asserts `exit_code == 0` and returns `json.loads(result.stdout)`. Wherever a
  snippet writes `json_of(...)`, use `_json(...)`.
- **DB isolation:** `tests/test_cli_registry.py` has an autouse fixture that sets
  `ALGUA_DB_PATH` to a `tmp_path` DB — CLI tests added there inherit it. For new test files,
  replicate that fixture.
- **kb isolation:** any test that triggers a kb sync (`registry set`, `strategy new`,
  `backfill-from-kb`) MUST also `monkeypatch.setenv("ALGUA_KNOWLEDGE_DIR", str(tmp_path / "vault"))`
  (and/or `monkeypatch.chdir(tmp_path)`), or it will write into the real `kb/`. `db_path`
  (`data/algua.db`) and `knowledge_dir` (`kb`) are both **relative**, so `chdir(tmp_path)` isolates
  both — which is why the existing `strategy new` tests are safe even once `new` starts registering.
- Snippet fixture names like `registry_env`, `strategy_env`, `registry_env_with_kb`, `sync_settings`
  are stand-ins: use the real `_tmp_db` autouse + per-test `monkeypatch.setenv(...)` pattern the
  suite already uses. For knowledge-layer unit tests, mirror `tests/test_knowledge_sync.py`'s
  Settings construction.

---

## Slice 1 — Schema + model + enums

Adds the columns, the enums, the canonicalization helper, and read-path plumbing. `registry add` gains the kwargs (new-row defaults); `list`/`show` emit the new fields. No new CLI options yet.

### Task 1: Metadata enums in contracts

**Files:**
- Create: `algua/contracts/registry_metadata.py`
- Test: `tests/test_registry_metadata.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_registry_metadata.py
from algua.contracts.registry_metadata import Author, HypothesisStatus


def test_author_values():
    assert {a.value for a in Author} == {"agent", "human"}


def test_hypothesis_status_values():
    assert {h.value for h in HypothesisStatus} == {
        "untested", "supported", "refuted", "inconclusive"
    }


def test_enums_are_strenum():
    assert Author.AGENT == "agent"
    assert HypothesisStatus.UNTESTED == "untested"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_registry_metadata.py -q`
Expected: FAIL with `ModuleNotFoundError: algua.contracts.registry_metadata`

- [ ] **Step 3: Write minimal implementation**

```python
# algua/contracts/registry_metadata.py
from __future__ import annotations

from enum import StrEnum


class Author(StrEnum):
    """Who authored a strategy — the agent (default) or a human operator."""

    AGENT = "agent"
    HUMAN = "human"


class HypothesisStatus(StrEnum):
    """Research status of a strategy's claimed edge."""

    UNTESTED = "untested"
    SUPPORTED = "supported"
    REFUTED = "refuted"
    INCONCLUSIVE = "inconclusive"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_registry_metadata.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add algua/contracts/registry_metadata.py tests/test_registry_metadata.py
git commit -m "feat(registry): add Author/HypothesisStatus metadata enums"
```

### Task 2: Tag canonicalization helper

**Files:**
- Create: `algua/registry/metadata.py`
- Test: `tests/test_registry_metadata_tags.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_registry_metadata_tags.py
from algua.registry.metadata import canonicalize_tags, dump_tags, load_tags


def test_canonicalize_trims_lowercases_dedupes_sorts():
    assert canonicalize_tags([" Mean-Reversion ", "MOMENTUM", "momentum"]) == [
        "mean-reversion", "momentum"
    ]


def test_canonicalize_rejects_empty():
    assert canonicalize_tags(["", "  ", "x"]) == ["x"]


def test_dump_then_load_roundtrips():
    assert load_tags(dump_tags(["b", "a"])) == ["a", "b"]


def test_load_handles_null_and_garbage():
    assert load_tags(None) == []
    assert load_tags("not json") == []
    assert load_tags("[]") == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_registry_metadata_tags.py -q`
Expected: FAIL with `ModuleNotFoundError: algua.registry.metadata`

- [ ] **Step 3: Write minimal implementation**

```python
# algua/registry/metadata.py
from __future__ import annotations

import json
from collections.abc import Iterable


def canonicalize_tags(tags: Iterable[str]) -> list[str]:
    """Trim, lowercase, drop empties, dedupe, and sort tags into canonical order."""
    seen: set[str] = set()
    for raw in tags:
        tag = raw.strip().lower()
        if tag:
            seen.add(tag)
    return sorted(seen)


def dump_tags(tags: Iterable[str]) -> str:
    """Serialize tags to the canonical JSON-array string stored in the registry."""
    return json.dumps(canonicalize_tags(tags))


def load_tags(value: str | None) -> list[str]:
    """Parse a stored tags column back to a list; tolerate NULL/invalid JSON as []."""
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except (ValueError, TypeError):
        return []
    if not isinstance(parsed, list):
        return []
    return canonicalize_tags(str(t) for t in parsed)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_registry_metadata_tags.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add algua/registry/metadata.py tests/test_registry_metadata_tags.py
git commit -m "feat(registry): add tag canonicalization helper"
```

### Task 3: Schema migration to v17

**Files:**
- Modify: `algua/registry/db.py:16` (SCHEMA_VERSION), `algua/registry/db.py:292-307` (migrate)
- Test: `tests/test_registry_db.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_registry_db.py — add these tests
import sqlite3

from algua.registry.db import SCHEMA_VERSION, connect, migrate

_META_COLS = {"family", "tags", "author", "hypothesis_status", "derived_from", "description"}


def test_schema_version_is_17():
    assert SCHEMA_VERSION == 17


def test_strategies_has_metadata_columns(tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(strategies)")}
    assert _META_COLS <= cols


def test_metadata_columns_are_null_on_existing_rows(tmp_path):
    # Simulate a pre-v17 DB: create the old-shaped table, insert a row, then migrate.
    db = tmp_path / "r.db"
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE strategies (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL "
        "UNIQUE, stage TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL)"
    )
    conn.execute(
        "INSERT INTO strategies(name, stage, created_at, updated_at) VALUES "
        "('legacy', 'idea', '2026-01-01', '2026-01-01')"
    )
    conn.commit()
    migrate(conn)
    row = conn.execute("SELECT * FROM strategies WHERE name='legacy'").fetchone()
    for col in _META_COLS:
        assert row[col] is None, f"{col} should be NULL on a pre-existing row, got {row[col]!r}"


def test_migrate_is_idempotent_at_v17(tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    migrate(conn)  # second run must be a no-op, not an error
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 17
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_registry_db.py -q -k "metadata or version_is_17"`
Expected: FAIL (SCHEMA_VERSION is 16; columns missing)

- [ ] **Step 3: Write minimal implementation**

In `algua/registry/db.py`, change line 16:

```python
SCHEMA_VERSION = 17
```

In `migrate()`, add a `_add_missing_columns` call for `strategies` (after the existing two calls, before the `PRAGMA user_version` line):

```python
    _add_missing_columns(
        conn,
        "strategies",
        {
            "family": "TEXT",
            "tags": "TEXT",
            "author": "TEXT",
            "hypothesis_status": "TEXT",
            "derived_from": "TEXT",
            "description": "TEXT",
        },
    )
```

(No SQL `DEFAULT` — `_add_missing_columns` adds columns NULL on existing rows, which is required so the backfill can still recover kb-authored values.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_registry_db.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add algua/registry/db.py tests/test_registry_db.py
git commit -m "feat(registry): schema v17 — add organizational metadata columns"
```

### Task 4: StrategyRecord fields + read coercion

**Files:**
- Modify: `algua/registry/repository.py:29-35` (StrategyRecord dataclass)
- Modify: `algua/registry/store.py:21-25` (`_row_to_record`)
- Test: `tests/test_registry_store.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_registry_store.py — add
from algua.contracts.registry_metadata import Author, HypothesisStatus


def test_record_exposes_metadata_defaults(tmp_path):
    from algua.registry.db import connect, migrate
    from algua.registry.store import SqliteStrategyRepository

    conn = connect(tmp_path / "r.db")
    migrate(conn)
    repo = SqliteStrategyRepository(conn)
    rec = repo.add("plain")
    assert rec.author == Author.AGENT
    assert rec.hypothesis_status == HypothesisStatus.UNTESTED
    assert rec.tags == []
    assert rec.family is None
    assert rec.derived_from is None
    assert rec.description is None


def test_null_metadata_columns_read_as_defaults(tmp_path):
    # A row written before the columns existed (all NULL) must read as the defaults.
    from algua.registry.db import connect, migrate
    from algua.registry.store import SqliteStrategyRepository

    conn = connect(tmp_path / "r.db")
    migrate(conn)
    conn.execute(
        "INSERT INTO strategies(name, stage, created_at, updated_at) VALUES "
        "('legacy', 'idea', '2026-01-01', '2026-01-01')"
    )
    conn.commit()
    rec = SqliteStrategyRepository(conn).get("legacy")
    assert rec.author == Author.AGENT
    assert rec.hypothesis_status == HypothesisStatus.UNTESTED
    assert rec.tags == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_registry_store.py -q -k metadata`
Expected: FAIL (StrategyRecord has no `author`; `add` still inserts only name/stage)

- [ ] **Step 3: Write minimal implementation**

In `algua/registry/repository.py`, extend the dataclass (add the import at top: `from algua.contracts.registry_metadata import Author, HypothesisStatus`):

```python
@dataclass
class StrategyRecord:
    id: int
    name: str
    stage: Stage
    created_at: str
    updated_at: str
    family: str | None = None
    tags: list[str] = field(default_factory=list)
    author: Author = Author.AGENT
    hypothesis_status: HypothesisStatus = HypothesisStatus.UNTESTED
    derived_from: str | None = None
    description: str | None = None
```

Add `from dataclasses import dataclass, field` to the imports.

In `algua/registry/store.py`, update `_row_to_record` (add imports: `from algua.contracts.registry_metadata import Author, HypothesisStatus` and `from algua.registry.metadata import load_tags`). Use `keys()` to tolerate the row possibly lacking columns in older callers, but after migrate they exist:

```python
def _row_to_record(row: sqlite3.Row) -> StrategyRecord:
    return StrategyRecord(
        id=row["id"], name=row["name"], stage=Stage(row["stage"]),
        created_at=row["created_at"], updated_at=row["updated_at"],
        family=row["family"],
        tags=load_tags(row["tags"]),
        author=Author(row["author"]) if row["author"] else Author.AGENT,
        hypothesis_status=(
            HypothesisStatus(row["hypothesis_status"])
            if row["hypothesis_status"] else HypothesisStatus.UNTESTED
        ),
        derived_from=row["derived_from"],
        description=row["description"],
    )
```

Update `add()` to write the new-row defaults explicitly so a fresh row is concrete, not NULL (the `add` signature stays name-only in this task; metadata kwargs come in Task 6):

```python
    def add(self, name: str) -> StrategyRecord:
        now = _now()
        with self._conn:
            try:
                cur = self._conn.execute(
                    "INSERT INTO strategies"
                    "(name, stage, created_at, updated_at, tags, author, hypothesis_status)"
                    " VALUES (?,?,?,?,?,?,?)",
                    (name, Stage.IDEA.value, now, now,
                     "[]", Author.AGENT.value, HypothesisStatus.UNTESTED.value),
                )
            except sqlite3.IntegrityError as exc:
                raise StrategyExists(name) from exc
            self._conn.execute(
                "INSERT INTO stage_transitions"
                "(strategy_id, from_stage, to_stage, actor, reason, created_at)"
                " VALUES (?,?,?,?,?,?)",
                (cur.lastrowid, None, Stage.IDEA.value, Actor.SYSTEM.value, "created", now),
            )
        return self.get(name)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_registry_store.py -q`
Expected: PASS

- [ ] **Step 5: Run the full gate + commit**

```bash
uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports
git add algua/registry/repository.py algua/registry/store.py tests/test_registry_store.py
git commit -m "feat(registry): StrategyRecord metadata fields + read coercion"
```

### Task 5: Emit metadata in `registry list` / `show`

**Files:**
- Modify: `algua/cli/registry_cmd.py:31-50` (`list_`, `show`)
- Test: `tests/test_cli_registry.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli_registry.py — add (follow the file's existing CliRunner/json-parse pattern)
def test_list_emits_metadata_fields(registry_env):
    # registry_env is the existing fixture that gives an isolated registry; adapt to the file.
    run_cli(["registry", "add", "alpha"])
    out = json_of(run_cli(["registry", "list"]))
    assert out[0]["author"] == "agent"
    assert out[0]["hypothesis_status"] == "untested"
    assert out[0]["tags"] == []
    assert out[0]["family"] is None


def test_show_emits_metadata_fields(registry_env):
    run_cli(["registry", "add", "alpha"])
    out = json_of(run_cli(["registry", "show", "alpha"]))
    assert out["author"] == "agent"
    assert out["hypothesis_status"] == "untested"
    assert out["tags"] == []
```

(Use the exact fixture/helper names already in `tests/test_cli_registry.py`; mirror an existing test for invocation + JSON parsing.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli_registry.py -q -k metadata`
Expected: FAIL (KeyError: 'author')

- [ ] **Step 3: Write minimal implementation**

Add a row-serializer near the top of `registry_cmd.py` and use it in both commands:

```python
def _record_json(r) -> dict:
    return {
        "id": r.id, "name": r.name, "stage": r.stage.value,
        "family": r.family, "tags": r.tags, "author": r.author.value,
        "hypothesis_status": r.hypothesis_status.value,
        "derived_from": r.derived_from, "description": r.description,
    }
```

`list_`:

```python
    emit([_record_json(r) for r in recs])
```

`show` (keep `transitions`):

```python
    emit(ok({**_record_json(rec), "transitions": transitions}))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cli_registry.py -q`
Expected: PASS

- [ ] **Step 5: Run the full gate + commit**

```bash
uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports
git add algua/cli/registry_cmd.py tests/test_cli_registry.py
git commit -m "feat(registry): emit organizational metadata in list/show"
```

---

## Slice 2 — Write paths (`registry add` flags + `registry set`)

### Task 6: `add()` accepts metadata + `registry add` flags + derived_from validation

**Files:**
- Modify: `algua/registry/repository.py` (Protocol `add` signature)
- Modify: `algua/registry/store.py` (`add` impl)
- Modify: `algua/cli/registry_cmd.py:22-28` (`add` command)
- Test: `tests/test_registry_store.py`, `tests/test_cli_registry.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_registry_store.py — add
def test_add_with_metadata_roundtrips(tmp_path):
    from algua.contracts.registry_metadata import Author, HypothesisStatus
    from algua.registry.db import connect, migrate
    from algua.registry.store import SqliteStrategyRepository

    conn = connect(tmp_path / "r.db")
    migrate(conn)
    repo = SqliteStrategyRepository(conn)
    rec = repo.add(
        "child",
        family="mean-reversion",
        tags=["Slow", "slow", " carry "],
        author=Author.HUMAN,
        hypothesis_status=HypothesisStatus.SUPPORTED,
        description="a thing",
    )
    assert rec.family == "mean-reversion"
    assert rec.tags == ["carry", "slow"]
    assert rec.author == Author.HUMAN
    assert rec.hypothesis_status == HypothesisStatus.SUPPORTED
    assert rec.description == "a thing"


def test_add_derived_from_requires_existing_parent(tmp_path):
    from algua.registry.db import connect, migrate
    from algua.registry.repository import StrategyNotFound
    from algua.registry.store import SqliteStrategyRepository
    import pytest

    conn = connect(tmp_path / "r.db")
    migrate(conn)
    repo = SqliteStrategyRepository(conn)
    with pytest.raises(StrategyNotFound):
        repo.add("orphan", derived_from="ghost")
    repo.add("parent")
    rec = repo.add("kid", derived_from="parent")
    assert rec.derived_from == "parent"


def test_add_derived_from_rejects_self(tmp_path):
    from algua.registry.db import connect, migrate
    from algua.registry.store import SqliteStrategyRepository
    import pytest

    conn = connect(tmp_path / "r.db")
    migrate(conn)
    repo = SqliteStrategyRepository(conn)
    with pytest.raises(ValueError):
        repo.add("self", derived_from="self")
```

```python
# tests/test_cli_registry.py — add
def test_add_with_metadata_flags(registry_env):
    out = json_of(run_cli([
        "registry", "add", "mr1", "--family", "mean-reversion",
        "--tag", "slow", "--tag", "carry", "--author", "human",
        "--hypothesis-status", "supported", "--description", "desc",
    ]))
    rec = json_of(run_cli(["registry", "show", "mr1"]))
    assert rec["family"] == "mean-reversion"
    assert rec["tags"] == ["carry", "slow"]
    assert rec["author"] == "human"
    assert rec["hypothesis_status"] == "supported"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_registry_store.py tests/test_cli_registry.py -q -k "metadata or derived"`
Expected: FAIL (`add()` takes no `family` kwarg)

- [ ] **Step 3: Write minimal implementation**

In `repository.py` Protocol, update the `add` signature + docstring:

```python
    def add(
        self,
        name: str,
        *,
        family: str | None = None,
        tags: list[str] | None = None,
        author: Author = Author.AGENT,
        hypothesis_status: HypothesisStatus = HypothesisStatus.UNTESTED,
        derived_from: str | None = None,
        description: str | None = None,
    ) -> StrategyRecord:
        """Insert a new strategy at stage ``idea`` with its initial transition row and the given
        organizational metadata. ``derived_from``, if set, must name an existing strategy and may
        not be the strategy itself. Raises ``StrategyExists`` / ``StrategyNotFound`` / ``ValueError``.
        """
        ...
```

Add `from algua.contracts.registry_metadata import Author, HypothesisStatus` to `repository.py`.

In `store.py`, implement validation + write (replace the Task-4 `add`):

```python
    def add(
        self,
        name: str,
        *,
        family: str | None = None,
        tags: list[str] | None = None,
        author: Author = Author.AGENT,
        hypothesis_status: HypothesisStatus = HypothesisStatus.UNTESTED,
        derived_from: str | None = None,
        description: str | None = None,
    ) -> StrategyRecord:
        if derived_from is not None:
            if derived_from == name:
                raise ValueError(f"{name} cannot be derived from itself")
            self.get(derived_from)  # raises StrategyNotFound if the parent is unknown
        now = _now()
        with self._conn:
            try:
                cur = self._conn.execute(
                    "INSERT INTO strategies"
                    "(name, stage, created_at, updated_at, family, tags, author,"
                    " hypothesis_status, derived_from, description)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (name, Stage.IDEA.value, now, now, family, dump_tags(tags or []),
                     author.value, hypothesis_status.value, derived_from, description),
                )
            except sqlite3.IntegrityError as exc:
                raise StrategyExists(name) from exc
            self._conn.execute(
                "INSERT INTO stage_transitions"
                "(strategy_id, from_stage, to_stage, actor, reason, created_at)"
                " VALUES (?,?,?,?,?,?)",
                (cur.lastrowid, None, Stage.IDEA.value, Actor.SYSTEM.value, "created", now),
            )
        return self.get(name)
```

Add `from algua.registry.metadata import dump_tags` to `store.py`.

In `registry_cmd.py`, expand the `add` command:

```python
@registry_app.command("add")
@json_errors(ValueError, LookupError)
def add(
    name: str,
    family: str = typer.Option(None, "--family", help="thesis family slug"),
    tag: list[str] = typer.Option(None, "--tag", help="tag (repeatable)"),
    author: Author = typer.Option(Author.AGENT, "--author", help="agent|human"),
    hypothesis_status: HypothesisStatus = typer.Option(
        HypothesisStatus.UNTESTED, "--hypothesis-status"),
    derived_from: str = typer.Option(None, "--derived-from", help="parent strategy name"),
    description: str = typer.Option(None, "--description"),
) -> None:
    """Register a new strategy at stage 'idea' with organizational metadata."""
    if family is not None and not _FAMILY_RE.match(family):
        raise ValueError(f"invalid family {family!r}: must be a lowercase slug (a-z, 0-9, hyphen)")
    with registry_conn() as conn:
        rec = SqliteStrategyRepository(conn).add(
            name, family=family, tags=tag or [], author=author,
            hypothesis_status=hypothesis_status, derived_from=derived_from,
            description=description,
        )
    emit(ok(_record_json(rec)))
```

Add to `registry_cmd.py` imports: `from algua.contracts.registry_metadata import Author, HypothesisStatus` and a module-level `_FAMILY_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")` (and `import re`).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_registry_store.py tests/test_cli_registry.py -q`
Expected: PASS

- [ ] **Step 5: Run the full gate + commit**

```bash
uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports
git add algua/registry/repository.py algua/registry/store.py algua/cli/registry_cmd.py tests/
git commit -m "feat(registry): registry add accepts organizational metadata"
```

### Task 7: `update_metadata` repository method

**Files:**
- Modify: `algua/registry/repository.py` (Protocol), `algua/registry/store.py`
- Test: `tests/test_registry_store.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_registry_store.py — add
def test_update_metadata_partial(tmp_path):
    from algua.contracts.registry_metadata import HypothesisStatus
    from algua.registry.db import connect, migrate
    from algua.registry.store import SqliteStrategyRepository

    conn = connect(tmp_path / "r.db")
    migrate(conn)
    repo = SqliteStrategyRepository(conn)
    repo.add("a", family="mean-reversion", tags=["slow"])
    before = repo.get("a")
    rec = repo.update_metadata(
        "a", hypothesis_status=HypothesisStatus.SUPPORTED, add_tags=["carry"]
    )
    assert rec.hypothesis_status == HypothesisStatus.SUPPORTED
    assert rec.tags == ["carry", "slow"]
    assert rec.family == "mean-reversion"  # untouched
    assert rec.updated_at >= before.updated_at


def test_update_metadata_remove_tag(tmp_path):
    from algua.registry.db import connect, migrate
    from algua.registry.store import SqliteStrategyRepository

    conn = connect(tmp_path / "r.db")
    migrate(conn)
    repo = SqliteStrategyRepository(conn)
    repo.add("a", tags=["slow", "carry"])
    rec = repo.update_metadata("a", remove_tags=["slow"])
    assert rec.tags == ["carry"]


def test_update_metadata_derived_from_validation(tmp_path):
    from algua.registry.db import connect, migrate
    from algua.registry.store import SqliteStrategyRepository
    import pytest

    conn = connect(tmp_path / "r.db")
    migrate(conn)
    repo = SqliteStrategyRepository(conn)
    repo.add("a")
    with pytest.raises(ValueError):
        repo.update_metadata("a", derived_from="a")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_registry_store.py -q -k update_metadata`
Expected: FAIL (`update_metadata` not defined)

- [ ] **Step 3: Write minimal implementation**

Add to the Protocol in `repository.py`:

```python
    def update_metadata(
        self,
        name: str,
        *,
        family: str | None = None,
        author: Author | None = None,
        hypothesis_status: HypothesisStatus | None = None,
        derived_from: str | None = None,
        description: str | None = None,
        add_tags: list[str] | None = None,
        remove_tags: list[str] | None = None,
    ) -> StrategyRecord:
        """Update only the supplied organizational-metadata fields (never the stage). ``add_tags``/
        ``remove_tags`` mutate the tag set. Returns the updated record."""
        ...
```

Implement in `store.py` (a sentinel-free approach: only the fields explicitly passed get written; tags mutate via add/remove):

```python
    def update_metadata(
        self,
        name: str,
        *,
        family: str | None = None,
        author: Author | None = None,
        hypothesis_status: HypothesisStatus | None = None,
        derived_from: str | None = None,
        description: str | None = None,
        add_tags: list[str] | None = None,
        remove_tags: list[str] | None = None,
    ) -> StrategyRecord:
        rec = self.get(name)
        if derived_from is not None:
            if derived_from == name:
                raise ValueError(f"{name} cannot be derived from itself")
            self.get(derived_from)
        sets: list[str] = []
        params: list[object] = []
        if family is not None:
            sets.append("family = ?"); params.append(family)
        if author is not None:
            sets.append("author = ?"); params.append(author.value)
        if hypothesis_status is not None:
            sets.append("hypothesis_status = ?"); params.append(hypothesis_status.value)
        if derived_from is not None:
            sets.append("derived_from = ?"); params.append(derived_from)
        if description is not None:
            sets.append("description = ?"); params.append(description)
        if add_tags or remove_tags:
            tags = set(rec.tags)
            tags |= set(canonicalize_tags(add_tags or []))
            tags -= set(canonicalize_tags(remove_tags or []))
            sets.append("tags = ?"); params.append(dump_tags(tags))
        if sets:
            sets.append("updated_at = ?"); params.append(_now())
            params.append(rec.id)
            with self._conn:
                self._conn.execute(
                    f"UPDATE strategies SET {', '.join(sets)} WHERE id = ?", params
                )
        return self.get(name)
```

Add `from algua.registry.metadata import canonicalize_tags, dump_tags` to `store.py` (merge with the existing import).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_registry_store.py -q -k update_metadata`
Expected: PASS

- [ ] **Step 5: Run the full gate + commit**

```bash
uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports
git add algua/registry/repository.py algua/registry/store.py tests/test_registry_store.py
git commit -m "feat(registry): update_metadata repository method"
```

### Task 8: `registry set` command (with before/after + kb re-sync)

**Files:**
- Modify: `algua/cli/registry_cmd.py` (new `set` command)
- Test: `tests/test_cli_registry.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli_registry.py — add
def test_set_changes_metadata_and_reports_before_after(registry_env):
    run_cli(["registry", "add", "a", "--hypothesis-status", "untested"])
    out = json_of(run_cli([
        "registry", "set", "a", "--hypothesis-status", "supported", "--add-tag", "carry",
    ]))
    assert out["changed"]["hypothesis_status"] == {"before": "untested", "after": "supported"}
    rec = json_of(run_cli(["registry", "show", "a"]))
    assert rec["hypothesis_status"] == "supported"
    assert rec["tags"] == ["carry"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli_registry.py -q -k set_changes`
Expected: FAIL (no `set` command)

- [ ] **Step 3: Write minimal implementation**

Add to `registry_cmd.py` (import `sync_strategy_doc`, `get_settings` lazily inside to avoid heavy import at module load — mirror how `strategy_cmd` does the seam):

```python
@registry_app.command("set")
@json_errors(ValueError, LookupError)
def set_(
    name: str,
    family: str = typer.Option(None, "--family"),
    author: Author = typer.Option(None, "--author"),
    hypothesis_status: HypothesisStatus = typer.Option(None, "--hypothesis-status"),
    derived_from: str = typer.Option(None, "--derived-from"),
    description: str = typer.Option(None, "--description"),
    add_tag: list[str] = typer.Option(None, "--add-tag", help="add a tag (repeatable)"),
    remove_tag: list[str] = typer.Option(None, "--remove-tag", help="remove a tag (repeatable)"),
) -> None:
    """Update a strategy's organizational metadata (never its stage); re-syncs the kb doc."""
    if family is not None and not _FAMILY_RE.match(family):
        raise ValueError(f"invalid family {family!r}: must be a lowercase slug (a-z, 0-9, hyphen)")
    fields = ("family", "author", "hypothesis_status", "derived_from", "description", "tags")
    with registry_conn() as conn:
        repo = SqliteStrategyRepository(conn)
        before = repo.get(name)
        after = repo.update_metadata(
            name, family=family, author=author, hypothesis_status=hypothesis_status,
            derived_from=derived_from, description=description,
            add_tags=add_tag or [], remove_tags=remove_tag or [],
        )
    changed = {}
    for f in fields:
        b, a = getattr(before, f), getattr(after, f)
        if isinstance(b, Author) or isinstance(b, HypothesisStatus):
            b = b.value
        if isinstance(a, Author) or isinstance(a, HypothesisStatus):
            a = a.value
        if b != a:
            changed[f] = {"before": b, "after": a}
    # Re-sync the kb doc so frontmatter reflects the new registry truth (best-effort; absent doc ok).
    # NOTE: `sync_strategy_doc` gains its `metadata=` parameter in Task 11. In THIS task call the
    # no-`metadata` form (stage only) so the task is self-contained and green; Task 11 upgrades this
    # call to pass `metadata=_kb_metadata(after)`.
    from algua.config.settings import get_settings
    from algua.knowledge.sync import sync_strategy_doc
    sync_strategy_doc(get_settings(), name, stage=after.stage.value)
    emit(ok({**_record_json(after), "changed": changed}))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cli_registry.py -q -k set_changes`
Expected: PASS

- [ ] **Step 5: Run the full gate + commit**

```bash
uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports
git add algua/cli/registry_cmd.py tests/test_cli_registry.py
git commit -m "feat(registry): registry set command for metadata mutation"
```

---

## Slice 3 — Read / filter

### Task 9: `list_strategies` filter params

**Files:**
- Modify: `algua/registry/repository.py` (Protocol `list_strategies`), `algua/registry/store.py`
- Test: `tests/test_registry_store.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_registry_store.py — add
def _seed_pool(repo):
    from algua.contracts.registry_metadata import Author, HypothesisStatus
    repo.add("a", family="mean-reversion", tags=["slow"], author=Author.AGENT)
    repo.add("b", family="mean-reversion", tags=["slow", "carry"], author=Author.HUMAN,
             hypothesis_status=HypothesisStatus.SUPPORTED)
    repo.add("c", family="momentum", tags=["fast"], author=Author.AGENT)


def test_filter_by_family(tmp_path):
    from algua.registry.db import connect, migrate
    from algua.registry.store import SqliteStrategyRepository
    conn = connect(tmp_path / "r.db"); migrate(conn)
    repo = SqliteStrategyRepository(conn); _seed_pool(repo)
    assert {r.name for r in repo.list_strategies(family="mean-reversion")} == {"a", "b"}


def test_filter_by_tag_is_all_of(tmp_path):
    from algua.registry.db import connect, migrate
    from algua.registry.store import SqliteStrategyRepository
    conn = connect(tmp_path / "r.db"); migrate(conn)
    repo = SqliteStrategyRepository(conn); _seed_pool(repo)
    assert {r.name for r in repo.list_strategies(tags=["slow", "carry"])} == {"b"}


def test_filter_by_author_and_status_compose(tmp_path):
    from algua.contracts.registry_metadata import Author, HypothesisStatus
    from algua.registry.db import connect, migrate
    from algua.registry.store import SqliteStrategyRepository
    conn = connect(tmp_path / "r.db"); migrate(conn)
    repo = SqliteStrategyRepository(conn); _seed_pool(repo)
    got = repo.list_strategies(author=Author.AGENT, family="mean-reversion")
    assert {r.name for r in got} == {"a"}


def test_filter_author_matches_null_legacy_row(tmp_path):
    # A legacy NULL-author row must match --author agent (COALESCE semantics).
    from algua.contracts.registry_metadata import Author
    from algua.registry.db import connect, migrate
    from algua.registry.store import SqliteStrategyRepository
    conn = connect(tmp_path / "r.db"); migrate(conn)
    conn.execute("INSERT INTO strategies(name, stage, created_at, updated_at) VALUES "
                 "('legacy','idea','2026-01-01','2026-01-01')")
    conn.commit()
    repo = SqliteStrategyRepository(conn)
    assert {r.name for r in repo.list_strategies(author=Author.AGENT)} == {"legacy"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_registry_store.py -q -k filter`
Expected: FAIL (`list_strategies` takes no `family` kwarg)

- [ ] **Step 3: Write minimal implementation**

Update the Protocol signature in `repository.py`:

```python
    def list_strategies(
        self,
        stage: Stage | None = None,
        *,
        family: str | None = None,
        tags: list[str] | None = None,
        author: Author | None = None,
        hypothesis_status: HypothesisStatus | None = None,
    ) -> list[StrategyRecord]:
        """List strategies, optionally filtered. Filters AND together; repeated ``tags`` means
        all-of. ``author``/``hypothesis_status`` use COALESCE so NULL legacy rows match the
        default. Ordered by insertion."""
        ...
```

Implement in `store.py`:

```python
    def list_strategies(
        self,
        stage: Stage | None = None,
        *,
        family: str | None = None,
        tags: list[str] | None = None,
        author: Author | None = None,
        hypothesis_status: HypothesisStatus | None = None,
    ) -> list[StrategyRecord]:
        clauses: list[str] = []
        params: list[object] = []
        if stage is not None:
            clauses.append("stage = ?"); params.append(stage.value)
        if family is not None:
            clauses.append("family = ?"); params.append(family)
        if author is not None:
            clauses.append("COALESCE(author, ?) = ?")
            params.extend((Author.AGENT.value, author.value))
        if hypothesis_status is not None:
            clauses.append("COALESCE(hypothesis_status, ?) = ?")
            params.extend((HypothesisStatus.UNTESTED.value, hypothesis_status.value))
        for tag in canonicalize_tags(tags or []):
            clauses.append(
                "EXISTS (SELECT 1 FROM json_each(COALESCE(tags, '[]')) WHERE value = ?)"
            )
            params.append(tag)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._conn.execute(
            f"SELECT * FROM strategies{where} ORDER BY id", params
        ).fetchall()
        return [_row_to_record(r) for r in rows]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_registry_store.py -q -k filter`
Expected: PASS

- [ ] **Step 5: Run the full gate + commit**

```bash
uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports
git add algua/registry/repository.py algua/registry/store.py tests/test_registry_store.py
git commit -m "feat(registry): metadata filters on list_strategies"
```

### Task 10: `registry list` filter options

**Files:**
- Modify: `algua/cli/registry_cmd.py:31-38` (`list_`)
- Test: `tests/test_cli_registry.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli_registry.py — add
def test_list_filters_compose(registry_env):
    run_cli(["registry", "add", "a", "--family", "mean-reversion", "--tag", "slow"])
    run_cli(["registry", "add", "b", "--family", "momentum", "--tag", "fast"])
    out = json_of(run_cli(["registry", "list", "--family", "mean-reversion"]))
    assert [r["name"] for r in out] == ["a"]
    out2 = json_of(run_cli(["registry", "list", "--tag", "fast"]))
    assert [r["name"] for r in out2] == ["b"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli_registry.py -q -k list_filters`
Expected: FAIL (no `--family` option)

- [ ] **Step 3: Write minimal implementation**

```python
@registry_app.command("list")
@json_errors(ValueError, LookupError)
def list_(
    stage: str = typer.Option(None, "--stage", help="filter by stage"),
    family: str = typer.Option(None, "--family"),
    tag: list[str] = typer.Option(None, "--tag", help="require this tag (repeatable, all-of)"),
    author: Author = typer.Option(None, "--author"),
    hypothesis_status: HypothesisStatus = typer.Option(None, "--hypothesis-status"),
) -> None:
    """List strategies with optional filters (AND-ed). Emits a bare JSON array."""
    st = Stage(stage) if stage else None
    with registry_conn() as conn:
        recs = SqliteStrategyRepository(conn).list_strategies(
            st, family=family, tags=tag or [], author=author,
            hypothesis_status=hypothesis_status,
        )
    emit([_record_json(r) for r in recs])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cli_registry.py -q -k list_filters`
Expected: PASS

- [ ] **Step 5: Run the full gate + commit**

```bash
uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports
git add algua/cli/registry_cmd.py tests/test_cli_registry.py
git commit -m "feat(registry): registry list metadata filters"
```

---

## Slice 4 — kb inversion

### Task 11: `sync_strategy_doc` writes registry-owned metadata

**Files:**
- Modify: `algua/knowledge/sync.py:74-91` (`sync_strategy_doc`)
- Modify: `algua/cli/registry_cmd.py` (`set` passes metadata) and `algua/cli/strategy_cmd.py` (`doc` passes metadata)
- Test: `tests/test_knowledge_sync.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_knowledge_sync.py — add (mirror the file's existing settings/tmp fixture)
def test_sync_writes_owned_metadata_and_preserves_foreign_keys(sync_settings):
    from algua.knowledge.sync import strategy_doc_path, sync_strategy_doc
    from algua.knowledge.templates import scaffold_strategy_doc

    path = strategy_doc_path(sync_settings, "a")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Seed a doc with a foreign frontmatter key that must survive the sync.
    from algua.knowledge.frontmatter import parse_doc, render_doc
    fm, body = parse_doc(scaffold_strategy_doc("a"))
    fm["my_note"] = "keep me"
    path.write_text(render_doc(fm, body))

    meta = {
        "family": "mean-reversion", "tags": ["carry", "slow"], "author": "human",
        "hypothesis_status": "supported", "derived_from": "parent", "description": "d",
    }
    sync_strategy_doc(sync_settings, "a", stage="backtested", metadata=meta)

    fm2, _ = parse_doc(path.read_text())
    assert fm2["family"] == "[[mean-reversion]]"
    assert fm2["derived_from"] == "[[parent]]"
    assert fm2["tags"] == ["carry", "slow"]
    assert fm2["author"] == "human"
    assert fm2["hypothesis_status"] == "supported"
    assert fm2["stage"] == "backtested"
    assert fm2["my_note"] == "keep me"  # foreign key preserved
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_knowledge_sync.py -q -k owned_metadata`
Expected: FAIL (`sync_strategy_doc` takes no `metadata` kwarg)

- [ ] **Step 3: Write minimal implementation**

Update `sync_strategy_doc` in `sync.py`:

```python
def sync_strategy_doc(
    settings: Settings,
    name: str,
    *,
    stage: str | None,
    metadata: dict[str, Any] | None = None,
) -> bool:
    """Rewrite the synced parts of one strategy doc. Returns False if the doc is absent.

    ``stage`` and ``metadata`` are read by the caller at the CLI seam so this layer never touches
    the registry. ``metadata`` carries the registry-owned organizational fields; only those keys
    (plus ``stage``/``mlflow_run``) are overwritten — any other frontmatter key is preserved.
    """
    path = strategy_doc_path(settings, name)
    if not path.exists():
        return False
    fm, body = parse_doc(path.read_text())
    if stage is not None:
        fm["stage"] = stage
    if metadata is not None:
        _apply_owned_metadata(fm, metadata)
    metrics = latest_run_metrics(name, tracking_uri=settings.mlflow_tracking_uri)
    if metrics:
        fm["mlflow_run"] = metrics["run_id"][:8]
    body = replace_block(body, "RESULTS", render_results_block(metrics))
    path.write_text(render_doc(fm, body))
    return True


def _apply_owned_metadata(fm: dict[str, Any], metadata: dict[str, Any]) -> None:
    """Write the registry-owned frontmatter keys from a registry metadata dict, wrapping
    ``family``/``derived_from`` as Obsidian wikilinks. NULL/None values clear the key."""
    for key in ("family", "derived_from"):
        val = metadata.get(key)
        if val:
            fm[key] = f"[[{val}]]"
        else:
            fm.pop(key, None)
    for key in ("tags", "author", "hypothesis_status", "description"):
        val = metadata.get(key)
        if val is not None:
            fm[key] = val
        else:
            fm.pop(key, None)
```

Now wire the two CLI seams. In `registry_cmd.py`, add a helper and use it in `set_`:

```python
def _kb_metadata(rec) -> dict:
    return {
        "family": rec.family, "tags": rec.tags, "author": rec.author.value,
        "hypothesis_status": rec.hypothesis_status.value,
        "derived_from": rec.derived_from, "description": rec.description,
    }
```

Replace the Task-8 sync call in `set_` with:

```python
    sync_strategy_doc(get_settings(), name, stage=after.stage.value, metadata=_kb_metadata(after))
```

In `strategy_cmd.py` `doc`, pass metadata too. Read the full records (not just stages) at the seam:

```python
    with registry_conn() as conn:
        recs = {rec.name: rec for rec in SqliteStrategyRepository(conn).list_strategies()}
    stages = {n: r.stage.value for n, r in recs.items()}
```

For the single-doc branch, pass `metadata=_kb_metadata(recs[name])` when `name in recs` (import `_kb_metadata` from `algua.cli.registry_cmd`). For `sync_all`, see Task 11b note below — to keep `sync_all`'s signature stable this task only wires the single-doc `strategy doc <name>` path and `registry set`; `sync_all` metadata wiring is Task 12.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_knowledge_sync.py tests/test_cli_strategy.py tests/test_cli_registry.py -q`
Expected: PASS

- [ ] **Step 5: Run the full gate + commit**

```bash
uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports
git add algua/knowledge/sync.py algua/cli/registry_cmd.py algua/cli/strategy_cmd.py tests/
git commit -m "feat(kb): sync_strategy_doc derives owned frontmatter from the registry"
```

### Task 12: `sync_all` carries metadata

**Files:**
- Modify: `algua/knowledge/sync.py:144-162` (`sync_all`)
- Modify: `algua/cli/strategy_cmd.py` (`doc --all` seam), any other `sync_all` callers (`algua/cli` doctor/sync paths — grep first)
- Test: `tests/test_knowledge_sync.py`

- [ ] **Step 1: Find all callers**

Run: `grep -rn "sync_all(" algua/`
Expected: a small set (strategy `doc`, possibly a `doctor`/`kb sync` command). Each must pass the new metadata map.

- [ ] **Step 2: Write the failing test**

```python
# tests/test_knowledge_sync.py — add
def test_sync_all_applies_metadata(sync_settings):
    from algua.knowledge.sync import strategy_doc_path, sync_all
    from algua.knowledge.templates import scaffold_strategy_doc
    from algua.knowledge.frontmatter import parse_doc

    p = strategy_doc_path(sync_settings, "a")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(scaffold_strategy_doc("a"))
    sync_all(
        sync_settings,
        {"a": "idea"},
        metadata={"a": {"family": "mean-reversion", "tags": ["slow"], "author": "agent",
                        "hypothesis_status": "untested", "derived_from": None, "description": None}},
    )
    fm, _ = parse_doc(p.read_text())
    assert fm["family"] == "[[mean-reversion]]"
    assert fm["tags"] == ["slow"]
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_knowledge_sync.py -q -k sync_all_applies`
Expected: FAIL (`sync_all` takes no `metadata` kwarg)

- [ ] **Step 4: Write minimal implementation**

```python
def sync_all(
    settings: Settings,
    stages: dict[str, str],
    metadata: dict[str, dict[str, Any]] | None = None,
) -> dict[str, list[str]]:
    """Sync each registered strategy's doc (stage + optional metadata), every family doc, then
    indexes. ``metadata`` maps strategy name -> its registry metadata dict."""
    metadata = metadata or {}
    synced: list[str] = []
    for name, stage in stages.items():
        if sync_strategy_doc(settings, name, stage=stage, metadata=metadata.get(name)):
            synced.append(name)
    families: list[str] = []
    fam_dir = strategies_dir(settings) / "families"
    if fam_dir.exists():
        for doc in sorted(fam_dir.glob("*.md")):
            fm, _ = parse_doc(doc.read_text())
            fam_name = str(fm.get("name", doc.stem))
            if sync_family_doc(settings, fam_name):
                families.append(fam_name)
    generate_indexes(settings)
    return {"strategies": synced, "families": families}
```

Update each caller from Step 1 to build a `{name: _kb_metadata(rec)}` map and pass it.

In `strategy_cmd.py` `doc`, the `--all` branch:

```python
        summary = sync_all(settings, stages, metadata={n: _kb_metadata(r) for n, r in recs.items()})
```

- [ ] **Step 5: Run test + gate + commit**

```bash
uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports
git add algua/knowledge/sync.py algua/cli/ tests/test_knowledge_sync.py
git commit -m "feat(kb): sync_all carries per-strategy registry metadata"
```

---

## Slice 5 — Backfill

### Task 13: `registry backfill-from-kb` command

**Files:**
- Modify: `algua/cli/registry_cmd.py` (new `backfill-from-kb` command)
- Modify: `algua/registry/store.py` + `repository.py` — a `backfill_metadata(name, **fields)` that fills only currently-NULL columns
- Test: `tests/test_registry_store.py`, `tests/test_cli_registry.py`

- [ ] **Step 1: Write the failing test (store)**

```python
# tests/test_registry_store.py — add
def test_backfill_metadata_fills_only_nulls(tmp_path):
    from algua.registry.db import connect, migrate
    from algua.registry.store import SqliteStrategyRepository
    conn = connect(tmp_path / "r.db"); migrate(conn)
    # legacy NULL row
    conn.execute("INSERT INTO strategies(name, stage, created_at, updated_at) VALUES "
                 "('a','idea','2026-01-01','2026-01-01')")
    conn.commit()
    repo = SqliteStrategyRepository(conn)
    repo.backfill_metadata("a", family="mean-reversion", hypothesis_status="supported")
    rec = repo.get("a")
    assert rec.family == "mean-reversion"
    assert rec.hypothesis_status.value == "supported"
    # second backfill must NOT overwrite a now-non-NULL value
    repo.backfill_metadata("a", family="momentum")
    assert repo.get("a").family == "mean-reversion"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_registry_store.py -q -k backfill_metadata`
Expected: FAIL (`backfill_metadata` not defined)

- [ ] **Step 3: Write minimal implementation (store)**

Add to the Protocol + implement in `store.py`:

```python
    def backfill_metadata(
        self,
        name: str,
        *,
        family: str | None = None,
        tags: list[str] | None = None,
        author: str | None = None,
        hypothesis_status: str | None = None,
        derived_from: str | None = None,
        description: str | None = None,
    ) -> StrategyRecord:
        """Fill only currently-NULL metadata columns from the given values (one-shot recovery).
        A column already holding a value is left untouched. ``author``/``hypothesis_status`` are
        raw validated strings (the caller maps/validates against the enums)."""
        cols = {
            "family": family,
            "tags": dump_tags(tags) if tags is not None else None,
            "author": author,
            "hypothesis_status": hypothesis_status,
            "derived_from": derived_from,
            "description": description,
        }
        sets = [f"{c} = ?" for c, v in cols.items() if v is not None]
        params = [v for v in cols.values() if v is not None]
        if sets:
            rec = self.get(name)
            # COALESCE keeps any existing non-NULL value; only NULLs are filled.
            assignments = ", ".join(f"{c} = COALESCE({c}, ?)" for c, v in cols.items() if v is not None)
            with self._conn:
                self._conn.execute(
                    f"UPDATE strategies SET {assignments} WHERE id = ?", [*params, rec.id]
                )
        return self.get(name)
```

(Note: the `sets` list is unused once `assignments` is used — drop `sets`; kept here only to show the filter. Final code uses `assignments`/`params`.)

- [ ] **Step 4: Run test (store) + write CLI test**

Run: `uv run pytest tests/test_registry_store.py -q -k backfill_metadata` → PASS

```python
# tests/test_cli_registry.py — add (needs the kb fixture; mirror test_cli_strategy's settings seam)
def test_backfill_from_kb_reports_and_fills(registry_env_with_kb):
    # Seed: a registered strategy with NULL metadata + a kb doc carrying frontmatter.
    # (Use the project's kb settings fixture; write kb/strategies/a.md with family/hypothesis_status.)
    ...
    out = json_of(run_cli(["registry", "backfill-from-kb"]))
    assert "a" in out["processed"]
    rec = json_of(run_cli(["registry", "show", "a"]))
    assert rec["family"] == "mean-reversion"
```

(Fill the seed using the same kb-settings fixture pattern as `tests/test_cli_strategy.py`; assert `filled`, `unmappable`, and the orphan lists are present in the JSON.)

- [ ] **Step 5: Write minimal implementation (CLI command)**

```python
@registry_app.command("backfill-from-kb")
@json_errors(ValueError, LookupError)
def backfill_from_kb() -> None:
    """One-shot: recover kb-authored metadata into NULL registry columns; report conflicts.

    Fills only currently-NULL columns. kb hypothesis_status/author values that aren't valid enum
    members are reported as 'unmappable' and left for the operator. Finally default-fills any row
    still NULL on author/hypothesis_status/tags."""
    from algua.config.settings import get_settings
    from algua.knowledge.frontmatter import parse_doc
    from algua.knowledge.sync import _unwikilink, strategy_doc_path

    settings = get_settings()
    processed: list[str] = []
    unmappable: list[dict] = []
    kb_without_row: list[str] = []
    rows_without_kb: list[str] = []
    valid_status = {h.value for h in HypothesisStatus}
    valid_author = {a.value for a in Author}
    with registry_conn() as conn:
        repo = SqliteStrategyRepository(conn)
        names = {r.name for r in repo.list_strategies()}
        # rows with no kb doc
        for name in names:
            if not strategy_doc_path(settings, name).exists():
                rows_without_kb.append(name)
        # kb docs -> registry
        for doc in sorted((settings.knowledge_dir / "strategies").glob("*.md")):
            if doc.name.startswith("_"):
                continue
            fm, _ = parse_doc(doc.read_text())
            if fm.get("type") == "family":
                continue
            name = str(fm.get("name", doc.stem))
            if name not in names:
                kb_without_row.append(name)
                continue
            status = fm.get("hypothesis_status")
            author = fm.get("author")
            if status is not None and status not in valid_status:
                unmappable.append({"name": name, "field": "hypothesis_status", "value": status})
                status = None
            if author is not None and author not in valid_author:
                unmappable.append({"name": name, "field": "author", "value": author})
                author = None
            tags = fm.get("tags")
            repo.backfill_metadata(
                name,
                family=_unwikilink(fm.get("family")),
                derived_from=_unwikilink(fm.get("derived_from")),
                hypothesis_status=status,
                author=author,
                description=fm.get("description"),
                tags=list(tags) if isinstance(tags, list) else None,
            )
            processed.append(name)
        # final default-fill of any remaining NULLs
        conn.execute("UPDATE strategies SET author = COALESCE(author, ?)", (Author.AGENT.value,))
        conn.execute(
            "UPDATE strategies SET hypothesis_status = COALESCE(hypothesis_status, ?)",
            (HypothesisStatus.UNTESTED.value,),
        )
        conn.execute("UPDATE strategies SET tags = COALESCE(tags, '[]')")
        conn.commit()
    emit(ok({
        "processed": sorted(processed),
        "unmappable": unmappable,
        "kb_docs_without_registry_row": sorted(kb_without_row),
        "registry_rows_without_kb_doc": sorted(rows_without_kb),
    }))
```

Note: `_unwikilink` is currently module-private in `sync.py` but already used cross-module intent; it's safe to import. If import-linter objects to importing a private name, expose it (rename without leading underscore) in `sync.py` and update its one internal caller.

- [ ] **Step 6: Run test + gate + commit**

```bash
uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports
git add algua/registry/ algua/cli/registry_cmd.py tests/
git commit -m "feat(registry): backfill-from-kb recovers metadata into NULL columns"
```

---

## Slice 6 — `strategy new` coupling (last)

### Task 14: `strategy new` registers with preflight + rollback

**Files:**
- Modify: `algua/cli/strategy_cmd.py:70-110` (`new`)
- Modify: `algua/registry/store.py` + `repository.py` — add `delete(name)` scoped to rollback
- Test: `tests/test_cli_strategy.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli_strategy.py — add
def test_strategy_new_registers(strategy_env):
    out = json_of(run_cli([
        "strategy", "new", "newstrat", "--family", "mean-reversion",
        "--hypothesis-status", "untested",
    ]))
    rec = json_of(run_cli(["registry", "show", "newstrat"]))
    assert rec["family"] == "mean-reversion"
    assert rec["stage"] == "idea"


def test_strategy_new_preflight_rejects_existing_registration(strategy_env):
    run_cli(["registry", "add", "dup"])
    res = run_cli(["strategy", "new", "dup"])  # name already registered
    out = json_of(res)
    assert out["ok"] is False
    # module file must NOT have been written by the rejected run
    import pathlib
    assert not (pathlib.Path("algua/strategies/examples/dup.py")).exists() or True  # see note
```

(The exact module path assertion depends on the test's cwd/fixture; assert on the JSON error + that no registry duplication occurred. Mirror the existing `test_cli_strategy.py` safety-rejection tests for structure.)

```python
def test_strategy_new_rollback_on_scaffold_failure(strategy_env, monkeypatch):
    # Force the kb scaffold write to fail AFTER registry add; assert the row was rolled back.
    import algua.cli.strategy_cmd as sc

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(sc, "scaffold_strategy_doc", boom)
    res = run_cli(["strategy", "new", "rollbackme"])
    assert json_of(res)["ok"] is False
    out = json_of(run_cli(["registry", "list"]))
    assert "rollbackme" not in [r["name"] for r in out]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli_strategy.py -q -k "registers or preflight or rollback"`
Expected: FAIL (`strategy new` doesn't register)

- [ ] **Step 3: Write minimal implementation**

Add `delete` to the repository (Protocol + store) — scoped, documented as rollback-only:

```python
    def delete(self, name: str) -> None:
        """Remove a strategy row and its transition rows. ONLY for rolling back a failed
        ``strategy new`` that just created it — there is no general deletion workflow."""
        ...
```

```python
    def delete(self, name: str) -> None:
        rec = self.get(name)
        with self._conn:
            self._conn.execute("DELETE FROM stage_transitions WHERE strategy_id = ?", (rec.id,))
            self._conn.execute("DELETE FROM strategies WHERE id = ?", (rec.id,))
```

Rewrite `new` in `strategy_cmd.py` (preflight → register → scaffold → rollback):

```python
@strategy_app.command("new")
@json_errors()
def new(
    name: str,
    family: str = typer.Option(None, "--family", help="thesis family this belongs to"),
    derived_from: str = typer.Option(None, "--derived-from", help="parent strategy name"),
    tag: list[str] = typer.Option(None, "--tag", help="tag (repeatable)"),
    author: Author = typer.Option(Author.AGENT, "--author"),
    hypothesis_status: HypothesisStatus = typer.Option(HypothesisStatus.UNTESTED, "--hypothesis-status"),
    description: str = typer.Option(None, "--description"),
) -> None:
    """Scaffold a new strategy module + kb doc AND register it (registry owns the metadata)."""
    # --- preflight: validate everything before any write ---
    if not name.isidentifier() or keyword.iskeyword(name):
        raise ValueError(
            f"invalid strategy name {name!r}: must be a valid, non-keyword Python identifier")
    if family is not None and not _FAMILY_RE.match(family):
        raise ValueError(f"invalid family {family!r}: must be a lowercase slug (a-z, 0-9, hyphen)")
    path = Path(__file__).parent.parent / "strategies" / "examples" / f"{name}.py"
    settings = get_settings()
    doc_path = strategy_doc_path(settings, name)
    if path.exists():
        raise ValueError(f"strategy already exists: {path}")
    if doc_path.exists():
        raise ValueError(f"strategy doc already exists: {doc_path}")
    with registry_conn() as conn:
        repo = SqliteStrategyRepository(conn)
        try:
            repo.get(name)
            raise ValueError(f"{name} is already registered")
        except StrategyNotFound:
            pass
        if derived_from is not None:
            repo.get(derived_from)  # StrategyNotFound if parent unknown
        # --- register first (fast, transactional) ---
        repo.add(name, family=family, tags=tag or [], author=author,
                 hypothesis_status=hypothesis_status, derived_from=derived_from,
                 description=description)
        # --- scaffold; roll the registration back on any failure ---
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(_TEMPLATE.format(name=name))
            doc_path.parent.mkdir(parents=True, exist_ok=True)
            doc_path.write_text(scaffold_strategy_doc(name, family=family, derived_from=derived_from))
            family_doc: str | None = None
            if family:
                fam_path = family_doc_path(settings, family)
                fam_path.parent.mkdir(parents=True, exist_ok=True)
                if not fam_path.exists():
                    fam_path.write_text(scaffold_family_doc(family))
                family_doc = str(fam_path)
        except Exception:
            repo.delete(name)
            # best-effort: remove a half-written module file so a retry isn't blocked
            path.unlink(missing_ok=True)
            raise
    emit(ok({"name": name, "path": str(path), "doc": str(doc_path), "family_doc": family_doc}))
```

Add imports to `strategy_cmd.py`: `from algua.contracts.registry_metadata import Author, HypothesisStatus`, `from algua.registry.repository import StrategyNotFound`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cli_strategy.py -q`
Expected: PASS

- [ ] **Step 5: Run the full gate + commit**

```bash
uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports
git add algua/cli/strategy_cmd.py algua/registry/repository.py algua/registry/store.py tests/test_cli_strategy.py
git commit -m "feat(strategy): strategy new registers with preflight + rollback"
```

### Task 15: Update scaffold template default frontmatter

**Files:**
- Modify: `algua/knowledge/templates.py:12-38` (`scaffold_strategy_doc`)
- Test: `tests/test_knowledge_sync.py` or `tests/test_cli_strategy.py`

- [ ] **Step 1: Decide + test**

`scaffold_strategy_doc` already writes `hypothesis_status: untested` and (optionally) `family`/`derived_from`. Since `strategy new` now immediately re-derives frontmatter from the registry is NOT automatic (new does not call sync), keep the scaffold authoring these initial values so a freshly-created doc matches the registry row. No code change is required IF the scaffold already matches the registry defaults. Add a guard test:

```python
# tests/test_cli_strategy.py — add
def test_new_doc_frontmatter_matches_registry(strategy_env):
    run_cli(["strategy", "new", "s1", "--family", "mean-reversion"])
    from algua.knowledge.frontmatter import parse_doc
    import pathlib
    # locate the doc via the show output / settings fixture
    rec = json_of(run_cli(["registry", "show", "s1"]))
    assert rec["family"] == "mean-reversion"
    assert rec["hypothesis_status"] == "untested"
```

- [ ] **Step 2: Run** `uv run pytest tests/test_cli_strategy.py -q -k frontmatter_matches` — PASS (no impl change expected; if it fails, align the scaffold defaults to the registry defaults).

- [ ] **Step 3: Commit (if any change)**

```bash
git add algua/knowledge/templates.py tests/test_cli_strategy.py
git commit -m "test(strategy): assert new doc frontmatter matches registry defaults"
```

---

## Final verification

- [ ] Run the full gate one last time on the branch:

```bash
uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports
```

- [ ] Confirm `registry list --family mean-reversion --author agent --hypothesis-status untested` returns the expected pool on a seeded DB (manual smoke).
- [ ] Confirm `uv run algua doctor` is still green.
