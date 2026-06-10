# Rename lifecycle stage `shortlisted` → `candidate` (#120) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename the lifecycle stage value `shortlisted` → `candidate` everywhere (enum, code, DB data, tests, docs/skills), with an idempotent DB migration that rewrites existing rows.

**Architecture:** Hard rename, no compat alias. A `_migrate_shortlisted_to_candidate` runs before the schema bootstrap (per-table guarded, no `user_version` gating, idempotent); `SCHEMA_VERSION` 19→20. The enum + all code/test references change atomically so the suite stays green. The "shortlist gate" *mechanism* name (gate_evaluations / shortlist token) is preserved — only the *stage value* changes.

**Tech Stack:** Python 3.12, SQLite (sqlite3), pytest, typer CLI, StrEnum.

Spec: `docs/superpowers/specs/2026-06-09-rename-shortlisted-to-candidate-issue-120-design.md`

---

## File Structure

- `algua/contracts/lifecycle.py` — the `Stage` enum member + `_LIVE_TRANSITIONS`.
- `algua/registry/db.py` — new migration helper, `SCHEMA_VERSION`, one schema comment.
- `algua/registry/transitions.py`, `algua/registry/promotion.py`, `algua/research/gates.py`, `algua/cli/research_cmd.py` — `Stage.SHORTLISTED` → `Stage.CANDIDATE` + stage-value prose.
- `tests/` — 10 files reference the old name; repoint atomically with the enum.
- `tests/test_db_migration_candidate.py` — NEW migration test.
- Docs/skills/scripts: `CLAUDE.md`, `AGENTS.md`, `docs/agent/research-lifecycle.md`, `.claude/skills/*`, `.codex/skills/*`, `.codex/scripts/run-research-loop.sh`, `docs/algua-*.html`.

---

## Task 1: DB migration (TDD) + schema bump + gate comment

**Files:**
- Create: `tests/test_db_migration_candidate.py`
- Modify: `algua/registry/db.py`

- [ ] **Step 1: Write the failing migration test**

```python
# tests/test_db_migration_candidate.py
import sqlite3

from algua.registry.db import SCHEMA_VERSION, migrate


def _v19_with_rows(conn: sqlite3.Connection) -> None:
    """Minimal pre-rename shape: strategies + stage_transitions holding 'shortlisted'."""
    conn.executescript(
        """
        CREATE TABLE strategies (id INTEGER PRIMARY KEY, name TEXT NOT NULL, stage TEXT NOT NULL);
        CREATE TABLE stage_transitions (
            id INTEGER PRIMARY KEY, strategy_id INTEGER NOT NULL,
            from_stage TEXT, to_stage TEXT NOT NULL
        );
        INSERT INTO strategies(name, stage) VALUES ('s1', 'shortlisted'), ('s2', 'paper');
        INSERT INTO stage_transitions(strategy_id, from_stage, to_stage)
            VALUES (1, 'backtested', 'shortlisted'), (1, 'shortlisted', 'paper');
        """
    )
    conn.commit()


def test_migration_rewrites_shortlisted_rows():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    _v19_with_rows(conn)
    migrate(conn)
    stages = {r["name"]: r["stage"] for r in conn.execute("SELECT name, stage FROM strategies")}
    assert stages == {"s1": "candidate", "s2": "paper"}
    froms = [r["from_stage"] for r in conn.execute("SELECT from_stage FROM stage_transitions")]
    tos = [r["to_stage"] for r in conn.execute("SELECT to_stage FROM stage_transitions")]
    assert "shortlisted" not in froms and "shortlisted" not in tos
    assert "candidate" in froms and "candidate" in tos
    assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION


def test_migration_fresh_empty_db_is_clean():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    migrate(conn)  # no tables yet — must not raise "no such table"
    assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION


def test_migration_strategies_only_db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        "CREATE TABLE strategies (id INTEGER PRIMARY KEY, name TEXT NOT NULL, stage TEXT NOT NULL);"
        "INSERT INTO strategies(name, stage) VALUES ('s1', 'shortlisted');"
    )
    conn.commit()
    migrate(conn)  # stage_transitions absent — per-table guard must skip it, not raise
    assert conn.execute("SELECT stage FROM strategies WHERE name='s1'").fetchone()[0] == "candidate"


def test_migration_runs_even_when_already_stamped():
    """No user_version gating: a DB stamped at the new version but still holding old rows is fixed."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    _v19_with_rows(conn)
    conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
    conn.commit()
    migrate(conn)
    assert conn.execute("SELECT stage FROM strategies WHERE name='s1'").fetchone()[0] == "candidate"
```

- [ ] **Step 2: Run it — expect FAIL** (`SCHEMA_VERSION` still 19, rows stay `shortlisted`)

Run: `uv run pytest tests/test_db_migration_candidate.py -q`
Expected: FAIL (assertions on `candidate` fail; SCHEMA_VERSION mismatch).

- [ ] **Step 3: Add the migration helper + bump version + fix the comment**

In `algua/registry/db.py`:
- Change `SCHEMA_VERSION = 19` → `SCHEMA_VERSION = 20`.
- In the `gate_evaluations` schema comment, change `BACKTESTED->SHORTLISTED` → `BACKTESTED->CANDIDATE` (KEEP the words "shortlist gate" in the same comment — that names the mechanism).
- In `migrate()`, add `_migrate_shortlisted_to_candidate(conn)` as the FIRST call (before `_rekey_search_trials_to_name(conn)` and before `conn.executescript(_SCHEMA)`).
- Add the helper next to `_rekey_search_trials_to_name`:

```python
def _migrate_shortlisted_to_candidate(conn: sqlite3.Connection) -> None:
    """Rewrite the renamed lifecycle stage value `shortlisted` -> `candidate` (#120) in the typed
    stage columns. Runs BEFORE the `CREATE TABLE IF NOT EXISTS` bootstrap, so each table is guarded
    independently — a fresh DB has neither table yet. Idempotent: the `WHERE` matches nothing on a
    second run, and it does NOT gate on `user_version`, so a DB already stamped at the new version
    but still holding `shortlisted` rows is still corrected.

    Only the typed `stage` / `from_stage` / `to_stage` columns are rewritten — the free-text audit
    trail (`audit_log`, `stage_transitions.reason`) and `gate_evaluations.decision_json` are
    immutable history and intentionally left as written."""
    def _has(table: str) -> bool:
        return (
            conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
            ).fetchone()
            is not None
        )

    if _has("strategies"):
        conn.execute("UPDATE strategies SET stage='candidate' WHERE stage='shortlisted'")
    if _has("stage_transitions"):
        conn.execute(
            "UPDATE stage_transitions SET from_stage='candidate' WHERE from_stage='shortlisted'"
        )
        conn.execute(
            "UPDATE stage_transitions SET to_stage='candidate' WHERE to_stage='shortlisted'"
        )
```

- [ ] **Step 4: Run it — expect PASS**

Run: `uv run pytest tests/test_db_migration_candidate.py -q`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/test_db_migration_candidate.py algua/registry/db.py
git commit -m "feat(registry): migrate stage shortlisted->candidate; schema v20 (#120)"
```

---

## Task 2: Rename the enum + all code + all tests (atomic, suite stays green)

**Files (modify):** `algua/contracts/lifecycle.py`, `algua/registry/transitions.py`, `algua/registry/promotion.py`, `algua/research/gates.py`, `algua/cli/research_cmd.py`, and the 10 test files listed below.

**Rule:** Replace the *stage value* only. `Stage.SHORTLISTED` → `Stage.CANDIDATE`; the string `"shortlisted"` (when it denotes the stage) → `"candidate"`. **PRESERVE** the mechanism phrases "shortlist gate", "shortlist transition", "shortlist token", and the `gate_evaluations` table/`test_shortlist_gate.py` filename — they name the gate mechanism, not the stage.

- [ ] **Step 1: Edit `algua/contracts/lifecycle.py`**
  - `SHORTLISTED = "shortlisted"` → `CANDIDATE = "candidate"`.
  - `_LIVE_TRANSITIONS`: `Stage.BACKTESTED: {Stage.SHORTLISTED, Stage.IDEA}` → `{Stage.CANDIDATE, Stage.IDEA}`; key `Stage.SHORTLISTED:` → `Stage.CANDIDATE:`; `Stage.PAPER: {Stage.LIVE, Stage.SHORTLISTED}` → `{Stage.LIVE, Stage.CANDIDATE}`.

- [ ] **Step 2: Edit the four code files** — replace every `Stage.SHORTLISTED` with `Stage.CANDIDATE`, and in docstrings/comments/error-messages replace the stage word "shortlisted" with "candidate" while keeping mechanism phrases:
  - `algua/registry/transitions.py:36,37,90` (the `target == Stage.SHORTLISTED` branch; the error string "transition to shortlisted requires..." → "transition to candidate requires..."; the "Wall D: an agent reaches shortlisted ONLY..." comment → "...reaches candidate ONLY...").
  - `algua/registry/promotion.py:78,91,96,98,151,177` (`Stage.SHORTLISTED` → `Stage.CANDIDATE`; prose "BACKTESTED -> SHORTLISTED" → "BACKTESTED -> CANDIDATE"; "PAPER -> SHORTLISTED" → "PAPER -> CANDIDATE"; "the agent's only path to shortlisted" → "...to candidate". KEEP "the shortlist gate" on line 152.).
  - `algua/research/gates.py:23` docstring "backtested -> shortlisted" → "backtested -> candidate".
  - `algua/cli/research_cmd.py:63,72` docstrings "backtested->shortlisted" → "backtested->candidate"; KEEP "BACKTESTED->SHORTLISTED" only if it denotes the token mechanism — change it to "BACKTESTED->CANDIDATE" (it denotes the stage edge; the *token* is the "shortlist gate").

- [ ] **Step 3: Repoint the 10 test files** — `Stage.SHORTLISTED` → `Stage.CANDIDATE`; assertion/seed literals `"shortlisted"` → `"candidate"` EXCEPT the mechanism comments ("agent shortlist gate"). Files: `tests/test_cli_live.py`, `tests/test_cli_research.py`, `tests/test_cli_registry.py`, `tests/test_e2e_lifecycle.py`, `tests/test_promotion.py`, `tests/test_lifecycle.py`, `tests/test_registry_store.py`, `tests/test_cli_paper.py`, `tests/test_shortlist_gate.py`, `tests/test_registry_approvals.py`. Keep the filename `test_shortlist_gate.py` (mechanism).

- [ ] **Step 4: Verify no stray stage references remain**

Run: `grep -rn "SHORTLISTED\|shortlisted" algua/ tests/ | grep -v ".pyc"`
Expected: only mechanism phrases remain ("shortlist gate", "shortlist transition/token", "agent shortlist gate", `test_shortlist_gate`, and `db.py` "shortlist gate" comment). NO `Stage.SHORTLISTED`, no `= "shortlisted"`, no stage-edge `->shortlisted`.

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`
Expected: all pass (same count as baseline; migration test from Task 1 still green).

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor(lifecycle): rename stage shortlisted->candidate across code+tests (#120)"
```

---

## Task 3: Docs, skills, scripts, generated HTML

**Files (modify):** `CLAUDE.md`, `AGENTS.md`, `docs/agent/research-lifecycle.md`, `.claude/skills/operating-algua/SKILL.md`, `.claude/skills/run-the-research-loop/SKILL.md`, `.codex/skills/operating-algua/SKILL.md`, `.codex/skills/run-the-research-loop/SKILL.md`, `.codex/scripts/run-research-loop.sh`, `docs/algua-architecture.html`, `docs/algua-lifecycle.html`.

- [ ] **Step 1: Replace the stage word in each.** In the lifecycle chain `idea → backtested → shortlisted → paper → live → retired` and prose like "take a strategy to a **shortlist**" / "the `backtested → shortlisted` edge", replace the stage `shortlisted` → `candidate` (and "shortlist" the *noun for the stage* → "candidate"). PRESERVE "shortlist gate" where it names the gate mechanism. Leave dated specs/plans under `docs/superpowers/` untouched (historical).

- [ ] **Step 2: Grep the doc/skill surface**

Run: `grep -rn "shortlisted\|shortlist" CLAUDE.md AGENTS.md docs/agent .claude/skills .codex/skills .codex/scripts docs/algua-architecture.html docs/algua-lifecycle.html`
Expected: only "shortlist gate" mechanism references remain (if any).

- [ ] **Step 3: Note the rollout step in the PR body** — operators with a populated kb run `uv run algua strategy doc --all` after upgrading so the vault frontmatter `stage` resyncs from the migrated registry. (No code; documentation only — the migration handles the DB, the resync handles the vault projection.)

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "docs(120): rename shortlisted->candidate in docs, skills, scripts, generated HTML"
```

---

## Task 4: Final gate

- [ ] **Step 1: Run the full quality gate**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: all green.

- [ ] **Step 2: Final stray-reference sweep**

Run: `grep -rn "Stage.SHORTLISTED\|= \"shortlisted\"\|->shortlisted\|-> shortlisted\|→ shortlisted" algua/ tests/`
Expected: no matches.

---

## Self-review notes
- **Spec coverage:** enum (T2), migration+schema+comment (T1), code refs (T2), tests incl. migration cases (T1+T2), docs/skills/scripts/HTML + kb-resync note (T3), naming-boundary preservation (T2/T3 rules). ✓
- **Naming boundary:** every task explicitly preserves "shortlist gate/transition/token" + `test_shortlist_gate.py`. ✓
- **Atomicity:** enum + code + tests in one commit (T2) so the suite never goes red. ✓
