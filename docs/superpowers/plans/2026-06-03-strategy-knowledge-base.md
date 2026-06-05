# Strategy Knowledge Base Implementation Plan

> **Historical record — completed.** This plan was executed and merged (PR #105). **The vault
> root has since moved from `docs/strategies/` to `kb/strategies/` (PR #111, 2026-06-05);**
> every `docs/strategies/...` path below is stale. Do not run the commands here verbatim —
> they're kept for provenance, not execution.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A living, markdown-per-strategy knowledge base — authored prose plus synced fact blocks generated from the registry + MLflow — that renders as an emergent graph in Obsidian and feeds the agent's next-idea reasoning.

**Architecture:** A new pure-ish module `algua/knowledge/` owns the doc schema (frontmatter + marker-delimited synced blocks), scaffolding templates, an MLflow metrics reader, and a projection that writes synced blocks + generates `_index.md` / `_families.md`. The CLI gains one new command (`strategy doc`), extends `strategy new` to scaffold docs, and extends `doctor` with a KB drift check. The registry schema is **unchanged** — lineage lives in doc frontmatter. Import-linter keeps `backtest`/`live` off the new layer.

**Tech Stack:** Python 3.12, Typer CLI, PyYAML (already present) for frontmatter, MLflow 3.x file backend (already present), pytest, ruff, mypy, import-linter. Driven via `uv run`.

**Source-of-truth recap (from the spec):** stage ← registry; metrics/stamps ← MLflow; hypothesis/derivation/verdict/family-thesis/lineage ← the doc (authored). The synced blocks render the first two into the doc; everything else is authored prose never touched by sync.

**Refinement vs the spec mock:** the synced `## Results` block renders the **latest MLflow run** (kind, metrics, snapshot, seed, run id). The gate pass/fail is **not** synced (it isn't persisted in a syncable place, and persisting it would mean touching `research/`); it stays in the authored `## Verdict & next` prose, and the registry `stage` already reflects shortlist-or-not. This keeps the slice purely additive.

---

## File structure

**New module `algua/knowledge/`:**
- `__init__.py` — empty package marker.
- `frontmatter.py` — pure: `parse_doc`, `render_doc`, `replace_block`. No I/O.
- `templates.py` — pure: `scaffold_strategy_doc`, `scaffold_family_doc`. No I/O beyond reading the clock.
- `metrics.py` — `latest_run_metrics(strategy, *, tracking_uri)` — reads MLflow.
- `sync.py` — `sync_strategy_doc`, `sync_family_doc`, `render_results_block`, `generate_indexes`, `sync_all`, `kb_check`, plus path helpers. Reads registry + filesystem + `metrics.py`.

**Modified:**
- `algua/config/settings.py` — add `knowledge_dir`.
- `algua/cli/strategy_cmd.py` — extend `new`; add `doc`.
- `algua/cli/app.py` — extend `doctor` with a KB check.
- `pyproject.toml` — two import-linter contracts.
- `.claude/skills/run-the-research-loop/SKILL.md` and `.codex/skills/run-the-research-loop/SKILL.md` (identical files) — weave KB into the loop.

**New tests:** `tests/test_knowledge_frontmatter.py`, `tests/test_knowledge_templates.py`, `tests/test_knowledge_metrics.py`, `tests/test_knowledge_sync.py`; extend `tests/test_cli_strategy.py` and `tests/test_cli_core.py`.

**Quality gate (run before each commit):** `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`

---

## Task 1: Add `knowledge_dir` setting

**Files:**
- Modify: `algua/config/settings.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_config.py`:

```python
def test_knowledge_dir_default_and_override(monkeypatch):
    from algua.config.settings import Settings

    assert Settings().knowledge_dir == Path("docs/strategies")
    monkeypatch.setenv("ALGUA_KNOWLEDGE_DIR", "/tmp/vault")
    assert Settings().knowledge_dir == Path("/tmp/vault")
```

If `from pathlib import Path` is not already imported at the top of `tests/test_config.py`, add it.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py::test_knowledge_dir_default_and_override -v`
Expected: FAIL — `AttributeError: 'Settings' object has no attribute 'knowledge_dir'`.

- [ ] **Step 3: Add the field**

In `algua/config/settings.py`, add this line alongside the other path fields (next to `data_dir`):

```python
    knowledge_dir: Path = Path("docs/strategies")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_config.py::test_knowledge_dir_default_and_override -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add algua/config/settings.py tests/test_config.py
git commit -m "feat(knowledge): add knowledge_dir setting"
```

---

## Task 2: Frontmatter parse / render / marker-block replace (pure)

**Files:**
- Create: `algua/knowledge/__init__.py`
- Create: `algua/knowledge/frontmatter.py`
- Test: `tests/test_knowledge_frontmatter.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_knowledge_frontmatter.py`:

```python
from algua.knowledge.frontmatter import parse_doc, render_doc, replace_block


def test_parse_round_trips_frontmatter_and_body():
    text = "---\nname: x\nstage: idea\n---\n## Hypothesis\n\nbody here\n"
    fm, body = parse_doc(text)
    assert fm == {"name": "x", "stage": "idea"}
    assert body == "## Hypothesis\n\nbody here\n"


def test_parse_no_frontmatter_returns_empty_dict():
    fm, body = parse_doc("just text\n")
    assert fm == {}
    assert body == "just text\n"


def test_render_then_parse_is_stable():
    fm = {"name": "x", "family": "[[fam]]", "stage": "backtested"}
    body = "## Hypothesis\n\nclaim\n"
    text = render_doc(fm, body)
    again_fm, again_body = parse_doc(text)
    assert again_fm == fm
    assert again_body == body


def test_replace_block_replaces_between_markers():
    text = "before\n<!-- ALGUA:RESULTS -->\nold\n<!-- /ALGUA:RESULTS -->\nafter\n"
    out = replace_block(text, "RESULTS", "new content")
    assert "new content" in out
    assert "old" not in out
    assert out.startswith("before\n")
    assert out.rstrip().endswith("after")


def test_replace_block_inserts_when_markers_absent():
    out = replace_block("body only\n", "RESULTS", "fresh")
    assert "<!-- ALGUA:RESULTS -->\nfresh\n<!-- /ALGUA:RESULTS -->" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_knowledge_frontmatter.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'algua.knowledge'`.

- [ ] **Step 3: Write the implementation**

Create `algua/knowledge/__init__.py` (empty file).

Create `algua/knowledge/frontmatter.py`:

```python
from __future__ import annotations

from typing import Any

import yaml

_DELIM = "---"


def parse_doc(text: str) -> tuple[dict[str, Any], str]:
    """Split a markdown doc into (frontmatter dict, body). Empty dict if no frontmatter."""
    if not text.startswith(_DELIM):
        return {}, text
    parts = text.split(_DELIM, 2)
    if len(parts) < 3:
        return {}, text
    fm = yaml.safe_load(parts[1]) or {}
    body = parts[2]
    if body.startswith("\n"):
        body = body[1:]
    return fm, body


def render_doc(frontmatter: dict[str, Any], body: str) -> str:
    """Render frontmatter + body back into a markdown doc."""
    fm_text = yaml.safe_dump(frontmatter, sort_keys=False).strip()
    return f"{_DELIM}\n{fm_text}\n{_DELIM}\n{body}"


def replace_block(text: str, marker: str, content: str) -> str:
    """Replace the bytes between <!-- ALGUA:{marker} --> and <!-- /ALGUA:{marker} -->.

    If the markers are absent, append a fresh block at the end. Prose is never touched.
    """
    start = f"<!-- ALGUA:{marker} -->"
    end = f"<!-- /ALGUA:{marker} -->"
    block = f"{start}\n{content}\n{end}"
    i = text.find(start)
    j = text.find(end)
    if i != -1 and j != -1 and j > i:
        return text[:i] + block + text[j + len(end):]
    sep = "" if text.endswith("\n") or text == "" else "\n"
    return f"{text}{sep}{block}\n"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_knowledge_frontmatter.py -v`
Expected: PASS (all 5).

- [ ] **Step 5: Commit**

```bash
git add algua/knowledge/__init__.py algua/knowledge/frontmatter.py tests/test_knowledge_frontmatter.py
git commit -m "feat(knowledge): frontmatter + marker-block helpers"
```

---

## Task 3: Scaffold templates (pure)

**Files:**
- Create: `algua/knowledge/templates.py`
- Test: `tests/test_knowledge_templates.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_knowledge_templates.py`:

```python
from algua.knowledge.frontmatter import parse_doc
from algua.knowledge.templates import scaffold_family_doc, scaffold_strategy_doc


def test_scaffold_strategy_doc_has_frontmatter_and_sections():
    text = scaffold_strategy_doc("alpha_v1", family="momentum", derived_from="alpha_v0",
                                 created="2026-06-03")
    fm, body = parse_doc(text)
    assert fm["name"] == "alpha_v1"
    assert fm["stage"] == "idea"
    assert fm["hypothesis_status"] == "untested"
    assert fm["family"] == "[[momentum]]"
    assert fm["derived_from"] == "[[alpha_v0]]"
    assert fm["created"] == "2026-06-03"
    assert "## Hypothesis" in body
    assert "## Derivation" in body
    assert "## Verdict & next" in body
    assert "<!-- ALGUA:RESULTS -->" in body and "<!-- /ALGUA:RESULTS -->" in body


def test_scaffold_strategy_doc_omits_absent_lineage():
    fm, _ = parse_doc(scaffold_strategy_doc("root", created="2026-06-03"))
    assert "family" not in fm
    assert "derived_from" not in fm


def test_scaffold_family_doc_has_thesis_and_members_block():
    fm, body = parse_doc(scaffold_family_doc("momentum", created="2026-06-03"))
    assert fm["type"] == "family"
    assert fm["name"] == "momentum"
    assert fm["status"] == "exploring"
    assert "## Thesis" in body
    assert "## Open questions / next" in body
    assert "<!-- ALGUA:MEMBERS -->" in body and "<!-- /ALGUA:MEMBERS -->" in body
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_knowledge_templates.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'algua.knowledge.templates'`.

- [ ] **Step 3: Write the implementation**

Create `algua/knowledge/templates.py`:

```python
from __future__ import annotations

from datetime import UTC, datetime

from algua.knowledge.frontmatter import render_doc


def _today() -> str:
    return datetime.now(UTC).date().isoformat()


def scaffold_strategy_doc(
    name: str,
    *,
    family: str | None = None,
    derived_from: str | None = None,
    created: str | None = None,
) -> str:
    """Initial strategy doc: authored sections empty, synced RESULTS block empty."""
    fm: dict[str, object] = {
        "name": name,
        "stage": "idea",
        "hypothesis_status": "untested",
        "created": created or _today(),
    }
    if family:
        fm["family"] = f"[[{family}]]"
    if derived_from:
        fm["derived_from"] = f"[[{derived_from}]]"
    body = (
        "## Hypothesis\n\n_What edge is claimed, and why._\n\n"
        "## Derivation\n\n_Forked from what; what changed._\n\n"
        "## Results\n\n"
        "<!-- ALGUA:RESULTS -->\n_No tracked runs yet._\n<!-- /ALGUA:RESULTS -->\n\n"
        "## Verdict & next\n\n"
        "_What was learned; the next idea as a [[link]]._\n"
    )
    return render_doc(fm, body)


def scaffold_family_doc(name: str, *, created: str | None = None) -> str:
    """Initial thesis-family hub doc."""
    fm: dict[str, object] = {
        "type": "family",
        "name": name,
        "status": "exploring",
        "created": created or _today(),
    }
    body = (
        "## Thesis\n\n_The hypothesis for the whole family._\n\n"
        "## Members\n\n"
        "<!-- ALGUA:MEMBERS -->\n_No members yet._\n<!-- /ALGUA:MEMBERS -->\n\n"
        "## State of exploration\n\n_Members + one-line outcomes._\n\n"
        "## Open questions / next\n\n"
        "_Which axes are exhausted; what's left; when to park._\n"
    )
    return render_doc(fm, body)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_knowledge_templates.py -v`
Expected: PASS (all 3).

- [ ] **Step 5: Commit**

```bash
git add algua/knowledge/templates.py tests/test_knowledge_templates.py
git commit -m "feat(knowledge): strategy + family scaffold templates"
```

---

## Task 4: MLflow latest-run metrics reader

**Files:**
- Create: `algua/knowledge/metrics.py`
- Test: `tests/test_knowledge_metrics.py`

The reader queries the experiment named after the strategy and returns the most recent run as a flat dict, or `None`. Tests log a real run to a tmp file-backend tracking URI (no server needed).

- [ ] **Step 1: Write the failing test**

Create `tests/test_knowledge_metrics.py`:

```python
from algua.knowledge.metrics import latest_run_metrics


def _log_run(uri, strategy, *, kind, metrics, params):
    import mlflow

    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(strategy)
    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.set_tags({"kind": kind})


def test_latest_run_metrics_none_when_no_experiment(tmp_path):
    uri = str(tmp_path / "mlruns")
    assert latest_run_metrics("ghost", tracking_uri=uri) is None


def test_latest_run_metrics_reads_latest(tmp_path):
    uri = str(tmp_path / "mlruns")
    _log_run(uri, "alpha", kind="backtest",
             metrics={"sharpe": 0.4}, params={"snapshot_id": "ds_1", "seed": "7"})
    _log_run(uri, "alpha", kind="walk_forward",
             metrics={"holdout.sharpe": 0.28}, params={"snapshot_id": "ds_2", "seed": "7"})

    out = latest_run_metrics("alpha", tracking_uri=uri)
    assert out is not None
    assert out["kind"] == "walk_forward"
    assert out["snapshot_id"] == "ds_2"
    assert out["seed"] == "7"
    assert out["metrics"]["holdout.sharpe"] == 0.28
    assert isinstance(out["run_id"], str) and out["run_id"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_knowledge_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'algua.knowledge.metrics'`.

- [ ] **Step 3: Write the implementation**

Create `algua/knowledge/metrics.py`:

```python
from __future__ import annotations

from typing import Any


def latest_run_metrics(strategy: str, *, tracking_uri: str) -> dict[str, Any] | None:
    """Most recent MLflow run for `strategy` as a flat dict, or None if there are none.

    Reads the experiment named `strategy`. Works against a file backend — no server.
    """
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    exp = mlflow.get_experiment_by_name(strategy)
    if exp is None:
        return None
    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1,
        output_format="list",
    )
    if not runs:
        return None
    run = runs[0]
    return {
        "run_id": run.info.run_id,
        "kind": run.data.tags.get("kind", "unknown"),
        "snapshot_id": run.data.params.get("snapshot_id"),
        "seed": run.data.params.get("seed"),
        "metrics": dict(run.data.metrics),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_knowledge_metrics.py -v`
Expected: PASS (both).

- [ ] **Step 5: Commit**

```bash
git add algua/knowledge/metrics.py tests/test_knowledge_metrics.py
git commit -m "feat(knowledge): MLflow latest-run metrics reader"
```

---

## Task 5: Sync — write synced blocks + generate indexes

**Files:**
- Create: `algua/knowledge/sync.py`
- Test: `tests/test_knowledge_sync.py`

This is the projection. It reads the registry (for stage) and `metrics.py` (for the Results block), rewrites only marker blocks, and regenerates `_index.md` / `_families.md`. `kb_check` reports drift for `doctor`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_knowledge_sync.py`:

```python
from contextlib import closing
from pathlib import Path

from algua.config.settings import Settings
from algua.knowledge.frontmatter import parse_doc
from algua.knowledge.sync import (
    generate_indexes,
    kb_check,
    render_results_block,
    sync_all,
    sync_strategy_doc,
)
from algua.knowledge.templates import scaffold_family_doc, scaffold_strategy_doc
from algua.registry import store
from algua.registry.db import connect, migrate


def _settings(tmp_path) -> Settings:
    return Settings(
        db_path=tmp_path / "r.db",
        knowledge_dir=tmp_path / "vault",
        mlflow_tracking_uri=str(tmp_path / "mlruns"),
    )


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_render_results_block_handles_no_metrics():
    assert "No tracked runs" in render_results_block(None)


def test_render_results_block_renders_metrics():
    out = render_results_block(
        {"run_id": "abcd1234ef", "kind": "walk_forward",
         "snapshot_id": "ds_2", "seed": "7", "metrics": {"holdout.sharpe": 0.28}}
    )
    assert "walk_forward" in out
    assert "holdout.sharpe" in out
    assert "ds_2" in out


def test_sync_strategy_doc_updates_stage_and_preserves_prose(tmp_path):
    s = _settings(tmp_path)
    _write(s.knowledge_dir / "alpha.md",
           scaffold_strategy_doc("alpha", created="2026-06-03"))
    with closing(connect(s.db_path)) as conn:
        migrate(conn)
        store.add_strategy(conn, "alpha")
        store.transition(conn, "alpha", "backtested", "agent", "test")
        assert sync_strategy_doc(s, conn, "alpha") is True
    fm, body = parse_doc((s.knowledge_dir / "alpha.md").read_text())
    assert fm["stage"] == "backtested"          # synced from registry
    assert "## Hypothesis" in body              # prose preserved
    assert "No tracked runs yet" in body        # synced RESULTS block present


def test_sync_strategy_doc_false_when_doc_missing(tmp_path):
    s = _settings(tmp_path)
    s.knowledge_dir.mkdir(parents=True)
    with closing(connect(s.db_path)) as conn:
        migrate(conn)
        assert sync_strategy_doc(s, conn, "ghost") is False


def test_generate_indexes_lists_strategies_and_families(tmp_path):
    s = _settings(tmp_path)
    _write(s.knowledge_dir / "alpha.md",
           scaffold_strategy_doc("alpha", family="momentum", created="2026-06-03"))
    _write(s.knowledge_dir / "families" / "momentum.md",
           scaffold_family_doc("momentum", created="2026-06-03"))
    generate_indexes(s)
    index = (s.knowledge_dir / "_index.md").read_text()
    families = (s.knowledge_dir / "_families.md").read_text()
    assert "[[alpha]]" in index
    assert "[[momentum]]" in families


def test_kb_check_flags_missing_doc(tmp_path):
    s = _settings(tmp_path)
    s.knowledge_dir.mkdir(parents=True)
    with closing(connect(s.db_path)) as conn:
        migrate(conn)
        store.add_strategy(conn, "alpha")
    ok, detail = kb_check(s)
    assert ok is False
    assert "alpha" in detail


def test_sync_all_returns_summary(tmp_path):
    s = _settings(tmp_path)
    _write(s.knowledge_dir / "alpha.md",
           scaffold_strategy_doc("alpha", family="momentum", created="2026-06-03"))
    _write(s.knowledge_dir / "families" / "momentum.md",
           scaffold_family_doc("momentum", created="2026-06-03"))
    with closing(connect(s.db_path)) as conn:
        migrate(conn)
        store.add_strategy(conn, "alpha")
        summary = sync_all(s, conn)
    assert "alpha" in summary["strategies"]
    assert "momentum" in summary["families"]
    members = (s.knowledge_dir / "families" / "momentum.md").read_text()
    assert "1 members" in members or "members: idea 1" in members
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_knowledge_sync.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'algua.knowledge.sync'`.

- [ ] **Step 3: Write the implementation**

Create `algua/knowledge/sync.py`:

```python
from __future__ import annotations

from contextlib import closing
from pathlib import Path
from typing import Any

from algua.config.settings import Settings
from algua.knowledge.frontmatter import parse_doc, render_doc, replace_block
from algua.knowledge.metrics import latest_run_metrics
from algua.registry import store
from algua.registry.db import connect, migrate


def strategy_doc_path(settings: Settings, name: str) -> Path:
    return settings.knowledge_dir / f"{name}.md"


def family_doc_path(settings: Settings, name: str) -> Path:
    return settings.knowledge_dir / "families" / f"{name}.md"


def render_results_block(metrics: dict[str, Any] | None) -> str:
    """Markdown content (no markers) for the synced RESULTS block."""
    if not metrics:
        return "_No tracked runs yet._"
    head = (
        f"Latest run: `{metrics['kind']}` · snapshot `{metrics.get('snapshot_id')}` · "
        f"seed `{metrics.get('seed')}` · run `{metrics['run_id'][:8]}`"
    )
    lines = [head, ""]
    if metrics["metrics"]:
        lines += ["| metric | value |", "|---|---|"]
        for key in sorted(metrics["metrics"]):
            lines.append(f"| {key} | {metrics['metrics'][key]:.4g} |")
    return "\n".join(lines)


def sync_strategy_doc(settings: Settings, conn, name: str) -> bool:
    """Rewrite the synced parts of one strategy doc. Returns False if the doc is absent."""
    path = strategy_doc_path(settings, name)
    if not path.exists():
        return False
    fm, body = parse_doc(path.read_text())
    try:
        rec = store.get_strategy(conn, name)
        fm["stage"] = rec.stage.value
    except store.StrategyNotFound:
        pass  # doc scaffolded before registry add — leave authored stage untouched
    metrics = latest_run_metrics(name, tracking_uri=settings.mlflow_tracking_uri)
    if metrics:
        fm["mlflow_run"] = metrics["run_id"][:8]
    body = replace_block(body, "RESULTS", render_results_block(metrics))
    path.write_text(render_doc(fm, body))
    return True


def sync_family_doc(settings: Settings, name: str) -> bool:
    """Rewrite the synced MEMBERS roster of one family doc. False if the doc is absent."""
    path = family_doc_path(settings, name)
    if not path.exists():
        return False
    counts: dict[str, int] = {}
    for doc in settings.knowledge_dir.glob("*.md"):
        if doc.name.startswith("_"):
            continue
        fm, _ = parse_doc(doc.read_text())
        if fm.get("family") == f"[[{name}]]":
            stage = str(fm.get("stage", "?"))
            counts[stage] = counts.get(stage, 0) + 1
    total = sum(counts.values())
    detail = f"{total} members"
    if counts:
        detail += ": " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
    fm, body = parse_doc(path.read_text())
    body = replace_block(body, "MEMBERS", detail)
    path.write_text(render_doc(fm, body))
    return True


def generate_indexes(settings: Settings) -> None:
    """(Re)generate _index.md (strategies) and _families.md from vault frontmatter."""
    vault = settings.knowledge_dir
    vault.mkdir(parents=True, exist_ok=True)
    strat_lines: list[str] = []
    for doc in sorted(vault.glob("*.md")):
        if doc.name.startswith("_"):
            continue
        fm, _ = parse_doc(doc.read_text())
        if fm.get("type") == "family":
            continue
        name = fm.get("name", doc.stem)
        strat_lines.append(
            f"- [[{name}]] — {fm.get('family', '—')} · "
            f"{fm.get('stage', '?')} · {fm.get('hypothesis_status', '?')}"
        )
    (vault / "_index.md").write_text("# Strategies\n\n" + "\n".join(strat_lines) + "\n")

    fam_dir = vault / "families"
    fam_lines: list[str] = []
    if fam_dir.exists():
        for doc in sorted(fam_dir.glob("*.md")):
            fm, _ = parse_doc(doc.read_text())
            fam_lines.append(f"- [[{fm.get('name', doc.stem)}]] — {fm.get('status', '?')}")
    (vault / "_families.md").write_text("# Thesis families\n\n" + "\n".join(fam_lines) + "\n")


def sync_all(settings: Settings, conn) -> dict[str, list[str]]:
    """Sync every registry strategy that has a doc, every family doc, then the indexes."""
    synced: list[str] = []
    for rec in store.list_strategies(conn):
        if sync_strategy_doc(settings, conn, rec.name):
            synced.append(rec.name)
    families: list[str] = []
    fam_dir = settings.knowledge_dir / "families"
    if fam_dir.exists():
        for doc in sorted(fam_dir.glob("*.md")):
            fm, _ = parse_doc(doc.read_text())
            fam_name = str(fm.get("name", doc.stem))
            if sync_family_doc(settings, fam_name):
                families.append(fam_name)
    generate_indexes(settings)
    return {"strategies": synced, "families": families}


def kb_check(settings: Settings) -> tuple[bool, str]:
    """Flag registry strategies with no doc or a stale synced stage. For `doctor`."""
    with closing(connect(settings.db_path)) as conn:
        migrate(conn)
        records = store.list_strategies(conn)
    issues: list[str] = []
    for rec in records:
        path = strategy_doc_path(settings, rec.name)
        if not path.exists():
            issues.append(f"{rec.name}: no doc")
            continue
        fm, _ = parse_doc(path.read_text())
        if fm.get("stage") != rec.stage.value:
            issues.append(f"{rec.name}: doc stage {fm.get('stage')} != registry {rec.stage.value}")
    if issues:
        return False, "; ".join(issues)
    return True, f"{len(records)} strategies in sync"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_knowledge_sync.py -v`
Expected: PASS (all 7).

- [ ] **Step 5: Run the full quality gate**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: all pass. If mypy complains about untyped `conn` params, that is acceptable as written (`sqlite3.Connection` is fine to add if you prefer: `import sqlite3` and annotate `conn: sqlite3.Connection`). If mypy flags missing stubs for `yaml`/`mlflow`, confirm the repo already tolerates this (tracking uses `mlflow` and passes) — no new ignores should be needed.

- [ ] **Step 6: Commit**

```bash
git add algua/knowledge/sync.py tests/test_knowledge_sync.py
git commit -m "feat(knowledge): sync projection + index generation + kb_check"
```

---

## Task 6: Import-linter contracts isolating the knowledge layer

**Files:**
- Modify: `pyproject.toml` (under `[tool.importlinter]`)

- [ ] **Step 1: Add two contracts**

Append these two contracts to the end of the import-linter contracts list in `pyproject.toml`:

```toml
[[tool.importlinter.contracts]]
name = "knowledge layer stays off cli, backtest, live, execution"
type = "forbidden"
source_modules = ["algua.knowledge"]
forbidden_modules = ["algua.cli", "algua.backtest", "algua.live", "algua.execution"]

[[tool.importlinter.contracts]]
name = "pure + execution lanes stay off the knowledge layer"
type = "forbidden"
source_modules = ["algua.backtest", "algua.live", "algua.features", "algua.contracts"]
forbidden_modules = ["algua.knowledge"]
```

- [ ] **Step 2: Run import-linter to verify both pass**

Run: `uv run lint-imports`
Expected: all contracts kept (including the two new ones). If a contract is broken, the offending import is a real layering bug — fix the import, don't relax the contract.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "chore(knowledge): import-linter contracts for the knowledge layer"
```

---

## Task 7: Extend `strategy new` to scaffold docs

**Files:**
- Modify: `algua/cli/strategy_cmd.py`
- Test: `tests/test_cli_strategy.py`

`strategy new` keeps writing the `.py` module, and now also scaffolds `docs/strategies/<name>.md` (and a family hub doc if `--family` is given and missing).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_cli_strategy.py`:

```python
def test_strategy_new_scaffolds_doc_and_family(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ALGUA_KNOWLEDGE_DIR", str(tmp_path / "vault"))
    result = runner.invoke(
        app, ["strategy", "new", "alpha", "--family", "momentum", "--derived-from", "seed"]
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert (tmp_path / "vault" / "alpha.md").exists()
    assert (tmp_path / "vault" / "families" / "momentum.md").exists()
    assert payload["doc"].endswith("alpha.md")


def test_strategy_new_rejects_unsafe_family(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ALGUA_KNOWLEDGE_DIR", str(tmp_path / "vault"))
    result = runner.invoke(app, ["strategy", "new", "alpha", "--family", "../evil"])
    assert result.exit_code == 1, result.stdout
    assert json.loads(result.stdout)["ok"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli_strategy.py::test_strategy_new_scaffolds_doc_and_family -v`
Expected: FAIL — the `--family` option does not exist yet (Typer errors / exit 2, or no doc written).

- [ ] **Step 3: Implement**

In `algua/cli/strategy_cmd.py`, add imports near the top:

```python
import re

from algua.config.settings import get_settings
from algua.knowledge.templates import scaffold_family_doc, scaffold_strategy_doc
```

Add this module-level constant below the imports:

```python
_FAMILY_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
```

Replace the `new` command with:

```python
@strategy_app.command("new")
@json_errors()
def new(
    name: str,
    family: str = typer.Option(None, "--family", help="thesis family this belongs to"),  # noqa: B008
    derived_from: str = typer.Option(None, "--derived-from", help="parent strategy name"),  # noqa: B008
) -> None:
    """Scaffold a new strategy module + its knowledge-base doc (and family hub if needed)."""
    if not name.isidentifier() or keyword.iskeyword(name):
        raise ValueError(
            f"invalid strategy name {name!r}: must be a valid, non-keyword Python identifier"
        )
    if family is not None and not _FAMILY_RE.match(family):
        raise ValueError(
            f"invalid family {family!r}: must be a lowercase slug (a-z, 0-9, hyphen)"
        )
    path = Path("algua/strategies/examples") / f"{name}.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ValueError(f"strategy already exists: {path}")
    path.write_text(_TEMPLATE.format(name=name))

    vault = get_settings().knowledge_dir
    vault.mkdir(parents=True, exist_ok=True)
    doc_path = vault / f"{name}.md"
    if not doc_path.exists():
        doc_path.write_text(
            scaffold_strategy_doc(name, family=family, derived_from=derived_from)
        )
    family_doc: str | None = None
    if family:
        (vault / "families").mkdir(parents=True, exist_ok=True)
        fam_path = vault / "families" / f"{family}.md"
        if not fam_path.exists():
            fam_path.write_text(scaffold_family_doc(family))
        family_doc = str(fam_path)

    emit({
        "ok": True, "name": name, "path": str(path),
        "doc": str(doc_path), "family_doc": family_doc,
    })
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli_strategy.py -v`
Expected: PASS — including the pre-existing `test_strategy_new_scaffolds_loadable_module` and `test_strategy_new_rejects_unsafe_names`.

Note: the existing `test_strategy_new_scaffolds_loadable_module` does not set `ALGUA_KNOWLEDGE_DIR`, so the doc is written under the default `docs/strategies/` **inside the test's `tmp_path`** (the test `chdir`s into `tmp_path`). That is harmless and isolated.

- [ ] **Step 5: Commit**

```bash
git add algua/cli/strategy_cmd.py tests/test_cli_strategy.py
git commit -m "feat(cli): strategy new scaffolds knowledge-base docs"
```

---

## Task 8: Add the `strategy doc` projection command

**Files:**
- Modify: `algua/cli/strategy_cmd.py`
- Test: `tests/test_cli_strategy.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_cli_strategy.py`:

```python
def test_strategy_doc_syncs_and_builds_index(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ALGUA_KNOWLEDGE_DIR", str(tmp_path / "vault"))
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))

    assert runner.invoke(app, ["strategy", "new", "alpha"]).exit_code == 0
    assert runner.invoke(app, ["registry", "add", "alpha"]).exit_code == 0
    assert runner.invoke(
        app, ["registry", "transition", "alpha", "--to", "backtested",
              "--actor", "agent", "--reason", "x"]
    ).exit_code == 0

    result = runner.invoke(app, ["strategy", "doc", "--all"])
    assert result.exit_code == 0, result.stdout
    assert json.loads(result.stdout)["ok"] is True
    assert "[[alpha]]" in (tmp_path / "vault" / "_index.md").read_text()
    assert "stage: backtested" in (tmp_path / "vault" / "alpha.md").read_text()


def test_strategy_doc_missing_doc_errors(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ALGUA_KNOWLEDGE_DIR", str(tmp_path / "vault"))
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))
    result = runner.invoke(app, ["strategy", "doc", "ghost"])
    assert result.exit_code == 1, result.stdout
    assert json.loads(result.stdout)["ok"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli_strategy.py::test_strategy_doc_syncs_and_builds_index -v`
Expected: FAIL — `No such command 'doc'`.

- [ ] **Step 3: Implement**

In `algua/cli/strategy_cmd.py`, add imports:

```python
from contextlib import closing

from algua.knowledge.sync import generate_indexes, sync_all, sync_strategy_doc
from algua.registry.db import connect, migrate
```

Add the command (below `new`):

```python
@strategy_app.command("doc")
@json_errors()
def doc(
    name: str = typer.Argument(None, help="strategy to sync; omit (or --all) for all"),  # noqa: B008
    all_: bool = typer.Option(False, "--all", help="sync every strategy + family doc"),
) -> None:
    """Regenerate the synced blocks of strategy/family docs + rebuild the indexes."""
    settings = get_settings()
    with closing(connect(settings.db_path)) as conn:
        migrate(conn)
        if all_ or name is None:
            summary = sync_all(settings, conn)
        else:
            if not sync_strategy_doc(settings, conn, name):
                raise ValueError(f"no strategy doc for {name!r}; run `strategy new` first")
            generate_indexes(settings)
            summary = {"strategies": [name], "families": []}
    emit({"ok": True, **summary})
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli_strategy.py -v`
Expected: PASS (all, including the two new tests).

- [ ] **Step 5: Commit**

```bash
git add algua/cli/strategy_cmd.py tests/test_cli_strategy.py
git commit -m "feat(cli): add 'strategy doc' projection command"
```

---

## Task 9: Extend `doctor` with the KB drift check

**Files:**
- Modify: `algua/cli/app.py` (the `doctor` command)
- Test: `tests/test_cli_core.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_cli_core.py`:

```python
def test_doctor_reports_knowledge_base_check(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))
    monkeypatch.setenv("ALGUA_KNOWLEDGE_DIR", str(tmp_path / "vault"))
    result = runner.invoke(app, ["doctor"])
    payload = json.loads(result.stdout)
    assert "knowledge_base" in {c["check"] for c in payload["checks"]}


def test_doctor_flags_missing_strategy_doc(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))
    monkeypatch.setenv("ALGUA_KNOWLEDGE_DIR", str(tmp_path / "vault"))
    runner.invoke(app, ["registry", "add", "alpha"])  # registry row, no doc
    result = runner.invoke(app, ["doctor"])
    payload = json.loads(result.stdout)
    kb = next(c for c in payload["checks"] if c["check"] == "knowledge_base")
    assert kb["ok"] is False
    assert "alpha" in kb["detail"]
    assert result.exit_code == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli_core.py::test_doctor_reports_knowledge_base_check -v`
Expected: FAIL — no `knowledge_base` check present.

- [ ] **Step 3: Implement**

In `algua/cli/app.py`, inside `doctor`, after the `calendar` check block and before `all_ok = ...`, add:

```python
    try:
        from algua.knowledge.sync import kb_check

        kb_ok, kb_detail = kb_check(settings)
        checks.append({"check": "knowledge_base", "ok": kb_ok, "detail": kb_detail})
    except Exception as exc:  # noqa: BLE001
        checks.append({"check": "knowledge_base", "ok": False, "detail": str(exc)})
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli_core.py -v`
Expected: PASS — including the pre-existing `test_doctor_passes_in_clean_env` (a fresh empty registry yields zero strategies, so the KB check is `ok=True`).

- [ ] **Step 5: Run the full quality gate**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add algua/cli/app.py tests/test_cli_core.py
git commit -m "feat(cli): doctor flags knowledge-base drift"
```

---

## Task 10: Weave the KB into the research-loop skill

**Files:**
- Modify: `.claude/skills/run-the-research-loop/SKILL.md`
- Modify: `.codex/skills/run-the-research-loop/SKILL.md` (byte-identical to the `.claude` copy — apply the same edits)

No tests here (skill prose). Make the same three edits to **both** files.

- [ ] **Step 1: Edit step 1 (Ideate)**

In the `## The loop` list, replace the step 1 text (the line starting `1. **Ideate.**`) so it begins by reading the KB indexes. Replace:

```
1. **Ideate.** Form one concrete, testable hypothesis (e.g. "longer-lookback cross-sectional
   momentum with a tighter top-k concentrates into stronger trends"). Pick a unique strategy name;
   skip names already in `uv run algua registry list`.
```

with:

```
1. **Ideate.** First read the knowledge base: `docs/strategies/_index.md` and
   `_families.md`. Prefer a thesis family marked `exploring`/`promising` with an open axis
   in its `## Open questions`; read that family doc and the relevant strategy docs to avoid
   re-running a refuted idea. Form one concrete, testable hypothesis on the most promising
   open axis. Pick a unique strategy name; skip names already in `uv run algua registry list`.
```

- [ ] **Step 2: Edit step 2 (Author)**

Replace the step 2 text:

```
2. **Author.** Delegate to the `author` subagent (it follows `author-a-strategy`) to write
   `algua/strategies/examples/<name>.py`. Confirm it loads: `uv run algua backtest run <name> --demo`.
```

with:

```
2. **Author.** Scaffold with `uv run algua strategy new <name> --family <slug> --derived-from
   <parent>` (creates the module *and* the KB doc + family hub). Delegate to the `author`
   subagent (it follows `author-a-strategy`) to write `algua/strategies/examples/<name>.py`,
   then fill in the doc's `## Hypothesis` and `## Derivation` prose. Confirm it loads:
   `uv run algua backtest run <name> --demo`.
```

- [ ] **Step 3: Edit step 7 (Record)**

Replace the step 7 text:

```
7. **Record.** Append the hypothesis, params, key metrics, the gate decision, and your
   shortlist/discard rationale to `run-report.md`.
```

with:

```
7. **Record.** Sync the synced fact blocks: `uv run algua strategy doc <name>`. Then write
   the doc's `## Verdict & next` (what was learned + the next idea as a `[[dangling-link]]`),
   set `hypothesis_status`, and update the family doc's `## State of exploration` and
   `status`. Finally append the hypothesis, params, key metrics, the gate decision, and your
   shortlist/discard rationale to `run-report.md`.
```

- [ ] **Step 4: Verify both files received all three edits**

Run: `diff .codex/skills/run-the-research-loop/SKILL.md .claude/skills/run-the-research-loop/SKILL.md && echo IDENTICAL`
Expected: prints `IDENTICAL` (the two copies stay in sync).

Run: `grep -c "strategy doc <name>" .claude/skills/run-the-research-loop/SKILL.md`
Expected: `1`.

- [ ] **Step 5: Commit**

```bash
git add .claude/skills/run-the-research-loop/SKILL.md .codex/skills/run-the-research-loop/SKILL.md
git commit -m "docs(skill): weave knowledge base into the research loop"
```

---

## Task 11: End-to-end smoke + final gate

**Files:** none (verification only)

- [ ] **Step 1: Drive the real CLI end-to-end**

Run (from the repo root, against the real vault `docs/strategies/`):

```bash
uv run algua strategy new kb_smoke --family kb-demo --derived-from cross_sectional_momentum
uv run algua registry add kb_smoke
uv run algua registry transition kb_smoke --to backtested --actor agent --reason "smoke"
uv run algua strategy doc --all
uv run algua doctor
```

Expected: every command exits 0 and emits JSON; `docs/strategies/kb_smoke.md` shows `stage: backtested`; `docs/strategies/_index.md` lists `[[kb_smoke]]`; `docs/strategies/families/kb-demo.md` exists; `doctor`'s `knowledge_base` check is `ok: true`.

- [ ] **Step 2: Clean up the smoke artifacts**

```bash
rm -f docs/strategies/kb_smoke.md docs/strategies/families/kb-demo.md \
      algua/strategies/examples/kb_smoke.py
uv run algua strategy doc --all   # rebuild indexes without the smoke entry
git checkout -- data/algua.db 2>/dev/null || true
```

Note: if `kb_smoke` ended up persisted in the real registry DB, remove it (the registry has no delete command — restore the DB from git if it is tracked, otherwise leave it; it is harmless paper-side state). Confirm `git status` shows no stray smoke files before finishing.

- [ ] **Step 3: Full quality gate**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: all pass.

- [ ] **Step 4: Finish the branch**

Use the `superpowers:finishing-a-development-branch` skill to decide merge/PR. The branch is `sp-strategy-knowledge-base` (already created off `main`, with the spec commit on it).

---

## Self-review notes (for the implementer)

- **Spec coverage:** doc schema (Tasks 2–3), source-of-truth/sync (Tasks 4–5), `strategy new` scaffolding (Task 7), new `strategy doc` command (Task 8), `_index`/`_families` (Task 5/8), `doctor` check (Task 9), new `algua/knowledge/` module + import-linter rule (Tasks 2–6), research-loop integration (Task 10). The deferred web dashboard / PnL / approval cockpit are intentionally absent.
- **Deliberate simplification (stated, not silent):** the synced `## Results` block renders the latest MLflow run only; gate pass/fail stays authored prose. A richer train-vs-walk-forward table is a later refinement.
- **Naming consistency:** `knowledge_dir`, `parse_doc`/`render_doc`/`replace_block`, `scaffold_strategy_doc`/`scaffold_family_doc`, `latest_run_metrics`, `sync_strategy_doc`/`sync_family_doc`/`render_results_block`/`generate_indexes`/`sync_all`/`kb_check`, marker names `RESULTS`/`MEMBERS` — used identically across all tasks.
