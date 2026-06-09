# Structured Idea-Sourcing Pipeline (#126) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a structured idea pool — sourced, deduped, provenance-stamped hypothesis records that park when their data isn't available — as the disciplined top-of-funnel, without touching the CODEOWNERS-protected promotion gate.

**Architecture:** A new `ideas` sqlite table (schema v18→v19) in the registry DB, linked to `strategies` by FK. Pure contracts (`Idea`/enums/state-machine), a pure dedup module, a pure status-classifier, a `data`-owned capability vocabulary, an `IdeaRepository` (the only idea SQL, incl. a refuted-aware live join), and a `research idea ...` CLI that orchestrates them. A standalone `source-ideas` skill drives `deep-research → dedup → research idea add`. The research loop is left unchanged (collection-only; autopilot waits for the human gate-counting PR).

**Tech Stack:** Python 3.12, Typer CLI, sqlite3, pytest. Quality gate after every commit: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`.

**Design spec:** `docs/superpowers/specs/2026-06-08-idea-sourcing-pipeline-design.md`

**Import boundaries (must stay green):** `contracts/idea.py` pure (stdlib only). `data/capabilities.py` imports `contracts` + `data.models`. `research/idea_dedup.py` + `research/ideas.py` import `contracts` only. `registry/ideas.py` imports `contracts`, `registry.metadata`, `research.idea_dedup` (registry→research is already established by `promotion.py`). `cli/idea_cmd.py` imports all. **No CODEOWNERS-protected file is modified** (`store.py`, `lifecycle.py`, `backtest/engine.py`, `gates.py`, `approvers/`, `live_gate.py`, `transitions.py`, `promotion.py`).

---

### Task 1: Idea contracts (enums, state machine, dataclass)

**Files:**
- Create: `algua/contracts/idea.py`
- Test: `tests/test_idea_contracts.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_idea_contracts.py
from algua.contracts.idea import (
    ALLOWED_IDEA_TRANSITIONS,
    DataCapability,
    Idea,
    IdeaStatus,
    SourceType,
    can_change_status,
)


def test_enum_values():
    assert IdeaStatus.OPEN == "open"
    assert IdeaStatus.NEEDS_DATA == "needs_data"
    assert {s.value for s in IdeaStatus} == {
        "open", "needs_data", "authored", "refuted", "discarded"}
    assert SourceType.PAPER == "paper"
    assert DataCapability.OHLCV == "ohlcv"
    assert DataCapability.FORM_13F == "form_13f"


def test_status_transitions_legal():
    assert can_change_status(IdeaStatus.OPEN, IdeaStatus.AUTHORED)
    assert can_change_status(IdeaStatus.OPEN, IdeaStatus.NEEDS_DATA)
    assert can_change_status(IdeaStatus.NEEDS_DATA, IdeaStatus.OPEN)
    assert can_change_status(IdeaStatus.AUTHORED, IdeaStatus.REFUTED)


def test_status_transitions_illegal():
    # terminal states go nowhere
    assert ALLOWED_IDEA_TRANSITIONS[IdeaStatus.REFUTED] == set()
    assert ALLOWED_IDEA_TRANSITIONS[IdeaStatus.DISCARDED] == set()
    # cannot resurrect a refuted idea or re-open an authored one
    assert not can_change_status(IdeaStatus.REFUTED, IdeaStatus.OPEN)
    assert not can_change_status(IdeaStatus.AUTHORED, IdeaStatus.OPEN)
    # no-op is not a legal "change"
    assert not can_change_status(IdeaStatus.OPEN, IdeaStatus.OPEN)


def test_idea_dataclass_fields():
    idea = Idea(
        id=1, title="t", hypothesis="h", family="mom", tags=["x"],
        source_type=SourceType.PAPER, source_ref="u", source_date=None, source_note=None,
        required_data=[DataCapability.OHLCV], status=IdeaStatus.OPEN, signature="h t",
        authored_strategy_id=None, duplicate_of_idea_id=None, override_reason=None,
        created_at="2026-06-08", updated_at="2026-06-08",
    )
    assert idea.required_data == [DataCapability.OHLCV]
    assert idea.status is IdeaStatus.OPEN
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_idea_contracts.py -q`
Expected: FAIL (`ModuleNotFoundError: algua.contracts.idea`)

- [ ] **Step 3: Write the implementation**

```python
# algua/contracts/idea.py
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class IdeaStatus(StrEnum):
    """Lifecycle of a sourced idea before/at the registry's `idea` stage."""

    OPEN = "open"              # testable now (needs only platform-supported data)
    NEEDS_DATA = "needs_data"  # parked: needs a capability the platform can't provide yet
    AUTHORED = "authored"      # promoted into a registered strategy
    REFUTED = "refuted"        # rejected; a dedup sentinel
    DISCARDED = "discarded"    # dropped without testing


class SourceType(StrEnum):
    """Where a sourced idea came from (provenance)."""

    PAPER = "paper"
    URL = "url"
    FORUM = "forum"
    FILING = "filing"
    THESIS = "thesis"
    MANUAL = "manual"


class DataCapability(StrEnum):
    """Controlled vocabulary of strategy-input data kinds an idea may require. Only OHLCV is
    platform-supported today (see ``algua.data.capabilities``); the rest park ideas as
    ``needs_data``. A single vocabulary stops 13f / form_13f / filings_13f fragmentation."""

    OHLCV = "ohlcv"
    FUNDAMENTALS = "fundamentals"
    FORM_13F = "form_13f"
    OPTIONS_FLOW = "options_flow"
    DARK_POOL = "dark_pool"
    FORM_4 = "form_4"


# Allowed `set-status` moves. open<->needs_data on a capability re-check; open/needs_data advance
# to authored or discarded; an authored idea can only be refuted (its strategy failed) or
# discarded; refuted/discarded are terminal. A no-op (X -> X) is never a legal change.
ALLOWED_IDEA_TRANSITIONS: dict[IdeaStatus, set[IdeaStatus]] = {
    IdeaStatus.OPEN: {IdeaStatus.NEEDS_DATA, IdeaStatus.AUTHORED, IdeaStatus.DISCARDED},
    IdeaStatus.NEEDS_DATA: {IdeaStatus.OPEN, IdeaStatus.AUTHORED, IdeaStatus.DISCARDED},
    IdeaStatus.AUTHORED: {IdeaStatus.REFUTED, IdeaStatus.DISCARDED},
    IdeaStatus.REFUTED: set(),
    IdeaStatus.DISCARDED: set(),
}


def can_change_status(frm: IdeaStatus, to: IdeaStatus) -> bool:
    return to in ALLOWED_IDEA_TRANSITIONS[frm]


@dataclass
class Idea:
    id: int
    title: str
    hypothesis: str
    family: str | None
    tags: list[str]
    source_type: SourceType
    source_ref: str | None
    source_date: str | None
    source_note: str | None
    required_data: list[DataCapability]
    status: IdeaStatus
    signature: str
    authored_strategy_id: int | None
    duplicate_of_idea_id: int | None
    override_reason: str | None
    created_at: str
    updated_at: str
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_idea_contracts.py -q`
Expected: PASS (4 tests)

- [ ] **Step 5: Run the quality gate**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: all green

- [ ] **Step 6: Commit**

```bash
git add algua/contracts/idea.py tests/test_idea_contracts.py
git commit -m "feat(contracts): Idea record, status/source/capability enums, status state machine (#126)"
```

---

### Task 2: Platform data-capability vocabulary

**Files:**
- Create: `algua/data/capabilities.py`
- Test: `tests/test_data_capabilities.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_data_capabilities.py
from algua.contracts.idea import DataCapability
from algua.data.capabilities import supported_capabilities


def test_supported_is_ohlcv_only_today():
    supported = supported_capabilities()
    assert supported == frozenset({DataCapability.OHLCV})


def test_alt_data_is_not_supported():
    supported = supported_capabilities()
    assert DataCapability.OHLCV in supported
    assert DataCapability.FORM_13F not in supported
    assert DataCapability.OPTIONS_FLOW not in supported
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_data_capabilities.py -q`
Expected: FAIL (`ModuleNotFoundError: algua.data.capabilities`)

- [ ] **Step 3: Write the implementation**

```python
# algua/data/capabilities.py
from __future__ import annotations

from algua.contracts.idea import DataCapability
from algua.data.models import Dataset

# Maps a platform Dataset (an ingestion/serving path that EXISTS) to the strategy-input
# DataCapability it provides. Extend this when a new ingestion path lands (e.g. a filings
# Dataset -> FORM_13F); that single edit lets parked ideas needing it become testable.
_DATASET_CAPABILITY: dict[Dataset, DataCapability] = {
    Dataset.BARS: DataCapability.OHLCV,
}


def supported_capabilities() -> frozenset[DataCapability]:
    """DataCapability values the platform can provide to a backtest, derived from the data
    layer's dataset support. "Supported" = an ingestion/serving path EXISTS (demo mode serves
    OHLCV with no loaded snapshot), NOT "a snapshot is currently loaded". Today: {OHLCV}."""
    return frozenset(_DATASET_CAPABILITY.values())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_data_capabilities.py -q`
Expected: PASS (2 tests)

- [ ] **Step 5: Run the quality gate**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: all green (confirms `data` may import `contracts.idea`)

- [ ] **Step 6: Commit**

```bash
git add algua/data/capabilities.py tests/test_data_capabilities.py
git commit -m "feat(data): supported_capabilities() vocabulary derived from Dataset support (#126)"
```

---

### Task 3: Status classifier (open vs needs_data)

**Files:**
- Create: `algua/research/ideas.py`
- Test: `tests/test_research_ideas.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_research_ideas.py
from algua.contracts.idea import DataCapability, IdeaStatus
from algua.research.ideas import classify_status

SUPPORTED = frozenset({DataCapability.OHLCV})


def test_ohlcv_only_is_open():
    assert classify_status([DataCapability.OHLCV], SUPPORTED) is IdeaStatus.OPEN


def test_no_data_is_open():
    assert classify_status([], SUPPORTED) is IdeaStatus.OPEN


def test_unsupported_capability_parks():
    assert classify_status(
        [DataCapability.OHLCV, DataCapability.FORM_13F], SUPPORTED) is IdeaStatus.NEEDS_DATA
    assert classify_status([DataCapability.OPTIONS_FLOW], SUPPORTED) is IdeaStatus.NEEDS_DATA
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_research_ideas.py -q`
Expected: FAIL (`ModuleNotFoundError: algua.research.ideas`)

- [ ] **Step 3: Write the implementation**

```python
# algua/research/ideas.py
from __future__ import annotations

from collections.abc import Collection

from algua.contracts.idea import DataCapability, IdeaStatus


def classify_status(
    required_data: Collection[DataCapability],
    supported: Collection[DataCapability],
) -> IdeaStatus:
    """OPEN when every required capability is platform-supported; else NEEDS_DATA (parked).
    An idea requiring no data is trivially OPEN. Note: OPEN means "implementable by the
    platform", NOT "a covering snapshot exists" — real-data/PIT readiness stays the promotion
    gate's job."""
    return IdeaStatus.OPEN if set(required_data) <= set(supported) else IdeaStatus.NEEDS_DATA
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_research_ideas.py -q`
Expected: PASS (3 tests)

- [ ] **Step 5: Run the quality gate**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: all green

- [ ] **Step 6: Commit**

```bash
git add algua/research/ideas.py tests/test_research_ideas.py
git commit -m "feat(research): classify_status — park ideas needing unsupported data (#126)"
```

---

### Task 4: Deterministic dedup primitives

**Files:**
- Create: `algua/research/idea_dedup.py`
- Test: `tests/test_idea_dedup.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_idea_dedup.py
from algua.research.idea_dedup import (
    families_comparable,
    is_collision,
    jaccard,
    signature,
)


def test_signature_normalizes_and_is_order_independent():
    a = signature("Momentum Reversal", "Stocks that fell rebound strongly")
    b = signature("reversal momentum", "strongly rebound that fell stocks")
    assert a == b  # lowercased, stopword-stripped, sorted token set
    assert "the" not in a.split() and "that" not in a.split()  # stopwords removed


def test_jaccard_bounds():
    assert jaccard(set(), set()) == 1.0
    assert jaccard({"a"}, set()) == 0.0
    assert jaccard({"a", "b"}, {"a", "b"}) == 1.0
    assert jaccard({"a", "b"}, {"b", "c"}) == 1 / 3


def test_families_comparable_null_is_failsafe():
    assert families_comparable("mom", "mom") is True
    assert families_comparable("mom", "value") is False
    # NULL/unknown on EITHER side compares against everything (cannot suppress a collision)
    assert families_comparable(None, "value") is True
    assert families_comparable("mom", "unknown") is True
    assert families_comparable(None, None) is True


def test_is_collision_respects_family_and_threshold():
    sig = signature("low volatility anomaly", "low vol names outperform on risk-adjusted basis")
    near = signature("the low volatility anomaly", "low vol stocks outperform risk adjusted")
    far = signature("earnings drift", "prices drift after earnings surprises for weeks")
    # same family + high overlap -> collision
    assert is_collision(sig, "vol", near, "vol") is True
    # different concrete family -> not compared, no collision
    assert is_collision(sig, "vol", near, "momentum") is False
    # null family on the candidate -> compared anyway; low overlap -> no collision
    assert is_collision(sig, None, far, "drift") is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_idea_dedup.py -q`
Expected: FAIL (`ModuleNotFoundError: algua.research.idea_dedup`)

- [ ] **Step 3: Write the implementation**

```python
# algua/research/idea_dedup.py
from __future__ import annotations

import re

# Small, explicit stopword set: generic finance/strategy filler that would otherwise inflate
# token overlap between unrelated ideas. Not a linguistics project — just the obvious noise.
_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with", "by", "is", "are",
    "be", "that", "this", "as", "at", "from", "it", "its", "we", "using", "use", "based",
    "strategy", "signal", "returns", "return", "stock", "stocks", "market",
})
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_UNKNOWN_FAMILY = "unknown"


def _tokens(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1}


def signature(title: str, hypothesis: str) -> str:
    """Normalized, order-independent dedup signature: the sorted, deduped, stopword-stripped
    token set of title + hypothesis, space-joined. Stored on the idea so a collision is a cheap
    token-set comparison and the signature stays human-inspectable."""
    return " ".join(sorted(_tokens(f"{title} {hypothesis}")))


def _sig_tokens(sig: str) -> set[str]:
    return set(sig.split())


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def families_comparable(f1: str | None, f2: str | None) -> bool:
    """Whether two ideas are in scope to be compared. A NULL or "unknown" family on EITHER side
    is compared against everything (fail-safe: a missing/mis-tagged family can never silently
    suppress collision detection). Two concrete families compare only when equal."""
    if f1 in (None, _UNKNOWN_FAMILY) or f2 in (None, _UNKNOWN_FAMILY):
        return True
    return f1 == f2


def is_collision(
    cand_signature: str,
    cand_family: str | None,
    other_signature: str,
    other_family: str | None,
    *,
    threshold: float = 0.6,
) -> bool:
    """True when two ideas are in comparable families AND their signatures' Jaccard meets the
    threshold. Coarse and recall-oriented within a family; the agent semantic layer is the
    backstop for paraphrase-level evasion the token set can't catch."""
    if not families_comparable(cand_family, other_family):
        return False
    return jaccard(_sig_tokens(cand_signature), _sig_tokens(other_signature)) >= threshold
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_idea_dedup.py -q`
Expected: PASS (4 tests)

- [ ] **Step 5: Run the quality gate**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: all green

- [ ] **Step 6: Commit**

```bash
git add algua/research/idea_dedup.py tests/test_idea_dedup.py
git commit -m "feat(research): deterministic idea dedup (signature/jaccard/family-safe collision) (#126)"
```

---

### Task 5: `ideas` table migration (schema v18 → v19)

**Files:**
- Modify: `algua/registry/db.py` (add `ideas` to `_SCHEMA`; bump `SCHEMA_VERSION`)
- Modify: `tests/test_registry_db.py` (update the version pins)
- Test: `tests/test_registry_db.py` (add ideas-table assertions)

- [ ] **Step 1: Update the failing tests first**

In `tests/test_registry_db.py`, replace the version-pin test (currently `test_schema_version_is_18`):

```python
def test_schema_version_is_19():
    assert SCHEMA_VERSION == 19
```

and update `test_migrate_is_idempotent_at_v18` to assert 19:

```python
def test_migrate_is_idempotent_at_v19(tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    migrate(conn)  # second run must be a no-op, not an error
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 19
```

Then append a new test:

```python
def test_ideas_table_created_with_expected_columns(tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(ideas)")}
    assert {
        "id", "title", "hypothesis", "family", "tags", "source_type", "source_ref",
        "source_date", "source_note", "required_data", "status", "signature",
        "authored_strategy_id", "duplicate_of_idea_id", "override_reason",
        "created_at", "updated_at",
    } <= cols
    # FK to strategies(id) is declared
    fks = {row["table"] for row in conn.execute("PRAGMA foreign_key_list(ideas)")}
    assert "strategies" in fks
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_registry_db.py -q`
Expected: FAIL (`SCHEMA_VERSION == 18`; `no such table: ideas`)

- [ ] **Step 3: Implement — add the table and bump the version**

In `algua/registry/db.py`, change `SCHEMA_VERSION = 18` to:

```python
SCHEMA_VERSION = 19
```

In the `_SCHEMA` string, immediately before the closing `"""` (after the `live_reservations` table), add:

```sql
-- ideas is the structured top-of-funnel pool (#126): externally-sourced, deduped,
-- provenance-stamped hypothesis records that climb the normal gated ladder. authored_strategy_id
-- is the relational link to the strategy an idea became (NULL until authored); the dedup gate
-- resolves a refuted strategy through this FK (a live join), so a refuted strategy blocks its
-- idea's near-duplicates without mutating idea rows. duplicate_of_idea_id records a deliberate
-- --allow-duplicate override (paired with override_reason).
CREATE TABLE IF NOT EXISTS ideas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    hypothesis TEXT NOT NULL,
    family TEXT,
    tags TEXT NOT NULL DEFAULT '[]',
    source_type TEXT NOT NULL,
    source_ref TEXT,
    source_date TEXT,
    source_note TEXT,
    required_data TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL,
    signature TEXT NOT NULL,
    authored_strategy_id INTEGER REFERENCES strategies(id),
    duplicate_of_idea_id INTEGER REFERENCES ideas(id),
    override_reason TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_ideas_status ON ideas(status);
CREATE INDEX IF NOT EXISTS ix_ideas_family ON ideas(family);
```

(No `_add_missing_columns` entry is needed — this is a whole new table the
`CREATE TABLE IF NOT EXISTS` bootstrap handles.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_registry_db.py -q`
Expected: PASS (all, incl. the new ideas-table test)

- [ ] **Step 5: Run the quality gate**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: all green (watch for other tests that pinned v18 — none known besides those updated)

- [ ] **Step 6: Commit**

```bash
git add algua/registry/db.py tests/test_registry_db.py
git commit -m "feat(registry): schema v19 — ideas table (FK to strategies) (#126)"
```

---

### Task 6: `IdeaRepository` (sqlite store + refuted live join + breadth signal)

**Files:**
- Create: `algua/registry/ideas.py`
- Test: `tests/test_idea_repository.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_idea_repository.py
import pytest

from algua.contracts.idea import DataCapability, IdeaStatus, SourceType
from algua.registry.db import connect, migrate
from algua.registry.ideas import IdeaNotFound, IdeaRepository
from algua.registry.store import SqliteStrategyRepository


def _conns(tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    return IdeaRepository(conn), SqliteStrategyRepository(conn)


def _add(repo, *, title="low vol anomaly", hypothesis="low vol names outperform risk adjusted",
         family="vol", status=IdeaStatus.OPEN, required_data=None):
    return repo.add(
        title=title, hypothesis=hypothesis, family=family, tags=["factor"],
        source_type=SourceType.PAPER, source_ref="http://x", source_date="2020-01-01",
        source_note=None, required_data=required_data or [DataCapability.OHLCV], status=status)


def test_add_get_roundtrip(tmp_path):
    repo, _ = _conns(tmp_path)
    idea = _add(repo)
    assert idea.id == 1
    fetched = repo.get(idea.id)
    assert fetched.title == "low vol anomaly"
    assert fetched.required_data == [DataCapability.OHLCV]
    assert fetched.status is IdeaStatus.OPEN
    assert fetched.signature  # computed + stored


def test_get_missing_raises(tmp_path):
    repo, _ = _conns(tmp_path)
    with pytest.raises(IdeaNotFound):
        repo.get(999)


def test_list_filters_by_status_and_family(tmp_path):
    repo, _ = _conns(tmp_path)
    _add(repo, title="a", family="vol")
    _add(repo, title="b", family="mom", status=IdeaStatus.NEEDS_DATA,
         required_data=[DataCapability.FORM_13F])
    assert [i.title for i in repo.list(family="vol")] == ["a"]
    assert [i.title for i in repo.list(status=IdeaStatus.NEEDS_DATA)] == ["b"]


def test_find_collisions_same_family(tmp_path):
    repo, _ = _conns(tmp_path)
    _add(repo, title="low vol anomaly", hypothesis="low vol names outperform risk adjusted",
         family="vol")
    hits = repo.find_collisions(
        title="the low vol anomaly",
        hypothesis="low vol stocks outperform on a risk adjusted basis", family="vol")
    assert len(hits) == 1
    assert hits[0].effective_status is IdeaStatus.OPEN


def test_find_collisions_ignores_discarded(tmp_path):
    repo, _ = _conns(tmp_path)
    idea = _add(repo)
    repo.set_status(idea.id, to=IdeaStatus.DISCARDED)
    assert repo.find_collisions(
        title="low vol anomaly", hypothesis="low vol names outperform risk adjusted",
        family="vol") == []


def test_refuted_strategy_blocks_via_live_join(tmp_path):
    repo, strat = _conns(tmp_path)
    idea = _add(repo)
    s = strat.add("lowvol_v1", family="vol")
    repo.set_status(idea.id, to=IdeaStatus.AUTHORED, authored_strategy_id=s.id)
    # The authored idea's strategy gets refuted (registry set, elsewhere).
    strat.update_metadata("lowvol_v1", hypothesis_status="refuted")  # type: ignore[arg-type]
    hits = repo.find_collisions(
        title="low vol anomaly redux",
        hypothesis="low vol names outperform on risk adjusted returns", family="vol")
    assert len(hits) == 1
    assert hits[0].effective_status is IdeaStatus.REFUTED  # downgraded by the join


def test_set_status_rejects_illegal(tmp_path):
    repo, _ = _conns(tmp_path)
    idea = _add(repo)
    repo.set_status(idea.id, to=IdeaStatus.DISCARDED)
    with pytest.raises(ValueError, match="illegal idea status change"):
        repo.set_status(idea.id, to=IdeaStatus.OPEN)


def test_windowed_counts_by_status(tmp_path):
    repo, _ = _conns(tmp_path)
    _add(repo, title="a")
    _add(repo, title="b", status=IdeaStatus.NEEDS_DATA,
         required_data=[DataCapability.FORM_13F])
    counts = repo.windowed_idea_counts(90)
    assert counts["open"] == 1
    assert counts["needs_data"] == 1
    assert counts["total"] == 2
```

Note: `update_metadata`'s `hypothesis_status` takes a `HypothesisStatus`; in the test, import and pass `HypothesisStatus.REFUTED` rather than the string. Adjust the test import:
```python
from algua.contracts.registry_metadata import HypothesisStatus
...
strat.update_metadata("lowvol_v1", hypothesis_status=HypothesisStatus.REFUTED)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_idea_repository.py -q`
Expected: FAIL (`ModuleNotFoundError: algua.registry.ideas`)

- [ ] **Step 3: Write the implementation**

```python
# algua/registry/ideas.py
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from algua.contracts.idea import (
    DataCapability,
    Idea,
    IdeaStatus,
    SourceType,
    can_change_status,
)
from algua.registry.metadata import dump_tags, load_tags
from algua.research.idea_dedup import is_collision, signature


class IdeaNotFound(LookupError):
    def __init__(self, idea_id: int) -> None:
        super().__init__(f"idea {idea_id} not found")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _dump_caps(caps: list[DataCapability]) -> str:
    return json.dumps([c.value for c in caps])


def _load_caps(value: str | None) -> list[DataCapability]:
    if not value:
        return []
    return [DataCapability(c) for c in json.loads(value)]


def _row_to_idea(row: sqlite3.Row) -> Idea:
    return Idea(
        id=row["id"], title=row["title"], hypothesis=row["hypothesis"],
        family=row["family"], tags=load_tags(row["tags"]),
        source_type=SourceType(row["source_type"]),
        source_ref=row["source_ref"], source_date=row["source_date"],
        source_note=row["source_note"], required_data=_load_caps(row["required_data"]),
        status=IdeaStatus(row["status"]), signature=row["signature"],
        authored_strategy_id=row["authored_strategy_id"],
        duplicate_of_idea_id=row["duplicate_of_idea_id"],
        override_reason=row["override_reason"],
        created_at=row["created_at"], updated_at=row["updated_at"],
    )


@dataclass
class Collision:
    """A colliding existing idea, with its effective status: AUTHORED is downgraded to REFUTED
    when the linked strategy is currently refuted (the refuted wall)."""

    idea: Idea
    effective_status: IdeaStatus


class IdeaRepository:
    """sqlite-backed idea pool: the only module that embeds idea SQL. Shares the registry
    connection (same DB as strategies), so the refuted-aware dedup can join across the two."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def add(
        self, *, title: str, hypothesis: str, family: str | None, tags: list[str],
        source_type: SourceType, source_ref: str | None, source_date: str | None,
        source_note: str | None, required_data: list[DataCapability], status: IdeaStatus,
        duplicate_of_idea_id: int | None = None, override_reason: str | None = None,
    ) -> Idea:
        sig = signature(title, hypothesis)
        now = _now()
        with self._conn:
            cur = self._conn.execute(
                "INSERT INTO ideas(title, hypothesis, family, tags, source_type, source_ref,"
                " source_date, source_note, required_data, status, signature,"
                " authored_strategy_id, duplicate_of_idea_id, override_reason,"
                " created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (title, hypothesis, family, dump_tags(tags), source_type.value, source_ref,
                 source_date, source_note, _dump_caps(required_data), status.value, sig,
                 None, duplicate_of_idea_id, override_reason, now, now),
            )
        rowid = cur.lastrowid
        assert rowid is not None
        return self.get(int(rowid))

    def get(self, idea_id: int) -> Idea:
        row = self._conn.execute("SELECT * FROM ideas WHERE id=?", (idea_id,)).fetchone()
        if row is None:
            raise IdeaNotFound(idea_id)
        return _row_to_idea(row)

    def list(
        self, *, status: IdeaStatus | None = None, family: str | None = None
    ) -> list[Idea]:
        sql = "SELECT * FROM ideas"
        clauses: list[str] = []
        params: list[object] = []
        if status is not None:
            clauses.append("status = ?")
            params.append(status.value)
        if family is not None:
            clauses.append("family = ?")
            params.append(family)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY id"
        return [_row_to_idea(r) for r in self._conn.execute(sql, params)]

    def find_collisions(
        self, *, title: str, hypothesis: str, family: str | None, threshold: float = 0.6
    ) -> list[Collision]:
        """Non-discarded ideas colliding with the candidate (family-safe + Jaccard). Each
        collision's effective_status downgrades AUTHORED -> REFUTED when the idea's linked
        strategy is currently hypothesis_status='refuted' — the refuted wall, via a read-only
        join (no idea-row mutation, no protected-file change)."""
        cand_sig = signature(title, hypothesis)
        rows = self._conn.execute(
            "SELECT i.*, s.hypothesis_status AS strat_status FROM ideas i"
            " LEFT JOIN strategies s ON s.id = i.authored_strategy_id"
            " WHERE i.status != ?",
            (IdeaStatus.DISCARDED.value,),
        ).fetchall()
        out: list[Collision] = []
        for row in rows:
            if is_collision(cand_sig, family, row["signature"], row["family"],
                            threshold=threshold):
                idea = _row_to_idea(row)
                effective = idea.status
                if idea.status is IdeaStatus.AUTHORED and row["strat_status"] == "refuted":
                    effective = IdeaStatus.REFUTED
                out.append(Collision(idea=idea, effective_status=effective))
        return out

    def set_status(
        self, idea_id: int, *, to: IdeaStatus, authored_strategy_id: int | None = None
    ) -> Idea:
        idea = self.get(idea_id)
        if not can_change_status(idea.status, to):
            raise ValueError(
                f"illegal idea status change {idea.status.value} -> {to.value}")
        if to is IdeaStatus.AUTHORED and authored_strategy_id is None:
            raise ValueError("authored status requires a strategy link")
        with self._conn:
            self._conn.execute(
                "UPDATE ideas SET status=?,"
                " authored_strategy_id=COALESCE(?, authored_strategy_id), updated_at=?"
                " WHERE id=?",
                (to.value, authored_strategy_id, _now(), idea_id),
            )
        return self.get(idea_id)

    def windowed_idea_counts(self, window_days: int) -> dict[str, int]:
        """Idea counts BY STATUS created within the trailing window — the funnel-breadth signal
        the later (human, CODEOWNERS) gate change will consume. By-status (not one number) so the
        gate can pick the right denominator. ISO-8601 UTC strings compare chronologically, so a
        string >= on created_at is correct."""
        cutoff = (datetime.now(UTC) - timedelta(days=window_days)).isoformat()
        counts: dict[str, int] = {s.value: 0 for s in IdeaStatus}
        for row in self._conn.execute(
            "SELECT status, COUNT(*) AS n FROM ideas WHERE created_at >= ? GROUP BY status",
            (cutoff,),
        ):
            counts[row["status"]] = int(row["n"])
        counts["total"] = sum(counts[s.value] for s in IdeaStatus)
        return counts
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_idea_repository.py -q`
Expected: PASS (8 tests)

- [ ] **Step 5: Run the quality gate**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: all green (confirms registry→research import is allowed)

- [ ] **Step 6: Commit**

```bash
git add algua/registry/ideas.py tests/test_idea_repository.py
git commit -m "feat(registry): IdeaRepository — CRUD, refuted-aware dedup join, breadth counts (#126)"
```

---

### Task 7: `research idea` CLI surface

**Files:**
- Create: `algua/cli/idea_cmd.py`
- Modify: `algua/cli/main.py` (import `idea_cmd` to register the subcommands)
- Test: `tests/test_cli_idea.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli_idea.py
import json

import pytest
from typer.testing import CliRunner

from algua.cli.main import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def _tmp_db(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))


def _json(result):
    assert result.exit_code == 0, result.stdout
    return json.loads(result.stdout)


def _add(*args):
    base = ["research", "idea", "add", "--title", "low vol anomaly",
            "--hypothesis", "low vol names outperform risk adjusted",
            "--family", "vol", "--source-type", "paper", "--source-ref", "http://x",
            "--required-data", "ohlcv"]
    return runner.invoke(app, base + list(args))


def test_add_and_show():
    added = _json(_add())
    assert added["ok"] is True
    assert added["status"] == "open"
    shown = _json(runner.invoke(app, ["research", "idea", "show", str(added["id"])]))
    assert shown["title"] == "low vol anomaly"


def test_needs_data_parking():
    out = _json(runner.invoke(app, [
        "research", "idea", "add", "--title", "whale 13f", "--hypothesis", "follow institutions",
        "--family", "flow", "--source-type", "filing", "--required-data", "form_13f"]))
    assert out["status"] == "needs_data"


def test_unknown_capability_errors():
    result = runner.invoke(app, [
        "research", "idea", "add", "--title", "x", "--hypothesis", "y",
        "--source-type", "manual", "--required-data", "satellite_imagery"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["ok"] is False


def test_dedup_collision_fails_closed():
    _add()
    result = _add("--title", "the low vol anomaly",
                  "--hypothesis", "low vol stocks outperform on a risk adjusted basis")
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert len(payload["collisions"]) == 1


def test_allow_duplicate_override():
    _add()
    out = _json(_add("--title", "the low vol anomaly",
                     "--hypothesis", "low vol stocks outperform on a risk adjusted basis",
                     "--allow-duplicate", "--reason", "distinct universe"))
    assert out["duplicate_of_idea_id"] == 1
    assert out["override_reason"] == "distinct universe"


def test_allow_duplicate_requires_reason():
    _add()
    result = _add("--title", "the low vol anomaly",
                  "--hypothesis", "low vol stocks outperform on a risk adjusted basis",
                  "--allow-duplicate")
    assert result.exit_code == 1
    assert json.loads(result.stdout)["ok"] is False


def test_list_is_bare_array():
    _add()
    listed = _json(runner.invoke(app, ["research", "idea", "list"]))
    assert isinstance(listed, list)
    assert listed[0]["title"] == "low vol anomaly"


def test_dedup_check_no_write():
    _add()
    out = _json(runner.invoke(app, [
        "research", "idea", "dedup-check", "--title", "the low vol anomaly",
        "--hypothesis", "low vol stocks outperform on a risk adjusted basis", "--family", "vol"]))
    assert out["is_novel"] is False
    assert len(out["collisions"]) == 1
    # nothing was written
    assert len(_json(runner.invoke(app, ["research", "idea", "list"]))) == 1


def test_set_status_authored_requires_strategy():
    added = _json(_add())
    result = runner.invoke(app, [
        "research", "idea", "set-status", str(added["id"]), "--to", "authored"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["ok"] is False


def test_set_status_authored_links_strategy():
    added = _json(_add())
    runner.invoke(app, ["registry", "add", "lowvol_v1", "--family", "vol"])
    out = _json(runner.invoke(app, [
        "research", "idea", "set-status", str(added["id"]), "--to", "authored",
        "--strategy", "lowvol_v1"]))
    assert out["status"] == "authored"
    assert out["authored_strategy_id"] is not None


def test_stats_counts_by_status():
    _add()
    out = _json(runner.invoke(app, ["research", "idea", "stats"]))
    assert out["counts"]["open"] == 1
    assert out["counts"]["total"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli_idea.py -q`
Expected: FAIL (no `research idea` command — `add` exits non-zero / not found)

- [ ] **Step 3: Implement the CLI module**

```python
# algua/cli/idea_cmd.py
from __future__ import annotations

import re

import typer

from algua.cli._common import ok, registry_conn
from algua.cli.app import emit
from algua.cli.errors import json_errors
from algua.cli.research_cmd import research_app
from algua.contracts.idea import DataCapability, IdeaStatus, SourceType
from algua.data.capabilities import supported_capabilities
from algua.registry.ideas import Collision, IdeaRepository
from algua.registry.store import SqliteStrategyRepository
from algua.research.ideas import classify_status

_FAMILY_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")

idea_app = typer.Typer(
    help="Idea pool: source, dedup, and park research hypotheses", no_args_is_help=True)
research_app.add_typer(idea_app, name="idea")


def _idea_json(idea) -> dict:
    return {
        "id": idea.id, "title": idea.title, "hypothesis": idea.hypothesis,
        "family": idea.family, "tags": idea.tags, "source_type": idea.source_type.value,
        "source_ref": idea.source_ref, "source_date": idea.source_date,
        "source_note": idea.source_note,
        "required_data": [c.value for c in idea.required_data],
        "status": idea.status.value, "signature": idea.signature,
        "authored_strategy_id": idea.authored_strategy_id,
        "duplicate_of_idea_id": idea.duplicate_of_idea_id,
        "override_reason": idea.override_reason,
        "created_at": idea.created_at, "updated_at": idea.updated_at,
    }


def _collision_json(c: Collision) -> dict:
    return {"id": c.idea.id, "title": c.idea.title, "family": c.idea.family,
            "status": c.idea.status.value, "effective_status": c.effective_status.value}


def _parse_required_data(raw: str | None) -> list[DataCapability]:
    if not raw:
        return []
    caps: list[DataCapability] = []
    for token in raw.split(","):
        token = token.strip().lower()
        if not token:
            continue
        try:
            caps.append(DataCapability(token))
        except ValueError as exc:
            allowed = ", ".join(c.value for c in DataCapability)
            raise ValueError(
                f"unknown required-data capability {token!r}; allowed: {allowed}") from exc
    return caps


@idea_app.command("add")
@json_errors(ValueError, LookupError)
def add(
    title: str = typer.Option(..., "--title"),
    hypothesis: str = typer.Option(..., "--hypothesis"),
    family: str = typer.Option(None, "--family", help="thesis family slug"),
    source_type: SourceType = typer.Option(..., "--source-type"),
    source_ref: str = typer.Option(None, "--source-ref", help="url / citation / doi"),
    source_date: str = typer.Option(None, "--source-date", help="ISO date of the source"),
    source_note: str = typer.Option(None, "--source-note"),
    tag: list[str] = typer.Option(None, "--tag", help="tag (repeatable)"),
    required_data: str = typer.Option(
        None, "--required-data", help="comma-separated DataCapability values"),
    allow_duplicate: bool = typer.Option(False, "--allow-duplicate"),
    reason: str = typer.Option(None, "--reason", help="required with --allow-duplicate"),
) -> None:
    """Add a sourced idea. Auto-parks (needs_data) when it needs unsupported data. Fails closed on
    a dedup collision unless --allow-duplicate --reason."""
    if family is not None and not _FAMILY_RE.match(family):
        raise ValueError(f"invalid family {family!r}: must be a lowercase slug (a-z, 0-9, hyphen)")
    caps = _parse_required_data(required_data)
    status = classify_status(caps, supported_capabilities())
    with registry_conn() as conn:
        repo = IdeaRepository(conn)
        collisions = repo.find_collisions(title=title, hypothesis=hypothesis, family=family)
        dup_of: int | None = None
        if collisions:
            if not allow_duplicate:
                emit({
                    "ok": False,
                    "error": "dedup collision; pass --allow-duplicate --reason to override",
                    "collisions": [_collision_json(c) for c in collisions],
                })
                raise typer.Exit(code=1)
            if not reason:
                raise ValueError("--allow-duplicate requires --reason")
            dup_of = collisions[0].idea.id
        idea = repo.add(
            title=title, hypothesis=hypothesis, family=family, tags=tag or [],
            source_type=source_type, source_ref=source_ref, source_date=source_date,
            source_note=source_note, required_data=caps, status=status,
            duplicate_of_idea_id=dup_of, override_reason=reason if dup_of else None)
    emit(ok(_idea_json(idea)))


@idea_app.command("list")
@json_errors(ValueError, LookupError)
def list_(
    status: str = typer.Option(None, "--status", help="filter by idea status"),
    family: str = typer.Option(None, "--family", help="filter by thesis family"),
) -> None:
    """List ideas (optional filters). Emits a bare JSON array (collection convention)."""
    st = IdeaStatus(status) if status else None
    with registry_conn() as conn:
        ideas = IdeaRepository(conn).list(status=st, family=family)
    emit([_idea_json(i) for i in ideas])


@idea_app.command("show")
@json_errors(ValueError, LookupError)
def show(idea_id: int = typer.Argument(..., metavar="ID")) -> None:
    """Show one idea by id."""
    with registry_conn() as conn:
        idea = IdeaRepository(conn).get(idea_id)
    emit(ok(_idea_json(idea)))


@idea_app.command("dedup-check")
@json_errors(ValueError, LookupError)
def dedup_check(
    title: str = typer.Option(..., "--title"),
    hypothesis: str = typer.Option(..., "--hypothesis"),
    family: str = typer.Option(None, "--family"),
) -> None:
    """Preflight a candidate against the pool; no write. Reports collisions (incl. refuted)."""
    with registry_conn() as conn:
        collisions = IdeaRepository(conn).find_collisions(
            title=title, hypothesis=hypothesis, family=family)
    emit(ok({"is_novel": not collisions,
             "collisions": [_collision_json(c) for c in collisions]}))


@idea_app.command("set-status")
@json_errors(ValueError, LookupError)
def set_status(
    idea_id: int = typer.Argument(..., metavar="ID"),
    to: IdeaStatus = typer.Option(..., "--to"),
    strategy: str = typer.Option(
        None, "--strategy", help="strategy name (required for --to authored)"),
) -> None:
    """Move an idea along its lifecycle (state-machine checked). --to authored links a strategy."""
    with registry_conn() as conn:
        strat_id: int | None = None
        if to is IdeaStatus.AUTHORED:
            if not strategy:
                raise ValueError("--to authored requires --strategy <name>")
            strat_id = SqliteStrategyRepository(conn).get(strategy).id
        idea = IdeaRepository(conn).set_status(idea_id, to=to, authored_strategy_id=strat_id)
    emit(ok(_idea_json(idea)))


@idea_app.command("stats")
@json_errors(ValueError, LookupError)
def stats(window_days: int = typer.Option(90, "--window-days")) -> None:
    """Funnel-breadth signal: idea counts by status in the trailing window. EXPOSED for the future
    (human, CODEOWNERS) gate change; NOT yet consumed by the promotion gate."""
    with registry_conn() as conn:
        counts = IdeaRepository(conn).windowed_idea_counts(window_days)
    emit(ok({"window_days": window_days, "counts": counts}))
```

- [ ] **Step 4: Register the module in the CLI**

In `algua/cli/main.py`, add `idea_cmd` to the import block (alphabetical, AFTER `data_cmd` so `research_cmd` — which defines `research_app` — is imported before `idea_cmd` attaches to it; since `idea_cmd` imports `research_cmd` itself, order is safe either way):

```python
from algua.cli import (  # noqa: F401 - imports register subcommands
    backtest_cmd,
    data_cmd,
    idea_cmd,
    live_cmd,
    paper_cmd,
    registry_cmd,
    research_cmd,
    strategy_cmd,
)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_cli_idea.py -q`
Expected: PASS (11 tests)

- [ ] **Step 6: Run the quality gate**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: all green

- [ ] **Step 7: Commit**

```bash
git add algua/cli/idea_cmd.py algua/cli/main.py tests/test_cli_idea.py
git commit -m "feat(cli): research idea add/list/show/dedup-check/set-status/stats (#126)"
```

---

### Task 8: `source-ideas` skill (collection-only; loop unchanged)

**Files:**
- Create: `.codex/skills/source-ideas/SKILL.md`

This task has no automated test (it is an agent playbook). The verification is: the file exists,
it drives the CLI built above, and it explicitly states the research loop is NOT yet auto-wired
to it (the breadth-sequencing guard).

- [ ] **Step 1: Write the skill**

```markdown
---
name: source-ideas
description: Source external trading-idea priors (papers/filings/forums) into the structured idea pool — run deep-research, extract hypotheses, dedup them, and `research idea add` the survivors. Use to widen the top of the funnel with diversity under discipline. NOT yet the research loop's default ideation step.
---

# Source ideas (structured external priors → idea pool)

Widen the top of the funnel with DIVERSITY under DISCIPLINE — not raw volume. You source
external priors into structured, deduped, provenance-stamped records; the CLI is the
deterministic store, you are the semantic judge.

> Collection-only: this skill populates the pool for deliberate use. It is NOT yet wired in as
> the research loop's automatic ideation step — that waits until the promotion gate counts idea
> breadth (a human, CODEOWNERS change). Do not mass-produce ideas straight into authoring.

## Steps

1. **Pick a topic / thesis family.** A family slug from `kb/strategies/_families.md`, or a new
   external angle (a factor, an anomaly, an alt-data thesis).

2. **Run `deep-research`** on it (the deep-research skill). Capture, for each candidate edge:
   - a short `title` and a one-paragraph `hypothesis` (the claimed edge + why),
   - `source_type` (paper|url|forum|filing|thesis) + `source_ref` (url/doi) + `source_date`,
   - the `required_data` capabilities it needs (`ohlcv` today; alt-data like `form_13f`,
     `options_flow`, `dark_pool`, `form_4` will park as `needs_data`),
   - a candidate `family` slug.

3. **Dedup each candidate (deterministic + semantic):**
   - Run `uv run algua research idea dedup-check --title T --hypothesis H --family F`.
   - If `is_novel` is false, READ the returned collisions. If any collision's
     `effective_status` is `refuted`, DO NOT re-add — that idea was already rejected.
   - Cross-check the kb (`_index.md`, `_families.md`) for a refuted/duplicate you recognize that
     a token match would miss (paraphrase, synonym, ticker swap). You are the semantic backstop.

4. **Add the survivors:**
   `uv run algua research idea add --title T --hypothesis H --family F --source-type paper
   --source-ref URL --source-date D --required-data ohlcv [--tag t1]`
   - A genuine, distinct angle that nonetheless trips the token dedup may be added with
     `--allow-duplicate --reason "<why it's genuinely new>"` — use sparingly, with a real reason.

5. **Triage the pool:** `uv run algua research idea list --status open` are testable now;
   `--status needs_data` are parked until their data lands. Pick `open` ideas to author via the
   normal research loop (`strategy new` → backtest → walk-forward → sweep → `research promote`).
   On authoring, link the idea: `uv run algua research idea set-status <id> --to authored
   --strategy <name>`.

6. **Check funnel breadth:** `uv run algua research idea stats` shows idea counts by status — the
   discipline signal. A wide pool is healthy ONLY because the gate will (soon) count it.
```

- [ ] **Step 2: Verify the file exists and the loop is unchanged**

Run: `test -f .codex/skills/source-ideas/SKILL.md && echo OK`
Run: `git status --porcelain .codex/skills/run-the-research-loop` → expect NO output (loop untouched)
Expected: `OK`, and the research-loop skill is not modified.

- [ ] **Step 3: Run the quality gate**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: all green (a markdown-only change doesn't affect code, but confirm nothing drifted)

- [ ] **Step 4: Commit**

```bash
git add .codex/skills/source-ideas/SKILL.md
git commit -m "docs(skill): source-ideas playbook — deep-research into the idea pool (collection-only) (#126)"
```

---

## Self-Review (completed against the spec)

- **Spec coverage:** ideas table (T5), Idea/enums/state-machine (T1), DataCapability vocab + parking (T1/T2/T3), dedup incl. null-family-safe + refuted live-join (T4/T6), `research idea` CLI incl. fail-closed collision + override audit (T7), counts-by-status breadth signal (T6 repo + T7 `stats`), `source-ideas` skill collection-only (T8). Deferred items (gate wiring, loop auto-wiring, needs_data re-check, status-propagation, deeper dedup) are intentionally absent — see spec "Out of scope".
- **Placeholder scan:** none — every code/test step is complete.
- **Type consistency:** `signature`/`is_collision`/`families_comparable` (T4) match their `IdeaRepository` callers (T6); `classify_status` (T3) matches the CLI (T7); `Collision`/`Idea` field names match `_idea_json`/`_collision_json` (T7); `windowed_idea_counts` keys (status values + `total`) match the `stats` test (T7).
- **Protected files:** none modified (verified against CODEOWNERS).
