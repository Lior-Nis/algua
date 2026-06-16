# Holdout Data Identity (#205) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a strategy's overlapping out-of-sample (OOS) calendar window single-use **regardless of data provenance**, closing the snapshot-vs-provider holdout double-burn (#205), and require agents to promote off a reproducible data source.

**Architecture:** Two small, independent changes. (1) `reserve_holdout` drops the `data_source`/`snapshot_id` bucket from its match `SELECT`, so the existing #192 OOS-interval-overlap test applies across all provenance (plus a defensive inverted-interval guard). (2) `promotion_preflight` refuses an agent promote unless the provider is snapshot-backed or carries a `reproducible` marker; `SyntheticProvider` gets that marker. No schema change, no hashing.

**Tech Stack:** Python, SQLite (`algua/registry`), pytest. Design spec: `docs/superpowers/specs/2026-06-15-holdout-data-identity-205-design.md`.

**Quality gate (run before each commit):** `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`

---

## File Structure

- `algua/registry/store.py` — `reserve_holdout`: provenance-independent match + inverted-interval guard.
- `algua/registry/promotion.py` — `promotion_preflight`: reproducible-source guard for agents.
- `algua/backtest/_sample.py` — `SyntheticProvider`: add `reproducible = True`.
- `tests/test_registry_store.py` — invert the old provenance-precedence test; add cross-provenance + inverted-interval tests.
- `tests/test_promotion.py` — add reproducible-source guard tests.
- `algua/registry/db.py`, `algua/registry/repository.py` — docstring/comment updates only.

---

### Task 1: Provenance-independent holdout match + inverted-interval guard

**Files:**
- Modify: `algua/registry/store.py` (`reserve_holdout`, ~436–464)
- Test: `tests/test_registry_store.py` (~248 `test_reserve_snapshot_identity_precedence`, and new tests near the `#192` interval-matching block ~150)

- [ ] **Step 1: Replace the old provenance-precedence test with a provenance-independent one**

In `tests/test_registry_store.py`, delete `test_reserve_snapshot_identity_precedence` (the one whose comment says "does NOT block a non-snapshot probe") and replace it with:

```python
def test_reserve_is_provenance_independent(repo_with_strategy):
    # #205: the OOS calendar window is the single-use unit REGARDLESS of provenance. A snapshot-backed
    # burn now blocks a non-snapshot probe over the same interval (was: distinct provenance buckets).
    repo, sid = repo_with_strategy
    _reserve(repo, sid, hs="2022-06-01", he="2022-12-31", snap="snapA")
    with pytest.raises(ValueError, match="holdout already consumed"):
        _reserve(repo, sid, hs="2022-06-01", he="2022-12-31", snap=None)
    # A DIFFERENT snapshot of an overlapping window is also blocked.
    with pytest.raises(ValueError, match="holdout already consumed"):
        _reserve(repo, sid, hs="2022-09-01", he="2023-03-31", snap="snapB")


def test_partial_overlap_cross_provenance_blocks(repo_with_strategy):
    # GATE-1 CRITICAL: a burn over snapshot S blocks a PARTIALLY-overlapping probe via a different
    # source. Whole-window content hashing could not catch this; the interval-overlap test does.
    repo, sid = repo_with_strategy
    _reserve(repo, sid, hs="2023-01-01", he="2023-12-31", snap="snapS")
    with pytest.raises(ValueError, match="holdout already consumed"):
        _reserve(repo, sid, hs="2023-06-01", he="2024-06-30", snap=None, ds="yfinance")


def test_inverted_interval_rejected(repo_with_strategy):
    # Defensive (GATE-1 r2): an inverted incoming interval would slip both the NULL branch and the
    # overlap test and fail OPEN. reserve_holdout rejects start > end.
    repo, sid = repo_with_strategy
    with pytest.raises(ValueError, match="invalid holdout interval"):
        _reserve(repo, sid, hs="2023-12-31", he="2023-01-01")
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `uv run pytest tests/test_registry_store.py -q -k "provenance_independent or partial_overlap_cross_provenance or inverted_interval"`
Expected: FAIL — `test_reserve_is_provenance_independent` and `test_partial_overlap_cross_provenance_blocks` fail (old code lets the non-snapshot/cross-provenance probe through, so no `ValueError` is raised); `test_inverted_interval_rejected` fails (no such validation yet).

- [ ] **Step 3: Make the match provenance-independent and add the inverted-interval guard**

In `algua/registry/store.py`, replace this block (the `# Data identity:` comment through the `data_param = data_source` lines, ~436–443):

```python
        # Data identity: snapshot_id when the probe has one (a snapshot-backed row is a DISTINCT
        # identity from a non-snapshot probe), else data_source among rows lacking a snapshot.
        if snapshot_id is not None:
            data_match = "snapshot_id = ?"
            data_param: str = snapshot_id
        else:
            data_match = "snapshot_id IS NULL AND data_source = ?"
            data_param = data_source
```

with:

```python
        # Single-use identity is the OOS INTERVAL [holdout_start, holdout_end] for the strategy,
        # PROVENANCE-INDEPENDENT (#205): the same OOS calendar window is burn-once regardless of how
        # the bars were reached (snapshot S, a different snapshot S2, or a provider P). data_source/
        # snapshot_id are persisted as EVIDENCE only, never matched on (was: a per-provenance bucket,
        # which let the same physical window be burned twice across provenance — the #205 hole).
        # Defensive (GATE-1): an inverted incoming interval (start > end) would slip both the NULL
        # branch and the overlap test below and fail OPEN, so reject it. holdout_window yields a
        # well-formed interval (idx[train_n] <= idx[-1]); this guards the primitive against a caller.
        if holdout_start > holdout_end:
            raise ValueError(
                f"invalid holdout interval: start {holdout_start!r} > end {holdout_end!r}")
```

Then update the SELECT (~458–464). Replace:

```python
            row = self._conn.execute(
                f"SELECT 1 FROM holdout_evaluations WHERE strategy_id = ?"
                f" AND {data_match}"
                f" AND (holdout_start IS NULL OR holdout_end IS NULL"
                f"      OR (holdout_start <= ? AND ? <= holdout_end)) LIMIT 1",
                (strategy_id, data_param, holdout_end, holdout_start),
            ).fetchone()
```

with:

```python
            row = self._conn.execute(
                "SELECT 1 FROM holdout_evaluations WHERE strategy_id = ?"
                " AND (holdout_start IS NULL OR holdout_end IS NULL"
                "      OR (holdout_start <= ? AND ? <= holdout_end)) LIMIT 1",
                (strategy_id, holdout_end, holdout_start),
            ).fetchone()
```

Also update the existing comment a few lines below the SELECT: change the phrase "Match identity is the OOS INTERVAL ... #192" preamble so it no longer says period/data identity bucketing — specifically the `# Match identity is the OOS INTERVAL` comment block already describes interval matching; leave it, it is still accurate. (No other change in this method; `data_source`/`snapshot_id` are still passed to the INSERT below as evidence.)

- [ ] **Step 4: Run the full store test file to verify pass + no regressions**

Run: `uv run pytest tests/test_registry_store.py -q`
Expected: PASS (including `test_overlapping_interval_blocks`, `test_different_holdout_frac_same_interval_still_blocks`, `test_non_overlapping_interval_allowed`, the three new tests, and the release/finalize tests).

- [ ] **Step 5: Run the concurrency tests (same-provenance single-burn must still hold)**

Run: `uv run pytest tests/test_concurrency.py -q -k holdout`
Expected: PASS (`test_concurrent_reserve_holdout_single_burn`).

- [ ] **Step 6: Commit**

```bash
git add algua/registry/store.py tests/test_registry_store.py
git commit -m "fix(205): provenance-independent single-use holdout OOS window

reserve_holdout drops the data_source/snapshot_id bucket from its match: the OOS
interval [holdout_start, holdout_end] is the single-use unit regardless of how the
bars were reached (snapshot S, snapshot S2, or provider P), closing the #205
double-burn. Adds an inverted-interval fail-closed guard. data_source/snapshot_id
remain evidence-only columns.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Reproducible-source guard for agent promotes

**Files:**
- Modify: `algua/registry/promotion.py` (`promotion_preflight`, insert after `guard_agent_relaxations(...)` ~95)
- Modify: `algua/backtest/_sample.py` (`SyntheticProvider`, ~19)
- Test: `tests/test_promotion.py`

- [ ] **Step 1: Write the failing tests**

In `tests/test_promotion.py`, add near the other `promotion_preflight` tests:

```python
class _NonReproducibleProvider:
    """Neither snapshot_id nor a reproducible marker; get_bars must NOT be reached (the guard fires
    before any provider read)."""

    def get_bars(self, symbols, start, end, timeframe):
        raise AssertionError("provider must not be read: reproducible guard should fire first")


def test_agent_refused_non_reproducible_source(tmp_path):
    repo = _repo(tmp_path)
    repo.add("alpha")
    with pytest.raises(ValueError, match="reproducible data source"):
        promotion_preflight(repo, "alpha", actor=Actor.AGENT, declared_combos=None,
                            allow_holdout_reuse=False, allow_non_pit=False,
                            provider=_NonReproducibleProvider(), start=_START, end=_END)


def test_reproducible_guard_skipped_for_synthetic_agent_and_any_human(tmp_path):
    # SyntheticProvider is reproducible -> the guard does NOT fire for an agent (a later stage check
    # raises instead, since "alpha" is at stage idea). A human is exempt even for a non-reproducible
    # provider (get_bars is never reached because the stage check short-circuits first).
    repo = _repo(tmp_path)
    repo.add("alpha")
    with pytest.raises(Exception) as agent_ei:
        promotion_preflight(repo, "alpha", actor=Actor.AGENT, declared_combos=None,
                            allow_holdout_reuse=False, allow_non_pit=False,
                            provider=SyntheticProvider(seed=0), start=_START, end=_END)
    assert "reproducible data source" not in str(agent_ei.value)
    with pytest.raises(Exception) as human_ei:
        promotion_preflight(repo, "alpha", actor=Actor.HUMAN, declared_combos=None,
                            allow_holdout_reuse=False, allow_non_pit=False,
                            provider=_NonReproducibleProvider(), start=_START, end=_END)
    assert "reproducible data source" not in str(human_ei.value)
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `uv run pytest tests/test_promotion.py -q -k "non_reproducible or reproducible_guard_skipped"`
Expected: FAIL — `test_agent_refused_non_reproducible_source` does not raise the "reproducible data source" message (no guard yet); `test_reproducible_guard_skipped...` fails because `SyntheticProvider` has no `reproducible` attribute, so the new guard (once added) would wrongly fire — but at this step the guard doesn't exist, so the agent path hits `_NonReproducibleProvider`? No: it uses `SyntheticProvider`, which has a real `get_bars`; the stage check raises `TransitionError` first, so this assert passes prematurely. The decisive failing test here is `test_agent_refused_non_reproducible_source`.

- [ ] **Step 3: Add the `reproducible` marker to `SyntheticProvider`**

In `algua/backtest/_sample.py`, in `class SyntheticProvider`, add a class attribute directly under the class line (before `__init__`):

```python
class SyntheticProvider:
    # Deterministic (fixed seed) -> a reproducible data source for the agent promote guard (#205):
    # the OOS bars are identical on a re-run, so a burned holdout is reproducible without a snapshot.
    reproducible = True

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
```

- [ ] **Step 4: Add the reproducible-source guard to `promotion_preflight`**

In `algua/registry/promotion.py`, immediately AFTER the `guard_agent_relaxations(...)` call (~95) and BEFORE `rec = repo.get(name)`, insert:

```python
    # Reproducible-source guard (#205): an agent's holdout burn must be over reproducible bars — an
    # immutable snapshot (snapshot_id set) or a deterministic provider (reproducible marker) — so the
    # OOS truth it spends is identical on a re-run. Refuse a non-snapshot, non-reproducible provider
    # for an agent BEFORE any provider read (verify_signal_panel_parity below fetches bars). Humans
    # are exempt (they accept the cost, mirroring --allow-non-pit). select_provider exposes only
    # demo/snapshot today; this fail-closes a future mutable/live provider. Duck-typed getattr avoids
    # a registry->data import-boundary violation.
    if (actor is Actor.AGENT and getattr(provider, "snapshot_id", None) is None
            and not getattr(provider, "reproducible", False)):
        raise ValueError(
            "agent research promote requires a reproducible data source: an ingested snapshot "
            "(--snapshot) or a deterministic provider. A non-reproducible/live provider's bars may "
            "revise between runs; promote with --actor human to accept the cost.")
```

- [ ] **Step 5: Run the new tests to verify they pass**

Run: `uv run pytest tests/test_promotion.py -q -k "non_reproducible or reproducible_guard_skipped"`
Expected: PASS.

- [ ] **Step 6: Run the full promotion + CLI research suites (existing agent+synthetic paths must stay green)**

Run: `uv run pytest tests/test_promotion.py tests/test_cli_research.py tests/test_research_gates.py -q`
Expected: PASS — existing agent tests using `SyntheticProvider` still pass because it is now marked `reproducible`.

- [ ] **Step 7: Commit**

```bash
git add algua/registry/promotion.py algua/backtest/_sample.py tests/test_promotion.py
git commit -m "fix(205): agents must promote off a reproducible data source

promotion_preflight refuses an agent promote unless the provider is snapshot-backed
or carries a reproducible marker, fired before any provider read; humans exempt.
SyntheticProvider (fixed seed) is marked reproducible so --demo agent promotes and
existing tests keep working.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Documentation/comment cleanup + full gate

**Files:**
- Modify: `algua/registry/db.py` (holdout_evaluations table comment ~133–143)
- Modify: `algua/registry/repository.py` (`reserve_holdout` docstring ~200)

- [ ] **Step 1: Update the schema comment in `db.py`**

In `algua/registry/db.py`, the comment block above `CREATE TABLE ... holdout_evaluations` (~133–143) says the row is refused per "strategy + data identity". Update the wording so it states the single-use key is the OOS interval and provenance is evidence-only. Change the sentence that describes the match (the line mentioning "data identity" / per-provenance) to:

```
-- Single-use key: (strategy_id, OOS interval [holdout_start, holdout_end]) — PROVENANCE-INDEPENDENT
-- (#205). data_source/snapshot_id are recorded as EVIDENCE only, never matched on. A row is REFUSED
-- if its OOS interval overlaps a prior reservation/burn for the strategy (the exact bars, #192),
-- regardless of how the bars were reached.
```

(Match the exact existing comment lines when editing; keep surrounding lines intact.)

- [ ] **Step 2: Update the `reserve_holdout` docstring/comment in `repository.py`**

In `algua/registry/repository.py` (~200), ensure the `reserve_holdout` interface docstring describes the match as "provenance-independent OOS-interval single-use (#205); data_source/snapshot_id are stored as evidence". If the current docstring mentions per-provenance/data-source bucketing, replace that phrasing. If it has no such description, add a one-line note.

- [ ] **Step 3: Run the full quality gate**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: ALL PASS. If any pre-existing test elsewhere encoded the old provenance-bucket behavior (search: `uv run pytest -q -k holdout`), update it to the new provenance-independent expectation and note it in the commit.

- [ ] **Step 4: Commit**

```bash
git add algua/registry/db.py algua/registry/repository.py
git commit -m "docs(205): update holdout-evaluations comments to provenance-independent identity

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review Notes (coverage vs spec)

- Spec Component 1 (provenance-independent match) → Task 1.
- Spec inverted-interval fail-closed guard → Task 1 (Step 3 + test).
- Spec Component 2 (reproducible-source guard, before provider read; SyntheticProvider marker) → Task 2.
- Spec "no schema change, no hashing, SCHEMA_VERSION stays 23" → honored (no `db.py` schema/version edit, only comments).
- Spec testing matrix (cross-provenance block, partial overlap, #192 regression, non-overlap allowed, legacy NULL fail-closed, agent reproducible refusal/allow, concurrency) → Tasks 1–2 cover all except the legacy NULL-interval row, which is unchanged behavior already covered by the existing NULL branch and `test_concurrency`/store tests; the `holdout_start IS NULL` clause is retained verbatim.
- Declined items (agent-accessible corrected-snapshot override; composite index) → not in plan, by design.
