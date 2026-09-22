# Evidence Epoch and Source Normalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the forward gate crediting ticks that predate the current artifact, and stop a comment or reformat resetting every strategy's evidence clock.

**Architecture:** Two independent defect fixes in the identity/evidence path. Task 1 bounds forward evidence to the LAST contiguous run of the current identity, so a revert cannot re-credit an old run. Task 2 normalizes module source through an AST round-trip (dropping comments, formatting and docstrings) before it is hashed into `code_hash`. Neither introduces a schema change; both are prerequisites for, and independent of, the artifact freeze.

**Tech Stack:** Python 3.12, sqlite3, stdlib `ast`, pytest, uv.

**Spec:** `docs/superpowers/specs/2026-09-22-artifact-freeze-design.md` (slice 1 of the Decomposition section)

## Global Constraints

- Drive the system through `uv run algua ...`; never bypass the CLI.
- Quality gate before every commit: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`.
- Keep `algua/contracts` and `algua/features` pure (no I/O, no cross-module imports beyond contracts).
- Line length 100 (ruff).
- `tests/test_module_size_ratchet.py` is shrink-only. If a pinned module grows, raise the pin IN THE SAME COMMIT with a comment saying why.
- Never `git add -A` — stage named paths only.
- Both tasks CHANGE `code_hash`/evidence admission for every strategy. That is expected and already true of the current fleet (PRs #656 and #657 both moved the identity). Do not try to preserve old digests.

---

### Task 1: Bound forward evidence to the current identity's last contiguous run

**Files:**
- Modify: `algua/registry/forward_evidence.py` (`_EXCLUSION_FILTERS` at :57, the row loop at :201-215; the tuple as it stands on `main`, WITHOUT `venue_blocked` — that key lives on the unmerged #655 branch, so rebase before assuming otherwise)
- Test: `tests/test_forward_promotion.py` (existing `seed_tick` helper at :104, `EXCLUSION_KEYS` at :44)

**Interfaces:**
- Consumes: existing `_identity_matches(row, identity)` at `forward_evidence.py:98`.
- Produces: `_epoch_start_id(rows: list[sqlite3.Row], identity: ArtifactIdentity) -> int | None` — the `id` of the first tick in the last contiguous run of `identity`, or `None` when the newest tick is not that identity. Adds `"pre_epoch"` to `_EXCLUSION_FILTERS`.

**Why a contiguous run rather than a stored epoch:** there is no deployment table yet (slice 3 adds one). The run boundary is derivable from data already recorded, needs no migration, and closes the gaming vector directly: reverting code to an earlier identity starts a NEW run, because the intervening ticks under the other identity break the old one.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_forward_promotion.py`:

```python
def test_ticks_before_an_identity_change_are_not_back_credited(conn):
    """Anti-gaming. Ticks are admitted only from the LAST contiguous run of the current identity.

    Without this bound an operator could revert a strategy to an earlier artifact AFTER seeing how
    that artifact's forward period turned out, and the old run would be silently re-credited.
    """
    seed_tick(conn, date(2026, 6, 1), 100.0, code_hash="OLD")
    seed_tick(conn, date(2026, 6, 2), 101.0, code_hash="OLD")
    seed_tick(conn, date(2026, 6, 3), 102.0, code_hash="OTHER")   # the run breaker
    seed_tick(conn, date(2026, 6, 10), 100.0, code_hash="c")
    seed_tick(conn, date(2026, 6, 12), 101.0, code_hash="c")

    res = assemble(conn)   # assembles against identity code_hash="c"
    assert res.evidence.n_return_observations == 1, "only the final run of two sessions counts"


def test_a_revert_does_not_re_credit_the_old_run(conn):
    """The concrete attack: run under identity c, switch away, switch BACK to c."""
    seed_tick(conn, date(2026, 6, 1), 100.0)      # identity "c"
    seed_tick(conn, date(2026, 6, 2), 140.0)      # identity "c" -- a flattering old run
    seed_tick(conn, date(2026, 6, 3), 90.0, code_hash="OTHER")
    seed_tick(conn, date(2026, 6, 10), 100.0)     # back to "c"
    seed_tick(conn, date(2026, 6, 12), 101.0)

    res = assemble(conn)
    assert res.excluded["pre_epoch"] == 2
    assert res.evidence.n_return_observations == 1


def test_a_run_is_not_broken_by_a_bad_tick_of_the_SAME_identity(conn):
    """A local-clock or stale tick is a bad tick, not a different artifact. It must not restart the
    epoch -- only an identity CHANGE does."""
    seed_tick(conn, date(2026, 6, 10), 100.0)
    seed_tick(conn, date(2026, 6, 11), 99.0, clock_source="local")
    seed_tick(conn, date(2026, 6, 12), 101.0)

    res = assemble(conn)
    assert res.excluded["local_clock"] == 1
    assert res.excluded["pre_epoch"] == 0
    assert res.evidence.n_return_observations == 1


def test_no_evidence_when_the_newest_tick_is_a_different_identity(conn):
    """The strategy has been recoded and has not traded since. Nothing may be credited."""
    seed_tick(conn, date(2026, 6, 10), 100.0)
    seed_tick(conn, date(2026, 6, 12), 101.0)
    seed_tick(conn, date(2026, 6, 13), 102.0, code_hash="NEWER")

    res = assemble(conn)
    assert res.evidence.n_return_observations == 0
```

Also add `"pre_epoch"` to `EXCLUSION_KEYS` at `tests/test_forward_promotion.py:44`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_forward_promotion.py -q -k "back_credited or revert or same_identity or newest_tick"`
Expected: FAIL — `KeyError: 'pre_epoch'` and observation counts of 2 or 3 instead of 1/0.

- [ ] **Step 3: Write the implementation**

In `algua/registry/forward_evidence.py`, extend the filter tuple at :57:

```python
_EXCLUSION_FILTERS = ("local_clock", "identity_drift", "legacy_null", "bad_tick_ts",
                      "no_decision", "bad_decision_ts", "stale_decision", "pre_epoch")
```

Add, next to `_identity_matches`:

```python
def _epoch_start_id(rows: list[sqlite3.Row], identity: ArtifactIdentity) -> int | None:
    """The id of the first tick in the LAST contiguous run of `identity`, or None if the newest
    tick is a different identity.

    Evidence may not be back-credited across an identity change. Without this bound, reverting a
    strategy to an earlier artifact AFTER seeing how that artifact's forward period turned out
    would silently re-credit the old run -- choosing the artifact on the strength of the evidence
    it is about to be judged by.

    A run is broken ONLY by a tick of a different identity. A tick that is inadmissible for some
    other reason (a local clock, a stale decision) is a bad tick of the SAME artifact, so it must
    not restart the epoch.
    """
    start: int | None = None
    for row in rows:
        if _identity_matches(row, identity):
            if start is None:
                start = int(row["id"])
        else:
            start = None
    return start
```

Replace the partition loop at :209-215 with:

```python
    epoch_start = _epoch_start_id(rows, identity)
    excluded = dict.fromkeys(_EXCLUSION_FILTERS, 0)
    admissible: list[sqlite3.Row] = []
    for row in rows:
        reason = _inadmissible_reason(row, identity, calendar, now_utc)
        if reason is None and (epoch_start is None or int(row["id"]) < epoch_start):
            # Would have counted, but predates the current artifact's run. Reported separately so
            # the gate's own output shows the bound was applied rather than silently dropping rows.
            reason = "pre_epoch"
        if reason is None:
            admissible.append(row)
        else:
            excluded[reason] += 1
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_forward_promotion.py -q`
Expected: PASS, all tests in the file.

- [ ] **Step 5: Verify the guard catches its own removal**

Temporarily change `reason = "pre_epoch"` to `pass` and re-run the file. Expected: the two back-crediting tests FAIL and nothing else does. Restore the line.

This step exists because on this branch's sibling PRs, tests repeatedly passed for the wrong reason and concealed live bugs. Do not skip it.

- [ ] **Step 6: Run the full gate and commit**

```bash
uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports
git add algua/registry/forward_evidence.py tests/test_forward_promotion.py
git commit -m "fix(evidence): do not back-credit ticks from before the current artifact's run"
```

If `tests/test_module_size_ratchet.py` fails, raise the `algua/registry/forward_evidence.py` pin in the same commit with a one-line reason.

---

### Task 2: Normalize module source before hashing it into `code_hash`

**Files:**
- Modify: `algua/registry/approvals.py` (the `inspect.getsource` call at :102)
- Test: `tests/test_registry_approvals.py`

**Interfaces:**
- Consumes: nothing from Task 1; the tasks are independent.
- Produces: `_normalized_source(module: ModuleType) -> str` in `algua/registry/approvals.py`, replacing the raw `inspect.getsource(module)` at :102. Same return contract: `""` when the source is unavailable.

**Why:** `code_hash` hashes raw source text, so a comment edit, a reflow, or a docstring rewrite invalidates every prior approval and resets every strategy's evidence clock. This repo edits docstrings constantly, and the first-party closure changed on 24 distinct days in 180 — a large share of it cosmetic. An AST round-trip drops comments and formatting; stripping docstrings drops the rest. Neither can change what the code does.

- [ ] **Step 1: Write the failing test**

Create `tests/test_source_normalization.py`:

```python
"""`code_hash` must track BEHAVIOUR, not typography.

It hashes the strategy's first-party source closure. Hashing raw text meant a comment edit, a
reflow or a docstring rewrite invalidated every prior approval and reset every strategy's
forward-evidence clock -- against a gate that needs 250-500 observations under one unchanged
identity. A cosmetic edit cannot change a decision, so it must not move the identity.

The opposite error is worse: normalising away something that DOES change behaviour would let a
real code change slip past the live gate. The second half of this file is that direction.
"""
from __future__ import annotations

from algua.registry.approvals import _strip_cosmetics


def test_a_comment_does_not_change_the_normalized_source():
    assert _strip_cosmetics("x = 1  # why\n") == _strip_cosmetics("x = 1\n")


def test_reformatting_does_not_change_it():
    assert _strip_cosmetics("def f(a,b):\n    return a+b\n") == _strip_cosmetics(
        "def f(\n    a,\n    b,\n):\n    return a + b\n")


def test_a_docstring_rewrite_does_not_change_it():
    a = 'def f():\n    """One thing."""\n    return 1\n'
    b = 'def f():\n    """Something else entirely, at length."""\n    return 1\n'
    assert _strip_cosmetics(a) == _strip_cosmetics(b)


def test_a_module_docstring_rewrite_does_not_change_it():
    assert _strip_cosmetics('"""A."""\nx = 1\n') == _strip_cosmetics('"""B."""\nx = 1\n')


def test_a_docstring_only_function_still_parses():
    """Removing the docstring must not leave an empty body."""
    out = _strip_cosmetics('def f():\n    """Only a docstring."""\n')
    assert "pass" in out


def test_a_changed_literal_DOES_change_it():
    assert _strip_cosmetics("x = 1\n") != _strip_cosmetics("x = 2\n")


def test_a_changed_operator_DOES_change_it():
    assert _strip_cosmetics("y = a + b\n") != _strip_cosmetics("y = a - b\n")


def test_a_renamed_local_DOES_change_it():
    assert _strip_cosmetics("def f(a):\n    return a\n") != _strip_cosmetics(
        "def f(b):\n    return b\n")


def test_a_reordered_statement_DOES_change_it():
    assert _strip_cosmetics("a = 1\nb = 2\n") != _strip_cosmetics("b = 2\na = 1\n")


def test_unparseable_source_is_hashed_raw_rather_than_collapsing():
    """A module that cannot be parsed must not normalize to the empty string -- that would make
    every broken module share one identity."""
    broken = "def f(:\n"
    assert _strip_cosmetics(broken) == broken
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_source_normalization.py -q`
Expected: FAIL with `ImportError: cannot import name '_strip_cosmetics'`.

- [ ] **Step 3: Write the implementation**

In `algua/registry/approvals.py`, add `import ast` to the imports and add:

```python
def _strip_cosmetics(source: str) -> str:
    """Source with comments, formatting and docstrings removed, via an AST round-trip.

    `code_hash` must track BEHAVIOUR, not typography. Hashing raw text meant a comment edit or a
    docstring rewrite reset every strategy's forward-evidence clock, against a gate that needs
    250-500 observations under ONE unchanged identity.

    Comments are absent from the AST, and `ast.unparse` emits canonical formatting, so both vanish.
    Docstrings survive as `Expr(Constant(str))` and are removed explicitly -- they are the most
    frequently edited text in this repo and cannot change a decision.

    Unparseable source is returned RAW rather than normalised to "": collapsing it would give every
    broken module one shared identity.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source
    for node in ast.walk(tree):
        if not isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        body = node.body
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            node.body = body[1:] or [ast.Pass()]
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def _normalized_source(module: ModuleType) -> str:
    """The module's source, normalised. "" when it is unavailable (a namespace package, a module
    built at runtime) -- the pre-existing contract at this call site."""
    try:
        return _strip_cosmetics(inspect.getsource(module))
    except (OSError, TypeError):
        return ""
```

Replace lines :101-104 — the `try: sources[mod_name] = inspect.getsource(module) / except (OSError, TypeError): sources[mod_name] = ""` block — with a single line (the try/except moves into `_normalized_source`):

```python
        sources[mod_name] = _normalized_source(module)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_source_normalization.py -q`
Expected: PASS, 10 tests.

- [ ] **Step 5: Verify normalization did not neuter the identity**

Run: `uv run pytest tests/test_registry_approvals.py tests/test_strategy_overlays_identity.py tests/test_approvals_fundamentals.py -q`

Expected: the pinned no-overlays `config_hash` digest is unaffected (this task changes `code_hash`, not `config_hash`). If any test pins a `code_hash` VALUE, update it and record in the commit that normalization moved it — do not weaken an assertion that a real change moves the hash.

- [ ] **Step 6: Measure what it bought**

Run:

```bash
uv run python -c "
from algua.registry.approvals import compute_artifact_hashes
print(compute_artifact_hashes('cross_sectional_momentum').code_hash)"
```

Record the value in the commit message. Then confirm the win is real:

```bash
git stash && uv run python -c "
from algua.registry.approvals import compute_artifact_hashes
print('before:', compute_artifact_hashes('cross_sectional_momentum').code_hash)"; git stash pop
```

Expected: the two differ (normalization is a one-time identity move, as the Global Constraints say).

- [ ] **Step 7: Run the full gate and commit**

```bash
uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports
git add algua/registry/approvals.py tests/test_source_normalization.py
git commit -m "fix(identity): hash what the code does, not how it is typed"
```

If `tests/test_module_size_ratchet.py` fails, raise the `algua/registry/approvals.py` pin in the same commit with a one-line reason.

---

### Task 3: Protect the identity primitive

**Files:**
- Modify: `CODEOWNERS`
- Modify: `tests/test_repo_hygiene.py` (`INTEGRITY_CRITICAL_MODULES` at :218)

**Interfaces:** none — configuration only.

**Why:** `algua/registry/approvals.py` computes the artifact identity the live gate depends on, and Task 2 just added a normalization step to it. Widening what is normalized away silently widens what may change under a gated strategy — exactly what the executable denylist exists to catch. It is currently unprotected: CODEOWNERS covers `/algua/registry/store/`, which is a different file.

- [ ] **Step 1: Write the failing test**

`tests/test_repo_hygiene.py` already asserts every entry in `INTEGRITY_CRITICAL_MODULES` is denied by the parsed CODEOWNERS. Add the module to that set:

```python
        "algua/registry/approvals.py",
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_repo_hygiene.py -q`
Expected: FAIL — the module is listed as integrity-critical but not CODEOWNERS-protected.

- [ ] **Step 3: Add the owner**

In `CODEOWNERS`, after the `/algua/registry/transitions.py` line:

```
/algua/registry/approvals.py    @Lior-Nis   # computes the artifact identity the live gate verifies
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_repo_hygiene.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add CODEOWNERS tests/test_repo_hygiene.py
git commit -m "chore(codeowners): protect the artifact-identity primitive"
```

---

## After the plan

Open ONE PR for all three tasks, then request a Codex review before merging. State in the PR body that this moves `code_hash` for every strategy (a one-time re-identification, already true of the current fleet) and that `pre_epoch` is a new exclusion key in the gate's output.
