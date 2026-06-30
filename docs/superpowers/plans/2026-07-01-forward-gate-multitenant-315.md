# Forward gate: attribution-cleanliness replaces single-tenant (#315) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Relax the forward gate so multiple attributed strategies can co-tenant one paper account and each still certify — by making fill attribution account-level and replacing the `single_tenant` (no-siblings) check with `single_account` (one account per strategy's evidence).

**Architecture:** Two surgical edits to the CODEOWNERS-protected gate: (1) `_classify_activities` attributes a FILL to ANY recorded paper-venue order (not this strategy's); (2) `single_tenant_ok` (distinct-account AND no-siblings) becomes `single_account_ok` (distinct-account only). Each strategy's return series stays clean by construction (its own per-strategy NAV ticks).

**Tech Stack:** Python 3.12, SQLite (`paper_venue_orders`, `tick_snapshots`), pytest.

## Global Constraints

- Run via `uv run ...`. Full gate green before every commit: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`.
- Ruff ≤ 100 columns.
- `git add` only the named files — never `git add -A`.
- Commit trailer: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- **CODEOWNERS-protected files** (`algua/registry/forward_promotion.py`, `algua/research/forward_gates.py`): change ONLY what the relaxation requires; touch no other gate criterion (`reconcile_ok`, the Sharpe bar, observations/coverage/vol/drawdown/staleness, identity, qualified-holdout). No `n_concurrent_forward` deflation (deferred).
- Existing single-tenant tests are **re-expressed** to the new semantics (sibling → pass, mixed-account → fail), never weakened.
- Branch: `forward-gate-multitenant-315` (off main). Merge AFTER #322/#323; CODEOWNERS-authorized.

---

## File structure

- `algua/registry/forward_promotion.py` — **Modify.** `_classify_activities` (drop `strategy_id`, account-level query) + both call sites; the `single_tenant_ok` computation → `single_account_ok`; the `ForwardEvidence(...)` construction kwarg.
- `algua/research/forward_gates.py` — **Modify.** `ForwardEvidence.single_tenant_ok` → `single_account_ok`; the `single_tenant` check → `single_account` (incl. the emitted decision key + message).
- `tests/test_forward_promotion.py`, `tests/test_forward_gates.py` — **Modify.** Account-level attribution tests; sibling-passes / mixed-account-fails; decision-key rename.

---

### Task 1: Account-level fill attribution in `_classify_activities`

**Files:**
- Modify: `algua/registry/forward_promotion.py`
- Test: `tests/test_forward_promotion.py`

**Interfaces:**
- Produces: `_classify_activities(conn, acts: list[dict[str, Any]]) -> tuple[int, int]` (the `strategy_id` param is removed). A FILL is attributable iff its `order_id` matches **any** row in `paper_venue_orders` by `broker_order_id`; unmatched/missing → unattributable. Both call sites (evidence assembly ~318, certificate re-verify ~460) drop the `strategy_id` arg.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_forward_promotion.py` (import the private helper directly):

```python
def test_classify_activities_attributes_fill_to_any_paper_order(tmp_path):
    from algua.registry.forward_promotion import _classify_activities
    from algua.registry.db import connect, migrate
    from algua.execution.live_ledger import record_paper_venue_order

    conn = connect(tmp_path / "r.db")
    migrate(conn)
    conn.execute("INSERT INTO strategies(name, stage, created_at, updated_at) VALUES "
                 "('sib','paper','2026-01-01','2026-01-01')")
    conn.commit()
    # a SIBLING strategy's order owns broker order id 'bo-7'
    record_paper_venue_order(conn, "sib", "AAA", "buy", None, "coid-sib", strategy_id=1)
    conn.execute("UPDATE paper_venue_orders SET broker_order_id='bo-7' WHERE client_order_id='coid-sib'")
    conn.commit()

    acts = [
        {"activity_type": "FILL", "order_id": "bo-7"},      # sibling's fill -> ATTRIBUTABLE
        {"activity_type": "FILL", "order_id": "bo-orphan"}, # matches no recorded order -> unattributable
        {"activity_type": "FILL", "order_id": None},        # missing id -> unattributable (fail closed)
        {"activity_type": "DIV", "order_id": None},         # non-fill -> ignored
    ]
    n_external, n_unattributable = _classify_activities(conn, acts)
    assert n_external == 0
    assert n_unattributable == 2     # only the orphan + the null-id fill
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_forward_promotion.py::test_classify_activities_attributes_fill_to_any_paper_order -v`
Expected: FAIL — `_classify_activities()` still takes `strategy_id` (TypeError: missing positional arg) / attributes only this-strategy fills.

- [ ] **Step 3: Make attribution account-level**

In `algua/registry/forward_promotion.py`, change the `_classify_activities` definition (drop `strategy_id`; query any paper-venue order):

```python
def _classify_activities(
    conn: sqlite3.Connection, acts: list[dict[str, Any]],
) -> tuple[int, int]:
    """(n_external_cash_flows, n_unattributable_fills) over raw broker activities. External capital
    types always count; a FILL is attributable iff it reconciles to SOME recorded paper-venue order
    by broker order id (account-level — any current paper-book strategy's order; a missing or
    unmatched order_id is unattributable, fail closed). Everything else (DIV/INT/FEE/...) passes.
    Shared with the certificate re-verification path."""
    n_external = 0
    n_unattributable = 0
    for act in acts:
        activity_type = act.get("activity_type")
        if activity_type in EXTERNAL_CAPITAL_TYPES:
            n_external += 1
        elif activity_type == "FILL":
            order_id = act.get("order_id")
            matched = order_id is not None and conn.execute(
                "SELECT 1 FROM paper_venue_orders WHERE broker_order_id = ?",
                (order_id,),
            ).fetchone() is not None
            if not matched:
                n_unattributable += 1
    return n_external, n_unattributable
```

Then update the **two call sites** to drop the `strategy_id` argument:
- the evidence-assembly call (currently `n_external_cash_flows, n_unattributable_fills = _classify_activities(conn, strategy_id, acts)`) → `_classify_activities(conn, acts)`;
- the certificate re-verify call (currently `n_external, n_unattributable = _classify_activities(conn, strategy_id, acts)`) → `_classify_activities(conn, acts)`.

- [ ] **Step 4: Run test + regression**

Run: `uv run pytest tests/test_forward_promotion.py tests/test_forward_gates.py -v`
Expected: the new test passes; existing tests still pass (account-level attribution is behavior-identical for a single strategy — its own fills match either way). If an existing test referenced `_classify_activities(conn, strategy_id, acts)`, update the call to drop `strategy_id` (mechanical).

- [ ] **Step 5: Commit**

```bash
git add algua/registry/forward_promotion.py tests/test_forward_promotion.py
git commit -m "feat(gate): account-level fill attribution in _classify_activities #315

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: `single_tenant_ok` → `single_account_ok` (drop the no-siblings block)

**Files:**
- Modify: `algua/research/forward_gates.py` (the `ForwardEvidence` field + the gate check), `algua/registry/forward_promotion.py` (the computation + the evidence construction)
- Test: `tests/test_forward_promotion.py`, `tests/test_forward_gates.py`

**Interfaces:**
- Consumes: account-level `_classify_activities` (Task 1).
- Produces: `ForwardEvidence.single_account_ok: bool` (renamed from `single_tenant_ok`); the gate emits a check keyed `single_account`; `single_account_ok` is True iff the strategy's admissible ticks all share one `account_id` (siblings on that account no longer fail it).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_forward_promotion.py` (use the existing fixtures that build admissible `tick_snapshots`; seed a sibling tick on the same account, and a second test with two account_ids). Mirror the file's existing evidence-assembly test style:

```python
def test_sibling_on_same_account_now_passes_single_account(...):
    # Build a window of admissible paper ticks for strategy A on account 'acct-1'; ALSO record a
    # sibling strategy B's paper tick on 'acct-1' in the same window. A's evidence must have
    # single_account_ok True (the old single_tenant would have been False).
    ...
    assembled = assemble_forward_evidence(...)  # the existing assembly entrypoint
    assert assembled.evidence.single_account_ok is True


def test_mixed_account_evidence_fails_single_account(...):
    # A's admissible ticks span 'acct-1' and 'acct-2' -> single_account_ok False.
    ...
    assert assembled.evidence.single_account_ok is False
```

Add to `tests/test_forward_gates.py`:

```python
def test_single_account_check_key_and_failure_message(...):
    # An evidence with single_account_ok=False produces a failed check keyed "single_account".
    ev = _evidence(single_account_ok=False, ...)  # the file's evidence factory
    decision = evaluate_forward_gate(ev, criteria, ...)
    failed = {c["name"] for c in decision.checks if not c["passed"]}
    assert "single_account" in failed
    assert "single_tenant" not in {c["name"] for c in decision.checks}
```

> Implementer note: use the existing evidence factory / assembly helpers in those test files; if a
> helper passes `single_tenant_ok=`, rename the kwarg to `single_account_ok=`. Keep the assertions
> above intact (sibling→pass, mixed→fail, key renamed).

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_forward_promotion.py tests/test_forward_gates.py -k "single_account" -v`
Expected: FAIL — `ForwardEvidence` has no `single_account_ok`; the check is still keyed `single_tenant`.

- [ ] **Step 3: Rename the evidence field + the gate check**

In `algua/research/forward_gates.py`:
- In `ForwardEvidence`, rename the field `single_tenant_ok: bool` → `single_account_ok: bool`.
- In `evaluate_forward_gate`, replace the `single_tenant` check block:

```python
    checks.append(_bool_check(
        "single_account", evidence.single_account_ok,
        "evidence spans more than one paper account in the window"))
```

(Update the section-7 comment from "single-tenant account" to "single-account evidence; siblings on the same account are allowed".)

- [ ] **Step 4: Compute `single_account_ok` (drop the sibling query)**

In `algua/registry/forward_promotion.py`, replace the `single_tenant_ok` computation block:

```python
    account_id: str | None = None
    single_account_ok = True
    if admissible:
        account_id = admissible[-1]["account_id"]
        # A strategy's evidence must come from ONE account (mixed-account evidence is a tenancy
        # violation). Siblings on the same account are ALLOWED: the multi-tenant book attributes
        # each strategy's return series via its own per-strategy NAV ticks (#314/#316a-b) and
        # account-level fill attribution (_classify_activities), so a sibling cannot contaminate it.
        single_account_ok = len({row["account_id"] for row in admissible}) == 1
```

Then in the `ForwardEvidence(...)` construction, change `single_tenant_ok=single_tenant_ok` →
`single_account_ok=single_account_ok`. (The `account_id` local is still used downstream — leave it.)

- [ ] **Step 5: Run the new tests + full gate suite**

Run: `uv run pytest tests/test_forward_promotion.py tests/test_forward_gates.py -v`
Expected: the new `single_account` tests pass; existing tests pass after the faithful updates (any test asserting a sibling FAILS the gate is re-expressed to PASS; mixed-account still fails; the kwarg/key rename applied). Do NOT weaken an assertion to make it pass.

- [ ] **Step 6: Full quality gate**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: all green. (Grep `grep -rn single_tenant algua tests` — there should be no remaining `single_tenant` references in code/tests after the rename; any in `kb/` docs are out of scope / a follow-up.)

- [ ] **Step 7: Commit**

```bash
git add algua/research/forward_gates.py algua/registry/forward_promotion.py tests/test_forward_promotion.py tests/test_forward_gates.py
git commit -m "feat(gate): single_account replaces single_tenant — allow co-tenant forward evidence #315

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-review

- **Spec coverage:** §2(a) `single_tenant`→`single_account` (field + check + computation + key) → Task 2; §2(b) account-level `_classify_activities` + both call sites → Task 1; §3 testing (sibling-passes, mixed-fails, sibling-fill-attributable, orphan-fill-fails, key rename) → Tasks 1–2; §4 non-goals respected (no concurrency deflation, no other criterion touched); §5 risk (protected surface, merge order) reflected in Global Constraints.
- **Placeholder scan:** the Task-2 test bodies use the file's existing fixtures (`...` = the established assembly/evidence helpers); the assertions are concrete. The `_classify_activities` test (Task 1) is fully self-contained. No code-step placeholders.
- **Type consistency:** `_classify_activities(conn, acts) -> tuple[int, int]` (Task 1) called without `strategy_id` at both sites; `ForwardEvidence.single_account_ok` (Task 2) defined in forward_gates.py, built in forward_promotion.py, read in the gate check — name consistent across all three; decision key `single_account` matches the test assertion.
