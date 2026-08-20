# Stage 4b — `registry/store.py` Per-Protocol Carve Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Carve the ~2170-line `algua/registry/store.py` monolith (the single concrete implementation of all 8 role Protocols declared in `algua/registry/repository.py`) into `algua/registry/store/` — one mixin file per Protocol, a shared base for the one genuinely cross-Protocol helper, a tiny shared-constant module, and a facade `__init__.py` that composes them into the same `SqliteStrategyRepository` class at the same import path. Zero behavior change: this is a pure structural move, mirroring Stage 3's `algua/data/store.py` -> `algua/data/store/` carve exactly.

**Architecture:** `algua/registry/repository.py` (921 lines, untouched by this plan) already declares the target shape: 8 role Protocols (`StrategyStore` — itself composing `StrategyReader`+`StrategyLister` — `ApprovalLedger`, `SearchBreadthLedger`, `HoldoutLedger`, `GateLedger`, `ForwardGateLedger`, `FamilyGraph`, `BacktestReturnsLedger`) plus 2 pure-union composition Protocols that need no implementation file of their own. `SqliteStrategyRepository` in `store.py` is the one class implementing all 8. This plan splits that class into 8 domain mixins + 1 shared base mixin (for the one method — `_apply_transition_locked` — genuinely called from more than one domain) + 1 tiny module for the one genuinely cross-file plain helper (`_now`), composed by a facade `__init__.py` exactly like Stage 3's `DataStore`. Every mixin independently declares its own `_conn: sqlite3.Connection` class-level type annotation (never assigned in the mixin — the facade's `__init__` owns the real assignment), mirroring how Stage 3's dataset mixins each independently declare `data_dir`/`manifest`/`_staging`. Two methods that call into OTHER Protocols' domains from inside their own body (`_verify_funnel_snapshot`, `_mint_agent_novel_family` — both private, both called only from `GateLedger`'s `record_gate_with_fdr_and_maybe_promote`) stay in `gate.py` as GateLedger's own helpers, using the same stub-for-mypy pattern Stage 3 already proved (`bars.py`'s `get_snapshot` stub) rather than being pulled into the shared base — their caller determines their home, not their dependencies.

**Tech Stack:** Python 3.12, uv, pytest, ruff, mypy, import-linter, sqlite3.

**Spec:** `docs/superpowers/specs/2026-08-18-system-simplification-design.md` §8 ("registry/store per-Protocol carve (cut list = `repository.py` Protocols)").

**Ground truth this plan is written from:** a research pass against `main`@`be87e4b` (post-4a) that read `algua/registry/repository.py` and `algua/registry/store.py` in full, produced an exact method-to-Protocol map with line ranges, confirmed the cross-Protocol coupling shape, confirmed zero import-linter contract edit is needed, confirmed all 48 dependent test files import only the public `SqliteStrategyRepository` (+ one constant, `AGENT_NOVEL_MINT_CAP`) with zero private-symbol coupling, and — the single most consequential finding — that `/CODEOWNERS` protects the EXACT file glob `/algua/registry/store.py`, which silently stops matching anything the moment `store.py` becomes a directory. That research doc is not itself part of this repo; its findings are folded into this plan directly. Two design questions it flagged as open were resolved by the plan's author (this document), not left to the implementer:
1. Where `_verify_funnel_snapshot`/`_mint_agent_novel_family` live: **resolved to `gate.py`** (their sole caller's file), using cross-file `self.*` stub calls for the Protocols they read — mirrors Stage 3's proven pattern, lowest risk, keeps ownership aligned with "who calls this."
2. Whether to collapse `FdrGateOutcome`'s 13 write-only-and-never-read fields (of 16 total — confirmed independently in this plan's research; production reads only `.final_passed`, tests additionally read `.updated_rec`/`.gate_id`): **deferred to a future small follow-up, NOT this plan.** Collapsing it requires editing `repository.py`'s `GateLedger` Protocol declaration, which this carve deliberately leaves untouched — mixing an API-shrinking cleanup into a "zero behavior change" structural carve would blur the exact invariant that made Stage 3 and Stage 4a's reviews fast and clean-verdict. `repository.py` is **not touched anywhere in this plan.**

## Global Constraints

- Quality gate on EVERY task before commit: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`. All four must pass.
- **`algua/registry/repository.py` is NOT touched anywhere in this plan.** This carve moves the ~2170-line *implementation* file; the ~920-line *interface* file (the Protocol declarations, `FdrGateOutcome`/`FdrStreamState`/`FunnelSnapshot`/etc. value types, `ArtifactIdentity`, the exceptions) stays exactly as it is. Every import of these types from `algua.registry.repository` inside the new package must be preserved unchanged — do not move any of them.
- **`SqliteStrategyRepository`'s public import path does not change.** `from algua.registry.store import SqliteStrategyRepository` (and `AGENT_NOVEL_MINT_CAP`, `StrategyExists`, `StrategyNotFound`, `StrategyRecord`) must keep working identically from the new `algua/registry/store/__init__.py` facade. No caller, anywhere in `algua/` or `tests/`, needs to change its import statement.
- **Zero behavior change.** Every method's body, SQL text, docstring, and comment moves verbatim (byte-for-byte) into its new home — this is code motion, not a rewrite. The only new code is: the 9 files' import blocks, `_conn: sqlite3.Connection` annotations, the `base.py` shared mixin, the 8 mypy stub declarations in `gate.py` + 1 in `family.py` (see Task 1 Step 8/10), and the facade `__init__.py`.
- **CODEOWNERS must be updated in the SAME commit that creates the package** (Task 1), not deferred. `/CODEOWNERS` currently has `/algua/registry/store.py        @Lior-Nis   # the paper->live wall + approvals` — an exact single-file glob. The moment `store.py` becomes a directory, this line silently stops matching anything, and every new file loses CODEOWNERS protection with no error or warning. It must change to the directory glob `/algua/registry/store/`. `algua/operator/diff_policy.py` parses `CODEOWNERS` dynamically at runtime (confirmed: it does not hardcode any path), so fixing this one line is sufficient — no code change needed there. (`diff_policy.py`'s own docstring uses `algua/registry/store.py` as an illustrative example in prose; this becomes a stale example after the carve but is not a functional bug — leave it for a future doc pass, do not fix it in this plan.)
- `git add`/`git mv` is always scoped to the named files — never `git add -A`.
- Commits end with: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- Known pre-existing worktree hazard: some test writes a demo strategy file into the real `algua/strategies/momentum/` directory as a side effect. If `git status` shows an untracked file there after running tests, delete it before staging — don't commit it.
- **Process rule from prior stages' execution — read this before dispatching or resuming any subagent**: if a background command (e.g. the full test suite, ~7-8 minutes on this repo) is started, the implementer/reviewer MUST actively poll/re-check its own output with real tool calls in a loop. There is no notification that wakes a dispatched subagent when a background command finishes — ending a turn to "wait" for one stalls the task indefinitely until manually resumed. This has now happened in every single prior stage's execution (0-2, 3, and — worse — literally every implementer subagent dispatched on Stage 4a, 5 for 5), including to implementers whose dispatch prompt already contained this exact warning. Budget for needing a manual resume on almost every dispatch. The fix that has worked: before resuming, independently confirm via `ps aux`/`kill -0 <pid>` that the pytest process has actually exited, THEN resume with a message stating that fact — resuming while it's still running just produces a second stall.
- No import-linter contract needs to change: no contract in `pyproject.toml` names `algua.registry.store` or `algua.registry.repository` specifically — all matching contracts use the broader `algua.registry` prefix, and converting `store.py` into `store/` (a package) does not cross any contract boundary since both live under that same prefix.
- Every new mixin file determines its own import block from what it actually uses — this plan does not hand-enumerate every file's imports (unlike the exact code given for genuinely new content); `ruff`'s unused-import (F401) and undefined-name checks plus `mypy` are the correctness proof that each file's imports are complete and minimal, exactly as they would be for a human engineer doing this move.

---

### Task 1: Carve `algua/registry/store.py` into `algua/registry/store/`

**Files:**
- Create: `algua/registry/store/__init__.py`, `algua/registry/store/_util.py`, `algua/registry/store/base.py`, `algua/registry/store/crud.py`, `algua/registry/store/approvals.py`, `algua/registry/store/search_breadth.py`, `algua/registry/store/holdout.py`, `algua/registry/store/gate.py`, `algua/registry/store/forward_gate.py`, `algua/registry/store/family.py`, `algua/registry/store/backtest_returns.py`
- Delete: `algua/registry/store.py`
- Modify: `CODEOWNERS`

**Interfaces:**
- Produces: `algua.registry.store.SqliteStrategyRepository` (same public class, same methods, same signatures — composed from 8 domain mixins + the shared `base.TransitionMixin`), `algua.registry.store.AGENT_NOVEL_MINT_CAP`, `algua.registry.store.StrategyExists`, `algua.registry.store.StrategyNotFound`, `algua.registry.store.StrategyRecord` (all re-exported at the facade, matching the current `__all__`).
- Consumes: nothing from earlier tasks (first task). Everything it needs already exists in `algua/registry/repository.py`, `algua/registry/db.py`, `algua/registry/metadata.py`, `algua/contracts/*`, `algua/research/gates.py` — all untouched, all still imported the same way from the new files.

All line numbers below are a snapshot of `algua/registry/store.py` as of this plan's writing (`main`@`be87e4b`) — re-verify against the live file before moving anything; if a line number is off by a few lines, trust the method/function NAME and its surrounding content over the exact number.

- [ ] **Step 1: Read the whole file first**

Read `algua/registry/store.py` (2169 lines) and `algua/registry/repository.py` (921 lines) in full before making any edit. Read `algua/data/store/__init__.py` and `algua/data/store/bars.py` too — these are the ALREADY-PROVEN pattern this task mirrors exactly (mixins declare shared state as class-level type annotations only, never assigned; a mixin that needs a method living elsewhere in the composed class declares a one-line `raise NotImplementedError` stub with a comment identifying where the real implementation lives; the facade's `__init__` owns all real state assignment).

- [ ] **Step 2: Create `algua/registry/store/_util.py`**

One shared plain function, used by 7 of the 8 domain mixins (everything except `search_breadth.py`, `holdout.py`... actually verify via grep: `grep -n "_now()" algua/registry/store.py` before moving, to confirm every call site — do not trust this list without re-checking, but as of this plan's writing `_now` is called from `crud.py`'s future home (add, update_metadata, apply_transition, intake_candidate_to_paper), `approvals.py`'s (record_approval), `holdout.py`'s (reserve_holdout, finalize_holdout_reservation), `gate.py`'s (record_gate_evaluation, record_gate_with_fdr_and_maybe_promote), `forward_gate.py`'s (record_forward_gate_evaluation, record_forward_pass_and_promote, `_insert_forward_gate_row_locked`), `backtest_returns.py`'s (persist_backtest_returns), `family.py`'s (create_family, assign_strategy_to_family, add_parent_edge, materialise_legacy_member_profiles, `_mint_agent_novel_family`)):

```python
"""Shared plain helpers for the registry/store package — no state, no self, safe to import
from any mixin file with zero cross-mixin coupling."""
from __future__ import annotations

from datetime import UTC, datetime


def _now() -> str:
    return datetime.now(UTC).isoformat()
```

`_row_to_record` (currently store.py:94-107) is used ONLY by `get`/`list_strategies` (both StrategyStore) — move it into `crud.py` directly, not here. `_pool_trial_sharpe_var`/`_validated_triples` (currently 52-77) are used ONLY by `pooled_trial_sharpe_var`/`funnel_trial_sharpe_var` (both SearchBreadthLedger) — move them into `search_breadth.py` directly. `_parse_canonical_utc` (currently 84-91) is used ONLY by `agent_novel_mint_audit`/`check_agent_novel_mint_bounds` (both FamilyGraph) — move it into `family.py` directly. Verify each of these three "single-file-only" claims yourself with a grep before committing to the placement (`grep -n "_row_to_record(\|_pool_trial_sharpe_var(\|_validated_triples(\|_parse_canonical_utc(" algua/registry/store.py`) — if a call site turns up somewhere unexpected, move that helper to `_util.py` instead and say so in your report.

- [ ] **Step 3: Create `algua/registry/store/base.py`**

The ONE shared mixin, holding the ONE method genuinely called from more than one domain: `_apply_transition_locked`. Verify its exact current call sites first (`grep -n "_apply_transition_locked(" algua/registry/store.py`) — as of this plan's writing there are exactly 3: `apply_transition` (StrategyStore/`crud.py`), `intake_candidate_to_paper` (no-Protocol/`crud.py`), `record_forward_pass_and_promote` (ForwardGateLedger/`forward_gate.py`).

The method's last line, `return self.get(rec.name)`, calls `get` — a `StrategyStore`/`StrategyReader` method that lives in `crud.py`, not here. This makes `TransitionMixin` itself a cross-file caller: add ONE mypy stub for `get` (real signature from `repository.py:258-263`), same pattern as Task 1 Step 8/10 — this is a finding from this plan's own drafting, not previously flagged.

The exact current text (re-verify against the live file — this is copied verbatim from `store.py:477-589` as of this plan's writing):

```python
"""Shared base mixin for algua/registry/store/ — holds the one private helper genuinely called
from more than one Protocol's domain (crud.py's apply_transition + intake_candidate_to_paper,
forward_gate.py's record_forward_pass_and_promote). Every domain mixin that calls it declares a
mypy-only stub (mirrors algua/data/store/bars.py's get_snapshot stub pattern); the real
implementation here is the one that runs at composition time via the facade's MRO."""
from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta

from algua.contracts.lifecycle import Actor, Stage, TransitionError
from algua.contracts.types import PendingLiveAuthorization
from algua.registry.repository import StrategyRecord


class TransitionMixin:
    _conn: sqlite3.Connection

    def get(self, name: str) -> StrategyRecord:
        # provided by crud.py's CrudMixin (StrategyReader); stub for mypy only
        raise NotImplementedError

    def _apply_transition_locked(
        self,
        rec: StrategyRecord,
        to: Stage,
        actor: Actor,
        reason: str | None,
        code_hash: str | None,
        config_hash: str | None,
        dependency_hash: str | None,
        consume_gate_id: int | None,
        consume_forward_gate_id: int | None,
        now: str,
        *,
        revoke_allocation: bool = False,
        live_authorization: PendingLiveAuthorization | None = None,
    ) -> StrategyRecord:
        """``apply_transition``'s body, WITHOUT opening a transaction: the caller owns the
        ``with self._conn:`` scope, so a composite write (e.g.
        ``record_forward_pass_and_promote``) can put extra statements in the SAME transaction
        as the token consume + stage CAS + transition INSERT."""
        from_stage = rec.stage
        if live_authorization is not None:
            # Go-live signature path (#254): consume the challenge and write the live_authorizations
            # row in THIS transaction with the stage CAS below, so a raced/failed transition rolls
            # back BOTH — never burning the nonce or leaving an orphan authorization. The signature
            # was already verified (no DB writes) in live_gate.verify_pending. We deliberately do
            # NOT store the challenge text — trade-time verification rebuilds the signed payload
            # from the recomputed identity, never from agent-writable bytes (codex CRITICAL).
            # The consume re-asserts the FULL pending-challenge predicate (strategy + recomputed
            # identity + unexpired + unconsumed) at consume time — mirroring the forward-gate
            # consume — so a signature verified just before expiry, or against a drifted identity,
            # cannot be applied here (closing the validate-then-consume gap; codex #254 review).
            cur = self._conn.execute(
                "UPDATE live_challenges SET consumed_at=?"
                " WHERE nonce=? AND consumed_at IS NULL AND strategy_id=?"
                " AND code_hash=? AND config_hash=? AND dependency_hash IS ? AND expires_at > ?",
                (now, live_authorization.nonce, rec.id, code_hash, config_hash, dependency_hash,
                 now),
            )
            if cur.rowcount != 1:
                raise TransitionError(
                    "go-live challenge is not consumable for this strategy+identity (already "
                    "consumed, missing, identity-drifted, or expired); request a fresh challenge "
                    "and re-sign")
            self._conn.execute(
                "INSERT INTO live_authorizations(strategy_id, code_hash, config_hash,"
                " dependency_hash, nonce, expires_at, signature, principal, authorized_at)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                (rec.id, code_hash, config_hash, dependency_hash, live_authorization.nonce,
                 live_authorization.expires_at, live_authorization.signature_b64,
                 live_authorization.principal, now),
            )
        if consume_gate_id is not None:
            # Single-use, atomic with the stage change: flipping the token, the stage UPDATE,
            # and the transition INSERT all live in this one transaction. If the token row was
            # already consumed or is missing, raise so the whole transition rolls back — the
            # stage can never advance on a vanished token, nor a token be spent without the
            # stage advancing.
            # NOTE: this UPDATE does NOT re-check artifact identity (code/config/dependency) —
            # identity binding lives in `find_consumable_gate_evaluation`, so callers MUST
            # always pair find->consume and never pass a hand-held id.
            cur = self._conn.execute(
                "UPDATE gate_evaluations SET consumed=1"
                " WHERE id=? AND strategy_id=? AND passed=1 AND actor='agent' AND consumed=0",
                (consume_gate_id, rec.id))
            if cur.rowcount != 1:
                raise TransitionError(
                    f"gate evaluation {consume_gate_id} is not a consumable agent token for "
                    f"this strategy (already consumed, missing, or mismatched)")
        if consume_forward_gate_id is not None:
            # Single-use, atomic with the stage change — same shape as the shortlist consume
            # above, EXCEPT it deliberately does NOT copy that block's lookup-trust: the WHERE
            # re-checks the FULL predicate set (identity, actor, passed, unconsumed, TTL) at
            # consume time, closing the validate-then-consume gap. The caller passes the
            # RECOMPUTED identity through code_hash/config_hash/dependency_hash; a NULL
            # dependency_hash never matches (fail-closed, mirroring has_valid_approval).
            from algua.research.forward_gates import FORWARD_TOKEN_TTL_DAYS
            cutoff = (datetime.now(UTC) - timedelta(days=FORWARD_TOKEN_TTL_DAYS)).isoformat()
            cur = self._conn.execute(
                "UPDATE forward_gate_evaluations SET consumed=1"
                " WHERE id=? AND strategy_id=? AND passed=1 AND actor='agent' AND consumed=0"
                " AND code_hash=? AND config_hash=? AND dependency_hash=? AND created_at>=?",
                (consume_forward_gate_id, rec.id, code_hash, config_hash,
                 dependency_hash, cutoff))
            if cur.rowcount != 1:
                raise TransitionError(
                    f"forward gate evaluation {consume_forward_gate_id} is not a consumable"
                    " agent token for this strategy+identity (already consumed, missing,"
                    " identity-drifted, or expired)")
        if revoke_allocation:
            # Bench wind-down (#125): revoke the live capital reservation in the SAME transaction
            # as the stage CAS below, so a raced/failed transition leaves the allocation intact.
            from algua.registry import allocations
            allocations.revoke_active_locked(self._conn, rec.id)
        # Compare-and-swap on the stage the caller read: two sessions sharing this DB must not
        # silently overwrite each other's transitions. Inside the txn, so a raced transition
        # rolls back the token consume above too.
        cur = self._conn.execute(
            "UPDATE strategies SET stage = ?, updated_at = ? WHERE id = ? AND stage = ?",
            (to.value, now, rec.id, from_stage.value),
        )
        if cur.rowcount != 1:
            raise TransitionError(
                f"concurrent transition detected for {rec.name!r}: stage is no longer"
                f" {from_stage.value!r} (another session moved it); re-read and retry")
        self._conn.execute(
            "INSERT INTO stage_transitions"
            "(strategy_id, from_stage, to_stage, actor, reason, code_hash, config_hash,"
            " dependency_hash, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (rec.id, from_stage.value, to.value, actor.value, reason,
             code_hash, config_hash, dependency_hash, now),
        )
        return self.get(rec.name)
```

Re-verify this text against the live `store.py:477-589` before committing — this plan's copy is a snapshot; the live file is authoritative if they've diverged. The `get` stub above is new code this plan adds (it did not exist as a stub before — `_apply_transition_locked` previously called `self.get` as an ordinary same-class method, which worked with zero stub because everything was one class); do not skip it, mypy will fail on `self.get(rec.name)` without it since `TransitionMixin` alone doesn't define `get`.

- [ ] **Step 4: Create `algua/registry/store/crud.py`**

`StrategyStore` mixin (composes `StrategyReader`+`StrategyLister` per `repository.py:284-389`). Class must inherit from `base.TransitionMixin` (`class CrudMixin(TransitionMixin):`) since it calls `_apply_transition_locked` directly (real inheritance, not a stub — mirrors `algua/data/store/bars_streamed.py`'s `class BarsStreamedStoreMixin(BarsStoreMixin):`).

Move verbatim (exact current line ranges, re-verify against the live file): `add` (124-158), `get` (160-166), `update_metadata` (168-211), `list_strategies` (213-250), `backfill_metadata` (252-284), `default_fill_metadata_nulls` (286-301), `delete` (303-311), `list_transitions` (313-318), `apply_transition` (320-395), `intake_candidate_to_paper` (397-447 — implements NO Protocol; placed here because it shares `apply_transition`'s shape and its only caller, `algua/cli/paper_cmd.py:518`, references the concrete `SqliteStrategyRepository`, not any Protocol type), `_assert_flat_for_bench` (449-475 — private, its only caller is `apply_transition`, both in this same file, so it needs no stub or inheritance trick — just a normal private method).

Move `_row_to_record` (94-107, per Step 2's confirmation) here too, as a module-level function (not a method — it takes `row: sqlite3.Row` as its only argument and doesn't need `self`).

`_conn: sqlite3.Connection` class-level annotation required (inherited from `TransitionMixin`, but since Stage 3's proven pattern has every mixin independently declare its own state annotations rather than relying on inheritance for that, also declare it directly on `CrudMixin` for clarity — a redundant-but-harmless annotation, not a redundant assignment).

- [ ] **Step 5: Create `algua/registry/store/approvals.py`**

`ApprovalLedger` mixin, class name `ApprovalLedgerMixin` (must match this exactly — the facade's `__init__.py` in Step 12 imports this exact name). Move verbatim: `record_approval` (591-608), `has_valid_approval` (610-628). `_conn: sqlite3.Connection` annotation.

- [ ] **Step 6: Create `algua/registry/store/search_breadth.py`**

`SearchBreadthLedger` mixin, class name `SearchBreadthLedgerMixin` (must match exactly — Step 12's facade imports this name). Move verbatim: `record_search_trial` (630-652), `pooled_trial_sharpe_var` (654-669), `total_search_combos` (671-677), `funnel_lifetime_search_combos` (679-689), `search_trials_fingerprint` (722-729), `windowed_search_combos` (950-959), `funnel_trial_sharpe_var` (961-996). Move `_pool_trial_sharpe_var` (52-63) and `_validated_triples` (66-77) here too as module-level functions (per Step 2's confirmation). `_conn: sqlite3.Connection` annotation.

Note the line-number gap: `family_graph_fingerprint` (691-720) sits BETWEEN two SearchBreadthLedger methods in the current file but belongs to `FamilyGraph`, not `SearchBreadthLedger` — do not move it here, it goes in `family.py` (Step 10). Method order in the new files does not need to match the old file's order; group by Protocol, not by position.

- [ ] **Step 7: Create `algua/registry/store/holdout.py`**

`HoldoutLedger` mixin, class name `HoldoutLedgerMixin` (must match exactly — Step 12's facade imports this name). Move verbatim: `reserve_holdout` (731-802), `finalize_holdout_reservation` (804-819), `release_holdout_reservation` (821-828), `record_holdout_returns` (830-906), `overlapping_holdout_return_streams` (908-948). `_conn: sqlite3.Connection` annotation.

- [ ] **Step 8: Create `algua/registry/store/gate.py`**

`GateLedger` mixin, class name `GateLedgerMixin` (must match exactly — Step 12's facade imports this name) — the most complex file in this carve. Move verbatim: `record_gate_evaluation` (998-1049), `find_consumable_gate_evaluation` (1051-1070), `fdr_stream_state` (1270-1376), `_cas_funnel` (1378-1386, staticmethod, private, its only caller `_verify_funnel_snapshot` is also moving here), `_verify_funnel_snapshot` (1388-1432), `record_gate_with_fdr_and_maybe_promote` (1434-1615), `passing_gate_by_token` (1617-1636 — implements NO Protocol, placed here since it reads the same `gate_evaluations` table and is used by the merge-back driver + `paper_cmd.py` via the concrete class), `gate_exists_by_token` (1638-1652 — same reasoning). `_conn: sqlite3.Connection` annotation.

`record_gate_with_fdr_and_maybe_promote` and its two private helpers (`_verify_funnel_snapshot`, `_mint_agent_novel_family` — the latter moves here too, see below) call into `SearchBreadthLedger` and `FamilyGraph` methods that now live in sibling files (`search_breadth.py`, `family.py`). This is the resolved design decision from this plan's header: **keep these methods here, in `gate.py`, and add mypy-only stub declarations** for every cross-file method they call — exactly the `bars.py`-`get_snapshot` pattern. Add these 10 stubs to the `GateLedgerMixin` class body (grouped near the top, before the real methods, with a one-line comment each saying which sibling file provides the real implementation):

```python
    def search_trials_fingerprint(self, *args, **kwargs):
        # provided by search_breadth.py's SearchBreadthLedgerMixin; stub for mypy only
        raise NotImplementedError

    def windowed_search_combos(self, *args, **kwargs):
        # provided by search_breadth.py's SearchBreadthLedgerMixin; stub for mypy only
        raise NotImplementedError

    def total_search_combos(self, *args, **kwargs):
        # provided by search_breadth.py's SearchBreadthLedgerMixin; stub for mypy only
        raise NotImplementedError

    def pooled_trial_sharpe_var(self, *args, **kwargs):
        # provided by search_breadth.py's SearchBreadthLedgerMixin; stub for mypy only
        raise NotImplementedError

    def funnel_trial_sharpe_var(self, *args, **kwargs):
        # provided by search_breadth.py's SearchBreadthLedgerMixin; stub for mypy only
        raise NotImplementedError

    def strategy_family(self, *args, **kwargs):
        # provided by family.py's FamilyGraphMixin; stub for mypy only
        raise NotImplementedError

    def family_lifetime_combos(self, *args, **kwargs):
        # provided by family.py's FamilyGraphMixin; stub for mypy only
        raise NotImplementedError

    def family_graph_fingerprint(self, *args, **kwargs):
        # provided by family.py's FamilyGraphMixin; stub for mypy only
        raise NotImplementedError

    def agent_novel_mint_seed(self, *args, **kwargs):
        # provided by family.py's FamilyGraphMixin; stub for mypy only
        raise NotImplementedError

    def check_agent_novel_mint_bounds(self, *args, **kwargs):
        # provided by family.py's FamilyGraphMixin; stub for mypy only
        raise NotImplementedError
```

Do NOT use `*args, **kwargs` loosely without checking — replace each stub's signature with the REAL signature from `repository.py`'s corresponding Protocol method (e.g. `repository.py:420-471` for the `SearchBreadthLedger` ones, `repository.py:765-876` for the `FamilyGraph` ones) so mypy actually type-checks call sites against them, not just against `Any`. The `*args, **kwargs` shown above is illustrative shorthand for this plan, not what you should commit — copy each Protocol method's real parameter list and return type from `repository.py`.

Also move `_mint_agent_novel_family` (1975-2041, currently sits among the FamilyGraph methods positionally but its ONLY caller is `record_gate_with_fdr_and_maybe_promote`) here, into `gate.py`, for the same reason `_verify_funnel_snapshot` stays here — its caller determines its home. Its body is FamilyGraph-domain raw SQL and calls `self.agent_novel_mint_seed()`/`self.check_agent_novel_mint_bounds()`, both already stubbed above.

- [ ] **Step 9: Create `algua/registry/store/forward_gate.py`**

`ForwardGateLedger` mixin. Class must inherit from `base.TransitionMixin` (`class ForwardGateMixin(TransitionMixin):`) since `record_forward_pass_and_promote` calls `_apply_transition_locked` directly. Move verbatim: `record_forward_gate_evaluation` (1072-1123), `_insert_forward_gate_row_locked` (1125-1178, private, both `record_forward_gate_evaluation` and `record_forward_pass_and_promote` call it — both are in this same file, no stub needed), `record_forward_pass_and_promote` (1180-1219), `find_consumable_forward_gate_evaluation` (1221-1247), `latest_forward_gate_row` (1249-1268). `_conn: sqlite3.Connection` annotation.

- [ ] **Step 10: Create `algua/registry/store/family.py`**

`FamilyGraph` mixin, class name `FamilyGraphMixin` (must match exactly — Step 12's facade imports this name). Move verbatim: `family_graph_fingerprint` (691-720), `create_family` (1704-1729), `assign_strategy_to_family` (1731-1775), `strategy_family` (1777-1783), `family_count` (1785-1791), `family_ancestry` (1793-1810), `add_parent_edge` (1812-1865), `all_families_with_member_profiles` (1867-1897), `_live_member_profile` (1899-1917, staticmethod, private, both callers are FamilyGraph methods in this same file), `materialise_legacy_member_profiles` (1919-1940), `agent_novel_mint_audit` (1942-1973), `agent_novel_mint_seed` (2043-2047), `check_agent_novel_mint_bounds` (2049-2081), `_family_member_strategies` (2083-2092, staticmethod, private, both callers are FamilyGraph methods in this same file), `windowed_family_combos` (2094-2111), `lifetime_combos_for_families` (2113-2159), `family_lifetime_combos` (2161-2163), `family_names` (2165-2168).

Move `_parse_canonical_utc` (84-91, per Step 2's confirmation) here too, as a module-level function. Move the two module-level constants here too (currently store.py:37-42, including their full comment block — this is the CODEOWNERS-protected-rationale comment, move it verbatim and update its self-reference from "algua/registry/store.py" to "algua/registry/store/family.py"):

```python
# --- #524 agent-NOVEL mint governance constants (R9-M1) --------------------------------------
# CODEOWNERS-protected: these live in algua/registry/store/family.py as module constants — NOT a
# CLI flag, NOT an env var — so the autonomous loop has no surface to read or raise them. Changing
# either requires a human PR to this protected file (the same human-gate as the promote relaxation
# flags).
AGENT_NOVEL_MINT_WINDOW_DAYS = 90   # rolling window for the burst rate cap (matches FUNNEL_WINDOW)
AGENT_NOVEL_MINT_CAP = 8            # max agent mints per rolling window (SOLE automatic bound)
```

`agent_novel_mint_seed` (moving here) calls `self.funnel_lifetime_search_combos()`, which belongs to `SearchBreadthLedger`/`search_breadth.py`. Add ONE mypy stub for it here (same pattern as Step 8, real signature copied from `repository.py:461-464`):

```python
    def funnel_lifetime_search_combos(self, *args, **kwargs):
        # provided by search_breadth.py's SearchBreadthLedgerMixin; stub for mypy only — replace
        # *args, **kwargs with the real signature from repository.py before committing
        raise NotImplementedError
```

`_conn: sqlite3.Connection` annotation.

- [ ] **Step 11: Create `algua/registry/store/backtest_returns.py`**

`BacktestReturnsLedger` mixin, class name `BacktestReturnsLedgerMixin` (must match exactly — Step 12's facade imports this name). Move verbatim: `persist_backtest_returns` (1658-1681), `load_backtest_returns` (1683-1698). `_conn: sqlite3.Connection` annotation.

- [ ] **Step 12: Create `algua/registry/store/__init__.py`**

The facade. Composes all 8 domain mixins into `SqliteStrategyRepository`; owns `__init__` and the `connection` property (both currently store.py:110-122, moved verbatim); re-exports the same 4 names the current `__all__` exports, plus `AGENT_NOVEL_MINT_CAP` (needed by `tests/registry/test_novel_family_seed_524.py`, which imports it directly from `algua.registry.store` today — confirm this import site still resolves after your edit).

```python
"""sqlite-backed registry store, carved by Protocol (spec §8). Each Protocol's implementation
lives in its own module (crud.py, approvals.py, search_breadth.py, holdout.py, gate.py,
forward_gate.py, family.py, backtest_returns.py); a shared base.py holds the one helper genuinely
called from more than one domain (_apply_transition_locked); SqliteStrategyRepository composes
them via mixins and keeps only the truly Protocol-agnostic members (__init__, connection) on
itself directly."""
from __future__ import annotations

import sqlite3

from algua.registry.store.approvals import ApprovalLedgerMixin
from algua.registry.store.backtest_returns import BacktestReturnsLedgerMixin
from algua.registry.store.crud import CrudMixin
from algua.registry.store.family import (
    AGENT_NOVEL_MINT_CAP as AGENT_NOVEL_MINT_CAP,
    FamilyGraphMixin,
)
from algua.registry.store.forward_gate import ForwardGateMixin
from algua.registry.store.gate import GateLedgerMixin
from algua.registry.store.holdout import HoldoutLedgerMixin
from algua.registry.store.search_breadth import SearchBreadthLedgerMixin

# StrategyExists/StrategyNotFound/StrategyRecord are declared in repository.py, not store.py —
# re-exported here (as `... as ...` per this repo's existing re-export convention, e.g.
# algua/data/store/__init__.py's `normalize_symbols as normalize_symbols`) so the current
# `from algua.registry.store import StrategyExists, ...` call sites keep working unmodified.
from algua.registry.repository import (
    StrategyExists as StrategyExists,
    StrategyNotFound as StrategyNotFound,
    StrategyRecord as StrategyRecord,
)

__all__ = [
    "AGENT_NOVEL_MINT_CAP",
    "SqliteStrategyRepository",
    "StrategyExists",
    "StrategyNotFound",
    "StrategyRecord",
]


class SqliteStrategyRepository(
    CrudMixin,
    ApprovalLedgerMixin,
    SearchBreadthLedgerMixin,
    HoldoutLedgerMixin,
    GateLedgerMixin,
    ForwardGateMixin,
    FamilyGraphMixin,
    BacktestReturnsLedgerMixin,
):
    """sqlite-backed ``StrategyRepository``: the only module that embeds registry SQL."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    @property
    def connection(self) -> sqlite3.Connection:
        """Read-only handle to the underlying sqlite connection, for protected verifiers (the
        live wall's forward-certificate check) that read operational tables alongside the
        repository. Deliberately NOT part of the ``StrategyRepository`` Protocol — the seam
        stays I/O-agnostic; non-sqlite repos must inject their own verifier."""
        return self._conn
```

Verify the exact `import ... as ...` re-export style against `algua/data/store/__init__.py`'s existing convention (lines 23-31 of that file) before committing to the syntax above — match the established style precisely rather than inventing a new one. `CrudMixin` and `ForwardGateMixin` both inherit from `base.TransitionMixin` per Steps 4/9 — Python's MRO resolves the diamond (`TransitionMixin` appearing twice in the ancestry via both `CrudMixin` and `ForwardGateMixin`) without conflict since it declares no `__init__` and no state is assigned anywhere except the facade's own `__init__` — confirm this is genuinely true (no `__init__` override anywhere in `base.py`) before treating it as settled.

- [ ] **Step 13: Delete the old `algua/registry/store.py`**

```bash
git rm algua/registry/store.py
```

- [ ] **Step 14: Update `CODEOWNERS`**

Find:
```
/algua/registry/store.py        @Lior-Nis   # the paper->live wall + approvals
```
Replace with:
```
/algua/registry/store/        @Lior-Nis   # the paper->live wall + approvals
```

- [ ] **Step 15: Full quality gate**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`

Expected: all four pass, with the SAME test count as the pre-task baseline (no test was added, removed, or behaviorally changed — every file's methods moved verbatim). If `mypy` complains about a stub's `*args, **kwargs` not matching a real call site, fix the stub's signature to the real one (per Step 8/10's instruction) rather than loosening the caller. Check `git status` for the known momentum-strategy test-fixture hazard; delete if present, don't stage it.

- [ ] **Step 16: Commit**

```bash
git add algua/registry/store/ CODEOWNERS
```

```bash
git commit -m "$(cat <<'EOF'
refactor: carve registry/store.py into registry/store/ by Protocol (stage 4b)

Splits the ~2170-line SqliteStrategyRepository monolith into algua/registry/store/ — one mixin
per repository.py Protocol (crud.py=StrategyStore, approvals.py=ApprovalLedger,
search_breadth.py=SearchBreadthLedger, holdout.py=HoldoutLedger, gate.py=GateLedger,
forward_gate.py=ForwardGateLedger, family.py=FamilyGraph, backtest_returns.py=BacktestReturnsLedger)
+ base.py (the one helper genuinely called from more than one domain,
_apply_transition_locked) + _util.py (_now, the one plain helper shared across most mixins) +
an __init__.py facade composing them at the same import path. Mirrors Stage 3's
data/store.py -> data/store/ carve exactly: mixins declare shared state as class-level type
annotations only, a stub-for-mypy pattern handles cross-file self.* calls
(record_gate_with_fdr_and_maybe_promote's two private helpers stay in gate.py, their sole
caller's file, rather than moving to the shared base — ownership follows the caller, not the
callee's data dependencies).

Zero behavior change: every method's body moved verbatim. repository.py (the Protocol
declarations) is untouched. CODEOWNERS' /algua/registry/store.py exact-file glob updated to the
directory glob /algua/registry/store/ in this same commit — deferring it even one commit would
have silently dropped review protection on every new file with no error.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Close-out verification

**Files:** none expected (verification only; fix anything found)

- [ ] **Step 1: Confirm zero stray references to the old flat-file path or private cross-file symbols**

```bash
grep -rn "from algua\.registry\.store import\|algua\.registry\.store\." algua/ tests/ --include='*.py'
```
Read through the hits: every one should reference only the 5 public re-exported names (`SqliteStrategyRepository`, `AGENT_NOVEL_MINT_CAP`, `StrategyExists`, `StrategyNotFound`, `StrategyRecord`) or the module path itself (`algua.registry.store` as a mock-patch target, if any — check for `mock.patch("algua.registry.store...")`-style strings too via `grep -rn "algua\.registry\.store\." tests/ | grep -i mock`). Any hit importing a private (`_`-prefixed) symbol, or a symbol that used to live directly on `store.py` but now lives on a specific mixin module, is a stray reference this task's Task 1 missed.

- [ ] **Step 2: Confirm the facade imports and composes cleanly**

```bash
uv run python -c "
from algua.registry.store import SqliteStrategyRepository, AGENT_NOVEL_MINT_CAP, StrategyExists, StrategyNotFound, StrategyRecord
import inspect
mro = [c.__name__ for c in SqliteStrategyRepository.__mro__]
print('MRO:', mro)
print('AGENT_NOVEL_MINT_CAP:', AGENT_NOVEL_MINT_CAP)
"
```
Expected: exits 0, prints an MRO list containing all 8 domain mixins + `TransitionMixin` (deduplicated once by Python's C3 linearization even though both `CrudMixin` and `ForwardGateMixin` inherit from it) + `object`, and `AGENT_NOVEL_MINT_CAP: 8`.

- [ ] **Step 3: Confirm every Protocol method is actually present and callable on the composed class**

```bash
uv run python -c "
from algua.registry.repository import StrategyRepository, ApprovalRepository
from algua.registry.store import SqliteStrategyRepository
import inspect
# structural Protocol check: every method the Protocols declare must exist on the concrete class
for proto in (StrategyRepository, ApprovalRepository):
    for name in proto.__protocol_attrs__ if hasattr(proto, '__protocol_attrs__') else dir(proto):
        if name.startswith('_') and name not in ('__init__',):
            continue
        if not hasattr(SqliteStrategyRepository, name):
            raise AssertionError(f'{proto.__name__}.{name} missing from SqliteStrategyRepository')
print('OK — every Protocol method resolves on the composed class')
"
```
If `__protocol_attrs__` isn't available on this Python's `typing.Protocol` (it's a fairly recent addition), fall back to explicitly listing each of the 8 Protocols' method names from `repository.py`'s definitions (Task 1 Step 1's read gives you these) and checking `hasattr` for each — either way, the point is an exhaustive presence check, not a partial one.

- [ ] **Step 4: Full quality gate one more time**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`

Expected: all four pass (this re-confirms Task 1's gate on a clean tree, catching anything a `git status` oversight might have missed), same test count as the pre-Task-1 baseline (zero delta — this whole plan adds/removes no tests). Check `git status` for the momentum-strategy hazard.

- [ ] **Step 5: `_apply_transition_locked`'s transaction-boundary discipline survived the split**

Read `algua/registry/store/base.py`'s `_apply_transition_locked` and confirm its docstring still says it must be called from WITHIN an already-open transaction and does not itself open one (`BEGIN`/`self._conn.execute("BEGIN")` should NOT appear inside this method's body). Then confirm each of its 3 callers (`crud.py`'s `apply_transition` and `intake_candidate_to_paper`, `forward_gate.py`'s `record_forward_pass_and_promote`) still opens its own transaction before calling it — grep each caller's body for `BEGIN IMMEDIATE` or `with self._conn:` appearing BEFORE the `_apply_transition_locked(` call site, byte-for-byte matching what it did in the original `store.py`. This is a safety-critical invariant the ground-truth research explicitly flagged as needing to survive the split exactly — verify it by reading the actual code, not by trusting that "the tests passed" is sufficient (a subtle transaction-nesting bug can pass tests that don't specifically probe concurrent access and only surface under real contention).

- [ ] **Step 6: CLI smoke test**

```bash
uv run algua doctor
uv run algua registry list
```
Expected: both exit 0 (or a clean, unrelated non-zero on an empty/no-data worktree — anything mentioning a missing import, an attribute error on `SqliteStrategyRepository`, or a `ModuleNotFoundError` for `algua.registry.store` is a real regression, not expected).

- [ ] **Step 7: Commit any fixes**

If steps 1-6 forced fixes, commit them (scoped `git add`, correct trailer). If nothing needed fixing, this task makes no commit — that's expected and consistent with how Stage 3's and Stage 4a's equivalent close-out tasks landed.
