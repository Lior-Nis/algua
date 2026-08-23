"""``GateLedger`` — research/shortlist gate evaluations, the funnel-snapshot CAS, and the
atomic record-gate-and-maybe-promote transaction (plus the #524 agent-NOVEL family mint, whose
sole caller lives here)."""
from __future__ import annotations

import json
import math
import sqlite3
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from algua.contracts.lifecycle import Actor, Stage, TransitionError
from algua.registry.db import FDR_COHORT_SIZE
from algua.registry.repository import (
    FamilyGraphDriftError,
    FdrGateOutcome,
    FdrStreamState,
    FunnelDriftError,
    FunnelFloor,
    FunnelSnapshot,
    PendingNovelFamily,
    StrategyRecord,
)
from algua.registry.store._util import _now
from algua.registry.store.base import TransitionMixin


class GateLedgerMixin(TransitionMixin):
    _conn: sqlite3.Connection

    if TYPE_CHECKING:
        # Cross-file calls: the real implementations live in SIBLING mixins (search_breadth.py,
        # family.py) and are composed by the facade. Declared TYPE_CHECKING-only — unlike
        # algua/data/store/bars.py, whose stub points at the FACADE (which always wins the MRO),
        # a runtime stub here would SHADOW a sibling mixin declared later in the facade's base
        # list. Signatures are copied from repository.py's Protocol declarations.
        def search_trials_fingerprint(self) -> tuple[int, int]: ...
        def windowed_search_combos(self, window_days: int) -> int: ...
        def total_search_combos(self, strategy_name: str) -> int: ...
        def pooled_trial_sharpe_var(self, strategy_name: str) -> float | None: ...
        def funnel_trial_sharpe_var(self, window_days: int) -> FunnelFloor: ...
        def strategy_family(self, strategy_name: str) -> int | None: ...
        def family_lifetime_combos(self, family_id: int) -> int: ...
        def family_graph_fingerprint(self) -> tuple[int, ...]: ...
        def agent_novel_mint_seed(self) -> int: ...
        def check_agent_novel_mint_bounds(self) -> None: ...
        # runs.py (RunLedgerMixin) — not a repository.py Protocol method (it is store-internal:
        # only the gate's own BEGIN IMMEDIATE calls it, never CLI/research code).
        def _insert_run_locked(self, kind: str, strategy_name: str, **kwargs: Any) -> int: ...

    def record_gate_evaluation(
        self,
        strategy_id: int,
        *,
        passed: bool,
        n_funnel: int,
        own_lifetime_combos: int,
        windowed_total_combos: int,
        funnel_window_days: int,
        breadth_provenance: str,
        pit_ok: bool,
        pit_override: bool,
        holdout_n_bars: int,
        min_holdout_observations: int,
        code_hash: str,
        config_hash: str,
        dependency_hash: str | None,
        data_source: str,
        snapshot_id: str | None,
        period_start: str,
        period_end: str,
        holdout_frac: float,
        actor: str,
        decision_json: str,
        fundamentals_snapshot: str | None = None,
        news_snapshot: str | None = None,
        family_id: int | None = None,
        family_lifetime_effective: int | None = None,
        universe_name: str | None = None,
    ) -> int:
        """Persist one gate evaluation (pass or fail) and return its row id. A passing AGENT row is
        the single-use token the shortlist transition consumes."""
        with self._conn:
            cur = self._conn.execute(
                "INSERT INTO gate_evaluations"
                "(strategy_id, passed, n_funnel, own_lifetime_combos, windowed_total_combos,"
                " funnel_window_days, breadth_provenance, pit_ok, pit_override, holdout_n_bars,"
                " min_holdout_observations, code_hash, config_hash, dependency_hash, data_source,"
                " snapshot_id, period_start, period_end, holdout_frac, actor, decision_json,"
                " fundamentals_snapshot, news_snapshot, consumed, created_at,"
                " family_id, family_lifetime_effective, universe_name)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,0,?,?,?,?)",
                (strategy_id, int(passed), n_funnel, own_lifetime_combos, windowed_total_combos,
                 funnel_window_days, breadth_provenance, int(pit_ok), int(pit_override),
                 holdout_n_bars, min_holdout_observations, code_hash, config_hash, dependency_hash,
                 data_source, snapshot_id, period_start, period_end, holdout_frac, actor,
                 decision_json, fundamentals_snapshot, news_snapshot, _now(),
                 family_id, family_lifetime_effective, universe_name),
            )
        rowid = cur.lastrowid
        assert rowid is not None
        return rowid

    def find_consumable_gate_evaluation(
        self,
        strategy_id: int,
        code_hash: str,
        config_hash: str,
        dependency_hash: str | None,
    ) -> int | None:
        """Return the id of the most-recent AGENT passing unconsumed gate row whose identity matches
        the recomputed current (code, config, dependency), or None. The ``actor='agent'`` filter
        means a human/override promote's audit row is never an agent-consumable token. A NULL
        ``dependency_hash`` matches nothing — fail-closed, mirroring has_valid_approval."""
        if dependency_hash is None:
            return None
        row = self._conn.execute(
            "SELECT id FROM gate_evaluations WHERE strategy_id=? AND passed=1 AND consumed=0"
            " AND actor='agent' AND code_hash=? AND config_hash=? AND dependency_hash=?"
            " ORDER BY id DESC LIMIT 1",
            (strategy_id, code_hash, config_hash, dependency_hash),
        ).fetchone()
        return int(row["id"]) if row is not None else None

    def fdr_stream_state(self) -> FdrStreamState | None:
        """Read the LORD++ FDR stream from ``gate_evaluations WHERE fdr_binding=1``, SCOPED to the
        cohort the NEXT binding test will join (#324, count-triggered cohort restarts).

        Rows with ``fdr_binding`` NULL or 0 are excluded (legacy-safe). Binding rows are read in
        ``id`` order and partitioned into cohorts of ``FDR_COHORT_SIZE`` by their stored
        ``fdr_cohort``. Integrity is validated fail-closed (returns None): any binding row with
        NULL/non-finite p-value or alpha-level, ``fdr_rejected`` not in {0, 1}, NULL ``fdr_cohort``,
        NULL/non-positive ``fdr_test_index``, a cohort index that does not advance monotonically as
        rows fill (0,0,…,1,1,… — i.e. rows must be grouped by cohort in id order), a cohort holding
        more than ``FDR_COHORT_SIZE`` rows, or within-cohort ``fdr_test_index`` not contiguous
        1,2,…,k in id order. The (cohort, fdr_test_index) partial-unique index also enforces
        uniqueness at write time.

        The returned ``t``/``discovery_indices`` are WITHIN-COHORT for the cohort the next test
        joins: a partially-filled last cohort (< FDR_COHORT_SIZE rows) IS that cohort (next
        ``t`` = its size + 1); a full last cohort seals and the next test opens ``cohort + 1`` at
        ``t = 1`` with no inherited discoveries. Audit counters (cohorts_completed, binding_tests,
        discoveries) are lifetime totals."""
        rows = self._conn.execute(
            "SELECT fdr_p_value, fdr_alpha_level, fdr_rejected, fdr_test_index, fdr_cohort"
            " FROM gate_evaluations WHERE fdr_binding=1 ORDER BY id",
        ).fetchall()
        if not rows:
            return FdrStreamState(
                t=1, discovery_indices=[], cohort_index=0,
                cohorts_completed=0, binding_tests=0, discoveries=0,
                current_cohort_applied_alpha=0.0)
        discoveries = 0
        # Per-cohort bookkeeping validated in id order: each cohort's fdr_test_index must run
        # 1..size and no cohort may exceed FDR_COHORT_SIZE; cohort index must advance by exactly 1
        # each time a new cohort begins (rows grouped contiguously by cohort in id order).
        cohort_sizes: dict[int, int] = {}
        # Discovery positions for the CURRENT (highest) cohort only — that is all the next test's
        # α_t depends on. Rebuilt whenever a new cohort begins.
        cur_cohort = 0
        cur_discovery_indices: list[int] = []
        # #529: Σ of stored α over the CURRENT cohort's rows (active-cohort exposure audit). Reset
        # whenever a new cohort begins; accumulated per row of the current cohort.
        cur_applied_alpha = 0.0
        prev_cohort: int | None = None
        for r in rows:
            p, alpha, rejected, idx, cohort = (
                r["fdr_p_value"], r["fdr_alpha_level"], r["fdr_rejected"],
                r["fdr_test_index"], r["fdr_cohort"],
            )
            if p is None or alpha is None or not math.isfinite(p) or not math.isfinite(alpha):
                return None
            if rejected is None or int(rejected) not in (0, 1):
                return None
            if cohort is None or int(cohort) < 0:
                return None
            if idx is None or int(idx) < 1 or int(idx) > FDR_COHORT_SIZE:
                return None
            cohort = int(cohort)
            idx = int(idx)
            if prev_cohort is None:
                # First binding row must open cohort 0 at position 1.
                if cohort != 0 or idx != 1:
                    return None
                cur_cohort = 0
                cur_discovery_indices = []
                cur_applied_alpha = 0.0
            elif cohort == prev_cohort:
                # Same cohort: index must be the next contiguous position.
                if idx != cohort_sizes[cohort] + 1:
                    return None
            elif cohort == prev_cohort + 1:
                # New cohort must begin only after the prior one filled exactly FDR_COHORT_SIZE,
                # and start at position 1. (A partial cohort followed by a higher cohort index is a
                # corrupt/mixed stream — fail closed.)
                if cohort_sizes[prev_cohort] != FDR_COHORT_SIZE or idx != 1:
                    return None
                cur_cohort = cohort
                cur_discovery_indices = []
                cur_applied_alpha = 0.0
            else:
                return None
            cohort_sizes[cohort] = cohort_sizes.get(cohort, 0) + 1
            if cohort_sizes[cohort] > FDR_COHORT_SIZE:
                return None
            prev_cohort = cohort
            cur_applied_alpha += float(alpha)
            if int(rejected) == 1:
                discoveries += 1
                cur_discovery_indices.append(idx)

        binding_tests = len(rows)
        last_cohort = cur_cohort
        last_size = cohort_sizes[last_cohort]
        if last_size < FDR_COHORT_SIZE:
            # The last cohort is still filling — the next test joins it.
            next_cohort = last_cohort
            next_t = last_size + 1
            next_discovery_indices = cur_discovery_indices
            next_applied_alpha = cur_applied_alpha
        else:
            # The last cohort is sealed — the next test opens a fresh cohort at t=1.
            next_cohort = last_cohort + 1
            next_t = 1
            next_discovery_indices = []
            next_applied_alpha = 0.0
        cohorts_completed = sum(1 for s in cohort_sizes.values() if s == FDR_COHORT_SIZE)
        return FdrStreamState(
            t=next_t, discovery_indices=next_discovery_indices, cohort_index=next_cohort,
            cohorts_completed=cohorts_completed, binding_tests=binding_tests,
            discoveries=discoveries, current_cohort_applied_alpha=next_applied_alpha)

    @staticmethod
    def _cas_funnel(name: str, expected: object, actual: object) -> None:
        """Fail-closed exact-equality check for one funnel-snapshot field (#339). Float fields are
        compared exactly: they are recomputed from the SAME committed rows by the SAME deterministic
        code, so an unchanged funnel yields a bit-identical value; any real drift changes it."""
        if expected != actual:
            raise FunnelDriftError(
                f"{name} changed between promotion compute and commit "
                f"({expected!r} -> {actual!r}); re-run the promotion against fresh funnel state")

    def _verify_funnel_snapshot(self, funnel: FunnelSnapshot) -> None:
        """#339 serializability CAS — MUST run with the write lock held (inside BEGIN IMMEDIATE).
        Re-reads every mutable funnel-wide input the (lock-free) decision was computed against and
        raises ``FunnelDriftError`` on any drift, which the caller's ``except`` rolls the whole
        transaction back on. Conservative: it can only PREVENT a commit, never cause a false pass.

        Covers the complete decision-input set (the rest of the decision is pinned to immutable
        artifacts — fixed data snapshot, immutable strategy source/config hash, deterministic
        bootstrap seed, and the atomically pre-burned holdout reservation #161):
          * the append-only ``search_trials`` row-identity fingerprint (pins the whole row set, so
            it pins the measured own-combo count AND the pooled/funnel variances against a
            same-value collision);
          * ``windowed_search_combos`` + family breadth — feed ``n_funnel`` (the deflated bar) on
            EVERY path;
          * on the FDR-binding (measured) path, the own-combo count and the two DSR variances that
            feed the ``p_value``.
        The FDR stream itself is deliberately NOT snapshotted here — it is read live under this same
        lock because the online LORD++ position MUST reflect all prior commits."""
        cur_count, cur_max = self.search_trials_fingerprint()
        self._cas_funnel(
            "search_trials fingerprint",
            (funnel.search_trials_count, funnel.search_trials_max_id), (cur_count, cur_max))
        self._cas_funnel(
            "windowed_search_combos", funnel.windowed_total_combos,
            self.windowed_search_combos(funnel.funnel_window_days))
        family_id = self.strategy_family(funnel.strategy_name)
        self._cas_funnel("strategy_family", funnel.family_id, family_id)
        family_eff = self.family_lifetime_combos(family_id) if family_id is not None else 0
        self._cas_funnel("family_lifetime_combos", funnel.family_lifetime_effective, family_eff)
        if funnel.dsr_binding:
            self._cas_funnel(
                "total_search_combos", funnel.own_lifetime_combos,
                self.total_search_combos(funnel.strategy_name))
            self._cas_funnel(
                "pooled_trial_sharpe_var", funnel.dsr_trial_var_ann,
                self.pooled_trial_sharpe_var(funnel.strategy_name))
            floor = self.funnel_trial_sharpe_var(funnel.funnel_window_days)
            self._cas_funnel(
                "funnel_trial_sharpe_var.var_ann", funnel.funnel_floor_var_ann, floor.var_ann)
            self._cas_funnel(
                "funnel_trial_sharpe_var.n_strategies",
                funnel.funnel_floor_n_strategies, floor.n_strategies)
            self._cas_funnel(
                "funnel_trial_sharpe_var.n_total_rows",
                funnel.funnel_floor_n_total_rows, floor.n_total_rows)

    def record_gate_with_fdr_and_maybe_promote(
        self,
        rec: StrategyRecord,
        *,
        gate_row: dict[str, Any],
        funnel: FunnelSnapshot,
        actor: Actor,
        reason: str | None = None,
        pending_novel_family: PendingNovelFamily | None = None,
        # ``run_row`` is the economic-layer run payload for THIS decision (spec 2026-08-23 §4.1),
        # inserted inside this method's BEGIN IMMEDIATE so it shares the gate row's fate. It cannot
        # be written by the caller: this method refuses to run inside an open transaction, so a
        # caller-side write would either land in a separate transaction (a run row that survives a
        # rolled-back gate — a phantom evaluation) or trip the top-level guard above.
        run_row: dict[str, Any] | None = None,
    ) -> FdrGateOutcome:
        # #524: coerce the actor BEFORE the pending-mint boundary guard (callers may pass a raw
        # string; an identity test against an un-coerced string would mis-evaluate).
        actor = Actor(actor)
        # #524: the mint is an AGENT-only capability and THIS method is the safety boundary, not
        # just the (trusted) caller path. Fail-closed unless the method actor is the agent AND the
        # pending spec's own actor/verdict are internally consistent (a caller bug must not write
        # mismatched audit/classification metadata). 'novel' is a plain-string literal so the store
        # never imports algua.research.SimVerdict (registry→research boundary stays clean).
        if pending_novel_family is not None and not (
            actor is Actor.AGENT
            and pending_novel_family.actor == Actor.AGENT.value
            and pending_novel_family.verdict == "novel"
        ):
            raise ValueError(
                "record_gate_with_fdr_and_maybe_promote: a pending_novel_family mint requires "
                "actor=agent and a consistent agent/novel pending spec")
        # TOP-LEVEL ONLY — mirrors reserve_holdout's contract. A manual BEGIN IMMEDIATE inside an
        # already-open transaction would raise "cannot start a transaction within a transaction";
        # catching that instead of pre-checking would leave the caller's surrounding tx open in a
        # rolled-back state. Fail loudly so the contract is enforced, not assumed.
        if self._conn.in_transaction:
            raise RuntimeError(
                "record_gate_with_fdr_and_maybe_promote must be called at top level,"
                " not inside an open transaction")

        # #339 — bind the funnel snapshot to the strategy being promoted. The in-lock CAS re-reads
        # the funnel-wide state keyed by funnel.strategy_name; if that name did not match rec (a
        # caller passing the wrong snapshot), the CAS would validate a DIFFERENT strategy's inputs
        # and the promoted strategy's own drift could escape. Fail closed on mismatch — the store
        # method is the safety boundary, not just the run_gate caller.
        if funnel.strategy_name != rec.name:
            raise FunnelDriftError(
                f"funnel snapshot is for {funnel.strategy_name!r} but the promotion is for "
                f"{rec.name!r}; the funnel CAS must verify the strategy being promoted")

        provisional_passed = bool(gate_row.get("passed"))
        final_passed: bool

        # BEGIN IMMEDIATE takes the write lock up front so the gate row INSERT and the optional
        # stage CAS are one atomic critical section.
        # BaseException (not Exception) so KeyboardInterrupt/SystemExit still rolls back.
        try:
            self._conn.execute("BEGIN IMMEDIATE")
            # #524 step 1 — pending-NOVEL CAS UNDER the write lock (R3-HIGH + R5-HIGH-1). Before any
            # other work, prove the agent-NOVEL verdict is still valid against the graph
            # at commit: (a) the strategy is STILL unassigned (no concurrent assignment landed), and
            # (b) the full-classifier-read-set fingerprint EQUALS the one the NOVEL verdict was
            # computed on (captured before==after classification, §7.1). A mismatch means a family
            # was minted / a member (re)assigned or removed / a parentage edge added / a member's
            # returns refreshed since classification, so the verdict is stale and might now be
            # MERGE/PARENTAGE — fail closed (the CLI re-runs preflight against the current graph).
            if pending_novel_family is not None:
                if self.strategy_family(rec.name) is not None:
                    raise FamilyGraphDriftError(
                        f"strategy {rec.name!r} was assigned to a family since NOVEL "
                        "classification; re-run promote", axis="still_assigned")
                if self.family_graph_fingerprint() != pending_novel_family.graph_fingerprint:
                    raise FamilyGraphDriftError(
                        f"family graph changed since {rec.name!r} was classified NOVEL; re-run "
                        "promote", axis="graph_fingerprint")
            # #339 — serializability CAS: re-read the funnel-wide MUTABLE state the (lock-free)
            # decision was computed against and abort if any of it drifted, so a committed decision
            # is provably a pure function of ONE funnel snapshot. Runs BEFORE the INSERT/stage-CAS,
            # inside this one write-locked critical section.
            self._verify_funnel_snapshot(funnel)
            # Factory soft gate (#529): the LORD++ FDR-binding branch itself was deleted
            # (simplification stage 4a) — it was provably dead in production (run_gate never
            # supplied a real p_value). final_passed is now always the provisional integrity-floor
            # verdict. The FDR ledger machinery this branch used to write — fdr_stream_state's read
            # path, the gate_evaluations fdr_* columns, db.py's cohort backfills — is preserved for
            # future re-tightening; only the write path (this branch) and its LORD++ math
            # (research/fdr_lord.py) are gone.
            final_passed = provisional_passed

            # Patch decision_json so the stored audit record reflects final_passed — only known
            # after this transaction's checks run.
            raw_decision = json.loads(gate_row.get("decision_json") or "{}")
            raw_decision["passed"] = final_passed
            decision_json = json.dumps(raw_decision)

            cur = self._conn.execute(
                "INSERT INTO gate_evaluations"
                "(strategy_id, passed, n_funnel, own_lifetime_combos, windowed_total_combos,"
                " funnel_window_days, breadth_provenance, pit_ok, pit_override, holdout_n_bars,"
                " min_holdout_observations, code_hash, config_hash, dependency_hash, data_source,"
                " snapshot_id, period_start, period_end, holdout_frac, actor, decision_json,"
                " consumed, created_at,"
                " fdr_binding, fdr_p_value, fdr_alpha_level, fdr_rejected, fdr_test_index,"
                " fdr_cohort,"
                " family_id, family_lifetime_effective,"
                " fundamentals_snapshot, news_snapshot, attempt_token, universe_name)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                # Agent passing rows are born-consumed: the stage has already advanced inside
                # this transaction, so the token is spent. Leaving consumed=0 would let a
                # future `registry transition --to candidate --actor agent` reuse the old row
                # after a back-step, bypassing the gate re-run requirement.
                (rec.id, int(final_passed),
                 gate_row["n_funnel"], gate_row["own_lifetime_combos"],
                 gate_row["windowed_total_combos"], gate_row["funnel_window_days"],
                 gate_row["breadth_provenance"], int(gate_row["pit_ok"]),
                 int(gate_row.get("pit_override", False)),
                 gate_row["holdout_n_bars"], gate_row["min_holdout_observations"],
                 gate_row["code_hash"], gate_row["config_hash"], gate_row.get("dependency_hash"),
                 gate_row["data_source"], gate_row.get("snapshot_id"),
                 gate_row["period_start"], gate_row["period_end"], gate_row["holdout_frac"],
                 actor.value, decision_json,
                 int(actor is Actor.AGENT and final_passed), _now(),
                 # fdr_binding, fdr_p_value, fdr_alpha_level, fdr_rejected, fdr_test_index,
                 # fdr_cohort: the binding-write path is retired (stage 4a) — every binding row
                 # is NULL going forward. Historical binding rows (written before this change)
                 # keep their real values; fdr_stream_state's WHERE fdr_binding=1 filter already
                 # excludes NULL rows, so it continues to see only genuine historical binding
                 # evidence.
                 None, None, None, None, None, None,
                 gate_row.get("family_id"), gate_row.get("family_lifetime_effective"),
                 gate_row.get("fundamentals_snapshot"), gate_row.get("news_snapshot"),
                 gate_row.get("attempt_token"), gate_row.get("universe_name")),
            )
            gate_id = cur.lastrowid
            assert gate_id is not None

            if run_row is not None:
                # Keep the run row's `passed` in lockstep with the gate row it derives from — even
                # though final_passed == provisional_passed unconditionally today (the LORD++
                # binding-rewrite branch is retired, stage 4a), a run row whose `passed` disagrees
                # with its own gate row would be a corrupt audit trail if that ever changes back.
                row = dict(run_row)
                row["passed"] = final_passed
                self._insert_run_locked("gate", rec.name, **row)

            updated_rec: StrategyRecord | None = None
            if final_passed:
                # #246: re-assert the source-stage invariant INSIDE this locked critical section.
                # The CAS below only pins WHERE stage=rec.stage (the caller's post-walk_forward
                # re-read). If a concurrent transition drifted the strategy off BACKTESTED before
                # that re-read — e.g. to a terminal RETIRED — the CAS would happily apply a
                # forbidden RETIRED->CANDIDATE edge (ALLOWED_TRANSITIONS forbids it). Mirror
                # promotion_preflight here, atomic with the CAS, so the gate can never resurrect a
                # drifted/terminal stage. Require exactly BACKTESTED (not validate_transition, which
                # permits the legal PAPER->CANDIDATE back-step) — promotion_preflight's reasoning.
                if rec.stage is not Stage.BACKTESTED:
                    raise TransitionError(
                        f"research promote requires stage backtested at promote time, got "
                        f"{rec.stage.value} (source drifted off backtested under concurrency since "
                        f"preflight); refusing to promote")
                updated_rec = self._apply_transition_locked(
                    rec, Stage.CANDIDATE, actor, reason,
                    code_hash=gate_row["code_hash"], config_hash=gate_row["config_hash"],
                    dependency_hash=gate_row.get("dependency_hash"),
                    consume_gate_id=None, consume_forward_gate_id=None, now=_now())
                # #524 step 5 — mint the seeded agent-NOVEL family, IN THIS SAME TRANSACTION, only
                # on a pass. All-or-nothing with the gate row + stage CAS: a crash before commit
                # leaves no orphan family/membership.
                if pending_novel_family is not None:
                    self._mint_agent_novel_family(pending_novel_family, rec.name, gate_id)

            self._conn.commit()
        except BaseException:
            self._conn.rollback()
            raise

        return FdrGateOutcome(
            gate_id=gate_id,
            fdr_binding=False,
            fdr_test_index=None,
            fdr_p_value=None,
            fdr_alpha_level=None,
            fdr_rejected=None,
            final_passed=final_passed,
            updated_rec=updated_rec,
            fdr_cohort=None,
            fdr_cohorts_completed=None,
            fdr_binding_tests=None,
            fdr_discoveries=None,
            fdr_expected_false_discoveries=None,
            fdr_throttle_window_binding=None,
            fdr_throttle_tripped=None,
            fdr_active_cohort_position=None,
            fdr_active_cohort_applied_alpha=None,
            fdr_expected_false_discoveries_incl_active=None,
        )

    def passing_gate_by_token(self, strategy: str, attempt_token: str) -> int | None:
        """The ``gate_evaluations.id`` of the PASSING, STRICT-AGENT row stamped with this exact
        ``attempt_token`` for ``strategy`` — the authoritative read the merge-back driver (#485)
        uses to attribute a promote outcome to THIS attempt, never to the ambient stage nor to "did
        the call raise". Returns None if no such row exists.

        The match is by exact ``attempt_token`` equality AND strict-agent context
        (``actor='agent'``, ``passed=1``): the token is a caller-supplied opaque per-attempt key
        (enforced unique by the partial index on non-null ``(strategy_id, attempt_token)``), so a
        concurrent external ``research promote`` of the same strategy — which supplies a different
        token, or NULL — can never be mistaken for this attempt's own gate row even if it mints a
        higher-id passing row in the same window.
        """
        row = self._conn.execute(
            "SELECT g.id FROM gate_evaluations g JOIN strategies s ON s.id = g.strategy_id"
            " WHERE s.name = ? AND g.attempt_token = ? AND g.passed = 1 AND g.actor = 'agent'"
            " ORDER BY g.id DESC LIMIT 1",
            (strategy, attempt_token),
        ).fetchone()
        return int(row["id"]) if row is not None else None

    def gate_exists_by_token(self, strategy: str, attempt_token: str) -> bool:
        """True iff ANY ``gate_evaluations`` row — PASSING **or** FAILING — is already stamped with
        this exact ``attempt_token`` for ``strategy``. The crash-idempotency read the merge-back
        driver (#485) uses to avoid re-invoking a metered promote whose prior (crashed) attempt
        already inserted a row and burned the holdout: because ``(strategy_id, attempt_token)`` is
        partial-unique, a second promote under the same token would late-fail on the index AFTER its
        side effects, so the driver must detect the existing row and branch on it instead of
        re-invoking. Unlike :meth:`passing_gate_by_token` this ignores ``passed`` (a failing row
        still consumed the token and burned the holdout)."""
        row = self._conn.execute(
            "SELECT 1 FROM gate_evaluations g JOIN strategies s ON s.id = g.strategy_id"
            " WHERE s.name = ? AND g.attempt_token = ? LIMIT 1",
            (strategy, attempt_token),
        ).fetchone()
        return row is not None

    def _mint_agent_novel_family(
        self, pending: PendingNovelFamily, founder_name: str, gate_id: int,
    ) -> int:
        """Mint the seeded agent-NOVEL family + assign the founder, via RAW locked INSERTs INSIDE
        the caller's open promote transaction (#524 step 5). NOT to be called on its own — it does
        no BEGIN/commit and relies on the caller's BEGIN IMMEDIATE. Order: (5a) seed>0, (5b) the
        per-window rate cap, (5c) uuid-suffixed name + raw INSERTs. Returns the family id."""
        # (5a) filter-before-sum seed + positivity guard. A corrupt/overlarge legacy row is EXCLUDED
        # by the WHERE filter (contributes 0), never overflowing or shrinking the seed below the sum
        # of the legitimate rows; an all-corrupt/empty funnel legitimately fails closed.
        seed = self.agent_novel_mint_seed()
        if seed <= 0:
            raise ValueError(
                "agent-NOVEL mint requires a strictly-positive funnel-lifetime breadth seed; the "
                "funnel has no well-typed in-range search_trials to seed (fail closed)")
        # (5b) authoritative under-lock per-window rate cap (fail-closed).
        self.check_agent_novel_mint_bounds()
        # (5c) collision-RESISTANT uuid mint name (no bounded probe → no DoS on a passed gate),
        # regenerate-and-retry on the astronomically-unlikely UNIQUE clash so correctness never
        # depends on uuid uniqueness. RAW INSERTs (NOT create_family/assign helpers, which
        # open their own transactions and would break this single BEGIN IMMEDIATE).
        now = _now()
        family_id: int | None = None
        for _ in range(8):
            candidate = f"{pending.slug_base}__{uuid4().hex}"
            try:
                cur = self._conn.execute(
                    "INSERT INTO families(name, created_at, created_by_actor, created_by_strategy,"
                    " seeded_prior_combos, founder_gate_id) VALUES (?,?,?,?,?,?)",
                    (candidate, now, Actor.AGENT.value, founder_name, seed, gate_id),
                )
                family_id = cur.lastrowid
                break
            except sqlite3.IntegrityError as exc:
                # ONLY a families.name UNIQUE clash is a retryable uuid collision. Any OTHER
                # integrity failure (a bad founder_gate_id FK, the seeded_prior_combos CHECK, an
                # append-only trigger, schema corruption) is a real invariant break — re-raise it
                # so the mint fails LOUDLY, never masked as "could not obtain a unique name".
                if "families.name" not in str(exc):
                    raise
                continue  # UNIQUE name clash — regenerate the uuid and retry (statement-scoped)
        if family_id is None:
            raise ValueError("agent-NOVEL mint could not obtain a unique family name (fail closed)")
        # family_created event (strategy_name = founder for audit clarity).
        self._conn.execute(
            "INSERT INTO family_events(event_type, family_id, strategy_name, actor, created_at)"
            " VALUES ('family_created', ?, ?, ?, ?)",
            (family_id, founder_name, Actor.AGENT.value, now),
        )
        # founding member row, with the founder's classified profile DB-persisted from birth.
        self._conn.execute(
            "INSERT INTO family_members(family_id, strategy_name, joined_at, joined_by_actor,"
            " removed_at, member_code_hash, member_factors_json) VALUES (?,?,?,?,NULL,?,?)",
            (family_id, founder_name, now, Actor.AGENT.value,
             pending.founder_code_hash, pending.founder_factors_json),
        )
        # strategy_assigned event with the classification metadata.
        self._conn.execute(
            "INSERT INTO family_events(event_type, family_id, strategy_name, actor,"
            " clustering_verdict, similarity_score, clustering_version, clustering_config_json,"
            " axis_json, matched_family_id, created_at)"
            " VALUES ('strategy_assigned', ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?)",
            (family_id, founder_name, Actor.AGENT.value, pending.verdict,
             pending.similarity_score, pending.clustering_version,
             pending.clustering_config_json, pending.axis_json, now),
        )
        return int(family_id)
