from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, NamedTuple, Protocol

from algua.contracts.lifecycle import Actor, Stage
from algua.contracts.registry_metadata import Author, HypothesisStatus


class FdrGateOutcome(NamedTuple):
    """Return value of ``record_gate_with_fdr_and_maybe_promote``.

    Carries the gate row id, the FDR audit fields (None when non-binding), the composite
    ``final_passed`` verdict, and the updated ``StrategyRecord`` if the strategy was promoted
    (None otherwise). Task 5 surfaces these in ``GateDecision.fdr_*`` fields."""

    gate_id: int
    fdr_binding: bool
    fdr_test_index: int | None
    fdr_p_value: float | None
    fdr_alpha_level: float | None
    fdr_rejected: bool | None
    final_passed: bool
    updated_rec: StrategyRecord | None


class FdrStreamState(NamedTuple):
    """Snapshot of the global LORD++ alpha-wealth stream.

    Source: ``gate_evaluations WHERE fdr_binding=1``. ``t`` is the count of binding rows (the
    current stream position); ``discovery_indices`` is the ordered list of ``fdr_test_index``
    values where ``fdr_rejected=1`` (past rejections that replenish alpha-wealth for future
    tests). Together they fully determine ``lord_plus_plus_level`` for the NEXT test."""

    t: int
    discovery_indices: list[int]


class ArtifactIdentity(NamedTuple):
    """The full identity a human approval binds to and the live gate recomputes.

    A ``NamedTuple`` so callers can either unpack ``(code_hash, config_hash, dependency_hash)``
    or read fields by name. ``dependency_hash`` is ``None`` only when the lockfile is absent;
    such an identity can never match a stored approval (see ``has_valid_approval``)."""

    code_hash: str
    config_hash: str
    dependency_hash: str | None


class StrategyExists(ValueError):
    pass


class StrategyNotFound(LookupError):
    pass


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


class StrategyRepository(Protocol):
    """Persistence seam for the registry.

    The lifecycle policy (transitions, approvals) depends on this Protocol, never on a concrete
    database driver. The sqlite implementation lives in ``algua.registry.store`` and is the only
    place that knows SQL; swapping the backing store means writing another implementation, not
    touching policy code.
    """

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
        not be the strategy itself.
        Raises ``StrategyExists`` / ``StrategyNotFound`` / ``ValueError``.
        """
        ...

    def get(self, name: str) -> StrategyRecord:
        """Return the strategy by name, or raise ``StrategyNotFound``."""
        ...

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
        """Update only the supplied organizational-metadata fields (never the stage).
        ``add_tags``/``remove_tags`` mutate the tag set. Returns the updated record.
        """
        ...

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
        raw validated strings (the caller maps/validates against the enums). Idempotent: re-running
        is a no-op once columns are non-NULL.
        """
        ...

    def default_fill_metadata_nulls(self) -> None:
        """Fill every strategy row's author/hypothesis_status/tags column from its default when
        still NULL. Terminal step of the ``backfill-from-kb`` command. Idempotent."""
        ...

    def delete(self, name: str) -> None:
        """Remove a strategy row and its transition rows. ONLY for rolling back a failed
        ``strategy new`` that just created it — there is no general deletion workflow."""
        ...

    def list_transitions(self, name: str) -> list[dict]:
        """Return the strategy's ordered stage-transition history."""
        ...

    def apply_transition(
        self,
        rec: StrategyRecord,
        to: Stage,
        actor: Actor,
        reason: str | None = None,
        code_hash: str | None = None,
        config_hash: str | None = None,
        dependency_hash: str | None = None,
        consume_gate_id: int | None = None,
        consume_forward_gate_id: int | None = None,
        revoke_allocation: bool = False,
    ) -> StrategyRecord:
        """Atomically advance ``rec`` to ``to``, append a transition row, return the new state.

        The stage write is a compare-and-swap on ``rec.stage``: if another session moved the
        stage since the caller read ``rec``, raise ``TransitionError`` (the whole transaction —
        including any token consume — rolls back). At most one of ``consume_gate_id`` /
        ``consume_forward_gate_id`` may be set (``ValueError`` otherwise); the forward consume
        re-checks the full token predicate (identity, actor, passed, unconsumed, TTL) against the
        caller-supplied ``code_hash``/``config_hash``/``dependency_hash`` at consume time.

        When ``revoke_allocation`` is set, the strategy's active live allocation is revoked in the
        SAME transaction as the stage change (used by the live -> dormant bench edge)."""
        ...

    def record_approval(
        self,
        strategy_id: int,
        code_hash: str,
        config_hash: str,
        dependency_hash: str | None,
        approved_by: str,
    ) -> int:
        """Persist a human approval pinning ``code_hash``/``config_hash``/``dependency_hash``;
        return its row id."""
        ...

    def has_valid_approval(
        self,
        strategy_id: int,
        code_hash: str,
        config_hash: str,
        dependency_hash: str | None,
    ) -> bool:
        """True iff a non-revoked approval pins exactly this strategy + code + config +
        dependency set. A ``None`` ``dependency_hash`` (no lockfile) never matches, and a stored
        row with a NULL ``dependency_hash`` never matches a concrete hash — both fail closed."""
        ...

    def record_search_trial(
        self, strategy_name: str, n_combos: int, grid_json: str,
        *, trial_sharpe_count: int | None = None,
        trial_sharpe_mean: float | None = None,
        trial_sharpe_var_ann: float | None = None,
    ) -> int:
        """Persist one measured search-breadth row (size + grid + the sweep's trial-Sharpe
        (count, mean, annualized var) for the #211 DSR dispersion); return its row id. Keyed by
        strategy NAME so a sweep run BEFORE the strategy is registered still counts toward promotion
        breadth. Stats default to None for callers that record only breadth."""
        ...

    def pooled_trial_sharpe_var(self, strategy_name: str) -> float | None:
        """Exact pooled SAMPLE variance (ddof=1) of the strategy's trial Sharpes across all its
        search_trials rows, via the law of total variance over the per-row (count, mean, var)
        triples. Returns None (fail closed) if there are no rows OR any contributing row has a
        NULL/NaN/inf count/mean/var. ANNUALIZED units (caller converts)."""
        ...

    def total_search_combos(self, strategy_name: str) -> int:
        """Sum of ``n_combos`` across every recorded ``search_trials`` row for the strategy NAME —
        the cumulative count of parameter combinations searched in this family (0 if none)."""
        ...

    def reserve_holdout(
        self,
        strategy_id: int,
        *,
        data_source: str,
        snapshot_id: str | None,
        period_start: str,
        period_end: str,
        holdout_frac: float,
        holdout_start: str,
        holdout_end: str,
        allow_reuse: bool,
    ) -> tuple[int, bool]:
        """Atomically claim the holdout window; return ``(reservation_id, reused)``.

        Under ``BEGIN IMMEDIATE`` (write lock held): re-check overlap against ALL rows (pending
        reservation OR committed burn) for this strategy, matching on the OOS INTERVAL
        ``[holdout_start, holdout_end]`` — the exact bars ``walk_forward`` burns (#192). The match
        is PROVENANCE-INDEPENDENT (#205): ``data_source``/``snapshot_id`` are stored as evidence
        only, never matched on, so the same OOS window is burn-once regardless of how the bars were
        reached (snapshot S, snapshot S2, or provider P). Then INSERT a pending row
        (``committed_at=NULL``, placeholder ``config_hash=''``). Match is on the INTERVAL, never
        config: a different ``holdout_frac`` landing on overlapping OOS bars does NOT escape the
        guard. A stored row with a NULL interval (legacy/old-code reservation) matches
        UNCONDITIONALLY — fail closed. An
        inverted incoming interval (start > end) is rejected (fail closed). ``period_*``/
        ``holdout_frac`` are evidence only.

        Raises ``ValueError`` (fail closed) if an overlapping row exists and not ``allow_reuse``.
        ``reused`` is True iff an overlapping row existed and the human override let it proceed.

        TOP-LEVEL ONLY: must not be called inside an open transaction / ``with self._conn:`` block
        (raises ``RuntimeError`` if ``self._conn.in_transaction``)."""
        ...

    def finalize_holdout_reservation(self, reservation_id: int, *, config_hash: str) -> None:
        """Commit a reservation into a burn: set ``committed_at`` + the real evidentiary
        ``config_hash``. Raises if the row is missing or already committed (guards double-finalize).
        """
        ...

    def release_holdout_reservation(self, reservation_id: int) -> None:
        """Free a still-pending reservation (clean walk_forward failure). Never touches a committed
        burn; a release after finalize/crash is a harmless no-op."""
        ...

    def windowed_search_combos(self, window_days: int) -> int:
        """Sum of ``n_combos`` across ALL strategies' ``search_trials`` recorded within the trailing
        ``window_days`` — funnel-wide search effort for the breadth wall (0 if none)."""
        ...

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
    ) -> int:
        """Persist one gate evaluation (pass or fail) and return its row id. A passing AGENT row is
        the single-use token the shortlist transition consumes."""
        ...

    def find_consumable_gate_evaluation(
        self,
        strategy_id: int,
        code_hash: str,
        config_hash: str,
        dependency_hash: str | None,
    ) -> int | None:
        """Return the id of the most-recent AGENT passing unconsumed gate row whose identity matches
        the recomputed (code, config, dependency), or None. A human row and a NULL
        ``dependency_hash`` are never consumable tokens — fail-closed."""
        ...

    def record_forward_gate_evaluation(
        self,
        strategy_id: int,
        *,
        passed: bool,
        n_forward_observations: int,
        min_forward_observations: int,
        session_coverage: float | None,
        realized_sharpe: float | None,
        holdout_sharpe: float | None,
        degradation_factor: float,
        sharpe_floor: float,
        realized_vol: float | None,
        min_forward_vol: float,
        realized_max_drawdown: float | None,
        max_forward_drawdown: float,
        first_tick_id: int | None,
        last_tick_id: int | None,
        first_tick_ts: str | None,
        last_tick_ts: str | None,
        max_staleness_sessions: int,
        n_reconcile_failures: int,
        n_concurrent_forward: int,
        account_id: str | None,
        code_hash: str,
        config_hash: str,
        dependency_hash: str | None,
        actor: str,
        decision_json: str,
        consumable: bool,
    ) -> int:
        """Persist one forward-test gate evaluation (pass or fail) and return its row id. A
        passing AGENT row written ``consumable=True`` is the single-use token the paper ->
        forward_tested transition consumes; ``consumable=False`` writes the row already consumed
        — a CERTIFICATE for the live wall, never a re-entry token (#124 GATE-2)."""
        ...

    def record_forward_pass_and_promote(
        self,
        rec: StrategyRecord,
        *,
        gate_row: dict[str, Any],
        actor: Actor,
        reason: str | None = None,
    ) -> tuple[int, StrategyRecord]:
        """Record a PASSING forward-gate evaluation AND advance ``rec`` paper -> forward_tested
        ATOMICALLY — the row insert, the compare-and-swap stage change, and the transition row
        commit together or not at all (#124 GATE-2). ``gate_row`` carries
        ``record_forward_gate_evaluation``'s row kwargs minus ``actor``/``consumable``.

        The row is born consumed regardless of actor (born-and-spent in the one transaction): it
        is the live wall's certificate, never a re-entry token a later demotion could bank. If
        another session moved the stage since ``rec`` was read, ``TransitionError`` — and the
        loser leaves NO row at all (its decision is lost; re-run the gate). Raises ``ValueError``
        on a non-passing ``gate_row``."""
        ...

    def find_consumable_forward_gate_evaluation(
        self,
        strategy_id: int,
        code_hash: str,
        config_hash: str,
        dependency_hash: str | None,
        *,
        now: str,
        ttl_days: int,
    ) -> int | None:
        """Return the id of the most-recent AGENT passing unconsumed forward-gate row whose
        identity matches the recomputed (code, config, dependency) and whose ``created_at`` is
        within ``ttl_days`` of ``now`` (a stale token can never be banked), or None. A human row
        and a NULL ``dependency_hash`` are never consumable tokens — fail-closed."""
        ...

    def latest_forward_gate_row(
        self,
        strategy_id: int,
        code_hash: str,
        config_hash: str,
        dependency_hash: str | None,
    ) -> dict | None:
        """Return the newest forward-gate row (all columns, as a dict) for this strategy+identity
        regardless of passed/consumed, or None. The live wall's certificate selection —
        pass-or-fail on purpose: a newer failed re-evaluation must invalidate an older pass
        (#124). A NULL ``dependency_hash`` matches nothing — fail-closed."""
        ...

    def fdr_stream_state(self) -> FdrStreamState | None:
        """Read the global LORD++ FDR stream: all ``gate_evaluations`` rows where
        ``fdr_binding=1``, ordered by ``id``.

        Returns ``FdrStreamState(t, discovery_indices)`` where ``t`` is the total count of
        binding rows and ``discovery_indices`` is the ordered list of ``fdr_test_index`` values
        where ``fdr_rejected=1``. Returns ``None`` (fail-closed) on any stream integrity failure:
        a binding row with NULL or non-finite ``fdr_p_value``/``fdr_alpha_level``, ``fdr_rejected``
        not in {0, 1}, NULL or non-positive ``fdr_test_index``, or non-contiguous indices
        (``fdr_test_index`` must equal 1, 2, …, t in order). Rows with ``fdr_binding`` NULL or 0
        are invisible to the stream (legacy-safe)."""
        ...

    def record_gate_with_fdr_and_maybe_promote(
        self,
        rec: StrategyRecord,
        *,
        gate_row: dict[str, Any],
        p_value: float | None,
        level_fn: Callable[[int, list[int]], float],
        actor: Actor,
        reason: str | None = None,
    ) -> FdrGateOutcome:
        """Record a gate evaluation WITH LORD++ FDR accounting and optionally promote to
        ``candidate`` — all under a single **BEGIN IMMEDIATE** transaction (write lock held from
        the stream-state SELECT through the INSERT + stage CAS).

        ``gate_row`` carries ``record_gate_evaluation``'s keyword arguments (including the
        provisional ``passed`` flag). ``p_value`` is ``1 − dsr_confidence`` when the row is
        FDR-binding (``dsr_binding=True`` AND ``dsr_confidence`` is finite); ``None`` means
        non-binding and FDR is skipped entirely for this row.

        When binding: reads the prior stream state UNDER the write lock, computes
        ``t_next = prior.t + 1``, ``α_t = level_fn(t_next, prior.discovery_indices)``,
        ``fdr_rejected = (p_value ≤ α_t)``, and ``final_passed = provisional_passed AND
        fdr_rejected``. The gate row is inserted with ``passed = final_passed`` (never the
        provisional value — ``find_consumable_gate_evaluation`` reads this column). If
        ``final_passed`` is True, ``rec`` is atomically advanced to ``candidate``.

        TOP-LEVEL ONLY: raises ``RuntimeError`` if called inside an open transaction (mirrors
        ``reserve_holdout``). Crash semantics: a process crash before commit rolls back both
        the FDR row and the stage CAS — no orphaned stream position, no half-promoted strategy.
        """
        ...

    # -------------------------------------------------------------------------
    # Factor-evaluation ledger (#219, slice E of #140)
    # -------------------------------------------------------------------------

    def record_factor_evaluation(
        self,
        *,
        factor_name: str,
        import_path: str,
        code_hash: str,
        hypothesis_hash: str,
        period_start: str,
        period_end: str,
        horizon: int,
        params_json: str,
        construction: str,
        construction_params_json: str,
        n_obs: int | None,
        mean_ic: float | None,
        ic_ir: float | None,
        t_stat: float | None,
        ic_skew: float | None,
        ic_kurtosis: float | None,
        n_dependents: int,
        data_source: str,
        snapshot_id: str | None,
        actor: str,
        created_at: str,
    ) -> int:
        """Persist a factor evaluation row (correction cols NULL until finalize). Returns row id."""
        ...

    def factor_hypothesis_breadth(
        self, factor_name: str, window_days: int
    ) -> tuple[int, int]:
        """Return ``(own_lifetime, windowed_total)`` distinct hypothesis counts.

        ``own_lifetime``: COUNT(DISTINCT hypothesis_hash) for this factor, all time.
        ``windowed_total``: COUNT(DISTINCT hypothesis_hash) across ALL factors within the
        trailing ``window_days`` (funnel-wide, analogous to windowed_search_combos).
        """
        ...

    def windowed_factor_irs(self, window_days: int) -> list[float]:
        """Latest finite ic_ir per distinct hypothesis_hash within the trailing window.

        Deduplicates to the most-recent row per hypothesis_hash, then filters to finite
        ic_ir values. Used to estimate pooled IR dispersion for the DSR layer.
        """
        ...

    def finalize_factor_evaluation(
        self,
        row_id: int,
        n_hypotheses: int,
        dsr_confidence: float | None,
        significant: bool,
    ) -> None:
        """Write the correction columns (n_hypotheses, dsr_confidence, significant) to the row."""
        ...
