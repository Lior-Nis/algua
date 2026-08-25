from __future__ import annotations

import json
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime

from algua.backtest.bootstrap import stable_bootstrap_seed, stationary_bootstrap_dsr
from algua.backtest.decision_path import verify_signal_panel_parity
from algua.backtest.neff import estimate_n_eff
from algua.backtest.walkforward import WalkForwardResult
from algua.contracts.lifecycle import Actor, Stage, TransitionError, validate_transition
from algua.contracts.types import DataProvider, assert_gated_costs
from algua.registry.approvals import compute_artifact_hashes
from algua.registry.family_assignment import classify_and_assign_family
from algua.registry.repository import (
    FamilyGraphDriftError,
    FunnelSnapshot,
    PendingNovelFamily,
    StrategyRepository,
)
from algua.registry.runs import provenance_of, record_walk_forward_run, walk_forward_metrics
from algua.research.gates import (
    DSR_ALPHA,
    DSR_BOOTSTRAP_LOWER_QUANTILE,
    DSR_BOOTSTRAP_RESAMPLES,
    FUNNEL_WINDOW_DAYS,
    MIN_CORR_OVERLAP_BARS,
    MIN_N_EFF_SIBLINGS,
    RHO_BAR_SHRINKAGE_K,
    GateCriteria,
    GateDecision,
    dsr_sr_star_annualized,
    effective_funnel_breadth,
    evaluate_gate,
)
from algua.strategies.loader import StrategyNotFound, load_strategy


def guard_agent_relaxations(
    actor: Actor,
    *,
    declared_combos: int | None,
    allow_holdout_reuse: bool,
    allow_non_pit: bool,
) -> None:
    """Every gate RELAXATION (declared breadth, holdout reuse, non-PIT) requires a human actor. An
    agent passing any is refused — the agent only ever sees the strict gate. Call EARLY
    (pre-peek)."""
    if actor is Actor.HUMAN:
        return
    if declared_combos is not None or allow_holdout_reuse or allow_non_pit:
        raise ValueError(
            "gate relaxation requires --actor human: --n-combos (declared breadth), "
            "--allow-holdout-reuse and --allow-non-pit are human-only. For an agent, breadth must "
            "be measured (run `backtest sweep`), the holdout fresh, and the universe PIT."
        )


def resolve_pit_ok(
    universe_name: str | None,
    universe_snapshots: list[dict[str, str]] | None,
    period_start: date,
) -> bool:
    """Wall B: PIT-valid iff a universe was used AND its earliest membership snapshot is effective
    on or before the backtest start (coverage, not mere presence)."""
    if universe_name is None or not universe_snapshots:
        return False
    # Fail closed: a malformed/missing effective_date means we CANNOT prove PIT coverage, so treat
    # it as non-PIT (not promotable) rather than raising after the holdout has been recorded.
    try:
        earliest = min(date.fromisoformat(s["effective_date"]) for s in universe_snapshots)
    except (KeyError, ValueError, TypeError):
        return False
    return earliest <= period_start


@dataclass
class BreadthContext:
    n_funnel: int
    own: int
    windowed_total: int
    provenance: str
    family_id: int | None = field(default=None)         # resolved family after classification
    expected_family_id: int | None = field(default=None)  # CAS token for run_gate (Task 5)
    # #524: a deferred agent-NOVEL family spec (family created only at the pass moment, never in
    # preflight). None for every non-agent-NOVEL case.
    pending_novel_family: PendingNovelFamily | None = field(default=None)


def promotion_preflight(
    repo: StrategyRepository,
    name: str,
    *,
    actor: Actor,
    declared_combos: int | None,
    allow_holdout_reuse: bool,
    allow_non_pit: bool,
    provider: DataProvider,
    start: datetime,
    end: datetime,
    new_family_slug: str | None = None,   # human-only, for NOVEL verdict
    universe_name: str | None = None,
    universe_by_date: Mapping[date, Collection[str]] | None = None,
) -> BreadthContext:
    """Pre-peek phase — runs BEFORE walk_forward, so every hard refusal happens before the holdout
    is touched and before any gate row is minted: (1) relaxations-need-human; (2) stage legality
    (BACKTESTED -> CANDIDATE must be legal — never mint a passing token for an illegal source
    stage); (3) exhaustive signal_panel parity gate (raises
    BacktestError on divergence, no-op when no signal_panel); (5) breadth resolution (refuse "no
    measured breadth" here)."""
    # FIRST check, before any holdout-affecting work and before the relaxation guard: only an agent
    # or a human may promote. SYSTEM is not a valid promote actor — it would pass as "not human"
    # (strict), burn the holdout, mint a gate_evaluations row it can NEVER consume (consumable rows
    # are filtered actor='agent'), leaving an orphaned token.
    if actor not in (Actor.AGENT, Actor.HUMAN):
        raise ValueError(
            f"research promote requires --actor agent or human, got {actor.value}")
    guard_agent_relaxations(actor, declared_combos=declared_combos,
                            allow_holdout_reuse=allow_holdout_reuse, allow_non_pit=allow_non_pit)
    # Reproducible-source guard (#205): an agent's holdout burn must be over reproducible bars — an
    # immutable snapshot (snapshot_id set) or a deterministic provider (reproducible marker) — so
    # the OOS truth it spends is identical on a re-run. Refuse a non-snapshot, non-reproducible
    # provider for an agent BEFORE any provider read (verify_signal_panel_parity below reads bars).
    # Humans are exempt (they accept the cost, mirroring --allow-non-pit). select_provider exposes
    # only demo/snapshot today; this fail-closes a future mutable/live provider. Duck-typed getattr
    # avoids a registry->data import-boundary violation; `is not True` (not just falsy) so a future
    # provider with a truthy-but-not-True `reproducible` cannot slip the guard (fail closed).
    if (actor is Actor.AGENT and getattr(provider, "snapshot_id", None) is None
            and getattr(provider, "reproducible", False) is not True):
        raise ValueError(
            "agent research promote requires a reproducible data source: an ingested snapshot "
            "(--snapshot) or a deterministic provider. A non-reproducible/live provider's bars may "
            "revise between runs; promote with --actor human to accept the cost.")
    rec = repo.get(name)
    # Source stage MUST be exactly BACKTESTED. validate_transition alone is too permissive here:
    # PAPER -> CANDIDATE is a legal back-step, so promoting from `paper` would otherwise pass
    # preflight, burn the holdout, and mint a token. Require backtested explicitly, then validate.
    if rec.stage is not Stage.BACKTESTED:
        raise TransitionError(
            f"research promote requires stage backtested, got {rec.stage.value}")
    validate_transition(rec.stage, Stage.CANDIDATE)  # TransitionError (a ValueError) if illegal
    # Load the strategy module for parity-check and breadth resolution below.
    # Silently skip if the strategy is not a bundled module (e.g. tests using synthetic names).
    try:
        _loaded = load_strategy(name)
    except StrategyNotFound:
        _loaded = None
    # Transaction-cost floor (#325): an agent may only promote a backtest evaluated at a realistic
    # transaction cost. The whole statistical stack (Sharpe/DSR/bootstrap/FDR/regime) is calibrated
    # on the return stream this promote will peek; if that stream were (near-)cost-free, a strategy
    # whose only edge is a cost illusion could clear every gate. assert_gated_costs fails closed on
    # a sub-floor fees+slippage BEFORE the holdout is touched, so a frictionless config can't mint
    # a gate token. Agent-only — a human accepts any cost (exploration / sensitivity sweeps).
    if actor is Actor.AGENT and _loaded is not None:
        assert_gated_costs(_loaded.execution)
    # Exhaustive signal_panel parity gate (#178): a panel that diverges from its per-bar signal on
    # ANY bar cannot pass promotion. Runs on the already-loaded strategy, in static mode over the
    # promotion window, BEFORE the holdout is touched. No-op when the strategy has no signal_panel.
    # Raises BacktestError on divergence (caught by the `promote` CLI's @json_errors).
    if _loaded is not None:
        verify_signal_panel_parity(_loaded, provider, start, end)
    # Walk-forward embargo declaration (#345): an agent's holdout is gated on a purge gap sized by
    # the strategy's declared feature lookback. An UNDECLARED lookback (feature_lookback is None)
    # resolves the embargo to 0 (legacy zero-gap, see _resolve_embargo), leaving a real
    # ~60-bar-lookback signal with NO purge and re-opening the leak on the gate-bearing path. Fail
    # closed:
    # the agent must declare feature_lookback (it may declare 0 for a strategy with no rolling
    # feature window). Humans accept the cost. Only enforceable for a bundled module (load_strategy
    # resolved); synthetic-name test strategies are left as-is, matching the parity gate above.
    if (actor is Actor.AGENT and _loaded is not None
            and _loaded.config.feature_lookback is None):
        raise ValueError(
            f"agent research promote requires {name!r} to declare feature_lookback (the bars of "
            f"trailing history its signal reads) so the walk-forward embargo can purge the "
            f"in-sample/holdout boundary (#345). Declare it on the strategy CONFIG (0 if the "
            f"signal has no rolling feature window); promote with --actor human to accept the "
            f"default.")
    # --- Family classification (clustering verdict, #222 / #524) ---
    # Runs BEFORE breadth resolution and BEFORE the holdout is touched. A strategy already assigned
    # to a family is left as-is. An agent NOVEL verdict is DEFERRED (#524): no family is created
    # here — classification returns a PendingNovelFamily spec, and the seeded family is minted only
    # at the pass moment inside the atomic promote tx. A human NOVEL still creates the fresh 0-prior
    # family in preflight (requires --new-family <slug>, the human privilege).
    classify = classify_and_assign_family(
        repo, name, actor=actor, new_family_slug=new_family_slug,
    )
    windowed_total = repo.windowed_search_combos(FUNNEL_WINDOW_DAYS)
    measured = repo.total_search_combos(name)
    if measured > 0:
        own, provenance = measured, "measured"
    elif declared_combos is not None:  # human-only path (already guarded above)
        own, provenance = declared_combos, "declared"
    else:
        raise ValueError(
            f"no recorded search breadth for {name!r}; run `algua backtest sweep {name} ...` "
            f"(records breadth). Declaring via --n-combos requires --actor human."
        )
    # Gated-universe subset guard (#559): an AGENT promote on a PIT universe fails closed — still
    # BEFORE the holdout is touched and before any gate row is minted — when the loaded strategy's
    # CONFIG.universe contains a symbol that was NEVER a member of the gated universe's membership
    # timeline (the union over the same universe_by_date map the PIT walk-forward consumes). The
    # module's universe is agent-authored and can silently diverge from the universe the evidence
    # was produced on (e.g. a 5-symbol template incl. NVDA gated on liquid10); at deployment the
    # tick binds to the GATE universe, so a config symbol outside it was never validated at all.
    # Humans are exempt (exploration), mirroring the other agent-only integrity walls. Placed
    # AFTER the structural refusals above so the established refusal precedence (stage, costs,
    # parity, lookback, classification, breadth) is unchanged.
    if actor is Actor.AGENT and _loaded is not None and universe_by_date:
        gated_members: set[str] = set().union(*universe_by_date.values())
        offending = sorted(set(_loaded.config.universe) - gated_members)
        if offending:
            raise ValueError(
                f"strategy {name!r} CONFIG.universe contains symbols that were never members of "
                f"the gated universe {universe_name!r}: {offending}. The paper deployment is "
                f"bound to the gated universe (#559) — fix CONFIG.universe or gate on a universe "
                f"that covers it."
            )
    ctx = BreadthContext(effective_funnel_breadth(own, windowed_total), own, windowed_total,
                         provenance)
    ctx.family_id = classify.family_id
    ctx.expected_family_id = classify.family_id  # run_gate will CAS-verify this (Task 5)
    ctx.pending_novel_family = classify.pending_novel_family
    # #524: as the LAST preflight step BEFORE the holdout peek, re-validate a pending agent-NOVEL
    # spec so every hard refusal burns NO holdout and mints NO consumable gate row (R6-HIGH-3,
    # R7-HIGH-1, R8-HIGH): (1) the per-window rate cap, and (2) still-unassigned + the
    # full-classifier-read-set fingerprint unchanged since classification. The under-lock re-checks
    # in record_gate_with_fdr_and_maybe_promote remain the authoritative race-safe guards.
    if classify.pending_novel_family is not None:
        _revalidate_pending_novel(repo, name, classify.pending_novel_family)
    return ctx


def _revalidate_pending_novel(
    repo: StrategyRepository, name: str, pending: PendingNovelFamily,
) -> None:
    """Fail-closed re-check of a deferred agent-NOVEL spec: the per-window rate cap AND the
    still-unassigned + graph-fingerprint snapshot (#524). Raises ``AgentMintCapError`` /
    ``FamilyGraphDriftError``. All are pure DB reads (no holdout, no gate row), safe to call both
    pre-peek and at the atomic burn."""
    repo.check_agent_novel_mint_bounds()
    # #532 (Finding 3a): fail closed on a non-positive funnel-lifetime breadth seed BEFORE the
    # holdout peek. The authoritative guard lives under the promote write lock in
    # _mint_agent_novel_family (step 5a), but that runs AFTER the holdout is burned; a promote that
    # is certain to fail the seed guard would otherwise spend its single-use holdout before the mint
    # rejects it. This pure read is the early advisory copy (defense-in-depth, not a replacement — a
    # concurrent search_trials change between here and commit is still caught under the lock).
    if repo.agent_novel_mint_seed() <= 0:
        raise ValueError(
            f"strategy {name!r}: agent-NOVEL mint requires a strictly-positive funnel-lifetime "
            "breadth seed; the funnel has no well-typed in-range search_trials (fail closed)"
        )
    if repo.strategy_family(name) is not None:
        raise FamilyGraphDriftError(
            f"strategy {name!r} assigned to a family since NOVEL classification; re-run promote",
            axis="still_assigned")
    if repo.family_graph_fingerprint() != pending.graph_fingerprint:
        raise FamilyGraphDriftError(
            f"family graph changed since {name!r} was classified NOVEL; re-run promote",
            axis="graph_fingerprint")


@dataclass
class PromotionOutcome:
    decision: GateDecision
    promoted: bool


def run_gate(
    repo: StrategyRepository,
    wf: WalkForwardResult,
    *,
    name: str,
    actor: Actor,
    criteria: GateCriteria,
    breadth: BreadthContext,
    universe_name: str | None,
    universe_snapshots: list[dict[str, str]] | None,
    period_start: date,
    period_end: date,
    holdout_frac: float,
    data_source: str,
    snapshot_id: str | None,
    allow_non_pit: bool,
    reason_suffix: str,
    holdout_evaluation_id: int | None = None,
    attempt_token: str | None = None,
) -> PromotionOutcome:
    """Post-walk phase: resolve PIT, evaluate, record the gate_evaluations row (pass AND fail), and
    on pass transition BACKTESTED->CANDIDATE (which consumes the just-minted agent token).
    Identity is recomputed via compute_artifact_hashes(name) — the SAME function the shortlist gate
    matches against (NOT wf.code_hash, which is git-HEAD-based and would never match)."""
    pit_ok = resolve_pit_ok(universe_name, universe_snapshots, period_start)
    holdout_n_bars = int(wf.holdout_metrics["n_bars"])
    # Resolve family breadth for the 3-way max (breadth snapshotted here, not in preflight).
    # Re-query live DB state (not in-memory BreadthContext) so the CAS below detects concurrent
    # re-assignments between preflight and run_gate (R2-F5 concurrency safety).
    family_id = repo.strategy_family(name)
    family_lifetime_effective = (
        repo.family_lifetime_combos(family_id) if family_id is not None else 0
    )
    # #524: pending agent-NOVEL — pre-lock mirror of the under-lock step-1 CAS (a fast, lock-free
    # early reject). The founder is evaluated with family arm 0 / family_id None (the family does
    # not exist yet); require it is STILL unassigned AND the classifier read-set fingerprint is
    # unchanged since classification, else re-run preflight. The authoritative re-checks are the
    # under-lock step-1 in record_gate_with_fdr_and_maybe_promote and the atomic on_peek re-check.
    if breadth.pending_novel_family is not None:
        if family_id is not None:
            raise FamilyGraphDriftError(
                f"strategy {name!r} was assigned to a family since NOVEL classification; re-run "
                "promote", axis="still_assigned")
        if repo.family_graph_fingerprint() != breadth.pending_novel_family.graph_fingerprint:
            raise FamilyGraphDriftError(
                f"family graph changed since {name!r} was classified NOVEL; re-run promote",
                axis="graph_fingerprint")
    # CAS: verify the family hasn't changed since preflight (concurrent-preflight safety R2-F5).
    elif breadth.expected_family_id is not None and family_id != breadth.expected_family_id:
        raise ValueError(
            f"family assignment changed between preflight and gate evaluation "
            f"(expected {breadth.expected_family_id}, got {family_id}); re-run promote."
        )
    # 3-way max: recompute final n_funnel including family dimension (overrides breadth.n_funnel,
    # which was computed in preflight without the family component).
    n_funnel = effective_funnel_breadth(
        breadth.own, breadth.windowed_total, family_lifetime_effective,
    )
    # DSR evidence (#211): armed/evaluated iff breadth is MEASURED (the advisory dsr_evidence
    # check is appended only then). Declared breadth (human, no sweep) omits DSR entirely.
    dsr_binding = breadth.provenance == "measured"
    dsr_trial_var_ann = repo.pooled_trial_sharpe_var(name) if dsr_binding else None
    funnel_floor = repo.funnel_trial_sharpe_var(FUNNEL_WINDOW_DAYS) if dsr_binding else None
    # Serial-dependence bootstrap (#221 Slice 2): bind iff measured AND the in-process OOS vector
    # is present. Recompute DSR confidence against the SAME floored SR* the closed form uses;
    # gates.py gets only the pre-computed scalar (it does no resampling).
    # NOTE: pre-existing measured promotion tests that supply holdout_returns also exercise this
    # bootstrap path — a future reviewer adding a `checks` assertion for dsr_bootstrap should
    # account for that.
    holdout_rets = wf.holdout_returns  # local binding so mypy can narrow the tuple type below
    bootstrap_binding = dsr_binding and holdout_rets is not None
    boot_lower = boot_seed = boot_b = boot_block = None
    if bootstrap_binding and holdout_rets is not None:  # second guard narrows tuple type for mypy
        sr_star_pp = dsr_sr_star_annualized(
            n_funnel, dsr_trial_var_ann, funnel_floor.var_ann if funnel_floor else None)
        boot_seed = stable_bootstrap_seed(
            name, wf.holdout_metrics["start"], wf.holdout_metrics["end"], wf.config_hash)
        boot = stationary_bootstrap_dsr(
            holdout_rets[0], holdout_rets[1], sr_star_pp, DSR_ALPHA,
            DSR_BOOTSTRAP_RESAMPLES, boot_seed, lower_quantile=DSR_BOOTSTRAP_LOWER_QUANTILE)
        boot_lower, boot_b, boot_block = boot.lower_confidence, boot.b_used, boot.block_len
    decision = evaluate_gate(
        wf, criteria, n_combos=n_funnel, breadth_provenance=breadth.provenance,
        pit_ok=pit_ok, allow_non_pit=allow_non_pit, own_lifetime_combos=breadth.own,
        windowed_total_combos=breadth.windowed_total, funnel_window_days=FUNNEL_WINDOW_DAYS,
        dsr_binding=dsr_binding, dsr_trial_var_ann=dsr_trial_var_ann,
        dsr_funnel_floor_var_ann=(funnel_floor.var_ann if funnel_floor else None),
        dsr_funnel_floor_n_strategies=(funnel_floor.n_strategies if funnel_floor else None),
        dsr_funnel_floor_n_total_rows=(funnel_floor.n_total_rows if funnel_floor else None),
        bootstrap_binding=bootstrap_binding, bootstrap_lower_confidence=boot_lower,
        bootstrap_seed=boot_seed, bootstrap_b=boot_b, bootstrap_block_len=boot_block,
        market_returns=wf.market_returns,
    )
    # Factory soft gate (2026-08-10 spec): the statistical stack is ADVISORY. The LORD++
    # FDR-binding branch itself was deleted (simplification stage 4a — it was provably dead here,
    # this path never supplied a real p_value); record_gate_with_fdr_and_maybe_promote's
    # final_passed is now always the provisional integrity-floor verdict. dsr_confidence is still
    # computed and recorded in the decision above (advisory telemetry).
    decision.fdr_binding = False
    decision.fdr_skip_reason = "stats_advisory"

    identity = compute_artifact_hashes(name)
    rec = repo.get(name)

    # Effective independent trials N_eff (#221 Slice 3) — SHADOW-ONLY: recorded for the audit trail,
    # NEVER passed as the binding DSR trial count (a lower N_eff would loosen the gate; it goes
    # binding only at Slice 5 with haircut retirement). Sibling-only read (excludes own vector).
    # Guarded on holdout_metrics["start"/"end"] presence (omit-not-fail for legacy fixtures
    # that pre-date Slice 1 and don't carry the OOS interval in the WalkForwardResult).
    h_start_neff = wf.holdout_metrics.get("start")
    h_end_neff = wf.holdout_metrics.get("end")
    if dsr_binding and h_start_neff is not None and h_end_neff is not None:
        siblings = repo.overlapping_holdout_return_streams(
            rec.id, h_start_neff, h_end_neff, FUNNEL_WINDOW_DAYS)
        neff = estimate_n_eff(
            n_funnel, siblings, min_siblings=MIN_N_EFF_SIBLINGS,
            min_overlap_bars=MIN_CORR_OVERLAP_BARS, shrinkage_k=RHO_BAR_SHRINKAGE_K)
        decision.dsr_n_eff = neff.n_eff
        decision.dsr_rho_bar = neff.rho_bar
        decision.dsr_n_siblings = neff.n_siblings

    # Persist the OOS return vector for this burn (#221 Slice 1) — separate tx from the burn
    # (which committed at on_peek). Written on EVERY burn (pass or fail): the holdout was
    # revealed, so the vector exists and funnel siblings may use it. gates.py never sees the
    # vector; promotion is the sole writer. A missing row for a committed burn is a recoverable
    # inconsistency (UNIQUE guards a re-run). returns_available feeds Slices 2-4
    # (omit-not-fail for pre-Slice-1 promotions).
    returns_available = False
    holdout_returns_id: int | None = None
    if holdout_evaluation_id is not None and wf.holdout_returns is not None:
        rets, bar_dates = wf.holdout_returns
        if not (len(rets) == len(bar_dates) == holdout_n_bars):
            raise ValueError(
                f"holdout_returns length {len(rets)}/{len(bar_dates)}"
                f" != holdout n_bars {holdout_n_bars}")
        holdout_returns_id = repo.record_holdout_returns(
            holdout_evaluation_id, rec.id,
            holdout_start=wf.holdout_metrics["start"], holdout_end=wf.holdout_metrics["end"],
            returns=rets, bar_dates=bar_dates)
        returns_available = True
    decision.returns_available = returns_available

    # The walk-forward this decision was computed on, as its own run row. Recorded here rather
    # than by the CLI because `research promote` runs the walk-forward internally — this is the
    # only place its result exists. Routed through the shared recorder (not an inline
    # repo.record_run call) so a `walk_forward` row has ONE shape regardless of which command
    # produced it.
    wf_run_id = record_walk_forward_run(repo, rec.name, wf, strategy_id=rec.id)

    # Build gate_row (all record_gate_evaluation kwargs, including provisional passed flag).
    gate_row = {
        "passed": decision.passed,
        "n_funnel": n_funnel,
        "own_lifetime_combos": breadth.own,
        "windowed_total_combos": breadth.windowed_total,
        "funnel_window_days": FUNNEL_WINDOW_DAYS,
        "breadth_provenance": breadth.provenance,
        "pit_ok": bool(decision.pit_ok),
        "pit_override": bool(decision.pit_override),
        "holdout_n_bars": holdout_n_bars,
        "min_holdout_observations": criteria.min_holdout_observations,
        "code_hash": identity.code_hash,
        "config_hash": identity.config_hash,
        "dependency_hash": identity.dependency_hash,
        "data_source": data_source,
        "snapshot_id": snapshot_id,
        # #559: the gated-universe identity, stamped on every row (pass AND fail) so a deployment
        # can be bound to the universe its gate evidence was produced on. NULL for non-universe
        # runs (the tick binding treats NULL as config_legacy).
        "universe_name": universe_name,
        "period_start": period_start.isoformat(),
        "period_end": period_end.isoformat(),
        "holdout_frac": holdout_frac,
        "decision_json": json.dumps(decision.to_dict(), sort_keys=True),
        "family_id": family_id,
        "family_lifetime_effective": family_lifetime_effective,
        # PIT sidecar provenance (#132): the snapshot ids the walk-forward/holdout consumed, so the
        # gate audit row records exactly which fundamentals/news snapshot fed a needs_* promotion.
        "fundamentals_snapshot": wf.fundamentals_snapshot,
        "news_snapshot": wf.news_snapshot,
        # #485: opaque per-attempt idempotency key from the autonomous merge-back driver (NULL for
        # every ordinary caller), stamped on the gate row so promote-outcome attribution binds to
        # the branch identity, not the ambient stage.
        "attempt_token": attempt_token,
    }

    # Atomic gate-record-and-maybe-promote — always uses BEGIN IMMEDIATE for consistency
    # (negligible overhead for ≤ a few thousand gate_evaluations rows).
    # #339 — capture the funnel-wide MUTABLE snapshot this decision was computed against, so the
    # commit can CAS-verify it under the write lock and refuse to serialize a mixed-snapshot
    # (stale-breadth / stale-variance) outcome. own/windowed are the exact values evaluate_gate
    # used (from preflight); family + variances are the run_gate reads above; the search_trials
    # fingerprint (captured here) is the append-only row-identity guard. Any drift by commit -> the
    # store rolls back and raises FunnelDriftError, and the operator re-runs against fresh state.
    st_count, st_max = repo.search_trials_fingerprint()
    funnel = FunnelSnapshot(
        strategy_name=name,
        funnel_window_days=FUNNEL_WINDOW_DAYS,
        dsr_binding=dsr_binding,
        own_lifetime_combos=breadth.own,
        windowed_total_combos=breadth.windowed_total,
        family_id=family_id,
        family_lifetime_effective=family_lifetime_effective,
        dsr_trial_var_ann=dsr_trial_var_ann,
        funnel_floor_var_ann=(funnel_floor.var_ann if funnel_floor else None),
        funnel_floor_n_strategies=(funnel_floor.n_strategies if funnel_floor else 0),
        funnel_floor_n_total_rows=(funnel_floor.n_total_rows if funnel_floor else 0),
        search_trials_count=st_count,
        search_trials_max_id=st_max,
    )
    fdr_outcome = repo.record_gate_with_fdr_and_maybe_promote(
        rec, gate_row=gate_row, funnel=funnel, actor=actor,
        reason=(_gate_reason(decision) + reason_suffix) if decision.passed else None,
        pending_novel_family=breadth.pending_novel_family,  # #524: minted only on pass, in-tx
        # The same decision as an economic-layer run row. PASS AND FAIL — the rejections are the
        # dataset the IS-vs-OOS scatter is mostly made of.
        run_row={
            "strategy_id": rec.id,
            "derived_from": [wf_run_id],
            # NOTE: no "passed" key here — record_gate_with_fdr_and_maybe_promote (store/gate.py)
            # unconditionally overwrites it with final_passed, the only value known once this
            # transaction's checks have run. Setting it here would be dead and misleading.
            #
            # Base provenance off the SIBLING walk-forward run's own provenance_of(wf), not a
            # hand-copied field list: the two rows describe the SAME evaluation and must not drift
            # (a hand-copy here had already dropped seed/timeframe, silently). Override only the
            # fields that are GENUINELY different for the gate: code_hash/config_hash/
            # dependency_hash come from `identity`, recomputed via compute_artifact_hashes(name) —
            # deliberately NOT wf's own git-HEAD-based hashes (see run_gate's docstring); the
            # gate's data_source/snapshot_id/universe_name/period_start/period_end are read from
            # the SAME local variables that were passed into walk_forward(), so they are already
            # identical to wf's — provenance_of(wf) carries them without an override.
            "provenance": provenance_of(wf) | {
                "code_hash": identity.code_hash,
                "config_hash": identity.config_hash,
                "dependency_hash": identity.dependency_hash,
            },
            "metrics": walk_forward_metrics(wf),
            # The ~40 DSR / IR / regime diagnostics: queryable, but deliberately outside the fixed
            # vocabulary. Finite scalars only — decision.to_dict() also carries strings, lists and
            # bools, and `bool` is an `int` subclass so it must be excluded explicitly.
            "extra_metrics": {
                k: float(v) for k, v in decision.to_dict().items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            },
            # v44: the holdout_returns row this burn's OOS vector was written to above (NULL for a
            # pre-Slice-1 promotion with no returns_available) — resolves `runs series` to it.
            "series_holdout_id": holdout_returns_id,
        },
    )

    # With the LORD++ binding branch retired (stage 4a), final_passed == provisional_passed (the
    # integrity floor) unconditionally. Kept as an assignment (not an assert) so the decision
    # always mirrors what the store committed.
    decision.passed = fdr_outcome.final_passed

    return PromotionOutcome(decision=decision, promoted=fdr_outcome.final_passed)


def _gate_reason(decision: GateDecision) -> str:
    """Human-readable gate summary. Metric checks render value/op/threshold; boolean checks (e.g.
    pit_required) render name=pass|fail."""
    parts: list[str] = []
    for c in decision.checks:
        if "value" in c and c.get("value") is not None and c.get("threshold") is not None:
            parts.append(f"{c['name']}={c['value']:.4g}{c['op']}{c['threshold']:.4g}")
        else:
            parts.append(f"{c['name']}={'pass' if c['passed'] else 'fail'}")
    breadth = (
        f"; funnel_breadth={decision.n_combos}({decision.breadth_provenance}"
        f"; own={decision.own_lifetime_combos}, windowed={decision.windowed_total_combos}"
        f", window={decision.funnel_window_days}d)"
        f"; min_holdout_sharpe={decision.base_min_holdout_sharpe:.4g}"
        f"->{decision.effective_min_holdout_sharpe:.4g}"
        if decision.n_combos is not None else ""
    )
    return "gate pass: " + ", ".join(parts) + breadth
