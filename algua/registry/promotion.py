from __future__ import annotations

import functools
import json
import math
from dataclasses import dataclass, field
from datetime import date, datetime

from algua.backtest.bootstrap import stable_bootstrap_seed, stationary_bootstrap_dsr
from algua.backtest.engine import verify_signal_panel_parity
from algua.backtest.walkforward import WalkForwardResult
from algua.contracts.lifecycle import Actor, Stage, TransitionError, validate_transition
from algua.contracts.types import DataProvider
from algua.registry.approvals import compute_artifact_hashes
from algua.registry.lineage import factors_used_by
from algua.registry.repository import StrategyRepository
from algua.research.clustering import (
    MERGE_THRESHOLD,
    PARENTAGE_THRESHOLD,
    WEIGHT_CODE_ANCESTRY,
    WEIGHT_FACTOR_LINEAGE,
    WEIGHT_RETURN_CORRELATION,
    SimVerdict,
    clustering_version,
    family_similarity,
)
from algua.research.gates import (
    DSR_ALPHA,
    DSR_BOOTSTRAP_LOWER_QUANTILE,
    DSR_BOOTSTRAP_RESAMPLES,
    FDR_ALPHA,
    FDR_W0,
    FUNNEL_WINDOW_DAYS,
    GateCriteria,
    GateDecision,
    dsr_sr_star_annualized,
    effective_funnel_breadth,
    evaluate_gate,
    lord_plus_plus_level,
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


def _get_all_family_members_for_clustering(
    repo: StrategyRepository,
) -> list[tuple[int, list[dict]]]:
    """Return [(family_id, members_list)] for all families with active members.

    Each member dict: {"code_hash": str, "factors": set[str]}.
    Delegates to the repository so the Protocol seam is respected.
    """
    return repo.all_families_with_member_profiles()


def _classify_and_assign_family(
    repo: StrategyRepository,
    name: str,
    *,
    actor: Actor,
    new_family_slug: str | None,
) -> int | None:
    """Classify ``name`` against all known families and assign it if needed.

    Returns the resolved family_id (or None only if it somehow stays unassigned, which
    should not happen — every branch either raises or assigns).

    Decision tree:
    - Already assigned: return current family_id, no re-classification.
    - MERGE: assign to best-matching family.
    - PARENTAGE + agent: fold into best parent family (agents cannot mint child families).
    - PARENTAGE + human: create a child family with a parent edge, assign there.
    - NOVEL + agent: raise ValueError (agents cannot mint new families).
    - NOVEL + human: create a new root family using new_family_slug (required); assign.
    """
    current_family_id = repo.strategy_family(name)
    if current_family_id is not None:
        # Already assigned — skip reclassification, keep existing assignment.
        return current_family_id

    # Get the strategy's identity for clustering comparison. A strategy whose module cannot be
    # loaded (e.g. test-only names) gets code_hash="" and factors=set(): it will never match an
    # existing member's real code_hash, so these strategies get NOVEL verdict (fail-closed).
    try:
        strategy_code_hash = compute_artifact_hashes(name).code_hash
    except StrategyNotFound:
        strategy_code_hash = ""
    try:
        factor_specs = factors_used_by(name)
        # factors_used_by returns list[FactorSpec]; get names.
        strategy_factors: set[str] = {
            f.name if hasattr(f, "name") else str(f) for f in factor_specs
        }
    except Exception:  # noqa: BLE001 — unregistered/test strategies silently get no factors
        strategy_factors = set()

    best_family_id: int | None = None
    best_verdict = SimVerdict.NOVEL
    best_score = 0.0

    # Load family data once; also collect member names for return-correlation axis
    all_family_data = _get_all_family_members_for_clustering(repo)
    all_member_names: list[str] = [
        m["name"]
        for _fid, members in all_family_data
        for m in members
        if "name" in m
    ]

    # Build returns_lookup from stored backtest returns so the correlation axis is live
    # whenever prior run() or sweep results have been persisted (#222, Task 7)
    returns_lookup: dict[str, object] = {}
    strategy_stored_returns = repo.load_backtest_returns(name)
    if strategy_stored_returns is not None:
        returns_lookup["__strategy__"] = strategy_stored_returns
    for member_name in all_member_names:
        if member_name not in returns_lookup:
            member_returns = repo.load_backtest_returns(member_name)
            if member_returns is not None:
                returns_lookup[member_name] = member_returns

    has_any_family = False
    for fam_id, members in all_family_data:
        has_any_family = True
        verdict, score = family_similarity(
            strategy_code_hash, strategy_factors, members,
            returns_lookup=returns_lookup or None,
        )
        if score > best_score or (score == best_score and best_verdict == SimVerdict.NOVEL):
            best_score = score
            best_verdict = verdict
            best_family_id = fam_id

    cv = clustering_version()
    clustering_config_json = json.dumps({
        "version": cv,
        "merge_threshold": MERGE_THRESHOLD,
        "parentage_threshold": PARENTAGE_THRESHOLD,
        "weights": {
            "code_ancestry": WEIGHT_CODE_ANCESTRY,
            "factor_lineage": WEIGHT_FACTOR_LINEAGE,
            "return_correlation": WEIGHT_RETURN_CORRELATION,
        },
    }, sort_keys=True)
    axis_json = json.dumps({
        "verdict": best_verdict.value,
        "score": best_score,
        "has_returns_data": bool(returns_lookup),
    }, sort_keys=True)

    def _do_assign(target_family_id: int, *, matched_family_id: int | None = None) -> None:
        repo.assign_strategy_to_family(
            name, target_family_id, actor=actor.value,
            verdict=best_verdict.value, similarity_score=best_score,
            clustering_version=cv, clustering_config_json=clustering_config_json,
            axis_json=axis_json,
            matched_family_id=matched_family_id if matched_family_id is not None
            else target_family_id,
        )

    if best_verdict == SimVerdict.MERGE:
        assert best_family_id is not None
        _do_assign(best_family_id)
        return best_family_id

    if best_verdict == SimVerdict.PARENTAGE:
        assert best_family_id is not None
        if actor is Actor.AGENT:
            # Agent cannot mint a child family. Fold into the best parent.
            _do_assign(best_family_id)
            return best_family_id
        else:
            # Human: create a child family, add a parent edge, assign.
            child_name = new_family_slug or f"{name}_family"
            child_fam_id = repo.create_family(child_name, actor=actor.value,
                                               created_by_strategy=name)
            repo.add_parent_edge(child_fam_id, best_family_id)
            _do_assign(child_fam_id, matched_family_id=best_family_id)
            return child_fam_id

    # NOVEL verdict
    if actor is Actor.AGENT:
        if not has_any_family:
            raise ValueError(
                f"strategy {name!r}: the family registry is empty — no existing family to "
                "merge into. A human operator must create the first family via "
                "`research promote --actor human --new-family <slug>` before agents can promote."
            )
        raise ValueError(
            f"strategy {name!r} has no matching family (clustering verdict: NOVEL). "
            "Assign to a family or use --actor human with --new-family <slug>."
        )
    else:
        # Human: create a new root family using the provided slug.
        if new_family_slug is None:
            raise ValueError(
                f"strategy {name!r}: clustering verdict is NOVEL (no matching family). "
                "Provide --new-family <slug> to create a new family."
            )
        new_fam_id = repo.create_family(new_family_slug, actor=actor.value,
                                         created_by_strategy=name)
        _do_assign(new_fam_id, matched_family_id=None)
        return new_fam_id


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
) -> BreadthContext:
    """Pre-peek phase — runs BEFORE walk_forward, so every hard refusal happens before the holdout
    is touched and before any gate row is minted: (1) relaxations-need-human; (2) stage legality
    (BACKTESTED -> CANDIDATE must be legal — never mint a passing token for an illegal source
    stage); (3) fundamentals-lane guard; (4) exhaustive signal_panel parity gate (raises
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
    # Fundamentals strategies cannot be promoted past backtested until the paper/live fundamentals
    # lane exists (#132): block the agent's only path to candidate early, with a clear message.
    # Silently skip if the strategy is not a bundled module (e.g. tests using synthetic names).
    try:
        _loaded = load_strategy(name)
    except StrategyNotFound:
        _loaded = None
    if _loaded is not None and _loaded.config.needs_fundamentals:
        raise ValueError(
            f"strategy {name!r} declares needs_fundamentals; it cannot be promoted past "
            f"backtested until the paper/live fundamentals lane is built (#132)"
        )
    if _loaded is not None and _loaded.config.needs_news:
        raise ValueError(
            f"strategy {name!r} declares needs_news; it cannot be promoted past "
            f"backtested until the paper/live news lane is built (#132)"
        )
    # Exhaustive signal_panel parity gate (#178): a panel that diverges from its per-bar signal on
    # ANY bar cannot pass promotion. Runs on the already-loaded strategy, in static mode over the
    # promotion window, BEFORE the holdout is touched. No-op when the strategy has no signal_panel.
    # Raises BacktestError on divergence (caught by the `promote` CLI's @json_errors).
    if _loaded is not None:
        verify_signal_panel_parity(_loaded, provider, start, end)
    # --- Family classification (clustering verdict, #222) ---
    # Runs BEFORE breadth resolution (no holdout has been touched yet). A strategy already
    # assigned to a family is left as-is (no reclassification). A NOVEL verdict refuses an
    # agent (agents cannot mint new families). A human NOVEL requires --new-family <slug>.
    family_id_resolved = _classify_and_assign_family(
        repo, name, actor=actor, new_family_slug=new_family_slug
    )
    measured = repo.total_search_combos(name)
    windowed_total = repo.windowed_search_combos(FUNNEL_WINDOW_DAYS)
    if measured > 0:
        own, provenance = measured, "measured"
    elif declared_combos is not None:  # human-only path (already guarded above)
        own, provenance = declared_combos, "declared"
    else:
        raise ValueError(
            f"no recorded search breadth for {name!r}; run `algua backtest sweep {name} ...` "
            f"(records breadth). Declaring via --n-combos requires --actor human."
        )
    ctx = BreadthContext(effective_funnel_breadth(own, windowed_total), own, windowed_total,
                         provenance)
    ctx.family_id = family_id_resolved
    ctx.expected_family_id = family_id_resolved  # run_gate will CAS-verify this (Task 5)
    return ctx


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
    # CAS: verify the family hasn't changed since preflight (concurrent-preflight safety R2-F5).
    if breadth.expected_family_id is not None and family_id != breadth.expected_family_id:
        raise ValueError(
            f"family assignment changed between preflight and gate evaluation "
            f"(expected {breadth.expected_family_id}, got {family_id}); re-run promote."
        )
    # 3-way max: recompute final n_funnel including family dimension (overrides breadth.n_funnel,
    # which was computed in preflight without the family component).
    n_funnel = effective_funnel_breadth(
        breadth.own, breadth.windowed_total, family_lifetime_effective,
    )
    # DSR evidence (#211): binding iff breadth is MEASURED. Declared breadth (human, no sweep)
    # omits DSR — and consequently FDR too (p_value requires a finite dsr_confidence).
    dsr_binding = breadth.provenance == "measured"
    dsr_trial_var_ann = repo.pooled_trial_sharpe_var(name) if dsr_binding else None
    funnel_floor = repo.funnel_trial_sharpe_var(FUNNEL_WINDOW_DAYS) if dsr_binding else None
    # Serial-dependence bootstrap (#221 Slice 2): bind iff measured AND the in-process OOS vector
    # is present. Recompute DSR confidence against the SAME floored SR* the closed form uses;
    # gates.py gets only the pre-computed scalar (it does no resampling).
    holdout_rets = wf.holdout_returns  # local binding so mypy can narrow the tuple type
    bootstrap_binding = dsr_binding and holdout_rets is not None
    boot_lower = boot_seed = boot_b = boot_block = None
    if bootstrap_binding and holdout_rets is not None:
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
    )
    # LORD++ FDR binding (#220): dsr_confidence must be finite (not None and not ±inf).
    # p = 1 − dsr_confidence is P(SR_true ≤ SR*) under the DSR null — an explicit conversion
    # here guards the ≥/≤ inversion hazard (see GATE-1 finding H3 in the design doc).
    dsr_conf = decision.dsr_confidence
    if dsr_conf is not None and not (0.0 <= dsr_conf <= 1.0):
        raise ValueError(
            f"dsr_confidence={dsr_conf!r} is outside [0, 1]; this is a DSR computation bug"
        )
    fdr_binding_this_row = (
        dsr_binding and dsr_conf is not None and math.isfinite(dsr_conf)
    )
    p_value = (1.0 - dsr_conf) if (fdr_binding_this_row and dsr_conf is not None) else None

    # Pre-populate non-binding FDR skip reason before decision_json is serialized so the DB
    # audit record matches the in-memory GateDecision (binding fields are unknown until the
    # store call and are patched there).
    if not fdr_binding_this_row:
        decision.fdr_binding = False
        if not dsr_binding:
            decision.fdr_skip_reason = "no_measured_dispersion"
        else:
            decision.fdr_skip_reason = "no_dsr_confidence"

    identity = compute_artifact_hashes(name)
    rec = repo.get(name)

    # Persist the OOS return vector for this burn (#221 Slice 1) — separate tx from the burn
    # (which committed at on_peek). Written on EVERY burn (pass or fail): the holdout was
    # revealed, so the vector exists and funnel siblings may use it. gates.py never sees the
    # vector; promotion is the sole writer. A missing row for a committed burn is a recoverable
    # inconsistency (UNIQUE guards a re-run). returns_available feeds Slices 2-4
    # (omit-not-fail for pre-Slice-1 promotions).
    returns_available = False
    if holdout_evaluation_id is not None and wf.holdout_returns is not None:
        rets, bar_dates = wf.holdout_returns
        if not (len(rets) == len(bar_dates) == holdout_n_bars):
            raise ValueError(
                f"holdout_returns length {len(rets)}/{len(bar_dates)}"
                f" != holdout n_bars {holdout_n_bars}")
        repo.record_holdout_returns(
            holdout_evaluation_id, rec.id,
            holdout_start=wf.holdout_metrics["start"], holdout_end=wf.holdout_metrics["end"],
            returns=rets, bar_dates=bar_dates)
        returns_available = True
    decision.returns_available = returns_available

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
        "period_start": period_start.isoformat(),
        "period_end": period_end.isoformat(),
        "holdout_frac": holdout_frac,
        "decision_json": json.dumps(decision.to_dict(), sort_keys=True),
        "family_id": family_id,
        "family_lifetime_effective": family_lifetime_effective,
    }

    # Atomic FDR-test-and-maybe-promote. For non-binding rows (p_value=None), the method
    # behaves identically to the old record_gate_evaluation + conditional transition_strategy
    # pair, but always uses BEGIN IMMEDIATE for consistency (negligible overhead for ≤ a few
    # thousand gate_evaluations rows).
    level_fn = functools.partial(lord_plus_plus_level, alpha=FDR_ALPHA, w0=FDR_W0)
    fdr_outcome = repo.record_gate_with_fdr_and_maybe_promote(
        rec, gate_row=gate_row, p_value=p_value, level_fn=level_fn, actor=actor,
        reason=(_gate_reason(decision) + reason_suffix) if decision.passed else None,
    )

    # Fold binding FDR audit fields into the GateDecision so they surface in to_dict() → CLI
    # JSON. Non-binding fields (fdr_binding=False, fdr_skip_reason) were set above, before
    # decision_json was serialized, so the DB record and the return value are consistent.
    if fdr_binding_this_row:
        decision.fdr_binding = True
        decision.fdr_p_value = fdr_outcome.fdr_p_value
        decision.fdr_alpha_level = fdr_outcome.fdr_alpha_level
        decision.fdr_test_index = fdr_outcome.fdr_test_index
        decision.fdr_rejected = fdr_outcome.fdr_rejected
        decision.checks.append({
            "name": "fdr_evidence",
            "value": fdr_outcome.fdr_p_value,
            "threshold": fdr_outcome.fdr_alpha_level,
            "op": "<=",
            "passed": bool(fdr_outcome.fdr_rejected),
        })
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
