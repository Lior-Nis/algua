from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime

from algua.backtest.engine import verify_signal_panel_parity
from algua.backtest.walkforward import WalkForwardResult
from algua.contracts.lifecycle import Actor, Stage, TransitionError, validate_transition
from algua.contracts.types import DataProvider
from algua.registry.approvals import compute_artifact_hashes
from algua.registry.repository import StrategyRepository
from algua.registry.transitions import transition_strategy
from algua.research.gates import (
    FUNNEL_WINDOW_DAYS,
    GateCriteria,
    GateDecision,
    effective_funnel_breadth,
    evaluate_gate,
)


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
    from algua.strategies.loader import StrategyNotFound, load_strategy

    try:
        _loaded = load_strategy(name)
    except StrategyNotFound:
        _loaded = None
    # Exhaustive signal_panel parity gate (#178): a panel that diverges from its per-bar signal on
    # ANY bar cannot pass promotion. Runs on the already-loaded strategy, in static mode over the
    # promotion window, BEFORE the holdout is touched. No-op when the strategy has no signal_panel.
    # Raises BacktestError on divergence (caught by the `promote` CLI's @json_errors).
    if _loaded is not None:
        verify_signal_panel_parity(_loaded, provider, start, end)
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
    return BreadthContext(effective_funnel_breadth(own, windowed_total), own, windowed_total,
                          provenance)


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
) -> PromotionOutcome:
    """Post-walk phase: resolve PIT, evaluate, record the gate_evaluations row (pass AND fail), and
    on pass transition BACKTESTED->CANDIDATE (which consumes the just-minted agent token).
    Identity is recomputed via compute_artifact_hashes(name) — the SAME function the shortlist gate
    matches against (NOT wf.code_hash, which is git-HEAD-based and would never match)."""
    pit_ok = resolve_pit_ok(universe_name, universe_snapshots, period_start)
    holdout_n_bars = int(wf.holdout_metrics["n_bars"])
    decision = evaluate_gate(
        wf, criteria, n_combos=breadth.n_funnel, breadth_provenance=breadth.provenance,
        pit_ok=pit_ok, allow_non_pit=allow_non_pit, own_lifetime_combos=breadth.own,
        windowed_total_combos=breadth.windowed_total, funnel_window_days=FUNNEL_WINDOW_DAYS,
    )
    identity = compute_artifact_hashes(name)
    rec = repo.get(name)
    repo.record_gate_evaluation(
        rec.id, passed=decision.passed, n_funnel=breadth.n_funnel,
        own_lifetime_combos=breadth.own, windowed_total_combos=breadth.windowed_total,
        funnel_window_days=FUNNEL_WINDOW_DAYS, breadth_provenance=breadth.provenance,
        pit_ok=bool(decision.pit_ok), pit_override=bool(decision.pit_override),
        holdout_n_bars=holdout_n_bars, min_holdout_observations=criteria.min_holdout_observations,
        code_hash=identity.code_hash, config_hash=identity.config_hash,
        dependency_hash=identity.dependency_hash, data_source=data_source, snapshot_id=snapshot_id,
        period_start=period_start.isoformat(), period_end=period_end.isoformat(),
        holdout_frac=holdout_frac, actor=actor.value,
        decision_json=json.dumps(decision.to_dict(), sort_keys=True),
    )
    promoted = False
    if decision.passed:
        transition_strategy(repo, name, Stage.CANDIDATE, actor,
                            _gate_reason(decision) + reason_suffix)
        promoted = True
    return PromotionOutcome(decision=decision, promoted=promoted)


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
