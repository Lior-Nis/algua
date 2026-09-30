"""The frozen planner port: the supervisor side of the Story 1.3c dispatcher (contract §3, §7, §8).

`FrozenPlanner` answers the three `PlannerPort` calls for one frozen tenant over one tick by
running each planner phase in a fresh child from the tenant's bundle and environment
(`frozen_invocation`). Phase A launches one child. Closed bars are computed here with the
planner's own strategy-free helper. Phase B answers a *pending* venue belief without a child —
the resolved-belief call that `run_tick` makes next repeats every check a pending call would
run — and launches one child for the resolved belief.

A child's result is trusted only as far as the supervisor cannot recompute it without strategy
code: the target weights and the Phase A binding. Everything else the supervisor acts on is
re-derived from its own inputs with the planner's own helpers and must match — Phase A's decision
time, a no-decision state, Phase B's state, and the intents (`build_intents` of the child's
weights, stamped with Phase A's decision time, unique, inside the gate universe or current
holdings); the decision's weights are then re-validated with the tenant's execution contract, and
a rule they break is a risk breach exactly as in process. Any other disagreement is
`frozen_result_invalid`.

Every fault is a `FrozenTenantFailure` carrying a §8 code, the deployment id and a sanitized,
bounded diagnostic; a child's risk failure keeps `RiskBreach` semantics with its detail
sanitized the same way. Raw stderr is read only into that diagnostic, as its sanitized head.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Final

import pandas as pd

from algua.contracts.types import ExecutionContract
from algua.live.frozen_invocation import (
    Runner,
    launch_child,
    process_diagnostic,
    process_failure,
    sanitize_text,
    unsupported_content,
)
from algua.live.frozen_wire import (
    WireError,
    WireIdentity,
    WireTooLarge,
    encode_request,
)
from algua.live.frozen_wire_json import Phase, decode_pairs, encode_pairs
from algua.live.frozen_wire_result import PlannerRejected, decode_result, encode_result
from algua.live.planner_contract import (
    Decision,
    EarlyNoDecision,
    EarlyPlannerInput,
    EarlyPlannerResult,
    LateNoDecision,
    LatePlannerInput,
    LatePlannerResult,
    PlannerRiskFailure,
    PlannerState,
    SnapshotRequired,
    VenueBeliefPending,
    VenueBeliefRequired,
)
from algua.live.planner_decision import build_intents
from algua.live.planner_early import ClosedBars, _empty_state, closed_universe_bars
from algua.live.planner_late import _late_state
from algua.primitives.contained_process import run_contained
from algua.risk.limits import (
    DARK_FEED_KINDS,
    RISK_BREACH_KINDS,
    RiskBreach,
    validate_decision_weights,
)

#: The §8 tenant failure codes; each is stable and deployment-bound.
FROZEN_FAILURE_CODES: Final = frozenset(
    {
        "frozen_content_unavailable",
        "frozen_content_unsupported",
        "frozen_request_too_large",
        "frozen_launch_failed",
        "frozen_timeout",
        "frozen_exit_abnormal",
        "frozen_output_exceeded",
        "frozen_result_invalid",
        "frozen_planner_rejected",
        "frozen_live_unsupported",
    }
)

type _Result = EarlyNoDecision | SnapshotRequired | PlannerRiskFailure | LateNoDecision | Decision


class FrozenTenantFailure(Exception):
    """A frozen tenant's fault, isolated to that tenant before any effect (§8)."""

    def __init__(self, code: str, deployment_id: int, diagnostic: str) -> None:
        if code not in FROZEN_FAILURE_CODES:
            raise ValueError(f"unknown frozen failure code {code!r}")
        self.code = code
        self.deployment_id = deployment_id
        self.diagnostic = sanitize_text(diagnostic)
        super().__init__(f"{code} (deployment {deployment_id}): {self.diagnostic}")


@dataclass(frozen=True)
class FrozenTarget:
    """One frozen deployment as plain values: its wire identity, content and supervisor view."""

    identity: WireIdentity
    bundle_root: Path
    environment_root: Path
    interpreter: Path
    execution: ExecutionContract
    gate_universe: tuple[str, ...]


class FrozenPlanner:
    """`PlannerPort` for one frozen tenant over ONE tick; build a new instance for every tick."""

    def __init__(
        self, target: FrozenTarget, *, invocations_root: Path, run: Runner = run_contained
    ) -> None:
        self._target = target
        self._invocations_root = invocations_root
        self._run = run
        self._content_checked = False
        self._snapshot: SnapshotRequired | None = None

    # --- the port ------------------------------------------------------------------------------

    def phase_a(self, early: EarlyPlannerInput) -> EarlyPlannerResult:
        result = self._invoke("a", early, None)
        if isinstance(result, PlannerRiskFailure):
            return result
        if isinstance(result, SnapshotRequired):
            if result.decision_ts != self._closed(early).decision_ts:
                raise self._invalid("the snapshot is not at the closed-bar decision time")
            self._snapshot = result
            return result
        if isinstance(result, EarlyNoDecision):
            closed = self._closed(early)  # the planner answers no_bars before it answers warming
            expected = EarlyNoDecision(
                "no_bars" if closed.bars.empty else "warming",
                _empty_state(early, closed.decision_ts),
            )
            if encode_result("a", early.request_id, result) != encode_result(
                "a", early.request_id, expected
            ):
                raise self._invalid("the early no-decision is not the one these inputs give")
            return result
        raise self._invalid(f"{type(result).__name__} is not a Phase A result")

    def closed_bars(self, early: EarlyPlannerInput) -> pd.DataFrame:
        snapshot = self._require_snapshot()
        closed = self._closed(early)
        if closed.decision_ts != snapshot.decision_ts:
            raise self._invalid("the closed bars' decision time is not Phase A's decision_ts")
        return closed.bars

    def phase_b(self, late: LatePlannerInput) -> LatePlannerResult:
        if isinstance(late.captured.venue_belief, VenueBeliefPending):
            return VenueBeliefRequired()  # run_tick resolves the belief and calls again
        snapshot = self._require_snapshot()
        result = self._invoke("b", late.early, late)
        if isinstance(result, PlannerRiskFailure):
            return result
        if isinstance(result, LateNoDecision):
            state, _ = self._derived(late, snapshot.decision_ts)
            if not snapshot.warming or result.state != state:
                raise self._invalid("the late no-decision is not the one these inputs give")
            return result
        if isinstance(result, Decision):
            return self._checked_decision(result, late, snapshot)
        raise self._invalid(f"{type(result).__name__} is not a Phase B result")

    # --- one child ------------------------------------------------------------------------------

    def _invoke(
        self, phase: Phase, early: EarlyPlannerInput, late: LatePlannerInput | None
    ) -> _Result:
        target = self._target
        if not self._content_checked:
            problem = unsupported_content(target.bundle_root)
            if problem is not None:
                raise self._failure("frozen_content_unsupported", problem)
            self._content_checked = True
        if early.gate_universe != target.gate_universe:
            raise self._failure(
                "frozen_planner_rejected", "the early input's gate universe is not the tenant's"
            )
        try:
            request_json, bars_arrow = encode_request(phase, early, late, target.identity)
        except WireTooLarge as exc:
            raise self._failure("frozen_request_too_large", str(exc)) from exc
        except WireError as exc:  # the supervisor's own input is not a valid request
            raise self._failure("frozen_planner_rejected", f"unencodable input: {exc}") from exc
        try:
            ended = launch_child(
                interpreter=target.interpreter,
                bundle_root=target.bundle_root,
                environment_root=target.environment_root,
                request_json=request_json,
                bars_arrow=bars_arrow,
                invocations_root=self._invocations_root,
                run=self._run,
            )
        except OSError as exc:
            raise self._failure("frozen_launch_failed", str(exc)) from exc
        code = process_failure(ended)
        if code is not None:
            raise self._failure(code, process_diagnostic(ended))
        try:
            result = decode_result(ended.stdout, phase=phase, request_id=early.request_id)
        except WireError as exc:
            raise self._invalid(str(exc)) from exc
        if isinstance(result, PlannerRejected):
            raise self._failure("frozen_planner_rejected", f"{result.code}: {result.detail}")
        if isinstance(result, PlannerRiskFailure):
            return self._risk(result.kind, result.detail)
        return result

    # --- supervisor-side re-derivation ----------------------------------------------------------

    def _closed(self, early: EarlyPlannerInput) -> ClosedBars:
        try:
            return closed_universe_bars(early.raw_bars, early.now, early.gate_universe)
        except (TypeError, ValueError) as exc:
            raise self._invalid(f"the supervisor cannot select closed bars: {exc}") from exc

    def _derived(
        self, late: LatePlannerInput, decision_ts: datetime | None
    ) -> tuple[PlannerState, dict[str, float]]:
        """Phase B's state and current weights from the captured values exactly as sent."""
        captured = late.captured
        try:
            sent = replace(
                captured,
                quantities=decode_pairs(encode_pairs(captured.quantities, "q"), "q"),
                market_values=decode_pairs(encode_pairs(captured.market_values, "v"), "v"),
            )
            return _late_state(sent, decision_ts)
        except (TypeError, ValueError) as exc:  # includes RiskBreach and WireError
            raise self._invalid(f"the captured values admit no late result: {exc}") from exc

    def _checked_decision(
        self, result: Decision, late: LatePlannerInput, snapshot: SnapshotRequired
    ) -> LatePlannerResult:
        ts = snapshot.decision_ts
        if snapshot.warming or ts is None:
            raise self._invalid("a decision follows a warming or undated Phase A")
        intents = list(result.ordered_intents)
        if result.state.decision_ts != ts or any(i.decision_ts != ts for i in intents):
            raise self._invalid("a decision timestamp is not Phase A's decision_ts")
        symbols = [intent.symbol for intent in intents]
        if len(set(symbols)) != len(symbols):
            raise self._invalid("the intents repeat a symbol")
        state, current = self._derived(late, ts)
        if outside := sorted(set(symbols) - set(self._target.gate_universe) - set(current)):
            raise self._invalid(f"intents outside the gate universe and holdings: {outside}")
        if result.state != replace(state, target_weights=result.state.target_weights):
            raise self._invalid("the decision state is not the one the captured values give")
        weights = pd.Series(dict(result.state.target_weights), dtype="float64")
        if intents != build_intents(weights, current, ts):
            raise self._invalid("the intents are not build_intents of the target weights")
        target = self._target
        try:
            validate_decision_weights(
                weights, target.execution, target.identity.strategy_name,
                allowed_symbols=target.gate_universe,
            )
        except RiskBreach as exc:
            return self._risk(exc.kind, exc.detail)
        return result

    # --- helpers ------------------------------------------------------------------------------

    def _require_snapshot(self) -> SnapshotRequired:
        if self._snapshot is None:
            raise RuntimeError("frozen planner port called before a Phase A snapshot")
        return self._snapshot

    def _risk(self, kind: str, detail: str) -> PlannerRiskFailure:
        if kind not in RISK_BREACH_KINDS:
            raise self._invalid("the risk kind is not a known breach kind")
        return PlannerRiskFailure(kind, sanitize_text(detail), kind in DARK_FEED_KINDS)

    def _failure(self, code: str, diagnostic: str) -> FrozenTenantFailure:
        return FrozenTenantFailure(code, self._target.identity.deployment_id, diagnostic)

    def _invalid(self, diagnostic: str) -> FrozenTenantFailure:
        return self._failure("frozen_result_invalid", diagnostic)
