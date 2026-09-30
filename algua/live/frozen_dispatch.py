"""The frozen planner port: the supervisor side of the Story 1.3c dispatcher (contract §3, §7, §8).

`FrozenPlanner` answers the three `PlannerPort` calls for one frozen tenant over one tick, and the
current supervisor owns every strategy-free wall. Before any child it validates its own input and
computes the planner's strategy-free verdict with the planner's own code: a breach it finds is
returned with breach semantics whatever a child would say, with no child launched, and an input
the planner refuses is `frozen_planner_rejected`. A pending venue belief is answered the same way
(the pre-belief breach, else `VenueBeliefRequired`); closed bars are computed here too.

Otherwise one fresh child runs the phase from the tenant's bundle (`frozen_invocation`), and its
result must be the verdict: identical to a no-decision or snapshot (binding included), else a
decision or a breach only the strategy's weights can cause (`DECISION_BREACH_KINDS`). A decision's
intents must be `build_intents` of its weights from the supervisor's current weights at Phase A's
decision time, unique, inside the gate universe or holdings; its weights are re-validated with the
tenant's contract, a violation being a breach as in process. Any other disagreement is
`frozen_result_invalid`. Every fault is a `FrozenTenantFailure` carrying a §8 code, the deployment
id and a sanitized, bounded diagnostic (raw stderr is read only into its sanitized head).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final

import pandas as pd

from algua.contracts.types import ExecutionContract
from algua.live.frozen_invocation import (
    LaunchFailure,
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
    PhaseBindingFailure,
    PlannerInputFailure,
    PlannerRiskFailure,
    SnapshotRequired,
    VenueBeliefRequired,
)
from algua.live.planner_decision import build_intents
from algua.live.planner_early import EarlyVerdict, closed_universe_bars, early_verdict
from algua.live.planner_late import LateProceed, late_verdict, verified_phase_a
from algua.live.planner_validation import validate_early
from algua.primitives.contained_process import run_contained
from algua.risk.limits import (
    DARK_FEED_KINDS,
    DECISION_BREACH_KINDS,
    RISK_BREACH_KINDS,
    RiskBreach,
    validate_decision_weights,
)

#: The §8 tenant failure codes; each is stable and deployment-bound.
FROZEN_FAILURE_CODES: Final = frozenset({
    "frozen_content_unavailable", "frozen_content_unsupported", "frozen_request_too_large",
    "frozen_launch_failed", "frozen_timeout", "frozen_exit_abnormal", "frozen_output_exceeded",
    "frozen_result_invalid", "frozen_planner_rejected", "frozen_live_unsupported",
})

type _Result = EarlyNoDecision | SnapshotRequired | PlannerRiskFailure | LateNoDecision | Decision
type _LateVerdict = PlannerRiskFailure | VenueBeliefRequired | LateNoDecision | LateProceed


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
        verdict = self._early(early)
        if isinstance(verdict, PlannerRiskFailure):
            return verdict
        result = self._invoke("a", early, None)
        if _canonical("a", early, result) != _canonical("a", early, verdict):
            raise self._mismatch(result, type(verdict).__name__)
        if isinstance(verdict, SnapshotRequired):
            self._snapshot = verdict
        return verdict

    def closed_bars(self, early: EarlyPlannerInput) -> pd.DataFrame:
        snapshot = self._require_snapshot()
        try:
            closed = closed_universe_bars(early.raw_bars, early.now, early.gate_universe)
        except (TypeError, ValueError) as exc:
            raise self._invalid(f"the supervisor cannot select closed bars: {exc}") from exc
        if closed.decision_ts != snapshot.decision_ts:
            raise self._invalid("the closed bars' decision time is not Phase A's decision_ts")
        return closed.bars

    def phase_b(self, late: LatePlannerInput) -> LatePlannerResult:
        self._require_snapshot()
        verdict = self._late(late)
        if isinstance(verdict, (PlannerRiskFailure, VenueBeliefRequired)):
            return verdict  # run_tick resolves a requested belief and calls again
        result = self._invoke("b", late.early, late)
        if isinstance(verdict, LateNoDecision):
            if _canonical("b", late.early, result) != _canonical("b", late.early, verdict):
                raise self._mismatch(result, "LateNoDecision")
            return verdict
        if isinstance(result, Decision):
            return self._checked_decision(result, verdict)
        if isinstance(result, PlannerRiskFailure) and result.kind in DECISION_BREACH_KINDS:
            return result
        raise self._mismatch(result, "a decision")

    # --- the supervisor's own strategy-free verdict ---------------------------------------------

    def _early(self, early: EarlyPlannerInput) -> EarlyVerdict:
        if early.gate_universe != self._target.gate_universe:
            raise self._failure(
                "frozen_planner_rejected", "the early input's gate universe is not the tenant's"
            )
        failure = validate_early(early, None)
        if failure is not None:
            raise self._rejected(failure)
        verdict = early_verdict(early, self._target.execution)
        if isinstance(verdict, PlannerRiskFailure):
            return self._risk(verdict.kind, verdict.detail)
        return verdict

    def _late(self, late: LatePlannerInput) -> _LateVerdict:
        """Phase B's verdict over the captured values exactly as a child would receive them."""
        phase_a = verified_phase_a(late, self._early(late.early))
        if not isinstance(phase_a, SnapshotRequired):
            raise self._rejected(phase_a)
        captured = late.captured
        try:
            sent = replace(
                captured,
                quantities=decode_pairs(encode_pairs(captured.quantities, "q"), "q"),
                market_values=decode_pairs(encode_pairs(captured.market_values, "v"), "v"),
            )
        except WireError as exc:
            raise self._failure("frozen_planner_rejected", f"unencodable input: {exc}") from exc
        verdict = late_verdict(late.early, sent, phase_a, self._target.execution)
        if isinstance(verdict, PlannerInputFailure):
            raise self._rejected(verdict)
        if isinstance(verdict, PlannerRiskFailure):
            return self._risk(verdict.kind, verdict.detail)
        return verdict

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
        try:
            request_json, bars_arrow = encode_request(phase, early, late, target.identity)
        except WireTooLarge as exc:
            raise self._failure("frozen_request_too_large", str(exc)) from exc
        except WireError as exc:  # the supervisor's own input is not a valid request
            raise self._failure("frozen_planner_rejected", f"unencodable input: {exc}") from exc
        try:  # an OSError setting up the invocation directory is the supervisor's: systemic
            ended = launch_child(
                interpreter=target.interpreter,
                bundle_root=target.bundle_root,
                environment_root=target.environment_root,
                request_json=request_json,
                bars_arrow=bars_arrow,
                invocations_root=self._invocations_root,
                run=self._run,
            )
        except LaunchFailure as exc:
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

    # --- a decision against the verdict ---------------------------------------------------------

    def _checked_decision(self, result: Decision, verdict: LateProceed) -> LatePlannerResult:
        ts, current = verdict.decision_ts, verdict.current_weights
        intents = list(result.ordered_intents)
        if result.state.decision_ts != ts or any(i.decision_ts != ts for i in intents):
            raise self._invalid("a decision timestamp is not Phase A's decision_ts")
        symbols = [intent.symbol for intent in intents]
        if len(set(symbols)) != len(symbols):
            raise self._invalid("the intents repeat a symbol")
        if outside := sorted(set(symbols) - set(self._target.gate_universe) - set(current)):
            raise self._invalid(f"intents outside the gate universe and holdings: {outside}")
        if result.state != replace(verdict.state, target_weights=result.state.target_weights):
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

    def _mismatch(self, result: _Result, expected: str) -> FrozenTenantFailure:
        if isinstance(result, PlannerRiskFailure):
            return self._invalid(f"the child reports a {result.kind} breach the supervisor does "
                                 "not find")
        if type(result).__name__ != expected:
            return self._invalid(f"the child answered {type(result).__name__} where the "
                                 f"supervisor's verdict is {expected}")
        return self._invalid(f"the child's {expected} is not the one these inputs give")

    def _rejected(self, refusal: PlannerInputFailure | PhaseBindingFailure) -> FrozenTenantFailure:
        return self._failure("frozen_planner_rejected", f"{refusal.code}: {refusal.detail}")

    def _risk(self, kind: str, detail: str) -> PlannerRiskFailure:
        if kind not in RISK_BREACH_KINDS:
            raise self._invalid("the risk kind is not a known breach kind")
        return PlannerRiskFailure(kind, sanitize_text(detail), kind in DARK_FEED_KINDS)

    def _failure(self, code: str, diagnostic: str) -> FrozenTenantFailure:
        return FrozenTenantFailure(code, self._target.identity.deployment_id, diagnostic)

    def _invalid(self, diagnostic: str) -> FrozenTenantFailure:
        return self._failure("frozen_result_invalid", diagnostic)


def _canonical(phase: Phase, early: EarlyPlannerInput, result: object) -> bytes:
    return encode_result(phase, early.request_id, result)
