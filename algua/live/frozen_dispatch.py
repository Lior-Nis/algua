"""The frozen planner port: the supervisor side of the Story 1.3c dispatcher (contract §3, §7, §8).

`FrozenPlanner` answers the three `PlannerPort` calls for one frozen tenant over one tick, and the
current supervisor owns every strategy-free wall. Before any child it validates its own input and
computes the planner's strategy-free verdict with the planner's own code: a breach it finds is
returned with breach semantics whatever a child would say, with no child launched, and an input
the planner refuses is `frozen_planner_rejected`. A pending venue belief is answered the same way
(the pre-belief breach, else `VenueBeliefRequired`); closed bars are computed here too.

Otherwise one fresh child runs the phase from the tenant's bundle (`frozen_invocation`), and its
result must be the verdict: identical to a no-decision or snapshot (binding included), else a
decision (cross-checked, its weights re-validated, in `frozen_attempt`) or a breach only the
strategy's weights can cause (`DECISION_BREACH_KINDS`). Any other disagreement is
`frozen_result_invalid`. Every fault is a `FrozenTenantFailure` carrying a §8 code, the deployment
id and a sanitized, bounded diagnostic (raw stderr is read only into its sanitized head). Each such
child dispatch is an attempt, recorded once judged through the injected `record` (Story 1.3d §1).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final

import pandas as pd

from algua.contracts.types import ExecutionContract
from algua.live.frozen_attempt import (
    Attempt,
    AttemptRecorder,
    Record,
    decision_problem,
    mismatch,
    weight_breach,
)
from algua.live.frozen_invocation import (
    LaunchFailure,
    Runner,
    launch_child,
    process_diagnostic,
    process_failure,
    sanitize_text,
    unsupported_content,
)
from algua.live.frozen_wire import WireError, WireIdentity, WireTooLarge, encode_request
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
from algua.live.planner_early import EarlyVerdict, closed_universe_bars, early_verdict
from algua.live.planner_late import LateProceed, late_verdict, verified_phase_a
from algua.live.planner_validation import validate_early
from algua.primitives.contained_process import run_contained
from algua.risk.limits import DARK_FEED_KINDS, DECISION_BREACH_KINDS, RISK_BREACH_KINDS

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
    """`PlannerPort` for one frozen tenant over ONE tick; build a new instance for every tick.

    ``record`` persists a judged attempt, returning its id; the bars args name the tick's bars.
    """

    def __init__(
        self, target: FrozenTarget, *, invocations_root: Path, record: Record, snapshot_id: str,
        bars_start: str | None, bars_end: str | None, run: Runner = run_contained,
    ) -> None:
        self._target = target
        self._invocations_root = invocations_root
        self._run = run
        self._attempts = AttemptRecorder(
            record, deployment_id=target.identity.deployment_id, snapshot_id=snapshot_id,
            bars_start=bars_start, bars_end=bars_end)
        self._content_checked = False
        self._snapshot: SnapshotRequired | None = None

    @property
    def final_invocation_id(self) -> int | None:
        """The id ``record`` returned for this tick's successful Phase B attempt, else None."""
        return self._attempts.final_id

    # --- the port ------------------------------------------------------------------------------

    def phase_a(self, early: EarlyPlannerInput) -> EarlyPlannerResult:
        verdict = self._early(early)
        if isinstance(verdict, PlannerRiskFailure):
            return verdict
        with self._attempt("a", early.request_id, None) as attempt:
            result = self._invoke(attempt, early, None)
            if _canonical("a", early, result) != _canonical("a", early, verdict):
                raise self._invalid(mismatch(result, type(verdict).__name__))
            self._attempts.succeeded(attempt, result)
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
        with self._attempt("b", late.early.request_id, late.phase_a_binding) as attempt:
            result = self._invoke(attempt, late.early, late)
            answer = self._judged(result, verdict, late.early)
            self._attempts.succeeded(attempt, result)  # a weight-rule breach is still a success
        return answer

    def _judged(
        self, result: _Result, verdict: LateNoDecision | LateProceed, early: EarlyPlannerInput
    ) -> LatePlannerResult:
        """Phase B's answer when the child's result is the supervisor's verdict (§7)."""
        if isinstance(verdict, LateNoDecision):
            if _canonical("b", early, result) != _canonical("b", early, verdict):
                raise self._invalid(mismatch(result, "LateNoDecision"))
            return verdict
        if isinstance(result, Decision):
            target = self._target
            if problem := decision_problem(result, verdict, target.gate_universe):
                raise self._invalid(problem)
            breach = weight_breach(
                result, target.execution, target.identity.strategy_name, target.gate_universe)
            return result if breach is None else self._risk(*breach)
        if isinstance(result, PlannerRiskFailure) and result.kind in DECISION_BREACH_KINDS:
            return result
        raise self._invalid(mismatch(result, "a decision"))

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

    # --- one attempt: one child ----------------------------------------------------------------

    @contextmanager
    def _attempt(self, phase: Phase, request_id: str, binding: str | None) -> Iterator[Attempt]:
        """Record a tenant failure raised inside, before it propagates; nothing if systemic."""
        attempt = self._attempts.begin(phase, request_id, binding)
        try:
            yield attempt
        except FrozenTenantFailure as failure:
            self._attempts.failed(attempt, failure.code, failure.diagnostic)
            raise

    def _invoke(
        self, attempt: Attempt, early: EarlyPlannerInput, late: LatePlannerInput | None
    ) -> _Result:
        target, phase = self._target, attempt.phase
        if not self._content_checked:
            problem = unsupported_content(target.bundle_root)
            if problem is not None:
                raise self._failure("frozen_content_unsupported", problem)
            self._content_checked = True
        try:
            request_json, bars_arrow = attempt.request = encode_request(
                phase, early, late, target.identity)
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
        attempt.ended = ended
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

    # --- helpers ------------------------------------------------------------------------------

    def _require_snapshot(self) -> SnapshotRequired:
        if self._snapshot is None:
            raise RuntimeError("frozen planner port called before a Phase A snapshot")
        return self._snapshot

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
