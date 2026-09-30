"""One frozen planner attempt: its decision cross-check and its evidence record (Story 1.3d §1, §3).

An *attempt* is one phase dispatch the supervisor decided to run a child for: every entry into the
frozen port's invocation step, including one refused before launch (request too large, content
unsupported, unencodable input), which has no request bytes. :class:`AttemptRecorder` writes each
as exactly one :class:`~algua.contracts.frozen_evidence.FrozenAttempt` through the injected
``record`` once the supervisor has fully judged it. A *success* decoded and passed every Story 1.3c
§7 cross-check for its phase; it carries its result kind and the SHA-256 of the child's accepted
stdout bytes, never of a re-encoding. A *failure* carries the ``FrozenTenantFailure`` code and
bounded, sanitized diagnostic the supervisor raised. Request and bars digests are of the exact
bytes written for the child. The dispatcher opens no attempt for a phase it settles itself and
never catches a systemic exception, so neither leaves a row; ``record`` raising is itself systemic.

Judging a child's result against the supervisor's verdict (§7) lives here too. :func:`mismatch`
says why a result is not the verdict. :func:`decision_problem` cross-checks a decision against the
late verdict: every timestamp is Phase A's ``decision_ts``, the intents' symbols are unique and
inside the gate universe or holdings, the state is the one the captured values give, and the
intents are ``build_intents`` of the target weights from the supervisor's current weights. Then
:func:`weight_breach` re-validates those weights with the tenant's contract: a violation is a real
breach, which the dispatcher returns with breach semantics as the in-process planner would.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime

import pandas as pd

from algua.contracts.frozen_evidence import FrozenAttempt
from algua.contracts.types import ExecutionContract
from algua.live.frozen_wire_json import Phase
from algua.live.planner_contract import (
    Decision,
    EarlyNoDecision,
    LateNoDecision,
    PlannerRiskFailure,
    SnapshotRequired,
)
from algua.live.planner_decision import build_intents
from algua.live.planner_late import LateProceed
from algua.primitives.contained_process import ContainedResult
from algua.risk.limits import RiskBreach, validate_decision_weights

type Record = Callable[[FrozenAttempt], int]
type Accepted = EarlyNoDecision | SnapshotRequired | PlannerRiskFailure | LateNoDecision | Decision

_RESULT_KINDS: dict[type, str] = {
    EarlyNoDecision: "early_no_decision",
    SnapshotRequired: "snapshot_required",
    PlannerRiskFailure: "risk_failure",
    LateNoDecision: "late_no_decision",
    Decision: "decision",
}


def mismatch(result: object, expected: str) -> str:
    """Why a child's decoded result is not the supervisor's verdict, which ``expected`` names."""
    if isinstance(result, PlannerRiskFailure):
        return f"the child reports a {result.kind} breach the supervisor does not find"
    if type(result).__name__ != expected:
        return (f"the child answered {type(result).__name__} where the supervisor's verdict is "
                f"{expected}")
    return f"the child's {expected} is not the one these inputs give"


def _weights(result: Decision) -> pd.Series:
    """The decision's target weights as float64, the form both sides judge them in (§7)."""
    return pd.Series(dict(result.state.target_weights), dtype="float64")


def decision_problem(
    result: Decision, verdict: LateProceed, gate_universe: tuple[str, ...]
) -> str | None:
    """Why a child's decision is not one the supervisor's late verdict admits, else ``None``."""
    ts, current = verdict.decision_ts, verdict.current_weights
    intents = list(result.ordered_intents)
    if result.state.decision_ts != ts or any(i.decision_ts != ts for i in intents):
        return "a decision timestamp is not Phase A's decision_ts"
    symbols = [intent.symbol for intent in intents]
    if len(set(symbols)) != len(symbols):
        return "the intents repeat a symbol"
    if outside := sorted(set(symbols) - set(gate_universe) - set(current)):
        return f"intents outside the gate universe and holdings: {outside}"
    if result.state != replace(verdict.state, target_weights=result.state.target_weights):
        return "the decision state is not the one the captured values give"
    if intents != build_intents(_weights(result), current, ts):
        return "the intents are not build_intents of the target weights"
    return None


def weight_breach(
    result: Decision, execution: ExecutionContract, strategy_name: str,
    gate_universe: tuple[str, ...],
) -> tuple[str, str] | None:
    """The ``(kind, detail)`` of a weight rule the decision breaks, else ``None``."""
    try:
        validate_decision_weights(_weights(result), execution, strategy_name,
                                  allowed_symbols=gate_universe)
    except RiskBreach as exc:
        return exc.kind, exc.detail
    return None


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


@dataclass
class Attempt:
    """One phase dispatch under judgement: what its child was sent and how the child ended."""

    phase: Phase
    request_id: str
    phase_a_invocation_id: int | None
    phase_a_binding: str | None  # the binding a Phase B attempt received
    started_at: str
    request: tuple[bytes, bytes] | None = None  # request.json and bars.arrow, exactly as written
    ended: ContainedResult | None = None


class AttemptRecorder:
    """Records one tick's judged attempts and keeps the ids ``record`` returned for them."""

    def __init__(
        self, record: Record, *, deployment_id: int, snapshot_id: str, bars_start: str | None,
        bars_end: str | None,
    ) -> None:
        self._record = record
        self._deployment_id = deployment_id
        self._bars = (snapshot_id, bars_start, bars_end)
        self.phase_a_id: int | None = None  # the tick's successful Phase A attempt
        self.final_id: int | None = None  # the tick's successful Phase B attempt

    def begin(self, phase: Phase, request_id: str, phase_a_binding: str | None) -> Attempt:
        """Open the attempt at its entry into the invocation step."""
        if phase == "b" and self.phase_a_id is None:
            raise RuntimeError("a Phase B attempt must follow a recorded Phase A success")
        linked = self.phase_a_id if phase == "b" else None
        return Attempt(phase, request_id, linked, phase_a_binding, _utc_now())

    def succeeded(self, attempt: Attempt, result: Accepted) -> int:
        """Record an attempt whose result decoded and passed every §7 check of its phase."""
        if attempt.ended is None:
            raise RuntimeError("a successful attempt has its child's accepted stdout")
        binding = (result.phase_a_binding if isinstance(result, SnapshotRequired)
                   else attempt.phase_a_binding)  # Phase A: produced; Phase B: received
        row_id = self._write(attempt, binding, result_kind=_RESULT_KINDS[type(result)],
                             result_sha256=_sha256(attempt.ended.stdout))
        if attempt.phase == "a":
            self.phase_a_id = row_id
        else:
            self.final_id = row_id
        return row_id

    def failed(self, attempt: Attempt, code: str, diagnostic: str) -> int:
        """Record an attempt the supervisor failed with ``code`` (a Story 1.3c §8 code)."""
        return self._write(attempt, attempt.phase_a_binding, failure_code=code,
                           diagnostic=diagnostic)

    def _write(
        self, attempt: Attempt, phase_a_binding: str | None, *, result_kind: str | None = None,
        result_sha256: str | None = None, failure_code: str | None = None,
        diagnostic: str | None = None,
    ) -> int:
        request_json, bars_arrow = attempt.request or (None, None)
        ended = attempt.ended
        snapshot_id, bars_start, bars_end = self._bars
        return self._record(FrozenAttempt(
            deployment_id=self._deployment_id,
            request_id=attempt.request_id,
            phase=attempt.phase,
            phase_a_invocation_id=attempt.phase_a_invocation_id,
            snapshot_id=snapshot_id,
            bars_start=bars_start,
            bars_end=bars_end,
            request_json=None if request_json is None else request_json.decode("utf-8"),
            request_sha256=None if request_json is None else _sha256(request_json),
            bars_sha256=None if bars_arrow is None else _sha256(bars_arrow),
            phase_a_binding=phase_a_binding,
            result_kind=result_kind,
            result_sha256=result_sha256,
            failure_code=failure_code,
            returncode=None if ended is None else ended.returncode,
            signal=None if ended is None else ended.signal,
            timed_out=ended is not None and ended.timed_out,
            stdout_exceeded=ended is not None and ended.stdout_exceeded,
            stderr_truncated=ended is not None and ended.stderr_truncated,
            diagnostic=diagnostic,
            started_at=attempt.started_at,
            ended_at=_utc_now(),
        ))
