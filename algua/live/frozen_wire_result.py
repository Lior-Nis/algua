"""The frozen planner's result on stdout: one canonical JSON document per phase.

Story 1.3c contract §7. The child encodes its planner outcome with :func:`encode_result`; the
supervisor reads it with :func:`decode_result`, which is strict the same way the request decoder
is (canonical bytes with at most one trailing newline, echoed wire/phase/request_id, a kind the
phase permits, exact fields and JSON types, §6 limits). Planner input and binding refusals cross as
``planner_rejected`` and decode to :class:`PlannerRejected`. A risk failure's dark-feed flag is
not on the wire: the supervisor derives it from the kind. Semantic allowlists (risk kinds, planner
codes), detail sanitisation and cross-checks against Phase A belong to the dispatcher.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

from algua.contracts.canonical import FROZEN_WIRE
from algua.contracts.types import OrderIntent, Side
from algua.live.frozen_wire import MAX_STDOUT_BYTES
from algua.live.frozen_wire_json import (
    Phase,
    WireError,
    canonical_bytes,
    decode_float,
    decode_pairs,
    decode_ts,
    encode_float,
    encode_pairs,
    encode_ts,
    expect,
    expect_hex,
    expect_object,
    expect_phase,
    expect_wire,
    optional,
    parse_canonical,
)
from algua.live.planner_contract import (
    Decision,
    EarlyNoDecision,
    LateNoDecision,
    PhaseBindingFailure,
    PlannerInputFailure,
    PlannerRiskFailure,
    PlannerState,
    SnapshotRequired,
)
from algua.risk.limits import DARK_FEED_KINDS


@dataclass(frozen=True)
class PlannerRejected:
    """The child's planner refused its input or Phase A binding (`frozen_planner_rejected`)."""

    code: str
    detail: str


type WireResult = (
    EarlyNoDecision | SnapshotRequired | PlannerRiskFailure | PlannerRejected | LateNoDecision
    | Decision
)

_KIND_FIELDS: dict[str, frozenset[str]] = {
    "early_no_decision": frozenset({"kind", "reason", "state"}),
    "snapshot_required": frozenset({"kind", "decision_ts", "warming", "phase_a_binding"}),
    "risk_failure": frozenset({"kind", "risk_kind", "detail"}),
    "planner_rejected": frozenset({"kind", "code", "detail"}),
    "late_no_decision": frozenset({"kind", "reason", "state"}),
    "decision": frozenset({"kind", "state", "intents"}),
}
_PHASE_KINDS: dict[str, frozenset[str]] = {
    "a": frozenset({"early_no_decision", "snapshot_required", "risk_failure", "planner_rejected"}),
    "b": frozenset({"risk_failure", "planner_rejected", "late_no_decision", "decision"}),
}
_STATE_KEYS = frozenset(
    {"decision_ts", "target_weights", "positions_before", "equity", "peak_equity",
     "reconcile_ok", "realized_gross"}
)
_INTENT_KEYS = frozenset({"symbol", "side", "target_weight", "decision_ts"})
_ENVELOPE_KEYS = frozenset({"wire", "phase", "request_id", "result"})


def _timestamp(value: Any, where: str) -> pd.Timestamp:
    return pd.Timestamp(decode_ts(value, where))


def _encode_state(state: PlannerState) -> dict[str, Any]:
    if not isinstance(state, PlannerState):
        raise WireError("not_encodable", "state must be a PlannerState")
    return {
        "decision_ts": optional(state.decision_ts, encode_ts, "state.decision_ts"),
        "target_weights": encode_pairs(state.target_weights, "state.target_weights"),
        "positions_before": encode_pairs(state.positions_before, "state.positions_before"),
        "equity": encode_float(state.equity, "state.equity"),
        "peak_equity": optional(state.peak_equity, encode_float, "state.peak_equity"),
        "reconcile_ok": expect(state.reconcile_ok, bool, "state.reconcile_ok"),
        "realized_gross": encode_float(state.realized_gross, "state.realized_gross"),
    }


def _decode_state(value: Any) -> PlannerState:
    s = expect_object(value, _STATE_KEYS, "state")
    return PlannerState(
        decision_ts=optional(s["decision_ts"], _timestamp, "state.decision_ts"),
        target_weights=tuple(decode_pairs(s["target_weights"], "state.target_weights").items()),
        positions_before=tuple(
            decode_pairs(s["positions_before"], "state.positions_before").items()
        ),
        equity=decode_float(s["equity"], "state.equity"),
        peak_equity=optional(s["peak_equity"], decode_float, "state.peak_equity"),
        reconcile_ok=expect(s["reconcile_ok"], bool, "state.reconcile_ok"),
        realized_gross=decode_float(s["realized_gross"], "state.realized_gross"),
    )


def _encode_intent(intent: OrderIntent) -> dict[str, Any]:
    if not isinstance(intent, OrderIntent):
        raise WireError("not_encodable", "intents must be OrderIntent values")
    if intent.side not in (Side.BUY, Side.SELL):
        raise WireError("bad_value", "intent side must be buy or sell")
    return {
        "symbol": expect(intent.symbol, str, "intent.symbol"),
        "side": Side(intent.side).value,
        "target_weight": encode_float(intent.target_weight, "intent.target_weight"),
        "decision_ts": encode_ts(intent.decision_ts, "intent.decision_ts"),
    }


def _decode_intent(value: Any) -> OrderIntent:
    i = expect_object(value, _INTENT_KEYS, "intent")
    if type(i["side"]) is not str or i["side"] not in ("buy", "sell"):
        raise WireError("bad_value", "intent side must be 'buy' or 'sell'")
    return OrderIntent(
        symbol=expect(i["symbol"], str, "intent.symbol"),
        side=Side(i["side"]),
        target_weight=decode_float(i["target_weight"], "intent.target_weight"),
        decision_ts=_timestamp(i["decision_ts"], "intent.decision_ts"),
    )


def _reason(value: Any, allowed: tuple[str, ...]) -> str:
    if type(value) is not str or value not in allowed:
        raise WireError("bad_value", f"reason must be one of {list(allowed)}")
    return value


def _encode_body(result: Any) -> dict[str, Any]:
    if isinstance(result, EarlyNoDecision):
        reason = _reason(result.reason, ("no_bars", "warming"))
        return {"kind": "early_no_decision", "reason": reason, "state": _encode_state(result.state)}
    if isinstance(result, SnapshotRequired):
        return {
            "kind": "snapshot_required",
            "decision_ts": optional(result.decision_ts, encode_ts, "decision_ts"),
            "warming": expect(result.warming, bool, "warming"),
            "phase_a_binding": expect_hex(result.phase_a_binding, 64, "phase_a_binding"),
        }
    if isinstance(result, PlannerRiskFailure):
        return {"kind": "risk_failure", "risk_kind": expect(result.kind, str, "risk_kind"),
                "detail": expect(result.detail, str, "detail")}
    if isinstance(result, (PlannerInputFailure, PhaseBindingFailure, PlannerRejected)):
        return {"kind": "planner_rejected", "code": expect(result.code, str, "code"),
                "detail": expect(result.detail, str, "detail")}
    if isinstance(result, LateNoDecision):
        reason = _reason(result.reason, ("warming",))
        return {"kind": "late_no_decision", "reason": reason, "state": _encode_state(result.state)}
    if isinstance(result, Decision):
        return {"kind": "decision", "state": _encode_state(result.state),
                "intents": [_encode_intent(intent) for intent in result.ordered_intents]}
    raise WireError("not_encodable", f"{type(result).__name__} never crosses the wire")


def encode_result(phase: Phase, request_id: str, result: Any) -> bytes:
    """The child's stdout: the canonical result document and one trailing newline."""
    expect_phase(phase)
    body = _encode_body(result)
    if body["kind"] not in _PHASE_KINDS[phase]:
        raise WireError("forbidden_kind", f"{body['kind']} is not a phase {phase} result")
    envelope = {
        "wire": dict(FROZEN_WIRE),
        "phase": phase,
        "request_id": expect_hex(request_id, 32, "request_id"),
        "result": body,
    }
    return canonical_bytes(envelope) + b"\n"


def _decode_body(kind: str, r: dict[str, Any]) -> WireResult:
    if kind == "early_no_decision":
        reason: Literal["no_bars", "warming"] = (
            "no_bars" if _reason(r["reason"], ("no_bars", "warming")) == "no_bars" else "warming"
        )
        return EarlyNoDecision(reason, _decode_state(r["state"]))
    if kind == "snapshot_required":
        return SnapshotRequired(
            optional(r["decision_ts"], _timestamp, "decision_ts"),
            expect(r["warming"], bool, "warming"),
            expect_hex(r["phase_a_binding"], 64, "phase_a_binding"),
        )
    if kind == "risk_failure":
        risk_kind: str = expect(r["risk_kind"], str, "risk_kind")
        return PlannerRiskFailure(
            risk_kind, expect(r["detail"], str, "detail"), risk_kind in DARK_FEED_KINDS
        )
    if kind == "planner_rejected":
        return PlannerRejected(expect(r["code"], str, "code"), expect(r["detail"], str, "detail"))
    if kind == "late_no_decision":
        _reason(r["reason"], ("warming",))
        return LateNoDecision("warming", _decode_state(r["state"]))
    state = _decode_state(r["state"])
    intents = expect(r["intents"], list, "intents")
    return Decision(state, tuple(_decode_intent(intent) for intent in intents))


def decode_result(stdout: bytes, *, phase: Phase, request_id: str) -> WireResult:
    """The supervisor's strict reader of one child's stdout."""
    expect_phase(phase)
    if len(stdout) > MAX_STDOUT_BYTES:
        raise WireError("output_too_large", f"{len(stdout)} > {MAX_STDOUT_BYTES} bytes")
    body = stdout[:-1] if stdout.endswith(b"\n") else stdout
    root = expect_object(parse_canonical(body), _ENVELOPE_KEYS, "result envelope")
    expect_wire(root["wire"])
    if root["phase"] != phase or root["request_id"] != request_id:
        raise WireError("echo_mismatch", "result does not echo this phase and request_id")
    result = root["result"]
    if type(result) is not dict:
        raise WireError("bad_type", "result must be an object")
    kind = result.get("kind")
    if type(kind) is not str or kind not in _KIND_FIELDS:
        raise WireError("unknown_kind", f"unknown result kind {kind!r}")
    if kind not in _PHASE_KINDS[phase]:
        raise WireError("forbidden_kind", f"{kind} is not a phase {phase} result")
    return _decode_body(kind, expect_object(result, _KIND_FIELDS[kind], f"result.{kind}"))
