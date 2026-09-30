"""The frozen planner wire, version 1: protected limits, identity and the request codec.

Pure: bytes in, typed planner values out — no subprocess, no filesystem. The supervisor encodes
one child's ``request.json`` and ``bars.arrow`` with :func:`encode_request`; the child rebuilds the
planner inputs with :func:`decode_request`, which refuses anything but the exact canonical
encoding (Story 1.3c contract §4, §6). ``resolved_config`` is the recorded config object, never
overlaid with the gate universe; the child re-serializes it in the planner's own canonical form so
Phase A re-hashes it against ``config_hash`` exactly as in process. Strict JSON and typed fields
live in ``frozen_wire_json``, the bar file in ``frozen_wire_arrow``, results in
``frozen_wire_result``; every protected constant is importable from here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from typing import Any, Final

from algua.contracts.canonical import FROZEN_WIRE
from algua.live.frozen_wire_arrow import BARS_FILE as BARS_FILE
from algua.live.frozen_wire_arrow import bars_reference, decode_referenced_bars, encode_bars
from algua.live.frozen_wire_json import MAX_JSON_COLLECTION as MAX_JSON_COLLECTION
from algua.live.frozen_wire_json import MAX_JSON_DEPTH as MAX_JSON_DEPTH
from algua.live.frozen_wire_json import (
    Phase,
    WireError,
    WireTooLarge,
    canonical_bytes,
    check_limits,
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
    is_hex,
    loads_strict,
    optional,
    parse_canonical,
)
from algua.live.planner_contract import (
    BOUNDARY_VERSION,
    CapturedStrategyState,
    EarlyPlannerInput,
    LatePlannerInput,
    VenueBelief,
    VenueBeliefDisabled,
    VenueBeliefEnabled,
    VenueBeliefPending,
)

# Protected constants of wire version 1 (contract §6); the JSON limits and BARS_FILE live with the
# code that enforces them and are re-exported above.
TIMEOUT_SECONDS: Final = 60
MAX_REQUEST_BYTES: Final = 256 * 1024
MAX_BARS_BYTES: Final = 256 * 1024 * 1024
MAX_STDOUT_BYTES: Final = 1024 * 1024
STDERR_CAPTURE_BYTES: Final = 64 * 1024
MAX_DIAGNOSTIC_BYTES: Final = 8 * 1024
KILL_GRACE_SECONDS: Final = 2.0
REQUEST_FILE: Final = "request.json"
TIMEFRAME: Final = "1d"

_ROOT_KEYS = frozenset(
    {"wire", "phase", "request_id", "strategy_name", "deployment_id", "artifact_id",
     "manifest_digest", "bundle_digest", "environment_digest", "early", "late"}
)
_EARLY_KEYS = frozenset(
    {"boundary_version", "config_hash", "resolved_config", "now", "timeframe", "calendar_code",
     "early_positions", "gate_universe", "max_drawdown", "bars"}
)
_LATE_KEYS = frozenset({"phase_a_binding", "captured"})
_CAPTURED_KEYS = frozenset(
    {"request_id", "sizing_equity", "drawdown_equity", "quantities", "market_values",
     "persisted_peak_equity", "venue_belief"}
)


@dataclass(frozen=True)
class WireIdentity:
    strategy_name: str
    deployment_id: int
    artifact_id: int
    manifest_digest: str
    bundle_digest: str
    environment_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.strategy_name, str) or not self.strategy_name:
            raise WireError("bad_identity", "strategy_name must be a non-empty string")
        for name in ("deployment_id", "artifact_id"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise WireError("bad_identity", f"{name} must be a positive integer")
        for name in ("manifest_digest", "bundle_digest", "environment_digest"):
            if not is_hex(getattr(self, name), 64):
                raise WireError("bad_identity", f"{name} must be 64 lowercase hex chars")


@dataclass(frozen=True)
class DecodedRequest:
    phase: Phase
    identity: WireIdentity
    early: EarlyPlannerInput
    late: LatePlannerInput | None


def _check_literals(boundary_version: Any, timeframe: Any) -> None:
    if type(boundary_version) is not int or boundary_version != BOUNDARY_VERSION:
        raise WireError("unsupported_boundary_version", str(boundary_version))
    if type(timeframe) is not str or timeframe != TIMEFRAME:
        raise WireError("unsupported_timeframe", str(timeframe))


def _config_object(text: Any) -> dict[str, Any]:
    value = loads_strict(text) if isinstance(text, str) else None
    if type(value) is not dict:
        raise WireError("bad_resolved_config", "resolved_config_json must hold a JSON object")
    return value


def _planner_config_json(config: dict[str, Any]) -> str:
    """The planner's canonical form (`planner_binding.resolved_config_digest` accepts only it)."""
    return json.dumps(
        config, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def _encode_belief(belief: VenueBelief) -> dict[str, Any]:
    if isinstance(belief, VenueBeliefPending):
        raise WireError("pending_venue_belief", "a pending belief never crosses the wire")
    if isinstance(belief, VenueBeliefDisabled) and belief.tag == "disabled":
        return {"kind": "disabled"}
    if isinstance(belief, VenueBeliefEnabled) and belief.tag == "enabled":
        return {"kind": "enabled", "quantities": encode_pairs(belief.quantities, "venue_belief")}
    raise WireError("bad_venue_belief", "unknown venue-belief variant")


def _decode_belief(value: Any) -> VenueBelief:
    kind = value.get("kind") if type(value) is dict else None
    if kind == "disabled":
        expect_object(value, frozenset({"kind"}), "venue_belief")
        return VenueBeliefDisabled()
    if kind == "enabled":
        body = expect_object(value, frozenset({"kind", "quantities"}), "venue_belief")
        return VenueBeliefEnabled(decode_pairs(body["quantities"], "venue_belief.quantities"))
    raise WireError("bad_venue_belief", "venue_belief kind must be 'disabled' or 'enabled'")


def _encode_late(late: LatePlannerInput) -> dict[str, Any]:
    c = late.captured
    return {
        "phase_a_binding": expect_hex(late.phase_a_binding, 64, "late.phase_a_binding"),
        "captured": {
            "request_id": expect_hex(c.request_id, 32, "captured.request_id"),
            "sizing_equity": encode_float(c.sizing_equity, "captured.sizing_equity"),
            "drawdown_equity": encode_float(c.drawdown_equity, "captured.drawdown_equity"),
            "quantities": encode_pairs(c.quantities, "captured.quantities"),
            "market_values": encode_pairs(c.market_values, "captured.market_values"),
            "persisted_peak_equity": optional(
                c.persisted_peak_equity, encode_float, "captured.persisted_peak_equity"
            ),
            "venue_belief": _encode_belief(c.venue_belief),
        },
    }


def _decode_late(value: Any) -> tuple[str, CapturedStrategyState]:
    late = expect_object(value, _LATE_KEYS, "late")
    binding = expect_hex(late["phase_a_binding"], 64, "late.phase_a_binding")
    c = expect_object(late["captured"], _CAPTURED_KEYS, "late.captured")
    return binding, CapturedStrategyState(
        request_id=expect_hex(c["request_id"], 32, "captured.request_id"),
        sizing_equity=decode_float(c["sizing_equity"], "captured.sizing_equity"),
        drawdown_equity=decode_float(c["drawdown_equity"], "captured.drawdown_equity"),
        quantities=decode_pairs(c["quantities"], "captured.quantities"),
        market_values=decode_pairs(c["market_values"], "captured.market_values"),
        persisted_peak_equity=optional(
            c["persisted_peak_equity"], decode_float, "captured.persisted_peak_equity"
        ),
        venue_belief=_decode_belief(c["venue_belief"]),
    )


def encode_request(
    phase: Phase, early: EarlyPlannerInput, late: LatePlannerInput | None, identity: WireIdentity
) -> tuple[bytes, bytes]:
    """``(request.json, bars.arrow)`` for one child; WireTooLarge past a §6 bound."""
    expect_phase(phase)
    if (phase == "a") != (late is None):
        raise WireError("phase_late_mismatch", "phase 'b' and only phase 'b' carries late state")
    if late is not None and late.early is not early:
        raise WireError("late_early_mismatch", "late state belongs to another early input")
    carried = (early.strategy_name, early.deployment_id, early.artifact_id, early.manifest_digest)
    wanted = (identity.strategy_name, identity.deployment_id, identity.artifact_id,
              identity.manifest_digest)
    if carried != wanted or any(type(value) is not int for value in carried[1:3]):
        raise WireError("identity_mismatch", "early input does not carry the wire identity")
    _check_literals(early.boundary_version, early.timeframe)
    bars = encode_bars(early.raw_bars)
    if len(bars) > MAX_BARS_BYTES:
        raise WireTooLarge("bars_too_large", f"{len(bars)} > {MAX_BARS_BYTES} bytes")
    reference = bars_reference(early.raw_bars)
    universe = expect(early.gate_universe, tuple, "gate_universe")
    root = {
        "wire": dict(FROZEN_WIRE),
        "phase": phase,
        "request_id": expect_hex(early.request_id, 32, "request_id"),
        "strategy_name": identity.strategy_name,
        "deployment_id": identity.deployment_id,
        "artifact_id": identity.artifact_id,
        "manifest_digest": identity.manifest_digest,
        "bundle_digest": identity.bundle_digest,
        "environment_digest": identity.environment_digest,
        "early": {
            "boundary_version": BOUNDARY_VERSION,
            "config_hash": expect(early.config_hash, str, "config_hash"),
            "resolved_config": _config_object(early.resolved_config_json),
            "now": encode_ts(early.now, "now"),
            "timeframe": TIMEFRAME,
            "calendar_code": expect(early.calendar_code, str, "calendar_code"),
            "early_positions": encode_pairs(early.early_positions, "early_positions"),
            "gate_universe": [expect(symbol, str, "gate_universe") for symbol in universe],
            "max_drawdown": optional(early.max_drawdown, encode_float, "max_drawdown"),
            "bars": reference,
        },
        "late": None if late is None else _encode_late(late),
    }
    try:
        check_limits(root)
    except WireError as exc:
        raise WireTooLarge(exc.reason, str(exc)) from None
    data = canonical_bytes(root)
    if len(data) > MAX_REQUEST_BYTES:
        raise WireTooLarge("request_too_large", f"{len(data)} > {MAX_REQUEST_BYTES} bytes")
    return data, bars


def decode_request(request_json: bytes, bars_arrow: bytes) -> DecodedRequest:
    """The child's strict reader: exact canonical bytes, exact keys and types, matching bars."""
    if len(request_json) > MAX_REQUEST_BYTES:
        raise WireError("request_too_large", f"{len(request_json)} > {MAX_REQUEST_BYTES} bytes")
    if len(bars_arrow) > MAX_BARS_BYTES:
        raise WireError("bars_too_large", f"{len(bars_arrow)} > {MAX_BARS_BYTES} bytes")
    root = expect_object(parse_canonical(request_json), _ROOT_KEYS, "request")
    expect_wire(root["wire"])
    phase = expect_phase(root["phase"])
    if (phase == "a") != (root["late"] is None):
        raise WireError("phase_late_mismatch", "phase 'b' and only phase 'b' carries late state")
    request_id = expect_hex(root["request_id"], 32, "request_id")
    identity = WireIdentity(**{field.name: root[field.name] for field in fields(WireIdentity)})
    e = expect_object(root["early"], _EARLY_KEYS, "early")
    _check_literals(e["boundary_version"], e["timeframe"])
    config_hash: str = expect(e["config_hash"], str, "early.config_hash")
    config = _planner_config_json(expect(e["resolved_config"], dict, "early.resolved_config"))
    now = decode_ts(e["now"], "early.now")
    calendar_code: str = expect(e["calendar_code"], str, "early.calendar_code")
    positions = decode_pairs(e["early_positions"], "early.early_positions")
    gate_universe = tuple(
        expect(symbol, str, "early.gate_universe")
        for symbol in expect(e["gate_universe"], list, "early.gate_universe")
    )
    max_drawdown = optional(e["max_drawdown"], decode_float, "early.max_drawdown")
    late = None if root["late"] is None else _decode_late(root["late"])
    raw_bars = decode_referenced_bars(bars_arrow, e["bars"])  # last: the bulk read
    early = EarlyPlannerInput(
        boundary_version=BOUNDARY_VERSION,
        request_id=request_id,
        strategy_name=identity.strategy_name,
        deployment_id=identity.deployment_id,
        artifact_id=identity.artifact_id,
        manifest_digest=identity.manifest_digest,
        config_hash=config_hash,
        resolved_config_json=config,
        now=now,
        timeframe=TIMEFRAME,
        calendar_code=calendar_code,
        raw_bars=raw_bars,
        early_positions=positions,
        gate_universe=gate_universe,
        max_drawdown=max_drawdown,
    )
    return DecodedRequest(
        phase, identity, early, None if late is None else LatePlannerInput(early, *late)
    )
