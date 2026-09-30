"""Story 1.3c contract §4/§6: the frozen planner's request codec (request.json + bars.arrow).

The codec is pure: bytes in, typed planner values out. A request that survives `decode_request` is
planner-equivalent to the one the supervisor encoded (same Phase A binding, same Phase B result),
and every byte-level or structural deviation is refused with a machine reason.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import pytest

from algua.contracts.canonical import FROZEN_WIRE, canonical_json
from algua.contracts.types import ExecutionContract
from algua.live import frozen_wire
from algua.live.frozen_wire import (
    BARS_FILE,
    MAX_JSON_COLLECTION,
    MAX_JSON_DEPTH,
    REQUEST_FILE,
    WireError,
    WireIdentity,
    WireTooLarge,
    check_limits,
    decode_request,
    encode_request,
)
from algua.live.planner import phase_a, phase_b
from algua.live.planner_binding import bars_digest, resolved_config_digest
from algua.live.planner_contract import (
    BOUNDARY_VERSION,
    CapturedStrategyState,
    EarlyPlannerInput,
    LatePlannerInput,
    SnapshotRequired,
    VenueBeliefDisabled,
    VenueBeliefEnabled,
    VenueBeliefPending,
)
from algua.strategies.base import LoadedStrategy, StrategyConfig, config_hash

NOW = datetime(2023, 1, 5, tzinfo=UTC)
REQUEST_ID = "0123456789abcdef0123456789abcdef"
IDENTITY = WireIdentity(
    strategy_name="test",
    deployment_id=7,
    artifact_id=11,
    manifest_digest="a" * 64,
    bundle_digest="b" * 64,
    environment_digest="c" * 64,
)
DELETE = object()


def _identity(scores, view, params):
    return scores


def make_strategy(params: dict | None = None) -> LoadedStrategy:
    def signal(view, params):
        return pd.Series({"AAA": 0.5})

    return LoadedStrategy(
        config=StrategyConfig(
            name="test",
            universe=["AAA"],
            execution=ExecutionContract(rebalance_frequency="1d", warmup_bars=0),
            params={} if params is None else params,
            construction="top_k_equal_weight",
            construction_params={"top_k": 1},
        ),
        signal_fn=signal,
        construct_fn=_identity,
    )


def make_bars(*, reverse: bool = False) -> pd.DataFrame:
    rows = [
        {
            "timestamp": datetime(2023, 1, day, tzinfo=UTC),
            "symbol": symbol,
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "adj_close": price,
            "volume": 100.0,
        }
        for day in (2, 3, 4)
        for symbol, price in (("AAA", 10.0), ("OLD", 5.0))
    ]
    if reverse:
        rows.reverse()
    return pd.DataFrame(rows).set_index("timestamp")


def make_early(
    strategy: LoadedStrategy | None = None,
    *,
    bars: pd.DataFrame | None = None,
    positions: dict[str, Any] | None = None,
    now: datetime = NOW,
    max_drawdown: float | None = 0.1,
) -> EarlyPlannerInput:
    strategy = make_strategy() if strategy is None else strategy
    resolved = json.dumps(
        strategy.config.model_dump(mode="json"), sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    )
    return EarlyPlannerInput(
        boundary_version=BOUNDARY_VERSION,
        request_id=REQUEST_ID,
        strategy_name="test",
        deployment_id=7,
        artifact_id=11,
        manifest_digest="a" * 64,
        config_hash=config_hash(strategy),
        resolved_config_json=resolved,
        now=now,
        timeframe="1d",
        calendar_code="XNYS",
        raw_bars=make_bars() if bars is None else bars,
        early_positions={} if positions is None else positions,
        gate_universe=("AAA",),
        max_drawdown=max_drawdown,
    )


def make_captured(
    *,
    belief: Any = None,
    quantities: dict[str, Any] | None = None,
    market_values: dict[str, Any] | None = None,
    sizing: Any = 100.0,
) -> CapturedStrategyState:
    return CapturedStrategyState(
        request_id=REQUEST_ID,
        sizing_equity=sizing,
        drawdown_equity=100.0,
        quantities={"OLD": 2.0} if quantities is None else quantities,
        market_values={"OLD": 20.0} if market_values is None else market_values,
        persisted_peak_equity=100.0,
        venue_belief=VenueBeliefDisabled() if belief is None else belief,
    )


def make_late(
    strategy: LoadedStrategy, early: EarlyPlannerInput, captured: CapturedStrategyState
) -> LatePlannerInput:
    first = phase_a(strategy, early)
    assert isinstance(first, SnapshotRequired)
    return LatePlannerInput(early, first.phase_a_binding, captured)


def encoded(phase: str = "a") -> tuple[bytes, bytes]:
    strategy = make_strategy()
    early = make_early(strategy)
    late = None if phase == "a" else make_late(strategy, early, make_captured())
    return encode_request(phase, early, late, IDENTITY)


def edit(data: bytes, path: tuple[Any, ...], value: Any) -> bytes:
    """Re-encode a canonical document with one nested field replaced (or deleted)."""
    root = json.loads(data)
    node = root
    for key in path[:-1]:
        node = node[key]
    if value is DELETE:
        del node[path[-1]]
    else:
        node[path[-1]] = value
    return canonical_json(root).encode("utf-8")


def arrow_bytes(table: pa.Table) -> bytes:
    sink = pa.BufferOutputStream()
    with ipc.new_file(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes()


# --- constants ----------------------------------------------------------------------------------


def test_protected_constants_are_wire_version_1():
    assert frozen_wire.TIMEOUT_SECONDS == 60
    assert frozen_wire.MAX_REQUEST_BYTES == 256 * 1024
    assert frozen_wire.MAX_BARS_BYTES == 256 * 1024 * 1024
    assert frozen_wire.MAX_STDOUT_BYTES == 1024 * 1024
    assert frozen_wire.STDERR_CAPTURE_BYTES == 64 * 1024
    assert MAX_JSON_DEPTH == 16
    assert MAX_JSON_COLLECTION == 10_000
    assert frozen_wire.MAX_DIAGNOSTIC_BYTES == 8 * 1024
    assert frozen_wire.KILL_GRACE_SECONDS == 2.0
    assert (BARS_FILE, REQUEST_FILE) == ("bars.arrow", "request.json")


def test_wire_error_is_a_value_error_with_a_machine_reason():
    error = WireTooLarge("request_too_large", "detail")
    assert isinstance(error, WireError) and isinstance(error, ValueError)
    assert error.reason == "request_too_large"


# --- encoding shape -----------------------------------------------------------------------------


def test_phase_a_request_has_exactly_the_contract_fields():
    request_json, _ = encoded("a")
    root = json.loads(request_json)
    assert request_json == canonical_json(root).encode("utf-8")
    assert set(root) == {
        "wire", "phase", "request_id", "strategy_name", "deployment_id", "artifact_id",
        "manifest_digest", "bundle_digest", "environment_digest", "early", "late",
    }
    assert root["wire"] == FROZEN_WIRE and root["phase"] == "a" and root["late"] is None
    assert root["request_id"] == REQUEST_ID
    assert (root["deployment_id"], root["artifact_id"]) == (7, 11)
    assert (root["bundle_digest"], root["environment_digest"]) == ("b" * 64, "c" * 64)
    early = root["early"]
    assert set(early) == {
        "boundary_version", "config_hash", "resolved_config", "now", "timeframe",
        "calendar_code", "early_positions", "gate_universe", "max_drawdown", "bars",
    }
    assert early["now"] == "2023-01-05T00:00:00.000000+00:00"
    assert early["max_drawdown"] == (0.1).hex()
    assert early["bars"] == {
        "file": "bars.arrow", "rows": 6, "bars_digest": bars_digest(make_bars())
    }


def test_resolved_config_is_the_recorded_object_and_gate_universe_travels_alone():
    strategy = make_strategy()
    early = make_early(strategy)
    root = json.loads(encode_request("a", early, None, IDENTITY)[0])
    assert root["early"]["resolved_config"] == json.loads(early.resolved_config_json)
    assert root["early"]["gate_universe"] == ["AAA"]


def test_mappings_are_sorted_symbol_value_pairs_and_universe_keeps_its_order():
    early = make_early(positions={"ZZZ": 1.0, "AAA": 2})
    early = EarlyPlannerInput(**{**early.__dict__, "gate_universe": ("ZZZ", "AAA")})
    root = json.loads(encode_request("a", early, None, IDENTITY)[0])
    assert root["early"]["early_positions"] == [["AAA", (2.0).hex()], ["ZZZ", (1.0).hex()]]
    assert root["early"]["gate_universe"] == ["ZZZ", "AAA"]


def test_phase_b_carries_every_captured_field():
    strategy = make_strategy()
    early = make_early(strategy)
    late = make_late(strategy, early, make_captured(belief=VenueBeliefEnabled({"OLD": 2.0})))
    root = json.loads(encode_request("b", early, late, IDENTITY)[0])
    assert root["late"] == {
        "phase_a_binding": late.phase_a_binding,
        "captured": {
            "request_id": REQUEST_ID,
            "sizing_equity": (100.0).hex(),
            "drawdown_equity": (100.0).hex(),
            "quantities": [["OLD", (2.0).hex()]],
            "market_values": [["OLD", (20.0).hex()]],
            "persisted_peak_equity": (100.0).hex(),
            "venue_belief": {"kind": "enabled", "quantities": [["OLD", (2.0).hex()]]},
        },
    }
    disabled = make_late(strategy, early, make_captured())
    root = json.loads(encode_request("b", early, disabled, IDENTITY)[0])
    assert root["late"]["captured"]["venue_belief"] == {"kind": "disabled"}


def test_identical_inputs_encode_to_identical_bytes():
    assert encoded("a") == encoded("a")
    assert encoded("b") == encoded("b")


# --- round trips --------------------------------------------------------------------------------


def test_phase_a_round_trip_is_planner_equivalent():
    strategy = make_strategy()
    early = make_early(strategy, positions={"OLD": 2.0})
    request_json, bars_arrow = encode_request("a", early, None, IDENTITY)
    decoded = decode_request(request_json, bars_arrow)
    assert decoded.phase == "a" and decoded.late is None and decoded.identity == IDENTITY
    assert decoded.early.resolved_config_json == early.resolved_config_json
    resolved_config_digest(decoded.early.resolved_config_json)  # the planner's own form
    assert decoded.early.early_positions == {"OLD": 2.0}
    assert decoded.early.gate_universe == ("AAA",)
    pd.testing.assert_frame_equal(decoded.early.raw_bars, early.raw_bars, check_index_type=True)
    assert phase_a(strategy, decoded.early) == phase_a(strategy, early)  # same Phase A binding
    assert encode_request("a", decoded.early, None, decoded.identity) == (request_json, bars_arrow)


@pytest.mark.parametrize("belief", [None, VenueBeliefEnabled({"OLD": 2.0})])
def test_phase_b_round_trip_is_planner_equivalent(belief):
    strategy = make_strategy()
    early = make_early(strategy)
    late = make_late(strategy, early, make_captured(belief=belief))
    request_json, bars_arrow = encode_request("b", early, late, IDENTITY)
    decoded = decode_request(request_json, bars_arrow)
    assert decoded.phase == "b" and decoded.late is not None
    assert decoded.late.early is decoded.early
    assert decoded.late.captured == late.captured
    assert phase_b(strategy, decoded.late) == phase_b(strategy, late)
    assert encode_request("b", decoded.early, decoded.late, decoded.identity) == (
        request_json,
        bars_arrow,
    )


def test_non_ascii_recorded_config_reaches_the_planner_in_its_canonical_form():
    strategy = make_strategy(params={"label": "café"})
    early = make_early(strategy)
    request_json, bars_arrow = encode_request("a", early, None, IDENTITY)
    assert "café".encode() in request_json
    decoded = decode_request(request_json, bars_arrow)
    assert decoded.early.resolved_config_json == early.resolved_config_json
    assert phase_a(strategy, decoded.early) == phase_a(strategy, early)


def test_special_floats_cross_losslessly():
    special = {"A": math.nan, "B": math.inf, "C": -math.inf, "D": -0.0, "E": 5e-324}
    strategy = make_strategy()
    early = make_early(strategy, max_drawdown=-0.0)
    captured = make_captured(
        quantities=special, market_values=special, sizing=math.nan,
        belief=VenueBeliefEnabled(special),
    )
    late = LatePlannerInput(early, "f" * 64, captured)
    decoded = decode_request(*encode_request("b", early, late, IDENTITY))
    assert decoded.late is not None
    got = decoded.late.captured
    for mapping in (got.quantities, got.market_values, got.venue_belief.quantities):
        assert {k: v.hex() for k, v in mapping.items()} == {
            k: float(v).hex() for k, v in special.items()
        }
    assert math.isnan(got.sizing_equity)
    assert math.copysign(1.0, decoded.early.max_drawdown) == -1.0


def test_null_max_drawdown_and_peak_round_trip():
    strategy = make_strategy()
    early = make_early(strategy, max_drawdown=None)
    captured = CapturedStrategyState(REQUEST_ID, 100.0, 100.0, {}, {}, None, VenueBeliefDisabled())
    late = LatePlannerInput(early, "f" * 64, captured)
    decoded = decode_request(*encode_request("b", early, late, IDENTITY))
    assert decoded.early.max_drawdown is None
    assert decoded.late is not None and decoded.late.captured == captured


def test_empty_bars_round_trip():
    empty = make_bars().iloc[0:0]
    early = make_early(bars=empty)
    request_json, bars_arrow = encode_request("a", early, None, IDENTITY)
    decoded = decode_request(request_json, bars_arrow)
    assert len(decoded.early.raw_bars) == 0
    assert bars_digest(decoded.early.raw_bars) == bars_digest(empty)


def test_bar_row_order_is_preserved():
    bars = make_bars(reverse=True)
    decoded = decode_request(*encode_request("a", make_early(bars=bars), None, IDENTITY))
    assert list(decoded.early.raw_bars.index) == list(bars.index)
    assert list(decoded.early.raw_bars["symbol"]) == list(bars["symbol"])


# --- encode refusals ----------------------------------------------------------------------------


def _late_for(early: EarlyPlannerInput, **captured: Any) -> LatePlannerInput:
    return LatePlannerInput(early, "f" * 64, make_captured(**captured))


@pytest.mark.parametrize(
    ("phase", "build", "reason"),
    [
        ("a", lambda e: (e, _late_for(e)), "phase_late_mismatch"),
        ("b", lambda e: (e, None), "phase_late_mismatch"),
        ("c", lambda e: (e, None), "bad_phase"),
        ("b", lambda e: (e, _late_for(make_early())), "late_early_mismatch"),
        ("a", lambda e: (EarlyPlannerInput(**{**e.__dict__, "deployment_id": 8}), None),
         "identity_mismatch"),
        ("a", lambda e: (EarlyPlannerInput(**{**e.__dict__, "manifest_digest": None}), None),
         "identity_mismatch"),
        ("b", lambda e: (e, _late_for(e, belief=VenueBeliefPending())), "pending_venue_belief"),
        ("a", lambda e: (make_early(now=datetime(2023, 1, 5)), None), "bad_timestamp"),
        ("a", lambda e: (make_early(now=datetime(2023, 1, 5, tzinfo=timezone(timedelta(hours=2)))),
                         None), "bad_timestamp"),
        ("a", lambda e: (make_early(positions={"AAA": True}), None), "bad_float"),
        ("a", lambda e: (make_early(positions={"": 1.0}), None), "bad_mapping"),
        ("b", lambda e: (e, _late_for(e, sizing="100")), "bad_float"),
        ("a", lambda e: (EarlyPlannerInput(**{**e.__dict__, "resolved_config_json": "[]"}), None),
         "bad_resolved_config"),
        ("a", lambda e: (EarlyPlannerInput(**{**e.__dict__, "timeframe": "1h"}), None),
         "unsupported_timeframe"),
        ("a", lambda e: (EarlyPlannerInput(**{**e.__dict__, "boundary_version": 2}), None),
         "unsupported_boundary_version"),
        ("a", lambda e: (make_early(bars=pd.concat([make_bars(), make_bars()])), None),
         "invalid_bars"),
    ],
)
def test_encode_refuses_inputs_it_cannot_carry_faithfully(phase, build, reason):
    early, late = build(make_early())
    with pytest.raises(WireError) as caught:
        encode_request(phase, early, late, IDENTITY)
    assert caught.value.reason == reason
    assert not isinstance(caught.value, WireTooLarge)


def test_oversize_request_metadata_is_too_large_before_launch():
    early = make_early()
    wide = EarlyPlannerInput(
        **{**early.__dict__, "gate_universe": tuple(f"SYM{i:036d}" for i in range(9_000))}
    )
    with pytest.raises(WireTooLarge) as caught:
        encode_request("a", wide, None, IDENTITY)
    assert caught.value.reason == "request_too_large"


def test_oversize_bars_are_too_large_before_launch(monkeypatch):
    monkeypatch.setattr(frozen_wire, "MAX_BARS_BYTES", 64)
    with pytest.raises(WireTooLarge) as caught:
        encode_request("a", make_early(), None, IDENTITY)
    assert caught.value.reason == "bars_too_large"


def test_over_deep_request_is_too_large():
    nested: Any = 1
    for _ in range(MAX_JSON_DEPTH):
        nested = {"n": nested}
    early = make_early()
    deep = EarlyPlannerInput(
        **{**early.__dict__, "resolved_config_json": json.dumps({"n": nested})}
    )
    with pytest.raises(WireTooLarge) as caught:
        encode_request("a", deep, None, IDENTITY)
    assert caught.value.reason == "json_depth"


# --- decode refusals ----------------------------------------------------------------------------


def test_json_limits_are_inclusive_bounds():
    deepest: Any = []
    for _ in range(MAX_JSON_DEPTH - 1):
        deepest = [deepest]
    check_limits(deepest)
    with pytest.raises(WireError, match="json_depth"):
        check_limits([deepest])
    check_limits(list(range(MAX_JSON_COLLECTION)))
    with pytest.raises(WireError, match="json_collection"):
        check_limits({"k": list(range(MAX_JSON_COLLECTION + 1))})


def _nested(depth: int) -> Any:
    value: Any = 0
    for _ in range(depth):
        value = {"n": value}
    return value


HEX_ONE = (1.0).hex()


@pytest.mark.parametrize(
    ("phase", "path", "value", "reason"),
    [
        ("a", ("deployment_id",), True, "bad_identity"),
        ("a", ("strategy_name",), 5, "bad_identity"),
        ("a", ("early", "bars", "rows"), True, "bad_type"),
        ("a", ("wire", "version"), True, "bad_wire"),
        ("a", ("wire", "version"), 2, "bad_wire"),
        ("a", ("wire", "name"), "other-planner", "bad_wire"),
        ("a", ("early", "max_drawdown"), 0.1, "bad_float"),
        ("a", ("early", "max_drawdown"), "0x1p-4", "bad_float"),
        ("a", ("early", "max_drawdown"), "0.1", "bad_float"),
        ("a", ("early", "max_drawdown"), "-nan", "bad_float"),
        ("a", ("early", "max_drawdown"), "0X1.999999999999AP-4", "bad_float"),
        ("b", ("late", "captured", "sizing_equity"), False, "bad_float"),
        ("a", ("early", "now"), "2023-01-05T00:00:00.000000", "bad_timestamp"),
        ("a", ("early", "now"), "2023-01-05T02:00:00.000000+02:00", "bad_timestamp"),
        ("a", ("early", "now"), "2023-01-05T00:00:00+00:00", "bad_timestamp"),
        ("a", ("early", "now"), "2023-01-05T00:00:00.000000Z", "bad_timestamp"),
        ("a", ("early", "now"), "2023-02-30T00:00:00.000000+00:00", "bad_timestamp"),
        ("a", ("early", "early_positions"), [["ZZZ", HEX_ONE], ["AAA", HEX_ONE]], "mapping_order"),
        ("a", ("early", "early_positions"), [["AAA", HEX_ONE], ["AAA", HEX_ONE]], "mapping_order"),
        ("a", ("early", "early_positions"), [["", HEX_ONE]], "bad_mapping"),
        ("a", ("early", "early_positions"), [["AAA"]], "bad_mapping"),
        ("b", ("late", "captured", "quantities"), [["Z", HEX_ONE], ["A", HEX_ONE]],
         "mapping_order"),
        ("a", ("early", "gate_universe"), ["AAA", 1], "bad_type"),
        ("a", ("early", "resolved_config"), [], "bad_type"),
        ("a", ("request_id",), REQUEST_ID.upper(), "bad_hex"),
        ("a", ("early", "bars", "bars_digest"), "0" * 63, "bad_hex"),
        ("b", ("late", "phase_a_binding"), "x" * 64, "bad_hex"),
        ("a", ("manifest_digest",), "a" * 63, "bad_identity"),
        ("a", ("deployment_id",), 0, "bad_identity"),
        ("a", ("phase",), "c", "bad_phase"),
        ("a", ("phase",), "b", "phase_late_mismatch"),
        ("b", ("phase",), "a", "phase_late_mismatch"),
        ("a", ("extra",), 1, "unknown_key"),
        ("a", ("early", "extra"), 1, "unknown_key"),
        ("a", ("early", "bars", "extra"), 1, "unknown_key"),
        ("b", ("late", "captured", "extra"), 1, "unknown_key"),
        ("b", ("late", "captured", "venue_belief"), {"kind": "disabled", "quantities": []},
         "unknown_key"),
        ("a", ("late",), DELETE, "missing_key"),
        ("a", ("early", "now"), DELETE, "missing_key"),
        ("b", ("late", "captured", "venue_belief"), DELETE, "missing_key"),
        ("b", ("late", "captured", "venue_belief"), {"kind": "pending"}, "bad_venue_belief"),
        ("a", ("early", "boundary_version"), 2, "unsupported_boundary_version"),
        ("a", ("early", "timeframe"), "1h", "unsupported_timeframe"),
        ("a", ("early", "bars", "file"), "other.arrow", "bad_bars_file"),
        ("a", ("early", "bars", "rows"), 7, "bars_rows"),
        ("a", ("early", "bars", "bars_digest"), "0" * 64, "bars_digest"),
        ("a", ("early", "gate_universe"), ["S"] * (MAX_JSON_COLLECTION + 1), "json_collection"),
        ("a", ("early", "resolved_config"), _nested(MAX_JSON_DEPTH), "json_depth"),
    ],
)
def test_decode_refuses_structural_deviations(phase, path, value, reason):
    request_json, bars_arrow = encoded(phase)
    with pytest.raises(WireError) as caught:
        decode_request(edit(request_json, path, value), bars_arrow)
    assert caught.value.reason == reason


def _raw_cases() -> list[tuple[str, Any, str]]:
    def whitespace(data: bytes) -> bytes:
        return json.dumps(json.loads(data), sort_keys=True, ensure_ascii=False, indent=1).encode()

    def unsorted(data: bytes) -> bytes:
        root = json.loads(data)
        return json.dumps(dict(reversed(root.items())), separators=(",", ":")).encode()

    hex_dd = f'"max_drawdown":"{(0.1).hex()}"'.encode()
    return [
        ("duplicate_key", lambda d: b'{"phase":"a",' + d[1:], "duplicate_key"),
        ("trailing_data", lambda d: d + b"{}", "invalid_json"),
        ("trailing_newline", lambda d: d + b"\n", "not_canonical"),
        ("whitespace", whitespace, "not_canonical"),
        ("unsorted_keys", unsorted, "not_canonical"),
        ("nan_token", lambda d: d.replace(hex_dd, b'"max_drawdown":NaN'), "non_finite"),
        ("infinite_number", lambda d: d.replace(hex_dd, b'"max_drawdown":1e999'), "non_finite"),
        ("invalid_utf8", lambda d: d.replace(b'"test"', b'"t\xffst"'), "invalid_utf8"),
        ("not_json", lambda d: b"request", "invalid_json"),
        ("oversize", lambda d: b" " * (frozen_wire.MAX_REQUEST_BYTES + 1), "request_too_large"),
    ]


@pytest.mark.parametrize(("case", "mutate", "reason"), _raw_cases(), ids=lambda v: str(v)[:20])
def test_decode_refuses_non_canonical_bytes(case, mutate, reason):
    request_json, bars_arrow = encoded("a")
    mutated = mutate(request_json)
    assert mutated != request_json, case
    with pytest.raises(WireError) as caught:
        decode_request(mutated, bars_arrow)
    assert caught.value.reason == reason


def _drifted_bars() -> list[tuple[str, bytes, str]]:
    table = pa.ipc.open_file(pa.BufferReader(encoded("a")[1])).read_all()
    volume = table.schema.get_field_index("volume")
    close = table.schema.get_field_index("close")
    as_int = table.set_column(volume, "volume", table.column("volume").cast(pa.int64()))
    reordered = table.select(["symbol", "timestamp", *table.column_names[2:]])
    micros = table.set_column(0, "timestamp", table.column(0).cast(pa.timestamp("us", tz="UTC")))
    new_york = table.set_column(
        0, "timestamp", table.column(0).cast(pa.timestamp("ns", tz="America/New_York"))
    )
    tagged = table.replace_schema_metadata({"pandas": "{}"})
    closes = table.column("close").to_pylist()
    nulled = table.set_column(close, "close", pa.array([None, *closes[1:]], type=pa.float64()))
    return [
        ("int_volume", arrow_bytes(as_int), "arrow_schema"),
        ("reordered", arrow_bytes(reordered), "arrow_schema"),
        ("microseconds", arrow_bytes(micros), "arrow_schema"),
        ("new_york", arrow_bytes(new_york), "arrow_schema"),
        ("metadata", arrow_bytes(tagged), "arrow_schema"),
        ("null_close", arrow_bytes(nulled), "arrow_null"),
        ("garbage", b"ARROW1-not-really", "invalid_arrow"),
        ("other_rows", encode_request("a", make_early(bars=make_bars(reverse=True)), None,
                                      IDENTITY)[1], "bars_digest"),
    ]


@pytest.mark.parametrize(("case", "bars_arrow", "reason"), _drifted_bars(),
                         ids=lambda v: str(v)[:20])
def test_decode_refuses_bars_that_drift_from_the_contract(case, bars_arrow, reason):
    request_json, _ = encoded("a")
    with pytest.raises(WireError) as caught:
        decode_request(request_json, bars_arrow)
    assert caught.value.reason == reason, case


def test_decode_refuses_oversize_bars(monkeypatch):
    request_json, bars_arrow = encoded("a")
    monkeypatch.setattr(frozen_wire, "MAX_BARS_BYTES", len(bars_arrow) - 1)
    with pytest.raises(WireError) as caught:
        decode_request(request_json, bars_arrow)
    assert caught.value.reason == "bars_too_large"


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"deployment_id": True}, "bad_identity"),
        ({"artifact_id": -1}, "bad_identity"),
        ({"strategy_name": ""}, "bad_identity"),
        ({"bundle_digest": "B" * 64}, "bad_identity"),
    ],
)
def test_wire_identity_is_strict(change, reason):
    with pytest.raises(WireError) as caught:
        WireIdentity(**{**IDENTITY.__dict__, **change})
    assert caught.value.reason == reason
