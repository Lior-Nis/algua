"""The frozen planner child, wire version 1: one planner phase run from a Story 1.3b bundle.

The supervisor launches ``<env>/bin/python -I -B -c BOOTSTRAP <bundle_root> <invocation_dir>``
(Story 1.3c contract §5); ``BOOTSTRAP`` puts the bundle root first on ``sys.path`` and calls
:func:`main`. The child refuses with ``EXIT_UNSUPPORTED`` unless the running ``algua`` package is
the bundle's, the bundle's protocol stamp is wire and planner boundary version 1, the bundle and
interpreter environment are the ones the request names (their store locators end in their
digests), and both the bundle's recorded config and its strategy's ``CONFIG`` are exactly the
request's recorded config. It decodes the invocation's read-only ``request.json`` and
``bars.arrow`` (``EXIT_BAD_REQUEST`` when the wire codec refuses them), overlays the gate universe
as the paper runtime does, runs the phase, checks that every loaded ``algua`` module came from the
bundle, and writes the single result document to stdout. It writes no file and catches nothing
else: an unexpected exception exits 1 with a traceback on stderr (``frozen_exit_abnormal``).

Its import surface — the wire codec, the planner, the strategy loader and what they reach — never
includes the registry, data, CLI, execution or operator layers.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path, PurePath
from typing import Any

import algua
from algua.contracts.canonical import canonical_json
from algua.live.frozen_wire import (
    BARS_FILE,
    EXIT_BAD_REQUEST,
    EXIT_OK,
    EXIT_UNSUPPORTED,
    MAX_BARS_BYTES,
    MAX_REQUEST_BYTES,
    REQUEST_FILE,
    DecodedRequest,
    WireError,
    decode_request,
)
from algua.live.frozen_wire_json import expect_wire, parse_canonical
from algua.live.frozen_wire_result import encode_result
from algua.live.planner import phase_a, phase_b
from algua.live.planner_contract import BOUNDARY_VERSION
from algua.strategies.base import LoadedStrategy
from algua.strategies.loader import load_tradable_strategy

PROTOCOL_FILE = "_algua/protocol.json"
RESOLVED_CONFIG_FILE = "_algua/resolved-config.json"
# The keys Story 1.3b's preparation writes into protocol.json.
_PROTOCOL_KEYS = frozenset(
    {"descriptor_version", "planner_boundary_version", "planner_protocol_version", "frozen_wire"}
)
# A recorded config also travels inside request.json, so it can never be larger than that.
_MAX_BUNDLE_JSON_BYTES = MAX_REQUEST_BYTES


class _Refused(Exception):
    """The child refuses its bundle, protocol, identity or module origin (``EXIT_UNSUPPORTED``)."""


def foreign_algua_modules(modules: Mapping[str, object], bundle_root: str) -> list[str]:
    """Names of loaded ``algua`` modules whose resolved source file is not under ``bundle_root``.

    An ``algua`` module without a file (namespace, built-in or blocked entry) counts as foreign;
    other modules are not inspected.
    """
    root = PurePath(os.path.realpath(bundle_root))
    foreign = []
    for name, module in modules.items():
        if name != "algua" and not name.startswith("algua."):
            continue
        file = getattr(module, "__file__", None)
        if not isinstance(file, str) or not PurePath(os.path.realpath(file)).is_relative_to(root):
            foreign.append(name)
    return sorted(foreign)


def _read(path: Path, limit: int) -> bytes:
    """At most ``limit + 1`` bytes, so an oversize file is detected without reading all of it."""
    with path.open("rb") as handle:
        return handle.read(limit + 1)


def _bundle_root() -> Path:
    file = getattr(algua, "__file__", None)
    root = Path(file).parent.parent if isinstance(file, str) else None
    if root is None or not sys.path or Path(sys.path[0]) != root:
        raise _Refused(f"the running algua package ({file}) is not the bundle at sys.path[0]")
    return root


def _bundle_bytes(bundle_root: Path, relative: str) -> bytes:
    try:
        data = _read(bundle_root / relative, _MAX_BUNDLE_JSON_BYTES)
    except OSError as exc:
        raise _Refused(f"{relative} is unreadable: {exc.strerror}") from None
    if len(data) > _MAX_BUNDLE_JSON_BYTES:
        raise _Refused(f"{relative} exceeds {_MAX_BUNDLE_JSON_BYTES} bytes")
    return data


def _check_protocol(bundle_root: Path) -> None:
    try:
        protocol = parse_canonical(_bundle_bytes(bundle_root, PROTOCOL_FILE))
        if type(protocol) is not dict or set(protocol) != _PROTOCOL_KEYS:
            raise WireError("bad_protocol", "unexpected protocol fields")
        expect_wire(protocol["frozen_wire"])
        boundary = protocol["planner_boundary_version"]
        if type(boundary) is not int or boundary != BOUNDARY_VERSION:
            raise WireError("unsupported_boundary_version", str(boundary))
        if any(type(protocol[key]) is not int
               for key in ("descriptor_version", "planner_protocol_version")):
            raise WireError("bad_protocol", "protocol versions must be integers")
    except WireError as exc:
        raise _Refused(f"{PROTOCOL_FILE} is not frozen-planner wire 1, boundary 1: {exc}") from None


def _read_request(invocation_dir: Path) -> DecodedRequest:
    return decode_request(
        _read(invocation_dir / REQUEST_FILE, MAX_REQUEST_BYTES),
        _read(invocation_dir / BARS_FILE, MAX_BARS_BYTES),
    )


def _canonical_text(value: Any) -> str:
    try:
        return canonical_json(value)
    except ValueError as exc:
        raise _Refused(f"not canonical JSON: {exc}") from None


def _load_bundle_strategy(bundle_root: Path, request: DecodedRequest) -> LoadedStrategy:
    """The bundle's strategy, after refusing any identity that is not the request's."""
    identity = request.identity
    if bundle_root.name != identity.bundle_digest:
        raise _Refused("the bundle root is not the requested bundle_digest")
    if Path(sys.prefix).name != identity.environment_digest:
        raise _Refused("the interpreter environment is not the requested environment_digest")
    recorded = _canonical_text(json.loads(request.early.resolved_config_json))
    if _bundle_bytes(bundle_root, RESOLVED_CONFIG_FILE) != recorded.encode("utf-8"):
        raise _Refused(f"{RESOLVED_CONFIG_FILE} is not the request's recorded config")
    name = identity.strategy_name
    try:
        strategy = load_tradable_strategy(name)
    except (LookupError, ValueError) as exc:
        raise _Refused(f"the bundle holds no tradable strategy {name!r}: {exc}") from None
    if strategy.config.name != name:
        raise _Refused(f"the bundle's CONFIG.name is not {name!r}")
    if _canonical_text(strategy.config.model_dump(mode="json")) != recorded:
        raise _Refused(f"the bundle's {name} CONFIG does not dump to the recorded config")
    return strategy


def _with_gate_universe(strategy: LoadedStrategy, gate_universe: tuple[str, ...]) -> LoadedStrategy:
    """The gate-universe overlay, exactly as ``registry.paper_runtime.prepare_paper_runtime``."""
    universe = list(gate_universe)
    if universe != strategy.universe:
        strategy = replace(
            strategy, config=strategy.config.model_copy(update={"universe": universe}))
    return strategy


def _run(bundle_root: Path, request: DecodedRequest) -> bytes:
    strategy = _with_gate_universe(
        _load_bundle_strategy(bundle_root, request), request.early.gate_universe)
    late = request.late  # decode_request: None exactly for phase "a"
    result = phase_a(strategy, request.early) if late is None else phase_b(strategy, late)
    payload = encode_result(request.phase, request.early.request_id, result)
    foreign = foreign_algua_modules(dict(sys.modules), str(bundle_root))
    if foreign:
        raise _Refused(f"algua modules loaded from outside the bundle: {foreign}")
    return payload


def main(invocation_dir: str) -> int:
    """Run one planner phase; the wire-v1 entry point ``BOOTSTRAP`` calls."""
    try:
        bundle_root = _bundle_root()
        _check_protocol(bundle_root)
        try:
            request = _read_request(Path(invocation_dir))
        except WireError as exc:
            sys.stderr.write(f"frozen_child: bad request: {exc}\n")
            return EXIT_BAD_REQUEST
        payload = _run(bundle_root, request)
    except _Refused as exc:
        sys.stderr.write(f"frozen_child: refused: {exc}\n")
        return EXIT_UNSUPPORTED
    sys.stdout.buffer.write(payload)
    sys.stdout.buffer.flush()
    return EXIT_OK
