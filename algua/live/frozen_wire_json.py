"""Strict canonical JSON and typed wire fields for the frozen planner wire, version 1.

The base layer of the ``frozen_wire*`` codec (Story 1.3c contract §4, §6, §7). A document is
accepted only if it is byte-identical to the canonical encoding of what it parses to, so duplicate
keys, whitespace variants and non-finite numbers are refused before any field is read; then every
field is read with its exact JSON type (a boolean is never a number). Floats cross as exact
``float.hex()`` strings, timestamps as UTC ISO-8601 with microseconds and ``+00:00``, and
symbol/value mappings as ``[symbol, value]`` pairs sorted by symbol with unique symbols.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections.abc import Callable, Mapping
from datetime import datetime
from typing import Any, Final, Literal

import pandas as pd

from algua.contracts.canonical import (
    FROZEN_WIRE,
    FROZEN_WIRE_NAME,
    FROZEN_WIRE_VERSION,
    canonical_json,
)

# Protected constants of wire version 1 (contract §6); `frozen_wire` re-exports them.
MAX_JSON_DEPTH: Final = 16
MAX_JSON_COLLECTION: Final = 10_000

type Phase = Literal["a", "b"]

_TIMESTAMP = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{6}\+00:00")


class WireError(ValueError):
    """A wire document or value the codec refuses; ``reason`` is a short machine token."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


class WireTooLarge(WireError):
    """A request exceeds a §6 bound; the supervisor refuses it before launch."""


# --- documents ------------------------------------------------------------------------------


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = dict(pairs)
    if len(result) != len(pairs):
        raise WireError("duplicate_key")
    return result


def _reject_constant(token: str) -> Any:
    raise WireError("non_finite", token)


def _finite_number(token: str) -> float:
    value = float(token)
    if not math.isfinite(value):
        raise WireError("non_finite", token)
    return value


def loads_strict(text: str) -> Any:
    """Parse JSON text refusing duplicate keys and non-finite numbers."""
    try:
        return json.loads(
            text, object_pairs_hook=_unique_object, parse_constant=_reject_constant,
            parse_float=_finite_number,
        )
    except WireError:
        raise
    except RecursionError:
        raise WireError("json_depth") from None
    except ValueError as exc:
        raise WireError("invalid_json", str(exc)) from None


def check_limits(value: Any) -> None:
    """Refuse nesting deeper than MAX_JSON_DEPTH or any list/object over MAX_JSON_COLLECTION."""
    stack: list[tuple[Any, int]] = [(value, 1)]
    while stack:
        item, depth = stack.pop()
        if isinstance(item, (dict, list, tuple)):
            if depth > MAX_JSON_DEPTH:
                raise WireError("json_depth", f"nesting exceeds {MAX_JSON_DEPTH}")
            if len(item) > MAX_JSON_COLLECTION:
                raise WireError("json_collection", f"a collection exceeds {MAX_JSON_COLLECTION}")
            children = item.values() if isinstance(item, dict) else item
            stack.extend((child, depth + 1) for child in children)


def canonical_bytes(value: Any) -> bytes:
    try:
        return canonical_json(value).encode("utf-8")
    except ValueError as exc:
        raise WireError("not_encodable", str(exc)) from None


def parse_canonical(data: bytes) -> Any:
    """Parse one UTF-8 JSON document that must be exactly its own canonical encoding."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise WireError("invalid_utf8") from None
    value = loads_strict(text)
    check_limits(value)
    try:
        canonical = canonical_json(value).encode("utf-8")
    except ValueError:
        raise WireError("not_canonical") from None
    if canonical != data:
        raise WireError("not_canonical")
    return value


# --- typed fields ---------------------------------------------------------------------------


def is_hex(value: object, length: int) -> bool:
    return isinstance(value, str) and re.fullmatch(f"[0-9a-f]{{{length}}}", value) is not None


def expect_object(value: Any, keys: frozenset[str], where: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise WireError("bad_type", f"{where} must be an object")
    if unknown := set(value) - keys:
        raise WireError("unknown_key", f"{where}: {sorted(unknown)}")
    if missing := keys - set(value):
        raise WireError("missing_key", f"{where}: {sorted(missing)}")
    return value


def expect(value: Any, kind: type, where: str) -> Any:
    """``value`` with exactly JSON type ``kind`` (``type(True) is not int``)."""
    if type(value) is not kind:
        raise WireError("bad_type", f"{where} must be {kind.__name__}")
    return value


def expect_hex(value: Any, length: int, where: str) -> str:
    if not is_hex(value, length):
        raise WireError("bad_hex", f"{where} must be {length} lowercase hex chars")
    return str(value)


def expect_wire(value: Any) -> None:
    wire = expect_object(value, frozenset({"name", "version"}), "wire")
    if (
        type(wire["name"]) is not str or wire["name"] != FROZEN_WIRE_NAME
        or type(wire["version"]) is not int or wire["version"] != FROZEN_WIRE_VERSION
    ):
        raise WireError("bad_wire", f"expected {FROZEN_WIRE}")


def expect_phase(value: Any) -> Phase:
    if type(value) is str and value == "a":
        return "a"
    if type(value) is str and value == "b":
        return "b"
    raise WireError("bad_phase", "phase must be 'a' or 'b'")


def encode_float(value: Any, where: str) -> str:
    if type(value) not in (int, float):
        raise WireError("bad_float", f"{where} must be an int or float")
    return float(value).hex()


def decode_float(value: Any, where: str) -> float:
    if type(value) is str:
        try:
            number = float.fromhex(value)
        except (ValueError, OverflowError):
            pass
        else:
            if number.hex() == value:
                return number
    raise WireError("bad_float", f"{where} must be a canonical float.hex() string")


def encode_ts(value: Any, where: str) -> str:
    stamp = pd.Timestamp(value) if isinstance(value, datetime) else None
    if stamp is None or stamp.tzinfo is None or str(stamp.tz) != "UTC" or stamp.nanosecond:
        raise WireError("bad_timestamp", f"{where} must be a UTC datetime at microsecond precision")
    return str(stamp.isoformat(timespec="microseconds"))


def decode_ts(value: Any, where: str) -> datetime:
    if type(value) is str and _TIMESTAMP.fullmatch(value):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            pass
    raise WireError("bad_timestamp", f"{where} must be UTC ISO-8601 with microseconds")


def optional[T](value: Any, codec: Callable[[Any, str], T], where: str) -> T | None:
    return None if value is None else codec(value, where)


def encode_pairs(value: Any, where: str) -> list[list[str]]:
    """A mapping (or a tuple of ``(symbol, value)`` pairs) as sorted unique pairs."""
    if isinstance(value, Mapping):
        items = list(value.items())
    elif isinstance(value, tuple):
        items = list(value)
    else:
        raise WireError("bad_mapping", f"{where} must be a mapping")
    pairs: dict[str, str] = {}
    for item in items:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise WireError("bad_mapping", f"{where} entries must be (symbol, value) pairs")
        symbol, number = item
        if not isinstance(symbol, str) or not symbol:
            raise WireError("bad_mapping", f"{where} symbols must be non-empty strings")
        normalized = unicodedata.normalize("NFC", symbol)  # as the canonical encoder will
        if normalized in pairs:
            raise WireError("bad_mapping", f"{where} symbols must be unique")
        pairs[normalized] = encode_float(number, f"{where}[{symbol!r}]")
    return [[symbol, pairs[symbol]] for symbol in sorted(pairs)]


def decode_pairs(value: Any, where: str) -> dict[str, float]:
    result: dict[str, float] = {}
    previous: str | None = None
    for item in expect(value, list, where):
        if type(item) is not list or len(item) != 2 or type(item[0]) is not str or not item[0]:
            raise WireError("bad_mapping", f"{where} entries must be [symbol, value] pairs")
        if previous is not None and item[0] <= previous:
            raise WireError("mapping_order", f"{where} symbols must be sorted and unique")
        previous = item[0]
        result[item[0]] = decode_float(item[1], f"{where}[{item[0]!r}]")
    return result
