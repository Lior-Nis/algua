"""Canonical JSON encoding and the frozen-wire identity.

The one implementation shared by the registry (artifact digests, manifests, recorded configs) and
the frozen planner child, which may not import ``algua.registry``. Stdlib only and free of I/O.
Story 1.3b's golden digest vectors pin these bytes: any change here re-identifies every recorded
frozen artifact.
"""
from __future__ import annotations

import json
import math
import unicodedata
from typing import Any

FROZEN_WIRE_NAME = "frozen-planner"
FROZEN_WIRE_VERSION = 1
FROZEN_WIRE = {"name": FROZEN_WIRE_NAME, "version": FROZEN_WIRE_VERSION}


def _utf8_nfc(value: str, label: str) -> str:
    """NFC form of ``value``; a lone surrogate is not UTF-8 text and fails as a plain ValueError."""
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        raise ValueError(f"{label} must be valid UTF-8 text") from None
    return unicodedata.normalize("NFC", value)


def _normalized(value: Any) -> Any:
    if isinstance(value, str):
        return _utf8_nfc(value, "canonical JSON text")
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical JSON numbers must be finite")
        return value
    if isinstance(value, (list, tuple)):
        return [_normalized(item) for item in value]
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("canonical JSON object keys must be strings")
            normalized = _utf8_nfc(key, "canonical JSON text")
            if normalized in result:
                raise ValueError("canonical JSON keys collide after normalization")
            result[normalized] = _normalized(item)
        return result
    raise ValueError(f"unsupported canonical JSON value: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    return json.dumps(
        _normalized(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    )
