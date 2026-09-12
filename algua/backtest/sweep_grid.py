"""`--param KEY=v1,v2` grid parsing for `backtest sweep` (carved from backtest/sweep.py, overlays
PR). Pure string -> grid; the strategy-dependent half (`validate_sweep_grid`) stays in sweep.py."""
from __future__ import annotations

import math
from typing import Any


def _coerce(value: str) -> Any:
    """Coerce a grid value string to int, then float, else leave as str.

    Non-finite floats ('inf'/'nan'/'-inf'/'1e400') are rejected here with a clear message
    rather than coerced and carried downstream, where they only surface as an opaque
    JSON-serialization failure in config_hash (json.dumps(allow_nan=False)) (#258).
    """
    try:
        return int(value)
    except ValueError:
        pass
    try:
        f = float(value)
    except ValueError:
        return value
    if not math.isfinite(f):
        raise ValueError(f"non-finite grid value: {value!r}")
    return f


def _coerce_values(values: list[Any]) -> list[Any]:
    """Widen a homogeneous-numeric value list to float if any element is float.

    Prevents silent type mixing when a grid mixes e.g. "10,10.5" → [int(10), float(10.5)].
    Any list that already contains a non-numeric value is returned unchanged.
    """
    has_float = any(type(v) is float for v in values)
    if has_float and all(isinstance(v, (int, float)) for v in values):
        return [float(v) for v in values]
    return list(values)


def parse_grid(params: list[str]) -> dict[str, list[Any]]:
    """Parse repeatable `KEY=v1,v2,...` flags into a grid dict. Values coerced int->float->str."""
    if not params:
        raise ValueError("provide at least one --param KEY=v1,v2,...")
    grid: dict[str, list[Any]] = {}
    for item in params:
        if "=" not in item:
            raise ValueError(f"malformed --param {item!r}: expected KEY=v1,v2,...")
        key, _, raw = item.partition("=")
        key = key.strip()
        values = [v.strip() for v in raw.split(",") if v.strip() != ""]
        if not key or not values:
            raise ValueError(f"malformed --param {item!r}: empty key or value list")
        if key in grid:
            raise ValueError(f"duplicate --param key {key!r}: specify each key only once")
        grid[key] = _coerce_values([_coerce(v) for v in values])
    return grid
