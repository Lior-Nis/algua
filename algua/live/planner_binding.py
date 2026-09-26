"""Canonical logical Phase A binding for the in-process planner contract."""

from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from collections.abc import Mapping
from datetime import datetime
from typing import Any

import pandas as pd

from algua.live.planner_contract import EarlyPlannerInput

_BAR_COLUMNS = ("symbol", "open", "high", "low", "close", "adj_close", "volume")


def _text(value: str) -> str:
    return unicodedata.normalize("NFC", value)


def _datetime_ns(value: datetime) -> str:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise ValueError("planner datetime must be timezone-aware")
    return str(stamp.tz_convert("UTC").value)


def _float_token(value: float) -> str:
    if type(value) not in (int, float):
        raise ValueError(f"planner float must be numeric, got {type(value).__name__}")
    number = float(value)
    if math.isnan(number):
        return "nan"
    if math.isinf(number):
        return "+inf" if number > 0 else "-inf"
    if number == 0.0:
        number = 0.0
    return number.hex()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def bars_digest(frame: pd.DataFrame) -> str:
    """Bind the canonical bar shape while retaining the exact captured row order."""
    if frame.index.name != "timestamp":
        raise ValueError("raw_bars index must be named 'timestamp'")
    if tuple(frame.columns) != _BAR_COLUMNS:
        raise ValueError(f"raw_bars columns must be exactly {list(_BAR_COLUMNS)!r}")
    if not isinstance(frame.index, pd.DatetimeIndex) or str(frame.index.tz) != "UTC":
        raise ValueError("raw_bars index must be a UTC DatetimeIndex")
    normalized_symbols = frame["symbol"].map(
        lambda value: _text(value) if isinstance(value, str) else value
    )
    keyed = frame.assign(symbol=normalized_symbols).reset_index()
    if keyed.duplicated(subset=["timestamp", "symbol"]).any():
        raise ValueError("raw_bars contains duplicate (timestamp, symbol) rows")
    rows = []
    for row in frame.itertuples(index=True, name=None):
        ts, symbol, *numbers = row
        if not isinstance(symbol, str) or not symbol:
            raise ValueError("raw_bars symbols must be non-empty strings")
        rows.append([_datetime_ns(ts), _text(symbol), *(_float_token(v) for v in numbers)])
    return _digest({"index_name": "timestamp", "columns": list(_BAR_COLUMNS), "rows": rows})


def resolved_config_digest(value: str) -> str:
    """Verify canonical object JSON and hash its exact UTF-8 bytes."""
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("resolved_config_json must be valid canonical JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError("resolved_config_json must contain a JSON object")
    canonical = _canonical(parsed).decode("utf-8")
    if value != canonical:
        raise ValueError("resolved_config_json must use canonical sorted compact JSON")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _mapping(value: Mapping[str, float]) -> list[list[str]]:
    items: list[list[str]] = []
    for key, number in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError("planner mapping keys must be non-empty strings")
        normalized = _text(key)
        if any(item[0] == normalized for item in items):
            raise ValueError("planner mapping keys collide after Unicode normalization")
        items.append([normalized, _float_token(number)])
    return sorted(items, key=lambda item: item[0].encode("utf-8"))


def phase_a_binding(
    early: EarlyPlannerInput, *, decision_ts: datetime | None, warming: bool
) -> str:
    """Return the normative integrity-only binding for a snapshot-required outcome."""
    root = {
        "domain": "algua.phase-a-binding",
        "version": "1",
        "early": {
            "request_id": _text(early.request_id),
            "strategy_name": _text(early.strategy_name),
            "deployment_id": (None if early.deployment_id is None else str(early.deployment_id)),
            "artifact_id": None if early.artifact_id is None else str(early.artifact_id),
            "manifest_digest": (
                None if early.manifest_digest is None else _text(early.manifest_digest)
            ),
            "config_hash": _text(early.config_hash),
            "resolved_config_sha256": resolved_config_digest(early.resolved_config_json),
            "now_ns": _datetime_ns(early.now),
            "timeframe": _text(early.timeframe),
            "calendar_code": _text(early.calendar_code),
            "bars_sha256": bars_digest(early.raw_bars),
            "early_positions": _mapping(early.early_positions),
            "gate_universe": [_text(symbol) for symbol in early.gate_universe],
            "max_drawdown": (
                None if early.max_drawdown is None else _float_token(early.max_drawdown)
            ),
        },
        "outcome": {
            "kind": "snapshot_required",
            "decision_ts_ns": None if decision_ts is None else _datetime_ns(decision_ts),
            "warming": warming,
        },
    }
    return hashlib.sha256(_canonical(root)).hexdigest()
