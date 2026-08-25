"""Parse a driver subprocess's stdout into a JSON envelope, and classify a failed run.

A driver's single ``emit()`` call round-trips as one ``json.dumps(..., indent=2)`` document, so
the operator wrapper (:mod:`algua.cli.operator_cmd`) parses the FULL stdout in the common case,
falling back to the last balanced top-level ``{...}`` object if the driver interleaved extra
output. This module holds that untrusted-stdout parsing plus the best-effort failure-label
classifier — pure text/dict logic, no subprocess or filesystem I/O of its own.
"""

from __future__ import annotations

import json

__all__ = ["last_top_level_object", "parse_driver_payload", "classify_failure"]


def last_top_level_object(text: str) -> str | None:
    """Locate the last balanced top-level ``{...}`` in ``text`` via brace-depth counting.

    Scans from the END: finds the final ``}``, then walks backwards tracking brace depth (ignoring
    braces inside JSON string literals) until depth returns to zero, yielding the matching ``{``.
    Returns the substring, or ``None`` if no balanced object is found.
    """
    end = text.rfind("}")
    if end == -1:
        return None
    depth = 0
    in_string = False
    i = end
    while i >= 0:
        ch = text[i]
        if in_string:
            if ch == '"':
                backslashes = 0
                j = i - 1
                while j >= 0 and text[j] == "\\":
                    backslashes += 1
                    j -= 1
                if backslashes % 2 == 0:
                    in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "}":
            depth += 1
        elif ch == "{":
            depth -= 1
            if depth == 0:
                return text[i : end + 1]
        i -= 1
    return None


def parse_driver_payload(stdout: str) -> dict | None:
    """Best-effort recover the driver's JSON envelope from its stdout, or ``None`` if none parses.

    A driver's single ``emit()`` call round-trips as one ``json.dumps(..., indent=2)`` document, so
    the FULL stdout parses cleanly in the common case. If the driver interleaved extra output, fall
    back to the last balanced top-level ``{...}``. Returns ``None`` (NOT ``{}``) when nothing parses
    to a dict, so the caller can tell "the driver did not emit parseable JSON" (a completion it
    cannot confirm) from "the driver emitted a valid but non-deferred envelope".
    """
    text = stdout.strip()
    if not text:
        return None
    for candidate in (text, last_top_level_object(text)):
        if candidate is None:
            continue
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def classify_failure(payload: dict | None) -> str:
    """Best-effort alert label for an rc!=0 outcome (never load-bearing — the alert fires and
    carries rc+stdout_head regardless). ``halted`` → global_halt, ``ok:false`` → breach, else
    job_failed (also the parse-failure fallback)."""
    if payload is not None and payload.get("halted"):
        return "global_halt"
    if payload is not None and payload.get("ok") is False:
        return "breach"
    return "job_failed"
