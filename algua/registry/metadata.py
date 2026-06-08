from __future__ import annotations

import json
from collections.abc import Iterable


def canonicalize_tags(tags: Iterable[str]) -> list[str]:
    """Trim, lowercase, drop empties, dedupe, and sort tags into canonical order."""
    seen: set[str] = set()
    for raw in tags:
        tag = raw.strip().lower()
        if tag:
            seen.add(tag)
    return sorted(seen)


def dump_tags(tags: Iterable[str]) -> str:
    """Serialize tags to the canonical JSON-array string stored in the registry."""
    return json.dumps(canonicalize_tags(tags))


def load_tags(value: str | None) -> list[str]:
    """Parse a stored tags column back to a list; tolerate NULL/invalid JSON as []."""
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except (ValueError, TypeError):
        return []
    if not isinstance(parsed, list):
        return []
    return canonicalize_tags(str(t) for t in parsed)
