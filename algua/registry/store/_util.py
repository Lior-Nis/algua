"""Shared plain helpers for the registry/store package — no state, no self, safe to import
from any mixin file with zero cross-mixin coupling."""
from __future__ import annotations

from datetime import UTC, datetime


def _now() -> str:
    return datetime.now(UTC).isoformat()
