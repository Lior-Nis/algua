from __future__ import annotations

from enum import StrEnum


class Author(StrEnum):
    """Who authored a strategy — the agent (default) or a human operator."""

    AGENT = "agent"
    HUMAN = "human"


class HypothesisStatus(StrEnum):
    """Research status of a strategy's claimed edge."""

    UNTESTED = "untested"
    SUPPORTED = "supported"
    REFUTED = "refuted"
    INCONCLUSIVE = "inconclusive"
