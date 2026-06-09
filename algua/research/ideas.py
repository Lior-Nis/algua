# algua/research/ideas.py
from __future__ import annotations

from collections.abc import Collection

from algua.contracts.idea import DataCapability, IdeaStatus


def classify_status(
    required_data: Collection[DataCapability],
    supported: Collection[DataCapability],
) -> IdeaStatus:
    """OPEN when every required capability is platform-supported; else NEEDS_DATA (parked).
    An idea requiring no data is trivially OPEN. Note: OPEN means "implementable by the
    platform", NOT "a covering snapshot exists" — real-data/PIT readiness stays the promotion
    gate's job."""
    return IdeaStatus.OPEN if set(required_data) <= set(supported) else IdeaStatus.NEEDS_DATA
