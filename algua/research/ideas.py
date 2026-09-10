# algua/research/ideas.py
from __future__ import annotations

from collections.abc import Collection

from algua.contracts.idea import DataCapability, Horizon, IdeaStatus, Market


def classify_status(
    required_data: Collection[DataCapability],
    supported: Collection[DataCapability],
) -> IdeaStatus:
    """OPEN when every required capability is platform-supported; else NEEDS_DATA (parked).
    An idea requiring no data is trivially OPEN. Note: OPEN means "implementable by the
    platform", NOT "a covering snapshot exists" — real-data/PIT readiness stays the promotion
    gate's job. Thin wrapper over `classify_idea` (data-only eligibility)."""
    return classify_idea(
        list(required_data), frozenset(supported), market=None, supported_markets=frozenset(),
        horizon=None, supported_horizons=frozenset(),
    )[0]


def classify_idea(
    caps: list[DataCapability], supported: frozenset[DataCapability], *,
    market: Market | None, supported_markets: frozenset[Market],
    horizon: Horizon | None, supported_horizons: frozenset[Horizon],
) -> tuple[IdeaStatus, str | None]:
    """Eligibility predicate shared by add / import / claim / reclassify (spec §6).

    OPEN iff every required capability, the market and the horizon are supported; otherwise
    NEEDS_DATA with a `;`-joined reason naming every gap (`data:<cap>`, `market:<m>`,
    `horizon:<h>`) so a later capability flip can re-open exactly the right rows. A None
    market/horizon (legacy rows) counts as supported."""
    gaps = [f"data:{c.value}" for c in caps if c not in supported]
    if market is not None and market not in supported_markets:
        gaps.append(f"market:{market.value}")
    if horizon is not None and horizon not in supported_horizons:
        gaps.append(f"horizon:{horizon.value}")
    if gaps:
        return IdeaStatus.NEEDS_DATA, ";".join(gaps)
    return IdeaStatus.OPEN, None
