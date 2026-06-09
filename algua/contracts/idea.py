# algua/contracts/idea.py
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class IdeaStatus(StrEnum):
    """Lifecycle of a sourced idea before/at the registry's `idea` stage."""

    OPEN = "open"              # testable now (needs only platform-supported data)
    NEEDS_DATA = "needs_data"  # parked: needs a capability the platform can't provide yet
    AUTHORED = "authored"      # promoted into a registered strategy
    REFUTED = "refuted"        # rejected; a dedup sentinel
    DISCARDED = "discarded"    # dropped without testing


class SourceType(StrEnum):
    """Where a sourced idea came from (provenance)."""

    PAPER = "paper"
    URL = "url"
    FORUM = "forum"
    FILING = "filing"
    THESIS = "thesis"
    MANUAL = "manual"


class DataCapability(StrEnum):
    """Controlled vocabulary of strategy-input data kinds an idea may require. Only OHLCV is
    platform-supported today (see ``algua.data.capabilities``); the rest park ideas as
    ``needs_data``. A single vocabulary stops 13f / form_13f / filings_13f fragmentation."""

    OHLCV = "ohlcv"
    FUNDAMENTALS = "fundamentals"
    FORM_13F = "form_13f"
    OPTIONS_FLOW = "options_flow"
    DARK_POOL = "dark_pool"
    FORM_4 = "form_4"


# Allowed `set-status` moves. open<->needs_data on a capability re-check; open/needs_data advance
# to authored or discarded; an authored idea can only be refuted (its strategy failed) or
# discarded; refuted/discarded are terminal. A no-op (X -> X) is never a legal change.
ALLOWED_IDEA_TRANSITIONS: dict[IdeaStatus, set[IdeaStatus]] = {
    IdeaStatus.OPEN: {IdeaStatus.NEEDS_DATA, IdeaStatus.AUTHORED, IdeaStatus.DISCARDED},
    IdeaStatus.NEEDS_DATA: {IdeaStatus.OPEN, IdeaStatus.AUTHORED, IdeaStatus.DISCARDED},
    IdeaStatus.AUTHORED: {IdeaStatus.REFUTED, IdeaStatus.DISCARDED},
    IdeaStatus.REFUTED: set(),
    IdeaStatus.DISCARDED: set(),
}


def can_change_status(frm: IdeaStatus, to: IdeaStatus) -> bool:
    return to in ALLOWED_IDEA_TRANSITIONS[frm]


@dataclass
class Idea:
    id: int
    title: str
    hypothesis: str
    family: str | None
    tags: list[str]
    source_type: SourceType
    source_ref: str | None
    source_date: str | None
    source_note: str | None
    required_data: list[DataCapability]
    status: IdeaStatus
    signature: str
    authored_strategy_id: int | None
    duplicate_of_idea_id: int | None
    override_reason: str | None
    created_at: str
    updated_at: str
