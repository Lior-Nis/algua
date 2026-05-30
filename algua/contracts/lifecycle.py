from __future__ import annotations

from enum import StrEnum


class Stage(StrEnum):
    IDEA = "idea"
    BACKTESTED = "backtested"
    SHORTLISTED = "shortlisted"
    PAPER = "paper"
    LIVE = "live"
    RETIRED = "retired"


class Actor(StrEnum):
    HUMAN = "human"
    AGENT = "agent"
    SYSTEM = "system"


ALLOWED_TRANSITIONS: dict[Stage, set[Stage]] = {
    Stage.IDEA: {Stage.BACKTESTED, Stage.RETIRED},
    Stage.BACKTESTED: {Stage.SHORTLISTED, Stage.IDEA, Stage.RETIRED},
    Stage.SHORTLISTED: {Stage.PAPER, Stage.BACKTESTED, Stage.RETIRED},
    Stage.PAPER: {Stage.LIVE, Stage.SHORTLISTED, Stage.RETIRED},
    Stage.LIVE: {Stage.PAPER, Stage.RETIRED},
    Stage.RETIRED: set(),
}


class TransitionError(ValueError):
    pass


def can_transition(frm: Stage, to: Stage) -> bool:
    return to in ALLOWED_TRANSITIONS[frm]


def validate_transition(frm: Stage, to: Stage) -> None:
    if not can_transition(frm, to):
        raise TransitionError(f"illegal transition {frm.value} -> {to.value}")
