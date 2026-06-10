from __future__ import annotations

from enum import StrEnum


class Stage(StrEnum):
    IDEA = "idea"
    BACKTESTED = "backtested"
    CANDIDATE = "candidate"
    PAPER = "paper"
    LIVE = "live"
    RETIRED = "retired"


class Actor(StrEnum):
    HUMAN = "human"
    AGENT = "agent"
    SYSTEM = "system"


# Forward/back edges between live stages. The retire edge is *derived* below so adding a
# stage cannot silently forget it: every non-retired stage gains `-> RETIRED` automatically,
# and RETIRED is terminal.
_LIVE_TRANSITIONS: dict[Stage, set[Stage]] = {
    Stage.IDEA: {Stage.BACKTESTED},
    Stage.BACKTESTED: {Stage.CANDIDATE, Stage.IDEA},
    Stage.CANDIDATE: {Stage.PAPER, Stage.BACKTESTED},
    Stage.PAPER: {Stage.LIVE, Stage.CANDIDATE},
    Stage.LIVE: {Stage.PAPER},
}

ALLOWED_TRANSITIONS: dict[Stage, set[Stage]] = {
    stage: (_LIVE_TRANSITIONS.get(stage, set()) | {Stage.RETIRED})
    if stage is not Stage.RETIRED
    else set()
    for stage in Stage
}


class TransitionError(ValueError):
    pass


def can_transition(frm: Stage, to: Stage) -> bool:
    return to in ALLOWED_TRANSITIONS[frm]


def validate_transition(frm: Stage, to: Stage) -> None:
    if not can_transition(frm, to):
        raise TransitionError(f"illegal transition {frm.value} -> {to.value}")
