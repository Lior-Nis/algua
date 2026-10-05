"""Paper and live must enforce the SAME safety invariants.

Why this file exists: `paper` is the rehearsal for `live`. If an invariant holds in one lane and not
the other, paper-testing stops predicting live behaviour — and the failure is silent, because
nothing breaks when a fix reaches one lane only.

That is not hypothetical. #559 bound the paper tick to the GATED universe (the symbol set the
strategy's promotion evidence actually covered) and the same fix was never applied to live, so for a
period the real-money lane traded a raw CONFIG list while the rehearsal lane failed closed — the
safety ordering exactly inverted (#601). It survived because the two lane orchestrators are only
~38% structurally similar, so a fix to one has no mechanical reason to reach the other.

These tests are deliberately STRUCTURAL (they read the tick sources) rather than behavioural. A
behavioural test proves the invariant holds on the path it exercises; these prove neither lane can
QUIETLY DROP it. Both kinds are worth having — the behavioural ones live in test_cli_live.py and
test_cli_paper.py.

Book exits (Story 2.2, T16 at the end of this file): for a while only the live lane drained a
leaving strategy's resting orders, while paper selected no guard at all (#685). Three facts now tie
the edge set, the selector and the single production path into the store together:
_REVOKE_ON_EXIT is exactly the set of allowed edges that leave a lane's operating stages;
select_exit_guard answers every one with the SOURCE lane's guard; and only transitions.py
calls .apply_transition( while no module injects a selector, so every production exit reaches
the store through the default selection.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

from algua.contracts.lifecycle import ALLOWED_TRANSITIONS, Stage
from algua.execution import lane_exit
from algua.execution.lane_exit import LiveExitGuard, select_exit_guard
from algua.execution.paper_exit_drain import PaperExitGuard
from algua.registry import transitions
from algua.registry.db import connect, migrate
from algua.registry.store import SqliteStrategyRepository
from algua.registry.transitions import _REVOKE_ON_EXIT

REPO = pathlib.Path(__file__).resolve().parents[1]
LANES = {
    "live": (REPO / "algua/cli/live_cmd.py", "_run_strategy_tick"),
    "paper": (REPO / "algua/cli/paper_cmd.py", "_run_paper_strategy_tick"),
}


def _tick_source(lane: str) -> str:
    path, func = LANES[lane]
    tree = ast.parse(path.read_text())
    node = next(
        (n for n in ast.walk(tree)
         if isinstance(n, ast.FunctionDef) and n.name == func), None
    )
    assert node is not None, (
        f"{lane}: {func} not found in {path.name} — if the tick was renamed or moved, update "
        f"LANES rather than deleting this test; a parity test that silently stops finding its "
        f"subject is worse than no test."
    )
    return ast.get_source_segment(path.read_text(), node) or ""


def _calls(lane: str) -> set[str]:
    """Every function name called anywhere in the lane's tick body."""
    src = _tick_source(lane)
    tree = ast.parse("if 1:\n" + "\n".join("    " + ln for ln in src.splitlines()))
    out: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Name):
                out.add(f.id)
            elif isinstance(f, ast.Attribute):
                out.add(f.attr)
    return out


def test_both_lanes_bind_to_the_gated_universe():
    """Neither lane may trade its raw CONFIG universe (#559 in paper, #601 in live).

    A strategy is promoted on evidence gathered over the gate's universe. Trading a config list
    that has since drifted means trading symbols the promotion evidence never covered.
    """
    paper_setup = (REPO / "algua/registry/paper_runtime.py").read_text()
    missing = [lane for lane in LANES if (
        "resolve_operational_universe" not in (_tick_source(lane) + paper_setup)
    )]
    assert not missing, (
        f"these lanes do NOT bind to the gated universe: {missing}. This is the #601 defect "
        f"recurring: the lane trades whatever its CONFIG says, not what its promotion evidence "
        f"covered. Route the tick through `resolve_operational_universe` as the other lane does."
    )


def test_both_lanes_trip_and_halt_on_a_risk_breach():
    """Both lanes must trip the kill-switch AND engage the global halt on a RiskBreach.

    The dark-feed branch (stale/unvaluable marks) halts the whole account and PRESERVES positions —
    flattening blind off a dead feed dumps the book at unknown prices. A lane that trips without
    halting, or halts without tripping, has a different safety policy from its twin.
    """
    for required in ("trip_for_breach", "engage"):
        missing = [lane for lane in LANES if required not in _calls(lane)]
        assert not missing, (
            f"lanes missing `{required}` in their breach path: {missing}. Both lanes must route "
            f"a RiskBreach identically; a one-lane-only change to breach handling means the "
            f"rehearsal no longer predicts production."
        )


def test_neither_lane_defines_its_own_dark_feed_kind_set():
    """The dark-feed kind set must exist exactly ONCE, in `algua.risk.limits`.

    Originally both lanes wrote the literal `{"stale_marks", "unvaluable_marks"}` inline, and this
    test asserted the two literals matched. That only caught divergence AFTER someone edited one —
    so stage 8 moved the set to `DARK_FEED_KINDS` beside the exception that carries `.kind`, and
    both lanes now route on `exc.is_dark_feed`.

    This assertion is the stronger form: a lane cannot disagree about the policy because a lane no
    longer states the policy. If one re-introduces a local set, the same market condition would
    liquidate the book in one lane and preserve it in the other.
    """
    from algua.risk.limits import DARK_FEED_KINDS, RiskBreach

    assert DARK_FEED_KINDS == frozenset({"stale_marks", "unvaluable_marks"})
    assert RiskBreach("stale_marks", "d").is_dark_feed is True
    assert RiskBreach("unvaluable_marks", "d").is_dark_feed is True
    assert RiskBreach("drawdown", "d").is_dark_feed is False, (
        "an economic breach must NOT be treated as a dark feed — it trips and flattens"
    )

    offenders = []
    for lane in LANES:
        src = _tick_source(lane)
        for literal in ('"stale_marks"', "'stale_marks'"):
            if literal in src:
                offenders.append(lane)
                break
    assert not offenders, (
        f"these lanes hardcode a dark-feed kind instead of using `exc.is_dark_feed`: {offenders}. "
        f"The policy lives in `algua.risk.limits.DARK_FEED_KINDS` so the two lanes cannot "
        f"drift apart on whether a dead bar feed preserves the book or liquidates it."
    )


def test_both_lanes_route_dark_feed_through_the_shared_predicate():
    """Both ticks must consult `is_dark_feed` — the branch must still EXIST in each lane.

    Complements the test above: that one proves no lane defines the policy, this one proves no lane
    silently DROPPED the branch (which would send a dark feed down the trip-and-flatten path).
    """
    missing = [lane for lane in LANES if "is_dark_feed" not in _tick_source(lane)]
    assert not missing, (
        f"these lanes no longer branch on `is_dark_feed`: {missing}. A dark bar feed would fall "
        f"through to trip-and-flatten and dump the book at unknown prices (#452 HIGH#3)."
    )


# --- T16 (Story 2.2 §2.1, §2.3): both lanes' book exits carry their lane's drain ------------------

PAPER_LANE = frozenset({Stage.PAPER, Stage.FORWARD_TESTED})
LIVE_LANE = frozenset({Stage.LIVE})


def _lane_exits() -> set[tuple[Stage, Stage]]:
    return {
        (source, target)
        for source, targets in ALLOWED_TRANSITIONS.items()
        for target in targets
        for lane in (PAPER_LANE, LIVE_LANE)
        if source in lane and target not in lane
    }


def test_revoke_on_exit_is_exactly_the_set_of_lane_exits():
    """A new lifecycle edge out of a lane cannot be added without its allocation shed and drain."""
    assert _REVOKE_ON_EXIT == _lane_exits()


@pytest.mark.parametrize(("source", "target"), sorted(_REVOKE_ON_EXIT))
def test_the_selector_answers_every_lane_exit_with_the_source_lanes_guard(
        tmp_path, empty_exit_venues, source, target):
    conn = connect(tmp_path / "reg.db")
    migrate(conn)
    repo = SqliteStrategyRepository(conn)
    repo.add(name="s1")

    guard = select_exit_guard(repo, "s1", source, target)

    assert type(guard) is (LiveExitGuard if source in LIVE_LANE else PaperExitGuard)


def test_transition_strategy_selects_through_the_lane_exit_selector():
    assert transitions._default_exit_guard_selector() is lane_exit.select_exit_guard


def _algua_calls(predicate) -> list[str]:
    found = []
    for path in sorted((REPO / "algua").rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and predicate(node):
                found.append(str(path.relative_to(REPO)))
    return found


def test_only_transitions_reaches_the_store_transition_and_none_injects_a_selector():
    assert _algua_calls(lambda c: isinstance(c.func, ast.Attribute)
                        and c.func.attr == "apply_transition") == ["algua/registry/transitions.py"]
    assert _algua_calls(lambda c: any(k.arg == "exit_guard_selector" for k in c.keywords)) == []
