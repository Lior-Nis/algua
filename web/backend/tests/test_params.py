"""Unit pins for backend.params — the argv hygiene layer (slice C)."""

from __future__ import annotations

import pytest
from backend.params import (
    InvalidParam,
    validate_action,
    validate_actor,
    validate_lane,
    validate_since,
    validate_strategy_name,
)

# --- strategy name ---


@pytest.mark.parametrize("name", ["momo", "a", "A.b_c-d9", "x" * 64, "mo.mo-1"])
def test_valid_strategy_names_pass_through(name: str) -> None:
    assert validate_strategy_name(name) == name


@pytest.mark.parametrize(
    "name",
    [
        "-momo",  # leading dash (flag injection)
        "--allow-non-pit",  # flag injection
        "mo/mo",  # slash
        "mo mo",  # space
        "mo=mo",  # equals
        "",  # empty
        "x" * 65,  # too long
        "mo;mo",  # shell metachar
        "mo\nmo",  # newline
    ],
)
def test_bad_strategy_names_raise(name: str) -> None:
    with pytest.raises(InvalidParam):
        validate_strategy_name(name)


# --- actor ---


@pytest.mark.parametrize("actor", ["agent", "human", "system"])
def test_valid_actors(actor: str) -> None:
    assert validate_actor(actor) == actor


@pytest.mark.parametrize("actor", ["root", "AGENT", "", "agent "])
def test_bad_actors_raise(actor: str) -> None:
    with pytest.raises(InvalidParam):
        validate_actor(actor)


# --- action ---


@pytest.mark.parametrize("action", ["transition", "kill_switch:trip", "a.b-c_9", "x" * 64])
def test_valid_actions(action: str) -> None:
    assert validate_action(action) == action


@pytest.mark.parametrize("action", ["a b", "a/b", "", "x" * 65, "a=b", "a\nb"])
def test_bad_actions_raise(action: str) -> None:
    with pytest.raises(InvalidParam):
        validate_action(action)


# --- since ---


@pytest.mark.parametrize(
    "since",
    ["2026-08-01", "2026-08-01T00:00:00", "2026-08-01T00:00:00+00:00", "2026-08-01 12:30:00"],
)
def test_valid_since(since: str) -> None:
    assert validate_since(since) == since


@pytest.mark.parametrize("since", ["not-a-date", "yesterday", "", "2026-13-01", "--since"])
def test_bad_since_raises(since: str) -> None:
    with pytest.raises(InvalidParam):
        validate_since(since)


# --- lane ---


@pytest.mark.parametrize("lane", ["paper", "live"])
def test_valid_lanes(lane: str) -> None:
    assert validate_lane(lane) == lane


@pytest.mark.parametrize("lane", ["margin", "PAPER", "", "paper "])
def test_bad_lanes_raise(lane: str) -> None:
    with pytest.raises(InvalidParam):
        validate_lane(lane)
