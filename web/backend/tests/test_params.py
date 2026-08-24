"""Unit pins for backend.params — the argv hygiene layer (slice C)."""

from __future__ import annotations

import pytest
from backend.params import (
    MAX_SERIES_RUN_IDS,
    InvalidParam,
    validate_action,
    validate_actor,
    validate_lane,
    validate_run_ids,
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


# --- run ids ---


def test_valid_run_ids_parsed_in_order() -> None:
    assert validate_run_ids("3,1,2") == [3, 1, 2]


def test_valid_single_run_id() -> None:
    assert validate_run_ids("7") == [7]


def test_run_ids_deduped_order_preserving() -> None:
    assert validate_run_ids("1,2,1,3,2") == [1, 2, 3]


def test_run_ids_tolerates_surrounding_whitespace() -> None:
    assert validate_run_ids(" 1 , 2 ") == [1, 2]


def test_run_ids_at_cap_passes() -> None:
    ids = ",".join(str(i) for i in range(1, MAX_SERIES_RUN_IDS + 1))
    assert validate_run_ids(ids) == list(range(1, MAX_SERIES_RUN_IDS + 1))


@pytest.mark.parametrize(
    "ids",
    [
        "",  # empty
        "1,,2",  # empty element
        "1,abc",  # non-integer element
        "1, 2.5",  # float, not int
        ",",  # only a separator
    ],
)
def test_malformed_run_ids_raise(ids: str) -> None:
    with pytest.raises(InvalidParam):
        validate_run_ids(ids)


def test_run_ids_over_cap_raises() -> None:
    ids = ",".join(str(i) for i in range(1, MAX_SERIES_RUN_IDS + 2))  # one over the cap
    with pytest.raises(InvalidParam):
        validate_run_ids(ids)


def test_run_ids_over_cap_after_dedup_still_raises() -> None:
    # Duplicates that would dedup UNDER the cap must not sneak an oversized raw list through:
    # the cap is enforced on the de-duplicated set, but a caller sending 20 distinct ids is
    # still over cap even though the CLI itself would dedup first too.
    ids = ",".join(str(i) for i in range(1, MAX_SERIES_RUN_IDS + 3))
    with pytest.raises(InvalidParam):
        validate_run_ids(ids)
