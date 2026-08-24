"""Unit pins for backend.params — the argv hygiene layer (slice C)."""

from __future__ import annotations

import pytest
from backend.params import (
    MAX_SERIES_RUN_IDS,
    InvalidParam,
    validate_action,
    validate_actor,
    validate_family,
    validate_lane,
    validate_run_ids,
    validate_run_kind,
    validate_since,
    validate_sort,
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


# --- run kind ---


@pytest.mark.parametrize(
    "kind", ["backtest", "walk_forward", "sweep", "sweep_trial", "gate"]
)
def test_valid_run_kinds(kind: str) -> None:
    assert validate_run_kind(kind) == kind


@pytest.mark.parametrize("kind", ["backtests", "BACKTEST", "", "-gate", "gate;drop"])
def test_bad_run_kinds_raise(kind: str) -> None:
    with pytest.raises(InvalidParam):
        validate_run_kind(kind)


# --- family (syntactic guard only) ---


@pytest.mark.parametrize("family", ["trend", "trend-1", "a.b_c", "x" * 64])
def test_valid_families_pass_through(family: str) -> None:
    assert validate_family(family) == family


@pytest.mark.parametrize(
    "family",
    [
        "-trend",  # leading dash (flag injection)
        "--allow-non-pit",  # flag injection
        "",  # empty
        "x" * 65,  # too long
        "a/b",  # slash
        "a b",  # space
        "a\nb",  # newline
    ],
)
def test_bad_families_raise(family: str) -> None:
    with pytest.raises(InvalidParam):
        validate_family(family)


# --- sort (syntactic guard only — NOT the METRIC_COLUMNS vocabulary check) ---


@pytest.mark.parametrize("sort", ["sharpe_oos", "mean_window_sharpe", "a.b-c_9", "x" * 64])
def test_valid_sorts_pass_through(sort: str) -> None:
    assert validate_sort(sort) == sort


@pytest.mark.parametrize(
    "sort",
    [
        "-sharpe_oos",  # leading dash (flag injection)
        "--limit=1",  # flag injection
        "",  # empty
        "x" * 65,  # too long
        "a b",  # space
        "a;drop",  # shell metachar
    ],
)
def test_bad_sorts_raise(sort: str) -> None:
    with pytest.raises(InvalidParam):
        validate_sort(sort)


def test_sort_does_not_check_metric_vocabulary() -> None:
    """`validate_sort` is deliberately syntactic-only — a made-up-but-well-shaped metric name
    passes here and is rejected downstream by the store's METRIC_COLUMNS allow-list instead."""
    assert validate_sort("not_a_real_metric") == "not_a_real_metric"


# --- run ids ---


def test_valid_run_ids_are_sorted_regardless_of_input_order() -> None:
    """FIX 2 (final review wave): ids are returned SORTED, not in input order — `algua_cli._cache`
    /`_locks` key on the exact argv tuple, so "3,1,2" and "1,2,3" must resolve to the same argv."""
    assert validate_run_ids("3,1,2") == [1, 2, 3]


def test_valid_single_run_id() -> None:
    assert validate_run_ids("7") == [7]


def test_run_ids_deduped_and_sorted() -> None:
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


@pytest.mark.parametrize("ids", ["0", "-1", "1,-2", "1,0,2"])
def test_run_ids_below_one_raise(ids: str) -> None:
    """FIX 2 (final review wave): a run id must be >= 1. A negative id would also become a
    leading bare argv token for `runs_series`'s positional first id (the hazard
    `validate_strategy_name`'s no-leading-dash rule exists to prevent)."""
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
