"""run_views: shaping, allow-listing, and the payload-size contract."""
from __future__ import annotations

import json
import sqlite3

import pandas as pd
import pytest

from algua.registry.db.migrate import migrate
from algua.registry.gate_history import GATE_DECISION_ALLOWLIST
from algua.registry.run_views import (
    run_detail_payload,
    run_list_payload,
    run_series_payload,
)
from algua.registry.store import SqliteStrategyRepository


@pytest.fixture()
def repo() -> SqliteStrategyRepository:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    migrate(conn)
    return SqliteStrategyRepository(conn)


def test_list_payload_never_contains_a_series(repo: SqliteStrategyRepository) -> None:
    """The payload-size contract: `runs list` returns scalars only."""
    repo.record_run("backtest", "alpha", metrics={"sharpe_is": 1.0})
    payload = run_list_payload(repo, kind=None, strategy=None, family=None, sort=None, limit=10)
    (row,) = payload["runs"]
    # Asserted structurally, not by substring: a repr() grep passes on the wrong thing the moment
    # a column is renamed. series_backtest_id is an INT POINTER and is allowed here; a series
    # PAYLOAD is not.
    assert "returns" not in row
    assert "returns_json" not in row
    assert isinstance(row.get("series_backtest_id"), (int, type(None)))
    for key, value in row.items():
        if isinstance(value, list):
            assert key in {"derived_from", "components"}, f"{key} leaked a list into runs list"


def test_list_payload_rejects_a_non_vocabulary_sort(repo: SqliteStrategyRepository) -> None:
    with pytest.raises(ValueError, match="not a sortable metric"):
        run_list_payload(
            repo, kind=None, strategy=None, family=None, sort="1; DROP TABLE runs", limit=10)


def test_list_payload_filters_by_family(repo: SqliteStrategyRepository) -> None:
    """`family` is resolved the same way `registry list --family` does: `strategies.family`."""
    repo.add("alpha", family="momentum")
    repo.add("beta", family="mean_reversion")
    repo.record_run("backtest", "alpha")
    repo.record_run("backtest", "beta")
    payload = run_list_payload(
        repo, kind=None, strategy=None, family="momentum", sort=None, limit=10)
    names = {row["strategy_name"] for row in payload["runs"]}
    assert names == {"alpha"}
    assert payload["count"] == 1


def test_list_payload_family_filter_applies_before_limit(repo: SqliteStrategyRepository) -> None:
    """FIX 1 regression: `--family` must be resolved to strategy names and pushed into
    `list_runs`'s own SQL filter BEFORE the LIMIT, not applied in Python over rows `list_runs`
    already truncated. With `sort` set, a post-hoc Python filter inverts the semantics: it returns
    the globally best N runs that happen to be in the family, which is systematically EMPTY for a
    small family ranked below the top `limit` runs — not merely truncated. This test plants exactly
    that shape (one low-`sharpe_oos` run in `trend`, `limit` runs with a higher `sharpe_oos` in a
    different family) and must fail against the pre-fix code."""
    repo.add("lonely", family="trend")
    repo.record_run("backtest", "lonely", metrics={"sharpe_oos": -1.0})
    repo.add("crowd", family="other")
    for i in range(20):
        repo.record_run("backtest", "crowd", metrics={"sharpe_oos": float(i)})
    payload = run_list_payload(
        repo, kind=None, strategy=None, family="trend", sort="sharpe_oos", limit=20)
    names = {row["strategy_name"] for row in payload["runs"]}
    assert names == {"lonely"}
    assert payload["count"] == 1


def test_list_payload_family_excludes_unregistered_runs(repo: SqliteStrategyRepository) -> None:
    """A run for a strategy with no `strategies` row (exploration precedes registration) has no
    family, so it must not slip through a family filter."""
    repo.record_run("backtest", "unregistered")
    payload = run_list_payload(
        repo, kind=None, strategy=None, family="momentum", sort=None, limit=10)
    assert payload["runs"] == []


def test_detail_payload_of_a_missing_run_is_an_error(repo: SqliteStrategyRepository) -> None:
    with pytest.raises(ValueError, match="no run"):
        run_detail_payload(repo, 9999)


def test_detail_payload_parses_lineage_as_lists(repo: SqliteStrategyRepository) -> None:
    parent = repo.record_run("walk_forward", "alpha")
    child = repo.record_run("gate", "alpha", derived_from=[parent])
    payload = run_detail_payload(repo, child)
    assert payload["derived_from"] == [parent]
    assert payload["components"] == []


def test_detail_payload_carries_extra_metrics_overflow(repo: SqliteStrategyRepository) -> None:
    run_id = repo.record_run(
        "gate", "alpha", extra_metrics={"dsr_t": 1.5, "fdr_p_value": 0.02})
    payload = run_detail_payload(repo, run_id)
    assert payload["extra_metrics"] == {"dsr_t": 1.5, "fdr_p_value": 0.02}


def test_detail_payload_gate_projects_allowlisted_decision(repo: SqliteStrategyRepository) -> None:
    """Reuses gate_history's real allowlist — asserted against the actual constant, not a
    hand-copied list, so a future field addition there is caught here too."""
    rec = repo.add("alpha")
    decision_json = json.dumps({
        "passed": True,
        "checks": [{
            "name": "holdout_sharpe", "op": ">", "threshold": 0.0, "value": 1.2,
            "passed": True, "advisory": False,
        }],
        # Deliberately NOT in GATE_DECISION_ALLOWLIST (per_regime_sharpes is a bare float list —
        # gate_history.py excludes it on purpose) and a made-up key that never belongs.
        "per_regime_sharpes": [0.1, 0.2, 0.3],
        "not_a_real_field": "smuggled",
    })
    gate_id = repo.record_gate_evaluation(
        rec.id, passed=True, n_funnel=1, own_lifetime_combos=1, windowed_total_combos=1,
        funnel_window_days=90, breadth_provenance="measured", pit_ok=True, pit_override=False,
        holdout_n_bars=63, min_holdout_observations=63, code_hash="c", config_hash="cfg",
        dependency_hash="d", data_source="SyntheticProvider", snapshot_id=None,
        period_start="2024-01-01", period_end="2024-06-01", holdout_frac=0.2, actor="agent",
        decision_json=decision_json,
    )
    run_id = repo.record_run("gate", "alpha", strategy_id=rec.id, gate_id=gate_id)
    payload = run_detail_payload(repo, run_id)
    decision = payload["gate_decision"]
    assert decision["checks"][0]["name"] == "holdout_sharpe"
    assert decision["passed"] is True
    assert set(decision) <= GATE_DECISION_ALLOWLIST
    assert "per_regime_sharpes" not in decision
    assert "not_a_real_field" not in decision
    assert "per_regime_sharpes" in payload["gate_decision_dropped_keys"]
    assert "not_a_real_field" in payload["gate_decision_dropped_keys"]


def test_detail_payload_non_gate_run_has_no_gate_decision(repo: SqliteStrategyRepository) -> None:
    run_id = repo.record_run("backtest", "alpha")
    payload = run_detail_payload(repo, run_id)
    assert "gate_decision" not in payload


def test_series_payload_reports_a_run_with_no_series_honestly(
    repo: SqliteStrategyRepository,
) -> None:
    run_id = repo.record_run("backtest", "alpha")
    payload = run_series_payload(repo, [run_id])
    assert payload["series"][str(run_id)] is None


def test_series_payload_returns_a_backtest_series(repo: SqliteStrategyRepository) -> None:
    returns = pd.Series(
        [0.01, -0.02], index=pd.to_datetime(["2024-01-02", "2024-01-03"]), dtype=float)
    series_id = repo.persist_backtest_returns("alpha", "2024-01-01", "2024-01-05", returns)
    run_id = repo.record_run("backtest", "alpha", series_backtest_id=series_id)
    payload = run_series_payload(repo, [run_id])
    entry = payload["series"][str(run_id)]
    assert entry["kind"] == "backtest"
    assert len(entry["returns"]) == 2
    assert entry["returns"][0][1] == pytest.approx(0.01)
    assert entry["returns"][1][1] == pytest.approx(-0.02)


def test_series_payload_returns_holdout_interval_context_without_per_bar_values(
    repo: SqliteStrategyRepository,
) -> None:
    """SENSITIVE: holdout_returns.returns_blob is a single-use OOS vector — see the DDL comment
    in algua/registry/db/holdout.py and the "ONLY method that reads returns_blob" docstring on
    overlapping_holdout_return_streams in algua/registry/store/holdout.py. run_series_payload must
    NEVER hand back a strategy's own per-bar OOS vector (that would re-open the single-use
    best-of-N surface sweep()'s holdout burn exists to prevent); it may only return the
    non-sensitive interval/n_bars scalars, enough to shade a chart's OOS region."""
    rec = repo.add("alpha")
    reservation_id, _reused = repo.reserve_holdout(
        rec.id, data_source="synthetic", snapshot_id=None,
        period_start="2024-01-01", period_end="2024-06-01", holdout_frac=0.2,
        holdout_start="2024-05-01", holdout_end="2024-05-03", allow_reuse=False,
    )
    repo.finalize_holdout_reservation(reservation_id, config_hash="cfg", strategy_id=rec.id)
    holdout_returns_id = repo.record_holdout_returns(
        reservation_id, rec.id, holdout_start="2024-05-01", holdout_end="2024-05-03",
        returns=[0.01, -0.02], bar_dates=["2024-05-01", "2024-05-02"],
    )
    run_id = repo.record_run(
        "gate", "alpha", strategy_id=rec.id, series_holdout_id=holdout_returns_id)
    payload = run_series_payload(repo, [run_id])
    entry = payload["series"][str(run_id)]
    assert entry["kind"] == "holdout"
    assert entry["holdout_start"] == "2024-05-01"
    assert entry["holdout_end"] == "2024-05-03"
    assert entry["n_bars"] == 2
    # Structural guard, not a substring check: no per-bar-derived value anywhere in the payload.
    # This is the regression test — it fails the moment someone re-adds a "returns" key here.
    assert "returns" not in entry
    assert "bar_dates" not in entry
    for value in entry.values():
        assert not isinstance(value, (list, tuple)), (
            f"holdout series entry leaked a per-bar sequence: {entry!r}")


def test_series_payload_missing_run_is_an_error(repo: SqliteStrategyRepository) -> None:
    with pytest.raises(ValueError, match="no run"):
        run_series_payload(repo, [9999])
