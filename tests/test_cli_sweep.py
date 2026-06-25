import json
import os
import sqlite3

import pytest
from typer.testing import CliRunner

from algua.cli.main import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def _tmp(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path))


def _ensure_family(strategy_name: str = "cross_sectional_momentum",
                   family_name: str = "csm_family") -> None:
    """Pre-assign a strategy to a family. Task 4 (#222) prerequisite for promote tests."""
    from datetime import UTC, datetime
    db_path = os.environ["ALGUA_DB_PATH"]
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    now = datetime.now(UTC).isoformat()
    try:
        row = conn.execute(
            "SELECT family_id FROM family_members WHERE strategy_name=? AND removed_at IS NULL",
            (strategy_name,),
        ).fetchone()
        if row is not None:
            return
        with conn:
            cur = conn.execute(
                "INSERT INTO families(name, created_at, created_by_actor, created_by_strategy)"
                " VALUES (?,?,?,?)",
                (family_name, now, "agent", strategy_name),
            )
            fam_id = cur.lastrowid
            conn.execute(
                "INSERT INTO family_events(event_type, family_id, actor, created_at)"
                " VALUES (?,?,?,?)",
                ("family_created", fam_id, "agent", now),
            )
            conn.execute(
                "INSERT INTO family_members(family_id, strategy_name, joined_at, joined_by_actor)"
                " VALUES (?,?,?,?)",
                (fam_id, strategy_name, now, "agent"),
            )
            conn.execute(
                "INSERT INTO family_events"
                "(event_type, family_id, strategy_name, actor,"
                " clustering_verdict, similarity_score, clustering_version,"
                " clustering_config_json, axis_json, matched_family_id, created_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                ("strategy_assigned", fam_id, strategy_name, "agent",
                 "NOVEL", 0.0, "v0", "{}", "{}", None, now),
            )
    finally:
        conn.close()


def test_sweep_demo_emits_ranked():
    result = runner.invoke(app, ["backtest", "sweep", "cross_sectional_momentum", "--demo",
                                 "--start", "2022-01-01", "--end", "2023-12-31",
                                 "--param", "lookback=20,40", "--param", "construction.top_k=1,3",
                                 "--top", "2"])
    assert result.exit_code == 0, result.stdout
    d = json.loads(result.stdout)
    assert d["n_combos"] == 4
    assert len(d["ranked"]) == 2
    assert d["best"]["params"]
    assert d["rank_by"] == "mean_sharpe"


def test_sweep_combos_carry_no_holdout():
    # The OOS holdout is WITHHELD from sweep entirely: ranking is on window/stability, and
    # exposing a per-combo holdout would let an agent SELECT on the holdout across combos (the
    # exact overfitting the breadth gate fights). It is revealed only by `research promote`.
    result = runner.invoke(app, ["backtest", "sweep", "cross_sectional_momentum", "--demo",
                                 "--start", "2022-01-01", "--end", "2023-12-31",
                                 "--param", "lookback=20,40", "--param", "construction.top_k=1,3"])
    assert result.exit_code == 0, result.stdout
    d = json.loads(result.stdout)
    assert d["ranked"]
    for entry in d["ranked"]:
        assert "holdout" not in entry
        assert "stability" in entry  # ranking signal is still present


def test_sweep_of_unregistered_strategy_still_records_breadth():
    # Breadth is keyed by NAME, so a sweep BEFORE registration still records (closes the leak
    # where an agent sweeps broadly first, then registers and declares a smaller --n-combos).
    result = runner.invoke(app, ["backtest", "sweep", "cross_sectional_momentum", "--demo",
                                 "--start", "2022-01-01", "--end", "2023-12-31",
                                 "--param", "lookback=20,40"])
    assert result.exit_code == 0, result.stdout
    assert json.loads(result.stdout)["recorded_breadth"] == {"n_combos": 2, "cumulative": 2}


def test_pre_registration_sweep_breadth_used_by_promote():
    # Sweep broadly while UNREGISTERED, then register + backtest, then promote: the measured sum
    # must be used (provenance "measured"), and a smaller --n-combos cannot undercut it.
    sweep = runner.invoke(app, ["backtest", "sweep", "cross_sectional_momentum", "--demo",
                                "--start", "2022-01-01", "--end", "2023-12-31",
                                "--param", "lookback=20,40", "--param", "construction.top_k=1,3"])
    assert sweep.exit_code == 0, sweep.stdout
    assert json.loads(sweep.stdout)["recorded_breadth"] == {"n_combos": 4, "cumulative": 4}

    backtested = runner.invoke(app, ["backtest", "run", "cross_sectional_momentum", "--demo",
                                     "--start", "2022-01-01", "--end", "2023-12-31", "--register"])
    assert backtested.exit_code == 0, backtested.stdout
    _ensure_family()

    promote = runner.invoke(app, ["research", "promote", "cross_sectional_momentum", "--demo",
                                  "--start", "2022-01-01", "--end", "2023-12-31",
                                  "--min-holdout-sharpe", "-100", "--min-holdout-return", "-100",
                                  "--min-pct-positive", "0", "--min-window-sharpe", "-100",
                                  "--n-combos", "1",  # smaller declaration must NOT undercut
                                  "--allow-non-pit", "--actor", "human"])
    assert promote.exit_code == 0, promote.stdout
    payload = json.loads(promote.stdout)
    assert payload["breadth_provenance"] == "measured"
    assert payload["n_funnel"] == 4


def test_sweep_of_registered_strategy_records_breadth():
    assert runner.invoke(app, ["registry", "add", "cross_sectional_momentum"]).exit_code == 0
    result = runner.invoke(app, ["backtest", "sweep", "cross_sectional_momentum", "--demo",
                                 "--start", "2022-01-01", "--end", "2023-12-31",
                                 "--param", "lookback=20,40", "--param", "construction.top_k=1,3"])
    assert result.exit_code == 0, result.stdout
    recorded = json.loads(result.stdout)["recorded_breadth"]
    assert recorded == {"n_combos": 4, "cumulative": 4}

    # A second sweep accumulates the cumulative family total.
    result2 = runner.invoke(app, ["backtest", "sweep", "cross_sectional_momentum", "--demo",
                                  "--start", "2022-01-01", "--end", "2023-12-31",
                                  "--param", "lookback=20,40"])
    assert json.loads(result2.stdout)["recorded_breadth"] == {"n_combos": 2, "cumulative": 6}


def test_sweep_requires_param():
    result = runner.invoke(app, ["backtest", "sweep", "cross_sectional_momentum", "--demo"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["ok"] is False


def test_sweep_malformed_param_is_json_error():
    result = runner.invoke(app, ["backtest", "sweep", "cross_sectional_momentum", "--demo",
                                 "--param", "lookback"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["ok"] is False


def test_sweep_requires_data_source():
    result = runner.invoke(app, ["backtest", "sweep", "cross_sectional_momentum",
                                 "--param", "lookback=20,40"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["ok"] is False


def test_sweep_top_zero_is_json_error():
    result = runner.invoke(app, ["backtest", "sweep", "cross_sectional_momentum", "--demo",
                                 "--param", "lookback=20,40", "--top", "0"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["ok"] is False
