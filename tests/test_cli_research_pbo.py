"""CLI integration tests for `research pbo` (#467).

Aggregate-only advisory output; records search breadth (metered); burns no holdout; transitions
nothing; fully-reconstructable provenance (base config_hash + full grid_hash + delisting inputs).
"""

import json

import pytest
from typer.testing import CliRunner

from algua.cli._common import registry_conn
from algua.cli.main import app
from algua.data.store import DataStore
from algua.registry.store import SqliteStrategyRepository
from algua.strategies.base import config_hash
from algua.strategies.loader import load_strategy

runner = CliRunner()

STRATEGY = "cross_sectional_momentum"

_AGGREGATE_KEYS = {
    "note", "pbo", "split_count", "trial_count", "window_count", "subperiod_count",
    "rank_by", "warnings", "provenance",
}
_PROVENANCE_KEYS = {
    "config_hash", "code_hash", "dependency_hash", "data_source", "snapshot_id", "timeframe",
    "seed", "period", "universe_name", "universe_snapshots", "fundamentals_snapshot",
    "news_snapshot", "windows", "holdout_frac", "rank_by", "grid_hash", "delisting_snapshot",
    "delistings_name", "assume_terminal_last_close",
}
# Anything that would leak a per-combo selection oracle.
_FORBIDDEN_KEYS = ("grid", "best", "ranked", "matrix", "trial_window_sharpes", "logits",
                   "degradation_curve", "recorded_breadth")


@pytest.fixture(autouse=True)
def _tmp(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path))


def _run(*extra):
    return runner.invoke(app, ["research", "pbo", STRATEGY, "--demo",
                               "--start", "2022-01-01", "--end", "2023-12-31",
                               "--param", "lookback=20,40", "--windows", "4", *extra])


def _seed_delistings(tmp_path):
    csv = tmp_path / "d.csv"
    csv.write_text("symbol,delisting_date,delisting_value\nB,2020-01-02,5.0\n")
    r = runner.invoke(app, [
        "data", "import-delistings", "--file", str(csv), "--source", "v",
        "--as-of", "2019-12-31T00:00:00+00:00",
    ])
    assert r.exit_code == 0, r.output


def test_pbo_payload_is_aggregate_only():
    result = _run()
    assert result.exit_code == 0, result.stdout
    d = json.loads(result.stdout)
    assert d["ok"] is True
    # Exactly the aggregate + provenance keys (plus the ok wrapper), nothing else.
    assert set(d) == _AGGREGATE_KEYS | {"ok"}
    for k in _FORBIDDEN_KEYS:
        assert k not in d, f"leaked forbidden key {k!r}"
    assert d["pbo"] is not None and 0.0 <= d["pbo"] <= 1.0
    assert d["trial_count"] == 2
    assert d["window_count"] == 4
    assert d["subperiod_count"] == 4
    assert d["split_count"] == 6  # C(4, 2)
    assert d["rank_by"] == "mean_sharpe"
    assert d["warnings"] == []


def test_pbo_provenance_is_reconstructable():
    d = json.loads(_run().stdout)
    prov = d["provenance"]
    assert set(prov) == _PROVENANCE_KEYS
    # Full untruncated SHA-256 grid hash.
    assert len(prov["grid_hash"]) == 64
    assert all(c in "0123456789abcdef" for c in prov["grid_hash"])
    # Base strategy config identity.
    assert prov["config_hash"] == config_hash(load_strategy(STRATEGY))
    # No delistings passed -> null snapshot + null handle + false flag.
    assert prov["delisting_snapshot"] is None
    assert prov["delistings_name"] is None
    assert prov["assume_terminal_last_close"] is False
    assert prov["windows"] == 4
    assert prov["rank_by"] == "mean_sharpe"


def test_pbo_records_search_breadth():
    # ADVISORY but METERED (#467 R2-3): unlike the round-1 design it RECORDS breadth so repeated
    # runs self-penalize at promotion.
    result = _run()
    assert result.exit_code == 0, result.stdout
    with registry_conn() as conn:
        repo = SqliteStrategyRepository(conn)
        assert repo.total_search_combos(STRATEGY) == 2
    # A second run accumulates (metering).
    _run()
    with registry_conn() as conn:
        assert SqliteStrategyRepository(conn).total_search_combos(STRATEGY) == 4


def test_pbo_burns_no_holdout_and_transitions_nothing():
    # Register the strategy so we can observe it never transitions and no holdout is burned.
    reg = runner.invoke(app, ["registry", "add", STRATEGY])
    assert reg.exit_code == 0, reg.stdout
    d = json.loads(_run().stdout)
    assert d["ok"] is True
    show = json.loads(runner.invoke(app, ["registry", "show", STRATEGY]).stdout)
    assert show["stage"] == "idea"  # unchanged — pbo transitions nothing
    with registry_conn() as conn:
        # No holdout evaluation row was written by pbo.
        row = conn.execute(
            "SELECT COUNT(*) FROM holdout_evaluations WHERE strategy_id="
            "(SELECT id FROM strategies WHERE name=?)", (STRATEGY,)).fetchone()
        assert row[0] == 0


def test_pbo_delistings_provenance_stamps_resolved_snapshot(tmp_path):
    _seed_delistings(tmp_path)
    real = DataStore(tmp_path).latest_delistings_snapshot_id()
    assert real is not None
    d = json.loads(_run("--delistings", "ANYLABEL").stdout)
    prov = d["provenance"]
    # The RESOLVED snapshot id (not the user label) is stamped; the raw handle is kept separately.
    assert prov["delisting_snapshot"] == real
    assert prov["delisting_snapshot"] != "ANYLABEL"
    assert prov["delistings_name"] == "ANYLABEL"


def test_pbo_single_combo_fails_closed_exit_zero():
    # A 1-combo grid gives cscv.pbo < 2 trials -> FAIL CLOSED: pbo=null + warning, exit 0.
    result = runner.invoke(app, ["research", "pbo", STRATEGY, "--demo",
                                 "--start", "2022-01-01", "--end", "2023-12-31",
                                 "--param", "lookback=20", "--windows", "4"])
    assert result.exit_code == 0, result.stdout
    d = json.loads(result.stdout)
    assert d["ok"] is True
    assert d["pbo"] is None
    assert any("2 trials" in w for w in d["warnings"])


def test_pbo_too_few_windows_fails_closed_exit_zero():
    result = runner.invoke(app, ["research", "pbo", STRATEGY, "--demo",
                                 "--start", "2022-01-01", "--end", "2023-12-31",
                                 "--param", "lookback=20,40", "--windows", "3"])
    assert result.exit_code == 0, result.stdout
    d = json.loads(result.stdout)
    assert d["ok"] is True
    assert d["pbo"] is None
    assert any("windows" in w for w in d["warnings"])


@pytest.mark.parametrize("bad_windows", ["1", "0"])
def test_pbo_sub_two_windows_fails_closed_advisory_not_error(bad_windows):
    # walk_forward's _segment_bounds requires windows >= 2 and would raise a ValueError (exit-1
    # error envelope) before cscv.pbo could fail closed. `research pbo` must short-circuit a < 2
    # --windows into the SAME advisory fail-closed contract: pbo=null + warning, exit 0, aggregate
    # payload shape intact, and NO breadth recorded (no sweep ran → no real search to meter).
    result = runner.invoke(app, ["research", "pbo", STRATEGY, "--demo",
                                 "--start", "2022-01-01", "--end", "2023-12-31",
                                 "--param", "lookback=20,40", "--windows", bad_windows])
    assert result.exit_code == 0, result.stdout
    d = json.loads(result.stdout)
    assert d["ok"] is True
    assert d["pbo"] is None
    assert d["window_count"] == int(bad_windows)
    assert any("windows" in w for w in d["warnings"])
    # Payload shape is the normal aggregate+provenance surface (no leaked oracle keys).
    assert set(d.keys()) - {"ok"} == _AGGREGATE_KEYS
    assert set(d["provenance"].keys()) == _PROVENANCE_KEYS
    assert len(d["provenance"]["grid_hash"]) == 64
    for k in _FORBIDDEN_KEYS:
        assert k not in d
    # No sweep ran, so nothing was metered.
    with registry_conn() as conn:
        assert SqliteStrategyRepository(conn).total_search_combos(STRATEGY) == 0
