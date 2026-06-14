import json

import pytest
from typer.testing import CliRunner

from algua.cli.main import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def _tmp(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path))


def _stage(name="cross_sectional_momentum"):
    show = runner.invoke(app, ["registry", "show", name])
    return json.loads(show.stdout)["stage"]


def _backtest_to_backtested():
    return runner.invoke(app, ["backtest", "run", "cross_sectional_momentum", "--demo",
                               "--start", "2022-01-01", "--end", "2023-12-31", "--register"])


def _sweep():
    return runner.invoke(app, ["backtest", "sweep", "cross_sectional_momentum", "--demo",
                               "--start", "2022-01-01", "--end", "2023-12-31",
                               "--param", "lookback=20,40", "--param", "construction.top_k=1,3"])


def test_agent_promote_demo_refuses_relaxation():
    assert _backtest_to_backtested().exit_code == 0
    r = runner.invoke(app, ["research", "promote", "cross_sectional_momentum", "--demo",
                            "--start", "2022-01-01", "--end", "2023-12-31", "--n-combos", "9"])
    assert r.exit_code != 0
    assert "human" in r.stdout.lower()


def test_human_promote_demo_overrides_shortlists():
    assert _backtest_to_backtested().exit_code == 0
    r = runner.invoke(app, ["research", "promote", "cross_sectional_momentum", "--demo",
                            "--start", "2022-01-01", "--end", "2023-12-31",
                            "--min-holdout-sharpe", "-100", "--min-holdout-return", "-100",
                            "--min-pct-positive", "0", "--min-window-sharpe", "-100",
                            "--n-combos", "9", "--allow-non-pit", "--actor", "human"])
    assert r.exit_code == 0, r.stdout
    p = json.loads(r.stdout)
    assert p["promoted"] is True and p["n_funnel"] == 9 and p["pit_override"] is True
    assert _stage() == "candidate"


def test_agent_promote_blocked_without_pit():
    assert _backtest_to_backtested().exit_code == 0
    assert _sweep().exit_code == 0
    r = runner.invoke(app, ["research", "promote", "cross_sectional_momentum", "--demo",
                            "--start", "2022-01-01", "--end", "2023-12-31",
                            "--min-holdout-sharpe", "-100", "--min-holdout-return", "-100",
                            "--min-pct-positive", "0", "--min-window-sharpe", "-100"])
    assert r.exit_code == 0, r.stdout
    p = json.loads(r.stdout)
    assert p["promoted"] is False
    assert next(c for c in p["checks"] if c["name"] == "pit_required")["passed"] is False


def test_promote_passes_and_shortlists():
    assert _backtest_to_backtested().exit_code == 0
    # No sweep recorded yet, so declare breadth explicitly via --n-combos.
    result = runner.invoke(app, ["research", "promote", "cross_sectional_momentum", "--demo",
                                 "--start", "2022-01-01", "--end", "2023-12-31",
                                 "--min-holdout-sharpe", "-100", "--min-holdout-return", "-100",
                                 "--min-pct-positive", "0", "--min-window-sharpe", "-100",
                                 "--n-combos", "9", "--allow-non-pit", "--actor", "human"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["passed"] is True
    assert payload["promoted"] is True
    assert payload["n_funnel"] == 9
    assert payload["breadth_provenance"] == "declared"
    # A declared breadth still raises the bar: effective > base.
    assert payload["effective_min_holdout_sharpe"] > payload["base_min_holdout_sharpe"]
    assert _stage() == "candidate"


def test_promote_uses_measured_breadth_from_sweep():
    assert _backtest_to_backtested().exit_code == 0
    assert _sweep().exit_code == 0  # records a 4-combo search_trial
    result = runner.invoke(app, ["research", "promote", "cross_sectional_momentum", "--demo",
                                 "--start", "2022-01-01", "--end", "2023-12-31",
                                 "--min-holdout-sharpe", "-100", "--min-holdout-return", "-100",
                                 "--min-pct-positive", "0", "--min-window-sharpe", "-100",
                                 "--allow-non-pit", "--actor", "human"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["breadth_provenance"] == "measured"
    assert payload["n_funnel"] == 4


def test_measured_breadth_wins_over_declaration():
    assert _backtest_to_backtested().exit_code == 0
    assert _sweep().exit_code == 0  # 4 combos measured
    result = runner.invoke(app, ["research", "promote", "cross_sectional_momentum", "--demo",
                                 "--start", "2022-01-01", "--end", "2023-12-31",
                                 "--min-holdout-sharpe", "-100", "--min-holdout-return", "-100",
                                 "--min-pct-positive", "0", "--min-window-sharpe", "-100",
                                 "--n-combos", "999",  # declaration ignored
                                 "--allow-non-pit", "--actor", "human"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["breadth_provenance"] == "measured"
    assert payload["n_funnel"] == 4


def test_two_sweeps_accumulate_breadth():
    assert _backtest_to_backtested().exit_code == 0
    assert _sweep().exit_code == 0
    assert _sweep().exit_code == 0  # second sweep accumulates
    result = runner.invoke(app, ["research", "promote", "cross_sectional_momentum", "--demo",
                                 "--start", "2022-01-01", "--end", "2023-12-31",
                                 "--min-holdout-sharpe", "-100", "--min-holdout-return", "-100",
                                 "--min-pct-positive", "0", "--min-window-sharpe", "-100",
                                 "--allow-non-pit", "--actor", "human"])
    payload = json.loads(result.stdout)
    assert payload["n_funnel"] == 8  # 4 + 4


def test_promote_refuses_with_no_breadth():
    assert _backtest_to_backtested().exit_code == 0
    result = runner.invoke(app, ["research", "promote", "cross_sectional_momentum", "--demo",
                                 "--start", "2022-01-01", "--end", "2023-12-31",
                                 "--min-holdout-sharpe", "-100", "--min-holdout-return", "-100",
                                 "--min-pct-positive", "0", "--min-window-sharpe", "-100"])
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert "search breadth" in payload["error"]
    assert _stage() == "backtested"  # not transitioned


def test_promote_fails_does_not_transition():
    assert _backtest_to_backtested().exit_code == 0
    result = runner.invoke(app, ["research", "promote", "cross_sectional_momentum", "--demo",
                                 "--start", "2022-01-01", "--end", "2023-12-31",
                                 "--min-holdout-sharpe", "999", "--n-combos", "1",
                                 "--allow-non-pit", "--actor", "human"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["passed"] is False
    assert payload["promoted"] is False
    assert _stage() == "backtested"


def test_promote_from_idea_is_json_error():
    runner.invoke(app, ["registry", "add", "cross_sectional_momentum"])
    result = runner.invoke(app, ["research", "promote", "cross_sectional_momentum", "--demo",
                                 "--start", "2022-01-01", "--end", "2023-12-31",
                                 "--min-holdout-sharpe", "-100", "--min-holdout-return", "-100",
                                 "--min-pct-positive", "0", "--min-window-sharpe", "-100"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["ok"] is False


def test_promote_rejects_bad_n_combos():
    result = runner.invoke(app, ["research", "promote", "cross_sectional_momentum",
                                 "--demo", "--n-combos", "0"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["ok"] is False


def test_promote_rejects_out_of_range_pct_positive():
    result = runner.invoke(app, ["research", "promote", "cross_sectional_momentum",
                                 "--demo", "--min-pct-positive", "1.5"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["ok"] is False


# --- Holdout-reuse (single-use holdout) -------------------------------------

_PASS = ["--min-holdout-sharpe", "-100", "--min-holdout-return", "-100",
         "--min-pct-positive", "0", "--min-window-sharpe", "-100"]


def _holdout_rows(tmp_path):
    import sqlite3
    conn = sqlite3.connect(tmp_path / "r.db")
    try:
        return conn.execute(
            "SELECT period_start, period_end, holdout_frac, reused, data_source, snapshot_id"
            " FROM holdout_evaluations ORDER BY id"
        ).fetchall()
    finally:
        conn.close()


def _gate_rows(tmp_path):
    import sqlite3
    conn = sqlite3.connect(tmp_path / "r.db")
    try:
        return conn.execute("SELECT passed FROM gate_evaluations ORDER BY id").fetchall()
    finally:
        conn.close()


def test_promote_system_actor_refused_before_holdout_and_gate_row(tmp_path):
    assert _backtest_to_backtested().exit_code == 0
    assert _sweep().exit_code == 0  # measured breadth, so the actor check is the refusal
    result = runner.invoke(app, ["research", "promote", "cross_sectional_momentum", "--demo",
                                 "--start", "2022-01-01", "--end", "2023-12-31",
                                 *_PASS, "--allow-non-pit", "--actor", "system"])
    assert result.exit_code == 1, result.stdout
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert "agent or human" in payload["error"]
    # Refused in preflight, before walk_forward: no gate row minted, no holdout burned.
    assert len(_gate_rows(tmp_path)) == 0
    assert len(_holdout_rows(tmp_path)) == 0
    assert _stage() == "backtested"


def test_gate_row_written_on_both_pass_and_fail(tmp_path):
    # A FAILING gate (impossible Sharpe) still records an audit row with passed=0; a PASSING gate
    # records one with passed=1. The strategy stays `backtested` after the fail so the second
    # (passing) promote on the same lifecycle is legal.
    assert _backtest_to_backtested().exit_code == 0
    fail = runner.invoke(app, ["research", "promote", "cross_sectional_momentum", "--demo",
                               "--start", "2022-01-01", "--end", "2023-12-31",
                               "--min-holdout-sharpe", "999", "--n-combos", "9",
                               "--allow-non-pit", "--actor", "human"])
    assert fail.exit_code == 0, fail.stdout
    assert json.loads(fail.stdout)["passed"] is False
    rows = _gate_rows(tmp_path)
    assert len(rows) == 1 and rows[0][0] == 0  # one row, passed=0
    assert _stage() == "backtested"

    # Passing promote on a DISJOINT window (avoids the burned-holdout refusal) records passed=1.
    ok = runner.invoke(app, ["research", "promote", "cross_sectional_momentum", "--demo",
                             "--start", "2022-01-01", "--end", "2023-12-31",
                             *_PASS, "--n-combos", "9", "--allow-holdout-reuse",
                             "--allow-non-pit", "--actor", "human"])
    assert ok.exit_code == 0, ok.stdout
    assert json.loads(ok.stdout)["passed"] is True
    rows = _gate_rows(tmp_path)
    assert len(rows) == 2 and rows[1][0] == 1  # second row, passed=1
    assert _stage() == "candidate"


def test_first_promote_records_holdout_evaluation(tmp_path):
    assert _backtest_to_backtested().exit_code == 0
    result = runner.invoke(app, ["research", "promote", "cross_sectional_momentum", "--demo",
                                 "--start", "2022-01-01", "--end", "2023-12-31",
                                 *_PASS, "--n-combos", "9", "--allow-non-pit", "--actor", "human"])
    assert result.exit_code == 0, result.stdout
    rows = _holdout_rows(tmp_path)
    assert len(rows) == 1
    assert rows[0][3] == 0  # reused == 0
    assert _stage() == "candidate"


def test_second_promote_same_window_refused(tmp_path):
    assert _backtest_to_backtested().exit_code == 0
    # First promote FAILS the gate (impossible Sharpe) so the strategy stays `backtested`; this
    # isolates the holdout-reuse refusal on the second run from the stage-legality preflight check.
    first = runner.invoke(app, ["research", "promote", "cross_sectional_momentum", "--demo",
                                "--start", "2022-01-01", "--end", "2023-12-31",
                                "--min-holdout-sharpe", "999", "--n-combos", "9",
                                "--allow-non-pit", "--actor", "human"])
    assert first.exit_code == 0, first.stdout
    assert json.loads(first.stdout)["passed"] is False
    second = runner.invoke(app, ["research", "promote", "cross_sectional_momentum", "--demo",
                                 "--start", "2022-01-01", "--end", "2023-12-31",
                                 *_PASS, "--n-combos", "9", "--allow-non-pit", "--actor", "human"])
    assert second.exit_code == 1, second.stdout
    payload = json.loads(second.stdout)
    assert payload["ok"] is False
    assert "holdout" in payload["error"].lower()
    # No second row written; stage unchanged (first promote failed the gate).
    assert len(_holdout_rows(tmp_path)) == 1


def test_failing_first_promote_still_burns_holdout(tmp_path):
    assert _backtest_to_backtested().exit_code == 0
    # Impossible Sharpe bar -> gate fails, but the holdout was looked at and is now burned.
    first = runner.invoke(app, ["research", "promote", "cross_sectional_momentum", "--demo",
                                "--start", "2022-01-01", "--end", "2023-12-31",
                                "--min-holdout-sharpe", "999", "--n-combos", "1",
                                "--allow-non-pit", "--actor", "human"])
    assert first.exit_code == 0, first.stdout
    assert json.loads(first.stdout)["passed"] is False
    assert len(_holdout_rows(tmp_path)) == 1
    second = runner.invoke(app, ["research", "promote", "cross_sectional_momentum", "--demo",
                                 "--start", "2022-01-01", "--end", "2023-12-31",
                                 *_PASS, "--n-combos", "1", "--allow-non-pit", "--actor", "human"])
    assert second.exit_code == 1, second.stdout
    assert json.loads(second.stdout)["ok"] is False
    assert len(_holdout_rows(tmp_path)) == 1


def test_allow_holdout_reuse_overrides(tmp_path):
    assert _backtest_to_backtested().exit_code == 0
    # First promote FAILS the gate (impossible Sharpe) so the strategy stays `backtested`, but the
    # holdout is burned. The override then re-evaluates the same window and promotes on the merits.
    first = runner.invoke(app, ["research", "promote", "cross_sectional_momentum", "--demo",
                                "--start", "2022-01-01", "--end", "2023-12-31",
                                "--min-holdout-sharpe", "999", "--n-combos", "9",
                                "--allow-non-pit", "--actor", "human"])
    assert first.exit_code == 0, first.stdout
    assert json.loads(first.stdout)["passed"] is False
    second = runner.invoke(app, ["research", "promote", "cross_sectional_momentum", "--demo",
                                 "--start", "2022-01-01", "--end", "2023-12-31",
                                 *_PASS, "--n-combos", "9", "--allow-holdout-reuse",
                                 "--allow-non-pit", "--actor", "human"])
    assert second.exit_code == 0, second.stdout
    payload = json.loads(second.stdout)
    assert payload["holdout_reuse"] == "override"
    rows = _holdout_rows(tmp_path)
    assert len(rows) == 2
    assert rows[1][3] == 1  # second row reused == 1
    # The override is recorded in the transition reason.
    show = runner.invoke(app, ["registry", "show", "cross_sectional_momentum"])
    history = json.loads(show.stdout)["transitions"]
    assert any("override" in (t.get("reason") or "") for t in history)


def test_non_overlapping_window_allowed(tmp_path):
    assert _backtest_to_backtested().exit_code == 0
    # Both gates fail (impossible Sharpe) so the strategy stays `backtested` and neither call
    # attempts a same-stage transition -- this isolates the holdout pre-check from the lifecycle.
    first = runner.invoke(app, ["research", "promote", "cross_sectional_momentum", "--demo",
                                "--start", "2022-01-01", "--end", "2022-12-31",
                                "--min-holdout-sharpe", "999", "--n-combos", "1",
                                "--allow-non-pit", "--actor", "human"])
    assert first.exit_code == 0, first.stdout
    # Disjoint period -> allowed without override (no refusal), records a second row.
    second = runner.invoke(app, ["research", "promote", "cross_sectional_momentum", "--demo",
                                 "--start", "2023-01-01", "--end", "2023-12-31",
                                 "--min-holdout-sharpe", "999", "--n-combos", "1",
                                 "--allow-non-pit", "--actor", "human"])
    assert second.exit_code == 0, second.stdout
    assert len(_holdout_rows(tmp_path)) == 2


def test_config_change_alone_does_not_bypass(tmp_path):
    assert _backtest_to_backtested().exit_code == 0
    first = runner.invoke(app, ["research", "promote", "cross_sectional_momentum", "--demo",
                                "--start", "2022-01-01", "--end", "2023-12-31",
                                *_PASS, "--n-combos", "9", "--allow-non-pit", "--actor", "human"])
    assert first.exit_code == 0, first.stdout
    # Same window + holdout_frac, tweaked gate params (config_hash unchanged here, but the rule
    # matches on the WINDOW regardless of config): still refused.
    second = runner.invoke(app, ["research", "promote", "cross_sectional_momentum", "--demo",
                                 "--start", "2022-01-01", "--end", "2023-12-31",
                                 "--min-holdout-sharpe", "0.1", "--min-holdout-return", "-100",
                                 "--min-pct-positive", "0", "--min-window-sharpe", "-100",
                                 "--n-combos", "9", "--allow-non-pit", "--actor", "human"])
    assert second.exit_code == 1, second.stdout
    assert json.loads(second.stdout)["ok"] is False


# --- Point-in-time universe wiring through the promotion gate -----------------


def _ingest_snapshot(tmp_path):
    """Ingest synthetic momentum-universe bars; return the snapshot id."""
    from datetime import UTC, datetime

    from algua.backtest._sample import SyntheticProvider
    from algua.data.store import DataStore

    symbols = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]
    start, end = datetime(2022, 1, 1, tzinfo=UTC), datetime(2023, 12, 31, tzinfo=UTC)
    bars = SyntheticProvider(seed=0).get_bars(symbols, start, end, "1d")
    frame = bars.reset_index().rename(columns={"timestamp": "ts"})
    rec = DataStore(tmp_path).ingest_bars(
        provider="synthetic", symbols=symbols, start="2022-01-01", end="2023-12-31",
        as_of="2024-01-01", source="test", frame=frame, timeframe="1d", adjustment="none",
    )
    return rec.snapshot_id


def _ingest_pit_universe(tmp_path):
    """Ingest a time-varying universe `pit_core`: AAPL/MSFT from 2022, NVDA added 2023.

    Returns the two snapshot ids in effective-date order.
    """
    from algua.data.store import DataStore

    store = DataStore(tmp_path)
    first = store.ingest_universe(
        universe="pit_core", symbols=["AAPL", "MSFT"], effective_date="2022-01-01",
        as_of="2022-01-02T00:00:00+00:00", source="test",
    )
    second = store.ingest_universe(
        universe="pit_core", symbols=["AAPL", "MSFT", "NVDA"], effective_date="2023-01-01",
        as_of="2023-01-02T00:00:00+00:00", source="test",
    )
    return first.snapshot_id, second.snapshot_id


def _register_backtested_on_snapshot(snap):
    return runner.invoke(app, ["backtest", "run", "cross_sectional_momentum",
                               "--snapshot", snap, "--register",
                               "--start", "2022-01-01", "--end", "2023-12-31"])


def test_promote_with_universe_threads_pit_provenance(tmp_path, monkeypatch):
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))
    snap = _ingest_snapshot(tmp_path)
    first_u, second_u = _ingest_pit_universe(tmp_path)
    assert _register_backtested_on_snapshot(snap).exit_code == 0
    result = runner.invoke(app, ["research", "promote", "cross_sectional_momentum",
                                 "--snapshot", snap, "--universe", "pit_core",
                                 "--start", "2022-01-01", "--end", "2023-12-31",
                                 *_PASS, "--n-combos", "9", "--actor", "human"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["passed"] is True
    assert payload["promoted"] is True
    assert payload["universe_name"] == "pit_core"
    eff = [s["effective_date"] for s in payload["universe_snapshots"]]
    assert eff == ["2022-01-01", "2023-01-01"]
    assert {s["snapshot_id"] for s in payload["universe_snapshots"]} == {first_u, second_u}
    # The bars snapshot_id is a SEPARATE provenance dimension — still the bars snapshot.
    assert payload["snapshot_id"] == snap
    assert _stage() == "candidate"


def test_promote_pit_membership_changes_holdout_outcome(tmp_path, monkeypatch):
    """A PIT-restricted universe (excluding symbols static mode would include) yields a
    different holdout metric than the static run — proving the map threads into the engine."""
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))
    snap = _ingest_snapshot(tmp_path)
    _ingest_pit_universe(tmp_path)  # AAPL/MSFT then +NVDA — excludes AMZN/GOOGL always
    assert _register_backtested_on_snapshot(snap).exit_code == 0

    static = runner.invoke(app, ["research", "promote", "cross_sectional_momentum",
                                 "--snapshot", snap,
                                 "--start", "2022-01-01", "--end", "2023-12-31",
                                 "--min-holdout-sharpe", "999", "--n-combos", "9",
                                 "--allow-non-pit", "--actor", "human"])
    assert static.exit_code == 0, static.stdout
    static_holdout = json.loads(static.stdout)["holdout"]

    pit = runner.invoke(app, ["research", "promote", "cross_sectional_momentum",
                              "--snapshot", snap, "--universe", "pit_core",
                              "--start", "2022-01-01", "--end", "2023-12-31",
                              "--min-holdout-sharpe", "999", "--n-combos", "9",
                              "--allow-holdout-reuse", "--actor", "human"])
    assert pit.exit_code == 0, pit.stdout
    pit_payload = json.loads(pit.stdout)
    assert pit_payload["universe_name"] == "pit_core"
    # PIT membership excludes AMZN/GOOGL the static run includes -> different holdout metrics.
    assert pit_payload["holdout"]["total_return"] != static_holdout["total_return"]


def test_promote_without_universe_has_null_universe_provenance(tmp_path, monkeypatch):
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))
    assert _backtest_to_backtested().exit_code == 0
    result = runner.invoke(app, ["research", "promote", "cross_sectional_momentum", "--demo",
                                 "--start", "2022-01-01", "--end", "2023-12-31",
                                 *_PASS, "--n-combos", "9", "--allow-non-pit", "--actor", "human"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["universe_name"] is None
    assert payload["universe_snapshots"] is None


def test_promote_unknown_universe_is_json_error(tmp_path, monkeypatch):
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))
    snap = _ingest_snapshot(tmp_path)
    assert _register_backtested_on_snapshot(snap).exit_code == 0
    result = runner.invoke(app, ["research", "promote", "cross_sectional_momentum",
                                 "--snapshot", snap, "--universe", "does_not_exist",
                                 "--start", "2022-01-01", "--end", "2023-12-31",
                                 *_PASS, "--n-combos", "9", "--actor", "human"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["ok"] is False


def test_promote_universe_burn_keyed_on_window_not_universe(tmp_path, monkeypatch):
    """The holdout burn is keyed on the data window, NOT the universe: a second promote on the
    same window+snapshot under a DIFFERENT universe is still refused (conservative)."""
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))
    snap = _ingest_snapshot(tmp_path)
    _ingest_pit_universe(tmp_path)
    assert _register_backtested_on_snapshot(snap).exit_code == 0
    # First promote FAILS the gate (impossible Sharpe) so the strategy stays `backtested`; this
    # isolates the holdout-burn refusal on the second run from the stage-legality preflight check.
    first = runner.invoke(app, ["research", "promote", "cross_sectional_momentum",
                                "--snapshot", snap,
                                "--start", "2022-01-01", "--end", "2023-12-31",
                                "--min-holdout-sharpe", "999", "--n-combos", "9",
                                "--allow-non-pit", "--actor", "human"])
    assert first.exit_code == 0, first.stdout
    assert json.loads(first.stdout)["passed"] is False
    # Same window/snapshot, now WITH a universe -> still refused (universe not in burn identity).
    second = runner.invoke(app, ["research", "promote", "cross_sectional_momentum",
                                 "--snapshot", snap, "--universe", "pit_core",
                                 "--start", "2022-01-01", "--end", "2023-12-31",
                                 *_PASS, "--n-combos", "9", "--actor", "human"])
    assert second.exit_code == 1, second.stdout
    assert "holdout" in json.loads(second.stdout)["error"].lower()


def test_walk_forward_does_not_burn_holdout_then_promote_succeeds(tmp_path):
    """`backtest walk-forward` must NOT consume the holdout: it records no holdout_evaluations row,
    and a later `promote` on the SAME window still succeeds (walk-forward didn't burn it)."""
    assert _backtest_to_backtested().exit_code == 0
    wf = runner.invoke(app, ["backtest", "walk-forward", "cross_sectional_momentum", "--demo",
                             "--start", "2022-01-01", "--end", "2023-12-31"])
    assert wf.exit_code == 0, wf.stdout
    assert "holdout_metrics" not in json.loads(wf.stdout)
    assert len(_holdout_rows(tmp_path)) == 0  # walk-forward burned nothing

    promote = runner.invoke(app, ["research", "promote", "cross_sectional_momentum", "--demo",
                                  "--start", "2022-01-01", "--end", "2023-12-31",
                                  *_PASS, "--n-combos", "9", "--allow-non-pit", "--actor", "human"])
    assert promote.exit_code == 0, promote.stdout  # not refused — holdout was still fresh
    payload = json.loads(promote.stdout)
    assert payload["promoted"] is True
    assert payload["holdout"]["n_bars"] > 0  # promote DOES reveal the holdout
    assert len(_holdout_rows(tmp_path)) == 1  # and promote burns it


def test_promote_with_universe_refuses_with_no_breadth_before_walkforward(tmp_path, monkeypatch):
    """Breadth refusal still fires before walk_forward even with --universe present."""
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))
    snap = _ingest_snapshot(tmp_path)
    _ingest_pit_universe(tmp_path)
    assert _register_backtested_on_snapshot(snap).exit_code == 0
    result = runner.invoke(app, ["research", "promote", "cross_sectional_momentum",
                                 "--snapshot", snap, "--universe", "pit_core",
                                 "--start", "2022-01-01", "--end", "2023-12-31", *_PASS])
    assert result.exit_code == 1, result.stdout
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert "search breadth" in payload["error"]
    # No holdout row written: refusal happened before walk_forward burned anything.
    assert len(_holdout_rows(tmp_path)) == 0


def test_promote_different_holdout_frac_is_refused_as_reburn(tmp_path):
    """A second `research promote` on the same date window with a DIFFERENT --holdout-frac must be
    refused as a holdout re-burn: the OOS tails overlap, so the interval-based guard fires.

    First promote uses --holdout-frac 0.2 (burns the last 20% as OOS).  Second uses --holdout-frac
    0.4 (the last 40% overlaps the first OOS tail).  The second must be refused even though the
    holdout_frac parameter differs — the guard matches on the ACTUAL bar interval, not the fraction.
    """
    assert _backtest_to_backtested().exit_code == 0
    # First promote: impossible Sharpe bar -> gate fails, strategy stays `backtested`.
    # But the holdout IS peeked (walk_forward ran), so the burn is committed before the gate check.
    first = runner.invoke(app, ["research", "promote", "cross_sectional_momentum", "--demo",
                                "--start", "2022-01-01", "--end", "2023-12-31",
                                "--holdout-frac", "0.2",
                                "--min-holdout-sharpe", "999", "--n-combos", "1",
                                "--allow-non-pit", "--actor", "human"])
    assert first.exit_code == 0, first.stdout
    first_payload = json.loads(first.stdout)
    # The first call must NOT itself be a "holdout already consumed" refusal.
    first_error = first_payload.get("error", "")
    assert first_payload.get("ok") is not False or "holdout already consumed" not in first_error
    # Holdout is burned — exactly one row committed.
    assert len(_holdout_rows(tmp_path)) == 1

    # Second promote: same date window, DIFFERENT holdout_frac (0.4 > 0.2 -> larger OOS tail that
    # overlaps the first burn). Must be refused as a holdout re-burn.
    second = runner.invoke(app, ["research", "promote", "cross_sectional_momentum", "--demo",
                                 "--start", "2022-01-01", "--end", "2023-12-31",
                                 "--holdout-frac", "0.4",
                                 *_PASS, "--n-combos", "1",
                                 "--allow-non-pit", "--actor", "human"])
    assert second.exit_code == 1, second.stdout
    payload = json.loads(second.stdout)
    assert payload["ok"] is False
    assert "holdout already consumed" in payload["error"]
    # No second holdout row written; still only the one from the first promote.
    assert len(_holdout_rows(tmp_path)) == 1


# --- dormant-sweep -----------------------------------------------------------


def _to_dormant(name="cross_sectional_momentum"):
    """Register `name` and drive it idea->backtested->candidate->paper->dormant via the CLI.
    Human actor is exempt from the agent token gates up to paper; paper->dormant is any-actor but
    requires a reason."""
    assert runner.invoke(app, ["registry", "add", name]).exit_code == 0
    chain = [("backtested", "human"), ("candidate", "human"),
             ("paper", "human"), ("dormant", "agent")]
    for to, actor in chain:
        r = runner.invoke(app, ["registry", "transition", name, "--to", to,
                                "--actor", actor, "--reason", "test"])
        assert r.exit_code == 0, r.stdout


def test_dormant_sweep_empty_pool():
    r = runner.invoke(app, ["research", "dormant-sweep", "--demo",
                            "--start", "2022-01-01", "--end", "2023-12-31"])
    assert r.exit_code == 0, r.stdout
    p = json.loads(r.stdout)
    assert p["ok"] is True
    assert p["total_dormant"] == 0
    assert p["passed"] == [] and p["failed"] == [] and p["skipped"] == [] and p["errors"] == []


def test_dormant_sweep_routes_pass():
    _to_dormant()
    r = runner.invoke(app, ["research", "dormant-sweep", "--demo",
                            "--start", "2022-01-01", "--end", "2023-12-31",
                            "--min-window-sharpe", "-100", "--min-pct-positive", "0"])
    assert r.exit_code == 0, r.stdout
    p = json.loads(r.stdout)
    assert p["total_dormant"] == 1 and p["evaluated"] == 1
    assert [x["strategy"] for x in p["passed"]] == ["cross_sectional_momentum"]
    assert p["failed"] == []
    assert p["passed"][0]["screen_passed"] is True
    assert "stability" in p["passed"][0]


def test_dormant_sweep_routes_fail():
    _to_dormant()
    r = runner.invoke(app, ["research", "dormant-sweep", "--demo",
                            "--start", "2022-01-01", "--end", "2023-12-31",
                            "--min-window-sharpe", "100", "--min-pct-positive", "1.0"])
    assert r.exit_code == 0, r.stdout
    p = json.loads(r.stdout)
    assert p["evaluated"] == 1
    assert [x["strategy"] for x in p["failed"]] == ["cross_sectional_momentum"]
    assert p["passed"] == []


def test_dormant_sweep_never_reveals_holdout():
    _to_dormant()
    r = runner.invoke(app, ["research", "dormant-sweep", "--demo",
                            "--start", "2022-01-01", "--end", "2023-12-31",
                            "--min-window-sharpe", "-100", "--min-pct-positive", "0"])
    assert r.exit_code == 0, r.stdout
    p = json.loads(r.stdout)
    # The per-strategy entries must contain ONLY window/stability data, never a "holdout" key.
    for entry in p["passed"] + p["failed"]:
        assert "holdout" not in entry, (
            f"strategy entry for {entry.get('strategy')} leaks a 'holdout' key: {list(entry)}"
        )


def test_dormant_sweep_has_no_side_effects():
    _to_dormant()
    import os
    from contextlib import closing
    from pathlib import Path

    from algua.registry.db import connect

    def _counts():
        with closing(connect(Path(os.environ["ALGUA_DB_PATH"]))) as conn:
            ge = conn.execute("SELECT COUNT(*) FROM gate_evaluations").fetchone()[0]
            ho = conn.execute("SELECT COUNT(*) FROM holdout_evaluations").fetchone()[0]
            stage = conn.execute(
                "SELECT stage FROM strategies WHERE name='cross_sectional_momentum'"
            ).fetchone()[0]
        return ge, ho, stage

    before = _counts()
    r = runner.invoke(app, ["research", "dormant-sweep", "--demo",
                            "--start", "2022-01-01", "--end", "2023-12-31"])
    assert r.exit_code == 0, r.stdout
    after = _counts()
    assert after == before
    assert after[2] == "dormant"


def test_dormant_sweep_is_repeatable():
    _to_dormant()
    args = ["research", "dormant-sweep", "--demo", "--start", "2022-01-01", "--end", "2023-12-31",
            "--min-window-sharpe", "-100", "--min-pct-positive", "0"]
    p1 = json.loads(runner.invoke(app, args).stdout)
    p2 = json.loads(runner.invoke(app, args).stdout)
    assert [x["strategy"] for x in p1["passed"]] == [x["strategy"] for x in p2["passed"]]
    assert p1["evaluated"] == p2["evaluated"] == 1


def test_dormant_sweep_skips_fundamentals_and_evaluates_others_in_one_run():
    _to_dormant("cross_sectional_momentum")
    _to_dormant("fundamentals_earnings_tilt")
    r = runner.invoke(app, ["research", "dormant-sweep", "--demo",
                            "--start", "2022-01-01", "--end", "2023-12-31",
                            "--min-window-sharpe", "-100", "--min-pct-positive", "0"])
    assert r.exit_code == 0, r.stdout
    p = json.loads(r.stdout)
    assert p["total_dormant"] == 2
    assert [s["strategy"] for s in p["skipped"]] == ["fundamentals_earnings_tilt"]
    assert "needs_fundamentals" in p["skipped"][0]["reason"]
    evaluated_names = [x["strategy"] for x in p["passed"]] + [x["strategy"] for x in p["failed"]]
    assert "cross_sectional_momentum" in evaluated_names


def test_dormant_sweep_ignores_non_dormant_strategies():
    assert runner.invoke(app, ["registry", "add", "cross_sectional_momentum"]).exit_code == 0
    assert runner.invoke(app, ["registry", "transition", "cross_sectional_momentum",
                               "--to", "backtested", "--actor", "human",
                               "--reason", "x"]).exit_code == 0
    r = runner.invoke(app, ["research", "dormant-sweep", "--demo",
                            "--start", "2022-01-01", "--end", "2023-12-31"])
    assert r.exit_code == 0, r.stdout
    p = json.loads(r.stdout)
    assert p["total_dormant"] == 0
    names = [x["strategy"] for x in p["passed"] + p["failed"]]
    names += [s["strategy"] for s in p["skipped"]]
    assert "cross_sectional_momentum" not in names
