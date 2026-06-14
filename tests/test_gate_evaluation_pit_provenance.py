from algua.registry.db import connect, migrate
from algua.registry.store import SqliteStrategyRepository


def _repo(tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    return SqliteStrategyRepository(conn)


def test_record_gate_evaluation_persists_pit_snapshots(tmp_path):
    repo = _repo(tmp_path)
    repo.add("s")
    sid = repo.get("s").id
    rid = repo.record_gate_evaluation(
        sid, passed=True, n_funnel=1, own_lifetime_combos=1, windowed_total_combos=1,
        funnel_window_days=30, breadth_provenance="measured", pit_ok=True, pit_override=False,
        holdout_n_bars=63, min_holdout_observations=63, code_hash="c", config_hash="cfg",
        dependency_hash="d", data_source="SyntheticProvider", snapshot_id="bars1",
        period_start="2023-01-01", period_end="2023-12-31", holdout_frac=0.2, actor="agent",
        decision_json="{}", news_snapshot="news1", fundamentals_snapshot=None)
    row = repo._conn.execute(
        "SELECT news_snapshot, fundamentals_snapshot FROM gate_evaluations WHERE id=?",
        (rid,)).fetchone()
    assert row["news_snapshot"] == "news1"
    assert row["fundamentals_snapshot"] is None


def test_record_gate_evaluation_defaults_pit_snapshots_to_null(tmp_path):
    # existing callers that don't pass the new kwargs still work; columns default NULL
    repo = _repo(tmp_path)
    repo.add("s2")
    sid = repo.get("s2").id
    rid = repo.record_gate_evaluation(
        sid, passed=False, n_funnel=1, own_lifetime_combos=1, windowed_total_combos=1,
        funnel_window_days=30, breadth_provenance="measured", pit_ok=True, pit_override=False,
        holdout_n_bars=63, min_holdout_observations=63, code_hash="c", config_hash="cfg",
        dependency_hash="d", data_source="SyntheticProvider", snapshot_id=None,
        period_start="2023-01-01", period_end="2023-12-31", holdout_frac=0.2, actor="agent",
        decision_json="{}")
    row = repo._conn.execute(
        "SELECT news_snapshot, fundamentals_snapshot FROM gate_evaluations WHERE id=?",
        (rid,)).fetchone()
    assert row["news_snapshot"] is None
    assert row["fundamentals_snapshot"] is None
