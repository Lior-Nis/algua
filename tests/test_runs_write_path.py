"""Every evaluation lands a run row — including for an UNREGISTERED strategy."""
from __future__ import annotations

import pytest

from algua.evaluation.backtest_run import run_backtest_task
from algua.registry.db import registry_conn
from algua.registry.store import SqliteStrategyRepository

STRATEGY = "cross_sectional_momentum"


@pytest.fixture(autouse=True)
def _isolated_db(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:  # noqa: ANN001
    """The established DB-isolation idiom (see tests/test_cli_backtest.py)."""
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))


def test_backtest_records_a_run() -> None:
    run_backtest_task(STRATEGY, demo=True)
    with registry_conn() as conn:
        rows = SqliteStrategyRepository(conn).list_runs(kind="backtest")
    assert len(rows) == 1
    assert rows[0]["strategy_name"] == STRATEGY
    assert rows[0]["sharpe_is"] is not None
    assert rows[0]["code_hash"] is not None


def test_backtest_records_a_run_for_an_unregistered_strategy() -> None:
    """Exploration precedes registration — that evidence must not be discarded."""
    run_backtest_task(STRATEGY, demo=True, register=False)
    with registry_conn() as conn:
        repo = SqliteStrategyRepository(conn)
        assert STRATEGY not in {s.name for s in repo.list_strategies()}
        assert len(repo.list_runs(kind="backtest")) == 1


def test_walk_forward_records_oos_and_window_metrics() -> None:
    from typer.testing import CliRunner

    from algua.cli.main import app

    res = CliRunner().invoke(app, ["backtest", "walk-forward", STRATEGY, "--demo"])
    assert res.exit_code == 0, res.output
    with registry_conn() as conn:
        rows = SqliteStrategyRepository(conn).list_runs(kind="walk_forward")
    assert len(rows) == 1
    row = rows[0]
    assert row["sharpe_oos"] is not None
    assert row["n_obs_oos"] is not None
    assert row["mean_window_sharpe"] is not None
    # A walk-forward measures no full-period in-sample figure; it must not invent one.
    assert row["sharpe_is"] is None


def _run_sweep() -> None:
    from typer.testing import CliRunner

    from algua.cli.main import app

    res = CliRunner().invoke(
        app, ["backtest", "sweep", STRATEGY, "--demo", "--param", "lookback=20,40,60"])
    assert res.exit_code == 0, res.output


def test_sweep_records_a_parent_and_one_child_per_combo() -> None:
    import json

    _run_sweep()
    with registry_conn() as conn:
        repo = SqliteStrategyRepository(conn)
        parents = repo.list_runs(kind="sweep")
        children = repo.list_runs(kind="sweep_trial")
    assert len(parents) == 1
    assert len(children) == 3
    assert parents[0]["trials_truncated_at"] is None
    assert all(json.loads(c["derived_from"]) == [parents[0]["id"]] for c in children)
    assert all(c["mean_window_sharpe"] is not None for c in children)


def test_sweep_parent_does_not_conflate_trial_and_window_sharpe() -> None:
    """FIX 2: `mean_window_sharpe` on a sweep_trial/walk_forward row is the mean ACROSS
    WALK-FORWARD WINDOWS of ONE evaluation; the sweep parent's own cross-COMBO mean (`SweepResult
    .trial_sharpe_mean`) is a different statistic and must never share that column (spec §3.1 —
    a single sortable column must be comparable within itself)."""
    _run_sweep()
    with registry_conn() as conn:
        repo = SqliteStrategyRepository(conn)
        parent = repo.list_runs(kind="sweep")[0]
        extra = {r["key"]: r["value"] for r in conn.execute(
            "SELECT key, value FROM run_metrics WHERE run_id=?", (parent["id"],))}
    assert parent["mean_window_sharpe"] is None
    assert extra["mean_trial_sharpe"] is not None


def test_sweep_trial_count_matches_the_recorded_breadth() -> None:
    """The point of the slice: n_combos stops being an assertion and becomes a countable set."""
    _run_sweep()
    with registry_conn() as conn:
        repo = SqliteStrategyRepository(conn)
        n_children = len(repo.list_runs(kind="sweep_trial"))
        declared = repo.total_search_combos(STRATEGY)
    assert n_children == declared


def _ensure_family(strategy_name: str = STRATEGY, family_name: str = "csm_family") -> None:
    """Pre-assign a strategy to a family. Copied from tests/test_cli_research.py's helper of the
    same name (not imported: that module's fixtures/globals are not meant to travel) — family
    classification runs unconditionally in promotion_preflight, so every promote-driving test needs
    this regardless of which check is expected to fail the gate."""
    from datetime import UTC, datetime

    now = datetime.now(UTC).isoformat()
    with registry_conn() as conn:
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


def _promote(*, expect_pass: bool) -> None:
    """Drive one `research promote` attempt for STRATEGY to a completed gate decision, reusing the
    recipe from tests/test_cli_research.py::test_agent_promote_blocked_without_pit: register +
    backtest, assign a family, sweep for measured breadth, then promote as the DEFAULT agent actor
    with no `--allow-non-pit`/`--universe` — every current strategy fails the binding `pit_required`
    check under that recipe, which is exactly the "easy path" the task brief calls for. Only
    `expect_pass=False` is exercised by this module; a caller asking for a pass would need the
    human-actor signed-challenge dance (tests/_human_actor_helpers.py) this helper does not do.

    The final promote step calls `promote_run.promote_task` DIRECTLY rather than through the CLI
    (`typer`'s `research promote` command): that command is wrapped in `@json_errors`, a catch-all
    that turns ANY exception into a JSON error + exit(1) rather than letting it propagate. The
    rollback test needs a real exception to reach `pytest.raises`, and `promote_task` — the plain
    function `@json_errors` calls into — has no such wrapper of its own.

    Imported from `algua.registry.promote_run`, its canonical home since the stage-6a extraction
    (#591); `algua.cli.research_cmd` merely re-exports it.
    """
    from typer.testing import CliRunner

    from algua.cli.main import app
    from algua.registry.promote_run import promote_task

    if expect_pass:
        raise NotImplementedError(
            "no strategy in this suite passes the gate without a human-actor relaxation; "
            "this helper only drives the (task-relevant) failing path")
    runner = CliRunner()
    bt = runner.invoke(app, ["backtest", "run", STRATEGY, "--demo",
                              "--start", "2022-01-01", "--end", "2023-12-31", "--register"])
    assert bt.exit_code == 0, bt.stdout
    _ensure_family()
    sweep = runner.invoke(app, ["backtest", "sweep", STRATEGY, "--demo",
                                 "--start", "2022-01-01", "--end", "2023-12-31",
                                 "--param", "lookback=20,40", "--param", "construction.top_k=1,3"])
    assert sweep.exit_code == 0, sweep.stdout
    payload = promote_task(
        STRATEGY, start="2022-01-01", end="2023-12-31", demo=True,
        min_holdout_sharpe=-100, min_holdout_return=-100,
        min_pct_positive=0, min_window_sharpe=-100,
    )
    assert payload["passed"] is False


def test_failing_gate_still_records_a_run() -> None:
    """The rejections ARE the dataset — a gate that records nothing on failure is useless."""
    _promote(expect_pass=False)
    with registry_conn() as conn:
        rows = SqliteStrategyRepository(conn).list_runs(kind="gate")
    assert len(rows) == 1
    assert rows[0]["passed"] == 0
    assert rows[0]["sharpe_oos"] is not None


def test_gate_run_links_to_its_walk_forward_run() -> None:
    """derived_from is what makes the IS-vs-OOS scatter joinable at all."""
    import json

    _promote(expect_pass=False)
    with registry_conn() as conn:
        repo = SqliteStrategyRepository(conn)
        gate = repo.list_runs(kind="gate")[0]
        wf = repo.list_runs(kind="walk_forward")[0]
    assert json.loads(gate["derived_from"]) == [wf["id"]]


def test_gate_run_names_its_own_gate_evaluations_row() -> None:
    """FIX 1: a `gate` run must be able to NAME its own gate_evaluations row — the join
    `runs show` / the gate bullet card (spec §5, §6 view 5) need to reach decision_json (the 11
    gate checks, per-regime Sharpes) that the run row's own fixed scalar columns deliberately do
    not carry. Exercised on the FAILING path (like the sibling test above) — a failing gate still
    writes both rows in the same transaction."""
    _promote(expect_pass=False)
    with registry_conn() as conn:
        repo = SqliteStrategyRepository(conn)
        gate_run = repo.list_runs(kind="gate")[0]
        ge = conn.execute("SELECT id FROM gate_evaluations").fetchone()
    assert ge is not None
    assert gate_run["gate_id"] == ge["id"]


def test_a_rolled_back_gate_leaves_no_phantom_run() -> None:
    """The run row shares the gate's transaction; if the gate rolls back, so must the run.

    `GateLedgerMixin` has no `_commit_gate_hook` seam (the brief's first-choice patch target does
    not exist in the real code), so this uses the brief's stated alternative: force
    `_insert_run_locked` to blow up INSIDE the gate's open transaction and assert that
    `gate_evaluations` is ALSO empty afterwards. That only holds if the run insert and the gate
    insert share one transaction — proving constraint 2 from the other direction.

    Patched on the FACADE (`SqliteStrategyRepository`), not `GateLedgerMixin`: `_insert_run_locked`
    is defined on the sibling `RunLedgerMixin` and only reachable through the facade's MRO — the
    gate mixin alone does not have the attribute (mirrors the composition note atop gate.py's
    TYPE_CHECKING block).

    The patch only raises for `kind="gate"`, not unconditionally: `_promote`'s setup (the
    `backtest run` CLI step, then the `walk_forward` run recorded at TOP LEVEL before the gate
    transaction opens) also goes through `_insert_run_locked` and must keep succeeding, or the
    CLI's catch-all error handling would swallow the injected failure long before it ever reaches
    the gate's own transaction and `pytest.raises` would never see it.
    """
    import sqlite3
    from unittest.mock import patch

    from algua.registry.store import SqliteStrategyRepository
    from algua.registry.store.runs import RunLedgerMixin

    _original_insert = RunLedgerMixin._insert_run_locked

    def _boom_only_for_gate(self: SqliteStrategyRepository, kind: str, strategy_name: str,
                             **kwargs: object) -> int:
        if kind == "gate":
            raise sqlite3.OperationalError("boom")
        return _original_insert(self, kind, strategy_name, **kwargs)

    with patch.object(SqliteStrategyRepository, "_insert_run_locked", _boom_only_for_gate):
        with pytest.raises(sqlite3.OperationalError):
            _promote(expect_pass=False)
    with registry_conn() as conn:
        repo = SqliteStrategyRepository(conn)
        assert repo.list_runs(kind="gate") == []
        gate_evaluations = conn.execute("SELECT * FROM gate_evaluations").fetchall()
        assert gate_evaluations == []
