"""Unit tests for the merge-back authoritative-intake chokepoint helpers
(:mod:`algua.registry.mergeback_intake`): the one-tx ``ensure_backtested`` registration and the
marker-driven, crash-idempotent ``produce_evidence`` recipe. Hermetic: a tmp registry DB, fake
sweep/backtest task callables, and a stub LoadedStrategy — no engine, no CLI, no git.
"""

from __future__ import annotations

import json
from contextlib import contextmanager

import pandas as pd
import pytest

from algua.contracts.types import ExecutionContract
from algua.portfolio.construction import get_construction_policy
from algua.registry.db import connect, migrate
from algua.registry.mergeback_intake import (
    INTAKE_TAG,
    MergeBackIntakeError,
    ensure_backtested,
    evidence_marker,
    produce_evidence,
)
from algua.registry.store import SqliteStrategyRepository
from algua.strategies.base import LoadedStrategy, StrategyConfig

_STRAT = "factory_momo"
_TIP = "BRANCHTIP0"
_GRID = ["lookback=10,20"]
_CANONICAL = json.dumps({"lookback": [10, 20]}, sort_keys=True)
# The full transported data context bound into the marker's recipe hash (GATE-2 #4).
_CTX = {"demo": True, "snapshot": None, "fundamentals_snapshot": None, "news_snapshot": None,
        "delistings": None, "rank_by": "mean_sharpe", "universe": "sp500",
        "start": "2024-01-01", "end": "2024-06-01"}


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "registry.db"


@pytest.fixture
def conn_factory(db_path):
    @contextmanager
    def factory():
        conn = connect(db_path)
        try:
            migrate(conn)
            yield conn
        finally:
            conn.close()

    return factory


def _ensure(conn_factory, **over):
    kwargs = dict(strategy=_STRAT, branch="research-run/1", branch_tip=_TIP,
                  merge_sha="MERGE0", base_sha="BASE0")
    kwargs.update(over)
    with conn_factory() as conn:
        return ensure_backtested(conn, **kwargs)


def _stub_strategy(**execution_kw) -> LoadedStrategy:
    cfg = StrategyConfig(
        name=_STRAT,
        universe=["AAA", "BBB"],
        execution=ExecutionContract(rebalance_frequency="1d", **execution_kw),
        params={"lookback": 10},
        construction="equal_weight_positive",
    )
    return LoadedStrategy(
        config=cfg,
        signal_fn=lambda v, p: pd.Series(dtype="float64"),
        construct_fn=get_construction_policy(cfg.construction),
    )


class _Task:
    """A fake sweep/backtest task callable that records invocations and can mimic the REAL task's
    autocommitted evidence insert (a search_trials row / a backtest_returns row) before optionally
    raising — so every crash point between the task's own commit and the marker flip is
    modelable."""

    def __init__(self, conn_factory, *, record: str | None = None,
                 raises: BaseException | None = None) -> None:
        self.conn_factory = conn_factory
        self.record = record
        self.raises = raises
        self.calls = 0

    def __call__(self) -> dict:
        self.calls += 1
        if self.record == "trial":
            with self.conn_factory() as conn:
                SqliteStrategyRepository(conn).record_search_trial(_STRAT, 2, _CANONICAL)
        elif self.record == "returns":
            with self.conn_factory() as conn:
                SqliteStrategyRepository(conn).persist_backtest_returns(
                    _STRAT, "2024-01-01", "2024-06-01",
                    pd.Series([0.01, -0.02], index=pd.to_datetime(["2024-01-02", "2024-01-03"])))
        if self.raises is not None:
            raise self.raises
        return {"ok": True}


def _produce(conn_factory, *, ensure_status="created", sweep_params=_GRID, sweep=None,
             backtest=None, strategy_stub=None, eval_context=_CTX, produce_conn_factory=None):
    sweep = sweep if sweep is not None else _Task(conn_factory, record="trial")
    backtest = backtest if backtest is not None else _Task(conn_factory, record="returns")
    status = produce_evidence(
        strategy=_STRAT, branch_tip=_TIP, ensure_status=ensure_status,
        sweep_params=sweep_params, eval_context=eval_context,
        conn_factory=produce_conn_factory if produce_conn_factory is not None else conn_factory,
        sweep_fn=sweep, backtest_fn=backtest,
        load_strategy_fn=lambda name: (strategy_stub or _stub_strategy()),
    )
    return status, sweep, backtest


def _marker(conn_factory):
    with conn_factory() as conn:
        rec = SqliteStrategyRepository(conn).get(_STRAT)
        row = evidence_marker(conn, rec.id, _TIP)
        return dict(row) if row is not None else None


def _trial_count(conn_factory):
    with conn_factory() as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM search_trials WHERE strategy_name=?", (_STRAT,)).fetchone()[0]


# ------------------------------------------------------------------------- ensure_backtested


def test_ensure_creates_row_at_backtested_with_provenance(conn_factory):
    assert _ensure(conn_factory) == "created"
    with conn_factory() as conn:
        repo = SqliteStrategyRepository(conn)
        rec = repo.get(_STRAT)
        assert rec.stage.value == "backtested"
        assert INTAKE_TAG in rec.tags
        transitions = repo.list_transitions(_STRAT)
    # created (system) + idea->backtested (agent) landed together.
    assert [(t["from_stage"], t["to_stage"], t["actor"]) for t in transitions] == [
        (None, "idea", "system"), ("idea", "backtested", "agent")]
    reason = transitions[-1]["reason"]
    for token in (INTAKE_TAG, "research-run/1", _TIP, "MERGE0", "BASE0"):
        assert token in reason


def test_ensure_is_idempotent_and_reports_existed(conn_factory):
    assert _ensure(conn_factory) == "created"
    assert _ensure(conn_factory) == "existed"
    with conn_factory() as conn:
        assert len(SqliteStrategyRepository(conn).list_transitions(_STRAT)) == 2  # no new rows


def test_ensure_advances_a_preexisting_idea_row(conn_factory):
    with conn_factory() as conn:
        SqliteStrategyRepository(conn).add(_STRAT)
    assert _ensure(conn_factory) == "existed"
    with conn_factory() as conn:
        repo = SqliteStrategyRepository(conn)
        assert repo.get(_STRAT).stage.value == "backtested"
        transitions = repo.list_transitions(_STRAT)
    assert transitions[-1]["to_stage"] == "backtested"
    assert INTAKE_TAG in (transitions[-1]["reason"] or "")


@pytest.mark.parametrize("stage", ["candidate", "paper", "forward_tested", "live", "retired",
                                   "dormant"])
def test_ensure_fails_closed_on_any_other_stage(conn_factory, stage):
    with conn_factory() as conn:
        repo = SqliteStrategyRepository(conn)
        repo.add(_STRAT)
        rec = repo.get(_STRAT)
        conn.execute("UPDATE strategies SET stage=? WHERE id=?", (stage, rec.id))
        conn.commit()
    with pytest.raises(MergeBackIntakeError, match="fail closed"):
        _ensure(conn_factory)
    with conn_factory() as conn:
        repo = SqliteStrategyRepository(conn)
        assert repo.get(_STRAT).stage.value == stage  # rolled back, untouched
        assert len(repo.list_transitions(_STRAT)) == 1


def test_ensure_refuses_to_run_inside_an_open_transaction(conn_factory):
    with conn_factory() as conn:
        conn.execute("BEGIN IMMEDIATE")
        with pytest.raises(RuntimeError, match="top level"):
            ensure_backtested(conn, strategy=_STRAT, branch="b", branch_tip=_TIP,
                              merge_sha="M", base_sha="B")
        conn.rollback()


# ------------------------------------------------------------------------- produce_evidence


def test_produce_fresh_records_evidence_and_completes_marker(conn_factory):
    _ensure(conn_factory)
    status, sweep, backtest = _produce(conn_factory)
    assert status == "produced"
    assert sweep.calls == 1 and backtest.calls == 1
    marker = _marker(conn_factory)
    assert marker["status"] == "completed" and marker["completed_at"] is not None
    assert _trial_count(conn_factory) == 1
    with conn_factory() as conn:
        assert SqliteStrategyRepository(conn).load_backtest_returns(_STRAT) is not None


def test_produce_completed_marker_blocks_any_rerecord(conn_factory):
    _ensure(conn_factory)
    _produce(conn_factory)
    status, sweep, backtest = _produce(conn_factory, ensure_status="existed")
    assert status == "already_produced"
    assert sweep.calls == 0 and backtest.calls == 0
    assert _trial_count(conn_factory) == 1  # breadth NOT double-counted


def test_produce_skips_preexisting_strategy_with_authoritative_breadth(conn_factory):
    # The direct-authoritative-funnel no-op: a pre-existing strategy that already carries measured
    # breadth AND a persisted returns series is never re-swept — and no marker is minted for it.
    with conn_factory() as conn:
        repo = SqliteStrategyRepository(conn)
        repo.add(_STRAT)
        repo.record_search_trial(_STRAT, 7, json.dumps({"other": [1]}))
        repo.persist_backtest_returns(
            _STRAT, "2024-01-01", "2024-06-01",
            pd.Series([0.01, -0.02], index=pd.to_datetime(["2024-01-02", "2024-01-03"])))
    _ensure(conn_factory)
    status, sweep, backtest = _produce(conn_factory, ensure_status="existed")
    assert status == "authoritative_breadth"
    assert sweep.calls == 0 and backtest.calls == 0
    assert _marker(conn_factory) is None
    assert _trial_count(conn_factory) == 1


def test_produce_backfills_missing_returns_when_breadth_exists(conn_factory):
    # GATE-2 #2: authoritative breadth proves search_trials, NOT the classifier's
    # return-correlation axis. Missing backtest_returns -> ONLY the backtest runs (never a
    # re-sweep, never a marker), and the cost floor is asserted first.
    with conn_factory() as conn:
        repo = SqliteStrategyRepository(conn)
        repo.add(_STRAT)
        repo.record_search_trial(_STRAT, 7, json.dumps({"other": [1]}))
    _ensure(conn_factory)
    status, sweep, backtest = _produce(conn_factory, ensure_status="existed")
    assert status == "authoritative_breadth_returns_backfilled"
    assert sweep.calls == 0 and backtest.calls == 1
    assert _marker(conn_factory) is None
    assert _trial_count(conn_factory) == 1  # breadth untouched
    with conn_factory() as conn:
        assert SqliteStrategyRepository(conn).load_backtest_returns(_STRAT) is not None
    # Re-run: returns now exist -> pure skip.
    status2, sweep2, backtest2 = _produce(conn_factory, ensure_status="existed")
    assert status2 == "authoritative_breadth"
    assert sweep2.calls == 0 and backtest2.calls == 0


def test_returns_backfill_asserts_the_cost_floor_before_persisting(conn_factory):
    with conn_factory() as conn:
        repo = SqliteStrategyRepository(conn)
        repo.add(_STRAT)
        repo.record_search_trial(_STRAT, 7, json.dumps({"other": [1]}))
    _ensure(conn_factory)
    with pytest.raises(ValueError, match="fees"):
        _produce(conn_factory, ensure_status="existed",
                 strategy_stub=_stub_strategy(fees=0.0, slippage=0.0))
    with conn_factory() as conn:
        assert SqliteStrategyRepository(conn).load_backtest_returns(_STRAT) is None


def test_produce_runs_for_a_created_row_even_with_stale_same_name_breadth(conn_factory):
    # Stale breadth under the SAME NAME (a long-gone strategy) never masks a freshly CREATED row's
    # evidence production — the skip predicate requires ensure_status == "existed".
    with conn_factory() as conn:
        SqliteStrategyRepository(conn).record_search_trial(_STRAT, 3, json.dumps({"old": [1]}))
    _ensure(conn_factory)
    status, sweep, _ = _produce(conn_factory, ensure_status="created")
    assert status == "produced"
    assert sweep.calls == 1


def test_produce_without_grid_is_no_context_and_records_nothing(conn_factory):
    _ensure(conn_factory)
    status, sweep, backtest = _produce(conn_factory, sweep_params=None)
    assert status == "no_context"
    assert sweep.calls == 0 and backtest.calls == 0
    assert _marker(conn_factory) is None


def test_crash_before_any_recording_leaves_clean_rerun(conn_factory):
    # Crash mid-compute BEFORE the sweep task's own insert: marker stays 'started', nothing is
    # recorded, and the re-run runs the full recipe cleanly.
    _ensure(conn_factory)
    boom = _Task(conn_factory, raises=RuntimeError("sweep crashed"))
    with pytest.raises(RuntimeError, match="sweep crashed"):
        _produce(conn_factory, sweep=boom)
    assert _marker(conn_factory)["status"] == "started"
    assert _trial_count(conn_factory) == 0

    status, sweep, backtest = _produce(conn_factory)
    assert status == "produced"
    assert sweep.calls == 1 and backtest.calls == 1
    assert _trial_count(conn_factory) == 1
    assert _marker(conn_factory)["status"] == "completed"


def test_crash_after_trial_recorded_dedups_the_trial_layer_on_resume(conn_factory):
    # Crash AFTER the sweep task autocommitted its trial row but BEFORE the marker flip (here: the
    # backtest task blows up). The resume must NOT re-sweep — duplicate trials permanently inflate
    # funnel/window breadth and the agent-NOVEL lifetime seed.
    _ensure(conn_factory)
    sweep1 = _Task(conn_factory, record="trial")
    backtest_boom = _Task(conn_factory, raises=RuntimeError("backtest crashed"))
    with pytest.raises(RuntimeError, match="backtest crashed"):
        _produce(conn_factory, sweep=sweep1, backtest=backtest_boom)
    assert sweep1.calls == 1
    assert _trial_count(conn_factory) == 1
    assert _marker(conn_factory)["status"] == "started"

    status, sweep2, backtest2 = _produce(conn_factory)
    assert status == "produced"
    assert sweep2.calls == 0            # deduped on (strategy, grid_json, id > watermark)
    assert backtest2.calls == 1
    assert _trial_count(conn_factory) == 1  # exactly one breadth row, ever
    assert _marker(conn_factory)["status"] == "completed"


def test_resume_with_a_different_grid_fails_closed(conn_factory):
    _ensure(conn_factory)
    with pytest.raises(RuntimeError):
        _produce(conn_factory, sweep=_Task(conn_factory, raises=RuntimeError("crash")))
    with pytest.raises(MergeBackIntakeError, match="inconsistent resume"):
        _produce(conn_factory, sweep_params=["lookback=30,40"])


def test_resume_without_the_grid_that_started_it_fails_closed(conn_factory):
    _ensure(conn_factory)
    with pytest.raises(RuntimeError):
        _produce(conn_factory, sweep=_Task(conn_factory, raises=RuntimeError("crash")))
    with pytest.raises(MergeBackIntakeError, match="transported none"):
        _produce(conn_factory, sweep_params=None)


def test_produce_asserts_the_agent_cost_floor_before_anything_persists(conn_factory):
    _ensure(conn_factory)
    frictionless = _stub_strategy(fees=0.0, slippage=0.0)
    with pytest.raises(ValueError, match="fees"):
        _produce(conn_factory, strategy_stub=frictionless)
    assert _marker(conn_factory) is None
    assert _trial_count(conn_factory) == 0


def test_produce_validates_grid_keys_against_the_merged_module(conn_factory):
    _ensure(conn_factory)
    with pytest.raises(ValueError, match="not a base signal param"):
        _produce(conn_factory, sweep_params=["not_a_param=1,2"])
    assert _marker(conn_factory) is None


def test_fresh_marker_always_sweeps_despite_concurrent_same_grid_trial(conn_factory, db_path):
    # GATE-2 #3: the watermark dedup is a RESUME-only device. On a marker NEWLY created by this
    # call, a concurrent/manual same-grid trial landing AFTER the watermark must NOT be
    # misattributed to this attempt — the authoritative sweep still runs. Simulated with a racing
    # conn_factory that injects a same-grid trial row immediately after the marker's creation
    # connection closes (id > watermark by construction).
    _ensure(conn_factory)
    injected = {"done": False}

    @contextmanager
    def racing_factory():
        with conn_factory() as conn:
            yield conn
        if not injected["done"]:
            with conn_factory() as probe:
                rec = SqliteStrategyRepository(probe).get(_STRAT)
                if evidence_marker(probe, rec.id, _TIP) is not None:
                    SqliteStrategyRepository(probe).record_search_trial(_STRAT, 2, _CANONICAL)
                    injected["done"] = True

    status, sweep, backtest = _produce(conn_factory, produce_conn_factory=racing_factory)
    assert status == "produced"
    assert injected["done"] is True     # the concurrent same-grid trial DID land after the marker
    assert sweep.calls == 1             # ...and the authoritative sweep still ran (no dedup skip)
    assert backtest.calls == 1
    assert _trial_count(conn_factory) == 2  # concurrent row + this attempt's own row


def test_resume_with_same_grid_but_different_context_fails_closed(conn_factory):
    # GATE-2 #4: the marker binds the FULL recipe. A resume transporting the SAME grid but a
    # DIFFERENT data context (here: another end date) must fail closed, never silently reuse the
    # marker — evidence produced against other data is not this attempt's evidence.
    _ensure(conn_factory)
    with pytest.raises(RuntimeError):
        _produce(conn_factory, sweep=_Task(conn_factory, raises=RuntimeError("crash")))
    drifted = dict(_CTX, end="2024-12-31")
    with pytest.raises(MergeBackIntakeError, match="data context drifted"):
        _produce(conn_factory, eval_context=drifted)
    # The matching recipe still resumes fine.
    status, _, _ = _produce(conn_factory)
    assert status == "produced"


def test_completed_marker_with_drifted_context_fails_closed_not_already_produced(conn_factory):
    _ensure(conn_factory)
    _produce(conn_factory)
    assert _marker(conn_factory)["status"] == "completed"
    drifted = dict(_CTX, snapshot="bars-other", demo=False)
    with pytest.raises(MergeBackIntakeError, match="data context drifted"):
        _produce(conn_factory, ensure_status="existed", eval_context=drifted)
