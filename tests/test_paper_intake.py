"""Tests for `paper intake` (#317): deterministic candidate → paper book admission.

Three required behaviours:
  1. Empty book, headroom for all: every candidate is admitted, equal slice, `queued` empty; each
     admitted strategy is now Stage.PAPER with a non-None active allocation.
  2. The --max-concurrent count cap binds: exactly one candidate is admitted (FIFO, lower sid), the
     other stays queued and remains Stage.CANDIDATE with no allocation.
  3. An already-occupied slot counts against the cap: with the sole slot already taken by an
     allocated paper-lane strategy, the queued candidate is not admitted.
"""
from __future__ import annotations

import json
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

import algua.strategies.momentum as _momentum_pkg
from algua.cli.main import app
from algua.config.settings import get_settings
from algua.execution.alpaca_broker import AccountState
from algua.registry.allocations import active_allocation
from algua.registry.db import connect, migrate
from algua.registry.store import SqliteStrategyRepository

runner = CliRunner()

# _S1 is registered first → lower DB id → FIFO tie-break admits it before _S2.
_S1 = "cross_sectional_momentum"
_S2 = "liquid10_momentum"


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "p.db"))
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ALGUA_ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALGUA_ALPACA_API_SECRET", "s")


@pytest.fixture(autouse=True)
def _second_strategy():
    """`_S2` is a second REAL, loadable, demo-backtestable strategy in the momentum family so
    `_index()` discovers it (mirrors tests/test_paper_run_all._second_strategy)."""
    p = Path(_momentum_pkg.__path__[0]) / f"{_S2}.py"
    p.write_text(
        '"""Second demo strategy for paper intake tests: trailing-return momentum, top-k."""\n'
        "from __future__ import annotations\n"
        "from typing import Any\n"
        "import pandas as pd\n"
        "from algua.contracts.types import ExecutionContract\n"
        "from algua.features.alphas import xs_trailing_return\n"
        "from algua.strategies.base import StrategyConfig\n"
        f"CONFIG = StrategyConfig(name={_S2!r},\n"
        "    universe=['AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL'],\n"
        "    execution=ExecutionContract(rebalance_frequency='1d', decision_lag_bars=1),\n"
        "    params={'lookback': 60}, construction='top_k_equal_weight',\n"
        "    construction_params={'top_k': 3}, feature_lookback=60)\n"
        "def signal(view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:\n"
        "    return xs_trailing_return(view, params)\n"
    )
    try:
        yield
    finally:
        p.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeBroker:
    """Minimal paper broker: `intake` reads only `.account().equity` (READ-ONLY, no trading)."""

    def __init__(self, equity: float) -> None:
        self._equity = equity

    def account(self) -> AccountState:
        return AccountState(equity=self._equity, cash=self._equity,
                            buying_power=self._equity, account_id="t")


def _to_candidate(name: str) -> None:
    """Register a real strategy via a demo backtest, then transition it to `candidate` (human
    bypasses the shortlist gate — test setup only)."""
    assert runner.invoke(app, ["backtest", "run", name, "--demo", "--register",
                               "--start", "2022-01-01", "--end", "2023-12-31"]).exit_code == 0
    assert runner.invoke(app, ["registry", "transition", name, "--to", "candidate",
                               "--actor", "human", "--reason", "ok"]).exit_code == 0


def _force_stage(name: str, stage_value: str) -> None:
    """Force a strategy's lifecycle stage directly (bypasses the promote gate for test setup)."""
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        rec = SqliteStrategyRepository(conn).get(name)
        conn.execute("UPDATE strategies SET stage = ? WHERE id = ?", (stage_value, rec.id))
        conn.commit()


def _seed_allocation(name: str, capital: float = 10_000.0) -> None:
    """Insert a strategy_allocations row directly (no paper-allocate CLI dependency)."""
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        rec = SqliteStrategyRepository(conn).get(name)
        conn.execute(
            "INSERT INTO strategy_allocations(strategy_id, capital, effective_ts, actor) "
            "VALUES (?,?,?,?)",
            (rec.id, capital, datetime.now(UTC).isoformat(), "agent"),
        )
        conn.commit()


def _stage_of(name: str):
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        return SqliteStrategyRepository(conn).get(name).stage


def _has_allocation(name: str) -> bool:
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        rec = SqliteStrategyRepository(conn).get(name)
        return active_allocation(conn, rec.id) is not None


# ---------------------------------------------------------------------------
# Test 1: empty book, headroom for all candidates
# ---------------------------------------------------------------------------

def test_intake_admits_all_candidates_when_book_empty(monkeypatch):
    """Two candidates, empty book, cap 5, equity 100k → BOTH admitted with an equal 20k slice,
    `queued` empty; both are now Stage.PAPER and each carries an active allocation."""
    _to_candidate(_S1)
    _to_candidate(_S2)

    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings",
                        lambda: _FakeBroker(100_000.0))

    result = runner.invoke(app, ["paper", "intake", "--max-concurrent", "5"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload.get("ok") is True

    admitted = {a["strategy"]: a["capital"] for a in payload["admitted"]}
    assert admitted == {_S1: 20_000.0, _S2: 20_000.0}
    assert payload["queued"] == []
    assert payload["slice"] == 20_000.0
    assert payload["occupied_before"] == 0
    assert payload["equity"] == 100_000.0

    from algua.contracts.lifecycle import Stage
    for name in (_S1, _S2):
        assert _stage_of(name) is Stage.PAPER
        assert _has_allocation(name)


# ---------------------------------------------------------------------------
# Test 2: the concurrency cap binds — only the FIFO-first candidate is admitted
# ---------------------------------------------------------------------------

def test_intake_cap_admits_only_first_candidate(monkeypatch):
    """cap 1, two candidates → exactly ONE admitted (the earlier-registered / lower-sid _S1); the
    other stays queued AND remains Stage.CANDIDATE with no allocation."""
    _to_candidate(_S1)
    _to_candidate(_S2)

    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings",
                        lambda: _FakeBroker(100_000.0))

    result = runner.invoke(app, ["paper", "intake", "--max-concurrent", "1"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)

    assert [a["strategy"] for a in payload["admitted"]] == [_S1]
    assert payload["queued"] == [_S2]

    from algua.contracts.lifecycle import Stage
    assert _stage_of(_S1) is Stage.PAPER
    assert _has_allocation(_S1)
    # The un-admitted candidate is untouched: still candidate, still unallocated.
    assert _stage_of(_S2) is Stage.CANDIDATE
    assert not _has_allocation(_S2)


# ---------------------------------------------------------------------------
# Test 3: an already-occupied slot counts against the cap
# ---------------------------------------------------------------------------

def test_intake_occupied_slot_blocks_admission(monkeypatch):
    """The sole slot (cap 1) is already taken by an allocated paper-lane strategy → the queued
    candidate is NOT admitted; it stays Stage.CANDIDATE with no allocation."""
    # _S2 is an already-admitted paper-lane tenant (forced stage + a seeded allocation row).
    _to_candidate(_S2)
    _force_stage(_S2, "paper")
    _seed_allocation(_S2)
    # _S1 is the queued candidate.
    _to_candidate(_S1)

    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings",
                        lambda: _FakeBroker(100_000.0))

    result = runner.invoke(app, ["paper", "intake", "--max-concurrent", "1"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)

    assert payload["admitted"] == []
    assert payload["queued"] == [_S1]
    assert payload["occupied_before"] == 1

    from algua.contracts.lifecycle import Stage
    assert _stage_of(_S1) is Stage.CANDIDATE
    assert not _has_allocation(_S1)


# ---------------------------------------------------------------------------
# The atomic admit primitive directly (findings #1/#3/#4)
# ---------------------------------------------------------------------------

def test_intake_reports_empty_stale_bucket_on_clean_run(monkeypatch):
    """`skipped_stale` is always present in the envelope (empty on a race-free run)."""
    _to_candidate(_S1)
    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings",
                        lambda: _FakeBroker(100_000.0))
    payload = json.loads(runner.invoke(app, ["paper", "intake"]).output)
    assert payload["skipped_stale"] == []
    assert [a["strategy"] for a in payload["admitted"]] == [_S1]


def test_primitive_rejects_non_candidate():
    """`intake_candidate_to_paper` fails closed (TransitionError) on a non-candidate stage — the
    'stale selection' signal the intake loop treats as skipped_stale."""
    from algua.contracts.lifecycle import Actor, TransitionError
    _to_candidate(_S1)
    _force_stage(_S1, "paper")  # no longer a candidate
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        repo = SqliteStrategyRepository(conn)
        with pytest.raises(TransitionError):
            repo.intake_candidate_to_paper(
                repo.get(_S1), capital=10_000.0, actor=Actor.AGENT,
                account_equity=100_000.0, max_concurrent=5)


def test_primitive_count_cap_is_atomic_and_rolls_back():
    """At the count cap the primitive raises CountCapReached and leaves the candidate exactly
    candidate with NO allocation (the allocation insert is rolled back with the failed txn)."""
    from algua.contracts.lifecycle import Actor, Stage
    from algua.registry.allocations import CountCapReached
    # _S2 occupies the sole slot (allocated paper tenant); _S1 is the queued candidate.
    _to_candidate(_S2)
    _force_stage(_S2, "paper")
    _seed_allocation(_S2)
    _to_candidate(_S1)
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        repo = SqliteStrategyRepository(conn)
        with pytest.raises(CountCapReached):
            repo.intake_candidate_to_paper(
                repo.get(_S1), capital=10_000.0, actor=Actor.AGENT,
                account_equity=100_000.0, max_concurrent=1)
        assert repo.get(_S1).stage is Stage.CANDIDATE
        assert active_allocation(conn, repo.get(_S1).id) is None


def test_primitive_capital_bound_rolls_back():
    """When the slice would breach Σ ≤ equity the primitive raises AllocationError and leaves the
    strategy candidate + unallocated (atomic rollback of the whole admit)."""
    from algua.contracts.lifecycle import Actor, Stage
    from algua.registry.allocations import AllocationError
    _to_candidate(_S1)
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        repo = SqliteStrategyRepository(conn)
        with pytest.raises(AllocationError):
            repo.intake_candidate_to_paper(
                repo.get(_S1), capital=200_000.0, actor=Actor.AGENT,
                account_equity=100_000.0, max_concurrent=5)
        assert repo.get(_S1).stage is Stage.CANDIDATE
        assert active_allocation(conn, repo.get(_S1).id) is None


# ---------------------------------------------------------------------------
# `paper allocate` (#497): the lane-scoped re-admission path for recovery/demotion
# re-entrants (dormant→paper, live→paper land UNALLOCATED) and manual paper-book resizes.
# ---------------------------------------------------------------------------

def _seed_paper_allocation(name: str, capital: float = 10_000.0) -> None:
    """Force `name` to stage paper and give it an active allocation via the shared
    ``allocate_locked`` body (caller-owns-txn) wrapped in a `with conn:` commit — so it counts as
    an active paper-lane tenant against the concurrency cap."""
    from algua.registry.allocations import allocate_locked
    _force_stage(name, "paper")
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        rec = SqliteStrategyRepository(conn).get(name)
        with conn:
            allocate_locked(conn, rec.id, capital, "agent", 100_000.0)


def _capital_of(name: str) -> float:
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        rec = SqliteStrategyRepository(conn).get(name)
        row = active_allocation(conn, rec.id)
        assert row is not None
        return float(row["capital"])


def test_paper_allocate_sets_then_resizes_paper_stage(monkeypatch):
    """`paper allocate` on a paper-stage, unallocated strategy sets the capital base and emits
    prior_capital == 0.0; a second allocate RESIZES it and emits the prior capital."""
    _to_candidate(_S1)
    _force_stage(_S1, "paper")
    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings",
                        lambda: _FakeBroker(100_000.0))

    r1 = runner.invoke(app, ["paper", "allocate", _S1, "--capital", "10000"])
    assert r1.exit_code == 0, r1.output
    p1 = json.loads(r1.output)
    assert p1["ok"] is True
    assert p1["strategy"] == _S1
    assert p1["capital"] == 10_000.0
    assert p1["prior_capital"] == 0.0
    assert _has_allocation(_S1)
    assert _capital_of(_S1) == 10_000.0

    r2 = runner.invoke(app, ["paper", "allocate", _S1, "--capital", "20000"])
    assert r2.exit_code == 0, r2.output
    p2 = json.loads(r2.output)
    assert p2["capital"] == 20_000.0
    assert p2["prior_capital"] == 10_000.0
    assert _capital_of(_S1) == 20_000.0


def test_paper_allocate_rejected_on_candidate_stage(monkeypatch):
    """A candidate-stage strategy has no paper re-admission path here — it enters only via
    `paper intake`. `paper allocate` fails closed (exit != 0), message names the stage."""
    _to_candidate(_S1)  # stays candidate
    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings",
                        lambda: _FakeBroker(100_000.0))
    result = runner.invoke(app, ["paper", "allocate", _S1, "--capital", "10000"])
    assert result.exit_code != 0
    payload = json.loads(result.output)
    assert payload["ok"] is False
    assert "candidate" in payload["error"]
    assert not _has_allocation(_S1)


def test_paper_allocate_rejected_on_live_stage(monkeypatch):
    """A live-stage strategy is out of the paper lane — `live allocate` owns it. `paper allocate`
    fails closed (exit != 0), message names the live stage."""
    _to_candidate(_S1)
    _force_stage(_S1, "live")
    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings",
                        lambda: _FakeBroker(100_000.0))
    result = runner.invoke(app, ["paper", "allocate", _S1, "--capital", "10000"])
    assert result.exit_code != 0
    payload = json.loads(result.output)
    assert payload["ok"] is False
    assert "live" in payload["error"]
    assert not _has_allocation(_S1)


def test_paper_allocate_count_cap_blocks_new_tenant_but_allows_resize(monkeypatch):
    """At the max-concurrent cap, a count-INCREASING allocation (a currently-unallocated paper
    strategy) is refused (CountCapReached surfaced, exit != 0), while RESIZING an already-allocated
    tenant at the cap succeeds (it admits no new tenant)."""
    # _S2 occupies the sole slot as an allocated paper tenant; _S1 is a paper strategy with no
    # active allocation yet.
    _to_candidate(_S2)
    _seed_paper_allocation(_S2, capital=10_000.0)
    _to_candidate(_S1)
    _force_stage(_S1, "paper")
    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings",
                        lambda: _FakeBroker(100_000.0))

    blocked = runner.invoke(app, ["paper", "allocate", _S1, "--capital", "10000",
                                  "--max-concurrent", "1"])
    assert blocked.exit_code != 0
    blocked_payload = json.loads(blocked.output)
    assert blocked_payload["ok"] is False
    assert "capacity" in blocked_payload["error"]
    assert not _has_allocation(_S1)

    # Resizing the already-allocated tenant at the same cap succeeds (no new tenant admitted).
    resized = runner.invoke(app, ["paper", "allocate", _S2, "--capital", "20000",
                                  "--max-concurrent", "1"])
    assert resized.exit_code == 0, resized.output
    assert json.loads(resized.output)["prior_capital"] == 10_000.0
    assert _capital_of(_S2) == 20_000.0
