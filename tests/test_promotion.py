from datetime import UTC, date, datetime
from pathlib import Path

import pytest

import algua.strategies.momentum as momentum_pkg
from algua.backtest._sample import SyntheticProvider
from algua.backtest.engine import BacktestError
from algua.contracts.lifecycle import Actor, Stage, TransitionError
from algua.registry.db import connect, migrate
from algua.registry.promotion import (
    BreadthContext,
    guard_agent_relaxations,
    promotion_preflight,
    resolve_pit_ok,
    run_gate,
)
from algua.registry.store import SqliteStrategyRepository
from algua.research.gates import (
    FUNNEL_WINDOW_DAYS,
    GateCriteria,
    effective_funnel_breadth,
)

_START = datetime(2024, 1, 1, tzinfo=UTC)
_END = datetime(2024, 6, 1, tzinfo=UTC)


def _repo(tmp_path):
    conn = connect(tmp_path / "p.db")
    migrate(conn)
    return SqliteStrategyRepository(conn)


def test_guard_agent_relaxations_refuses_agent():
    for kw in (dict(declared_combos=9, allow_holdout_reuse=False, allow_non_pit=False),
               dict(declared_combos=None, allow_holdout_reuse=True, allow_non_pit=False),
               dict(declared_combos=None, allow_holdout_reuse=False, allow_non_pit=True)):
        with pytest.raises(ValueError, match="human"):
            guard_agent_relaxations(Actor.AGENT, **kw)


def test_guard_allows_clean_agent_and_any_human():
    guard_agent_relaxations(Actor.AGENT, declared_combos=None, allow_holdout_reuse=False,
                            allow_non_pit=False)
    guard_agent_relaxations(Actor.HUMAN, declared_combos=9, allow_holdout_reuse=True,
                            allow_non_pit=True)


def test_resolve_pit_ok_requires_coverage():
    cover = [{"snapshot_id": "u1", "effective_date": "2021-06-01"}]
    late = [{"snapshot_id": "u1", "effective_date": "2022-06-01"}]
    assert resolve_pit_ok("sp", cover, date(2022, 1, 1)) is True
    assert resolve_pit_ok("sp", late, date(2022, 1, 1)) is False
    assert resolve_pit_ok(None, None, date(2022, 1, 1)) is False


def test_resolve_pit_ok_fails_closed_on_malformed_snapshot():
    # A malformed/missing effective_date must NOT raise (the holdout was already recorded) — it
    # fails closed to non-PIT (not promotable).
    missing = [{"snapshot_id": "u1"}]  # KeyError
    bad_format = [{"snapshot_id": "u1", "effective_date": "not-a-date"}]  # ValueError
    wrong_type = [{"snapshot_id": "u1", "effective_date": None}]  # TypeError
    assert resolve_pit_ok("sp", missing, date(2022, 1, 1)) is False
    assert resolve_pit_ok("sp", bad_format, date(2022, 1, 1)) is False
    assert resolve_pit_ok("sp", wrong_type, date(2022, 1, 1)) is False


@pytest.mark.parametrize("stages", [
    (),                                              # idea
    (Stage.BACKTESTED, Stage.CANDIDATE),           # candidate
    (Stage.BACKTESTED, Stage.CANDIDATE, Stage.PAPER),  # paper (PAPER->CANDIDATE is legal!)
])
def test_preflight_refuses_non_backtested_source(tmp_path, stages):
    repo = _repo(tmp_path)
    rec = repo.add("alpha")
    repo.record_search_trial("alpha", 4, "{}")  # measured breadth present (so stage is the refusal)
    for s in stages:
        rec = repo.apply_transition(rec, s, Actor.HUMAN, "setup")
    with pytest.raises(TransitionError, match="backtested"):
        promotion_preflight(repo, "alpha", actor=Actor.AGENT, declared_combos=None,
                            allow_holdout_reuse=False, allow_non_pit=False,
                            provider=SyntheticProvider(seed=0), start=_START, end=_END)
    # Refused in preflight => no gate row, no holdout burn.
    assert repo._conn.execute("SELECT COUNT(*) c FROM gate_evaluations").fetchone()["c"] == 0
    assert repo._conn.execute("SELECT COUNT(*) c FROM holdout_evaluations").fetchone()["c"] == 0


def test_preflight_refuses_system_actor_before_any_holdout_or_gate_row(tmp_path):
    # SYSTEM is not a valid promote actor. The refusal is the FIRST check in preflight (before the
    # relaxation guard and before walk_forward), so no holdout is burned and no gate row is minted.
    repo = _repo(tmp_path)
    rec = repo.add("alpha")
    repo.apply_transition(rec, Stage.BACKTESTED, Actor.AGENT, "bt")
    repo.record_search_trial("alpha", 4, "{}")  # measured breadth present (isolate the actor check)
    with pytest.raises(ValueError, match="agent or human"):
        promotion_preflight(repo, "alpha", actor=Actor.SYSTEM, declared_combos=None,
                            allow_holdout_reuse=False, allow_non_pit=False,
                            provider=SyntheticProvider(seed=0), start=_START, end=_END)
    assert repo._conn.execute("SELECT COUNT(*) c FROM gate_evaluations").fetchone()["c"] == 0
    assert repo._conn.execute("SELECT COUNT(*) c FROM holdout_evaluations").fetchone()["c"] == 0


class _NonReproducibleProvider:
    """Neither snapshot_id nor a reproducible marker; get_bars must NOT be reached (the guard fires
    before any provider read)."""

    def get_bars(self, symbols, start, end, timeframe):
        raise AssertionError("provider must not be read: reproducible guard should fire first")


def test_agent_refused_non_reproducible_source(tmp_path):
    repo = _repo(tmp_path)
    repo.add("alpha")
    with pytest.raises(ValueError, match="reproducible data source"):
        promotion_preflight(repo, "alpha", actor=Actor.AGENT, declared_combos=None,
                            allow_holdout_reuse=False, allow_non_pit=False,
                            provider=_NonReproducibleProvider(), start=_START, end=_END)


def test_reproducible_guard_skipped_for_synthetic_agent_and_any_human(tmp_path):
    # SyntheticProvider is reproducible -> the guard does NOT fire for an agent (a later stage check
    # raises instead, since "alpha" is at stage idea). A human is exempt even for a non-reproducible
    # provider (get_bars is never reached because the stage check short-circuits first).
    repo = _repo(tmp_path)
    repo.add("alpha")
    with pytest.raises(Exception) as agent_ei:
        promotion_preflight(repo, "alpha", actor=Actor.AGENT, declared_combos=None,
                            allow_holdout_reuse=False, allow_non_pit=False,
                            provider=SyntheticProvider(seed=0), start=_START, end=_END)
    assert "reproducible data source" not in str(agent_ei.value)
    with pytest.raises(Exception) as human_ei:
        promotion_preflight(repo, "alpha", actor=Actor.HUMAN, declared_combos=None,
                            allow_holdout_reuse=False, allow_non_pit=False,
                            provider=_NonReproducibleProvider(), start=_START, end=_END)
    assert "reproducible data source" not in str(human_ei.value)


class _NonReproducibleWorkingProvider(SyntheticProvider):
    """A working (deterministic) provider that deliberately does NOT advertise reproducibility —
    stands in for a future mutable/live provider for the human-exemption test."""

    reproducible = False


def test_human_exempt_from_reproducible_guard_through_preflight(tmp_path):
    # A human may promote off a non-reproducible source: the guard is agent-only, so preflight runs
    # to COMPLETION (breadth resolved) rather than refusing — proving the exemption holds all the
    # way through, not just that it doesn't raise at the guard. ("alpha" is not a bundled module, so
    # signal-panel parity step is skipped and the provider is never read.)
    repo = _repo(tmp_path)
    rec = repo.add("alpha")
    repo.apply_transition(rec, Stage.BACKTESTED, Actor.HUMAN, "bt")
    repo.record_search_trial("alpha", 4, "{}")
    ctx = promotion_preflight(repo, "alpha", actor=Actor.HUMAN, declared_combos=None,
                              allow_holdout_reuse=False, allow_non_pit=False,
                              provider=_NonReproducibleWorkingProvider(), start=_START, end=_END)
    assert ctx.own == 4 and ctx.provenance == "measured"


def test_preflight_refuses_agent_without_measured_breadth(tmp_path):
    repo = _repo(tmp_path)
    rec = repo.add("alpha")
    repo.apply_transition(rec, Stage.BACKTESTED, Actor.AGENT, "bt")
    with pytest.raises(ValueError, match="search breadth"):
        promotion_preflight(repo, "alpha", actor=Actor.AGENT, declared_combos=None,
                            allow_holdout_reuse=False, allow_non_pit=False,
                            provider=SyntheticProvider(seed=0), start=_START, end=_END)


def test_preflight_resolves_measured_funnel_breadth(tmp_path):
    repo = _repo(tmp_path)
    rec = repo.add("alpha")
    repo.apply_transition(rec, Stage.BACKTESTED, Actor.AGENT, "bt")
    repo.record_search_trial("alpha", 4, "{}")
    repo.record_search_trial("beta", 10, "{}")  # sibling effort raises the funnel bar
    ctx = promotion_preflight(repo, "alpha", actor=Actor.AGENT, declared_combos=None,
                              allow_holdout_reuse=False, allow_non_pit=False,
                              provider=SyntheticProvider(seed=0), start=_START, end=_END)
    assert ctx.own == 4 and ctx.windowed_total == 14 and ctx.n_funnel == 14
    assert ctx.provenance == "measured"


def _write_tmp_strategy(filename: str, body: str) -> Path:
    """Write a temp strategy module into the momentum family dir so load_strategy can find it.
    Caller must unlink it in a finally block (mirrors tests/test_fast_path.py loader tests)."""
    p = Path(momentum_pkg.__path__[0]) / filename
    p.write_text(body)
    return p


def test_preflight_refuses_divergent_signal_panel_before_holdout(tmp_path):
    """A strategy whose signal_panel diverges from its per-bar signal is refused in preflight --
    before any holdout or gate row -- by the exhaustive parity gate (#178)."""
    mod = _write_tmp_strategy(
        "tmp_divergent_panel.py",
        "from typing import Any\n"
        "import pandas as pd\n"
        "from algua.contracts.types import ExecutionContract\n"
        "from algua.strategies.base import StrategyConfig\n"
        "CONFIG = StrategyConfig(name='tmp_divergent_panel', universe=['AAA', 'BBB'],\n"
        "    execution=ExecutionContract("
        "rebalance_frequency='1d', decision_lag_bars=1, warmup_bars=0),\n"
        "    construction='equal_weight_positive')\n"
        "def signal(view, params):\n"
        "    syms = sorted(view['symbol'].unique())\n"
        "    return pd.Series(1.0, index=syms)\n"
        "def signal_panel(bars, params):\n"
        "    adj = bars.reset_index().pivot("
        "index='timestamp', columns='symbol', values='adj_close')\n"
        "    out = pd.DataFrame(0.0, index=adj.index, columns=adj.columns)\n"
        "    out['AAA'] = 1.0\n"
        "    return out\n",
    )
    try:
        repo = _repo(tmp_path)
        rec = repo.add("tmp_divergent_panel")
        repo.apply_transition(rec, Stage.BACKTESTED, Actor.AGENT, "bt")
        repo.record_search_trial("tmp_divergent_panel", 4, "{}")
        with pytest.raises(BacktestError, match="parity"):
            promotion_preflight(
                repo, "tmp_divergent_panel", actor=Actor.AGENT, declared_combos=None,
                allow_holdout_reuse=False, allow_non_pit=False,
                provider=SyntheticProvider(seed=2), start=_START, end=_END)
        assert repo._conn.execute("SELECT COUNT(*) c FROM gate_evaluations").fetchone()["c"] == 0
        assert repo._conn.execute("SELECT COUNT(*) c FROM holdout_evaluations").fetchone()["c"] == 0
    finally:
        mod.unlink()


def test_preflight_passes_parity_for_faithful_bundled_strategy(tmp_path):
    """A real bundled strategy with a faithful signal_panel passes the parity gate and preflight
    resolves breadth as usual (no false positive)."""
    repo = _repo(tmp_path)
    rec = repo.add("cross_sectional_momentum")
    repo.apply_transition(rec, Stage.BACKTESTED, Actor.AGENT, "bt")
    repo.record_search_trial("cross_sectional_momentum", 4, "{}")
    ctx = promotion_preflight(
        repo, "cross_sectional_momentum", actor=Actor.AGENT, declared_combos=None,
        allow_holdout_reuse=False, allow_non_pit=False,
        provider=SyntheticProvider(seed=7), start=_START, end=_END)
    assert ctx.own == 4 and ctx.provenance == "measured"


# --- run_gate DSR binding (#211) ---------------------------------------------------------------
# run_gate calls compute_artifact_hashes(name) -> load_strategy(name), so these tests use a real
# bundled strategy ("cross_sectional_momentum"). The holdout Sharpe is set far above the deflated
# bar at the n_combos used, so the DSR check under test is the only thing that can flip `passed`.

_GATE_START = date(2024, 1, 1)
_GATE_END = date(2024, 6, 1)
_GATE_NAME = "cross_sectional_momentum"

# a walk-forward that passes everything-but-DSR; high Sharpe clears the breadth-deflated bar.
_GATE_HOLDOUT = {
    "sharpe": 7.0, "total_return": 0.2, "n_bars": 252, "skewness": 0.0, "kurtosis": 3.0}
_GATE_STAB = {"pct_positive_windows": 0.8, "min_sharpe": 0.1}


def _gate_wf(holdout=None, stability=None):
    from algua.backtest.walkforward import WalkForwardResult

    return WalkForwardResult(
        strategy=_GATE_NAME, config_hash="c", data_source="synthetic", snapshot_id=None,
        timeframe="1d", seed=None,
        period={"start": "2024-01-01", "end": "2024-06-01"}, windows=4, holdout_frac=0.2,
        window_metrics=[], holdout_metrics=holdout or dict(_GATE_HOLDOUT),
        stability=stability or dict(_GATE_STAB))


def _gate_repo(tmp_path):
    repo = _repo(tmp_path)
    rec = repo.add(_GATE_NAME)
    repo.apply_transition(rec, Stage.BACKTESTED, Actor.AGENT, "bt")
    return repo


def _breadth(repo, provenance: str, *, n: int = 5) -> BreadthContext:
    windowed_total = repo.windowed_search_combos(FUNNEL_WINDOW_DAYS)
    n_funnel = effective_funnel_breadth(n, windowed_total)
    return BreadthContext(n_funnel, n, windowed_total, provenance)


def _run(repo, breadth, wf=None):
    return run_gate(
        repo, wf or _gate_wf(), name=_GATE_NAME, actor=Actor.AGENT, criteria=GateCriteria(),
        breadth=breadth, universe_name=None, universe_snapshots=None,
        period_start=_GATE_START, period_end=_GATE_END, holdout_frac=0.2,
        data_source="synthetic", snapshot_id=None, allow_non_pit=True, reason_suffix="")


def test_run_gate_measured_binds_dsr(tmp_path):
    repo = _gate_repo(tmp_path)
    repo.record_search_trial(_GATE_NAME, 5, "{}", trial_sharpe_count=5,
                             trial_sharpe_mean=0.5, trial_sharpe_var_ann=0.04)
    breadth = _breadth(repo, "measured")
    outcome = _run(repo, breadth)
    d = outcome.decision
    assert d.dsr_binding is True
    assert any(c["name"] == "dsr_evidence" for c in d.checks)


def test_run_gate_declared_breadth_omits_dsr(tmp_path):
    repo = _gate_repo(tmp_path)
    breadth = _breadth(repo, "declared")
    outcome = _run(repo, breadth)
    d = outcome.decision
    assert d.dsr_binding is False
    assert all(c["name"] != "dsr_evidence" for c in d.checks)
    assert d.dsr_skip_reason == "no_measured_dispersion"


def test_run_gate_measured_but_missing_stats_fails_closed(tmp_path):
    repo = _gate_repo(tmp_path)
    repo.record_search_trial(_GATE_NAME, 5, "{}")  # measured row, NULL stats (pre-migration)
    breadth = _breadth(repo, "measured")
    outcome = _run(repo, breadth)
    d = outcome.decision
    assert d.dsr_binding is True
    assert any(c["name"] == "dsr_evidence" and c["passed"] is False for c in d.checks)
    assert d.passed is False
    assert outcome.promoted is False
