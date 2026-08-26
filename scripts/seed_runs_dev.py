#!/usr/bin/env python3
"""Dev-only seeding script for the run-ledger views (slice 3, Task 1).

The `runs` table (v44) on the operator's real registry has zero rows: it fills only when the
operator loop restarts, which is a human decision that has not happened. Without seed data the
five run-ledger charts (slice 3) have nothing to render and their "render it and look at it"
verification step cannot complete. This script seeds a SCRATCH database with plausible runs via
the public repository API only (`SqliteStrategyRepository.record_run`, `record_sweep_trials`,
`record_gate_evaluation`, `reserve_holdout`/`record_holdout_returns`, ...) — never a raw INSERT —
so seeded rows obey exactly the validation real rows do, and the seed cannot silently drift from
the schema when the schema next changes.

SAFETY (hard requirement): this script must be INCAPABLE of writing to the operator's real
registry. This repo's dominant workflow is git WORKTREES, and every worktree carries its own copy
of this file — a guard computed only from `__file__`'s location protects that one worktree's own
`data/algua.db` and is INERT against a different checkout's registry (including the operator's
real one, e.g. `~/Projects/algua/data/algua.db`) passed in as an absolute `--db` path: it does not
equal the script-relative path as a string, and it is not a hardlink of it, so a pure
`__file__`-relative check lets it through. The refusal therefore checks `--db` (resolved, never
string-compared) against EVERY plausible real-registry path: the script-relative path, the
CURRENT WORKING DIRECTORY's `data/algua.db` (whatever shell the invoker happens to run from), and
`Settings.db_path` (which honours `ALGUA_DB_PATH`/`.env`, exactly the knob this script's own usage
instructions tell you to set) — so an operator-configured path is covered too. A second,
INODE-level check (`os.path.samefile`) catches a hardlink sharing any candidate's inode under a
different path — `Path.resolve()` alone dereferences symlinks but says nothing about hardlinks,
since a hardlink IS a distinct path to the same file, not a link to resolve. `--yes` is required to
proceed. All checks run BEFORE anything touches disk — `connect()` (which creates the file) is
called only after every check passes.

Usage:
    uv run python scripts/seed_runs_dev.py --db /tmp/slice3-dev.db --yes

Then, to look at it:
    ALGUA_DB_PATH=/tmp/slice3-dev.db uv run algua runs list --limit 5

Frontend against the seeded DB — the backend (a standalone uv project under web/) shells out to
the CLI, so it needs ALGUA_DB_PATH in ITS OWN environment:
    ALGUA_DB_PATH=/tmp/slice3-dev.db uv run --project web python -m backend.main
    cd web/frontend && npm run dev
(run the two in separate terminals; the frontend dev server proxies /api to 127.0.0.1:8787).
CONSTRAINT: if a deployed `algua-web` service is already bound to 127.0.0.1:8787 on this host
(e.g. a systemd user unit), the backend command above will fail with "address already in use" —
stop that service for the session, or run the backend on another port and point the frontend's
dev proxy at it instead.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from algua.config.settings import get_settings
from algua.registry.db import connect, migrate
from algua.registry.store import SqliteStrategyRepository

DEFAULT_SEED = 20260826  # today's date, at authorship time — fixed, not "whatever day this runs"
REPO_ROOT = Path(__file__).resolve().parent.parent


def _real_registry_candidates() -> list[Path]:
    """Every path that could plausibly BE (or alias) the operator's real registry, resolved.
    See the module docstring's SAFETY section for why a single `__file__`-relative check is not
    enough: this repo is worked from git worktrees, each carrying its own copy of this script, so
    the guard must also cover the invoking shell's cwd and whatever `Settings.db_path` resolves to
    (which honours `ALGUA_DB_PATH`/`.env`) — not just this copy's own worktree-relative guess."""
    candidates = [REPO_ROOT / "data" / "algua.db", Path.cwd() / "data" / "algua.db"]
    try:
        candidates.append(get_settings().db_path)
    except Exception:
        # Settings construction failing (e.g. a malformed .env) must never WEAKEN the guard —
        # the two path-based candidates above still stand.
        pass
    return [c.resolve() for c in candidates]


def _short_hash(*parts: str) -> str:
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:16]


def _business_dates(start: date, n: int) -> list[date]:
    """n consecutive weekdays starting at `start` (no holiday calendar — a dev seed doesn't need
    one; the point is monotonic dates, not a real trading calendar)."""
    out: list[date] = []
    d = start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def _gauss_returns(rng: random.Random, n: int, mu: float, sigma: float) -> list[float]:
    return [rng.gauss(mu, sigma) for _ in range(n)]


def _series_metrics(returns: list[float]) -> dict[str, float | int | None]:
    """Plausible full-period IS stats from a generated return series. Not a reimplementation of
    `algua.backtest`'s real metrics math — this is seed data, not a fixture for testing that
    math — just internally consistent numbers for the equity-curve view."""
    s = pd.Series(returns, dtype=float)
    ann_vol = float(s.std(ddof=0) * (252**0.5))
    mean_ann = float(s.mean() * 252)
    sharpe = mean_ann / ann_vol if ann_vol > 0 else 0.0
    downside = s[s < 0]
    downside_dev = float(downside.std(ddof=0) * (252**0.5)) if len(downside) > 1 else 0.0
    sortino = mean_ann / downside_dev if downside_dev > 0 else 0.0
    cum = (1.0 + s).cumprod()
    total_return = float(cum.iloc[-1] - 1.0)
    running_max = cum.cummax()
    drawdown = (cum / running_max) - 1.0
    max_dd = float(abs(drawdown.min()))
    n_years = len(s) / 252
    cagr = float(cum.iloc[-1] ** (1 / n_years) - 1.0) if n_years > 0 and cum.iloc[-1] > 0 else 0.0
    calmar = cagr / max_dd if max_dd > 0 else 0.0
    return {
        "sharpe_is": round(sharpe, 4), "sortino_is": round(sortino, 4),
        "total_return_is": round(total_return, 4), "max_drawdown_is": round(max_dd, 4),
        "ann_vol_is": round(ann_vol, 4), "cagr_is": round(cagr, 4), "calmar_is": round(calmar, 4),
        "n_obs_is": len(returns),
    }


@dataclass(frozen=True)
class StrategySpec:
    name: str
    family: str
    registered: bool
    regime: str  # key into REGIME_PARAMS — drives the IS-vs-OOS scatter placement


# mean_window_sharpe (x) vs sharpe_oos (y) placement for the IS-vs-OOS scatter (view 2). The
# diagonal is y=x. `None` means "no walk-forward evidence" (exercises NULL handling). Regimes are
# deliberately on BOTH sides of the diagonal — task-1 brief: "a seed that only produces one regime
# makes the most valuable view untestable."
REGIME_PARAMS: dict[str, dict[str, float | None]] = {
    # honest: OOS tracks the walk-forward mean-window figure closely (near the diagonal).
    "honest_pos": {"mean_window_sharpe": 0.52, "sharpe_oos": 0.61},
    "honest_pos_2": {"mean_window_sharpe": 0.29, "sharpe_oos": 0.24},
    "honest_neg": {"mean_window_sharpe": -0.18, "sharpe_oos": -0.24},
    # mined: OOS is anomalously far ABOVE the walk-forward figure — the multiple-testing fluke
    # this platform's breadth deflation exists to catch (a single lucky holdout burn out of many
    # combos searched, not a genuine edge the walk-forward windows ever showed).
    "mined_above_1": {"mean_window_sharpe": 0.08, "sharpe_oos": 1.42},
    "mined_above_2": {"mean_window_sharpe": -0.05, "sharpe_oos": 0.98},
    "mined_above_3": {"mean_window_sharpe": 0.22, "sharpe_oos": 1.71},
    # overfit: the mirror image — strong across the walk-forward windows, decays hard on the
    # untouched holdout. Genuine spread on the OTHER side of the diagonal too.
    "overfit_below": {"mean_window_sharpe": 1.35, "sharpe_oos": 0.11},
    # no walk-forward evidence at all.
    "null_metrics": {"mean_window_sharpe": None, "sharpe_oos": None},
}

STRATEGIES: list[StrategySpec] = [
    StrategySpec("trend_breakout_v1", "trend-following", True, "honest_pos"),
    StrategySpec("trend_breakout_v2", "trend-following", True, "honest_neg"),
    StrategySpec("trend_pullback_v1", "trend-following", True, "mined_above_1"),
    StrategySpec("trend_pullback_v2", "trend-following", False, "mined_above_2"),
    StrategySpec("vol_carry_v1", "mean-reversion", True, "overfit_below"),
    StrategySpec("vol_carry_v2", "mean-reversion", True, "null_metrics"),
    StrategySpec("rsi_reversion_v1", "mean-reversion", True, "honest_pos_2"),
    StrategySpec("rsi_reversion_v2", "mean-reversion", False, "mined_above_3"),
]

# Which registered strategies also get a persisted holdout return series (series_holdout_id on
# their gate run) — exercises view 4's shaded-OOS-region branch across an honest/mined/overfit
# spread, while others deliberately exercise the "no series pointer" branch.
STRATEGIES_WITH_HOLDOUT_SERIES = {"trend_breakout_v1", "trend_pullback_v1", "vol_carry_v1"}

# Strategies whose sweep gets one deliberately-metric-less trial (a trial that produced no
# stability stats) — "a few runs with NULL metrics" per the brief, spread beyond just one strategy.
STRATEGIES_WITH_A_NULL_TRIAL = {"rsi_reversion_v1", "trend_pullback_v2"}

# Per-regime in-sample drift/vol for the generated backtest return series — plausible, not a
# reimplementation of any real metric math (see `_series_metrics`).
REGIME_BACKTEST_DRIFT: dict[str, tuple[float, float]] = {
    "honest_pos": (0.0006, 0.010), "honest_pos_2": (0.0005, 0.010),
    "honest_neg": (-0.0003, 0.012),
    "mined_above_1": (0.0004, 0.014), "mined_above_2": (0.0003, 0.014),
    "mined_above_3": (0.0005, 0.014),
    "overfit_below": (0.0012, 0.009),
}


def _oos_metrics(
    rng: random.Random, mean_ws: float | None, sharpe_oos: float | None, n_bars: int,
) -> dict[str, float | int | None]:
    if mean_ws is None or sharpe_oos is None:
        return {
            "sharpe_oos": None, "sortino_oos": None, "total_return_oos": None,
            "max_drawdown_oos": None, "ann_vol_oos": None, "n_obs_oos": n_bars,
            "mean_window_sharpe": None, "std_window_sharpe": None,
            "min_window_sharpe": None, "pct_positive_windows": None,
        }
    ann_vol = round(rng.uniform(0.08, 0.22), 4)
    total_return = round(sharpe_oos * ann_vol * (n_bars / 252) ** 0.5, 4)
    max_dd = round(abs(rng.uniform(0.03, 0.18)) + (0.05 if sharpe_oos < 0 else 0.0), 4)
    sortino = round(sharpe_oos * rng.uniform(1.05, 1.3), 4)
    std_ws = round(abs(mean_ws) * rng.uniform(0.2, 0.5) + 0.05, 4)
    min_ws = round(mean_ws - std_ws * 1.5, 4)
    pct_pos = round(min(0.95, max(0.05, 0.5 + mean_ws * 0.25)), 4)
    return {
        "sharpe_oos": sharpe_oos, "sortino_oos": sortino, "total_return_oos": total_return,
        "max_drawdown_oos": max_dd, "ann_vol_oos": ann_vol, "n_obs_oos": n_bars,
        "mean_window_sharpe": mean_ws, "std_window_sharpe": std_ws,
        "min_window_sharpe": min_ws, "pct_positive_windows": pct_pos,
    }


def _gate_checks(
    mean_ws: float | None, sharpe_oos: float | None, effective_floor: float, n_bars: int,
    min_bars: int,
) -> list[dict[str, Any]]:
    """Mirrors `algua.research.gates`' check shape: binding integrity-floor checks
    (`min_holdout_observations`, `holdout_sharpe_floor`, `pit_required`) plus advisory
    statistical checks (`holdout_sharpe`, `holdout_return`, `pct_positive_windows`,
    `min_window_sharpe`). A NULL metric fails closed (never a pass), matching the real gate's
    non-finite-value handling.

    Evidence presence is checked ONCE, as an explicit `is not None` guard, so `mean_ws`/
    `sharpe_oos` are narrowed to `float` (not `float | None`) everywhere they're used below —
    a separately-computed bool (the earlier `has_evidence` pattern) doesn't narrow anything for
    a type checker, since it can't prove the bool tracks the Optionals' nullness.
    """
    obs_passed = n_bars >= min_bars
    pit_passed = True
    holdout_return_value: float | None
    pct_pos_value: float | None
    min_ws_value: float | None
    if mean_ws is not None and sharpe_oos is not None:
        floor_passed = sharpe_oos > 0.0
        holdout_sharpe_passed = sharpe_oos >= effective_floor
        holdout_return_value = round(sharpe_oos * 0.1, 4)
        holdout_return_passed = sharpe_oos > 0.0
        pct_pos_value = round(min(0.95, max(0.05, 0.5 + mean_ws * 0.25)), 4)
        pct_pos_passed = mean_ws >= 0.0
        min_ws_value = round(mean_ws - 0.3, 4)
        min_ws_passed = (mean_ws - 0.3) >= -0.5
    else:
        floor_passed = False
        holdout_sharpe_passed = False
        holdout_return_value = None
        holdout_return_passed = False
        pct_pos_value = None
        pct_pos_passed = False
        min_ws_value = None
        min_ws_passed = False
    checks: list[dict[str, Any]] = [
        {"name": "min_holdout_observations", "value": n_bars, "threshold": min_bars, "op": ">=",
         "passed": obs_passed},
        {"name": "holdout_sharpe_floor", "value": sharpe_oos, "threshold": 0.0, "op": ">",
         "passed": floor_passed},
        {"name": "pit_required", "passed": pit_passed},
        {"name": "holdout_sharpe", "value": sharpe_oos, "threshold": effective_floor, "op": ">=",
         "passed": holdout_sharpe_passed, "advisory": True},
        {"name": "holdout_return", "value": holdout_return_value, "threshold": 0.0, "op": ">",
         "passed": holdout_return_passed, "advisory": True},
        {"name": "pct_positive_windows", "value": pct_pos_value, "threshold": 0.5, "op": ">=",
         "passed": pct_pos_passed, "advisory": True},
        {"name": "min_window_sharpe", "value": min_ws_value, "threshold": -0.5, "op": ">=",
         "passed": min_ws_passed, "advisory": True},
    ]
    return checks


def _sweep_trials(
    rng: random.Random, strategy: StrategySpec, mean_ws: float | None,
) -> list[dict[str, Any]]:
    n_trials = rng.randint(5, 8)
    center = mean_ws if mean_ws is not None else rng.uniform(-0.3, 0.3)
    trials: list[dict[str, Any]] = []
    for i in range(n_trials):
        lookback = rng.choice([20, 30, 60, 90, 120, 180, 252])
        threshold = round(rng.uniform(0.5, 2.5), 2)
        config = {"lookback": lookback, "threshold": threshold}
        config_hash = _short_hash(strategy.name, "trial", str(i))
        if strategy.name in STRATEGIES_WITH_A_NULL_TRIAL and i == 0:
            trials.append({"config": config, "config_hash": config_hash, "metrics": {}})
            continue
        trial_ws = round(center + rng.uniform(-0.6, 0.6), 4)
        std_ws = round(abs(rng.uniform(0.05, 0.4)), 4)
        min_ws = round(trial_ws - std_ws * 1.8, 4)
        pct_pos = round(min(0.95, max(0.05, 0.5 + trial_ws * 0.2)), 4)
        trials.append({
            "config": config, "config_hash": config_hash,
            "metrics": {
                "mean_window_sharpe": trial_ws, "std_window_sharpe": std_ws,
                "min_window_sharpe": min_ws, "pct_positive_windows": pct_pos,
            },
        })
    return trials


def _seed_strategy(
    repo: SqliteStrategyRepository, rng: random.Random, spec: StrategySpec,
) -> dict[str, int]:
    counts = {"backtest": 0, "walk_forward": 0, "sweep": 0, "sweep_trial": 0, "gate": 0}
    strategy_id: int | None = None
    if spec.registered:
        rec = repo.add(spec.name, family=spec.family)
        strategy_id = rec.id

    is_start = date(2024, 1, 2)
    is_dates = _business_dates(is_start, 252)
    period_start, period_end = is_dates[0].isoformat(), is_dates[-1].isoformat()
    code_hash = _short_hash(spec.name, "code")
    dependency_hash = _short_hash(spec.name, "deps")
    base_provenance = {
        "code_hash": code_hash, "dependency_hash": dependency_hash,
        "data_source": "SyntheticProvider", "universe_name": "seed_universe_2024",
        "seed": DEFAULT_SEED, "timeframe": "1d",
        "period_start": period_start, "period_end": period_end,
    }

    # -- backtest -----------------------------------------------------------------------------
    bt_config_hash = _short_hash(spec.name, "backtest")
    if spec.regime == "null_metrics":
        bt_run_id = repo.record_run(
            "backtest", spec.name, strategy_id=strategy_id,
            provenance=base_provenance | {"config_hash": bt_config_hash},
            config={"lookback": 60}, metrics=None,
        )
    else:
        mu, sigma = REGIME_BACKTEST_DRIFT[spec.regime]
        returns = _gauss_returns(rng, len(is_dates), mu, sigma)
        bt_metrics = _series_metrics(returns)
        series_backtest_id = None
        if spec.registered:
            series = pd.Series(returns, index=pd.to_datetime(is_dates), dtype=float)
            series_backtest_id = repo.persist_backtest_returns(
                spec.name, period_start, period_end, series)
        bt_run_id = repo.record_run(
            "backtest", spec.name, strategy_id=strategy_id,
            provenance=base_provenance | {"config_hash": bt_config_hash},
            config={"lookback": 60}, metrics=bt_metrics,
            series_backtest_id=series_backtest_id,
        )
    counts["backtest"] += 1

    # -- walk_forward ---------------------------------------------------------------------------
    params = REGIME_PARAMS[spec.regime]
    n_bars = 63
    wf_metrics = _oos_metrics(rng, params["mean_window_sharpe"], params["sharpe_oos"], n_bars)
    wf_config_hash = _short_hash(spec.name, "walk_forward")
    wf_run_id = repo.record_run(
        "walk_forward", spec.name, strategy_id=strategy_id,
        provenance=base_provenance | {"config_hash": wf_config_hash},
        metrics=wf_metrics, derived_from=[bt_run_id],
    )
    counts["walk_forward"] += 1

    # -- sweep + sweep_trial ----------------------------------------------------------------------
    trials = _sweep_trials(rng, spec, params["mean_window_sharpe"])
    sweep_config_hash = _short_hash(spec.name, "sweep")
    sweep_run_id = repo.record_run(
        "sweep", spec.name, strategy_id=strategy_id,
        provenance=base_provenance | {"config_hash": sweep_config_hash},
        config={"grid": {"lookback": [20, 30, 60, 90, 120, 180, 252]},
                "rank_by": "mean_window_sharpe", "windows": 4, "holdout_frac": 0.2},
        extra_metrics={"mean_trial_sharpe": round(rng.uniform(-0.2, 0.6), 4)},
    )
    counts["sweep"] += 1
    n_written, _truncated_at = repo.record_sweep_trials(sweep_run_id, spec.name, trials)
    counts["sweep_trial"] += n_written

    # -- gate (registered strategies only — record_gate_evaluation requires a real strategy_id) -
    if strategy_id is not None:
        own_lifetime_combos = n_written
        windowed_total_combos = own_lifetime_combos + rng.randint(20, 50)
        base_floor = 0.5
        haircut = round(0.05 * (own_lifetime_combos**0.5), 4)
        effective_floor = round(base_floor + haircut, 4)
        checks = _gate_checks(
            params["mean_window_sharpe"], params["sharpe_oos"], effective_floor, n_bars, n_bars)
        binding_names = {"min_holdout_observations", "holdout_sharpe_floor", "pit_required"}
        overall_passed = all(c["passed"] for c in checks if c["name"] in binding_names)
        decision = {
            "passed": overall_passed, "checks": checks,
            "n_combos": windowed_total_combos, "breadth_provenance": "measured",
            "base_min_holdout_sharpe": base_floor, "effective_min_holdout_sharpe": effective_floor,
            "own_lifetime_combos": own_lifetime_combos,
            "windowed_total_combos": windowed_total_combos,
            "funnel_window_days": 90, "pit_ok": True, "pit_override": False,
            "returns_available": spec.name in STRATEGIES_WITH_HOLDOUT_SERIES,
        }
        holdout_dates = _business_dates(is_dates[-1] + timedelta(days=3), n_bars)
        holdout_start, holdout_end = holdout_dates[0].isoformat(), holdout_dates[-1].isoformat()
        gate_config_hash = _short_hash(spec.name, "gate")
        gate_id = repo.record_gate_evaluation(
            strategy_id, passed=overall_passed, n_funnel=len(STRATEGIES),
            own_lifetime_combos=own_lifetime_combos,
            windowed_total_combos=windowed_total_combos, funnel_window_days=90,
            breadth_provenance="measured", pit_ok=True, pit_override=False,
            holdout_n_bars=n_bars, min_holdout_observations=n_bars,
            code_hash=code_hash, config_hash=gate_config_hash, dependency_hash=dependency_hash,
            data_source="SyntheticProvider", snapshot_id=None,
            period_start=period_start, period_end=period_end, holdout_frac=0.2, actor="agent",
            decision_json=json.dumps(decision),
        )

        series_holdout_id = None
        if spec.name in STRATEGIES_WITH_HOLDOUT_SERIES and params["sharpe_oos"] is not None:
            reservation_id, _reused = repo.reserve_holdout(
                strategy_id, data_source="SyntheticProvider", snapshot_id=None,
                period_start=period_start, period_end=period_end, holdout_frac=0.2,
                holdout_start=holdout_start, holdout_end=holdout_end, allow_reuse=False,
            )
            repo.finalize_holdout_reservation(
                reservation_id, config_hash=gate_config_hash, strategy_id=strategy_id)
            oos_mu = float(params["sharpe_oos"]) * 0.15 / 252
            oos_returns = _gauss_returns(rng, n_bars, oos_mu, 0.012)
            series_holdout_id = repo.record_holdout_returns(
                reservation_id, strategy_id, holdout_start=holdout_start, holdout_end=holdout_end,
                returns=oos_returns, bar_dates=[d.isoformat() for d in holdout_dates],
            )

        repo.record_run(
            "gate", spec.name, strategy_id=strategy_id, gate_id=gate_id,
            derived_from=[wf_run_id],
            provenance=base_provenance | {"config_hash": gate_config_hash},
            metrics=wf_metrics, passed=overall_passed, series_holdout_id=series_holdout_id,
        )
        counts["gate"] += 1

    return counts


def seed(repo: SqliteStrategyRepository, seed_value: int) -> dict[str, int]:
    rng = random.Random(seed_value)
    totals = {"backtest": 0, "walk_forward": 0, "sweep": 0, "sweep_trial": 0, "gate": 0}
    for spec in STRATEGIES:
        counts = _seed_strategy(repo, rng, spec)
        for k, v in counts.items():
            totals[k] += v
    return totals


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Seed a SCRATCH registry DB with plausible run-ledger rows for slice-3 "
                    "chart development. Refuses to touch the operator's real registry.")
    parser.add_argument(
        "--db", required=True,
        help="Path to the SCRATCH database to seed (e.g. /tmp/slice3-dev.db). Required — there "
             "is no default, so this can never silently target the real registry.")
    parser.add_argument(
        "--yes", action="store_true",
        help="Confirm you want to write to the resolved --db path printed above.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="RNG seed (deterministic).")
    args = parser.parse_args(argv)

    target = Path(args.db).resolve()
    real_candidates = _real_registry_candidates()
    print(f"seed target (resolved): {target}", file=sys.stderr)

    for real_db in real_candidates:
        if target == real_db:
            print(
                f"REFUSING: --db resolves to a real-registry candidate ({real_db}). "
                "This script may only write to a scratch path. Nothing was written.",
                file=sys.stderr,
            )
            return 1
        # Inode-level check: a hardlink at `target` sharing a candidate's inode resolves to a
        # DIFFERENT path (samefile, not the same string), so the equality check above alone would
        # miss it and `connect()` would open the real registry's actual bytes under a
        # scratch-looking name. Only meaningful when both paths already exist (a hardlink target
        # must pre-exist) — `samefile` raises on a missing path, and a fresh scratch DB not
        # existing yet is the normal case.
        if real_db.exists() and target.exists() and os.path.samefile(target, real_db):
            print(
                f"REFUSING: --db ({target}) is a hardlink to a real-registry candidate "
                f"({real_db}) — same file, different path. Nothing was written.",
                file=sys.stderr,
            )
            return 1
    if not args.yes:
        print(
            "Refusing to write without --yes. Re-run with --yes once you've reviewed the "
            "resolved path above. Nothing was written.",
            file=sys.stderr,
        )
        return 1

    conn = connect(target)
    try:
        migrate(conn)
        repo = SqliteStrategyRepository(conn)
        counts = seed(repo, args.seed)
    finally:
        conn.close()

    print(json.dumps({"db": str(target), "seeded": counts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
