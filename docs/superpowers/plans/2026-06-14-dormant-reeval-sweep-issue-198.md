# Dormant Re-eval Sweep (Slice B) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `algua research dormant-sweep` — a read-only advisory command that screens every `dormant` strategy's walk-forward window/stability metrics on a common window and prints a ranked recovery report, without ever reading/revealing/burning the holdout, writing a ledger row, or transitioning anything.

**Architecture:** One new Typer command in `algua/cli/research_cmd.py` reusing `promote`'s input helpers (`select_provider`, `resolve_universe_inputs`, `utc`, `walk_forward`) but **none** of its holdout/gate/transition machinery and **not** `evaluate_gate` (which is holdout-coupled). It lists `Stage.DORMANT`, runs `walk_forward` per strategy (provider + optional PIT universe resolved once), labels each by `wf.stability` thresholds, and emits a single JSON report. Per-strategy `try/except Exception` isolation; `needs_fundamentals` strategies are skipped.

**Tech Stack:** Python 3.12, Typer CLI, pytest (`typer.testing.CliRunner`). Quality gate: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`.

**Spec:** `docs/superpowers/specs/2026-06-13-dormant-reeval-sweep-issue-198-design.md`

---

## File Structure

- `algua/cli/research_cmd.py` — add the `dormant-sweep` command + a few imports. *Task 1.*
- `tests/test_cli_research.py` — append a `_to_dormant` helper + tests. *Tasks 1 & 2.*

No new modules, no schema change, no protected files.

---

## Task 1: The `research dormant-sweep` command + routing/empty tests

**Files:**
- Modify: `algua/cli/research_cmd.py` (add imports + the command)
- Test: `tests/test_cli_research.py` (append helper + tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cli_research.py` (it already has `runner = CliRunner()`, `from algua.cli.main import app`, `import json`, and the autouse `_tmp` fixture pointing `ALGUA_DB_PATH`/`ALGUA_DATA_DIR` at tmp):

```python
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
    # very low thresholds -> the strategy's windows screen as "passed"
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
    # impossibly high thresholds -> the strategy screens as "failed"
    r = runner.invoke(app, ["research", "dormant-sweep", "--demo",
                            "--start", "2022-01-01", "--end", "2023-12-31",
                            "--min-window-sharpe", "100", "--min-pct-positive", "1.0"])
    assert r.exit_code == 0, r.stdout
    p = json.loads(r.stdout)
    assert p["evaluated"] == 1
    assert [x["strategy"] for x in p["failed"]] == ["cross_sectional_momentum"]
    assert p["passed"] == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_cli_research.py -k dormant_sweep -q`
Expected: FAIL — `research dormant-sweep` is not a known command (non-zero exit / usage error).

- [ ] **Step 3: Add the imports**

In `algua/cli/research_cmd.py`, extend the existing `from algua.cli._common import (...)` block to also import `select_provider` and `utc`, and add `load_strategy` + `Stage` imports. The block becomes:

```python
from algua.cli._common import (
    ok,
    registry_conn,
    resolve_eval_inputs,
    resolve_universe_inputs,
    select_provider,
    utc,
)
```
and add, alongside the existing imports:
```python
from algua.contracts.lifecycle import Actor, Stage
from algua.strategies.loader import load_strategy
```
(Verify `select_provider` and `utc` are exported by `algua/cli/_common.py` — both are defined there: `select_provider` at `_common.py:61`, and `utc` is used inside `resolve_eval_inputs`. If `utc` is re-exported from another module, import it from wherever `_common` gets it. `Actor` is already imported on the existing line — extend it to `Actor, Stage` rather than duplicating.)

- [ ] **Step 4: Add the command**

Append to `algua/cli/research_cmd.py`:

```python
@research_app.command("dormant-sweep")
@json_errors(ValueError, LookupError, BacktestError, sqlite3.OperationalError)
def dormant_sweep(
    start: str = typer.Option("2023-01-01", "--start"),
    end: str = typer.Option("2023-12-31", "--end"),
    demo: bool = typer.Option(False, "--demo", help="use the synthetic data provider"),
    snapshot: str = typer.Option(None, "--snapshot", help="screen an ingested bars snapshot id"),
    universe: str = typer.Option(
        None, "--universe",
        help="optional single point-in-time universe applied to ALL dormant strategies; "
             "omit to use each strategy's own static universe"),
    windows: int = typer.Option(4, "--windows", help="walk-forward windows"),
    holdout_frac: float = typer.Option(
        0.2, "--holdout-frac",
        help="walk-forward holdout fraction (shapes the windows; the holdout is NOT revealed)"),
    min_window_sharpe: float = typer.Option(
        0.0, "--min-window-sharpe", help="screen threshold on MEAN walk-forward window Sharpe"),
    min_pct_positive: float = typer.Option(
        0.6, "--min-pct-positive",
        help="screen threshold on the fraction of positive walk-forward windows"),
    top: int = typer.Option(
        None, "--top", help="cap passed/failed lists to the top N by mean window Sharpe"),
) -> None:
    """Advisory STABILITY screen over the dormant pool. For each dormant strategy, re-run
    walk-forward on a common window and report whether its WINDOW/stability metrics look healthy
    again. This is NOT a gate: it never reads, reveals, or burns the single-use holdout, writes no
    ledger rows, and transitions nothing. A pass is a prioritization signal (re-audition via
    `registry transition --to paper`), not a guarantee of re-promotion or forward-gate clearance."""
    if not 0.0 <= min_pct_positive <= 1.0:
        raise ValueError("--min-pct-positive must be in [0, 1]")
    start_dt, end_dt = utc(start), utc(end)
    provider = select_provider(demo, snapshot)
    data_source = type(provider).__name__
    snapshot_id = getattr(provider, "snapshot_id", None)
    # Resolve the optional PIT universe ONCE (common to the whole pool); None => each strategy's
    # own static universe is used by walk_forward.
    universe_by_date, universe_prov = (
        resolve_universe_inputs(universe, start_dt, end_dt) if universe else (None, None))

    with registry_conn() as conn:
        dormant = SqliteStrategyRepository(conn).list_strategies(Stage.DORMANT)

    passed: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for rec in dormant:
        try:
            strategy = load_strategy(rec.name)
            if getattr(strategy.config, "needs_fundamentals", False):
                skipped.append({"strategy": rec.name,
                                "reason": "needs_fundamentals: walk-forward lane not wired"})
                continue
            wf = walk_forward(
                strategy, provider, start_dt, end_dt, windows=windows,
                holdout_frac=holdout_frac, universe_by_date=universe_by_date,
                universe_name=universe, universe_snapshots=universe_prov)
            stability = wf.stability  # window-only metrics; the holdout is deliberately untouched
            screen_passed = (stability["mean_sharpe"] >= min_window_sharpe
                             and stability["pct_positive_windows"] >= min_pct_positive)
            result = {
                "strategy": rec.name, "screen_passed": screen_passed,
                "stability": stability, "windows": wf.window_metrics,
                "config_hash": wf.config_hash, "universe_name": wf.universe_name,
                "universe_snapshots": wf.universe_snapshots,
                "pit": universe_prov is not None,
            }
            (passed if screen_passed else failed).append(result)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:  # noqa: BLE001 - per-strategy isolation: one bad strategy must not abort the sweep
            errors.append({"strategy": rec.name, "error": f"{type(e).__name__}: {e}"})

    evaluated = len(passed) + len(failed)  # BEFORE any --top truncation
    passed.sort(key=lambda r: r["stability"]["mean_sharpe"], reverse=True)
    failed.sort(key=lambda r: r["stability"]["mean_sharpe"], reverse=True)
    if top is not None:
        passed, failed = passed[:top], failed[:top]
    emit(ok({
        "note": ("advisory stability screen over walk-forward windows; NOT the holdout gate. A pass "
                 "means the strategy's windows look healthy again - worth re-auditioning via "
                 "`registry transition --to paper` - it does NOT guarantee it will clear "
                 "re-promotion (which burns a fresh holdout) or the #124 forward gate. Residual "
                 "multiple-testing risk: acting on top-ranked names is a human judgement."),
        "period": {"start": start_dt.date().isoformat(), "end": end_dt.date().isoformat()},
        "data_source": data_source, "snapshot_id": snapshot_id,
        "thresholds": {"min_window_sharpe": min_window_sharpe,
                       "min_pct_positive": min_pct_positive},
        "total_dormant": len(dormant), "evaluated": evaluated,
        "passed": passed, "failed": failed, "skipped": skipped, "errors": errors,
    }))
```

Note: `Actor` may now be unused if no other code in the file referenced it — it IS used by `promote` (`Actor(actor)`), so keep it. `load_strategy` and `Stage` are new uses.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_cli_research.py -k dormant_sweep -q`
Expected: PASS (empty pool, routes-pass, routes-fail).

- [ ] **Step 6: Commit**

```bash
git add algua/cli/research_cmd.py tests/test_cli_research.py
git commit -m "feat(198): research dormant-sweep advisory stability screen over the dormant pool"
```
(End the commit message with a blank line then: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`)

---

## Task 2: Invariant + edge-case tests

**Files:**
- Test: `tests/test_cli_research.py` (append; reuses `_to_dormant` from Task 1)

These exercise the already-implemented command — they must pass without changing production code (if one fails, fix the test wiring, not the command; a genuine invariant break is a real bug → stop and report).

- [ ] **Step 1: Write the tests**

Append to `tests/test_cli_research.py`:

```python
def test_dormant_sweep_never_reveals_holdout():
    _to_dormant()
    r = runner.invoke(app, ["research", "dormant-sweep", "--demo",
                            "--start", "2022-01-01", "--end", "2023-12-31",
                            "--min-window-sharpe", "-100", "--min-pct-positive", "0"])
    assert r.exit_code == 0, r.stdout
    # The single-use holdout must never leak through this advisory screen.
    assert "holdout" not in r.stdout.lower()


def test_dormant_sweep_has_no_side_effects():
    _to_dormant()
    from contextlib import closing
    import os
    from algua.registry.db import connect

    def _counts():
        with closing(connect(os.environ["ALGUA_DB_PATH"])) as conn:
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
    assert after == before                      # no gate/holdout rows written
    assert after[2] == "dormant"                # stage unchanged


def test_dormant_sweep_is_repeatable():
    _to_dormant()
    args = ["research", "dormant-sweep", "--demo", "--start", "2022-01-01", "--end", "2023-12-31",
            "--min-window-sharpe", "-100", "--min-pct-positive", "0"]
    p1 = json.loads(runner.invoke(app, args).stdout)
    p2 = json.loads(runner.invoke(app, args).stdout)
    assert [x["strategy"] for x in p1["passed"]] == [x["strategy"] for x in p2["passed"]]
    assert p1["evaluated"] == p2["evaluated"] == 1   # no holdout exhaustion on the 2nd run


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
    # the non-fundamentals strategy still got evaluated
    evaluated_names = [x["strategy"] for x in p["passed"]] + [x["strategy"] for x in p["failed"]]
    assert "cross_sectional_momentum" in evaluated_names


def test_dormant_sweep_ignores_non_dormant_strategies():
    # a strategy left at backtested must never appear in the sweep
    assert runner.invoke(app, ["registry", "add", "cross_sectional_momentum"]).exit_code == 0
    assert runner.invoke(app, ["registry", "transition", "cross_sectional_momentum",
                               "--to", "backtested", "--actor", "human",
                               "--reason", "x"]).exit_code == 0
    r = runner.invoke(app, ["research", "dormant-sweep", "--demo",
                            "--start", "2022-01-01", "--end", "2023-12-31"])
    assert r.exit_code == 0, r.stdout
    p = json.loads(r.stdout)
    assert p["total_dormant"] == 0
    names = [x["strategy"] for x in p["passed"] + p["failed"]] + [s["strategy"] for s in p["skipped"]]
    assert "cross_sectional_momentum" not in names
```

- [ ] **Step 2: Run the tests to verify they pass**

Run: `uv run pytest tests/test_cli_research.py -k dormant_sweep -q`
Expected: PASS (all Task 1 + Task 2 dormant-sweep tests).

Note on the `holdout_evaluations`/`gate_evaluations` tables in `test_dormant_sweep_has_no_side_effects`: they exist after `migrate()` runs (the CLI runs it on connect). If a table name differs, confirm against `algua/registry/db.py` and adjust the `_counts()` query — do NOT relax the assertion.

- [ ] **Step 3: Commit**

```bash
git add tests/test_cli_research.py
git commit -m "test(198): dormant-sweep invariants (no holdout leak, no side effects, repeatable, skips, isolation)"
```
(End with a blank line then the `Co-Authored-By:` trailer as above.)

---

## Task 3: Full quality gate

- [ ] **Step 1: Run the complete gate**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: all green. The `noqa: BLE001` on the broad `except Exception` keeps ruff quiet about blind-except (intentional per-strategy isolation). `lint-imports` stays green — the new imports (`load_strategy`, `Stage`, `select_provider`, `utc`) are all within layers the CLI already depends on.

- [ ] **Step 2: Fix any failures**

If `mypy` complains about `Any` in the result-dict lists, the `list[dict[str, Any]]` annotations cover it. If `ruff` flags an unused import, remove it. If `--top` interplay with `evaluated` looks off, confirm `evaluated` is computed before truncation (it is). Re-run until green.

- [ ] **Step 3: Final commit (if Step 2 changed anything)**

```bash
git add -A
git commit -m "chore(198): satisfy quality gate for dormant-sweep"
```

---

## Self-Review notes (reconciled against the spec)

- **Advisory stability screen, no holdout** (spec §Decision, §"Why a stability screen") → Task 1 (`wf.stability` only; no `holdout` key) + `test_dormant_sweep_never_reveals_holdout`.
- **Read-only / no side effects** (spec §Architecture last para) → Task 1 (no transition/reserve/run_gate/evaluate_gate) + `test_dormant_sweep_has_no_side_effects`.
- **Repeatable** (spec §Decision) → `test_dormant_sweep_is_repeatable`.
- **Common window + per-strategy static universe; optional single --universe resolved once** (spec §Decision, §Architecture step 1) → Task 1 (`resolve_universe_inputs` once; `walk_forward` uses static universe when `universe_prov is None`).
- **Per-strategy isolation (broad except)** (spec §Error handling) → Task 1 (`except Exception`, re-raise `KeyboardInterrupt`/`SystemExit`). Note: not separately unit-tested (hard to force a mid-eval crash on a bundled strategy without a fixture); the broad-except + the multi-strategy skip test cover the structure.
- **Fundamentals skipped** (spec §Decision) → `test_dormant_sweep_skips_fundamentals_and_evaluates_others_in_one_run`.
- **Non-dormant excluded** (spec §Testing) → `test_dormant_sweep_ignores_non_dormant_strategies`.
- **Empty pool** (spec §Error handling) → `test_dormant_sweep_empty_pool`.
- **Report shape + note + thresholds + ranking** (spec §Output) → Task 1 emit block.
- **Ranking across multiple PASSED strategies:** only one non-fundamentals bundled strategy exists, so multi-element pass-ordering isn't asserted; the sort is applied unconditionally and exercised on single-element lists. Acceptable test limitation, not a spec gap.
- **Deferred (not in this plan):** scheduling, auto-transition, holdout-burning re-gate, per-strategy stored windows, structured `dormancy_reason`, fundamentals re-eval.
