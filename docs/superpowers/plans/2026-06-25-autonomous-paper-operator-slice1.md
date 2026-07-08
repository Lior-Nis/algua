# Autonomous Paper Operator — Slice 1 (Paper Allocation + Schedule Predicate) Implementation Plan

> **Status (2026-07-04):** superseded — historical. PR #288 closed; implementation tracked under epic #318 (#316/#317 + follow-ons).

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the two self-contained, no-LLM, no-attribution primitives the autonomous paper operator needs first — per-strategy paper capital allocation (`paper allocate`, Σ ≤ account equity) and the schedule-class `is_due` predicate — so concurrent paper strategies have a bounded sizing denominator and the future `run-all` driver has a "tick this session?" seam.

**Architecture:** `paper allocate` reuses the existing venue-agnostic `algua/registry/allocations.py` module (it already enforces Σ(active capital) ≤ a caller-supplied `account_equity` atomically under `BEGIN IMMEDIATE`), passing the **paper** account equity read from the Alpaca paper broker. `is_due` is a pure predicate over a strategy's `rebalance_frequency`, fail-closed on anything it doesn't recognize. Both are independently testable and merge on their own.

**Tech Stack:** Python 3.12, Typer (CLI), SQLite (`strategy_allocations` table — already migrated), pytest, Typer `CliRunner`.

## Global Constraints

- Drive everything through `uv run algua ...`; never bypass the CLI. (CLAUDE.md)
- Quality gate green before every commit: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`.
- Ruff line length ceiling: **100 columns**.
- Additions-only for strategies/policies; never weaken an existing gate or risk wall.
- `git add` is **scoped to named files** — never `git add -A` (concurrent sessions leave untracked WIP).
- This box is **paper-only**; it never holds live allocations, so paper reuses the shared `strategy_allocations` table. (Documented limitation — see Task 2 note.)
- Work on branch `autonomous-paper-operator` (already created).

---

## File structure

- `algua/execution/schedule.py` — **Create.** Pure schedule-class predicate `is_due`. One responsibility: "should a strategy with this execution contract decide on this session?" Depends only on the contract type.
- `tests/test_schedule.py` — **Create.** Unit tests for `is_due`.
- `algua/cli/paper_cmd.py` — **Modify.** Add the `paper allocate` command + a `_paper_account_equity()` helper. Follows the existing `live allocate` pattern (`live_cmd.py:76`).
- `tests/test_cli_paper.py` — **Modify.** Add allocate command tests (mirror `tests/test_allocations.py` + the existing paper-CLI broker-mock style).

---

### Task 1: Schedule-class predicate (`is_due`)

**Files:**
- Create: `algua/execution/schedule.py`
- Test: `tests/test_schedule.py`

**Interfaces:**
- Consumes: nothing (pure).
- Produces: `is_due(rebalance_frequency: str) -> bool` — `True` for `"1d"`; raises `ValueError` for any unrecognized frequency (fail-closed: never silently tick or silently skip). This is the seam the future `paper run-all` calls per strategy; multi-cadence support is added here later without touching callers.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_schedule.py
from __future__ import annotations

import pytest

from algua.execution.schedule import is_due


def test_daily_is_always_due():
    assert is_due("1d") is True


def test_unknown_frequency_fails_closed():
    with pytest.raises(ValueError, match="unsupported rebalance_frequency"):
        is_due("1w")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_schedule.py -v`
Expected: FAIL — `ModuleNotFoundError: algua.execution.schedule` (module not defined).

- [ ] **Step 3: Write minimal implementation**

```python
# algua/execution/schedule.py
"""Schedule-class predicate: should a strategy decide on the current session?

Today every strategy is daily/XNYS, so the only supported class is "1d" (decide every session).
Multi-cadence (weekly, intraday, crypto 24/7) is added here later WITHOUT changing callers — the
future `paper run-all` only asks `is_due(strategy.execution.rebalance_frequency)`. Fail closed on
anything unrecognized so a typo'd or future frequency can never silently tick or silently skip."""
from __future__ import annotations


def is_due(rebalance_frequency: str) -> bool:
    if rebalance_frequency == "1d":
        return True
    raise ValueError(f"unsupported rebalance_frequency {rebalance_frequency!r}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_schedule.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Quality gate + commit**

```bash
uv run ruff check algua/execution/schedule.py tests/test_schedule.py
uv run mypy algua/execution/schedule.py
git add algua/execution/schedule.py tests/test_schedule.py
git commit -m "feat(execution): is_due schedule-class predicate (fail-closed, daily-only)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: `paper allocate` command

**Files:**
- Modify: `algua/cli/paper_cmd.py` (add `_paper_account_equity()` and the `allocate` command; add imports)
- Test: `tests/test_cli_paper.py` (add three tests)

**Interfaces:**
- Consumes: `algua.registry.allocations.allocate(conn, strategy_id, capital, actor, account_equity)` and `AllocationError` (existing — `allocations.py`); `_alpaca_broker_from_settings()` (existing in `paper_cmd.py:69`) → `.account().equity`; `SqliteStrategyRepository`, `Stage`, `registry_conn`, `ok`, `emit`, `json_errors` (existing).
- Produces: CLI command `paper allocate <name> --capital $X`. Allocates only to a **`paper`-stage** strategy (the driver transitions `candidate→paper` first, then allocates); rejects any other stage. Enforces Σ paper allocations ≤ paper account equity (delegated to `allocations.allocate`). Emits `{"ok": true, "strategy": name, "capital": X}`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_cli_paper.py` (it already has `runner`, `_isolated`, and the `AlpacaPaperBroker.account` monkeypatch style — see `tests/test_cli_paper.py:153`). These mirror `tests/test_allocations.py` semantics through the CLI:

```python
# tests/test_cli_paper.py — append

def _paper_strategy(name="alpha"):
    # Promote a fresh strategy all the way to the `paper` stage via the repository, so allocate has
    # a legal target. Uses the same repo/transition helpers the other paper tests use.
    from algua.contracts.lifecycle import Actor, Stage
    from algua.registry.store import SqliteStrategyRepository
    from algua.registry.transitions import transition_strategy
    from algua.cli._common import registry_conn

    with registry_conn() as conn:
        repo = SqliteStrategyRepository(conn)
        repo.add(name)
        for to in (Stage.BACKTESTED, Stage.CANDIDATE, Stage.PAPER):
            transition_strategy(repo, name, to, Actor.HUMAN, reason="test setup")
    return name


def test_paper_allocate_sets_active_allocation(monkeypatch, tmp_path):
    _isolated(monkeypatch, tmp_path)
    monkeypatch.setenv("ALGUA_ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALGUA_ALPACA_API_SECRET", "s")
    monkeypatch.setattr(
        "algua.cli.paper_cmd.AlpacaPaperBroker.account",
        lambda self: AccountState(equity=50_000.0, cash=50_000.0, buying_power=50_000.0,
                                  account_id="paper-1"),
    )
    name = _paper_strategy()
    result = runner.invoke(app, ["paper", "allocate", name, "--capital", "10000"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert payload["capital"] == 10_000.0


def test_paper_allocate_rejects_non_paper_stage(monkeypatch, tmp_path):
    _isolated(monkeypatch, tmp_path)
    monkeypatch.setenv("ALGUA_ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALGUA_ALPACA_API_SECRET", "s")
    monkeypatch.setattr(
        "algua.cli.paper_cmd.AlpacaPaperBroker.account",
        lambda self: AccountState(equity=50_000.0, cash=50_000.0, buying_power=50_000.0,
                                  account_id="paper-1"),
    )
    from algua.cli._common import registry_conn
    from algua.registry.store import SqliteStrategyRepository
    with registry_conn() as conn:
        SqliteStrategyRepository(conn).add("idea_only")  # stays at `idea`
    result = runner.invoke(app, ["paper", "allocate", "idea_only", "--capital", "1000"])
    assert result.exit_code != 0
    assert json.loads(result.output)["ok"] is False


def test_paper_allocate_sum_capped_at_equity(monkeypatch, tmp_path):
    _isolated(monkeypatch, tmp_path)
    monkeypatch.setenv("ALGUA_ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALGUA_ALPACA_API_SECRET", "s")
    monkeypatch.setattr(
        "algua.cli.paper_cmd.AlpacaPaperBroker.account",
        lambda self: AccountState(equity=50_000.0, cash=50_000.0, buying_power=50_000.0,
                                  account_id="paper-1"),
    )
    a, b = _paper_strategy("a"), _paper_strategy("b")
    assert runner.invoke(app, ["paper", "allocate", a, "--capital", "40000"]).exit_code == 0
    over = runner.invoke(app, ["paper", "allocate", b, "--capital", "20000"])
    assert over.exit_code != 0
    assert "exceeds" in over.output
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cli_paper.py -k allocate -v`
Expected: FAIL — `No such command 'allocate'` (Typer rejects the subcommand) / non-zero exit.

- [ ] **Step 3: Add imports to `paper_cmd.py`**

At the top of `algua/cli/paper_cmd.py`, ensure these are imported (add only the ones missing — check the existing import block first):

```python
from algua.contracts.lifecycle import Stage
from algua.registry import allocations
from algua.registry.allocations import AllocationError
from algua.registry.store import SqliteStrategyRepository
```

- [ ] **Step 4: Add the helper + command**

Append to `algua/cli/paper_cmd.py` (near the other `@paper_app.command(...)` definitions):

```python
def _paper_account_equity() -> float:
    """Paper account equity = the Σ-allocation cap. Read once from the broker (read-only)."""
    return float(_alpaca_broker_from_settings().account().equity)


@paper_app.command("allocate")
@json_errors(ValueError, LookupError, AllocationError, BrokerError)
def allocate(
    name: str,
    capital: float = typer.Option(..., "--capital", help="paper capital base $"),
) -> None:
    """Set a paper strategy's capital base (its fixed sizing denominator). Enforces that the sum of
    all active paper allocations does not exceed paper-account equity. Allocate AFTER the strategy
    has reached the `paper` stage. (Paper-only box: no live allocations coexist in this table.)"""
    with registry_conn() as conn:
        rec = SqliteStrategyRepository(conn).get(name)  # unknown name -> LookupError -> {ok:false}
        if rec.stage is not Stage.PAPER:
            raise ValueError(
                f"can only allocate paper capital to a paper-stage strategy; "
                f"{name!r} is at stage {rec.stage.value}"
            )
        allocations.allocate(conn, rec.id, capital=capital, actor="agent",
                             account_equity=_paper_account_equity())
    emit(ok({"strategy": name, "capital": capital}))
```

Note: `BrokerError` is already imported in `paper_cmd.py` (used by `account`/`trade-tick`). If `typer` is not yet imported at module top, it is — `paper_cmd.py` defines other Typer commands.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli_paper.py -k allocate -v`
Expected: PASS (3 passed).

- [ ] **Step 6: Full quality gate**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: all green. (If ruff flags line length in the new test/command, wrap to ≤100 cols.)

- [ ] **Step 7: Commit**

```bash
git add algua/cli/paper_cmd.py tests/test_cli_paper.py
git commit -m "feat(paper): paper allocate — per-strategy capital base, Σ ≤ paper equity

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-review

- **Spec coverage (this slice):** §5 Group 1 — `paper allocate` (Task 2) and `is_due` (Task 1) are covered. `paper run-all` is intentionally deferred (see roadmap below). No other §-section is in scope for Slice 1.
- **Placeholder scan:** none — every step has real code/commands.
- **Type consistency:** `is_due(rebalance_frequency: str) -> bool` defined in Task 1 and referenced (future) by `run-all`; `allocate` consumes the existing `allocations.allocate(...)` signature verbatim from `algua/registry/allocations.py`; `AccountState(equity=...)` matches `algua/execution/alpaca_broker.py`.

---

## Roadmap — subsequent slices (each gets its own plan)

In Codex's verified dependency order; this plan delivers the prerequisites for Slice 2.

1. **Slice 1 (this plan)** — `paper allocate` + `is_due`. *Done when merged.*
2. **Slice 2 — `paper run-all`.** Tick the due set, sizing each strategy against its allocation (not full account equity). **Confronts the hard part:** correct per-strategy fill attribution on the shared paper account is entangled with the **open #249 paper-reconcile defect** (phantom reconcile breaches). This slice must either depend on, or carve out, the attributed-net paper-venue fill ledger — its own design note before coding.
3. **Slice 3 — research-cycle merge driver + file lock.** The crash-safe gate→merge→promote→allocate sequence (§6.1) with sweep-before-promote ordering (§6.3) and the load-bearing lock (§6.4).
4. **Slice 4 — agent NOVEL-family enablement** in `promotion.py` (§3.2 teardown).
5. **Slice 5 — quarantine-flag schema + allocator check** (the family-audit guard's teeth, §6.5).
6. **Slice 6 — always-on driver:** systemd timers, calendar gate, session-idempotency, ingest refresh, forward-promote wiring, alerts (§5 Group 4, §4).
7. **Slice 7 — analysis job** (§6 Group 6, optional).
