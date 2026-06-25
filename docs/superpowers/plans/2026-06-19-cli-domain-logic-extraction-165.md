# Issue #165 — CLI Domain-Logic Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lift domain logic out of the thin CLI command modules into the layers that own their dependencies, and remove every cli→cli sibling import, locking it in with an import-linter `independence` contract.

**Architecture:** Pure relocation — each helper moves to the layer that already imports its dependencies (verified cycle-free in the spec). No behavior changes; the existing CLI test suite is the safety net, and each genuinely-new pure function additionally gets a focused unit test. The new import-linter contract is added LAST, after all three cli→cli edges are removed, so `lint-imports` stays green at every commit.

**Tech Stack:** Python 3.12, Typer CLI, SQLite registry, pytest, ruff, mypy, import-linter.

**Spec:** `docs/superpowers/specs/2026-06-19-cli-domain-logic-extraction-165-design.md`

## Global Constraints

- Quality gate at every commit: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`. (For per-task speed you MAY run only the named tests + `ruff`/`mypy`/`lint-imports`; the FINAL task runs the full `pytest -q`.)
- `algua/contracts` and `algua/features` stay pure (no I/O, no cross-module imports beyond contracts).
- No backwards-compat shims, dead fallbacks, or dual code paths — move the symbol, update every call site, delete the original.
- Every new module starts with `from __future__ import annotations`.
- The `independence` import-linter contract (Task 8) must pass with ZERO `ignore_imports`.

---

### Task 1: `load_tradable_strategy` shared primitive (strategies/loader.py) + wire live

**Files:**
- Modify: `algua/strategies/loader.py` (add function; it already imports `LoadedStrategy` from `strategies.base`)
- Modify: `algua/cli/live_cmd.py:113-119` (`_run_strategy_tick` — replace inline load+asserts)
- Test: `tests/test_strategy_loader.py` (add a unit test)

**Interfaces:**
- Produces: `load_tradable_strategy(name: str) -> LoadedStrategy` in `algua.strategies.loader` — calls `load_strategy(name)` then `assert_tradable_without_fundamentals` and `assert_tradable_without_news`; raises `ValueError` for a `needs_fundamentals`/`needs_news` strategy.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_strategy_loader.py`:

```python
def test_load_tradable_strategy_loads_plain_strategy():
    from algua.strategies.loader import load_tradable_strategy
    s = load_tradable_strategy("cross_sectional_momentum")
    assert s.config.name == "cross_sectional_momentum"


def test_load_tradable_strategy_rejects_fundamentals_strategy(monkeypatch):
    # A needs_fundamentals strategy must be refused by the tradability gate.
    import algua.strategies.loader as loader
    from algua.strategies.base import LoadedStrategy

    sentinel = object()

    def _fake_load(name):
        return sentinel

    def _boom(strategy):
        assert strategy is sentinel
        raise ValueError("needs_fundamentals: not tradable without a fundamentals lane")

    monkeypatch.setattr(loader, "load_strategy", _fake_load)
    monkeypatch.setattr(loader, "assert_tradable_without_fundamentals", _boom)
    with pytest.raises(ValueError, match="needs_fundamentals"):
        loader.load_tradable_strategy("x")
```

Ensure `import pytest` is present at the top of the file.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_strategy_loader.py -k tradable -v`
Expected: FAIL with `ImportError`/`AttributeError` (`load_tradable_strategy` not defined).

- [ ] **Step 3: Implement the function**

In `algua/strategies/loader.py`, extend the base import and add the function. Change line 14:

```python
from algua.strategies.base import (
    LoadedStrategy,
    StrategyConfig,
    assert_tradable_without_fundamentals,
    assert_tradable_without_news,
)
```

Then add (after `load_strategy`):

```python
def load_tradable_strategy(name: str) -> LoadedStrategy:
    """Load a strategy AND assert it can trade off bars alone.

    The shared paper/live preamble: a ``needs_fundamentals``/``needs_news`` strategy has no
    paper/live data lane yet, so it must be refused before any order work. Kept beside
    ``load_strategy`` because the tradability assertions are a strategies-layer concern.
    """
    strategy = load_strategy(name)
    assert_tradable_without_fundamentals(strategy)
    assert_tradable_without_news(strategy)
    return strategy
```

- [ ] **Step 4: Run the unit test to verify it passes**

Run: `uv run pytest tests/test_strategy_loader.py -k tradable -v`
Expected: PASS.

- [ ] **Step 5: Wire `live_cmd._run_strategy_tick` to use it**

In `algua/cli/live_cmd.py`, replace the inline block at lines 113-119:

```python
    strategy = load_strategy(name)
    from algua.strategies.base import (
        assert_tradable_without_fundamentals,
        assert_tradable_without_news,
    )
    assert_tradable_without_fundamentals(strategy)
    assert_tradable_without_news(strategy)
```

with:

```python
    strategy = load_tradable_strategy(name)
```

Update the import at line 47 from `from algua.strategies.loader import load_strategy` to also expose the new symbol:

```python
from algua.strategies.loader import load_strategy, load_tradable_strategy
```

(`load_strategy` is still used by `resume` in live? No — it is used elsewhere; keep it only if another call site remains. Verify with `grep -n "load_strategy(" algua/cli/live_cmd.py`; if `_run_strategy_tick` was the only user, import just `load_tradable_strategy`.)

- [ ] **Step 6: Run the live CLI tests + gate**

Run: `uv run pytest tests/test_cli_live.py tests/test_strategy_loader.py -q && uv run ruff check algua/strategies/loader.py algua/cli/live_cmd.py && uv run mypy algua && uv run lint-imports`
Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add algua/strategies/loader.py algua/cli/live_cmd.py tests/test_strategy_loader.py
git commit -m "refactor(165): extract load_tradable_strategy shared primitive

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: `load_gated_strategy` → registry/gating.py (paper gate)

**Files:**
- Create: `algua/registry/gating.py`
- Modify: `algua/cli/paper_cmd.py` (import + replace `_load_gated_strategy` calls at :224 and :377; delete the def at :128-160)
- Test: `tests/test_registry_gating.py` (new)

**Interfaces:**
- Consumes: `load_tradable_strategy` (Task 1).
- Produces: `load_gated_strategy(conn: sqlite3.Connection, name: str, command: str) -> tuple[LoadedStrategy, StrategyRecord]` in `algua.registry.gating` — loads via `load_tradable_strategy`, then requires stage ∈ {PAPER, FORWARD_TESTED}, `not global_halt.is_engaged`, `not kill_switch.is_tripped`; else `ValueError`. `command` is a caller-supplied label that only colours the stage-error text.

- [ ] **Step 1: Write the failing test**

Create `tests/test_registry_gating.py`:

```python
import pytest

from algua.contracts.lifecycle import Stage
from algua.registry.db import connect, migrate
from algua.registry.gating import load_gated_strategy
from algua.registry.store import SqliteStrategyRepository
from algua.risk import global_halt, kill_switch


def _conn(tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    return conn


def _register_paper(conn, name="cross_sectional_momentum"):
    repo = SqliteStrategyRepository(conn)
    repo.add(name)
    repo.set_stage(name, Stage.PAPER)
    return repo


def test_load_gated_strategy_returns_strategy_and_record(tmp_path):
    conn = _conn(tmp_path)
    _register_paper(conn)
    strategy, rec = load_gated_strategy(conn, "cross_sectional_momentum", "paper run")
    assert strategy.config.name == "cross_sectional_momentum"
    assert rec.stage is Stage.PAPER


def test_load_gated_strategy_rejects_wrong_stage(tmp_path):
    conn = _conn(tmp_path)
    repo = SqliteStrategyRepository(conn)
    repo.add("cross_sectional_momentum")  # stage = idea
    with pytest.raises(ValueError, match="requires 'paper'"):
        load_gated_strategy(conn, "cross_sectional_momentum", "paper run")


def test_load_gated_strategy_rejects_tripped_kill_switch(tmp_path):
    conn = _conn(tmp_path)
    _register_paper(conn)
    kill_switch.trip(conn, "cross_sectional_momentum", reason="x", actor="agent")
    with pytest.raises(ValueError, match="kill-switch"):
        load_gated_strategy(conn, "cross_sectional_momentum", "paper run")


def test_load_gated_strategy_rejects_global_halt(tmp_path):
    conn = _conn(tmp_path)
    _register_paper(conn)
    global_halt.engage(conn, reason="x", actor="agent")
    with pytest.raises(ValueError, match="global halt"):
        load_gated_strategy(conn, "cross_sectional_momentum", "paper run")
```

> NOTE to implementer: confirm the exact `SqliteStrategyRepository` registration API (`add` / `set_stage` names + the transition rules) with `grep -n "def add\|def set_stage\|def get\b" algua/registry/store.py algua/registry/repository.py`. If direct `set_stage` to PAPER is disallowed, register via the CliRunner helper `_to_paper` pattern from `tests/test_cli_paper.py` instead.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_registry_gating.py -v`
Expected: FAIL (`ModuleNotFoundError: algua.registry.gating`).

- [ ] **Step 3: Create the module**

Create `algua/registry/gating.py` (move the body verbatim from `paper_cmd._load_gated_strategy`, swapping the inline load+asserts for `load_tradable_strategy`):

```python
from __future__ import annotations

import sqlite3

from algua.contracts.lifecycle import Stage
from algua.registry.repository import StrategyRecord
from algua.registry.store import SqliteStrategyRepository
from algua.risk import global_halt, kill_switch
from algua.strategies.base import LoadedStrategy
from algua.strategies.loader import load_tradable_strategy


def load_gated_strategy(
    conn: sqlite3.Connection, name: str, command: str,
) -> tuple[LoadedStrategy, StrategyRecord]:
    """Load a strategy and clear the two gates every paper trading command shares: it must be at
    the PAPER or FORWARD_TESTED stage and its kill-switch (and the global halt) must not be
    engaged. ``command`` is a caller-supplied label that only colours the stage-error text.

    Returns ``(strategy, rec)`` so callers can read the registry record (e.g. ``rec.id``) without
    a second DB round-trip. A forward_tested strategy keeps accumulating evidence ticks while
    awaiting the go-live signature, so it is treated the same as paper for trading purposes.

    Lives in ``registry`` (not ``cli``) so any non-CLI consumer shares the SAME gate — paper/live
    gating can no longer drift via a copy in a command module.
    """
    strategy = load_tradable_strategy(name)
    rec = SqliteStrategyRepository(conn).get(name)
    if rec.stage not in (Stage.PAPER, Stage.FORWARD_TESTED):
        raise ValueError(
            f"{name} is at stage '{rec.stage.value}'; "
            f"{command} requires 'paper' or 'forward_tested'"
        )
    if global_halt.is_engaged(conn):
        raise ValueError("global halt active; clear with 'algua paper resume-all'")
    if kill_switch.is_tripped(conn, name):
        raise ValueError(f"kill-switch tripped for {name}; reset with 'algua paper resume {name}'")
    return strategy, rec
```

- [ ] **Step 4: Run the unit test to verify it passes**

Run: `uv run pytest tests/test_registry_gating.py -v`
Expected: PASS.

- [ ] **Step 5: Rewire `paper_cmd` and delete the old def**

In `algua/cli/paper_cmd.py`:
1. Delete the `_load_gated_strategy` def (lines 128-160) AND its now-unused local imports of `assert_tradable_without_*` (they moved into `load_tradable_strategy`).
2. Add `from algua.registry.gating import load_gated_strategy`.
3. Replace both call sites:
   - `:224` `strategy, _rec = _load_gated_strategy(conn, name, "paper run")` → `strategy, _rec = load_gated_strategy(conn, name, "paper run")`
   - `:377` `strategy, rec = _load_gated_strategy(conn, name, "trade-tick")` → `strategy, rec = load_gated_strategy(conn, name, "trade-tick")`
4. If `load_strategy` is now unused in `paper_cmd` except by `resume`/`flatten`, keep its import (those two still call it directly — see spec; do NOT change them).

- [ ] **Step 6: Run paper CLI tests + gate**

Run: `uv run pytest tests/test_cli_paper.py tests/test_registry_gating.py -q && uv run ruff check algua/registry/gating.py algua/cli/paper_cmd.py && uv run mypy algua && uv run lint-imports`
Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add algua/registry/gating.py algua/cli/paper_cmd.py tests/test_registry_gating.py
git commit -m "refactor(165): move paper gate to registry/gating.load_gated_strategy

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: `tick_clock` → execution/tick_clock.py

**Files:**
- Create: `algua/execution/tick_clock.py`
- Modify: `algua/cli/paper_cmd.py` (import + replace `_tick_clock` at :432; delete def at :163-173)
- Modify: `algua/cli/live_cmd.py` (import + replace `_tick_clock` at :193; shrink the paper_cmd import line)
- Test: `tests/test_tick_clock.py` (new)

**Interfaces:**
- Produces: `tick_clock(clock: Callable[[], str]) -> tuple[str, str]` in `algua.execution.tick_clock` — returns `(utc_iso, "broker")` from the venue clock, or `(now_utc_iso, "local")` if the clock raises `BrokerError`/`ValueError`/`TypeError` or yields a tz-naive/malformed timestamp.

- [ ] **Step 1: Write the failing test**

Create `tests/test_tick_clock.py`:

```python
import pandas as pd

from algua.execution.alpaca_broker import BrokerError
from algua.execution.tick_clock import tick_clock


def test_tick_clock_uses_broker_clock_when_valid():
    ts, source = tick_clock(lambda: "2023-06-01T14:30:00+00:00")
    assert source == "broker"
    assert ts == pd.Timestamp("2023-06-01T14:30:00+00:00").tz_convert("UTC").isoformat()


def test_tick_clock_falls_back_on_broker_error():
    def _boom():
        raise BrokerError("clock down")
    ts, source = tick_clock(_boom)
    assert source == "local"
    assert ts.endswith("+00:00") or "T" in ts


def test_tick_clock_falls_back_on_naive_timestamp():
    # tz-naive -> tz_convert raises TypeError -> local fallback
    ts, source = tick_clock(lambda: "2023-06-01T14:30:00")
    assert source == "local"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tick_clock.py -v`
Expected: FAIL (`ModuleNotFoundError: algua.execution.tick_clock`).

- [ ] **Step 3: Create the module** (move body verbatim from `paper_cmd._tick_clock`)

```python
from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

import pandas as pd

from algua.execution.alpaca_broker import BrokerError


def tick_clock(clock: Callable[[], str]) -> tuple[str, str]:
    """``(tick_ts, clock_source)`` for the evidence-tick stamp: the venue's clock normalized to a
    UTC ISO timestamp (``clock_source="broker"``), or the local clock (``clock_source="local"``)
    when the venue's is unusable. ValueError/TypeError cover a malformed or tz-naive venue
    timestamp — that is a clock failure too, and it must never kill the tick record after orders
    already went out. Shared by the paper and live lanes so their stamping semantics cannot drift.

    Coupling note: imports ``BrokerError`` from ``alpaca_broker`` — not broker-agnostic today
    (only one broker exists; extracting a shared exceptions leaf is deferred — YAGNI).
    """
    try:
        return pd.Timestamp(clock()).tz_convert("UTC").isoformat(), "broker"
    except (BrokerError, ValueError, TypeError):
        return datetime.now(UTC).isoformat(), "local"
```

- [ ] **Step 4: Run the unit test to verify it passes**

Run: `uv run pytest tests/test_tick_clock.py -v`
Expected: PASS.

- [ ] **Step 5: Rewire both lanes, delete old def, shrink the live→paper import**

- `algua/cli/paper_cmd.py`: delete the `_tick_clock` def (lines 163-173); add `from algua.execution.tick_clock import tick_clock`; replace `_tick_clock(broker.clock)` at :432 with `tick_clock(broker.clock)`.
- `algua/cli/live_cmd.py`: add `from algua.execution.tick_clock import tick_clock`; replace `_tick_clock(broker.clock)` at :193 with `tick_clock(broker.clock)`; change the line `from algua.cli.paper_cmd import _breach_payload, _tick_clock, _trip` to `from algua.cli.paper_cmd import _breach_payload, _trip` (drops `_tick_clock` only — `_breach_payload`/`_trip` leave in Task 4).

- [ ] **Step 6: Run tests + gate**

Run: `uv run pytest tests/test_cli_paper.py tests/test_cli_live.py tests/test_tick_clock.py -q && uv run ruff check algua/execution/tick_clock.py algua/cli/paper_cmd.py algua/cli/live_cmd.py && uv run mypy algua && uv run lint-imports`
Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add algua/execution/tick_clock.py algua/cli/paper_cmd.py algua/cli/live_cmd.py tests/test_tick_clock.py
git commit -m "refactor(165): move tick_clock to execution/tick_clock

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: breach handling → `risk/breach.py` (`trip_for_breach`) + `cli/_common.py` (`breach_payload`); remove the live→paper import

**Files:**
- Create: `algua/risk/breach.py`
- Modify: `algua/cli/_common.py` (add `breach_payload`)
- Modify: `algua/cli/paper_cmd.py` (imports + replace `_trip`/`_breach_payload`; delete defs at :176-190)
- Modify: `algua/cli/live_cmd.py` (imports + replace; **delete the `from algua.cli.paper_cmd import …` line entirely**)
- Test: `tests/test_risk_breach.py` (new), `tests/test_cli_common.py` (new or append)

**Interfaces:**
- Produces: `trip_for_breach(conn: sqlite3.Connection, name: str, exc: RiskBreach) -> None` in `algua.risk.breach` — trips the kill-switch THEN appends a `kill_switch_trip` audit row (mutate-then-audit; must NOT be reversed).
- Produces: `breach_payload(error: str, **extra: object) -> dict` in `algua.cli._common` — returns `{"ok": False, "kill_switch": "tripped", "error": error, **extra}`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_risk_breach.py`:

```python
from algua.audit import log as audit
from algua.registry.db import connect, migrate
from algua.risk import kill_switch
from algua.risk.breach import trip_for_breach
from algua.risk.limits import RiskBreach


def _conn(tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    return conn


def test_trip_for_breach_trips_and_audits(tmp_path):
    conn = _conn(tmp_path)
    trip_for_breach(conn, "s", RiskBreach("drawdown", "equity below floor"))
    assert kill_switch.is_tripped(conn, "s") is True
    rows = audit.read(conn, strategy="s")
    assert any(r["action"] == "kill_switch_trip" and "drawdown" in r["reason"] for r in rows)
```

Create (or append to) `tests/test_cli_common.py`:

```python
from algua.cli._common import breach_payload


def test_breach_payload_shape():
    p = breach_payload("boom", kind="drawdown", strategy="s")
    assert p == {"ok": False, "kill_switch": "tripped", "error": "boom",
                 "kind": "drawdown", "strategy": "s"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_risk_breach.py tests/test_cli_common.py -v`
Expected: FAIL (`ModuleNotFoundError` / `ImportError`).

- [ ] **Step 3: Create `risk/breach.py`** (move `_trip` verbatim, rename)

```python
from __future__ import annotations

import sqlite3

from algua.audit.log import append as audit_append
from algua.risk import kill_switch
from algua.risk.limits import RiskBreach


def trip_for_breach(conn: sqlite3.Connection, name: str, exc: RiskBreach) -> None:
    """Trip the kill-switch for a risk breach and write the matching audit row.

    Ordering is INTENTIONAL: mutate (trip) THEN audit. For a trip, fail-safe means the switch is
    persisted (halted) even if the audit write then fails; reversing to audit-first would leave the
    worse state (audited-as-tripped but switch unpersisted). The divergent emit/flatten stays in
    each caller.
    """
    kill_switch.trip(conn, name, reason=exc.detail, actor="system")
    audit_append(conn, actor="system", action="kill_switch_trip",
                 reason=f"{exc.kind}: {exc.detail}", strategy=name)
```

- [ ] **Step 4: Add `breach_payload` to `cli/_common.py`** (move `_breach_payload` verbatim, rename)

```python
def breach_payload(error: str, **extra: object) -> dict:
    """A failure envelope for a tripped kill-switch: ``{"ok": false, "kill_switch": "tripped"...}``.

    The shared skeleton of every paper/live-command halt/breach emit; callers pass the
    human-readable ``error`` plus whatever variant keys (``kind``, ``strategy``, ``halted``, ...)
    that path adds. Pure presentation — lives beside ``ok`` in the CLI infrastructure, not in a
    command module (so paper and live share it without a cli→cli import).
    """
    return {"ok": False, "kill_switch": "tripped", "error": error, **extra}
```

- [ ] **Step 5: Run the new unit tests to verify they pass**

Run: `uv run pytest tests/test_risk_breach.py tests/test_cli_common.py -v`
Expected: PASS.

- [ ] **Step 6: Rewire `paper_cmd`, delete its defs**

In `algua/cli/paper_cmd.py`:
1. Delete the `_breach_payload` (lines 176-182) and `_trip` (lines 185-190) defs.
2. Add imports: `from algua.risk.breach import trip_for_breach` and add `breach_payload` to the existing `from algua.cli._common import ok, registry_conn, utc` line → `from algua.cli._common import breach_payload, ok, registry_conn, utc`.
3. Replace every `_trip(` → `trip_for_breach(` (at :230, :408) and every `_breach_payload(` → `breach_payload(` (at :231, :405, :423, :561).

- [ ] **Step 7: Rewire `live_cmd`, remove the cli→cli import line**

In `algua/cli/live_cmd.py`:
1. Delete the line `from algua.cli.paper_cmd import _breach_payload, _trip` entirely.
2. Add `from algua.risk.breach import trip_for_breach`; add `breach_payload` to the existing `from algua.cli._common import ok, registry_conn, utc` import.
3. Replace `_trip(` → `trip_for_breach(` (at :161) and `_breach_payload(` → `breach_payload(` (at :158, :181, :258).

- [ ] **Step 8: Run tests + gate (verify the live→paper import is gone)**

Run: `uv run pytest tests/test_cli_paper.py tests/test_cli_live.py tests/test_risk_breach.py tests/test_cli_common.py -q && uv run ruff check algua/risk/breach.py algua/cli/_common.py algua/cli/paper_cmd.py algua/cli/live_cmd.py && uv run mypy algua && uv run lint-imports`
Expected: all PASS.
Verify: `grep -n "from algua.cli.paper_cmd" algua/cli/live_cmd.py` returns nothing.

- [ ] **Step 9: Commit**

```bash
git add algua/risk/breach.py algua/cli/_common.py algua/cli/paper_cmd.py algua/cli/live_cmd.py tests/test_risk_breach.py tests/test_cli_common.py
git commit -m "refactor(165): breach handling to risk/breach + cli/_common; drop live->paper import

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: `record_search_breadth` → registry/search_breadth.py

**Files:**
- Create: `algua/registry/search_breadth.py`
- Modify: `algua/cli/backtest_cmd.py` (import + replace call at :228; delete def at :240-257)
- Test: `tests/test_registry_search_breadth.py` (new)

**Interfaces:**
- Produces: `record_search_breadth(repo: SqliteStrategyRepository, name: str, result: SweepResult) -> dict[str, int]` in `algua.registry.search_breadth` — calls `repo.record_search_trial(...)` then returns `{"n_combos": result.n_combos, "cumulative": repo.total_search_combos(name)}`. Caller owns the `registry_conn` transaction.

- [ ] **Step 1: Write the failing test**

Create `tests/test_registry_search_breadth.py`:

```python
from algua.backtest.sweep import SweepResult
from algua.registry.db import connect, migrate
from algua.registry.search_breadth import record_search_breadth
from algua.registry.store import SqliteStrategyRepository


def _conn(tmp_path):
    conn = connect(tmp_path / "r.db")
    migrate(conn)
    return conn


def test_record_search_breadth_records_and_returns_cumulative(tmp_path):
    conn = _conn(tmp_path)
    repo = SqliteStrategyRepository(conn)
    # Confirm the SweepResult fields with: grep -n "class SweepResult\|n_combos\|grid\|trial_sharpe" algua/backtest/sweep.py
    result = SweepResult(...)  # IMPLEMENTER: build a minimal SweepResult per its actual ctor
    out = record_search_breadth(repo, "momo", result)
    assert out["n_combos"] == result.n_combos
    assert out["cumulative"] == result.n_combos
    out2 = record_search_breadth(repo, "momo", result)
    assert out2["cumulative"] == 2 * result.n_combos
```

> IMPLEMENTER: inspect `SweepResult` (`grep -n "class SweepResult" -A20 algua/backtest/sweep.py`) and the existing `test_sweep*.py` to construct a minimal instance; reuse a factory if those tests have one.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_registry_search_breadth.py -v`
Expected: FAIL (`ModuleNotFoundError: algua.registry.search_breadth`).

- [ ] **Step 3: Create the module** (move body from `backtest_cmd._record_search_breadth`, swap self-opened conn for injected `repo`)

```python
from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from algua.backtest.sweep import SweepResult
    from algua.registry.store import SqliteStrategyRepository


def record_search_breadth(
    repo: SqliteStrategyRepository, name: str, result: SweepResult,
) -> dict[str, int]:
    """Record this sweep's measured breadth into the registry, keyed by strategy NAME.

    Recorded UNCONDITIONALLY — even for a not-yet-registered strategy. Exploration precedes
    registration: keying by name (not the registry id) means a pre-registration sweep still
    counts toward the promotion breadth, so an agent can't sweep broadly first and then promote a
    freshly-registered strategy under a smaller declared --n-combos. Returns the recorded count
    plus the new cumulative family total for the emitted JSON.

    The transaction is CALLER-OWNED: the CLI wraps this in ``with registry_conn() as conn:`` and
    passes ``SqliteStrategyRepository(conn)``.
    """
    repo.record_search_trial(
        name, result.n_combos, json.dumps(result.grid, sort_keys=True),
        trial_sharpe_count=result.trial_sharpe_count,
        trial_sharpe_mean=result.trial_sharpe_mean,
        trial_sharpe_var_ann=result.trial_sharpe_var_ann,
    )
    return {"n_combos": result.n_combos, "cumulative": repo.total_search_combos(name)}
```

- [ ] **Step 4: Run the unit test to verify it passes**

Run: `uv run pytest tests/test_registry_search_breadth.py -v`
Expected: PASS.

- [ ] **Step 5: Rewire `backtest_cmd`, delete the old def**

In `algua/cli/backtest_cmd.py`:
1. Delete the `_record_search_breadth` def (lines 240-257).
2. Add imports: `from algua.registry.search_breadth import record_search_breadth` and (if not already imported) `from algua.cli._common import registry_conn` and `from algua.registry.store import SqliteStrategyRepository`. Verify `json` is still needed elsewhere in the file; if not, remove the now-unused `import json`.
3. Replace the call at :228 `recorded = _record_search_breadth(name, result)` with:

```python
    with registry_conn() as conn:
        recorded = record_search_breadth(SqliteStrategyRepository(conn), name, result)
```

- [ ] **Step 6: Run tests + gate**

Run: `uv run pytest tests/test_cli_sweep.py tests/test_registry_search_breadth.py -q && uv run ruff check algua/registry/search_breadth.py algua/cli/backtest_cmd.py && uv run mypy algua && uv run lint-imports`
Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add algua/registry/search_breadth.py algua/cli/backtest_cmd.py tests/test_registry_search_breadth.py
git commit -m "refactor(165): move record_search_breadth to registry/search_breadth

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: `kb_metadata` → registry/repository.py; remove strategy→registry_cmd import

**Files:**
- Modify: `algua/registry/repository.py` (add `kb_metadata` free function beside `StrategyRecord`)
- Modify: `algua/cli/registry_cmd.py` (import + replace `_kb_metadata` at :209; delete def at :37-44)
- Modify: `algua/cli/strategy_cmd.py` (swap the cli→cli import at :12; 3 call sites at :159/:198/:200 unchanged in body)
- Test: `tests/test_registry_repository.py` (new or append)

**Interfaces:**
- Produces: `kb_metadata(rec: StrategyRecord) -> dict` in `algua.registry.repository` — returns `{"family", "tags", "author", "hypothesis_status", "derived_from", "description"}` (registry-owned kb frontmatter; no id/name/stage).

- [ ] **Step 1: Write the failing test**

Create (or append to) `tests/test_registry_repository.py`:

```python
from algua.contracts.registry_metadata import Author, HypothesisStatus
from algua.registry.repository import StrategyRecord, kb_metadata


def test_kb_metadata_projects_frontmatter_fields():
    # IMPLEMENTER: confirm StrategyRecord's required ctor args with
    # `grep -n "class StrategyRecord" -A25 algua/registry/repository.py` and fill them in.
    rec = StrategyRecord(...)
    meta = kb_metadata(rec)
    assert set(meta) == {"family", "tags", "author", "hypothesis_status",
                         "derived_from", "description"}
    assert "id" not in meta and "stage" not in meta
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_registry_repository.py -k kb_metadata -v`
Expected: FAIL (`ImportError: cannot import name 'kb_metadata'`).

- [ ] **Step 3: Add the function to `registry/repository.py`** (move body verbatim from `registry_cmd._kb_metadata`, drop the leading underscore)

```python
def kb_metadata(rec: StrategyRecord) -> dict:
    """Return the registry-owned frontmatter fields for kb sync (no id/name/stage).

    Lives beside ``StrategyRecord`` so both ``registry_cmd`` and ``strategy_cmd`` import it from the
    registry layer rather than from each other.
    """
    return {
        "family": rec.family, "tags": rec.tags, "author": rec.author.value,
        "hypothesis_status": rec.hypothesis_status.value,
        "derived_from": rec.derived_from, "description": rec.description,
    }
```

- [ ] **Step 4: Run the unit test to verify it passes**

Run: `uv run pytest tests/test_registry_repository.py -k kb_metadata -v`
Expected: PASS.

- [ ] **Step 5: Rewire both CLI modules, delete the old def**

- `algua/cli/registry_cmd.py`: delete the `_kb_metadata` def (lines 37-44); add `kb_metadata` to the existing `from algua.registry.repository import StrategyRecord` import → `from algua.registry.repository import StrategyRecord, kb_metadata`; replace `_kb_metadata(after)` at :209 with `kb_metadata(after)`.
- `algua/cli/strategy_cmd.py`: replace `from algua.cli.registry_cmd import _kb_metadata` (line 12) with `from algua.registry.repository import kb_metadata`; replace `_kb_metadata(` → `kb_metadata(` at :159, :198, :200.

- [ ] **Step 6: Run tests + gate (verify strategy→registry_cmd import is gone)**

Run: `uv run pytest tests/test_cli_registry.py tests/test_cli_strategy.py tests/test_registry_repository.py -q && uv run ruff check algua/registry/repository.py algua/cli/registry_cmd.py algua/cli/strategy_cmd.py && uv run mypy algua && uv run lint-imports`
Expected: all PASS.
Verify: `grep -n "from algua.cli.registry_cmd" algua/cli/strategy_cmd.py` returns nothing.

- [ ] **Step 7: Commit**

```bash
git add algua/registry/repository.py algua/cli/registry_cmd.py algua/cli/strategy_cmd.py tests/test_registry_repository.py
git commit -m "refactor(165): move kb_metadata to registry/repository; drop strategy->registry_cmd import

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7: Composition root — hoist idea→research mount into main.py

**Files:**
- Modify: `algua/cli/idea_cmd.py` (remove `research_app` import + the mount at :21)
- Modify: `algua/cli/main.py` (add the mount after the import block)
- Test: existing `tests/test_cli_idea.py` (must still pass — it exercises `algua research idea …`)

**Interfaces:**
- Consumes: `research_cmd.research_app` (Typer), `idea_cmd.idea_app` (Typer).
- Produces: the `algua research idea …` command path, now wired in the composition root.

- [ ] **Step 1: Remove the cross-import + self-mount from `idea_cmd.py`**

In `algua/cli/idea_cmd.py`:
1. Delete the import `from algua.cli.research_cmd import research_app` (line 9/10).
2. Delete the mount line `research_app.add_typer(idea_app, name="idea")` (line 21). Keep the `idea_app = typer.Typer(...)` definition.

- [ ] **Step 2: Add the mount to the composition root `main.py`**

In `algua/cli/main.py`, immediately AFTER the `from algua.cli import (…)` block (the modules must be imported first) and before `main()` is defined, add:

```python
# Composition root: mount idea_app under research_app HERE (not inside idea_cmd) so no cli command
# module imports a sibling. Typer builds the command tree lazily at get_command(app) inside main(),
# so this only needs to run before that call. MUST stay after the `from algua.cli import (…)` block.
research_cmd.research_app.add_typer(idea_cmd.idea_app, name="idea")
```

Note: the existing import is `from algua.cli import (backtest_cmd, … idea_cmd, … research_cmd, …)` so `research_cmd` and `idea_cmd` are in scope as module objects.

- [ ] **Step 3: Run the idea CLI tests to verify the command path survives**

Run: `uv run pytest tests/test_cli_idea.py -q`
Expected: PASS (the `algua research idea …` subcommands resolve exactly as before).

- [ ] **Step 4: Gate (verify idea→research import is gone)**

Run: `uv run ruff check algua/cli/idea_cmd.py algua/cli/main.py && uv run mypy algua && uv run lint-imports`
Expected: all PASS.
Verify: `grep -n "from algua.cli.research_cmd" algua/cli/idea_cmd.py` returns nothing.

- [ ] **Step 5: Commit**

```bash
git add algua/cli/idea_cmd.py algua/cli/main.py
git commit -m "refactor(165): hoist idea->research Typer mount into the main.py composition root

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 8: Lock it in — import-linter `independence` contract + full gate

**Files:**
- Modify: `pyproject.toml` (add the contract under `[tool.importlinter]`)

**Interfaces:**
- Consumes: the three removed cli→cli edges (Tasks 4, 6, 7).

- [ ] **Step 1: Add the contract**

Append to `pyproject.toml` (after the last existing `[[tool.importlinter.contracts]]`):

```toml
[[tool.importlinter.contracts]]
# The CLI is a thin JSON seam: command modules must not import one another (domain logic belongs in
# the layer that owns it, shared CLI helpers in cli._common/app/errors). This kills paper/live
# gating drift at the structural level (issue #165). Shared infra (app, _common, errors, main) is
# NOT listed, so command modules may still import it.
name = "cli command modules are independent of one another (no cli->cli sibling imports)"
type = "independence"
modules = [
    "algua.cli.backtest_cmd",
    "algua.cli.data_cmd",
    "algua.cli.factor_cmd",
    "algua.cli.idea_cmd",
    "algua.cli.live_cmd",
    "algua.cli.paper_cmd",
    "algua.cli.registry_cmd",
    "algua.cli.research_cmd",
    "algua.cli.strategy_cmd",
]
```

- [ ] **Step 2: Run lint-imports to verify the contract passes with zero exceptions**

Run: `uv run lint-imports`
Expected: PASS — all contracts kept, including the new independence contract (no `ignore_imports` needed).

- [ ] **Step 3: Prove the contract has teeth (temporary negative check)**

Temporarily add `from algua.cli import paper_cmd  # noqa` to the top of `algua/cli/live_cmd.py`, run `uv run lint-imports`, and confirm it now FAILS on the independence contract. Then REMOVE the temporary line and re-run to confirm PASS again. (Do not commit the temporary line.)

- [ ] **Step 4: Run the FULL quality gate**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml
git commit -m "refactor(165): forbid cli->cli sibling imports via import-linter independence contract

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review notes (author)

- **Spec coverage:** all 6 symbol moves (Tasks 1–6), the composition-root fix (Task 7), and the import-linter contract (Task 8) are covered; `breach_payload`→`cli/_common`, `load_strategy`-kept-by-`flatten`/`resume`, mutate-then-audit ordering, and the `main.py` mount-ordering note are all reflected.
- **Edge-removal ordering:** the three cli→cli edges are removed in Tasks 4 (live→paper), 6 (strategy→registry), 7 (idea→research); the contract (Task 8) is added only after — so `lint-imports` stays green at every commit.
- **Implementer verification hooks:** Tasks 2/5/6 flag the exact `grep` to confirm `SqliteStrategyRepository`/`SweepResult`/`StrategyRecord` constructor APIs before finalizing the test fixtures (those are the only spots the plan can't fully pin without reading the class bodies).
