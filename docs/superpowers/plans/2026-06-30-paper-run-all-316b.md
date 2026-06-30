# paper run-all: multi-tenant batch + scoped cancel (#316b) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `paper run-all` — a per-session batch that ticks every active paper strategy on the shared account (mirror of `live run-all`), reusing #316a's `_run_paper_strategy_tick`, and land the scoped-cancel fix the batch requires.

**Architecture:** Generalize `owned_open_order_ids` by `LedgerKind`; add `_paper_scoped_cancel`; harden `_run_paper_strategy_tick` (add `reserve_buy` hook; scope the breach-handler cancel via the supplied `cancel`); add `paper run-all` (ingest-once → reconcile-once → shared buying-power pool → loop the helper → one envelope).

**Tech Stack:** Python 3.12, Typer, SQLite, pytest. Builds on #316a (PR#322, stacked), #313, #314, the allocations module. Decoupled from #288 (ticks all paper strategies; `is_due` deferred).

## Global Constraints

- Run via `uv run ...`. Full gate green before each commit: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`.
- Ruff ≤ 100 columns.
- `git add` only the named files — never `git add -A`.
- Commit trailer: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- **Do not touch the live lane**; do not import `live_cmd` from `paper_cmd` (cli→cli banned).
- run-all ticks **all** `Stage.PAPER` strategies (no `is_due` — deferred; all daily/XNYS today).
- A not-clean account reconcile **defers the whole cycle** (no trades); `halt` engages the global halt. Only a clean reconcile ticks. Mirror `live run-all`.
- Branch: `paper-run-all-316b` (stacked on `paper-multitenant-tick-316a`; rebase onto main after #322 merges).

---

## File structure

- `algua/execution/live_ledger.py` — **Modify.** Generalize `owned_open_order_ids(..., *, kind=LedgerKind.LIVE)`.
- `algua/cli/paper_cmd.py` — **Modify.** Harden `_run_paper_strategy_tick`; add `_paper_scoped_cancel`, `_paper_reserve_for`/pool, the `paper run-all` command.
- `tests/test_live_ledger_ledgerkind.py` — **Modify.** `owned_open_order_ids(kind=PAPER)` test.
- `tests/test_paper_run_all.py` — **Create.** run-all batch tests.
- `tests/test_cli_paper.py` — **Modify.** helper-hardening test (breach uses scoped `cancel`).

---

### Task 1: Generalize `owned_open_order_ids` by `LedgerKind`

**Files:**
- Modify: `algua/execution/live_ledger.py`
- Test: `tests/test_live_ledger_ledgerkind.py`

**Interfaces:**
- Consumes: `LedgerKind`, `_TABLES` (private, same module).
- Produces: `owned_open_order_ids(conn, broker, strategy, *, kind: LedgerKind = LedgerKind.LIVE) -> list[str]` — broker order ids of THIS strategy's open orders, read from `_TABLES[kind].orders`. Live caller (default kind) unchanged.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_live_ledger_ledgerkind.py`:

```python
def test_owned_open_order_ids_paper_kind_reads_paper_venue_orders(tmp_path):
    from algua.execution.live_ledger import LedgerKind, owned_open_order_ids, record_paper_venue_order
    from algua.registry.db import connect, migrate

    conn = connect(tmp_path / "r.db")
    migrate(conn)
    conn.execute("INSERT INTO strategies(name, stage, created_at, updated_at) VALUES "
                 "('s1','paper','2026-01-01','2026-01-01'),('s2','paper','2026-01-01','2026-01-01')")
    conn.commit()
    record_paper_venue_order(conn, "s1", "AAA", "buy", None, "coid-s1", strategy_id=1)
    record_paper_venue_order(conn, "s2", "BBB", "buy", None, "coid-s2", strategy_id=2)

    class _B:
        def list_open_orders(self):
            return [{"id": "o1", "client_order_id": "coid-s1"},
                    {"id": "o2", "client_order_id": "coid-s2"}]

    assert owned_open_order_ids(conn, _B(), "s1", kind=LedgerKind.PAPER) == ["o1"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_live_ledger_ledgerkind.py::test_owned_open_order_ids_paper_kind_reads_paper_venue_orders -v`
Expected: FAIL — `owned_open_order_ids() got an unexpected keyword argument 'kind'`.

- [ ] **Step 3: Generalize the function**

In `algua/execution/live_ledger.py`, change `owned_open_order_ids` to take a keyword-only `kind` and read the kind's orders table:

```python
def owned_open_order_ids(
    conn: sqlite3.Connection, broker: object, strategy: str,
    *, kind: LedgerKind = LedgerKind.LIVE,
) -> list[str]:
    """The broker order ids of THIS strategy's currently-open orders: list the account's open
    orders and keep those whose client_order_id maps (via the kind's order ledger) to `strategy`.
    Used to scope cancellation so one strategy never cancels a sibling's orders. `kind` selects the
    order ledger (live_orders / paper_venue_orders)."""
    open_orders = broker.list_open_orders()  # type: ignore[attr-defined]
    owned = {
        r["client_order_id"]
        for r in conn.execute(
            f"SELECT client_order_id FROM {_TABLES[kind].orders} WHERE strategy = ?", (strategy,)
        )
    }
    return [o["id"] for o in open_orders if o.get("client_order_id") in owned]
```

- [ ] **Step 4: Run test + the live regression**

Run: `uv run pytest tests/test_live_ledger_ledgerkind.py tests/test_cli_live.py -k "owned or cancel or run_all" -v`
Expected: PASS — new test passes; the live caller (default kind) is unaffected.

- [ ] **Step 5: Lint/type + commit**

```bash
uv run ruff check algua/execution/live_ledger.py tests/test_live_ledger_ledgerkind.py && uv run mypy algua/execution/live_ledger.py
git add algua/execution/live_ledger.py tests/test_live_ledger_ledgerkind.py
git commit -m "feat(execution): owned_open_order_ids parameterized by LedgerKind #316b

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Harden `_run_paper_strategy_tick` (reserve_buy hook + scoped breach cancel)

**Files:**
- Modify: `algua/cli/paper_cmd.py`
- Test: `tests/test_cli_paper.py`

**Interfaces:**
- Consumes: existing `_run_paper_strategy_tick`, `TickHooks` (has a `reserve_buy` field).
- Produces: `_run_paper_strategy_tick(..., *, cancel=None, reserve_buy=None, start=..., end=...)` — now forwards `reserve_buy` into `TickHooks`, and its breach handler cancels via the supplied `cancel` callable when present (account-wide only as the fallback).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_cli_paper.py` (reuse the existing `_run_paper_strategy_tick`-driving fakes/fixtures; a `force_breach` broker triggers the breach path):

```python
def test_run_paper_strategy_tick_breach_uses_scoped_cancel(monkeypatch, tmp_path):
    # On breach, when a `cancel` callable is supplied, the helper must call IT (scoped),
    # not the broker's account-wide cancel_open_orders().
    name = _paper_strategy_with_allocation(monkeypatch, tmp_path, capital=10_000.0,
                                           account_equity=100_000.0)
    broker = _FakePaperBroker(account_equity=100_000.0, positions={}, marks={"AAA": 100.0},
                              force_breach=True)
    called = {"scoped": 0}
    from algua.cli import paper_cmd
    from algua.cli._common import registry_conn
    from algua.registry.store import SqliteStrategyRepository
    with registry_conn() as conn:
        rec = SqliteStrategyRepository(conn).get(name)
        strategy, _ = _load_for_tick(conn, name)  # existing helper / load_gated_strategy
        acct = broker.account()
        out = paper_cmd._run_paper_strategy_tick(
            conn, name, strategy, rec, broker, provider_for(_SNAP), 0.01,
            "tick-ts", "broker", acct,
            cancel=lambda: called.__setitem__("scoped", called["scoped"] + 1),
            start="2026-01-01", end="2026-02-01")
    assert out["ok"] is False
    assert called["scoped"] >= 1           # scoped cancel used
    assert broker.account_wide_cancels == 0  # account-wide cancel NOT used when cancel supplied
```

> Implementer note: adapt the harness to the existing `tests/test_cli_paper.py` fakes — the key assertions are (1) breach returns `ok False`, (2) the supplied `cancel` callable was invoked, (3) the broker's account-wide `cancel_open_orders` was NOT invoked. Add an `account_wide_cancels` counter to the fake broker if absent.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli_paper.py::test_run_paper_strategy_tick_breach_uses_scoped_cancel -v`
Expected: FAIL — the breach handler currently calls `broker.cancel_open_orders()` unconditionally.

- [ ] **Step 3: Harden the helper**

In `algua/cli/paper_cmd.py`, (a) add `reserve_buy=None` to the keyword-only params of `_run_paper_strategy_tick`; (b) add `reserve_buy=reserve_buy` to its `TickHooks(...)`; (c) in the `except RiskBreach` handler, replace the cancel line:

```python
            # scoped cancel when the caller supplies one (run-all); account-wide fallback for the
            # single-strategy trade-tick path (cancel=None) — never cancel a sibling's orders.
            if cancel is not None:
                cancel()
            else:
                broker.cancel_open_orders()
```

(Replace the existing bare `broker.cancel_open_orders()` inside the breach `try:` with the block above.)

- [ ] **Step 4: Run test + single-strategy regression**

Run: `uv run pytest tests/test_cli_paper.py -k "scoped_cancel or trade_tick or breach" -v`
Expected: PASS — new test passes; existing single-strategy `trade-tick` breach tests (cancel=None → account-wide) unchanged.

- [ ] **Step 5: Commit**

```bash
git add algua/cli/paper_cmd.py tests/test_cli_paper.py
git commit -m "feat(paper): _run_paper_strategy_tick reserve_buy hook + scoped breach cancel #316b

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: `paper run-all` command (+ `_paper_scoped_cancel`, reservation pool)

**Files:**
- Modify: `algua/cli/paper_cmd.py`
- Test: `tests/test_paper_run_all.py` (create)

**Interfaces:**
- Consumes: `owned_open_order_ids(kind=LedgerKind.PAPER)` (Task 1); hardened `_run_paper_strategy_tick` (Task 2); `_paper_broker_net`, `_ingest_paper_venue`, `paper_reconcile`, `_alpaca_broker_from_settings`, `_select_provider`, `load_gated_strategy`, `SqliteStrategyRepository`, `Stage`, `global_halt`, `audit_append`, `tick_clock`.
- Produces: `paper run-all --snapshot ID [--start --end --max-drawdown]` — one sequenced cycle over all `Stage.PAPER` strategies.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_paper_run_all.py` (reuse the `_FakePaperBroker` + allocation-seeding patterns from `tests/test_cli_paper.py`; a multi-strategy broker tracks per-strategy submits + cancels):

```python
import json
from typer.testing import CliRunner
from algua.cli.app import app

runner = CliRunner()


def test_run_all_ticks_all_paper_strategies(...):
    # Two paper strategies, each allocated; clean reconcile. Both tick; envelope lists both.
    ... # seed two paper strategies + allocations + fake broker (clean account)
    result = runner.invoke(app, ["paper", "run-all", "--snapshot", _SNAP,
                                 "--start", "2026-01-01", "--end", "2026-02-01"])
    assert result.exit_code == 0, result.output
    names = {s["strategy"] for s in json.loads(result.output)["strategies"]}
    assert names == {"s1", "s2"}


def test_run_all_defers_whole_cycle_on_unreconciled_account(...):
    # Broker shows an unattributable holding -> reconcile not clean -> NO strategy trades.
    ...
    result = runner.invoke(app, ["paper", "run-all", "--snapshot", _SNAP, ...])
    payload = json.loads(result.output)
    assert payload["strategies"] == [] and payload.get("deferred") is True
    assert broker.submitted == []


def test_run_all_breach_scoped_flatten_surfaces_siblings(...):
    # s1 breaches: it trips + scoped-flattens; the envelope still includes the sibling ticked
    # before it, and exit is non-zero. A sibling's resting order is NOT cancelled by s1's breach.
    ...
    assert result.exit_code != 0
    assert any(s.get("ok") is False for s in payload["strategies"])
    assert sibling_open_order_still_present  # scoped cancel did not touch the sibling


def test_run_all_reservation_pool_caps_concurrent_buys(...):
    # Pool = buying_power; first strategy consumes most of it; second strategy's buy is trimmed.
    ...
    assert second_strategy_buy_notional <= remaining_pool_after_first
```

> Implementer note: build the multi-strategy fake broker + seeding helpers from the existing
> `tests/test_cli_paper.py` / `tests/test_paper_venue_reconcile.py` fakes. Keep the four behaviors'
> assertions concrete (both-tick, defer-all, scoped-breach-surfaces-siblings, pool-cap). Seed
> allocations directly (no `paper allocate` CLI dependency).

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_paper_run_all.py -v`
Expected: FAIL — `No such command 'run-all'`.

- [ ] **Step 3: Add `_paper_scoped_cancel` + the command**

In `algua/cli/paper_cmd.py`, add (ensure `owned_open_order_ids` is imported from `algua.execution.live_ledger`):

```python
def _paper_scoped_cancel(conn, broker, name: str) -> None:
    """Cancel only THIS strategy's open paper-venue orders (never a sibling's)."""
    for oid in owned_open_order_ids(conn, broker, name, kind=LedgerKind.PAPER):
        broker.cancel_order(oid)


@paper_app.command("run-all")
@json_errors(ValueError, LookupError, BrokerError)
def run_all(
    snapshot: str = typer.Option(..., "--snapshot", help="ingested bars snapshot id"),
    start: str = typer.Option("2023-01-01", "--start"),
    end: str = typer.Option("2023-12-31", "--end"),
    max_drawdown: float = typer.Option(None, "--max-drawdown",
                                       help="halt + flatten a strategy if equity falls this fraction below its peak"),  # noqa: E501
) -> None:
    """One sequenced cycle over ALL paper strategies: ingest venue fills, reconcile the account,
    then tick each (scoped cancel, shared buying-power pool). Trades only when the account
    reconciles clean; a persistent unexplained drift engages the global halt."""
    if max_drawdown is not None and not 0.0 < max_drawdown <= 1.0:
        raise ValueError("--max-drawdown must be in (0, 1]")
    with registry_conn() as conn:
        repo = SqliteStrategyRepository(conn)
        paper = repo.list_strategies(Stage.PAPER)
        if not paper:
            emit(ok({"strategies": [], "note": "no paper strategies"}))
            return
        if global_halt.is_engaged(conn):
            emit(breach_payload("global halt engaged", halted=True))
            raise typer.Exit(1)
        broker = _alpaca_broker_from_settings()
        provider = _select_provider(False, snapshot)
        acct = broker.account()
        tick_ts, clock_source = tick_clock(broker.clock)
        try:
            _ingest_paper_venue(conn, broker, tick_ts)
        except Exception as exc:
            audit_append(conn, actor="system", action="venue_ingest_failed", reason=str(exc))
            emit(breach_payload(str(exc), kind="venue_ingest_failed"))
            raise typer.Exit(1) from exc

        cycle = paper_reconcile.next_cycle(conn)
        recon = paper_reconcile.reconcile(conn, _paper_broker_net(broker), cycle)
        if recon.halt:
            global_halt.engage(conn, reason=f"paper reconcile drift {recon.mismatches}",
                               actor="system")
            emit({"ok": False, "deferred": True, "halted": True, "reconcile": recon.mismatches})
            raise typer.Exit(1)
        if not recon.clean:
            emit(ok({"strategies": [], "deferred": True, "reconcile": recon.mismatches,
                     "note": "reconcile pending; deferring trades this cycle"}))
            return

        pool = {"available": float(acct.buying_power)}

        def _paper_reserve_for(strategy_name):
            def _reserve(symbol: str, notional: float) -> float:
                permitted = min(notional, max(0.0, pool["available"]))
                pool["available"] -= permitted
                if permitted < notional:
                    audit_append(conn, actor="system", action="paper_reserve_trim",
                                 reason=f"{symbol} {notional}->{permitted}", strategy=strategy_name)
                return permitted
            return _reserve

        results: list[dict] = []
        breached = False
        for prec in paper:
            name = prec.name
            strategy, rec = load_gated_strategy(conn, name, "paper run-all")
            out = _run_paper_strategy_tick(
                conn, name, strategy, rec, broker, provider, max_drawdown,
                tick_ts, clock_source, acct,
                reserve_buy=_paper_reserve_for(name),
                cancel=lambda n=name: _paper_scoped_cancel(conn, broker, n),
                start=start, end=end)
            results.append(out)
            if out.get("ok") is False:
                breached = True
                break
    envelope = {"reconcile": recon.mismatches, "strategies": results}
    if breached:
        emit({"ok": False, **envelope})
        raise typer.Exit(1)
    emit(ok(envelope))
```

- [ ] **Step 4: Run the new tests + paper regression**

Run: `uv run pytest tests/test_paper_run_all.py tests/test_cli_paper.py tests/test_paper_venue_reconcile.py -v`
Expected: PASS — the four run-all tests pass and existing paper tests are unaffected.

- [ ] **Step 5: Full quality gate**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add algua/cli/paper_cmd.py tests/test_paper_run_all.py
git commit -m "feat(paper): paper run-all — multi-tenant batch + scoped cancel + reservation pool #316b

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-review

- **Spec coverage:** §2 scoped-cancel fix → Task 1 (generalize) + Task 2 (helper breach cancel) + Task 3 (`_paper_scoped_cancel`); §3 run-all flow (ingest-once → reconcile-once → pool → loop → envelope) → Task 3; §4 components all mapped; §5 testing (both-tick, defer-all, scoped-breach-surfaces-siblings, pool-cap, owned_open_order_ids paper, helper scoped-cancel) → Tasks 1–3 tests. §6 non-goals respected (no is_due, no gate, no operator intake, no run_tick body change).
- **Placeholder scan:** the two implementer notes (test-harness mechanics) are unavoidable given existing fakes; the assertions are concrete. No code-step placeholders.
- **Type consistency:** `owned_open_order_ids(conn, broker, strategy, *, kind=LedgerKind.LIVE)` (Task 1) used by `_paper_scoped_cancel` (Task 3); `_run_paper_strategy_tick(..., *, cancel=None, reserve_buy=None, start, end)` (Task 2) called with those kwargs in run-all (Task 3); `paper_reconcile.next_cycle`/`reconcile` + `_paper_broker_net` signatures match #313/#316a.
