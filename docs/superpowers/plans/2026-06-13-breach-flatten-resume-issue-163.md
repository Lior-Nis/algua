# Breach → Flatten → Resume (issue #163) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** End-to-end test the live breach→flatten→resume money path, fix the paper/forward_tested flatten to close the strategy's own held-but-dropped symbols and reset its stale belief, and make `resume`/`resume-all` reconcile the live ledger against broker truth so the operator is never trapped.

**Architecture:** The live `run-all` breach handler already offsets over `believed_positions` + records the offsets (correct, just untested). We (1) add a per-strategy paper flatten scope + belief reset, (2) add an ingest + account-wide broker-truth reconcile to `resume`/`resume-all` via a read-only live client, and (3) cover the whole chain with a real integration test.

**Tech Stack:** Python, Typer CLI, SQLite (`registry_conn`), pytest + `typer.testing.CliRunner`. All data commands emit JSON on stdout.

**Spec:** `docs/superpowers/specs/2026-06-13-breach-flatten-resume-issue-163-design.md`

**Quality gate (run after each task's commit):**
`uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`

---

## File Structure

- `algua/execution/order_state.py` — add `clear_derived_positions(conn, strategy)` (zero a strategy's derived paper belief).
- `algua/execution/live_ledger.py` — add `strategy_live_symbols(conn, strategy)` (the symbols a strategy is responsible for: its `live_orders` ∪ `live_fills`).
- `algua/execution/alpaca_broker.py` — add `AlpacaLiveReadOnlyBroker(_AlpacaBroker)` (live host, no `LiveAuthorization`, for read-only reconcile).
- `algua/cli/paper_cmd.py` — `_strategy_held_symbols` helper; per-strategy close + belief reset at both flatten sites; `resume`/`resume-all` ingest + broker-truth reconcile; widen decorators to `BrokerError`.
- `tests/test_order_state.py`, `tests/test_live_ledger_orders.py`, `tests/test_alpaca_broker.py`, `tests/test_cli_paper.py`, `tests/test_cli_live.py` — tests.

---

### Task 1: `clear_derived_positions` — zero a strategy's derived paper belief

**Files:**
- Modify: `algua/execution/order_state.py`
- Test: `tests/test_order_state.py` (create if absent)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_order_state.py` (create the file with this content if it does not exist):

```python
from contextlib import closing

from algua.execution.order_state import clear_derived_positions, derive_positions
from algua.registry.db import connect, migrate


def _seed_fill(conn, strategy, symbol, qty):
    cur = conn.execute(
        "INSERT INTO paper_orders(strategy, symbol, side, target_weight, decision_ts, "
        "submitted_ts, status, broker_order_id) VALUES (?,?,?,?,?,?,?,?)",
        (strategy, symbol, "buy", 0.5, "2023-01-01T00:00:00Z", "2023-01-01T00:00:00Z",
         "filled", f"bo-{strategy}-{symbol}"),
    )
    conn.execute(
        "INSERT INTO paper_fills(order_id, symbol, qty, price, fill_ts) VALUES (?,?,?,?,?)",
        (cur.lastrowid, symbol, qty, 100.0, "2023-01-01T00:00:00Z"),
    )
    conn.commit()


def test_clear_derived_positions_zeros_only_target_strategy(tmp_path):
    db = tmp_path / "p.db"
    with closing(connect(str(db))) as conn:
        migrate(conn)
        _seed_fill(conn, "alpha", "AAA", 5.0)
        _seed_fill(conn, "beta", "BBB", 3.0)
        assert derive_positions(conn, "alpha") == {"AAA": 5.0}

        clear_derived_positions(conn, "alpha")

        assert derive_positions(conn, "alpha") == {}        # alpha belief gone
        assert derive_positions(conn, "beta") == {"BBB": 3.0}  # sibling untouched
        # paper_orders rows are kept (audit trail / symbol source)
        n = conn.execute(
            "SELECT COUNT(*) FROM paper_orders WHERE strategy='alpha'"
        ).fetchone()[0]
        assert n == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_order_state.py::test_clear_derived_positions_zeros_only_target_strategy -v`
Expected: FAIL with `ImportError: cannot import name 'clear_derived_positions'`.

- [ ] **Step 3: Add the function**

In `algua/execution/order_state.py`, add after `derive_positions` (around line 96):

```python
def clear_derived_positions(conn: sqlite3.Connection, strategy: str) -> None:
    """Zero a strategy's DERIVED paper belief by dropping its fills (the source `derive_positions`
    sums), keeping its `paper_orders` rows as the audit trail and the flatten symbol source. Used
    after a successful paper flatten so a subsequent `trade-tick` reconcile starts from flat
    instead of re-tripping on stale simulated positions. Mirrors `persist_run`'s leading DELETE."""
    conn.execute(
        "DELETE FROM paper_fills WHERE order_id IN "
        "(SELECT id FROM paper_orders WHERE strategy = ?)",
        (strategy,),
    )
    conn.commit()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_order_state.py::test_clear_derived_positions_zeros_only_target_strategy -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add algua/execution/order_state.py tests/test_order_state.py
git commit -m "feat(163): clear_derived_positions to zero a strategy's paper belief"
```

---

### Task 2: `strategy_live_symbols` — the symbols a strategy is responsible for

**Files:**
- Modify: `algua/execution/live_ledger.py`
- Test: `tests/test_live_ledger_orders.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_live_ledger_orders.py`:

```python
def test_strategy_live_symbols_unions_orders_and_fills(tmp_path):
    from contextlib import closing

    from algua.execution.live_ledger import record_live_order, strategy_live_symbols
    from algua.registry.db import connect, migrate
    with closing(connect(str(tmp_path / "p.db"))) as conn:
        migrate(conn)
        record_live_order(conn, "alpha", "AAA", "buy", None, "coid-aaa")
        # a fill in a symbol with no surviving order row (e.g. dropped from the universe)
        conn.execute(
            "INSERT INTO live_fills(activity_id, broker_order_id, strategy, symbol, qty, price, "
            "fill_ts) VALUES (?,?,?,?,?,?,?)",
            ("act-zzz", "bo-zzz", "alpha", "ZZZ", 3.0, 100.0, "2023-01-01T00:00:00Z"),
        )
        conn.commit()
        record_live_order(conn, "beta", "BBB", "buy", None, "coid-bbb")

        assert strategy_live_symbols(conn, "alpha") == {"AAA", "ZZZ"}
        assert strategy_live_symbols(conn, "beta") == {"BBB"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_live_ledger_orders.py::test_strategy_live_symbols_unions_orders_and_fills -v`
Expected: FAIL with `ImportError: cannot import name 'strategy_live_symbols'`.

- [ ] **Step 3: Add the function**

In `algua/execution/live_ledger.py`, add after `believed_positions` (around line 100):

```python
def strategy_live_symbols(conn: sqlite3.Connection, strategy: str) -> set[str]:
    """Every symbol a strategy is responsible for = the union of symbols in its live_orders and its
    live_fills. Used by the resume reconcile to scope the broker-truth check to the strategy's own
    symbols (a held-but-dropped symbol is in its fills even after it left the universe)."""
    rows = conn.execute(
        "SELECT symbol FROM live_orders WHERE strategy = ? "
        "UNION SELECT symbol FROM live_fills WHERE strategy = ?",
        (strategy, strategy),
    ).fetchall()
    return {r["symbol"] for r in rows}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_live_ledger_orders.py::test_strategy_live_symbols_unions_orders_and_fills -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add algua/execution/live_ledger.py tests/test_live_ledger_orders.py
git commit -m "feat(163): strategy_live_symbols for per-strategy reconcile scoping"
```

---

### Task 3: `AlpacaLiveReadOnlyBroker` — read-only live client (no authorization)

**Files:**
- Modify: `algua/execution/alpaca_broker.py`
- Test: `tests/test_alpaca_broker.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_alpaca_broker.py`:

```python
def test_live_readonly_broker_requires_live_https_host():
    from algua.execution.alpaca_broker import AlpacaLiveReadOnlyBroker, BrokerError

    # accepts the live host with no LiveAuthorization
    b = AlpacaLiveReadOnlyBroker("k", "s", base_url="https://api.alpaca.markets")
    assert b.base_url == "https://api.alpaca.markets"
    # refuses a non-live / non-https host (the platform invariant)
    import pytest
    with pytest.raises(BrokerError):
        AlpacaLiveReadOnlyBroker("k", "s", base_url="https://paper-api.alpaca.markets")
    with pytest.raises(BrokerError):
        AlpacaLiveReadOnlyBroker("k", "s", base_url="http://api.alpaca.markets")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_alpaca_broker.py::test_live_readonly_broker_requires_live_https_host -v`
Expected: FAIL with `ImportError: cannot import name 'AlpacaLiveReadOnlyBroker'`.

- [ ] **Step 3: Add the class**

In `algua/execution/alpaca_broker.py`, add after the `AlpacaLiveBroker` class definition (end of file region, after the `AlpacaLiveBroker.__init__`):

```python
class AlpacaLiveReadOnlyBroker(_AlpacaBroker):
    """READ-ONLY view of the Alpaca LIVE venue for reconcile (get_positions + account_activities),
    constructed WITHOUT a LiveAuthorization because it never places an order — both endpoints are
    GETs. Reuses the base host allowlist (live host + https only), so it cannot be pointed at a
    wrong endpoint. Used by `resume`/`resume-all` to confirm a live strategy is flat at the broker
    before clearing the kill-switch. The live API keys remain the real wall (trusted env only)."""

    _ALLOWED_HOSTS = frozenset({"api.alpaca.markets"})

    def __init__(self, api_key: str, api_secret: str, base_url: str = _LIVE_DEFAULT_URL) -> None:
        super().__init__(api_key, api_secret, base_url)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_alpaca_broker.py::test_live_readonly_broker_requires_live_https_host -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add algua/execution/alpaca_broker.py tests/test_alpaca_broker.py
git commit -m "feat(163): AlpacaLiveReadOnlyBroker for read-only reconcile reads"
```

---

### Task 4: Paper flatten + trade-tick breach — per-strategy close + belief reset

**Files:**
- Modify: `algua/cli/paper_cmd.py` (`flatten` ~L454-481; `trade-tick` breach handler ~L331-348)
- Test: `tests/test_cli_paper.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cli_paper.py`:

```python
def _seed_paper_order(db_path, strategy, symbol):
    from contextlib import closing

    from algua.registry.db import connect, migrate
    with closing(connect(str(db_path))) as conn:
        migrate(conn)
        cur = conn.execute(
            "INSERT INTO paper_orders(strategy, symbol, side, target_weight, decision_ts, "
            "submitted_ts, status, broker_order_id) VALUES (?,?,?,?,?,?,?,?)",
            (strategy, symbol, "buy", 0.5, "2023-01-01T00:00:00Z", "2023-01-01T00:00:00Z",
             "filled", f"bo-{strategy}-{symbol}"),
        )
        conn.execute(
            "INSERT INTO paper_fills(order_id, symbol, qty, price, fill_ts) VALUES (?,?,?,?,?)",
            (cur.lastrowid, symbol, 5.0, 100.0, "2023-01-01T00:00:00Z"),
        )
        conn.commit()


def test_paper_flatten_closes_dropped_symbol_not_siblings(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALGUA_ALPACA_API_SECRET", "s")
    _to_paper()
    db = tmp_path / "p.db"
    # the strategy holds ZZZ (a symbol no longer in its universe); a SIBLING strategy holds SIB
    _seed_paper_order(db, "cross_sectional_momentum", "ZZZ")
    _seed_paper_order(db, "sibling_strat", "SIB")

    broker = _FlattenBroker()
    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings", lambda: broker)
    result = runner.invoke(app, ["paper", "flatten", "cross_sectional_momentum"])
    assert result.exit_code == 0, result.stdout

    closed = set(broker.closed_symbols)
    assert "ZZZ" in closed                     # held-but-dropped symbol IS closed
    assert "SIB" not in closed                 # sibling's symbol is NOT closed
    # the closed set is surfaced in the payload
    assert "ZZZ" in json.loads(result.stdout)["closed_symbols"]

    # the strategy's derived belief is reset to flat after the close
    show = json.loads(runner.invoke(app, ["paper", "show", "cross_sectional_momentum"]).stdout)
    assert show["positions"] == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli_paper.py::test_paper_flatten_closes_dropped_symbol_not_siblings -v`
Expected: FAIL — `closed_symbols` not in payload and/or ZZZ not closed (current code closes `strategy.universe` only).

- [ ] **Step 3: Add the shared helper and wire both flatten sites**

In `algua/cli/paper_cmd.py`, add these imports to the existing `from algua.execution.order_state import (...)` block: `clear_derived_positions`. Then add the helper near `_trip` (around line 132):

```python
def _strategy_held_symbols(
    conn: sqlite3.Connection, strategy: str, universe: list[str]
) -> list[str]:
    """Universe ∪ every symbol THIS strategy has submitted a paper order for — so a held-but-dropped
    symbol (traded before its universe changed) is still exited, WITHOUT closing a sibling's
    positions on a shared paper account. Closing a flat name is a 404 no-op, so over-including
    never-filled names is harmless."""
    rows = conn.execute(
        "SELECT DISTINCT symbol FROM paper_orders WHERE strategy = ?", (strategy,)
    ).fetchall()
    return sorted(set(universe) | {r["symbol"] for r in rows})
```

Replace the `flatten` command body's close block (currently lines ~474-481):

```python
        try:
            broker.cancel_open_orders()
            broker.close_positions(strategy.universe)
        except BrokerError as exc:
            emit(_breach_payload(str(exc), strategy=name, liquidation_submitted=False))
            raise typer.Exit(1) from exc
    # liquidation_submitted: Alpaca accepted the close orders; fills land async (may be next open).
    emit(ok({"strategy": name, "kill_switch": "tripped", "liquidation_submitted": True}))
```

with:

```python
        symbols = _strategy_held_symbols(conn, name, strategy.universe)
        try:
            broker.cancel_open_orders()
            broker.close_positions(symbols)
        except BrokerError as exc:
            emit(_breach_payload(str(exc), strategy=name, liquidation_submitted=False))
            raise typer.Exit(1) from exc
        # Close succeeded: reset the derived belief so a later trade-tick reconcile starts flat.
        clear_derived_positions(conn, name)
    # liquidation_submitted: Alpaca accepted the close orders; fills land async (may be next open).
    emit(ok({"strategy": name, "kill_switch": "tripped", "liquidation_submitted": True,
             "closed_symbols": symbols}))
```

Replace the `trade-tick` breach handler's close block (currently lines ~335-347):

```python
            try:
                broker.cancel_open_orders()
                broker.close_positions(strategy.universe)
            except BrokerError as fexc:
                liquidation_submitted = False
                flatten_error = str(fexc)
                audit_append(conn, actor="system", action="flatten_failed",
                             reason=str(fexc), strategy=name)
            payload = _breach_payload(exc.detail, kind=exc.kind,
                                      liquidation_submitted=liquidation_submitted)
            if flatten_error is not None:
                payload["flatten_error"] = flatten_error
            emit(payload)
            raise typer.Exit(1) from exc
```

with:

```python
            symbols = _strategy_held_symbols(conn, name, strategy.universe)
            try:
                broker.cancel_open_orders()
                broker.close_positions(symbols)
            except BrokerError as fexc:
                liquidation_submitted = False
                flatten_error = str(fexc)
                audit_append(conn, actor="system", action="flatten_failed",
                             reason=str(fexc), strategy=name)
            else:
                # Close succeeded: reset the derived belief so the next reconcile starts flat.
                clear_derived_positions(conn, name)
            payload = _breach_payload(exc.detail, kind=exc.kind,
                                      liquidation_submitted=liquidation_submitted,
                                      closed_symbols=symbols)
            if flatten_error is not None:
                payload["flatten_error"] = flatten_error
            emit(payload)
            raise typer.Exit(1) from exc
```

- [ ] **Step 4: Run tests to verify they pass (incl. existing flatten tests)**

Run: `uv run pytest tests/test_cli_paper.py -k flatten -v`
Expected: PASS — new test plus existing `test_paper_flatten_closes_and_trips`, `test_paper_flatten_allowed_at_forward_tested_stage`, `test_paper_flatten_close_failure_stays_tripped`, `test_paper_flatten_rejects_non_paper_stage`.

- [ ] **Step 5: Commit**

```bash
git add algua/cli/paper_cmd.py tests/test_cli_paper.py
git commit -m "feat(163): per-strategy paper flatten close + belief reset"
```

---

### Task 5: `resume` (Stage.LIVE) — ingest + broker-truth reconcile

**Files:**
- Modify: `algua/cli/paper_cmd.py` (`resume` ~L243-273; add factory + reconcile helper)
- Test: `tests/test_cli_paper.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cli_paper.py`:

```python
class _ReadOnlyLiveBroker:
    """Fake read-only live broker: scripted activities + broker net positions for resume tests."""
    def __init__(self, activities, positions):
        self._activities = activities
        self._positions = positions
    def account_activities(self, after=None):
        return self._activities
    def get_positions(self):
        import pandas as pd
        return pd.Series(self._positions, dtype="float64")


def _seed_live_killed(db_path, name, fills):
    """Seed a tripped live strategy with believed fills (symbol -> qty)."""
    from contextlib import closing

    from algua.registry.db import connect, migrate
    from algua.risk import kill_switch
    with closing(connect(str(db_path))) as conn:
        migrate(conn)
        conn.execute("UPDATE strategies SET stage='live' WHERE name=?", (name,))
        for i, (sym, qty) in enumerate(fills.items()):
            conn.execute(
                "INSERT INTO live_fills(activity_id, broker_order_id, strategy, symbol, qty, "
                "price, fill_ts) VALUES (?,?,?,?,?,?,?)",
                (f"seed-{i}", f"bo-{i}", name, sym, qty, 100.0, "2023-01-01T00:00:00Z"),
            )
        kill_switch.trip(conn, name, reason="flatten", actor="system")
        conn.commit()


def test_resume_live_refuses_when_broker_still_holds(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_ALPACA_LIVE_API_KEY", "lk")
    monkeypatch.setenv("ALGUA_ALPACA_LIVE_API_SECRET", "ls")
    name = "cross_sectional_momentum"
    _to_paper()
    _seed_live_killed(tmp_path / "p.db", name, {"AAA": 5.0})
    # broker still reports AAA held, no new activities -> ledger non-flat AND broker exposed
    broker = _ReadOnlyLiveBroker(activities=[], positions={"AAA": 5.0})
    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_live_readonly_from_settings", lambda: broker)
    r = runner.invoke(app, ["paper", "resume", name])
    assert r.exit_code == 1
    assert json.loads(r.stdout)["ok"] is False
    assert "not flat" in r.stdout.lower()


def test_resume_live_refuses_when_creds_missing(monkeypatch, tmp_path):
    name = "cross_sectional_momentum"
    _to_paper()
    _seed_live_killed(tmp_path / "p.db", name, {"AAA": 5.0})
    # no live creds set -> cannot confirm flat -> refuse
    r = runner.invoke(app, ["paper", "resume", name])
    assert r.exit_code == 1
    assert json.loads(r.stdout)["ok"] is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cli_paper.py -k "resume_live" -v`
Expected: FAIL — `_alpaca_live_readonly_from_settings` does not exist / resume does not yet reconcile.

- [ ] **Step 3: Add the factory, reconcile helper, and rewrite resume's live branch**

In `algua/cli/paper_cmd.py`, add to the top-level imports:

```python
from algua.execution.alpaca_broker import AlpacaLiveBroker  # noqa: F401  (if not already imported)
from algua.execution.alpaca_broker import AlpacaLiveReadOnlyBroker
from algua.execution.live_ledger import (
    believed_positions,
    fill_cursor,
    ingest_activities,
    strategy_live_symbols,
)
from algua.execution.live_reconcile import account_expected_net
```

(Drop the now-redundant lazy `from algua.execution.live_ledger import believed_positions` inside `resume`/`resume-all`/`show`.) Add near `_alpaca_broker_from_settings` (around line 70):

```python
_RECONCILE_TOL = 1e-6


def _alpaca_live_readonly_from_settings() -> AlpacaLiveReadOnlyBroker:
    s = get_settings()
    if not s.alpaca_live_api_key or not s.alpaca_live_api_secret:
        raise ValueError(
            "Alpaca LIVE credentials not configured; cannot confirm the strategy is flat at the "
            "broker — set ALGUA_ALPACA_LIVE_API_KEY and ALGUA_ALPACA_LIVE_API_SECRET"
        )
    return AlpacaLiveReadOnlyBroker(s.alpaca_live_api_key, s.alpaca_live_api_secret,
                                    base_url=s.alpaca_live_url)


def _live_strategy_flat(
    conn: sqlite3.Connection, name: str, universe: list[str], broker: object,
) -> tuple[bool, dict]:
    """Ingest pending broker activities, then ACCOUNT-WIDE reconcile: the strategy is flat iff its
    own believed_positions is empty AND the broker holds no UNEXPLAINED qty (broker net minus the
    books' expected net = Σ all live_fills) in any symbol it is responsible for. A sibling that
    legitimately holds the same symbol explains the broker qty and does not block resume."""
    ingest_activities(conn, broker.account_activities(after=fill_cursor(conn)))  # type: ignore[attr-defined]
    own = believed_positions(conn, name)
    broker_net = {s: float(q) for s, q in broker.get_positions().items()  # type: ignore[attr-defined]
                  if float(q) != 0.0}
    expected = account_expected_net(conn)
    syms = set(universe) | strategy_live_symbols(conn, name)
    unexplained = {
        s: broker_net.get(s, 0.0) - expected.get(s, 0.0)
        for s in syms
        if abs(broker_net.get(s, 0.0) - expected.get(s, 0.0)) > _RECONCILE_TOL
    }
    is_flat = (not own) and (not unexplained)
    return is_flat, {"believed": own, "broker_unexplained": unexplained}
```

Change the `resume` decorator and rewrite its live branch. Replace (lines ~243-254):

```python
@paper_app.command("resume")
@json_errors(ValueError)
def resume(name: str) -> None:
    """Reset (clear) a strategy's kill-switch so paper runs may resume. Human action."""
    from algua.execution.live_ledger import believed_positions
    with registry_conn() as conn:
        rec = SqliteStrategyRepository(conn).get(name)
        if rec.stage is Stage.LIVE and believed_positions(conn, name):
            raise ValueError(
                f"{name} is not flat (believed positions: {believed_positions(conn, name)}); "
                "re-flatten before resuming a live strategy"
            )
```

with:

```python
@paper_app.command("resume")
@json_errors(ValueError, BrokerError)
def resume(name: str) -> None:
    """Reset (clear) a strategy's kill-switch so paper runs may resume. Human action. For a LIVE
    strategy this first ingests pending broker activities and reconciles against broker truth — it
    refuses unless the strategy is flat both in the ledger AND at the broker (zero drift)."""
    with registry_conn() as conn:
        rec = SqliteStrategyRepository(conn).get(name)
        if rec.stage is Stage.LIVE:
            strategy = load_strategy(name)
            broker = _alpaca_live_readonly_from_settings()
            is_flat, residual = _live_strategy_flat(conn, name, strategy.universe, broker)
            if not is_flat:
                raise ValueError(
                    f"{name} is not flat after reconcile: {residual}; offset fills pending or "
                    "liquidation incomplete — re-flatten or retry after fills land"
                )
```

The rest of `resume` (the `was_tripped` block) is unchanged.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli_paper.py -k "resume" -v`
Expected: PASS — new live-refuse tests plus existing `test_resume_rebases_drawdown_peak`, `test_manual_kill_blocks_run_then_resume_allows`, etc. (paper-stage resume unchanged).

- [ ] **Step 5: Commit**

```bash
git add algua/cli/paper_cmd.py tests/test_cli_paper.py
git commit -m "feat(163): resume reconciles a live strategy against broker truth"
```

---

### Task 6: `resume-all` — ingest before computing `not_flat`

**Files:**
- Modify: `algua/cli/paper_cmd.py` (`resume-all` ~L507-541)
- Test: `tests/test_cli_paper.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_cli_paper.py`:

```python
def test_resume_all_ingests_before_warning(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_ALPACA_LIVE_API_KEY", "lk")
    monkeypatch.setenv("ALGUA_ALPACA_LIVE_API_SECRET", "ls")
    from algua.risk import global_halt
    name = "cross_sectional_momentum"
    _to_paper()
    # one live strategy holding AAA; an ingest delivers the offsetting fill so it nets flat
    _seed_live_killed(tmp_path / "p.db", name, {"AAA": 5.0})
    from contextlib import closing

    from algua.config.settings import get_settings
    from algua.registry.db import connect, migrate
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        # record the order so the ingested offset fill attributes back to the strategy
        from algua.execution.live_ledger import backfill_broker_order_id, record_live_order
        record_live_order(conn, name, "AAA", "sell", None, "coid-off")
        backfill_broker_order_id(conn, "coid-off", "bo-off")
        global_halt.engage(conn, reason="halt-all", actor="agent")

    offset_fill = [{"id": "act-off", "activity_type": "FILL", "side": "sell", "qty": "5",
                    "price": "100", "symbol": "AAA", "order_id": "bo-off",
                    "transaction_time": "2023-01-02T00:00:00Z"}]
    broker = _ReadOnlyLiveBroker(activities=offset_fill, positions={})
    monkeypatch.setattr("algua.cli.paper_cmd._maybe_live_readonly", lambda: broker)

    r = runner.invoke(app, ["paper", "resume-all"])
    assert r.exit_code == 0, r.stdout
    payload = json.loads(r.stdout)
    assert payload["global_halt"] == "reset"
    # after ingest the offset fill landed -> strategy is flat -> NOT listed as not_flat
    assert "live_not_flat" not in payload
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli_paper.py::test_resume_all_ingests_before_warning -v`
Expected: FAIL — `_maybe_live_readonly` does not exist; without the ingest the strategy still shows as `not_flat`.

- [ ] **Step 3: Add `_maybe_live_readonly` and wire the ingest**

In `algua/cli/paper_cmd.py`, add near `_alpaca_live_readonly_from_settings`:

```python
def _maybe_live_readonly() -> AlpacaLiveReadOnlyBroker | None:
    """A read-only live client if live creds are configured, else None (resume-all stays lenient:
    with no creds it just computes not_flat from the current belief)."""
    s = get_settings()
    if not s.alpaca_live_api_key or not s.alpaca_live_api_secret:
        return None
    return AlpacaLiveReadOnlyBroker(s.alpaca_live_api_key, s.alpaca_live_api_secret,
                                    base_url=s.alpaca_live_url)
```

Change the `resume-all` decorator to `@json_errors(ValueError, BrokerError)` and remove its lazy `from algua.execution.live_ledger import believed_positions`. Then, inside `with registry_conn() as conn:`, immediately after `live_rows = conn.execute(...).fetchall()`, add the ingest BEFORE the `not_flat` list comprehension:

```python
        live_rows = conn.execute(
            "SELECT name FROM strategies WHERE stage = 'live'"
        ).fetchall()
        if live_rows:
            broker = _maybe_live_readonly()
            if broker is not None:
                # account-wide ingest so not_flat reflects post-ingest belief (landed offset fills)
                ingest_activities(conn, broker.account_activities(after=fill_cursor(conn)))
        not_flat = [
            r["name"] for r in live_rows if believed_positions(conn, r["name"])
        ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli_paper.py -k "resume_all" -v`
Expected: PASS — new test plus existing `test_resume_all_clears_and_wipes_peaks_but_keeps_strategy_switch`.

- [ ] **Step 5: Commit**

```bash
git add algua/cli/paper_cmd.py tests/test_cli_paper.py
git commit -m "feat(163): resume-all ingests live activities before computing not_flat"
```

---

### Task 7: End-to-end live breach→flatten→resume test + reconcile negatives

**Files:**
- Test: `tests/test_cli_live.py`

- [ ] **Step 1: Write the end-to-end test**

Append to `tests/test_cli_live.py`:

```python
class _BreachThenFlatBroker:
    """Fake live broker for the full breach->flatten->resume chain. Before the offsets are
    submitted it reports the seeded holdings (AAA+5, ZZZ+3) and no new activities; once the breach
    handler submits the offsets it flips to flat and its activity feed returns the offset FILLs."""
    def __init__(self):
        self.offsets = []
        self._closed = False
    def account_activities(self, after=None):
        if not self._closed:
            return []
        return [
            {"id": f"act-off-{sym}", "activity_type": "FILL", "side": "sell",
             "qty": str(abs(qty)), "price": "100", "symbol": sym, "order_id": f"off-{sym}",
             "transaction_time": "2023-01-02T00:00:00Z"}
            for sym, qty in self.offsets
        ]
    def get_positions(self):
        import pandas as pd
        if self._closed:
            return pd.Series(dtype="float64")
        return pd.Series({"AAA": 5.0, "ZZZ": 3.0})
    def list_open_orders(self):
        return []
    def cancel_order(self, oid):
        pass
    def submit_offset(self, symbol, qty, coid):
        self.offsets.append((symbol, qty))
        self._closed = True
        return f"off-{symbol}"
    def account(self):
        from algua.execution.alpaca_broker import AccountState
        return AccountState(equity=100_000.0, cash=100_000.0, buying_power=100_000.0)


def _seed_live_fills(name, fills):
    from contextlib import closing

    from algua.config.settings import get_settings
    from algua.registry.db import connect, migrate
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        for i, (sym, qty) in enumerate(fills.items()):
            conn.execute(
                "INSERT INTO live_fills(activity_id, broker_order_id, strategy, symbol, qty, "
                "price, fill_ts) VALUES (?,?,?,?,?,?,?)",
                (f"seed-{sym}", f"bo-seed-{sym}", name, sym, qty, 100.0,
                 "2023-01-01T00:00:00Z"),
            )
        conn.commit()


def test_breach_flatten_resume_end_to_end(monkeypatch):
    from algua.live.live_loop import RiskBreach
    name = "cross_sectional_momentum"
    _to_live(name)
    # strategy believes it holds AAA (5) and ZZZ (3); ZZZ is held-but-dropped (not in universe)
    _seed_live_fills(name, {"AAA": 5.0, "ZZZ": 3.0})

    broker = _BreachThenFlatBroker()
    monkeypatch.setattr("algua.cli.live_cmd.verify_live_authorization", lambda *a, **k: _auth())
    monkeypatch.setattr("algua.cli.live_cmd._alpaca_live_broker", lambda auth: broker)
    monkeypatch.setattr("algua.cli.live_cmd._select_provider", lambda demo, snapshot: object())
    monkeypatch.setattr("algua.cli.live_cmd.active_allocation",
                        lambda conn, sid: {"capital": 10_000.0})
    monkeypatch.setattr("algua.cli.live_cmd.run_tick",
                        lambda *a, **k: (_ for _ in ()).throw(RiskBreach("drawdown", "dd")))
    # resume reads the SAME fake broker (read-only path)
    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_live_readonly_from_settings", lambda: broker)

    # 1) run-all breaches -> kill-switch trips, offsets submitted over BOTH believed symbols
    r = runner.invoke(app, ["live", "run-all", "--snapshot", "x"])
    assert r.exit_code == 1
    assert sorted(broker.offsets) == [("AAA", 5.0), ("ZZZ", 3.0)]  # held-but-dropped ZZZ included

    from contextlib import closing

    from algua.config.settings import get_settings
    from algua.execution.live_ledger import believed_positions
    from algua.registry.db import connect, migrate
    from algua.risk import kill_switch
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        assert kill_switch.is_tripped(conn, name)
        # operator-trap: offset fills have NOT been ingested yet -> belief still non-flat
        assert believed_positions(conn, name) == {"AAA": 5.0, "ZZZ": 3.0}

    # 2) resume ingests the offset fills, reconciles to flat, clears the kill-switch (zero drift)
    r2 = runner.invoke(app, ["paper", "resume", name])
    assert r2.exit_code == 0, r2.stdout
    assert json.loads(r2.stdout)["kill_switch"] == "reset"
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        assert not kill_switch.is_tripped(conn, name)
        assert believed_positions(conn, name) == {}   # ledger flat after resume


def test_resume_sibling_holds_same_symbol_does_not_block(monkeypatch):
    """Strategy A is flat after ingest; sibling B legitimately holds AAA. The account-wide reconcile
    explains the broker's AAA via B's ledger, so resuming A is NOT blocked."""
    name = "cross_sectional_momentum"
    _to_live(name)
    from contextlib import closing

    from algua.config.settings import get_settings
    from algua.registry.db import connect, migrate
    from algua.risk import kill_switch
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        # sibling B holds AAA (5) in its own ledger; A holds nothing
        conn.execute(
            "INSERT INTO live_fills(activity_id, broker_order_id, strategy, symbol, qty, price, "
            "fill_ts) VALUES (?,?,?,?,?,?,?)",
            ("sib-aaa", "bo-sib", "sibling_live", "AAA", 5.0, 100.0, "2023-01-01T00:00:00Z"),
        )
        kill_switch.trip(conn, name, reason="manual", actor="system")
        conn.commit()

    # broker shows AAA+5 (B's), no activities; A's own ledger is empty
    from tests.test_cli_paper import _ReadOnlyLiveBroker  # reuse the fake
    broker = _ReadOnlyLiveBroker(activities=[], positions={"AAA": 5.0})
    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_live_readonly_from_settings", lambda: broker)

    r = runner.invoke(app, ["paper", "resume", name])
    assert r.exit_code == 0, r.stdout
    assert json.loads(r.stdout)["kill_switch"] == "reset"
```

Note: `test_resume_sibling_holds_same_symbol_does_not_block` imports `_ReadOnlyLiveBroker` from `tests/test_cli_paper.py` (defined in Task 5). If cross-test imports are undesirable in this repo's style, copy the small fake into `tests/test_cli_live.py` instead.

- [ ] **Step 2: Run tests to verify they fail (then pass after Tasks 4–6 are in)**

Run: `uv run pytest tests/test_cli_live.py -k "end_to_end or sibling" -v`
Expected: PASS once Tasks 4–6 are implemented (this task adds no production code — it is the integration coverage). If run before Tasks 5–6, it FAILs at the `paper resume` step.

- [ ] **Step 3: Run the whole breach/flatten/resume surface**

Run: `uv run pytest tests/test_cli_live.py tests/test_cli_paper.py -v`
Expected: PASS (all new + existing).

- [ ] **Step 4: Run the full quality gate**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add tests/test_cli_live.py
git commit -m "test(163): end-to-end live breach->flatten->resume + reconcile negatives"
```

---

## Self-Review notes (carried into execution)

- **Spec coverage:** Gap-1 paper close (Task 4), Gap-2 paper belief reset (Task 1+4), resume broker-truth reconcile + read-only client (Tasks 3+5), resume-all ingest (Task 6), E2E + negatives incl. held-but-dropped + sibling + residual + partial (Tasks 5+7). Partial-offset-fill refusal is covered by `test_resume_live_refuses_when_broker_still_holds` (belief non-flat + broker residual) — the same refusal path; an explicit partial variant can be added in execution if desired.
- **Type consistency:** helper names used identically across tasks — `clear_derived_positions`, `strategy_live_symbols`, `AlpacaLiveReadOnlyBroker`, `_strategy_held_symbols`, `_alpaca_live_readonly_from_settings`, `_maybe_live_readonly`, `_live_strategy_flat`.
- **Live path untouched:** no change to `live_cmd.py`'s offset/ingest breach logic; Task 7 only tests it.
- **Deferred (spec):** real wall-clock paper position ledger; activity-feed pagination robustness (now fails closed); `live resume` alias.
