# Auto-Flatten on Kill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A kill flattens the live book — `AlpacaPaperBroker.close_all_positions()`, a `paper flatten <name>` command (close + halt, fail-safe), and auto-flatten when `trade-live` trips on a `RiskBreach`.

**Architecture:** `close_all_positions` mirrors `cancel_open_orders` (DELETE + 207 per-item parse). `paper flatten` and the `trade-live` breach path both trip the kill-switch *before* closing (fail-safe: halted even if the close fails). Global kill-switch + drawdown breaker are B2b-2.

**Tech Stack:** Python 3.12, requests, Typer, sqlite3, pytest, ruff, mypy, import-linter. Spec: `docs/superpowers/specs/2026-06-03-auto-flatten-on-kill-design.md`.

---

## File structure

| File | Responsibility |
|---|---|
| `algua/execution/alpaca_broker.py` (modify) | `close_all_positions()`. |
| `algua/cli/paper_cmd.py` (modify) | `paper flatten` command + `trade-live` auto-flatten. |

---

### Task 1: `close_all_positions()` on the adapter

**Files:** Modify `algua/execution/alpaca_broker.py`; Test `tests/test_alpaca_broker.py`.

- [ ] **Step 1: Add failing tests** — append to `tests/test_alpaca_broker.py`:

```python
def test_close_all_positions_ok(monkeypatch):
    fake = _FakeRequestsWithDelete(_FakeResp(207, [{"symbol": "AAA", "status": 200}]))
    monkeypatch.setattr(ab, "requests", fake)
    _broker().close_all_positions()
    assert fake.deleted == ["https://paper-api.alpaca.markets/v2/positions?cancel_orders=true"]


def test_close_all_positions_empty_is_noop(monkeypatch):
    monkeypatch.setattr(ab, "requests", _FakeRequestsWithDelete(_FakeResp(207, [])))
    _broker().close_all_positions()  # no positions -> no error


def test_close_all_positions_partial_failure_raises(monkeypatch):
    fake = _FakeRequestsWithDelete(_FakeResp(207, [{"symbol": "AAA", "status": 200},
                                                   {"symbol": "BBB", "status": 500}]))
    monkeypatch.setattr(ab, "requests", fake)
    with pytest.raises(BrokerError):
        _broker().close_all_positions()
```

- [ ] **Step 2: Run** `uv run pytest tests/test_alpaca_broker.py -q` → FAIL.

- [ ] **Step 3: Implement** — in `algua/execution/alpaca_broker.py`, add after `cancel_open_orders`:

```python
    def close_all_positions(self) -> None:
        """Liquidate ALL positions and cancel open orders (DELETE /v2/positions?cancel_orders=true).
        Alpaca returns 207 multi-status with a per-position result list; raise if ANY close failed.
        Idempotent: no open positions -> empty list -> no-op."""
        results = self._read(
            self._delete("/v2/positions?cancel_orders=true"), "/v2/positions", ok=(200, 207)
        )
        if isinstance(results, list):
            failed = [r for r in results if int(r.get("status", 500)) not in (200, 204)]
            if failed:
                raise BrokerError(f"alpaca failed to close some positions: {failed}")
```

- [ ] **Step 4: Run** `uv run pytest tests/test_alpaca_broker.py -q` → PASS.

- [ ] **Step 5: Gate + commit** — `uv run ruff check . && uv run mypy algua && uv run pytest tests/test_alpaca_broker.py -q`

```bash
git add algua/execution/alpaca_broker.py tests/test_alpaca_broker.py
git commit -m "feat(broker): AlpacaPaperBroker.close_all_positions (liquidate + cancel)"
```

---

### Task 2: `paper flatten` + `trade-live` auto-flatten

**Files:** Modify `algua/cli/paper_cmd.py`; Test `tests/test_cli_paper.py`.

- [ ] **Step 1: Add failing tests** — append to `tests/test_cli_paper.py`:

```python
from algua.execution.alpaca_broker import BrokerError


class _FlattenBroker:
    def __init__(self, fail=False):
        self.fail = fail
        self.flattened = False

    def close_all_positions(self):
        if self.fail:
            raise BrokerError("alpaca failed to close some positions: [...]")
        self.flattened = True


def test_paper_flatten_closes_and_trips(monkeypatch):
    monkeypatch.setenv("ALGUA_ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALGUA_ALPACA_API_SECRET", "s")
    _to_paper()
    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings", _FlattenBroker)
    result = runner.invoke(app, ["paper", "flatten", "cross_sectional_momentum"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["flattened"] is True and payload["kill_switch"] == "tripped"
    show = json.loads(runner.invoke(app, ["paper", "show", "cross_sectional_momentum"]).stdout)
    assert show["kill_switch"]["tripped"] is True


def test_paper_flatten_rejects_non_paper_stage(monkeypatch):
    monkeypatch.setenv("ALGUA_ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALGUA_ALPACA_API_SECRET", "s")
    runner.invoke(app, ["registry", "add", "cross_sectional_momentum"])  # idea
    result = runner.invoke(app, ["paper", "flatten", "cross_sectional_momentum"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["ok"] is False


def test_paper_flatten_close_failure_stays_tripped(monkeypatch):
    monkeypatch.setenv("ALGUA_ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALGUA_ALPACA_API_SECRET", "s")
    _to_paper()
    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings",
                        lambda: _FlattenBroker(fail=True))
    result = runner.invoke(app, ["paper", "flatten", "cross_sectional_momentum"])
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False and payload["kill_switch"] == "tripped"
    # switch stayed tripped (fail-safe: halted even though the close failed)
    show = json.loads(runner.invoke(app, ["paper", "show", "cross_sectional_momentum"]).stdout)
    assert show["kill_switch"]["tripped"] is True
```

(Note: `_FlattenBroker` is passed as the factory; in `test_paper_flatten_closes_and_trips` it's used directly as the callable `_alpaca_broker_from_settings` — calling it returns a `_FlattenBroker()` instance.)

- [ ] **Step 2: Run** `uv run pytest tests/test_cli_paper.py -q` → FAIL.

- [ ] **Step 3: Implement** — in `algua/cli/paper_cmd.py`:

(a) Add the `paper flatten` command at the end of the file:

```python
@paper_app.command("flatten")
@json_errors(ValueError, LookupError, BrokerError)
def flatten(
    name: str,
    actor: str = typer.Option("agent", "--actor", help="human | agent"),
) -> None:
    """Emergency: close ALL live positions for a strategy and trip its kill-switch (halt)."""
    load_strategy(name)
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        rec = store.get_strategy(conn, name)
        if rec.stage is not Stage.PAPER:
            raise ValueError(f"{name} is at stage '{rec.stage.value}'; flatten requires 'paper'")
        broker = _alpaca_broker_from_settings()
        # Halt first (fail-safe): the strategy is stopped even if the close call then fails.
        kill_switch.trip(conn, name, reason="flatten", actor=actor)
        audit_append(conn, actor=actor, action="flatten", reason="manual flatten", strategy=name)
        try:
            broker.close_all_positions()
        except BrokerError as exc:
            emit({"ok": False, "strategy": name, "kill_switch": "tripped",
                  "flattened": False, "error": str(exc)})
            raise typer.Exit(1) from exc
    emit({"strategy": name, "kill_switch": "tripped", "flattened": True})
```

(b) Replace the `trade-live` `RiskBreach` handler. Change this block:

```python
        except RiskBreach as exc:
            kill_switch.trip(conn, name, reason=exc.detail, actor="system")
            audit_append(conn, actor="system", action="kill_switch_trip",
                         reason=f"{exc.kind}: {exc.detail}", strategy=name)
            emit({"ok": False, "kind": exc.kind, "kill_switch": "tripped", "error": exc.detail})
            raise typer.Exit(1) from exc
```

to:

```python
        except RiskBreach as exc:
            kill_switch.trip(conn, name, reason=exc.detail, actor="system")
            audit_append(conn, actor="system", action="kill_switch_trip",
                         reason=f"{exc.kind}: {exc.detail}", strategy=name)
            flattened = True
            try:
                broker.close_all_positions()
            except BrokerError as fexc:
                flattened = False
                audit_append(conn, actor="system", action="flatten_failed",
                             reason=str(fexc), strategy=name)
            emit({"ok": False, "kind": exc.kind, "kill_switch": "tripped",
                  "flattened": flattened, "error": exc.detail})
            raise typer.Exit(1) from exc
```

- [ ] **Step 4: Run** `uv run pytest tests/test_cli_paper.py -q` → PASS.

- [ ] **Step 5: Verify** — `uv run algua paper flatten --help` shows the command.

- [ ] **Step 6: Full gate + commit** — `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports` (all pass; `lint-imports` stays `10 kept, 0 broken`).

```bash
git add algua/cli/paper_cmd.py tests/test_cli_paper.py
git commit -m "feat(live): `paper flatten` + auto-flatten on live breach (close + halt)"
```

---

## Self-review notes

- **Spec coverage:** `close_all_positions` (§3 → Task 1); `paper flatten` fail-safe ordering
  (§4 → Task 2a); `trade-live` auto-flatten (§5 → Task 2b); tests incl. partial-failure and
  close-failure-stays-tripped (§6 → Tasks 1,2). Live acceptance is documented in the spec (§6),
  not code. No new import contract (§2).
- **Type consistency:** `close_all_positions()` (no args, returns None, raises `BrokerError`) is
  identical across Task 1 and the Task 2 callers; the `_FlattenBroker` test double exposes exactly
  `close_all_positions`. The `trade-live` handler reuses the in-scope `broker` from the same `with`
  block.
- **No placeholders:** every code step is complete.
- **Note:** in `test_paper_flatten_closes_and_trips`, `_alpaca_broker_from_settings` is monkeypatched
  to the `_FlattenBroker` class itself (so calling the factory returns a fresh `_FlattenBroker()`);
  the failure test patches it to `lambda: _FlattenBroker(fail=True)`.
