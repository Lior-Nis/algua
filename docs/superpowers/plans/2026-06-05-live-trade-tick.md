# `live trade-tick` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `algua live trade-tick <name>` places real-money orders on Alpaca's live venue — gated by `verify_live_authorization` (full at tick start, cheap revoke-check per order), driving the existing `run_tick` against `AlpacaLiveBroker`, with Alpaca as the source of truth and the same kill-switch/drawdown/scoped-flatten safety as paper.

**Architecture:** Add `live_gate.authorization_active` (cheap unrevoked-row check) + a per-order `should_halt` re-check in `run_tick`. Add `algua/cli/live_cmd.py` with the `live` Typer group, the `_alpaca_live_broker` factory (live keys + the `LiveAuthorization` tollbooth), and the `live trade-tick` command reusing `run_tick`, the shared breach helpers from `paper_cmd`, and the drawdown breaker. No local position ledger (`derived_positions=None`); orders recorded to `audit_log` only.

**Tech Stack:** Python 3.12, Typer, sqlite3, requests, ssh-keygen, pytest, ruff, mypy, import-linter. Spec: `docs/superpowers/specs/2026-06-05-live-trade-tick-design.md`.

---

## File structure

| File | Responsibility |
|---|---|
| `algua/registry/live_gate.py` (modify) | `authorization_active(conn, authorization) -> bool` (cheap revoke check). |
| `algua/live/live_loop.py` (modify) | re-check `should_halt` per submit-loop iteration. |
| `algua/cli/live_cmd.py` (new) | `live` Typer group; `_alpaca_live_broker`; `live trade-tick`. |
| `algua/cli/main.py` (modify) | import `live_cmd` so the `live` group registers. |

---

### Task 1: cheap revoke check + per-order `should_halt`

**Files:** Modify `algua/registry/live_gate.py`, `algua/live/live_loop.py`; Test `tests/test_live_gate.py`, `tests/test_live_loop.py`.

Context: `algua/registry/live_gate.py` already imports `LiveAuthorization` (from `algua.contracts.types`) and `sqlite3`; `LiveAuthorization` has `.strategy_id`, `.code_hash`, `.config_hash`, `.dependency_hash`. `tests/test_live_gate.py` has `_conn(tmp_path)` and `_seed_authorization(conn, tmp_path, *, code, cfg, dep, principal)` (issues a challenge, signs, calls `verify_and_consume` → writes a `live_authorizations` row for strategy_id=1) and `_live_strategy(conn)`. In `algua/live/live_loop.py` `run_tick`, the submit loop is `for intent in intents:` (currently the `should_halt` hook is checked only twice BEFORE the loop, raising `TickHalted`); `tests/test_live_loop.py` has `_FakeBroker`, `_FakeProvider`, `_strategy(weights, warmup_bars=0)`, `_bars`, `DATES`, and `TickHooks`/`TickHalted` imports.

- [ ] **Step 1: Add failing tests.** Append to `tests/test_live_gate.py`:
```python
def test_authorization_active_true_until_revoked(tmp_path):
    from algua.contracts.types import LiveAuthorization
    conn = _conn(tmp_path)
    _seed_authorization(conn, tmp_path, code="ch", cfg="cfg", dep="dep")
    auth = LiveAuthorization(strategy_id=1, code_hash="ch", config_hash="cfg",
                             dependency_hash="dep", principal="lior", authorized_at="t")
    assert live_gate.authorization_active(conn, auth) is True
    conn.execute("UPDATE live_authorizations SET revoked_at='2026-06-05' WHERE strategy_id=1")
    conn.commit()
    assert live_gate.authorization_active(conn, auth) is False
    # an identity with no row at all -> False
    other = LiveAuthorization(1, "OTHER", "cfg", "dep", "lior", "t")
    assert live_gate.authorization_active(conn, other) is False
```
Append to `tests/test_live_loop.py` (FIRST read `_FakeBroker` to confirm it records submitted orders and supports `snapshot`/`submit_sized`; mirror however the existing run_tick submit tests assert what was sent — adapt the assertion to the real interface):
```python
def test_run_tick_should_halt_aborts_between_orders():
    from algua.live.live_loop import TickHalted, TickHooks
    broker = _FakeBroker()
    bars = _bars({"AAA": [100.0, 100.0, 100.0], "BBB": [100.0, 100.0, 100.0]})  # 2 symbols
    calls = {"n": 0}

    def should_halt():
        calls["n"] += 1
        return calls["n"] > 3  # False for the 2 pre-loop checks + the 1st order; True before the 2nd

    hooks = TickHooks(should_halt=should_halt)
    with pytest.raises(TickHalted):
        run_tick(_strategy({"AAA": 0.5, "BBB": 0.5}), broker, _FakeProvider(bars),
                 DATES[0], DATES[-1], hooks=hooks)
    # only the FIRST order was submitted before the mid-loop halt
    assert len(broker.submitted) == 1
```
(If `_FakeBroker.submitted` is named differently or the symbol count/threshold needs tweaking so exactly one order lands before the halt, adjust `calls["n"] > N` and the assertion to match — the intent is: a `should_halt` that flips True mid-loop stops further orders and raises `TickHalted`.)

- [ ] **Step 2: Run** `uv run pytest tests/test_live_gate.py tests/test_live_loop.py -q` → FAIL.

- [ ] **Step 3: Add `authorization_active`** — append to `algua/registry/live_gate.py`:
```python
def authorization_active(conn: sqlite3.Connection, authorization: LiveAuthorization) -> bool:
    """Cheap mid-tick check: does an UNREVOKED live_authorizations row matching the (already
    re-verified) authorization's identity still exist? No ssh-keygen, no hash recompute — for the
    per-order should_halt hook so a revocation aborts the rest of a tick."""
    row = conn.execute(
        "SELECT 1 FROM live_authorizations WHERE strategy_id=? AND code_hash=? AND config_hash=? "
        "AND dependency_hash IS ? AND revoked_at IS NULL LIMIT 1",
        (authorization.strategy_id, authorization.code_hash, authorization.config_hash,
         authorization.dependency_hash),
    ).fetchone()
    return row is not None
```

- [ ] **Step 4: Per-order `should_halt`** — in `algua/live/live_loop.py`, at the TOP of the `for intent in intents:` loop (before computing `coid`), add:
```python
    for intent in intents:
        # Re-check before EACH order so a halt / authorization-revoke mid-loop stops further orders.
        if hooks.should_halt is not None and hooks.should_halt():
            raise TickHalted("kill-switch tripped during submit phase")
```
(Keep the two existing pre-loop checks; this adds the in-loop one.)

- [ ] **Step 5: Run** `uv run pytest tests/test_live_gate.py tests/test_live_loop.py -q` → PASS.

- [ ] **Step 6: Gate + commit** — `uv run ruff check . && uv run mypy algua && uv run pytest tests/test_live_gate.py tests/test_live_loop.py -q && uv run lint-imports`
```bash
git add algua/registry/live_gate.py algua/live/live_loop.py tests/test_live_gate.py tests/test_live_loop.py
git commit -m "feat(live): authorization_active cheap revoke-check + per-order should_halt in run_tick"
```

---

### Task 2: `live trade-tick` command + live broker factory

**Files:** Create `algua/cli/live_cmd.py`; Modify `algua/cli/main.py`; Test `tests/test_cli_live.py` (new).

Context: `paper_cmd.py` exposes `_trip(conn, name, exc)` (trips kill-switch + audits) and `_breach_payload(error, **extra) -> dict` (`{"ok": False, "kill_switch": "tripped", "error": ..., **extra}`). `algua/cli/_common.py` has `ok`, `registry_conn`, `utc`, `select_provider`. `algua/cli/app.py` has `app`, `emit`. `algua/cli/errors.py` has `json_errors`. `verify_live_authorization(conn, repo, name, allowed_signers_path) -> LiveAuthorization` and `authorization_active(conn, auth)` + `ALLOWED_SIGNERS_PATH` + `LiveAuthorizationError` are in `algua/registry/live_gate.py`. `AlpacaLiveBroker(authorization, api_key, api_secret, base_url)` and `BrokerError` are in `algua/execution/alpaca_broker.py`. `get_settings()` has `alpaca_live_api_key`/`alpaca_live_api_secret`/`alpaca_live_url`. `run_tick`, `TickHooks`, `TickHalted`, `SubmittedOrder` are in `algua/live/live_loop.py`; `RiskBreach` in `algua/risk/limits.py`; `client_order_id`, `get_peak_equity`, `update_peak_equity`, `record_tick_snapshot` in `algua/execution/order_state.py`; `kill_switch`, `global_halt` in `algua/risk`; `SqliteStrategyRepository` in `algua/registry/store.py`; `load_strategy` in `algua/strategies.loader`; `audit_append` = `from algua.audit.log import append as audit_append`. `main.py` imports the `_cmd` modules in a `from algua.cli import (backtest_cmd, data_cmd, paper_cmd, registry_cmd, research_cmd, strategy_cmd)` tuple.

- [ ] **Step 1: Add failing tests.** Create `tests/test_cli_live.py`:
```python
import json
from datetime import UTC, datetime

import pytest
from typer.testing import CliRunner

from algua.cli.main import app
from algua.contracts.types import LiveAuthorization
from algua.live.live_loop import TickResult
from algua.risk.limits import RiskBreach

runner = CliRunner()


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "p.db"))
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ALGUA_ALPACA_LIVE_API_KEY", "lk")
    monkeypatch.setenv("ALGUA_ALPACA_LIVE_API_SECRET", "ls")


def _to_live(name="cross_sectional_momentum"):
    # bring a strategy to 'live' stage in the DB directly (the signed ceremony is tested elsewhere)
    from contextlib import closing

    from algua.config.settings import get_settings
    from algua.registry.db import connect, migrate
    assert runner.invoke(app, ["backtest", "run", name, "--demo", "--register",
                               "--start", "2022-01-01", "--end", "2023-12-31"]).exit_code == 0
    for to in ("shortlisted", "paper"):
        runner.invoke(app, ["registry", "transition", name, "--to", to, "--actor", "agent",
                            "--reason", "x"])
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        conn.execute("UPDATE strategies SET stage='live' WHERE name=?", (name,))
        conn.commit()


def _auth():
    return LiveAuthorization(1, "c", "cf", "d", "lior", "t")


def test_live_trade_tick_refused_without_authorization(monkeypatch):
    from algua.registry.live_gate import LiveAuthorizationError
    _to_live()
    monkeypatch.setattr("algua.cli.live_cmd.verify_live_authorization",
                        lambda *a, **k: (_ for _ in ()).throw(LiveAuthorizationError("nope")))
    r = runner.invoke(app, ["live", "trade-tick", "cross_sectional_momentum", "--snapshot", "x"])
    assert r.exit_code == 1 and json.loads(r.stdout)["ok"] is False


def test_live_trade_tick_refused_when_killed(monkeypatch):
    _to_live()
    monkeypatch.setattr("algua.cli.live_cmd.verify_live_authorization", lambda *a, **k: _auth())
    runner.invoke(app, ["paper", "kill", "cross_sectional_momentum", "--reason", "x"])
    r = runner.invoke(app, ["live", "trade-tick", "cross_sectional_momentum", "--snapshot", "x"])
    assert r.exit_code == 1 and json.loads(r.stdout)["ok"] is False


def test_live_trade_tick_missing_live_keys(monkeypatch):
    _to_live()
    monkeypatch.setenv("ALGUA_ALPACA_LIVE_API_KEY", "")
    monkeypatch.setattr("algua.cli.live_cmd.verify_live_authorization", lambda *a, **k: _auth())
    r = runner.invoke(app, ["live", "trade-tick", "cross_sectional_momentum", "--snapshot", "x"])
    assert r.exit_code == 1 and json.loads(r.stdout)["ok"] is False


def test_live_trade_tick_happy_path(monkeypatch):
    _to_live()
    ts = datetime(2023, 6, 1, tzinfo=UTC)
    fake = TickResult(decision_ts=ts, target_weights={"AAA": 1.0}, positions_before={},
                      submitted=[{"symbol": "AAA", "side": "buy", "target_weight": 1.0,
                                  "order_id": "o-1", "client_order_id": "c"}],
                      equity=50000.0, peak_equity=50000.0)
    monkeypatch.setattr("algua.cli.live_cmd.verify_live_authorization", lambda *a, **k: _auth())
    monkeypatch.setattr("algua.cli.live_cmd._alpaca_live_broker", lambda auth: object())
    monkeypatch.setattr("algua.cli.live_cmd._select_provider", lambda demo, snapshot: object())
    monkeypatch.setattr("algua.cli.live_cmd.run_tick", lambda *a, **k: fake)
    r = runner.invoke(app, ["live", "trade-tick", "cross_sectional_momentum", "--snapshot", "x"])
    assert r.exit_code == 0, r.stdout
    payload = json.loads(r.stdout)
    assert payload["submitted"][0]["order_id"] == "o-1"
    # the live order was audited and a tick snapshot recorded
    from contextlib import closing

    from algua.config.settings import get_settings
    from algua.execution.order_state import latest_tick_snapshot
    from algua.registry.db import connect, migrate
    with closing(connect(get_settings().db_path)) as conn:
        migrate(conn)
        assert latest_tick_snapshot(conn, "cross_sectional_momentum") is not None
        n = conn.execute("SELECT COUNT(*) FROM audit_log WHERE action='live_order'").fetchone()[0]
        assert n == 1


def test_live_trade_tick_breach_trips_and_flattens(monkeypatch):
    _to_live()
    monkeypatch.setattr("algua.cli.live_cmd.verify_live_authorization", lambda *a, **k: _auth())

    class _FlatBroker:
        def __init__(self):
            self.closed = None
        def cancel_open_orders(self):
            pass
        def close_positions(self, syms):
            self.closed = list(syms)

    broker = _FlatBroker()
    monkeypatch.setattr("algua.cli.live_cmd._alpaca_live_broker", lambda auth: broker)
    monkeypatch.setattr("algua.cli.live_cmd._select_provider", lambda demo, snapshot: object())
    monkeypatch.setattr("algua.cli.live_cmd.run_tick",
                        lambda *a, **k: (_ for _ in ()).throw(RiskBreach("drawdown", "dd")))
    r = runner.invoke(app, ["live", "trade-tick", "cross_sectional_momentum", "--snapshot", "x"])
    assert r.exit_code == 1
    payload = json.loads(r.stdout)
    assert payload["ok"] is False and payload["kind"] == "drawdown"
    assert broker.closed is not None  # scoped flatten ran on the live broker
```

- [ ] **Step 2: Run** `uv run pytest tests/test_cli_live.py -q` → FAIL.

- [ ] **Step 3: Create `algua/cli/live_cmd.py`:**
```python
from __future__ import annotations

from datetime import UTC, datetime

import typer

from algua.audit.log import append as audit_append
from algua.cli._common import ok, registry_conn, utc
from algua.cli._common import select_provider as _select_provider
from algua.cli.app import app, emit
from algua.cli.errors import json_errors
from algua.cli.paper_cmd import _breach_payload, _trip
from algua.config.settings import get_settings
from algua.contracts.types import LiveAuthorization
from algua.execution.alpaca_broker import AlpacaLiveBroker, BrokerError
from algua.execution.order_state import (
    client_order_id,
    get_peak_equity,
    record_tick_snapshot,
    update_peak_equity,
)
from algua.live.live_loop import SubmittedOrder, TickHalted, TickHooks, run_tick
from algua.registry.live_gate import (
    ALLOWED_SIGNERS_PATH,
    LiveAuthorizationError,
    authorization_active,
    verify_live_authorization,
)
from algua.registry.store import SqliteStrategyRepository
from algua.risk import global_halt, kill_switch
from algua.risk.limits import RiskBreach
from algua.strategies.loader import load_strategy

live_app = typer.Typer(help="LIVE (real-money) trading — human-authorized strategies only",
                       no_args_is_help=True)
app.add_typer(live_app, name="live")


def _alpaca_live_broker(authorization: LiveAuthorization) -> AlpacaLiveBroker:
    s = get_settings()
    if not s.alpaca_live_api_key or not s.alpaca_live_api_secret:
        raise ValueError(
            "Alpaca LIVE credentials not configured; set ALGUA_ALPACA_LIVE_API_KEY "
            "and ALGUA_ALPACA_LIVE_API_SECRET"
        )
    return AlpacaLiveBroker(authorization, s.alpaca_live_api_key, s.alpaca_live_api_secret,
                            base_url=s.alpaca_live_url)


@live_app.command("trade-tick")
@json_errors(ValueError, LookupError, BrokerError, LiveAuthorizationError)
def trade_tick(
    name: str,
    snapshot: str = typer.Option(..., "--snapshot", help="ingested bars snapshot id"),
    start: str = typer.Option("2023-01-01", "--start"),
    end: str = typer.Option("2023-12-31", "--end"),
    max_drawdown: float = typer.Option(None, "--max-drawdown",
                                       help="halt + flatten if equity falls this fraction below the persisted peak"),  # noqa: E501
) -> None:
    """Run ONE wall-clock tick against the Alpaca LIVE venue (REAL MONEY). Re-verifies the human
    go-live signature against the trust anchor before trading; Alpaca is the source of truth (no
    local ledger). A drawdown/exposure breach trips the kill-switch and scoped-flattens."""
    if max_drawdown is not None and not 0.0 < max_drawdown <= 1.0:
        raise ValueError("--max-drawdown must be in (0, 1]")
    with registry_conn() as conn:
        repo = SqliteStrategyRepository(conn)
        # THE WALL: re-verify the human signature for the current artifact (also requires Stage.LIVE).
        authorization = verify_live_authorization(conn, repo, name, ALLOWED_SIGNERS_PATH)
        if kill_switch.is_tripped(conn, name) or global_halt.is_engaged(conn):
            raise ValueError(f"{name} is halted; resume before live trading")
        strategy = load_strategy(name)
        broker = _alpaca_live_broker(authorization)
        provider = _select_provider(False, snapshot)

        def _persist(record: SubmittedOrder) -> None:
            audit_append(conn, actor="agent", action="live_order",
                         reason=f"{record.side} {record.symbol} {record.order_id}", strategy=name)

        hooks = TickHooks(
            client_order_id_for=client_order_id,
            on_submitted=_persist,
            should_halt=lambda: (kill_switch.is_tripped(conn, name)
                                 or global_halt.is_engaged(conn)
                                 or not authorization_active(conn, authorization)),
            peak_equity=get_peak_equity(conn, name),
            derived_positions=None,  # Alpaca is the sole source of truth; no local-ledger reconcile
        )
        try:
            result = run_tick(strategy, broker, provider, utc(start), utc(end),
                              hooks=hooks, max_drawdown=max_drawdown)
        except TickHalted as exc:
            audit_append(conn, actor="system", action="live_trade_tick_halted",
                         reason=str(exc), strategy=name)
            emit(_breach_payload(str(exc), strategy=name, halted=True))
            raise typer.Exit(1) from exc
        except RiskBreach as exc:
            _trip(conn, name, exc)
            liquidation_submitted = True
            flatten_error = None
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
        if result.peak_equity is not None:
            update_peak_equity(conn, name, result.peak_equity)
            record_tick_snapshot(
                conn, name, tick_ts=datetime.now(UTC).isoformat(),
                decision_ts=result.decision_ts.isoformat() if result.decision_ts else None,
                equity=result.equity, peak_equity=result.peak_equity,
                positions=result.positions_before, n_submitted=len(result.submitted),
                reconcile_ok=result.reconcile_ok,
            )
        audit_append(conn, actor="agent", action="live_trade_tick",
                     reason=f"{len(result.submitted)} live orders submitted", strategy=name)

    emit(ok({
        "strategy": name,
        "venue": "live",
        "decision_ts": result.decision_ts.isoformat() if result.decision_ts else None,
        "submitted": result.submitted,
        "reconcile_ok": result.reconcile_ok,
    }))
```

- [ ] **Step 4: Register the group** — in `algua/cli/main.py`, add `live_cmd,` to the `from algua.cli import (...)` tuple (keep it sorted: after `data_cmd,`).

- [ ] **Step 5: Run** `uv run pytest tests/test_cli_live.py -q` → PASS.

- [ ] **Step 6: Full gate + commit** — `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports` (all pass; contracts kept, 0 broken).
```bash
git add algua/cli/live_cmd.py algua/cli/main.py tests/test_cli_live.py
git commit -m "feat(cli): live trade-tick — gated real-money loop (verify per tick + per-order revoke)"
```

---

## Self-review notes

- **Spec coverage:** the gate (verify at start + kill/halt) (§3 → Task 2); `_alpaca_live_broker` factory + hooks with `derived_positions=None` + audit-only persistence + the cheap revoke check in `should_halt` (§4 → Tasks 1–2); scoped flatten on breach (§5 → Task 2); shared helpers imported from `paper_cmd` + the `live` group registered + the per-order `should_halt` in `run_tick` (§6 → Tasks 1–2); the walls (§7); tests for refused-without-auth / killed / missing-keys / happy / breach + `authorization_active` + per-order halt (§8 → Tasks 1–2). `--max-drawdown` validation mirrors paper.
- **Type consistency:** `authorization_active(conn, authorization) -> bool` (Task 1) is called in `live_cmd`'s `should_halt` (Task 2); `verify_live_authorization(...) -> LiveAuthorization` is consumed by `_alpaca_live_broker(authorization)` and the tollbooth; `_breach_payload`/`_trip` are imported from `paper_cmd` (same signatures the paper command uses); `record_tick_snapshot`/`update_peak_equity`/`get_peak_equity` match `order_state`.
- **No placeholders:** the live_cmd body is complete. The two "match the real `_FakeBroker`" / "adjust the halt threshold" notes in Task 1 are verifications against the existing test double, not gaps. CLI tests monkeypatch `verify_live_authorization` (the gate's crypto is exhaustively unit-tested in `test_live_gate.py`); the command test exercises the WIRING (gate called, broker built, run_tick driven, audit/snapshot written, breach scoped-flattens).
- **Security invariant:** the command cannot trade unless `verify_live_authorization` passes (re-verifies the human signature against the CODEOWNERS anchor for the current code) AND live keys are present AND it isn't killed/halted/revoked; an agent has none of these, and the per-order revoke check stops a tick the instant authorization is pulled.
