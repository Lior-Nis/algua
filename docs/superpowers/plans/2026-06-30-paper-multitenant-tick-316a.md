# Multi-tenant per-strategy paper tick + trade-tick rework (#316a) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rework the per-strategy paper tick to multi-tenant: size off the strategy's own allocation/NAV (#314) and do an account-wide reconcile (#313) in `trade-tick`, replacing #249's single-tenant `venue_belief`; extract a reusable `_run_paper_strategy_tick` helper for #316b's run-all.

**Architecture:** `run_tick` is unchanged — it already sizes off `live_snapshot` and skips the in-tick reconcile when `venue_belief` is unset. The change is in `paper_cmd.py`: extract the per-strategy tick into a helper that supplies `build_paper_sizing_snapshot` + `paper_believed_positions` (no `venue_belief`), and rework `trade-tick` to ingest → account-reconcile via `paper_reconcile` → call the helper.

**Tech Stack:** Python 3.12, Typer, SQLite, pytest. Builds on #313/#314/#249 (all on main).

## Global Constraints

- Run via `uv run ...`. Full gate green before every commit: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`.
- Ruff line length ≤ 100 columns.
- `git add` only the named files — never `git add -A`.
- Commit trailer: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- **Do not touch the live lane** (`live_cmd.py`, `_run_strategy_tick`, `live_loop.run_tick`); its tests must stay green.
- **Do not import `live_cmd` from `paper_cmd`** (cli→cli is import-linter-banned) — hence the local `_paper_broker_net`.
- On a not-clean account reconcile, `trade-tick` **defers** (emit, no orders, no tick snapshot); a `halt` (persistent drift) engages the global halt. Only a clean reconcile ticks. (Matches the spec + live `run-all`.)
- Branch: `paper-multitenant-tick-316a` (already created off main).

---

## File structure

- `algua/cli/paper_cmd.py` — **Modify.** Add `_paper_broker_net`; add `_run_paper_strategy_tick`; rework `trade_tick`. Add imports (`active_allocation`, `build_paper_sizing_snapshot`, `paper_reconcile`, `global_halt` if not present).
- `tests/test_cli_paper.py` — **Modify.** Add multi-tenant `trade-tick` tests (NAV sizing, reconcile defer, breach) reusing a paper fake broker.

---

### Task 1: `_paper_broker_net(broker)` helper

**Files:**
- Modify: `algua/cli/paper_cmd.py`
- Test: `tests/test_cli_paper.py`

**Interfaces:**
- Consumes: a broker exposing `get_positions() -> pandas.Series` (symbol→qty), like `AlpacaPaperBroker`.
- Produces: `_paper_broker_net(broker) -> dict[str, float]` — nonzero net positions per symbol (paper analog of live's `_broker_net_positions`).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_cli_paper.py`:

```python
def test_paper_broker_net_drops_zero_positions():
    import pandas as pd
    from algua.cli.paper_cmd import _paper_broker_net

    class _B:
        def get_positions(self):
            return pd.Series({"AAA": 10.0, "BBB": 0.0, "CCC": -3.0})

    assert _paper_broker_net(_B()) == {"AAA": 10.0, "CCC": -3.0}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli_paper.py::test_paper_broker_net_drops_zero_positions -v`
Expected: FAIL — `ImportError: cannot import name '_paper_broker_net'`.

- [ ] **Step 3: Add the helper**

In `algua/cli/paper_cmd.py`, add near the other module-level helpers (e.g. just below `_alpaca_broker_from_settings`):

```python
def _paper_broker_net(broker) -> dict[str, float]:
    """The paper broker's net positions per symbol (nonzero only) — fed to the account reconcile.
    Local to paper_cmd because the live analog (_broker_net_positions) can't be imported (cli->cli)."""
    pos = broker.get_positions()  # pandas Series symbol -> qty
    return {sym: float(q) for sym, q in pos.items() if float(q) != 0.0}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cli_paper.py::test_paper_broker_net_drops_zero_positions -v`
Expected: PASS.

- [ ] **Step 5: Lint/type + commit**

```bash
uv run ruff check algua/cli/paper_cmd.py tests/test_cli_paper.py && uv run mypy algua/cli/paper_cmd.py
git add algua/cli/paper_cmd.py tests/test_cli_paper.py
git commit -m "feat(paper): _paper_broker_net helper for the account reconcile #316a

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Extract `_run_paper_strategy_tick` + rework `trade-tick` (multi-tenant)

**Files:**
- Modify: `algua/cli/paper_cmd.py`
- Test: `tests/test_cli_paper.py`

**Interfaces:**
- Consumes: `_paper_broker_net` (Task 1); `build_paper_sizing_snapshot` (#314, from `algua.execution.live_sizing`); `paper_reconcile` (#313, `algua.execution.paper_reconcile` — `next_cycle`, `reconcile`); `active_allocation` (`algua.registry.allocations`); existing `paper_believed_positions`, `record_paper_venue_order`, `backfill_paper_venue_broker_order_id`, `_ingest_paper_venue`, `client_order_id`, `get_peak_equity`/`update_peak_equity`, `record_tick_snapshot`, `kill_switch`, `global_halt`, `trip_for_breach`, `TickHooks`, `run_tick`, `TickHalted`, `RiskBreach`, `_RECONCILE_TOL`, `tick_clock`, `compute_artifact_hashes`.
- Produces: `_run_paper_strategy_tick(conn, name, strategy, rec, broker, provider, max_drawdown, tick_ts, clock_source, acct, *, cancel=None) -> dict` — the reusable per-strategy multi-tenant tick (#316b will call it per strategy). Returns `ok({...})` on success or `{"ok": False, ...}` on halt/breach (never raises for those).

- [ ] **Step 1: Add the imports**

In `algua/cli/paper_cmd.py`, add to the existing import groups:

```python
from algua.execution.live_sizing import build_paper_sizing_snapshot
from algua.execution import paper_reconcile
from algua.registry.allocations import active_allocation
```

(Verify `global_halt` is already imported — it is used by the current `should_halt`. If not, add `from algua.risk import global_halt`.)

- [ ] **Step 2: Write the failing multi-tenant tests**

Add to `tests/test_cli_paper.py`. These drive `trade-tick` through real `run_tick` with a fake paper broker (mirror the `_PaperVenueTestBroker` in `tests/test_paper_venue_reconcile.py`; that fake needs a `get_positions()` returning a `pandas.Series` — add it if absent). Seed a `strategy_allocations` row directly (no dependency on the `paper allocate` CLI). Full helper/fixtures per the existing file's style:

```python
def test_trade_tick_sizes_off_allocation_not_account(monkeypatch, tmp_path):
    # A paper strategy at PAPER stage with a $10k allocation, account funded at $1M.
    # Orders must target the $10k allocation/NAV, not the $1M account equity.
    name = _paper_strategy_with_allocation(monkeypatch, tmp_path, capital=10_000.0,
                                           account_equity=1_000_000.0)
    broker = _FakePaperBroker(account_equity=1_000_000.0, positions={}, marks={"AAA": 100.0})
    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings", lambda: broker)
    result = runner.invoke(app, ["paper", "trade-tick", name, "--snapshot", _SNAP,
                                 "--start", "2026-01-01", "--end", "2026-02-01"])
    assert result.exit_code == 0, result.output
    # the recorded tick snapshot equity is the per-strategy NAV (~allocation), not 1_000_000
    snap = _latest_tick(name)
    assert snap["equity"] <= 10_000.0


def test_trade_tick_defers_on_unattributable_holding(monkeypatch, tmp_path):
    # Broker shows a holding no paper strategy owns -> reconcile not clean -> defer, no orders.
    name = _paper_strategy_with_allocation(monkeypatch, tmp_path, capital=10_000.0,
                                           account_equity=100_000.0)
    broker = _FakePaperBroker(account_equity=100_000.0, positions={"ZZZ": 5.0}, marks={"AAA": 100.0})
    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings", lambda: broker)
    result = runner.invoke(app, ["paper", "trade-tick", name, "--snapshot", _SNAP,
                                 "--start", "2026-01-01", "--end", "2026-02-01"])
    payload = json.loads(result.output)
    assert payload.get("traded") is False or payload.get("deferred") is True
    assert broker.submitted == []   # nothing traded on an unreconciled account


def test_trade_tick_breach_trips_and_scoped_flattens(monkeypatch, tmp_path):
    name = _paper_strategy_with_allocation(monkeypatch, tmp_path, capital=10_000.0,
                                           account_equity=100_000.0)
    broker = _FakePaperBroker(account_equity=100_000.0, positions={}, marks={"AAA": 100.0},
                              force_breach=True)
    monkeypatch.setattr("algua.cli.paper_cmd._alpaca_broker_from_settings", lambda: broker)
    result = runner.invoke(app, ["paper", "trade-tick", name, "--snapshot", _SNAP,
                                 "--start", "2026-01-01", "--end", "2026-02-01", "--max-drawdown", "0.01"])
    assert result.exit_code != 0
    assert kill_switch_is_tripped(name)
```

> Implementer note: build `_FakePaperBroker`, `_paper_strategy_with_allocation`, `_latest_tick`, `_SNAP`, and `kill_switch_is_tripped` by reusing the patterns already in `tests/test_cli_paper.py` and `tests/test_paper_venue_reconcile.py` (the `_PaperVenueTestBroker` + the existing `_isolated` autouse fixture + an ingested demo snapshot). Keep the three assertions above intact; adapt the harness mechanics as the existing fixtures require. The deferred-payload key (`traded`/`deferred`) must match what `trade_tick` emits in Step 4 — keep them consistent.

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_cli_paper.py -k "trade_tick_sizes or trade_tick_defers or trade_tick_breach" -v`
Expected: FAIL (helper not present / behavior not implemented).

- [ ] **Step 4: Implement the helper + rework `trade_tick`**

Add the helper and replace the body of `trade_tick`. The helper absorbs the current tick + breach/flatten + snapshot logic, switching the sizing hooks and dropping `venue_belief`:

```python
def _run_paper_strategy_tick(  # noqa: PLR0913
    conn, name: str, strategy, rec, broker, provider, max_drawdown,
    tick_ts, clock_source, acct, *, cancel=None,
    start: str = "2023-01-01", end: str = "2023-12-31",
) -> dict:
    """ONE strategy's multi-tenant paper tick: NAV-snapshot sizing (#314), crash-safe ledger
    recording, breach trip + scoped flatten, tick-snapshot persistence (equity = per-strategy NAV).
    Returns ok({...}) on success, or an {"ok": False, ...} marker on TickHalted/RiskBreach (so
    #316b's run-all can surface siblings on a breach, like live #270). Caller does the account
    reconcile BEFORE calling this; no venue_belief here."""
    alloc = active_allocation(conn, rec.id)
    if alloc is None:
        raise ValueError(f"{name} has no paper allocation")
    allocation = float(alloc["capital"])
    identity = compute_artifact_hashes(name)

    hooks = TickHooks(
        client_order_id_for=client_order_id,
        before_submit=lambda intent, coid: (
            record_paper_venue_order(conn, name, intent.symbol, intent.side.value, None,
                                     coid, strategy_id=rec.id) if coid is not None else None),
        on_submitted=lambda rec_: backfill_paper_venue_broker_order_id(
            conn, rec_.client_order_id, rec_.order_id),
        should_halt=lambda: kill_switch.is_tripped(conn, name) or global_halt.is_engaged(conn),
        cancel=cancel,
        peak_equity=get_peak_equity(conn, name),
        live_snapshot=lambda bars: build_paper_sizing_snapshot(
            conn, name, allocation, bars, strategy.universe),
        live_positions=lambda: paper_believed_positions(conn, name),
    )
    try:
        result = run_tick(strategy, broker, provider, utc(start), utc(end),
                          hooks=hooks, max_drawdown=max_drawdown)
    except TickHalted as exc:
        audit_append(conn, actor="system", action="trade_tick_halted", reason=str(exc),
                     strategy=name)
        return {"ok": False, "strategy": name, **breach_payload(str(exc), strategy=name,
                                                                halted=True)}
    except RiskBreach as exc:
        trip_for_breach(conn, name, exc)
        n_offsets = 0
        flatten_error = None
        try:
            broker.cancel_open_orders()
            breach_ts, _ = tick_clock(broker.clock)
            _ingest_paper_venue(conn, broker, breach_ts)
            for sym, qty in paper_believed_positions(conn, name).items():
                if abs(qty) <= _RECONCILE_TOL:
                    continue
                coid = client_order_id(name, datetime.now(UTC), sym)
                record_paper_venue_order(conn, name, sym, "sell" if qty > 0 else "buy",
                                         None, coid, strategy_id=rec.id)
                oid = broker.submit_offset(sym, qty, coid)
                backfill_paper_venue_broker_order_id(conn, coid, oid)
                n_offsets += 1
        except Exception as fexc:
            flatten_error = str(fexc)
            audit_append(conn, actor="system", action="flatten_failed", reason=str(fexc),
                         strategy=name)
        payload = breach_payload(exc.detail, kind=exc.kind, liquidation_submitted=n_offsets > 0,
                                 offsets_submitted=n_offsets)
        if flatten_error is not None:
            payload["flatten_error"] = flatten_error
        return {"ok": False, "strategy": name, **payload}

    if result.peak_equity is not None:
        update_peak_equity(conn, name, result.peak_equity)
        record_tick_snapshot(
            conn, name, tick_ts=tick_ts,
            decision_ts=result.decision_ts.isoformat() if result.decision_ts else None,
            equity=result.equity, peak_equity=result.peak_equity,
            positions=result.positions_before, n_submitted=len(result.submitted),
            reconcile_ok=result.reconcile_ok, lane="paper", strategy_id=rec.id,
            code_hash=identity.code_hash, config_hash=identity.config_hash,
            dependency_hash=identity.dependency_hash, account_id=acct.account_id,
            cash=acct.cash, clock_source=clock_source)
    audit_append(conn, actor="agent", action="trade_tick",
                 reason=f"{len(result.submitted)} orders submitted", strategy=name)
    return ok({
        "strategy": name,
        "decision_ts": result.decision_ts.isoformat() if result.decision_ts else None,
        "target_weights": result.target_weights, "positions_before": result.positions_before,
        "submitted": result.submitted, "reconcile_ok": result.reconcile_ok,
        "realized_gross": result.realized_gross})
```

Then replace the body of `trade_tick` (keep its signature + the `--max-drawdown` validation) with: load → ingest → account reconcile → defer/halt/clean → helper:

```python
    if max_drawdown is not None and not 0.0 < max_drawdown <= 1.0:
        raise ValueError("--max-drawdown must be in (0, 1]")
    with registry_conn() as conn:
        strategy, rec = load_gated_strategy(conn, name, "trade-tick")
        broker = _alpaca_broker_from_settings()
        provider = _select_provider(False, snapshot)
        acct = broker.account()
        tick_ts, clock_source = tick_clock(broker.clock)
        try:
            _ingest_paper_venue(conn, broker, tick_ts)
        except Exception as exc:   # fail closed on ANY ingest/transport error
            audit_append(conn, actor="system", action="venue_ingest_failed", reason=str(exc),
                         strategy=name)
            emit(breach_payload(str(exc), strategy=name, kind="venue_ingest_failed"))
            raise typer.Exit(1) from exc

        # Account-wide reconcile (multi-tenant): attributed_paper_net vs the broker book, grace
        # window. halt -> global halt; not clean -> defer (no trade); clean -> tick.
        cycle = paper_reconcile.next_cycle(conn)
        recon = paper_reconcile.reconcile(conn, _paper_broker_net(broker), cycle)
        if recon.halt:
            global_halt.engage(conn, reason=f"paper reconcile drift {recon.mismatches}",
                               actor="system")
            emit({"ok": False, "strategy": name, "deferred": True, "halted": True,
                  "reconcile": recon.mismatches})
            raise typer.Exit(1)
        if not recon.clean:
            audit_append(conn, actor="system", action="trade_tick_deferred",
                         reason="reconcile pending", strategy=name)
            emit(ok({"strategy": name, "traded": False, "deferred": True,
                     "reconcile": recon.mismatches}))
            return

        out = _run_paper_strategy_tick(conn, name, strategy, rec, broker, provider, max_drawdown,
                                       tick_ts, clock_source, acct)
    emit(out)
    if not out.get("ok", False):
        raise typer.Exit(1)
```

Remove the now-unused `identity = compute_artifact_hashes(name)` line from `trade_tick` (it moved into the helper) and any import left unused after the rework (e.g. `attributed_live_net` if it is no longer referenced — check before deleting).

- [ ] **Step 5: Run the new tests + paper regression**

Run: `uv run pytest tests/test_cli_paper.py tests/test_paper_venue_reconcile.py -v`
Expected: PASS — the three new tests pass and the existing paper-CLI / venue-reconcile tests still pass (adjust any existing trade-tick test that asserted the old single-tenant `venue_belief` behavior, if present, to the new defer/clean semantics; do NOT weaken an assertion to pass).

- [ ] **Step 6: Full quality gate**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add algua/cli/paper_cmd.py tests/test_cli_paper.py
git commit -m "feat(paper): multi-tenant per-strategy tick + trade-tick account reconcile #316a

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-review

- **Spec coverage:** §3 `_run_paper_strategy_tick` (NAV snapshot, drop venue_belief, scoped flatten, NAV-peak snapshot) → Task 2; reworked `trade-tick` (ingest → reconcile → helper) → Task 2; `_paper_broker_net` → Task 1; peak-store refinement (keep `get_peak_equity`/`update_peak_equity`) → Task 2 helper. §5 testing (NAV sizing, reconcile defer, breach) → Task 2 Step 2. §6 non-goals respected (no run-all/reservation pool; no run_tick body change; no rename). §7 risk (order path) mitigated by reusing the proven breach/flatten/ledger blocks verbatim.
- **Placeholder scan:** the only prose-not-code spot is the Task 2 Step 2 implementer note for harness mechanics (fakes/fixtures), which is unavoidable given the existing test infra — the three assertions are concrete; flagged, not a silent gap.
- **Type consistency:** `_run_paper_strategy_tick(conn, name, strategy, rec, broker, provider, max_drawdown, tick_ts, clock_source, acct, *, cancel=None, start, end) -> dict` defined and called consistently; `_paper_broker_net(broker) -> dict[str, float]` defined in Task 1 and used in Task 2; `paper_reconcile.next_cycle`/`reconcile` and `build_paper_sizing_snapshot`/`active_allocation` signatures match the merged #313/#314 + allocations module.

## Note for #316b (run-all)

`_run_paper_strategy_tick` takes `tick_ts`/`clock_source`/`acct` as parameters precisely so run-all can ingest + reconcile + resolve the clock ONCE per cycle, then call it per due strategy with a `cancel`/(future) `reserve_buy`. Keep that boundary stable.
