# Multi-tenant per-strategy paper tick + trade-tick rework (#316a) — design

**Status:** Draft (design approved in brainstorming; pending human review → implementation plan)
**Date:** 2026-06-30
**Issue:** #321 (#316a) — split from #316 (epic #318). #316b = the `paper run-all` batch, follow-up.
**Builds on:** #313 (`paper_reconcile`, merged 450071f), #314 (`build_paper_sizing_snapshot`, merged 9640d03), #249 (paper-venue ledger), the allocations module — all on main.

## 1. Problem

Under multi-tenant paper, each strategy must size off its own allocation/NAV and the account integrity
must be checked account-wide — not per #249's single-tenant `venue_belief` (which compared one
strategy's belief against the whole-account broker snapshot). This issue reworks the **per-strategy
paper tick** (exercised via `trade-tick`) to the multi-tenant model, replacing #249's single-tenant
path. The batch loop + shared buying-power reservation pool is the follow-up #316b (`paper run-all`),
which reuses the per-strategy helper this issue extracts.

## 2. Key enabler: `run_tick` already supports this

`run_tick` already sizes off `live_snapshot` (a ledger NAV snapshot) when supplied and reconciles
`venue_belief` only when supplied. So **`run_tick` itself is unchanged.** The switch is in the
paper lane's hook wiring (the caller), mirroring how live's `_run_strategy_tick` + `live run-all`
already work. (`live_snapshot`/`live_positions` are now used by paper too — a naming smell; the
lane-agnostic rename is deferred, like `LiveSizingError`, to avoid churning live.)

## 3. Components (`algua/cli/paper_cmd.py`)

### `_run_paper_strategy_tick(conn, name, broker, provider, max_drawdown, start, end, reserve_buy=None, cancel=None) -> dict` (new)
The reusable per-strategy multi-tenant tick — mirror of live's `_run_strategy_tick`:
- Read `active_allocation(conn, rec.id)`; raise if none (like live's "no allocation").
- Hooks: `live_snapshot = lambda bars: build_paper_sizing_snapshot(conn, name, allocation, bars,
  strategy.universe)`; `live_positions = lambda: paper_believed_positions(conn, name)`;
  `before_submit`/`on_submitted` = the existing paper-venue ledger recording (`record_paper_venue_order`
  / `backfill_paper_venue_broker_order_id`); `should_halt` (kill-switch + global halt); `cancel` (the
  scoped cancel, defaulting to this strategy's own open orders); `peak_equity = get_peak_equity(conn,
  name)`. **No `venue_belief`.**
- `run_tick(...)` → sizes off the per-strategy NAV.
- On `TickHalted`/`RiskBreach`: trip + **scoped** paper flatten (existing offset-flatten over the
  strategy's own believed positions) + audit; return a `{"ok": False, ...}` marker (so #316b's
  run-all can surface siblings on a breach, like live #270).
- Persist the tick snapshot with `equity = result.equity` (the per-strategy **NAV**, not whole-account)
  and `update_peak_equity(conn, name, result.peak_equity)`.

### `trade-tick` reworked
- Ingest venue fills (existing `_ingest_paper_venue`).
- **Account reconcile:** `cycle = paper_reconcile.next_cycle(conn)`;
  `recon = paper_reconcile.reconcile(conn, _paper_broker_net(broker), cycle)`. `recon.halt` →
  `global_halt.engage` + exit non-zero; `not recon.clean` → defer (emit, trade nothing this tick);
  clean → proceed. This account-wide check (via `attributed_paper_net`) preserves the `reconcile_ok`
  the forward gate requires, replacing #249's `venue_belief`.
- Call `_run_paper_strategy_tick(conn, name, ...)` once; emit its result.

### `_paper_broker_net(broker) -> dict[str, float]` (new, local)
The paper broker's net positions per symbol, fed to the reconcile. Local to `paper_cmd` because the
live analog (`_broker_net_positions` in `live_cmd`) can't be imported (cli→cli is import-linter-banned).

### Drawdown peak (refinement)
Keep the existing paper peak store (`get_peak_equity`/`update_peak_equity`) — **not** a switch to
`get_nav_peak`. Once `live_snapshot` is supplied, `run_tick` computes drawdown off **NAV**, so the
value flowing into the paper peak store is already NAV-based. Reusing the existing store keeps
`paper show` / `paper resume` (which read `get_peak_equity`) consistent without touching them.

## 4. Data flow (one `trade-tick`)

```
ingest venue fills (paginated, fail-closed)
  → reconcile(attributed_paper_net vs broker net, grace window)   [halt | defer | clean]
  → _run_paper_strategy_tick:
        build_paper_sizing_snapshot (NAV, attributed positions)
        run_tick (size off NAV; before_submit→record; on_submitted→backfill)
        breach → trip + scoped flatten
        persist tick_snapshot (equity = NAV, reconcile_ok)
```

## 5. Testing

- **`tests/test_cli_paper.py` (extend)** against a sim/fake paper broker + a seeded allocation row
  (inserted directly — no dependency on the `paper allocate` CLI):
  - `trade-tick` sizes off the **allocation/NAV**, not whole-account equity (account funded well above
    the allocation → orders target the allocation);
  - the tick snapshot's `equity` is the per-strategy NAV;
  - **account reconcile fires**: an unattributable broker holding → not clean → **defers, no trade**
    (the #249 phantom-flatten regression stays fixed) and `reconcile_ok` is recorded; a persistent
    unexplained residual → `halt` (global halt);
  - a breach trips the kill-switch + scoped-flattens only this strategy.
- **`_run_paper_strategy_tick` unit coverage:** happy tick (orders toward target), no-allocation →
  error, breach → trip + scoped flatten + `{"ok": False}` marker.
- **Live regression:** `live run-all` / `_run_strategy_tick` / live sizing tests unchanged (live is
  not touched).
- TDD; `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports` green.

## 6. Non-goals (this issue)

- The batch `paper run-all` loop + shared buying-power **reservation pool** → #316b.
- The forward-gate relaxation (`single_tenant` → attribution-clean) → #315.
- No `run_tick` body change; no schema change; no rename of `live_snapshot`/`live_positions`/
  `LiveSizingError` (deferred).

## 7. Risk

- **Touches the order path** (`trade-tick`) — the highest-risk change in the epic so far. Mitigated by:
  `run_tick` itself unchanged; the per-strategy helper mirrors the proven live `_run_strategy_tick`;
  the account reconcile reuses #313 (already reviewed); the existing paper breach/flatten/ledger
  recording is reused, not rewritten.
- A one-time drawdown-peak discontinuity for any in-flight paper strategy (account-equity peak →
  NAV peak). Acceptable: paper has not been operated in production, so no live evidence depends on a
  pre-existing peak.
