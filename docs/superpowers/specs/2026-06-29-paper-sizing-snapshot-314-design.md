# Per-strategy paper sizing snapshot + virtual NAV (#314) — design

**Status:** Draft (design approved in brainstorming; pending human review → implementation plan)
**Date:** 2026-06-29
**Issue:** #314 (epic #318 — multi-tenant attributed paper forward-testing, live-parity)
**Builds on:** #249 / PR#310 (the paper-venue fill ledger + `LedgerKind` seam). Independent of #313.

## 1. Problem

Under multi-tenant paper, each strategy must size off **its own** attributed book and a clean
**per-strategy virtual NAV**, not the shared account. Live already does this:
`build_live_sizing_snapshot` returns a `SizingSnapshot{equity, market_values, qtys}` + `nav`, where
`nav = allocation + Σ position_pnl(own fills marked to latest close)` and
`equity = min(allocation, nav)`, failing closed when a held symbol has no usable mark. Paper has no
equivalent — the wall-clock lane sizes off the whole-account broker snapshot (`snap.equity`), which
is wrong the moment two strategies share an account.

#249 already generalized the ledger with a `LedgerKind` enum (LIVE/PAPER), `believed_positions(conn,
strategy, kind)`, and lane-agnostic `position_pnl`. So the paper sizing snapshot is a faithful
generalization of the live one over `LedgerKind.PAPER`.

## 2. Scope & boundaries

This issue is the **per-strategy sizing-snapshot primitive only** — additive, unit-tested, unwired.
The `run_tick`/`trade-tick` change that consumes it (paper sizing off this snapshot instead of the
whole-account broker snapshot, dropping #249's single-tenant in-tick reconcile, and recording
per-strategy attributed equity in `tick_snapshots`) **must change together with the driver** (#316),
atomically — isolating it here would leave `trade-tick` sizing per-strategy with no integrity check
until #316 adds the account reconcile (a broken intermediate). So:

- **#314 (this):** `build_sizing_snapshot(..., kind)` + `build_paper_sizing_snapshot` alias. No
  `run_tick`/CLI change; nothing wired. The existing single-tenant `trade-tick` is untouched.
- **#316:** wires the snapshot + #313's `paper_reconcile` into `run_tick` + `paper run-all`; replaces
  #249's single-tenant path; records per-strategy equity. This is where the override lands.

`build_paper_sizing_snapshot` takes `allocation: float` as a parameter (the driver supplies it from
`active_allocation`), so #314 does **not** depend on Slice 1's `paper allocate` CLI (#288) — only on
#249's ledger.

## 3. Approach: generalize by `LedgerKind` (DRY)

The NAV/PnL/fail-closed accounting (~50 lines) is subtle; the only lane-specific parts are the
positions source and the fills table. So generalize, rather than duplicate — consistent with #313's
shared-core decision, and idiomatic to the codebase's existing `believed_positions` /
`paper_believed_positions` alias pattern.

### Modify `algua/execution/live_sizing.py`
- `build_sizing_snapshot(conn, strategy, allocation, bars, universe, *, kind: LedgerKind) -> tuple[SizingSnapshot, float]`
  — the generalized core. Positions via `believed_positions(conn, strategy, kind)`; the per-symbol
  PnL fill sequence read from the kind's fills table (via the accessor below); everything else
  (marks from the latest closed bar, `nav = allocation + realized + unrealized`,
  `equity = min(allocation, nav)`, fail-closed on a held symbol with no usable mark, non-positive
  equity fails closed) unchanged from today's live logic.
- `build_live_sizing_snapshot(conn, strategy, allocation, bars, universe)` → thin alias =
  `build_sizing_snapshot(..., kind=LedgerKind.LIVE)`. **The live caller (`live_cmd.py:139`) and
  `tests/test_live_sizing.py` stay untouched** — the strongest behavior-preserving guard.
- `build_paper_sizing_snapshot(conn, strategy, allocation, bars, universe)` → thin alias =
  `build_sizing_snapshot(..., kind=LedgerKind.PAPER)`.
- `SizingSnapshot` and `LiveSizingError` unchanged. The paper path raises the same `LiveSizingError`;
  a lane-agnostic rename is deferred to avoid churning the live lane (noted, not done here).

### Modify `algua/execution/live_ledger.py`
- Add `fills_table(kind: LedgerKind) -> str` returning `_TABLES[kind].fills`, so `live_sizing` reads
  the per-kind fills table without importing the private `_TABLES`. Additive.

## 4. Testing

- **`tests/test_paper_sizing.py`** (new) — mirror `test_live_sizing.py` over the paper ledger: seed
  `paper_venue_fills` + a strategy, `build_paper_sizing_snapshot` →
  - `nav == allocation + realized + unrealized` (PnL of the strategy's own fills marked to the latest
    closed bar);
  - `equity == min(allocation, nav)`;
  - correct `qtys` / `market_values` for held symbols (including a held-but-out-of-universe symbol);
  - **fail-closed** (`LiveSizingError`) when a held symbol has no usable mark;
  - non-positive resulting equity fails closed.
- **Parity test** — identical fills written under `LedgerKind.LIVE` (`live_fills`) vs
  `LedgerKind.PAPER` (`paper_venue_fills`) yield identical `SizingSnapshot` + `nav`, proving the
  generalization is faithful to the live original.
- **Live regression** — `tests/test_live_sizing.py` passes unchanged (the alias routes through the
  generalized core).
- TDD throughout; `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
  green before every commit.

## 5. Non-goals (this issue)

- No `run_tick`/`trade-tick`/CLI change; no `paper run-all` (→ #316).
- No `tick_snapshots` per-strategy-equity recording yet (a consequence of #316 wiring run_tick to the
  snapshot).
- No `paper_reconcile` use — that is #313's primitive, wired by #316.
- No forward-gate change (→ #315). No schema change (the ledger tables exist from #249).

## 6. Risk

- **Touches `live_sizing` / `live_ledger`** (load-bearing live code) via additive generalization. The
  live caller and `tests/test_live_sizing.py` are untouched (call the unchanged alias), so the live
  sizing path is behavior-preserving; the existing live sizing tests are the guard.
- Everything else is additive (a new alias, a new accessor, a new test file), unwired, so the blast
  radius on the running system is limited to the behavior-preserving generalization.
