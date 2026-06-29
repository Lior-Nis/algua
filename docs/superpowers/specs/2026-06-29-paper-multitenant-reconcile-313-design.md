# Multi-tenant attributed paper reconcile (#313) — design

**Status:** Draft (design approved in brainstorming; pending human review → implementation plan)
**Date:** 2026-06-29
**Issue:** #313 (epic #318 — multi-tenant attributed paper forward-testing, live-parity)
**Builds on:** #249 / PR#310 (the paper-venue fill ledger; kept, not reverted)

## 1. Problem

algua runs many strategies simultaneously — `live` does (shared account + `attributed_live_net` +
`live run-all`), but **paper forward-testing does not**: #249 made the paper reconcile single-tenant.
The forward gate requires `single_tenant_ok = n_siblings == 0` (`forward_promotion.py:278-283`) and
`run_tick` reconciles one strategy's belief against the **whole-account** snapshot
(`live_loop.py:167,179-190`). So ≥2 strategies on one paper account fail closed.

The epic (#318) restores multi-tenant attributed paper, overriding #249's single-tenant decision
where needed. **This issue (#313) is the foundation**: the account-level integrity primitive that a
multi-tenant driver needs. It is purely additive — it does **not** touch `run_tick` or the CLI, so
the existing single-tenant `trade-tick` keeps working unchanged until later issues wire the new path.

## 2. Scope & boundaries

#249 already attributes fills per-strategy (`paper_venue_fills.strategy`/`strategy_id`/
`broker_order_id`; `paper_believed_positions(conn, name)` returns one strategy's own positions). The
multi-tenant override is narrow and split across three issues:

- **#313 (this):** the account-level integrity primitives — `paper_account_expected_net`,
  `attributed_paper_net`, and a paper account `reconcile()` (grace-window + persisted state). Additive
  library code, unit-tested, driver-facing. No `run_tick`/CLI change.
- **#314:** `build_paper_sizing_snapshot` (attributed positions **+** virtual NAV) and the `run_tick`
  paper change. *This* is where #249's single-tenant in-tick comparison is replaced — because correct
  multi-tenant sizing needs per-strategy NAV as the denominator, which only #314 provides.
- **#316:** `paper run-all` calls #313's `reconcile` once per cycle, then ticks each strategy with
  #314's snapshot.

Splitting the `run_tick` change out of #313 avoids a broken intermediate (sizing off attributed
positions but whole-account equity). #313 alone changes no runtime behavior.

## 3. Approach: a shared reconcile core (DRY)

The grace-window + persisted-state reconcile algorithm is subtle (that is where bugs hide); the
per-lane net-sum queries are trivial. So extract the algorithm once and have both lanes delegate,
rather than duplicate it.

### New `algua/execution/reconcile_core.py` (lane-agnostic)
- `@dataclass(frozen=True) ReconcileResult` — `clean: bool`, `halt: bool`, `mismatches: list[dict]`
  (moved here from `live_reconcile`).
- `reconcile_account(conn, broker_net, expected, cycle, *, state_table, tolerance, grace_cycles) -> ReconcileResult`
  — the extracted algorithm. `expected: dict[str, float]` is supplied by the caller (its lane's
  expected-net). `state_table` parameterizes the per-symbol pending-state table (a controlled string
  constant, never user input — f-string interpolation is safe here). Same semantics as today's
  `live_reconcile.reconcile`: within tolerance clears the pending row; otherwise records/keeps a row
  keyed by `first_seen_cycle`; once it persists `grace_cycles` it becomes `unexplained` → `halt=True`.
  `clean` is True only when nothing mismatches.
- `next_cycle(conn, *, table) -> int` — the monotonic persisted cycle counter, table parameterized.

### `algua/execution/live_reconcile.py` (behavior-preserving refactor)
- Keep `account_expected_net`, `attributed_live_net`.
- `reconcile(conn, broker_net, cycle, tolerance=_TOLERANCE, grace_cycles=_GRACE_CYCLES)` →
  `reconcile_account(conn, broker_net, account_expected_net(conn), cycle,
  state_table="live_reconcile_state", tolerance=tolerance, grace_cycles=grace_cycles)`.
- `next_cycle(conn)` → `core.next_cycle(conn, table="live_cycle")`.
- The existing live reconcile tests are the regression guard (behavior must be unchanged).

### New `algua/execution/paper_reconcile.py`
- `paper_account_expected_net(conn) -> dict[str, float]` — Σ all `paper_venue_fills.qty` (signed) per
  symbol; zero nets omitted (paper analog of `account_expected_net`).
- `attributed_paper_net(conn) -> dict[str, float]` — Σ fills `JOIN strategies s ON s.name = f.strategy
  AND s.stage = 'paper'`; orphan (`strategy IS NULL`) and non-paper fills excluded, so they cannot
  "explain" a broker position (paper analog of `attributed_live_net`).
- `reconcile(conn, broker_net, cycle, tolerance=_TOLERANCE, grace_cycles=_GRACE_CYCLES)` →
  `reconcile_account(conn, broker_net, attributed_paper_net(conn), cycle,
  state_table="paper_reconcile_state", ...)`. **Uses `attributed_paper_net` (not the all-fills sum)**
  so a broker holding no current-paper strategy owns leaves a residual and fails closed — the
  multi-tenant safety semantics. The grace window absorbs transient skew (a just-ingested fill whose
  order's `broker_order_id` is not yet backfilled is briefly orphan/unattributed → `pending`, not
  `halt`, until attribution catches up within `grace_cycles`).
- `next_cycle(conn)` → `core.next_cycle(conn, table="paper_cycle")`.

`reconcile` takes `broker_net` as a parameter (the driver supplies it from the broker snapshot); #313
makes no broker calls and stays pure/testable.

## 4. Data model (SCHEMA_VERSION 30 → 31)

Two new tables in `algua/registry/db.py`, mirroring `live_reconcile_state` + `live_cycle`:

```sql
CREATE TABLE IF NOT EXISTS paper_reconcile_state (
    symbol           TEXT PRIMARY KEY,
    expected_qty     REAL NOT NULL,
    broker_qty       REAL NOT NULL,
    first_seen_cycle INTEGER NOT NULL,
    status           TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS paper_cycle (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    n  INTEGER NOT NULL
);
```

## 5. Testing

- **`tests/test_reconcile_core.py`** — clean match → `clean=True, halt=False`; mismatch within grace →
  `pending`, `halt=False`; mismatch persisting ≥ `grace_cycles` → `unexplained`, `halt=True`; a
  resolved mismatch clears its stale state row; `1e-6` tolerance absorbed. Parameterized over a
  throwaway `state_table`/`cycle` table to prove lane-agnosticism.
- **Live regression** — the existing `live_reconcile` tests must pass unchanged, proving the refactor
  is behavior-preserving.
- **`tests/test_paper_reconcile.py`** — `paper_account_expected_net` sums all fills; `attributed_paper_net`
  excludes an orphan (`strategy IS NULL`) fill and a non-`paper`-stage strategy's fills; `reconcile`
  fails closed (residual, eventually `unexplained`→halt) on an unattributable broker holding, passes
  when the attributed book explains the broker net, and grace-tolerates a not-yet-attributed fill.
- TDD throughout; `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
  green before every commit.

## 6. Non-goals (this issue)

- No `run_tick` change, no CLI command, no `paper run-all` (→ #314 / #316).
- No per-strategy NAV / sizing snapshot (→ #314).
- No forward-gate change — `single_tenant_ok` stays as-is until #315 (which depends on #313 + #314).
- The reconcile is wired by no caller yet; it is a tested primitive the multi-tenant driver will use.

## 7. Risk

- **Touches `live_reconcile`** (load-bearing live code) via the behavior-preserving extraction. The
  existing live reconcile tests are the guard; the refactor must not change live behavior.
- Everything else is additive (new modules, two new tables) and unwired, so the blast radius on the
  running system is limited to the live-reconcile refactor.
