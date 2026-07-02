# Plan — Book-Level Aggregate Risk Limits (#389)

Spec: `docs/superpowers/specs/2026-07-02-book-level-risk-limits-389.md`

## Task 1 — pure `algua/risk/book_limits.py` + unit tests (DONE via subagent, TDD)
- `BookRiskLimits` frozen dataclass (validated conservative defaults).
- `BookExposure` accumulator: `permit_buy(symbol, requested)` → trimmed permitted notional,
  mutates the running book. Long-only precondition; min-of-four-headrooms.
- `tests/test_book_limits.py`: validation, each cap binding, sequential compounding across
  strategies, already-breached seed, fail-closed edges.
- Gate: `uv run pytest -q tests/test_book_limits.py && ruff + mypy` on the two files.

## Task 2 — settings caps (env-overridable, conservative defaults)
- `algua/config/settings.py`: add `book_max_gross=2.0`, `book_max_net=1.0`,
  `book_max_symbol_concentration=0.25`, `book_max_symbol_notional=0.50` (prefix `ALGUA_BOOK_*`).

## Task 3 — wire the book layer into live `run_all` (`algua/cli/live_cmd.py`)
- After clean reconcile, before the pool/reserve closures:
  - `net_positions = _broker_net_positions(broker)` (already computed for reconcile — reuse).
  - Build marks for the union of `net_positions` symbols + all verified strategy universes from a
    single bars fetch (`provider.get_bars`, latest closed bar per symbol). Reuse `_latest_marks`
    logic (import or replicate minimally).
  - Fail-closed guards → emit a note + skip trading this cycle (mirror reconcile-pending path):
    any nonzero net position with `qty < 0` (short → long-only precondition), or with a
    missing/non-finite/≤0 mark.
  - `book_notionals = {sym: qty * mark}` for nonzero positions; build `BookExposure(equity,
    book_notionals, BookRiskLimits(from settings))`. Equity = `_live_account_equity()` /
    `broker.account().equity` (account-level denominator, NOT a subaccount).
- `_reserve_for(name)`: compose book into the existing pool closure —
  `pool_permitted = min(notional, max(0, pool.available))`;
  `permitted = book.permit_buy(symbol, pool_permitted)`; `pool.available -= permitted`;
  audit shortfall (`permitted < notional`) via `record_reservation` (existing table; reason auto
  'trimmed'/'skipped'). Book mutates by the FINAL permitted (after pool trim).

## Task 4 — integration test for run_all wiring
- `tests/test_live_run_all_book_limits.py` (or extend existing live run-all test): a fake broker +
  two strategies both targeting the same name; assert the second strategy's buy is trimmed/skipped
  at the book level; assert a short account position fails the cycle closed; assert a missing mark
  fails closed.

## Gate (full, before push)
`uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`

## Touched files (all NON-CODEOWNERS-protected)
`algua/risk/book_limits.py` (new), `algua/config/settings.py`, `algua/cli/live_cmd.py`,
`tests/test_book_limits.py` (new), integration test (new). No store/lifecycle/engine/gates/
promotion/etc.
