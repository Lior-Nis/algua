# Plan: book circuit breaker + default-ON drawdown (#390)

Spec: `docs/superpowers/specs/2026-07-02-book-circuit-breaker-390.md`. Worktree off `origin/main`.
Quality gate between tasks: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`.

## Task 1 — pure book-breaker helper + tests
- `algua/risk/book_breaker.py`: `BookBreakerLimits`, `BookBreach`, `evaluate_book_breaker`.
- `tests/test_book_breaker.py`: valid/invalid limits; unusable equity; drawdown boundary
  (just under / just over / fresh ATH dd=0); daily-loss boundary; unusable baseline; None-clean.

## Task 2 — persisted account high-water mark + tests
- `algua/registry/db.py`: `book_equity_peak` table (id=1 single row).
- `algua/risk/book_equity.py`: `update_book_peak` (ratchet, reject bad equity), `get_book_peak`,
  `clear_book_peak`.
- `tests/test_book_equity.py`: ratchet up only; reject non-finite/<=0; clear; empty→None.

## Task 3 — broker `last_equity`
- `AccountState.last_equity: float = 0.0`; populate in `AlpacaLiveBroker.account()`.
- Extend an existing broker test (fake `/v2/account` payload) to assert `last_equity` flows.

## Task 4 — settings
- `strategy_max_drawdown_default = 0.25`, `book_max_drawdown = 0.15`, `book_max_daily_loss = 0.05`.

## Task 5 — Part A default-ON per-strategy drawdown
- live `run-all`, paper `trade-tick`, paper `run`: default `--max-drawdown` to the setting;
  add `--disable-drawdown-breaker` (audited when used).
- Update existing tests asserting the old `None` default; add default-on + disable tests.

## Task 6 — Part B wiring in live `run-all` + resume-all re-base
- After clean reconcile, before `_build_book_exposure`: validate equity, ratchet peak, evaluate,
  on breach engage global_halt + audit + `close_all_positions` + emit + exit 1.
- `live resume-all`: `clear_book_peak`.
- `tests/test_live_book_breaker.py`: breach → halt+close_all+exit1; unusable equity → halt;
  clean book → trades; resume-all clears peak.

## Task 7 — full gate, GATE-2 (Codex), PR.
