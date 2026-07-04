# #450 — Live breach-flatten fail-safe on non-BrokerError: verification + regression guard

## Finding: the production defect is already fixed on `origin/main`

Issue #450 (filed 2026-07-02, `risk-safe-scaling`, high) reports that the live breach-flatten
path caught only `BrokerError` while the paper path caught `Exception`, so a non-broker exception
(`sqlite3.OperationalError` from a DB lock, a `ValueError` from `record_live_order`) raised during
real-money liquidation would propagate: the kill-switch is tripped but the position is left
un-flattened and (per the issue) a raw traceback escapes `@json_errors`. It cited the narrow
`except BrokerError as fexc` at `algua/cli/live_cmd.py:185`.

That code no longer exists. The finding was filed against a stale tree:

1. **#336 / PR#372 (`852f77c`, merged 2026-07-01 — one day BEFORE #450 was filed)** single-sourced
   the breach-flatten loop into `algua/execution/flatten.py::flatten_strategy`. That helper wraps
   `cancel()` + `ingest()` + the record→`submit_offset`→backfill offset loop in
   `except Exception` (flatten.py:145), capturing ANY exception into `FlattenResult.flatten_error`
   plus an audited `flatten_failed` row, and NEVER propagating. Both lanes (live per-strategy breach
   at `live_cmd.py:196`, paper per-strategy breach, paper `flatten` command) now call this one
   helper, so the lane divergence #450 describes is structurally impossible.
2. **The live book-breach flatten** (`close_all_positions`, `live_cmd.py:490`) is wrapped in
   `except Exception` and emits a structured `flatten_error` payload.
3. **#449 / PR#482 (`a316902`, current `origin/main` HEAD)** added the live `flatten` and
   `halt-all` emergency commands — the live `flatten` command also delegates to `flatten_strategy`
   (`live_cmd.py:628`), inheriting the same `except Exception` fail-safe.
4. **#337** made `@json_errors` a no-arg catch-all: even a hypothetical escaped exception becomes a
   structured `{ok:false,error,code}` envelope with exit 1 — no raw traceback. So the second half of
   the issue's harm (traceback escaping §3.5) is also mitigated.

The only remaining `except BrokerError` in `live_cmd.py` is the book-breaker account read
(`_evaluate_book_loss_breaker`, `live_cmd.py:295`), which is fail-**closed** by design (a BrokerError
returns a `BookBreach` that halts the account) — the opposite of the fail-open concern.

Existing regression coverage already pins the contract: `tests/test_flatten.py::
test_flatten_held_getter_failure_fails_safe` injects a non-BrokerError (`RuntimeError`) and asserts
it is captured into `flatten_error` (not propagated) with a `flatten_failed` audit row; the
command-level `test_cli_live.py::test_live_flatten_error_stays_tripped` and
`test_live_flatten_close_failure_stays_tripped` assert the kill-switch stays tripped on a flatten
error.

## Decision

**Recommend closing #450 as already-resolved (by #336/#372, #449/#482, #337).** Ship no production
change — adding more defensive catches would be dead/duplicate code.

Add ONE small, additive, defense-in-depth regression test that pins the EXACT scenario #450 cites
by name — a non-`BrokerError` (`sqlite3.OperationalError`, modelling the #387 DB-lock, and a
`ValueError`) raised at the offset-**submit** step (not just the `held`-getter path already covered)
— and asserts LIVE↔PAPER **parity** (both `LedgerKind` values fail safe identically). This locks the
specific fail-open-on-non-broker-error regression the issue warns about so a future refactor cannot
silently narrow the catch back. Test-only; no production files; no CODEOWNERS-protected paths.

## Design forks (resolved)

- **Fork: write defensive production code vs. verify-and-regression-test?** → verify-and-test. The
  defect is genuinely absent on `origin/main`; the honest deliverable is a regression guard + an
  issue-close recommendation, not redundant hardening.
- **Fork: where to add the test?** → `tests/test_flatten.py` (the helper's own unit-test file),
  driving `flatten_strategy` directly. This is where all three call sites converge, so one test
  guards every lane. A CLI-level test would be redundant with the existing
  `test_run_all_breach_liquidates_per_strategy` / `test_live_flatten_error_stays_tripped`.
- **Fork: also touch the paper `halt-all` narrow `except BrokerError` (paper_cmd.py:711)?** → No.
  It is paper (no real money), out of #450's live-real-money scope, and the #337 catch-all already
  prevents a traceback. Keep the PR tightly scoped to #450.
- **Fork: assert the raise site via `submit_offset` or via a monkeypatched DB write?** → via
  `submit_offset` raising the exception, with the `record`/`backfill` helpers monkeypatched to
  no-ops so the test isolates the `except Exception` wrapper behaviour and needs no registered
  strategy row for the PAPER kind. Keeps the fixture trivial while proving lane parity.

## Non-goals

- No change to `flatten_strategy`, `live_cmd.py`, `paper_cmd.py`, or any production module.
- No schema change.
- Not touching the paper `halt-all` narrow catch (deferred, out of scope).

## Quality gate

`uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
(full suite — a targeted run misses far-file breakage).
