# Spec: default-ON drawdown breaker + aggregate/daily-loss book circuit breaker (#390)

## Problem
Two fail-open defects on the real-money / venue paths:

1. **Per-strategy drawdown breaker defaults OFF.** `check_drawdown` (`algua/risk/limits.py`)
   no-ops when `max_drawdown is None`. Live `run-all` and paper `trade-tick`/`run` all default
   the `--max-drawdown` Typer option to `None`, so absent an explicit operator flag every tick
   runs with the drawdown breaker fully OFF.
2. **No book-level aggregate breaker.** Nothing halts the *whole shared live account* on
   aggregate drawdown or daily loss. `global_halt` engages only on reconcile drift. A correlated
   crash across N strategies has zero automatic halt at the book level. #389 added a book-level
   *exposure* cap (gross/net/concentration) but no *loss/drawdown* breaker.

Institutional/prop practice hard-codes a daily-loss circuit breaker (~2-5%) that liquidates and
halts the whole book, precisely because per-strategy stops are insufficient under correlated stress.

## Part A — default-ON per-strategy drawdown breaker (fail-closed)
- New setting `strategy_max_drawdown_default: float = 0.25` (env `ALGUA_STRATEGY_MAX_DRAWDOWN_DEFAULT`).
  Reuses the conservative 0.25 forward bound *value* but is a SEPARATE setting so no
  CODEOWNERS-protected file (`research/forward_gates.py`) is touched.
- Live `run-all`, paper `trade-tick`, paper `run`: `--max-drawdown` default flips from `None`
  to `strategy_max_drawdown_default`. The breaker is now ON by default.
- New `--disable-drawdown-breaker` boolean flag (default `False`) on those three commands. When
  set, `max_drawdown` is passed as `None` (the `DRAWDOWN_DISABLED` sentinel) AND an
  `audit_append(action="drawdown_breaker_disabled", ...)` row is written. Documented as a
  HUMAN-ONLY emergency relaxation — loud and audited.
- `check_drawdown`'s `None`-sentinel semantics are unchanged.

## Part B — aggregate / daily-loss book circuit breaker (LIVE only)
A pure helper + a persisted account high-water mark + a broker-supplied daily baseline, wired
into `live run-all` after a clean reconcile and BEFORE any strategy can order.

### Pure helper — `algua/risk/book_breaker.py` (no I/O, stdlib only)
- `BookBreakerLimits(max_drawdown, max_daily_loss)` — frozen; `__post_init__` validates both are
  finite and in `(0, 1]`.
- `BookBreach(kind, detail)` — frozen result marker.
- `evaluate_book_breaker(equity, peak, last_equity, limits) -> BookBreach | None`:
  - **fail closed** — `equity` non-finite or `<= 0` → `BookBreach("book_equity_unusable", ...)`.
  - **book drawdown** — `peak > 0` and `equity < peak * (1 - max_drawdown)` →
    `BookBreach("book_drawdown", ...)` with `dd = 1 - equity/peak`.
  - **book daily loss** — baseline is `last_equity` (the broker's PRIOR trading-session close, an
    exchange-session-correct start-of-day that captures overnight gaps). If `last_equity`
    non-finite or `<= 0` → `BookBreach("book_baseline_unusable", ...)` (fail closed — cannot
    establish a daily baseline). Else if `equity < last_equity * (1 - max_daily_loss)` →
    `BookBreach("book_daily_loss", ...)` with `loss = 1 - equity/last_equity`.
  - else `None`.

### Account high-water mark — table `book_equity_peak` + `algua/risk/book_equity.py`
- `CREATE TABLE IF NOT EXISTS book_equity_peak(id INTEGER PRIMARY KEY CHECK(id=1),
  peak REAL NOT NULL, updated_at TEXT NOT NULL)` in `algua/registry/db.py`.
- `update_book_peak(conn, equity) -> float` — ratchets up only; **rejects** non-finite/`<= 0`
  equity with `ValueError` (defense-in-depth so a bad read can't corrupt the peak).
- `get_book_peak(conn) -> float | None`, `clear_book_peak(conn)`.
- No daily-baseline table — the daily baseline is the broker's `last_equity`, not persisted.

### Broker — expose `last_equity`
- `AccountState` gains `last_equity: float = 0.0` (default keeps legacy constructions valid);
  `AlpacaLiveBroker.account()` populates it from the Alpaca `/v2/account` `last_equity` field.

### Wiring — `live run-all`
After a clean reconcile, before `_build_book_exposure` and the per-strategy tick loop, on ONE
`broker.account()` snapshot:
1. **Validate equity BEFORE mutating the peak** (GATE-1 correction): if `equity` non-finite or
   `<= 0`, do not touch the peak — `global_halt.engage(...)`, emit, exit 1.
2. Otherwise `peak = update_book_peak(conn, equity)` (ratchet includes the current cycle, so a
   fresh all-time high has `dd = 0`).
3. `breach = evaluate_book_breaker(equity, peak, acct.last_equity, BookBreakerLimits(...))`.
4. On any breach: `global_halt.engage(conn, reason=breach.detail, actor="system")` FIRST
   (fail-safe — persistent no-trade even if the close then errors), audit
   `book_circuit_breaker`, then `broker.close_all_positions()` (account-wide cancel-all +
   close-all — flattens orphan/dormant/unverified holdings too, matching paper `halt-all`).
   Emit `{"ok": False, "book_breach": ...}` and exit 1. A close error is surfaced with
   `liquidation_submitted: False` and still exits 1 (the halt persists).

### Settings
- `book_max_drawdown: float = 0.15` (env `ALGUA_BOOK_MAX_DRAWDOWN`).
- `book_max_daily_loss: float = 0.05` (env `ALGUA_BOOK_MAX_DAILY_LOSS`) — conservative end of the
  institutional 2-5% band.

### resume-all re-base
`live resume-all` (in `paper_cmd`, which already clears live NAV peaks + the global halt) also
`clear_book_peak(conn)` so the account re-bases its book high-water mark after a flatten-to-cash
(else the breaker re-trips against a stale peak). The daily baseline auto-re-bases next session
via the broker's `last_equity`.

## Why once-tripped can't double-trip
`run-all` already checks `global_halt.is_engaged` at the top and exits. Once the book breaker
engages the halt, every later cycle exits at that top check — no second flatten. Within a cycle
there is a single evaluation and a single `close_all_positions`.

## Scope / deferrals (documented)
- **Paper aggregate breaker** — paper has no single shared real-money account book worth a
  daily-loss halt; the aggregate breaker is LIVE-only (matches #389, which wired book exposure
  into live only). The Part-A per-strategy default-ON change DOES cover paper.
- **Per-factor / sector aggregate loss** — deferred.
- **Cross-process `run-all` lock** — pre-existing single-runner assumption (#389); out of scope.

## CODEOWNERS
Touched: `book_breaker.py` (new), `book_equity.py` (new), `registry/db.py`, `config/settings.py`,
`execution/alpaca_broker.py`, `cli/live_cmd.py`, `cli/paper_cmd.py`. NONE are CODEOWNERS-protected
→ auto-merge eligible on green CI.

## Review record
GATE-1 (Codex, adversarial, two rounds): round 1 raised CRITICAL (per-strategy flatten misses the
book → use `close_all_positions`), HIGH (UTC/first-observed daily baseline wrong → use broker
`last_equity`), HIGH (unusable equity must persist-halt, not soft-defer). Round 2 APPROVED after
folds, with one correction: validate equity before mutating the peak (folded into the wiring +
`update_book_peak` rejecting bad equity).

GATE-2 (Codex, on the production diff): two HIGHs folded — (1) a `BrokerError` reading/parsing the
account (missing/malformed `equity`/`last_equity`) fell through to a retryable JSON error instead of
persist-halting; now `_evaluate_book_loss_breaker` catches it and returns a
`book_account_read_failed` breach → halt+flatten. (2) `strategy_max_drawdown_default` (env-override)
was unvalidated, so a `nan`/`>1` value could silently disable the default-ON breaker; now
`resolve_drawdown_breaker` fails closed on a non-finite / out-of-(0,1] default. The book caps are
validated at consumption via `BookBreakerLimits.__post_init__`.
