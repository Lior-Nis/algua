# Mark-freshness risk guard (#452) — design

## Problem
Every live/paper risk wall (drawdown breaker, realized-gross check, NAV, and the sizing
denominator) is computed from bar marks: `run_tick` fetches `provider.get_bars(universe, start,
end)`, drops today's partial session via the `ts.date() < now.date()` cutoff, then values the book
off `bars.groupby('symbol')['close'].last()` (`_latest_marks`). Nothing checks that those marks are
RECENT.

`live run-all` and `paper trade-tick` default `--start 2023-01-01 --end 2023-12-31`. Invoked
without overriding `--end`, every mark, NAV, drawdown and gross-exposure figure is computed from
2023-12-29 closes. A strategy down 80% in reality shows no drawdown against a stale 2023 peak; the
breaker never trips; the loop sizes and submits real-money orders against a 2.5-year-stale world.
`grep stale|freshness|days_since` over `algua/` returns zero matches. (BCBS 239 Principle 14 —
Timeliness.)

A second, sharper variant: a **dead feed**. If the provider returns an EMPTY frame (or a frame with
no bar for a held name) while the book is still holding positions, `run_tick` currently returns a
no-op `TickResult` early — it never establishes the risk state at all, and no wall runs. A held book
whose feed has gone dark is exactly the case the guard must fail closed on.

A third variant, structural rather than about mark VALUE: **warm-up suppresses the breakers for a
held book (CRITICAL).** `run_tick` returns a no-op `TickResult` the moment
`bars.index.nunique() <= warmup_bars`, BEFORE it builds the sizing snapshot, runs `check_drawdown`,
or runs the realized-gross check. Warm-up is a *decision* gate — it exists to withhold NEW orders
until the strategy has enough history to rank — but a strategy can be re-auditioned or re-listed
while still HOLDING an inherited book. In that window the marks can be perfectly fresh yet the
drawdown breaker never runs, so a held book down 80% is never valued or tripped simply because the
strategy is short of warm-up bars. Risk VALUATION of a held book must never be gated on warm-up;
only new-order emission may be.

A fourth variant, at the **account/book level**: `run-all`'s `_build_book_exposure` (#389) values
the whole reconciled account book off `provider.get_bars(...)` marks, and today only rejects a mark
that is missing / non-finite / `<= 0` — with NO staleness bound, and only by DEFERRING the cycle (a
benign skip), not by halting. A stale `--end` (or a single dead account name) therefore values the
entire account book off stale closes and, at worst, silently defers rather than treating an
unvaluable book as the systemic data-integrity failure it is.

## Chosen design — layered, defense in depth
Layer A (per-strategy `run_tick` wall + held-book valuation), Layer A2 (account-level
`_build_book_exposure` wall), and Layer B (remove the stale `--start`/`--end` default footgun).

### Layer A — PER-SYMBOL mark usability wall inside the shared `run_tick` (authoritative)
`run_tick` is the ONE wall-clock tick for BOTH the live and the paper-broker lanes, and it is where
every risk wall executes. The marks are consumed **per symbol**, so a global `bars.index.max()` is
NOT a sound freshness proxy: one fresh symbol pushes the global max forward while a different,
genuinely dead symbol is valued off a stale close. The wall therefore reasons about usability **per
consumed symbol** and fails closed if ANY consumed mark is missing, stale, unvaluable, or
future-dated — not just when the
freshest is stale.

**A mark is UNUSABLE if it is any of:**

- **absent** — the symbol has no bar at all in the (post-cutoff) frame (`no_mark`). A partial frame
  that silently drops a symbol we are valuing/deciding on is a data-integrity failure, NOT a symbol
  to skip over. (finding 3)
- **stale** — its latest bar is more than `MAX_STALE_SESSIONS` completed exchange sessions behind
  `now`.
- **unvaluable** — its latest bar's close is present but NOT a positive finite number, i.e.
  `not (math.isfinite(close) and close > 0.0)`. This rejects `<= 0.0` AND `+inf`/`NaN` — a `+inf`
  mark passes a bare `close > 0.0` test yet poisons NAV / drawdown / gross exposure just as badly as
  a stale mark, so the wall must reject it (it is STRICTER than `build_sizing_snapshot`'s legacy
  `mark <= 0.0` rule, which let `+inf` through).
- **future-dated** — its latest bar maps to a session AFTER `now`'s session (negative staleness).
  A bar "ahead" of the clock is bad data or a skewed clock, not fresh data; it fails closed rather
  than being clamped to "fresh". (finding 4)

Any unusable consumed mark ⇒ the risk state cannot be established off bars ⇒ **HALT-WITHOUT-FLATTEN**
(see policy below — a dark BAR feed halts the fleet but does not force-liquidate; only a
trustworthy-mark economic breach flattens). This is `RiskBreach`, not a silent skip.

**Three placements — the early-return paths must fail closed AND value a held book (findings 1 + CRITICAL).**

The old draft ran the wall ONLY after the sizing snapshot, so both early-return paths (empty frame,
warm-up not met) returned a no-op `TickResult` *before* the wall — or any risk valuation — ever ran.
That left two threats untouched: the dead-feed (zero recent bars while the book is held), and the
warm-up suppression of the breakers for a held book (CRITICAL). The wall AND the risk valuation are
therefore split so a held book is both freshness-checked AND valued BEFORE any early return:

1. **HELD-book gate — before any early return, before sizing.** Immediately after the cutoff is
   applied, compute the held set from the SAME ledger source the sizing snapshot uses
   (`hooks.live_positions()`, else broker positions — via `_early_positions`). If the book holds
   ANY position, every held symbol's mark must be usable (present, valuable, fresh, not
   future-dated). An empty/dead frame makes every held mark `no_mark` ⇒ `RiskBreach`. This runs
   before the empty-bars return AND before the warm-up return, so a held book can never slip out on
   an early-return path with an unpriced/stale mark. If the book is FLAT (no held symbols), the
   early returns proceed unchanged — nothing is at risk, nothing to value.

2. **HELD-book risk VALUATION — ahead of the warm-up early return whenever the book is held
   (CRITICAL).** Warm-up is a DECISION gate, not a risk gate: it may suppress `decide()` and new
   orders, but it must NEVER suppress valuation of a book that is already held. So the sizing
   snapshot + non-positive-equity guard + `check_drawdown` + realized-gross check run whenever
   `held or not warming` — i.e. for EVERY held book, warming or not, and for every past-warm-up
   book. A held-but-warming strategy therefore still values its book off (freshness-validated)
   marks, ratchets its drawdown peak, and TRIPS the drawdown / realized-gross breakers exactly as a
   past-warm-up tick would; it merely skips `decide()`/submit. Only a FLAT warming book takes the
   unchanged no-op early return (nothing held ⇒ nothing to value). The drawdown state computed on a
   held-warming tick is returned (equity + peak + realized_gross + positions_before) so the peak is
   persisted and the tick snapshot recorded, keeping the breaker basis continuous across warm-up.

3. **CONSUMED-set gate — on the DECISION path only (past warm-up), after the sizing snapshot, before
   `decide()` / submit.** Once the strategy is actually deciding, the consumed set widens to the
   **valued book ∪ decision universe** and the same usability check runs over all of it, so no
   stale/absent-priced ranked target or order ever reaches the venue. It runs ONLY on the decision
   path: a held-warming book does not rank its universe, so its not-yet-held candidate symbols are
   not "consumed" and are not freshness-checked (only the held book is).

**Which symbols are "consumed" on the decision path.** The union of

- the **valued book** — every symbol the sizing snapshot holds (`snap.qtys[s] != 0`); this is both
  the ledger-held set (NAV / drawdown / sizing denominator) and, since `positions_before` and
  `current_weights` are derived from the same `snap`, the book-held set behind the realized-gross
  check; and
- the **decision universe** — `strategy.universe`, every target/candidate symbol `decide()` ranks
  and values off `universe_bars.loc[:t]`. A universe symbol with NO bar in the frame is a `no_mark` offender
  and trips the wall (finding 3) — `decide()` ranking a candidate off a silently-missing series is
  the same unknowable-input failure as valuing a held name off a stale close; the recent-window
  default (Layer B) is what keeps this from tripping on ordinary operation.

```python
# --- inside run_tick ---------------------------------------------------------------
if timeframe != "1d":                      # daily-bar session semantics only (see "Timeframe")
    raise ValueError(f"mark-freshness wall supports only 1d bars; got {timeframe!r}")
now = now or datetime.now(UTC)

# Discover the held book BEFORE the fetch (ledger/broker positions need no bars), then fetch marks
# for the UNIVERSE ∪ HELD set — so a held INHERITED / out-of-universe symbol is actually requested
# and gets a real mark, instead of falsely reading as `no_mark` merely because it was never fetched
# (Round-2c GATE-1). The valuation reads this union frame; the DECISION reads a universe-only view.
held_qtys = _early_positions(hooks, broker)
held = {s for s, q in held_qtys.items() if q != 0.0}
bars = provider.get_bars(sorted(set(strategy.universe) | held), start, end, timeframe).sort_index()
if not bars.empty:
    bars = bars[[ts.date() < now.date() for ts in bars.index]]   # drop today's partial session

latest_ts = _latest_bar_ts(bars)     # {symbol: latest kept bar ts}; from the UNION frame; {} when empty
latest_close = _latest_marks(bars)   # {symbol: latest kept close};  from the UNION frame; {} when empty

# (1) HELD-book gate — BEFORE the empty-bars / warm-up early returns and BEFORE sizing.
if held:
    assert_marks_usable(held, latest_ts, latest_close, now)   # RiskBreach(stale/unvaluable) -> HALT-only

if bars.empty:                          # nothing fetched at all; FLAT here (held+empty already tripped)
    return TickResult(None, {}, held_qtys, [])

# Decision TIMING + warm-up come from the UNIVERSE-restricted history ONLY (Round-2d): a held
# out-of-universe symbol's longer/independent history must NOT make the universe look warmed or move
# the decision timestamp. Valuation (below) still reads the full union frame.
universe_bars = bars[bars["symbol"].isin(strategy.universe)]
t = universe_bars.index.max() if not universe_bars.empty else None
warming = universe_bars.index.nunique() <= strategy.execution.warmup_bars   # empty universe -> warming

# (2) HELD-book risk VALUATION — runs whenever the book is held, warm-up or not (CRITICAL).
valued = held or not warming
if valued:
    snap, drawdown_equity = ...        # build_sizing_snapshot (held marks already validated above)
    if not (snap.equity > 0.0): raise RiskBreach("non_positive_equity", ...)
    peak = drawdown_equity if hooks.peak_equity is None else max(hooks.peak_equity, drawdown_equity)
    check_drawdown(drawdown_equity, peak, max_drawdown)         # trustworthy marks -> flatten on trip
    positions_before = {s: q for s, q in snap.qtys.items() if q != 0.0}
    current_weights  = {s: mv / snap.equity for s, mv in snap.market_values.items() if mv != 0.0}
    ... reconcile (if venue_belief) ...
    realized_gross = sum(abs(w) for w in current_weights.values())
    check_gross_exposure_realized(realized_gross, strategy.execution.max_gross_exposure)

if warming:
    if valued:   # held-but-warming: breakers already ran; persist the drawdown state, no decide/submit
        return TickResult(t, {}, positions_before, [], equity=drawdown_equity,
                          peak_equity=peak, reconcile_ok=reconcile_ok, realized_gross=realized_gross)
    return TickResult(t, {}, held_qtys, [])   # FLAT + warming: unchanged no-op

# (3) CONSUMED-set gate — decision path only (past warm-up), after sizing, before decide/submit.
consumed = {s for s, q in snap.qtys.items() if q != 0.0} | set(strategy.universe)
assert_marks_usable(consumed, latest_ts, latest_close, now)       # RiskBreach(stale/unvaluable) -> HALT-only
# DECISION view is the universe-only history up to t — the widened fetch (held out-of-universe marks)
# NEVER reaches target_weights(); decide() sees exactly the universe bars it saw before this change.
weights, intents = decide(strategy, universe_bars.loc[:t], current_weights, t)
...
```

**Fetch scope — `universe ∪ held`, valued on the union, decided on the universe (Round-2c GATE-1).**
The held set is the ledger/broker position book, which can include an INHERITED or now-out-of-universe
symbol (a re-auditioned strategy, or a name dropped from the universe while still held — the exact
"held book while warming/re-listed" case the CRITICAL finding is about). If `run_tick` fetched only
`strategy.universe`, such a held symbol would carry NO bar and the HELD-book gate would read it as
`no_mark` — falsely classifying a *fetch-scope* omission as a *dark feed* and routing a trustworthy,
valuable held book to halt-without-flatten instead of valuing it and letting the drawdown breaker
flatten on a real loss. So the fetch is widened to `sorted(set(strategy.universe) | held)`, and:
- the **valuation** (held-book gate, `build_sizing_snapshot`, drawdown, realized-gross, consumed gate)
  reads the UNION frame — every held name has a real mark, so a genuinely dark feed still trips
  `no_mark` while a merely-out-of-universe held name is valued correctly; and
- the **decision** — its timing AND view are UNIVERSE-ONLY. Both the warm-up count and the decision
  timestamp `t` are derived from `universe_bars` (the union frame filtered to `strategy.universe`),
  NOT the union frame, so a held out-of-universe symbol's longer/independent history can neither make
  the universe look warmed nor move `t` (Round-2d); `decide()` then ranks over `universe_bars.loc[:t]`.
  So the widened fetch can never change what `target_weights()` sees or when it runs — identical to
  pre-change behavior. (`build_sizing_snapshot` already unions `universe | held` internally, so it
  consumes the extra marks natively for valuation.)

`assert_marks_usable` (exported without a leading underscore so `_build_book_exposure` reuses the
SAME wall — HIGH #2) builds the per-symbol staleness map (mapping absent → `math.inf`, and wrapping
the calendar call so a mapping error fails closed — see "Freshness math"), splits out the
present-but-unvaluable marks, and delegates to the pure `check_mark_freshness`:

```python
def assert_marks_usable(symbols, latest_ts, latest_close, now) -> None:
    cal = MarketCalendar()
    # A present mark must be a POSITIVE FINITE number: `not (isfinite and > 0)` rejects <=0 AND
    # +inf / NaN (a bare `> 0.0` would let +inf through to poison NAV/drawdown/gross exposure).
    unvaluable = {
        s for s in symbols
        if s in latest_close and not (math.isfinite(latest_close[s]) and latest_close[s] > 0.0)
    }
    if unvaluable:
        raise RiskBreach(
            "unvaluable_marks",
            f"held/consumed symbols have a non-positive / non-finite mark: {sorted(unvaluable)} — "
            f"refusing to value/size the book off an unvaluable feed",
        )
    stale_by_symbol: dict[str, float] = {}
    for s in symbols:
        ts = latest_ts.get(s)
        if ts is None:
            stale_by_symbol[s] = math.inf          # no bar at all -> no_mark offender (finding 3)
            continue
        try:
            stale_by_symbol[s] = float(cal.sessions_stale(ts, now))
        except Exception as exc:                   # MinuteOutOfBounds / unmappable ts (finding 5)
            raise RiskBreach(
                "stale_marks",
                f"cannot map {s} mark {ts} to an exchange session ({exc!r}) — "
                f"refusing to establish risk state off an unmappable timestamp",
            ) from exc
    check_mark_freshness(stale_by_symbol, MAX_STALE_SESSIONS)   # raises RiskBreach("stale_marks")
```

**The latest mark + timestamp must be read ATOMICALLY from the same row, nulls preserved (Round-2b
GATE-1 finding).** The legacy `_latest_marks` used `bars.groupby('symbol')['close'].last()`, and
pandas `GroupBy.last()` **skips NaN** — so a symbol whose NEWEST bar has a `NaN` close but an older
bar has a finite close would return the *older* finite close paired with the *newest* timestamp,
masking the NaN-latest and slipping past the `isfinite and > 0` wall (finding 2 would be reopened for
exactly the NaN-in-latest-bar case). Both `_latest_marks` and `_latest_bar_ts` therefore select each
symbol's **latest row by timestamp WITHOUT dropping nulls** — e.g. on the `sort_index()`-ed frame,
`bars.groupby('symbol').tail(1)` (last row per symbol, all columns, NaN preserved) → map
`symbol -> (latest_ts, latest_close)` from the SAME row. A `NaN`/`+inf` latest close is then kept and
correctly trips `unvaluable_marks`; the paired `latest_ts` is the genuine newest bar's timestamp so
staleness is measured against the real latest bar, not an older non-null one. (`build_sizing_snapshot`
and `_build_book_exposure` are updated to the same null-preserving latest-row selection so the wall
and the valuation agree on the mark.)

`check_mark_freshness` (in `algua/risk/limits.py`) takes the **per-symbol staleness mapping** as a
`dict[str, float]` (so `math.inf` cleanly encodes `no_mark` and negatives encode future-dated),
classifies every offender with a distinct reason, and reports them all:

```python
def check_mark_freshness(stale_by_symbol: dict[str, float], max_stale: int) -> None:
    offenders: dict[str, str] = {}
    for s, n in stale_by_symbol.items():
        if math.isinf(n):
            offenders[s] = "no_mark"                 # absent from the frame (finding 3)
        elif n < 0:
            offenders[s] = f"future_dated({n:.0f})"  # bar ahead of now -> clock/data bad (finding 4)
        elif n > max_stale:
            offenders[s] = f"stale({n:.0f})"
    if offenders:
        raise RiskBreach(
            "stale_marks",
            f"marks unusable beyond {max_stale} completed sessions: {offenders} — "
            f"refusing to value/size/decide off an unreliable feed",
        )
```

### Policy — dark BAR feed ⇒ HALT-WITHOUT-FLATTEN; only an economic/position breach flattens (finding HIGH #3)
An absent/stale/unvaluable/future-dated mark means the current price of a consumed name is UNKNOWN,
so NAV, drawdown and gross exposure are unknowable — the risk state cannot be established off bars.
The old draft's response was "trip + flatten." That is the WRONG proportion for the actual failure:
the offender is the **BAR feed**, and a stale bar feed is almost always a transient vendor delay
while the **broker is alive** (indeed `flatten_strategy` proves the broker is reachable — it submits
market orders). **Force-liquidating an entire book at market on a data outage is self-harm** — it
realizes costs/slippage and abandons positions to solve a *data* problem, not a *position* problem.
Holding still and stopping trading already removes the incremental risk (no new orders), and the
broker-truth defenses below still guard a real economic crash. So the policy splits by breach nature:

- **Data-integrity breach — a dark BAR feed** (`RiskBreach.kind ∈ {"stale_marks",
  "unvaluable_marks"}`): **HALT-WITHOUT-FLATTEN.** The CLI `except RiskBreach` handler special-cases
  these two kinds: it **engages the global halt** (`global_halt.engage`, stopping the WHOLE loop for
  human review — a dark feed is systemic since all strategies share one provider), trips the
  per-strategy kill-switch (`trip_for_breach`, so it stays halted), audits, and returns a halt
  marker with `liquidation_submitted=False` — it **does NOT call `flatten_strategy`**. Positions are
  preserved; the broker-truth book-loss breaker (`_evaluate_book_loss_breaker`, #390) — which reads
  `broker.account().equity`, NOT bars — continues to catch and flatten a REAL account drawdown
  independently of the dark bar feed. (When the broker itself is ALSO dark, flatten would fail
  anyway, so halt-only is strictly safe either way.)
- **Economic / position breach on TRUSTWORTHY marks** (`kind ∈ {"drawdown",
  "gross_exposure_realized", "reconcile", "non_positive_equity", ...}`): **trip + scoped flatten**,
  unchanged. These fire only AFTER the freshness gate has already confirmed the marks are usable, so
  reducing real exposure is the correct response and the flatten prices are trustworthy. This is the
  **explicit flatten condition** the finding asked for: flatten iff the breach kind is a
  trustworthy-mark economic/position breach, i.e. NOT a data-integrity kind. **`non_positive_equity`
  is in this set uniformly** — a wiped book (`NAV <= 0`) off trustworthy marks is a real economic
  wipe-out, so it flattens + halts, NOT a silent skip (see reconciliation below).

This split is a POLICY decision, verified by dedicated tests: (a) a stale-feed / dead-feed held-book
tick engages the global halt AND does NOT flatten (assert the halt is set and `flatten_strategy` is
NOT invoked / `liquidation_submitted=False`); (b) a genuine drawdown breach on a held (incl.
held-warming) book DOES trip + flatten (assert the flatten submission). It applies identically in
both `live_cmd._run_strategy_tick` and `paper_cmd._run_paper_strategy_tick`.

#### Reconciling the `LiveSizingError` skip path with the policy (finding 2)
`build_sizing_snapshot` raises `LiveSizingError` (a SKIP, not a flatten) when a held symbol has no
usable mark. Running the **HELD-book gate BEFORE `build_sizing_snapshot`** (placement 1) converts
every missing / stale / unvaluable / future-dated HELD mark into a `RiskBreach("stale_marks" |
"unvaluable_marks")` — which now routes to **HALT-without-flatten** (above), not a skip and not a
flatten. Consequently:

- `build_sizing_snapshot`'s `LiveSizingError`-for-a-held-mark branch becomes **unreachable on the
  wall-clock path** (the gate has already tripped) — retained as pure defense-in-depth for any
  non-wall caller.
- **Non-positive NAV is UNIFIED to a flatten, not a silent skip (Round-2 GATE-1 finding 1).** The old
  draft kept `NAV <= 0` as a `LiveSizingError` SKIP — but a skip neither halts nor flattens, so a
  wiped book that still holds positions would be skipped EVERY tick forever, un-halted and
  un-flattened. That is fixed by removing `build_sizing_snapshot`'s `equity <= 0.0`
  `LiveSizingError` raise and **returning the (non-positive-equity) snapshot**, so `run_tick`'s
  EXISTING `if not (snap.equity > 0.0): raise RiskBreach("non_positive_equity")` guard fires
  **uniformly for BOTH snapshot sources** (ledger AND `broker.snapshot`) — before any division /
  sign-flip — and routes to the trip + flatten economic-breach handler. (This is safe: the only
  callers of `build_sizing_snapshot` are the two wall-path CLI hooks, both feeding `run_tick`, whose
  guard sits ahead of every division; no non-wall caller relies on the raise.) A wiped book therefore
  flattens + trips the kill-switch, closing the skip-forever hole. `NAV <= 0` marks are trustworthy
  (the freshness/finite/positive wall already passed), so this is a genuine economic wipe, correctly
  in the flatten set — NOT a data-integrity halt.
- Consequently `LiveSizingError` is **fully unreachable on the wall path** (held-mark branch
  pre-empted by the gate; NAV branch removed). It is retained only as non-wall defense-in-depth.
- **`live_cmd`** keeps its `except LiveSizingError` purely as belt-and-suspenders (any residual
  non-wall `LiveSizingError` → skip, never a generic-exception escape); its comment is updated to
  "unreachable on the wall path — mark problems HALT, non-positive NAV FLATTENs; retained as
  defense-in-depth".
- **`paper_cmd._run_paper_strategy_tick` has NO `except LiveSizingError` today**, so historically a
  sizing error escaped to the generic `except Exception` (`cycle_failed`) → `@json_errors`
  `sizing_error` envelope (finding 2, second half). With NAV<=0 now a `RiskBreach`, paper's EXISTING
  `except RiskBreach` already handles it (trip + flatten) — but a matching
  `except LiveSizingError -> paper_sizing_skipped` (audit + `ok({"strategy": name, "traded": False,
  "skipped": str(exc)})`) is STILL ADDED as defense-in-depth so no residual `LiveSizingError` can
  ever escape as an unhandled generic exception (finding 2 stays resolved).

### Layer A2 — account-level book-exposure freshness wall (finding HIGH #2)
`run-all`'s `_build_book_exposure` (#389) values the WHOLE reconciled account book off
`provider.get_bars(...)` marks BEFORE the per-strategy loop runs, and today only rejects a mark that
is missing / non-finite / `<= 0`, and only by DEFERRING the cycle. The same per-symbol usability wall
Layer A applies to a strategy's held book must apply here to the account's `net_positions` marks:

- After fetching the book marks, run the **shared `assert_marks_usable`** helper over the reconciled
  nonzero `net_positions` symbols (every one is a genuinely account-held name, so `no_mark` =
  data-integrity failure, never a benign skip). Absent / stale (`> MAX_STALE_SESSIONS`) / unvaluable
  (`<= 0` / non-finite) / future-dated marks all raise `RiskBreach("stale_marks" |
  "unvaluable_marks")` — reusing the exact staleness math and thresholds as Layer A (single-sourced,
  not a second copy). This SUBSUMES the current bespoke `mark is None / not finite / <= 0` check.
- **An unusable account-held mark ⇒ engage the GLOBAL HALT (halt-only, no account-wide flatten).**
  The `run-all` caller wraps `_build_book_exposure` so a propagated `RiskBreach` engages the global
  halt (`global_halt.engage`), audits (`book_stale_marks_halt`), emits an `ok: False` book-breach
  envelope with `global_halt: "set"`, `liquidation_submitted: False`, and exits non-zero — it does
  NOT call `broker.close_all_positions()`. This mirrors the per-strategy Layer A policy (HIGH #3):
  the offender is the BAR feed while the broker is alive, and the broker-truth book-loss breaker
  (`_evaluate_book_loss_breaker`, which runs EARLIER in the same cycle off `broker.account().equity`)
  already halts+flattens the account on a REAL drawdown. Auto-liquidating the whole account on a
  transient bar-feed outage would be disproportionate self-harm; halting the fleet for human review
  while preserving positions is the proportionate response.
- The benign DEFERs that remain `(None, reason)` (short-position precondition; already-breaches-cap
  seed) are unchanged — those are policy/economic states, not data-integrity failures, and are out of
  scope for this wall.
- `_build_book_exposure` gains an injectable `now` (defaulting to `datetime.now(UTC)`) so the
  staleness math is testable, exactly like `run_tick`.

Tests: a stale / absent / future-dated `net_positions` mark engages the global halt AND does NOT call
`close_all_positions` (halt-only); a fresh account book builds the `BookExposure` unchanged.

### Timeframe — daily only
The freshness math maps a bar by `latest_bar.date()` and reasons in exchange **sessions**, which is
daily-bar semantics. The bar contract permits intraday timeframes, so the wall **fails closed
unless `timeframe == "1d"`** — a plain `raise ValueError` (NOT a bare `assert`, which `-O` strips)
at the top of `run_tick`, before the fetch, so the check governs the held-book gate too. `run_tick`'s
only wall-clock callers pass `timeframe="1d"` today; intraday freshness (a minutes/hours bound) is an
explicit deferred scope, not silently mis-handled. (A timeframe mismatch is a static misconfiguration,
not a runtime data failure, so it fails the tick closed without flattening — no trade, no venue call.)

`MAX_STALE_SESSIONS = 2` — a module constant in `algua/risk/limits.py`, un-relaxable (NOT a CLI flag
or settings field, so it can't be tuned into the footgun it defends against).

### Threshold justification — 2 completed sessions
Staleness is counted in **completed exchange sessions** by `sessions_between`, which maps each
endpoint through `session_on_or_before` and therefore **skips weekends and holidays entirely**. The
old "5 sessions to clear a weekend/holiday cluster" rationale was invalid on two counts: calendar
gaps never inflate a session count in the first place, and a bound of 5 would fail to trip on a full
DEAD trading week (5 missed sessions ≤ 5). Grounding the bound in actual session counts:

- **Normal live tick** — today's partial bar is dropped by the cutoff, so the freshest kept bar is
  the *prior* session ⇒ staleness **1**.
- **Weekend / holiday invocation** — the freshest kept bar is the last session, and `now` maps
  (in exchange time) to that same session with no intervening SESSIONS ⇒ staleness **0**. (Post-a-
  3-day-weekend the holiday contributes no session, so it is still 1, not 3 — precisely why the
  bound can be tight.)
- **Bound = 2** passes normal operation (1) and tolerates ONE missed / not-yet-published session
  (e.g. a single-session vendor delay), while a SECOND consecutive missing session (staleness 3)
  trips.
- Because the metric counts sessions, not calendar days, a genuinely dead feed trips within two
  further sessions regardless of weekends/holidays: a full dead trading week trips on its **3rd**
  session — the exact case the old bound of 5 silently tolerated.

Verified by an **exact-boundary** test: `sessions_stale == 2` passes, `== 3` raises; plus the
normal-op (staleness 1 passes), weekend (staleness 0 passes), future-dated (negative ⇒ fails
closed), and the post-close-before-UTC-midnight boundary cases below.

### Layer B — remove the stale `--start`/`--end` default footgun (required, not polish)
With Layer A live, the frozen 2023 defaults would make the DEFAULT wall-clock invocation trip
`stale_marks` on every strategy and flatten the whole book. So the wall-clock defaults MUST become a
recent rolling window. Because a Typer `Option` default must be a static value, `--start`/`--end`
become **sentinel `None`** options resolved at call time: default `--end` → today (UTC, ISO date),
default `--start` → `today − LOOKBACK_DAYS` (400 days — ~275 sessions, covering typical warm-ups
with slack; operators needing more pass `--start`). An explicitly-passed `--start`/`--end` is
honoured unchanged. (Layer B is unchanged from the reviewed draft — Codex confirmed the entry-point
table is complete and the sentinel-`None` resolution correctly defuses the frozen-2023 footgun.)

**Complete set of wall-clock venue entry points (all get the recent-window default):**

| entry point | file:line | kind |
|---|---|---|
| `live run-all` | `live_cmd.py:367-368` | `@command` (real user-facing default) |
| `live _run_strategy_tick` | `live_cmd.py:137` | internal helper (called by `run-all`) |
| `paper trade-tick` | `paper_cmd.py:461-462` | `@command` (real user-facing default) |
| `paper _run_paper_strategy_tick` | `paper_cmd.py:357` | internal helper (called by `trade-tick`) |

`paper trade-tick` was MISSED by the first draft of this design — it is the direct single-strategy
wall-clock paper-venue command and defaulted to the same 2023 window. **There is no `paper run-all`**
(an earlier draft referenced one that does not exist); `_run_paper_strategy_tick` is invoked only by
`trade-tick`. **There is no direct `live trade-tick` command** either — live's only wall-clock command
surface is `run-all` (which drives `_run_strategy_tick`). The two internal helpers already receive
`start=`/`end=` explicitly from their callers, so their own defaults are never the live path; they
are updated anyway so no latent 2023 default remains.

Everything else is a **historical replay / research** surface with NO wall-clock freshness wall, and
is deliberately LEFT on the frozen 2023 window:

- `paper run` (`paper_cmd.py:217-218`) — SimBroker REPLAY over historical bars, not a wall-clock
  venue tick (`run_paper` never calls `run_tick`).
- `backtest_cmd`, `research_cmd`, `factor_cmd`, `monitoring_cmd`, `shadow_cmd` — backtest / walk-
  forward / factor-eval / advisory-monitoring / shadow-replay over historical ranges.

A CLI test asserts that invoking `paper trade-tick` (and `live run-all`) WITHOUT `--end` resolves
the effective `end` to today, and WITHOUT `--start` resolves `start` to `today − 400d`; and that
`paper run` still defaults to the 2023 replay window.

## Freshness math
Daily bars are stamped at the session's UTC midnight (#262). `session_of_instant` on a UTC-midnight
bar would misattribute it to the PRIOR session (UTC midnight = prior evening ET), so map the BAR by
its session DATE (via `session_on_or_before`, applied inside `sessions_between`) and `now` in
exchange time:

```python
def sessions_stale(self, latest_bar: datetime, now: datetime) -> int:
    """Completed exchange sessions from the bar's session to now's session, in EXCHANGE time.
    NOT clamped: a NEGATIVE result (the bar maps to a session AFTER now — clock skew or bad
    data) is returned as-is so the caller fails closed rather than reading a future bar as
    fresh (finding 4). `session_of_instant` may raise MinuteOutOfBounds for an out-of-range
    instant — propagated for the caller to convert into a fail-closed RiskBreach (finding 5)."""
    return self.sessions_between(latest_bar.date(), self.session_of_instant(now))
```

`sessions_between` maps each endpoint via `session_on_or_before` (idempotent on a session date), so
`sessions_stale` returns 0 for the freshest closed session and grows one per completed session as a
feed goes stale, and returns a NEGATIVE value when the bar is ahead of `now`. It is computed once per
consumed symbol from that symbol's own latest kept bar; the caller (`assert_marks_usable`) wraps the
call so a `MinuteOutOfBounds` / unmappable timestamp becomes `RiskBreach("stale_marks")` that flows to
the halt-without-flatten policy rather than crashing as a generic exception (finding 5).

### Post-close-before-UTC-midnight boundary (finding 6)
The `ts.date() < now.date()` cutoff is a UTC-date comparison, while `sessions_stale` maps `now`
through the EXCHANGE calendar. In the window after the US close but before UTC midnight (e.g. `now`
= 2023-06-01 16:30 ET = 20:30 UTC, same UTC date; or the symmetric case where the exchange session
and the UTC date diverge), the freshest genuinely-fresh bar can read as staleness 1 rather than 0
because of the half-open `end=today` fetch plus the UTC-date cutoff. `MAX_STALE_SESSIONS = 2`
tolerates this (staleness 1 passes), but the interaction is subtle and MUST be pinned by explicit
tests so a future cutoff/`now`-mapping change cannot silently push a genuinely-fresh tick to
staleness 2+ and start spuriously flattening. Tests assert the effective staleness stays ≤ 1 across
that boundary window.

## Deliberate scope
- The account-level book-exposure seed (`_build_book_exposure`, #389) IS now freshness-walled
  (Layer A2, finding HIGH #2) — a stale/absent/future-dated account-held mark engages the global
  halt (halt-only). Any other bar-seeded book accumulator enforced before the per-strategy loop
  needs no separate guard: a stale-seed mis-valuation can only ever *permit* a buy that the
  per-strategy `run_tick` then trips on `stale_marks` before any order is submitted — the
  per-strategy wall is the authoritative backstop, and a falsely-conservative stale seed only defers
  (safe).
- Intraday (`timeframe != "1d"`) freshness is deferred (the wall fails closed on it, above).
- No DB schema change, no settings/schema bump.
- No CODEOWNERS-protected file is touched (`risk/limits.py`, `risk/breach.py`,
  `calendar/market_calendar.py`, `live/live_loop.py`, `execution/live_sizing.py`, `cli/live_cmd.py`,
  `cli/paper_cmd.py` are all unprotected) → auto-merge on green CI.

## Files / task list
1. `algua/calendar/market_calendar.py` — add `sessions_stale(latest_bar, now) -> int` (signed, NOT
   clamped; propagates `MinuteOutOfBounds`).
2. `algua/risk/limits.py` — add `MAX_STALE_SESSIONS = 2` and
   `check_mark_freshness(stale_by_symbol: dict[str, float], max_stale: int)` (pure; classifies
   `no_mark` via `math.inf`, `future_dated` via negatives, `stale` via `> max_stale`; raises
   `RiskBreach("stale_marks", ...)` listing every offender).
3. `algua/live/live_loop.py` — in `run_tick`: (a) fail closed unless `timeframe == "1d"` at entry;
   (a2) compute `held` BEFORE `get_bars` and fetch `sorted(set(strategy.universe) | held)`; value on
   the union frame but derive BOTH the warm-up count AND the decision timestamp `t` from the
   `strategy.universe`-restricted frame (`universe_bars`) and rank `decide()` over `universe_bars.loc[:t]`,
   so the widened fetch never pollutes `target_weights()` NOR the warm-up/timing gate (Round-2c/2d);
   (b) add the pure `_latest_bar_ts(bars)` helper and rework `_latest_marks` so BOTH read each
   symbol's latest row by timestamp with **nulls preserved** (e.g. `groupby('symbol').tail(1)` on the
   sorted frame → `symbol -> (ts, close)`), NOT `groupby(...).last()` which skips NaN (Round-2b
   finding); plus a **shared, importable** `assert_marks_usable(
   symbols, latest_ts, latest_close, now)` (unvaluable scan + staleness map with `math.inf` for
   absent + calendar-error wrapping → `RiskBreach`) — exported (no leading underscore, or re-exported)
   so `cli/live_cmd.py`'s `_build_book_exposure` reuses the SAME wall (HIGH #2); (c) HELD-book gate
   BEFORE the empty-bars / warm-up early returns and before sizing; **(d) HELD-book risk VALUATION
   (`build_sizing_snapshot` + non-positive-equity guard + `check_drawdown` + reconcile + realized-
   gross) ahead of the warm-up early return whenever the book is held — gate valuation on
   `valued = held or not warming`, not on warm-up (CRITICAL); a held-warming tick returns the computed
   equity/peak/realized_gross/positions_before so the drawdown peak persists**; (e) CONSUMED-set gate
   on the DECISION path only (past warm-up), after sizing, before `decide()`.
4. `algua/execution/live_sizing.py` — **remove the `equity <= 0.0` `LiveSizingError` raise; RETURN
   the non-positive-equity snapshot** so `run_tick`'s uniform `non_positive_equity` `RiskBreach` guard
   fires (NAV<=0 now FLATTENs, not skips — Round-2 finding 1). Rework its local `_latest_marks` to the
   same null-preserving latest-row selection (Round-2b finding). Keep the held-mark `LiveSizingError`
   branch as defense-in-depth (unreachable on the wall path, pre-empted by the HELD gate) and comment
   it so.
5. `algua/cli/live_cmd.py` —
   - `run-all` (+ `_run_strategy_tick`) recent rolling-window defaults via sentinel-`None`
     `--start`/`--end` resolved to `today − 400d` / today.
   - **`_run_strategy_tick`'s `except RiskBreach`: split by kind (HIGH #3)** — `kind ∈ {"stale_marks",
     "unvaluable_marks"}` ⇒ `global_halt.engage` + `trip_for_breach` + audit + return a halt marker
     with `liquidation_submitted=False` and NO `flatten_strategy` call; all other kinds keep the
     existing trip + scoped flatten. Update the `except LiveSizingError` comment to "non-positive NAV
     only; mark problems now HALT (no flatten)".
   - **`_build_book_exposure`: reuse `assert_marks_usable` over `net_positions` marks (HIGH #2)** —
     add an injectable `now`; a propagated `RiskBreach` from the caller wrapper engages the global
     halt (halt-only, NO `close_all_positions`), audits `book_stale_marks_halt`, emits an `ok:False`
     book-breach envelope + `raise typer.Exit(1)`. The bespoke `mark is None / not finite / <= 0`
     scan is subsumed by the shared wall; the short-precondition + seed-breach `(None, reason)` defers
     stay unchanged.
6. `algua/cli/paper_cmd.py` —
   - same recent-window defaults on `trade-tick` (+ `_run_paper_strategy_tick`); leave `paper run`
     (SimBroker replay) on the 2023 window.
   - **`_run_paper_strategy_tick`'s `except RiskBreach`: same kind-split as live (HIGH #3)** — data-
     integrity kinds ⇒ `global_halt.engage` + `trip_for_breach` + halt marker, NO flatten; other
     kinds keep the trip + flatten.
   - ADD `except LiveSizingError -> paper_sizing_skipped` returning
     `ok({...,"traded": False,"skipped":...})` as defense-in-depth so any residual `LiveSizingError`
     is a clean skip, never an unhandled generic exception (finding 2). (Non-positive NAV now flows to
     `except RiskBreach` → flatten, not here.)
7. Tests:
   - `tests/test_market_calendar.py` — `sessions_stale`: exact-boundary (2 passes / 3 raises),
     normal-op (1), weekend (0), future-dated (negative), and the post-close-before-UTC-midnight
     boundary (staleness ≤ 1) cases (finding 6).
   - `tests/test_risk_limits.py` — `check_mark_freshness`: all-fresh passes; **mixed fresh/stale**
     (one fresh symbol + one stale held symbol) raises `RiskBreach("stale_marks")` naming the stale
     symbol; a **`no_mark`** entry (`math.inf`) raises and is reported as `no_mark` (finding 3); a
     **future-dated** entry (negative) raises as `future_dated` (finding 4); empty map passes.
   - `tests/test_live_loop.py`:
     - **Dead-feed + held book** (empty frame, positions held) trips `RiskBreach("stale_marks")`
       with every held symbol reported `no_mark` — verifying the HELD-book gate runs BEFORE the
       empty-bars early return (finding 1).
     - **Stale-feed + held book** (all bars ≥ 3 sessions old) trips before sizing.
     - **Unvaluable held mark** — a present close that is `<= 0`, **`+inf`, OR `NaN`** trips
       `RiskBreach("unvaluable_marks")` (proving the `isfinite and > 0` predicate rejects non-finite
       positives, Round-2 finding 2). Includes the **NaN-masking** case (Round-2b finding): a held
       symbol whose OLDER bar has a finite close but whose LATEST bar has a `NaN` close still trips
       `unvaluable_marks` — proving the null-preserving latest-row selection does not backfill the
       older finite close via `groupby(...).last()`.
     - **Partial frame on the decision path** — a universe/consumed symbol absent from the frame
       trips `no_mark` after sizing (finding 3).
     - **Held-but-warming book still runs the breakers (CRITICAL)** — a strategy holding a position
       with `bars.index.nunique() <= warmup_bars` and FRESH marks: (a) a book down past the drawdown
       limit trips `RiskBreach("drawdown")` even though warm-up is not met (assert the breach); (b) a
       book within limits returns a `TickResult` with `decision_ts=t`, empty `submitted`, and a
       non-None `peak_equity`/`equity`/`realized_gross` reflecting the valued book (assert the peak is
       persisted), proving `check_drawdown`/realized-gross ran ahead of the warm-up early return.
     - **Flat + warming** still returns the no-op `TickResult` (`peak_equity is None`) — valuation
       is skipped only when nothing is held.
     - **Held out-of-universe symbol (Round-2c)** — a strategy holding a symbol NOT in
       `strategy.universe`, with a FRESH mark for it: `run_tick` fetches it (union), values the book,
       and does NOT false-trip `no_mark`; assert the provider was asked for `universe ∪ held` and that
       a drawdown on the held out-of-universe book flattens (trustworthy-mark path), while a genuinely
       absent bar for that held symbol DOES trip `no_mark`. Also assert `decide()`'s view excludes the
       out-of-universe symbol (decision unaffected by the widened fetch).
     - **Warm-up isolation (Round-2d)** — a held out-of-universe symbol with LONG history while the
       universe has FEWER than `warmup_bars` distinct sessions: `run_tick` still treats the strategy as
       warming (values the held book, runs the breakers, but NO `decide()`/submit), proving warm-up and
       `t` count universe sessions, not union sessions.
     - **Fresh book passes**; **`timeframe != "1d"` fails closed** (ValueError, no trade).
   - `tests/test_live_cmd.py` / `tests/test_paper_cmd.py` (handler policy, HIGH #3):
     - **Dark-feed halt-only** — a `stale_marks`/`unvaluable_marks` `RiskBreach` from `run_tick`
       engages the global halt AND does NOT call `flatten_strategy` (`liquidation_submitted=False`);
       asserted on BOTH `_run_strategy_tick` (live) and `_run_paper_strategy_tick` (paper).
     - **Economic breach still flattens** — a `drawdown`/`gross_exposure_realized` `RiskBreach` DOES
       trip + flatten (assert the flatten submission), unchanged.
   - `tests/test_live_cmd.py` — `_build_book_exposure` account-level wall (HIGH #2): a stale / absent
     / future-dated / `+inf` `net_positions` mark makes `run-all` engage the global halt AND NOT call
     `close_all_positions` (halt-only); a fresh account book builds the `BookExposure` unchanged;
     the short-precondition + seed-breach defers still return the benign skip note.
   - `tests/test_live_loop.py` / `test_live_sizing.py` — **non-positive NAV FLATTENs, not skips
     (Round-2 finding 1)**: a wall tick whose ledger NAV `<= 0` (held positions, trustworthy marks)
     raises `RiskBreach("non_positive_equity")` from `run_tick`, and the live AND paper handlers trip
     + flatten it (assert the flatten submission), rather than returning a silent skip; assert
     `build_sizing_snapshot` returns the non-positive-equity snapshot (no `LiveSizingError`). A
     separate belt-and-suspenders test: a directly-raised `LiveSizingError` (non-wall) still yields a
     clean skip envelope, never an unhandled generic exception (finding 2).
   - CLI tests — `paper trade-tick` and `live run-all` default `--end` resolves to today and
     `--start` to `today − 400d`; `paper run` still defaults to the 2023 replay window.

## GATE-1 findings resolved

### Round 2d (GATE-1 re-run)
- **widened fetch polluted the warm-up gate** — the Round-2c union fetch left
  `warming = bars.index.nunique() <= warmup_bars` and `t` computed on the union frame, so a held
  out-of-universe symbol's longer history could make the universe look warmed and run `decide()` on an
  under-warmed universe. Fixed by deriving BOTH the warm-up count and `t` from the universe-restricted
  `universe_bars`, keeping held-book valuation on the union frame.

### Round 2c (GATE-1 re-run)
- **held out-of-universe symbol false-tripped `no_mark`** — `run_tick` fetched only
  `strategy.universe`, so an inherited / out-of-universe held symbol carried no bar and read as a dark
  feed (`no_mark`), routing a trustworthy held book to halt-without-flatten instead of valuing it.
  Fixed by computing `held` before the fetch, fetching `universe ∪ held` (valuation on the union
  frame), and passing a universe-restricted view to `decide()` so the widened fetch never pollutes
  the decision.

### Round 2b (GATE-1 re-run on the Round-2 revision)
- **(finding 1) non-positive NAV was a silent skip** — a wiped book (`NAV <= 0`) that still holds
  positions was skipped every tick, never halted or flattened. Fixed by removing
  `build_sizing_snapshot`'s `equity <= 0.0` raise and letting `run_tick`'s uniform
  `RiskBreach("non_positive_equity")` guard fire for both snapshot sources → trip + flatten. `NAV<=0`
  off trustworthy marks is a genuine economic wipe, correctly in the flatten set.
- **(finding 2) positive non-finite marks passed the wall** — `assert_marks_usable`'s bare
  `close > 0.0` let `+inf` (and the account-level check let `+inf`/`NaN`) through to poison NAV /
  drawdown / gross. Fixed to require `math.isfinite(close) and close > 0.0` for every consumed /
  account-held mark; a `+inf`/`NaN` mark now trips `unvaluable_marks`.
- **(finding 3, Round-2b) NaN latest close masked by `groupby(...).last()`** — pandas
  `GroupBy.last()` skips NaN, so a symbol whose newest bar had a `NaN` close but an older finite close
  returned the older close paired with the newest timestamp, slipping past the finite-mark wall.
  Fixed by selecting each symbol's latest row by timestamp with **nulls preserved** (e.g.
  `groupby('symbol').tail(1)`) in `_latest_marks` / `_latest_bar_ts` (and the matching selection in
  `build_sizing_snapshot` / `_build_book_exposure`), so a `NaN`-latest correctly trips
  `unvaluable_marks`.

### Round 2 (this revision)
- **(CRITICAL) warm-up suppresses the breakers for a held book** — risk VALUATION
  (`build_sizing_snapshot` + non-positive-equity guard + `check_drawdown` + reconcile + realized-
  gross) now runs whenever `valued = held or not warming`, i.e. ahead of the warm-up early return for
  EVERY held book. Warm-up suppresses only `decide()`/new orders; a held-warming book is valued,
  ratchets its peak, and trips the breakers. Tested by a held-but-warming tick that trips `drawdown`
  and by one that persists a non-None `peak_equity` (placement 2, Layer A).
- **(HIGH #2) account book valued off unchecked marks** — `_build_book_exposure` reuses the shared
  `assert_marks_usable` wall over `net_positions` marks (Layer A2); an absent/stale/unvaluable/
  future-dated account-held mark engages the GLOBAL HALT (halt-only, no account-wide flatten, since
  only the bar feed is dark while the broker-truth book-loss breaker still guards). Tested to halt
  without `close_all_positions`.
- **(HIGH #3) auto-liquidate on a dead feed** — the held-book flatten is gated on an EXPLICIT
  condition: the CLI `except RiskBreach` handlers flatten ONLY for trustworthy-mark economic/position
  kinds (`drawdown`/`gross_exposure_realized`/`reconcile`/`non_positive_equity`); a data-integrity
  kind (`stale_marks`/`unvaluable_marks` — dark BAR feed, broker alive) is HALT-WITHOUT-FLATTEN
  (documented deliberate policy: forced market liquidation on a data outage is self-harm; the
  broker-truth book breaker guards real drawdown). Tested both ways in live and paper.

### Round 1 (prior revision)
- **(1) early-return dead-feed** — HELD-book gate runs before the empty-bars AND warm-up early
  returns; a held book on an empty/stale frame trips `RiskBreach("stale_marks")` instead of
  returning a no-op `TickResult`.
- **(2) LiveSizingError vs halt** — held mark problems are converted to `RiskBreach` before
  `build_sizing_snapshot` (option a), so they route to the halt-without-flatten policy via the
  existing handlers; the non-positive-NAV case is FLATTENed as an economic breach (superseded by
  Round-2b finding 1 — no longer a skip); paper gains an `except LiveSizingError` (defense-in-depth)
  so no residual `LiveSizingError` escapes as a generic exception.
- **(3) missing consumed mark** — the `if s in latest_by_symbol` filter is dropped; an absent
  consumed symbol maps to `math.inf` and is reported as a `no_mark` offender.
- **(4) future-dated clamp** — `sessions_stale` no longer `max(0, n)`-clamps; a negative (future)
  staleness fails closed as `future_dated`.
- **(5) calendar mapping crash** — `assert_marks_usable` wraps the `sessions_stale` /
  `session_of_instant` call so `MinuteOutOfBounds` / unmappable timestamps become
  `RiskBreach("stale_marks")` that flows to the halt-without-flatten policy.
- **(6) post-close-before-UTC-midnight** — explicit boundary tests pin the effective staleness ≤ 1
  across that window.
