# Survivorship-free 30-year backtests — PIT constituents + delisting-aware exits (#212)

**Status:** design (GATE-1 reviewed — Codex + Gemini panel; reshaped r1, refined r2, semantics tightened r3)
**Issue:** #212
**Date:** 2026-06-15

## Problem

A 30-year backtest is only honest if it is **point-in-time survivorship-free**. Otherwise
"robust across 30 years" silently means "robust among the names that survived to today" —
survivorship bias wearing a robustness costume. Two gaps remain after the existing PIT machinery:

1. **No bulk path for historical constituents.** `data ingest-universe` records ONE membership
   snapshot per call. A 30-year index history has thousands of add/drop events; we need a
   bulk importer: a constituents CSV (symbol, add_date, drop_date — including delisted tickers) →
   the universe-snapshot timeline the engine already consumes.
2. **No delisting-aware exit (a genuine correctness gap today).** When a held symbol's bars end
   mid-backtest, its column in the execution grid goes `NaN`. vectorbt cannot execute a sell at a
   `NaN` price, so the position **freezes** at its last value and `0 × NaN` poisons group equity.
   The position is never realized and never removed — a silent survivorship leak.

The actual survivorship-bias-free **dataset** (CRSP / Norgate / Sharadar) is human/vendor work,
explicitly deferred ("take care of later"). This spec builds the **code** that consumes it.

## GATE-1 reshape (why this design is not "infer-from-bars + realize-at-last-close")

The first design inferred a delisting from "a symbol's last valid bar precedes the panel's last
bar" and realized the position at its last close by default. The review panel (Codex + Gemini,
independent, in agreement) rated this **CRITICAL-unsound**: "bars end early" also describes coverage
truncation, illiquidity, acquisitions, suspensions, and ADR changes. Silently realizing any of those
at the last close *fabricates a clean exit* — itself a survivorship leak. And because `panel_end` is
set by the longest-lived survivor, in a 30-year union *most* names end before it.

The reshaped design therefore **never invents an exit**. For a symbol whose bars end before
`panel_end` exactly two things happen:

1. **Always** kill the `NaN`-poisoning (forward-fill the dead column; the position there is 0). This
   is a pure numerical bug fix with no behavioral effect on a live/healthy run.
2. **Force a liquidation only if the position is actually held past its last real bar** — and even
   then only with a **confirmed delisting record**; otherwise **fail closed**. The only *new* errors
   this introduces are runs that already produced `NaN`-poisoned garbage, so blast radius on
   existing backtests is ~nil.

## Existing machinery (consumed, not rebuilt)

- `DataStore.ingest_universe(name, symbols, effective_date, as_of, source)` →
  content-hashed parquet snapshot in `snapshots/manifest.jsonl` (`store.py:247`).
- `DataStore.read_universe(name)` → `list[UniverseSnapshot]` sorted by `effective_date`, with an
  ambiguity guard (two snapshots, same effective_date, different membership → `ValueError`)
  (`store.py:713`).
- `_members_as_of(universe_by_date, t)` → greatest `effective_date ≤ t.date()`, look-ahead-free
  (`backtest/engine.py:42`).
- `_decision_weights(...)` masks the strategy view to as-of members and calls the unified
  `validate_decision_weights(..., allowed_symbols=members)` (`engine.py:134`, `risk/limits.py:150`).
- `simulate(...)` fetches the union of all-ever-members, pivots `adj_close` to the `adj` grid
  `(timestamp × symbol)`, computes `weights_eff` (fast-path or loop), applies `decision_lag`, and runs
  `vbt.Portfolio.from_orders(close=adj, size=weights_eff, size_type="targetpercent",
  cash_sharing=True, group_by=True)` (`engine.py:530`).
- `resolve_universe_inputs(name, start, end)` (`cli/_common.py:82`) reads the timeline and returns
  `(universe_by_date, provenance)`, threaded into `simulate` / `walk_forward` / `sweep` via
  `--universe`.
- `Dataset` / `Kind` StrEnums + `SnapshotMetadata` (`data/models.py:12`). `SCHEMA_VERSION = 2`.

## Component A — Constituents bulk-importer (`data import-universe`)

New CLI subcommand (`cli/data_cmd.py`):

```
algua data import-universe --name NAME --file constituents.csv [--as-of TS] [--source LABEL]
```

- **CSV schema:** `symbol,add_date,drop_date`. Empty `drop_date` = still a member (open interval).
  **Multiple rows per symbol allowed** → multiple membership intervals (re-additions, e.g. S&P
  re-entries).
- **Membership convention:** `add_date` **inclusive**, `drop_date` **exclusive**.
  `membership@D = { s : ∃ interval(s) with add ≤ D and (drop is empty or D < drop) }`.
- **Pure transformer** `constituents_to_snapshots(rows) -> list[tuple[date, frozenset[str]]]`
  in a new pure module `algua/data/constituents.py` (no I/O):
  - `change_dates = sorted(unique(all add_dates ∪ all drop_dates))`.
  - For each change date `D`, compute `membership@D`; emit `(D, members)`.
  - **Collapse no-op change dates:** if `membership@D` equals the previously emitted membership,
    skip `D` (keeps the timeline minimal).
- **Symbol normalization first (LOW):** symbols are canonicalized (`normalize_symbols` semantics —
  strip/upper) at CSV-parse time, *before* duplicate/overlap validation, so two spellings of the
  same ticker can't slip a duplicate or overlap past the checks (the store normalizes later anyway).
- **Fail-closed validation** (raises before any ingest, on canonical symbols):
  - malformed/unparseable dates or missing `symbol`/`add_date`;
  - `add_date > drop_date`;
  - **`add_date == drop_date`** (degenerate zero-length interval) → **rejected** (C11: no silent
    drop, no ambiguous "flagged but kept");
  - **overlapping intervals for the same symbol** (ambiguous membership) → error;
  - duplicate identical rows → de-duplicated, not errored.
- **Atomicity (C9):** the transformer runs **fully in memory and validates the entire file first**;
  only then are snapshots ingested in change-date order. A bad row therefore never produces a
  partial timeline. A power-loss mid-ingest leaves a partial timeline that is **re-runnable**
  (content-hash dedup makes already-written snapshots no-ops) — the same durability contract every
  existing ingest path has; full transactional all-or-nothing across snapshots is out of scope.
- **Immutability enforced under the manifest lock (C10, round-2 HIGH + round-3 MEDIUM TOCTOU):**
  documenting immutability is not enough — `ingest_universe` appends without checking existing
  same-name dates, and `read_universe` only catches a conflict *after* the poisoning write. A bare
  pre-read is also racy: two concurrent imports can both read clean, then append conflicting same-date
  memberships. So the same-date compatibility check is a **compare-and-append performed under the
  manifest append lock** (`manifest.py` already serializes appends via flock): a new
  `DataStore.ingest_universe(..., require_immutable=True)` path that, *holding the lock*, rejects a
  change date whose membership differs from an already-stored snapshot on that date before appending.
  A cheap pre-read stays as a fail-fast for the common case. Corrections require a new universe name;
  a rejected import leaves the manifest untouched (asserted, including under a concurrent-writer test
  reusing the #164 harness).
- **Wiring:** for each emitted `(D, members)`, call `DataStore.ingest_universe(...)`. Reuses
  content-hash dedup, manifest append, fsync durability, and the `read_universe` timeline. Emits a
  summary JSON (`snapshots_written`, `change_dates`, `symbols_seen`). **No engine change.**

## Component B — Delisting-aware exit (the genuine correctness addition)

A pure overlay on the **execution grid**, applied inside `simulate()` immediately before
`from_orders`. New pure function in `algua/backtest/delisting.py`:

```python
@dataclass(frozen=True)
class DelistingRecord:
    delisting_date: date
    terminal_price: float        # per-share terminal proceeds, in adj_close units; strictly > 0
    source: str
    def __post_init__(self):
        # round-3 HIGH: enforce at the API boundary, not only at `import-delistings`, so a
        # programmatically-constructed record can never push a price <= 0 into adj[T].
        if not (self.terminal_price > 0) or not math.isfinite(self.terminal_price):
            raise ValueError("terminal_price must be finite and > 0 (zero-proceeds write-off deferred)")

def apply_delisting_exits(
    adj: pd.DataFrame,                                    # (timestamp × symbol) adj_close grid
    weights_eff: pd.DataFrame,                            # post-lag target-percent EXECUTION weights
    records: Mapping[str, list[DelistingRecord]] | None = None,  # multiple events per symbol
    *,
    assume_terminal_last_close: bool = False,             # human-only relaxation (see Component C)
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:       # (adj_exec, weights_exec, forced_exits)
    ...
```

There is exactly **one panel per simulation**: `build_portfolio` is an alias of `simulate`, which
runs `from_orders` once over the full period; `walk_forward` segments the *resulting* `pf.returns()`
into windows (it does **not** re-simulate per window — round-2 correction), and `sweep` runs one
`simulate` per combo over its own full period. So the overlay applies once per `simulate` call with
`panel_end = adj.index[-1]` (the full backtest end), and there is no per-window record filtering.

**Records are keyed `symbol -> list[DelistingRecord]`** (round-2 HIGH): a ticker can delist, be
reused, or re-enter an index across decades, so a single record per symbol is unsound. (Disambiguating
two *different companies* sharing a reused ticker on one continuous bar series is a deeper PIT-identity
problem — see #205 — and is out of scope.)

For each symbol column `c` let `T = adj[c].last_valid_index()` (skip entirely if `T is None` — never
traded in this panel) and `first_bar = adj[c].first_valid_index()`.

**Applicable-record selection (round-3 HIGH — exact):** a record `r` for `c` is a *candidate* iff
`first_bar.date() ≤ r.delisting_date ≤ panel_end.date()` (a date after `panel_end` is **not** a
candidate — it does not "resolve to `panel_end`") **and** its resolved bar `D_bar(r)` = greatest panel
bar `≤ r.delisting_date` equals `T`. Then:
- **0 candidates** → no applicable record.
- **exactly 1 candidate** → that is the applicable record.
- **≥ 2 candidates** (distinct dates collapsing to the same terminal bar `T`, e.g. weekend dates with
  different prices) → **fail closed** (ambiguous terminal valuation).

**Integrity check:** for *any* record `r` of `c` with `r.delisting_date ≤ panel_end.date()`, if a real
bar for `c` exists strictly after `D_bar(r)` → **fail closed** (the symbol traded past its stated
delisting — data inconsistency). This is independent of `T < panel_end` vs `T == panel_end`: a period
ending exactly on the delisting (`D_bar == T == panel_end`) is allowed; only bars *beyond* the
delisting are rejected.

**Action (applies whether or not `T == panel_end`):**
- **Applicable record exists and the position is held** (`held = weights_eff.loc[T, c] != 0` — a
  *detection* proxy on target state; the **runtime** post-sim guarantee below makes it load-bearing):
  `weights_exec.loc[T:, c] = 0.0` (liquidate **into** `T`, capturing `T`'s close-to-close return) and
  `adj_exec.loc[T, c] = record.terminal_price` (realize at terminal proceeds — `> 0` by construction).
  Append `{symbol, bar: T, terminal_price, source}` to `forced_exits`. (When `T == panel_end` there is
  no NaN tail, but the override + forced flat still realize the terminal price, which may differ from
  the last close.)
- **Bars end early (`T < panel_end`) and the position is held with NO applicable record** → **fail
  closed** (a held position into a terminal data gap cannot be honestly valued) — UNLESS
  `assume_terminal_last_close` (human-only): liquidate at `T`'s last close (`weights_exec.loc[T:, c]
  = 0.0`, no override) and append `{... source: "assumed_last_close"}`.
- **not held:** nothing to do (the position was already 0 at `T`, exited at a real price ≤ `T`).

**NaN-poison kill (always, whenever `T < panel_end`, independent of the above):**
`adj_exec.loc[after T, c] = adj[c].loc[T]` (forward-fill the last real price). The position there is 0,
so the price is inert for PnL; this removes the `0 × NaN` that corrupts group equity.

**Runtime post-sim guarantee (round-3 HIGH — not test-only):** after `from_orders`, `simulate`
verifies that every symbol in `forced_exits` has **actual position zero** on every bar after its forced
bar and that group equity/returns are finite; otherwise it raises `BacktestError`. The `held` proxy and
`call_seq="auto"` are thereby backstopped by a hard runtime check, not just integration tests.

**vectorbt sequencing (C3/C4):** the forced sell at `T` should free cash usable by any same-bar
rebalance of other symbols, and the override price must not mis-value the group. `from_orders` is
called with **`call_seq="auto"`** (sells before buys within a bar under `cash_sharing`) — the current
`simulate` passes no `call_seq`. This is the *right direction*, **not a correctness proof**: vectorbt
itself notes `TargetPercent` + auto sequencing relies on approximate order values. So it is backed by
tests, not asserted: a golden-backtest **regression guard** (does `call_seq="auto"` shift existing
results? if so it is surfaced and decided explicitly, never silently), plus a constrained-cash
forced-exit rebalance test. The realized exit is verified by asserting the **actual** post-sim
position is zero (no residual/rejected order), not merely that the target was set to 0.

**Decision vs execution weights (C6):** the overlay is an **exogenous corporate-action execution**,
not a strategy decision. The strategy's `decision_weights` (what #178 exhaustive parity and #179
decision parity check) are computed and validated **upstream and unchanged**; `apply_delisting_exits`
produces a distinct `execution_weights` grid downstream. The `forced_exits` list is surfaced in the
result JSON so a forced liquidation is never mistaken for a strategy decision.

**Properties:**
- **Path-agnostic / parity-safe:** applied *after* weight computation, identically for the fast-path
  and the loop; `verify_signal_panel_parity` (#178) compares *pre-overlay* weights — unaffected.
- **No false delistings:** an exit is forced only for a position *held past its last real bar*, and
  only with a confirmed record (or the human relaxation). Coverage truncation / illiquidity / index
  removal of an *unheld* name never triggers a forced exit.
- **Composes with PIT masking:** a name removed from the universe while still trading was already
  sold at the drop (weight 0 at a real bar) → not `held` at `T` → no forced exit; just the NaN-kill
  of its dangling column. Index-removal and delisting are orthogonal and layer cleanly.
- **No look-ahead:** uses only `adj` (engine-owned prices) and runs after the strategy decided; it
  feeds the strategy no future information.

**Threading:** `simulate(..., delisting_records=..., assume_terminal_last_close=...)`, passed through
`walk_forward` and `sweep`. Because `walk_forward` simulates **once** over the full period and only
segments the returns, the overlay runs once against the full-period panel; `sweep` passes the same
records into each combo's `simulate`. There are no per-window panels, so no per-window record
filtering — the only "filtering" is the per-symbol resolved-bar selection above.

**Out of scope:** interior data-gap handling (a hole that later resumes — `last_valid` is the true
last bar, so interior gaps never trigger; carrying a position across an interior gap is a pre-existing
data-quality concern, untouched here).

## Component C — Delisting-record ingestion (minimal, end-to-end)

The forced-exit path is load-bearing on these records (no longer optional), so the ingestion is built
now. Deliberately minimal and schema-versioned so the layout can evolve when the real vendor format
lands.

- **New `Dataset.DELISTINGS = "delistings"` and `Kind.DELISTING = "delisting"`** (`data/models.py`).
  Additive enum values; readers filter by kind, so **no `SCHEMA_VERSION` bump** (record shape
  unchanged).
- **CLI:** `algua data import-delistings --file delistings.csv [--as-of TS] [--source LABEL]`.
  CSV: `symbol,delisting_date,delisting_value`.
- **Value semantics (C5):** `delisting_value` is the **per-share terminal price in adj_close units**
  (terminal proceeds per share), NOT a raw vendor return. If a vendor (e.g. CRSP) supplies a
  delisting *return* `r`, the operator/importer converts it to a price
  (`terminal_price = last_adj_close * (1 + r)`) **before ingest**; the conversion and source are
  recorded in `source`/provenance so an already-adjusted close is never double-counted in-engine.
- **Store:** `DataStore.ingest_delistings(records, as_of, source)` → content-hashed parquet snapshot
  (`symbol, delisting_date, delisting_value`), manifest append, fsync durability — mirrors
  `ingest_universe`. `DataStore.read_delistings(as_of=None) -> Mapping[str, list[DelistingRecord]]`
  reads the latest delistings snapshot with `as_of ≤ given` (point-in-time); multiple rows for one
  symbol (re-additions / reuse) become a list.
- **Validation (fail-closed at ingest):** `delisting_value` finite and **strictly `> 0`** (round-2
  CRITICAL — vectorbt rejects an order price `≤ 0`, so a `0`/total-loss proceeds cannot be modeled
  by overriding `adj[T]`; we **fail closed at ingest** with a clear "zero-proceeds write-off not yet
  supported" message rather than fake it with an epsilon price — the true write-off path is
  deferred); `delisting_date` parseable; **duplicate `(symbol, delisting_date)` within one file →
  error** (the same event twice), but **distinct dates for one symbol are allowed** (the list above).
- **CLI threading:** `resolve_delisting_inputs(name_or_none, end_dt)` in `cli/_common.py` mirrors
  `resolve_universe_inputs` → `(records | None, provenance | None)`. New optional `--delistings NAME`
  flag on `backtest` / `walk-forward` / `sweep` / `research promote`.
- **Human-only relaxation:** `--assume-terminal-last-close` threads `assume_terminal_last_close=True`
  into the overlay. Like every other relaxation in the platform it is **human-only** — the agent
  research/promote path forbids it; a held-into-gap-without-record position fails closed for an agent,
  always. Its use is stamped in provenance (`source: "assumed_last_close"`).

## Lifecycle / boundary / quality

- `algua/contracts` and `algua/features` stay pure; the overlay lives in `algua/backtest`
  (pandas allowed), the transformer in `algua/data` (pure). No new cross-module imports beyond the
  existing seams. `lint-imports` stays green.
- Live/paper paths unchanged: a delisting is a broker event there, not a simulation concern.
- Provenance: result JSON already surfaces `universe_snapshots`; add `delisting_snapshot` and the
  `forced_exits` list (per-symbol bar + realized terminal price + source).

## Testing strategy (TDD)

- **Transformer (pure, unit):** open interval (survivor), closed interval (delisted), re-addition
  (two intervals → re-entry snapshot), simultaneous add+drop on one date (single snapshot), no-op
  collapse, overlapping-interval rejection, `add > drop` rejection, `add == drop` rejection,
  malformed-row rejection, duplicate-row de-dup.
- **Delisting overlay (pure, unit):**
  - held-past-last-bar **with** record → forced flat at `T`, realized at `terminal_price`, return
    into `T` captured, `forced_exits` recorded;
  - held-past-last-bar **without** applicable record → `BacktestError` (and, under
    `assume_terminal_last_close`, realized at last close + stamped);
  - **not held** at `T` (already exited) → only NaN-kill, no forced exit, no error;
  - NaN killed after `T` (assert no `0×NaN` reaches group equity);
  - **integrity:** a record with bars existing strictly *after* its resolved `D_bar` → fail closed;
  - **boundary:** period ends exactly on the delisting (`D_bar == T == panel_end`) → the
    `terminal_price` override **is** realized (assert the realized cash differs from last close when
    the prices differ), **not** a false integrity failure;
  - record dated **after** `panel_end` → not a candidate, skipped (no error);
  - record dated **before** the symbol's `first_bar` → not a candidate;
  - **multiple records per symbol** (delist → reuse → delist again) → the event whose `D_bar` matches
    the panel's terminal bar is selected;
  - **≥ 2 candidates** collapsing to the same terminal bar → fail closed (ambiguous);
  - `DelistingRecord(terminal_price ≤ 0 or non-finite)` → `ValueError` at construction (API boundary);
  - symbol never traded in panel (`last_valid is None`) → skipped.
- **Runtime post-sim guarantee:** a constructed case where the forced exit leaves a residual position
  (e.g. simulated rejected order) → `simulate` raises `BacktestError` (proves the guarantee is runtime,
  not test-only).
- **vectorbt order-level integration:** a 3-symbol panel, one delists mid-period held with a record —
  assert the **actual post-sim position is zero** (no residual/dust), cash realized at
  `terminal_price`, group equity finite throughout; a contrast run *without* the overlay shows the
  NaN freeze/poison (locks in the fix); a 4th symbol rebalancing on bar `T` under constrained cash
  exercises `call_seq="auto"` cash availability; a golden-backtest regression test guards that
  `call_seq="auto"` does not silently shift existing results.
- **Importers end-to-end:** `import-universe` → `read_universe` + `_members_as_of` resolve correctly
  across change dates and re-additions; symbol normalization happens before validation (a
  mixed-case duplicate is caught); the **pre-write immutability guard** trips on a same-name
  correction **and leaves the manifest unmutated**; `import-delistings` round-trips through
  `read_delistings` (incl. multi-row symbols → list); `import-delistings` rejects `value ≤ 0`;
  `--delistings` threads records into `simulate`.
- **Agent vs human:** the agent research/promote path rejects `--assume-terminal-last-close`;
  human path accepts and stamps it.

## Deferred (follow-ups, not this slice)

- Sourcing the actual CRSP / Norgate / Sharadar survivorship-bias-free dataset (the "data later"
  work), including the real per-symbol terminal prices / delisting returns.
- **Zero-proceeds (total-loss) write-off** (round-2 CRITICAL): a `terminal_price == 0` bankruptcy
  cannot be modeled via an `adj[T]` override (vectorbt rejects price `≤ 0`). This slice fails closed
  at ingest on `value ≤ 0`; a dedicated write-off mechanism (realize the position to zero cash
  outside the `from_orders` close-price path, e.g. a cash adjustment) is a follow-up. Until then a
  near-total loss is representable as a small strictly-positive terminal price.
- Interior data-gap exit policy (distinct from end-of-bars terminal gap).
- A delisting *classification* source (suspension vs acquisition vs bankruptcy) richer than a single
  terminal price — and an importer that converts vendor delisting *returns* to terminal prices
  natively rather than relying on operator pre-conversion.
- A trading-calendar-aware ingest-time `delisting_date` sanity check (current rail is panel-index-only
  at simulation time, which already catches misalignment).
