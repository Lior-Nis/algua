# Pure corporate-action back-adjustment engine (#149)

**Status:** design APPROVED — GATE-1 passed (panel: Codex + Gemini Flash + OpenCode/GLM, 3 rounds;
2 CRITICAL math/look-ahead + 1 CRITICAL validator-false-accept + assorted HIGH/MED all folded in)
**Issue:** #149 (slice 3 of #129). Also owns the economic/event-aligned `adj_close`
validation folded in from #148 / PR#152 GATE-1.
**Date:** 2026-06-10

## Summary

A **pure**, I/O-free corporate-action back-adjustment engine: given a raw OHLC frame plus a
typed split/dividend event list for one symbol, produce the **back-adjusted close** (`adj_close`)
and the **cumulative adjustment factor**. Plus a pure **validator** that checks a *vendor-supplied*
`adj_close` against the same event list (the folded-#148 economic validation), reverse-split-safe.

Both live in a new pure leaf `algua/data/corpactions.py` (imports only pandas + numpy; no algua
imports; lint-imports-clean). **No importer wiring in this slice** — FirstRate ships no event list,
so there is nothing to wire the validation into yet; wiring rides with the first vendor that ships
events (Databento, #150). This delivers the reusable keystone and the folded validator as ready,
exhaustively-tested building blocks.

## Why pure / standalone

The math is `(raw OHLC frame, events) -> (adj_close, factor)` with no I/O. Keeping it a pure helper
makes it exhaustively unit-testable on known splits/dividends and keeps it boundary-clean. It lands
with no vendor wiring of its own — matching #149's original framing.

## Event types — discriminated union

```python
@dataclass(frozen=True)
class Split:
    ex_date: pd.Timestamp   # tz-aware; normalized to UTC internally
    ratio: float            # new shares per old: 2.0 = 2:1; 0.1 = 1:10 reverse split

@dataclass(frozen=True)
class Dividend:
    ex_date: pd.Timestamp   # tz-aware
    cash: float             # per-share cash, in RAW-close (pre-split) price units

CorporateAction = Split | Dividend
```

Two distinct types (not one `value` that means different things by context) — self-documenting and
mypy-friendly; the engine dispatches on type. Each `__post_init__` validates so a malformed event
cannot exist: `ex_date` tz-aware; `Split.ratio` finite and `> 0`; `Dividend.cash` finite and `> 0`
(`math.isfinite` — `float('inf') > 0` is `True`, so the bare `> 0` check is not enough and an `inf`
ratio would silently zero the series). Events are an unordered iterable.

**`ex_date` convention.** `ex_date` is the **ex-dividend / ex-split session date** (not the record,
pay, or announcement date), at **UTC midnight** for the `1d` scope, matching the bar `ts` convention
so the `searchsorted` boundary is well-defined. The 1d engine `tz_convert`s `ex_date` to UTC and
**requires it to be UTC midnight** — a non-midnight `ex_date` (e.g. `12:00Z`) would `searchsorted`
*past* the midnight ex-date bar and wrongly scale the ex-date bar itself (an off-by-one against the
"strictly before" rule), so the engine **raises** rather than silently flooring it. Mapping a
vendor's date *types* and calendars to a canonical session ex-date is the #150 adapter's job and
must fail closed there. Intraday session mapping is out of scope (#151).

**`cash` units (same-date split + dividend).** `Dividend.cash` is stated in the **same units as the
raw close** — i.e. pre-split per-share cash for the share count *before* any same-date split. This
is the only convention under which `m_div = (P_prev − cash)/P_prev` with raw `P_prev` is correct
when a split shares the ex-date. The #150 adapter normalizes vendor cash to this convention.

## The engine — `back_adjust(raw, events) -> pd.DataFrame`

**Input:** `raw` — a DataFrame carrying at least `ts` (tz-aware UTC) and `close`. Other columns are
ignored. Preconditions (fail-closed, see below): `ts` strictly ascending and unique; `close`
**finite** (no NaN/±inf) and `> 0` — `inf` passes a bare `> 0` and would emit infinite/NaN
`adj_close`, so finiteness is checked explicitly.

**Group by ex-date first.** All events are grouped by their (UTC) `ex_date` into one combined
multiplier per date. Within a date: split ratios multiply together; dividend cash amounts **sum**
(`D1 + D2`), then a *single* dividend multiplier is computed from the summed cash. This is
load-bearing: multiplying per-dividend multipliers `(1−D1/P)(1−D2/P)` is **wrong** — it adds a bogus
`D1·D2/P²` cross-term; the correct same-date factor is `(P − (D1+D2))/P`. Grouping also makes the
"order doesn't matter" claim true. Note it makes legitimate same-date *composition* correct — it does
**not** make duplicate-event data errors safe: the engine cannot distinguish two genuine same-date
dividends from the same dividend listed twice by a torn feed (it would double the cash), nor two
"split" rows on one date from a duplicated split (it would double-adjust). De-duplicating
source-identified duplicate events is the #150 adapter's fail-closed responsibility, upstream of the
engine.

**Per-date multiplier `m_d`** (applied to all bars *strictly before* that `ex_date`):
- Split component → `∏ (1 / ratio)`. 2:1 → 0.5; 1:10 reverse → 10.0 (pre-event prices scale **up** —
  the case the naive #148 `r ≤ 1` check false-rejected).
- Dividend component → `(P_prev − Σcash) / P_prev`, where `P_prev` is the **raw close of the last
  bar strictly before `ex_date`** (CRSP/Yahoo total-return convention). `P_prev` is the **raw**
  close, never the adjusted close: the split component is already a separate multiplier in the
  product, so using adjusted `P_prev` would double-apply the split.
- Combined `m_d = split_component × dividend_component`.

**Cumulative factor:** `factor[i] = ∏ m_d` over every ex-date `> ts[i]`. Computed as a **suffix
product** over event boundaries, O(n + m·log n), deterministic:
- For each grouped ex-date, `idx = searchsorted_left(ts, ex_date)` = count of bars with `ts <
  ex_date`. The event scales bars `[0, idx − 1]`. The engine does **no** calendar resolution: an
  `ex_date` that falls on a non-trading day (no matching bar) is handled purely by `searchsorted`,
  scaling every bar strictly before it — correct, since those are exactly the pre-event bars.
- `idx == 0` (ex-date at/before the first bar) → **no-op** (scales nothing; `P_prev` undefined, so
  skipped without evaluation). `idx == n` (ex-date strictly **after** the last bar) → **also a
  no-op** — *symmetric* with the pre-range case. This is a correctness guard, not a convenience:
  applying a future-dated event would scale every bar (including the last) and leak a not-yet-
  observable event into the snapshot — look-ahead. By no-op-ing it, `factor[n−1] ≡ 1.0`, so
  `adj_close[-1] == close[-1]` **structurally** — the series is always anchored at the most recent
  bar, matching vendor `Adj Close` (e.g. yfinance).
- Build `A`, length `n + 1`, all `1.0`; for each in-range ex-date set `A[idx] *= m_d` (only `idx ∈
  [1, n−1]` are ever written — `idx==0` and `idx==n` are the no-ops above). Then
  `factor[i] = ∏_{k=i+1}^{n} A[k]` via one reverse cumulative product (`A[n]` is always `1.0`, the
  identity for the last bar).

**Output:** a DataFrame `[ts, adj_close, adj_factor]`, one row per input bar in input order, with
`adj_close = close * adj_factor`. Returning `adj_factor` keeps the adjustment auditable alongside
raw OHLC (the issue's requirement). **Persistence direction (decided, deferred):** `adj_factor` is
for in-process audit; when #150 wires this in, it persists as a **sidecar corporate-action / factor
artifact**, *not* by adding an `adj_factor` column to the cross-lane bar-schema (which would force a
coordinated `validate_bars` + consumers contract change). This commits the direction now so #150
need not decide it mid-build.

**Edge cases (deterministic, explicit):** empty `events` → `factor ≡ 1.0`, `adj_close ≡ close`
(identity). Empty `raw` → empty `[ts, adj_close, adj_factor]` frame. Pre-range and post-range
ex-dates → no-op (above). No NaN-fill anywhere. **Caller responsibility:** `back_adjust` operates on
exactly the bars passed; a pre-range event is a no-op, so the caller must supply the full raw history
spanning the factor horizon it wants (a truncated window silently yields window-local factors).

## The validator (folded #148) — `check_adj_close_consistent(...)`

```python
def check_adj_close_consistent(
    raw_close: pd.Series, vendor_adj: pd.Series, events,
    *, rtol: float = 1e-3, atol: float = 5e-3,
) -> None: ...
```

`raw_close` and `vendor_adj` share one `ts` index. **Input guards (fail-closed):** identical
tz-aware UTC `DatetimeIndex` on both, strictly increasing, unique; both series finite (no NaN/±inf)
and `> 0`. The check:

**Precondition — full series through the vendor's adjustment anchor.** This validator compares
against a *globally* back-adjusted vendor series, so it must be handed the **full symbol series
through the vendor file's adjustment horizon** (its most recent bar = the anchor), not an arbitrary
mid-history `get_bars(start, end)` slice. On a truncated slice the engine no-ops the events that fall
*after* the slice (they are out of window), while the vendor's `adj_close` still reflects them — so
`v[-1] ≠ 1.0` legitimately and the anchor assertion below would false-reject. At ingest time this is
natural: the adjusted file *is* the full series. (The engine itself has no such restriction — it
produces correct window-local factors for any window; only this cross-check against a global vendor
series needs the full horizon.)

1. Recompute the engine `factor` from `raw_close + events` (so `factor[-1] == 1.0` by construction).
2. Compute the vendor's *implied* factor `v[i] = vendor_adj[i] / raw_close[i]`.
3. **Anchor assertion:** require `v[-1] ≈ 1.0` (equivalently `vendor_adj[-1] ≈ raw_close[-1]`) within
   `(rtol, atol)`. The last bar is post-all-events, so a correctly back-adjusted vendor series has
   `adj == raw` there. This catches a **globally mis-scaled** series (e.g. `adj_close` in cents vs
   dollars) that a scale-removing normalization would silently accept — the bar-schema requires
   `adj_close` to be a price in the same units as `close`.
4. **Shape assertion:** `numpy.allclose(factor, v, rtol=rtol, atol=atol)` element-wise. On any
   mismatch (anchor or shape) raise `ValueError` naming the offending date(s) with expected (engine)
   vs actual (vendor) factor (same fail-closed shape as the importer's existing parity guards).

Because the comparison is against **real event-derived factors**, reverse splits **pass** — the
exact false-reject the naive #148 ratio/monotonicity checks could not avoid. Comparing the
**factor** (not absolute price) with a **combined `atol + rtol`** keeps the tolerance well-behaved
near `1.0` and sidesteps the #148 "tolerance scales as ¢/price" fragility (a single relative tol on
price is simultaneously too tight for cheap bars and too loose for expensive ones; a relative tol on
a factor near 1.0 needs the `atol` companion so low-priced/small-dividend steps don't false-reject).

**Sensitivity, stated honestly:** split steps are exact (pure ratios) → high-confidence detection of
wrong-magnitude or misaligned steps. Dividend steps depend on vendor dividend convention (rounding,
close vs. VWAP basis), so they are validated *within `(rtol, atol)`*. The validator is therefore a
sound **gross-error / torn-or-shifted-file / wrong-units detector**, not a penny-level
dividend-parity certifier — exactly the #148 failure mode it must catch. The tolerances are tunable
when the validator is wired to a real events feed (#150), which is when they can first be calibrated.

## Fail-closed guards (raise `ValueError`)

- `Split.ratio` non-finite or `≤ 0`; `Dividend.cash` non-finite or `≤ 0`; tz-naive `ex_date` — in
  `__post_init__`.
- `ex_date` not UTC midnight (carries a time-of-day) — in the 1d engine, raised rather than floored.
- Dividend where `P_prev − Σcash ≤ 0` (dividend ≥ prior close). Message is actionable: distinguishes
  a possible liquidating/return-of-capital distribution (exclude it from the event list) from a data
  misalignment (check alignment) — the engine models only ordinary cash dividends, so such an event
  must not be smuggled in as a `Dividend`.
- `raw` missing `ts`/`close`; `ts` not strictly ascending or has duplicates; `close` non-finite
  (NaN/±inf) or `≤ 0`.
- No silent coercion, no NaN-fill.

**Modeled scope:** ordinary forward/reverse splits and ordinary cash dividends only. Spin-offs,
rights, returns of capital, and other special distributions are **not** modeled; the #150 adapter
must classify and reject (or separately handle) them rather than coercing them into `Dividend`.

## Placement & boundaries

`algua/data/corpactions.py`, pandas + numpy only, importing no other `algua` module — a pure leaf
like `algua.data.schema`. No new import-linter contract is required (the existing
`features`/`provenance` purity contracts are unaffected); `lint-imports` stays green. No protected
files touched.

## Out of scope (deferred)

- Wiring into any importer (FirstRate has no events; Databento #150 computes adj from raw+events).
- Persisting `adj_factor` (committed direction above: sidecar artifact in #150, not a bar-schema
  column).
- Intraday timeframes / session mapping (#151); a corporate-action *dataset* ingestion path;
  multi-symbol batching (callers loop per symbol, matching the per-symbol importer seam).
- Special-distribution taxonomy (spin-off / rights / ROC) — #150 adapter scope.

## Testing (TDD, pure unit tests)

`tests/test_corpactions.py`:

- **Engine math:** 2:1 split; single dividend (factor & `adj_close` vs hand-computed values);
  **reverse split** (the #148 regression — `adj/raw > 1` historically, must compute, not reject);
  split + dividend on different dates; multiple stacked events; **same-ex-date two dividends** (must
  equal `(P−(D1+D2))/P`, *not* the cross-term product); **same-ex-date split + dividend** (cash in
  raw/pre-split units); **P_prev sitting between two prior events** (a split between P_prev's bar and
  the dividend ex-date — confirms `P_prev` is raw, split not double-applied); no-event identity.
- **Anchor / range:** event before the first bar (no-op); event **after the last bar** (no-op,
  `adj_close[-1] == close[-1]`); empty bars; dividend `ex_date` on a non-trading day (resolves to the
  prior bar).
- **Guards:** `ratio ≤ 0`, `ratio = inf/nan`, `cash ≤ 0`, `cash = inf/nan`, tz-naive `ex_date`,
  **non-midnight `ex_date` (time-of-day) raises**, dividend ≥ prior close (actionable message),
  unsorted/dup `ts`, non-positive/NaN/±inf `close` → each raises.
- **Validator:** consistent vendor series passes; reverse-split vendor series passes; torn/shifted
  vendor series (off-by-one rows) rejects; wrong-magnitude split step rejects; **globally mis-scaled
  series (cents vs dollars) rejects** (anchor assertion); low-priced stock with a small dividend
  within `atol` passes (no false-reject); mismatched/dup/NaN/±inf index or value inputs raise.

Gate: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`.
