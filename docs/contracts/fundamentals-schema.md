# Fundamentals schema (the non-tabular PIT seam)

Tidy/long, bitemporal. One row = one metric value for one issuer/period, stamped with when it
became knowable.

| column | type | meaning |
|---|---|---|
| `symbol` | str (upper-cased, non-null) | issuer ticker |
| `fiscal_period_end` | date (non-null) | the period the figure describes |
| `metric` | str (non-null) | metric name, e.g. `revenue`, `eps_diluted` |
| `value` | float64 (NaN allowed = reported-but-unavailable) | the figure |
| `knowable_at` | tz-aware UTC datetime (non-null) | report availability = filing + lag; the PIT key |
| `source` | str (non-null) | provenance label |

## Point-in-time rule
A record is visible at decision `t` iff `knowable_at <= t`. The backtest engine owns `t` (the bar
timestamp) and masks per bar — the strategy never chooses `t`. Because daily bars are midnight-UTC,
an intraday filing on day D is first visible at the decision for D+1 (conservative; never leaks).

## Restatements
A restatement is a NEW row sharing `(symbol, fiscal_period_end, metric)` with a later `knowable_at`.
The as-of mask keeps, per that key, the row with the greatest `knowable_at <= t` — originally-reported
before the restatement is knowable, restated after.

## Validation floor
`knowable_at >= fiscal_period_end` (UTC midnight) — a sanity floor, not a precise availability model.
Bitemporal key `(symbol, fiscal_period_end, metric, knowable_at)` is unique within a snapshot.

## Two access modes
- **As-of (signal):** `FundamentalsProvider.get_fundamentals` → engine mask → `compute_weights`.
- **Hindsight (analysis):** `algua data query-fundamentals` (full history) — agent post-mortems only,
  structurally walled from the engine (import-linter). The wall is a STATIC import-graph guarantee enforced by `lint-imports` (and an AST test); it is not a runtime sandbox — defending against a strategy that dynamically imports `algua.data.hindsight` (via `importlib`/`__import__`) is deferred to a strategy-purity-hardening follow-up.
