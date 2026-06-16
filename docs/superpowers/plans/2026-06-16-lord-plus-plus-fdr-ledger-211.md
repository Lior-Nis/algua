# LORD++ Online FDR Alpha-Wealth Ledger — Issue #211 Phase 2

**Date:** 2026-06-16
**Issue:** #220 (spec #211 Phase 2)
**Schema:** 24 → 25
**Merged:** (this PR)

## Problem

Phase 1 (#218, schema v24) added a per-strategy DSR evidence check beside the deflated-Sharpe
haircut. This controls *per-strategy* selection inflation but does nothing about **funnel-wide
multiplicity**: testing thousands of hypotheses against a shared holdout inflates false discoveries
no matter how tight each individual test is.

## Solution

A persistent **LORD++ alpha-wealth ledger** governing the `backtested → candidate` gate.

**Algorithm:** LORD++ (Ramdas et al. 2017, Biometrika 104:1). Chosen over BH because the funnel
is never-fixed-N; chosen over BY because BY is brutally conservative.

**p-value:** `p = 1 − dsr_confidence` — P(SR_true ≤ SR*) under the DSR null, stronger than the
simple null SR_true ≤ 0. Converted explicitly at the interface to guard ≥/≤ inversion hazard.

**Parameters:**
- `FDR_ALPHA = 0.05` (operating target, not a formal proof — shared-holdout dependence breaks the
  guarantee; Phase 3/#221 adds dependence-aware recalibration)
- `FDR_W0 = FDR_ALPHA / 2 = 0.025` (initial alpha-wealth)
- `FDR_GAMMA_TRUNCATION = 10_000` — discount sequence normalized over j=1..10_000

**Discount sequence:** `γ_j ∝ log(max(j, 2)) / (j · exp(√(log j)))`, the onlineFDR / Ramdas et al.
default. `max(j, 2)` ensures γ_j > 0 for all j. Sum over truncation ≤ 1.0 + 1e-9.

**LORD++ level formula:**
```
α_t = γ_t · W0
      + (FDR_ALPHA − W0) · γ_{t−τ_1}       (only if τ_1 exists)
      + FDR_ALPHA · Σ_{j≥2} γ_{t−τ_j}      (sum over j≥2 discovery positions)
```

Reject (discovery) iff `p_t ≤ α_t`. Wealth derived from ledger rows on each call — no caching,
same recompute-from-rows philosophy as `pooled_trial_sharpe_var`. Stream of ≤ a few thousand
rows; cheap.

**Discovery rule:** self-contained — `p ≤ α_t` is a "discovery" independent of whether the
strategy actually promotes. The gate ANDs this with other checks, so actual promotions are a
*subset* of FDR rejections. Keeps the LORD++ FDR-control argument valid.

**FDR binding condition:** `dsr_binding=True AND dsr_confidence is not None AND isfinite(dsr_confidence)`.
If DSR is binding but returns None, the promotion fails via DSR; FDR is non-binding for that row.
Declared/human breadth → FDR entirely skipped (non-binding).

## Architecture

### New schema columns (`gate_evaluations`, schema v26)

| Column | Type | Description |
|---|---|---|
| `fdr_binding` | INTEGER NULL | 1 if this row is in the LORD++ stream |
| `fdr_p_value` | REAL NULL | p = 1 − dsr_confidence |
| `fdr_alpha_level` | REAL NULL | α_t at evaluation time |
| `fdr_rejected` | INTEGER NULL | 1 if p ≤ α_t (FDR discovery) |
| `fdr_test_index` | INTEGER NULL | position t in the global stream |

Partial unique index `ix_gate_evaluations_fdr_index ON gate_evaluations(fdr_test_index)
WHERE fdr_binding=1` prevents duplicate stream positions.

### Stream read — `fdr_stream_state()` (store.py)

SELECT `gate_evaluations WHERE fdr_binding=1 ORDER BY id`. Returns `(t, discovery_indices)`.
Fail-closed validation: NULL/non-finite p/alpha → None; fdr_rejected ∉ {0,1} → None;
non-positive fdr_test_index → None; non-contiguous indices → None.

### Atomic write — `record_gate_with_fdr_and_maybe_promote()` (store.py)

Uses **explicit `BEGIN IMMEDIATE`** + manual commit/rollback, mirroring `reserve_holdout`.
Context-manager (`with self._conn:`) starts DEFERRED — a preceding SELECT can slip before
the write lock, enabling double-spend. Top-level guard: `if in_transaction: raise RuntimeError`.

Transaction sequence:
1. `BEGIN IMMEDIATE` (acquires write lock immediately)
2. `fdr_stream_state()` (SELECT — now safe under write lock)
3. `t_next = t + 1`; `α_t = level_fn(t_next, discovery_indices)`
4. `fdr_rejected = p_value ≤ α_t` (discovery rule)
5. `final_passed = provisional_passed AND fdr_rejected` (tighten-only AND-check)
6. INSERT `gate_evaluations` with `passed = final_passed` (never provisional)
7. If `final_passed`: `_apply_transition_locked(rec, CANDIDATE, …)`
8. COMMIT

Crash semantics: crash before commit rolls back both the FDR row and the stage CAS — no orphaned
row, no consumed stream position. Accepted audit gap: holdout is burned before this method; a
crash between holdout-burn and `record_gate_with_fdr_and_maybe_promote` omits that hypothesis from
the FDR stream. Single-use holdout (#192/#193) bounds the attack surface.

### Import boundary

`algua/registry/store.py` receives `level_fn(t, taus) → float` as an injected callable from
`promotion.py`. `promotion.py` constructs `functools.partial(lord_plus_plus_level, alpha=FDR_ALPHA,
w0=FDR_W0)`. This keeps `algua/registry` free of `algua/research` imports (lint-contracts KEPT).

## Tasks completed

1. **Pure LORD++ math** (`algua/research/gates.py`) — `lord_plus_plus_level`, `_lord_gamma_weights`,
   `FDR_ALPHA`, `FDR_W0`, `FDR_GAMMA_TRUNCATION`, `_LORD_GAMMA`.
2. **Schema 25→26** (`algua/registry/db.py`) — 5 FDR columns + partial unique index + migration.
3. **Stream read accessor** (`repository.py` Protocol + `store.py`) — `fdr_stream_state()` with
   fail-closed validation.
4. **Atomic FDR-test-and-maybe-promote** (`store.py`) — `record_gate_with_fdr_and_maybe_promote()`
   with BEGIN IMMEDIATE, top-level guard, and concurrency test (two concurrent binding evals get
   distinct serialized t values).
5. **Gate wiring** (`gates.py` + `promotion.py`) — `GateDecision` FDR fields, `fdr_evidence`
   check (tighten-only AND), `run_gate` routing through atomic composite.
6. **Surface + docs** — FDR fields flow in `decision.to_dict()`, `CLAUDE.md` updated.
7. **Quality gate** — `pytest -q` (1682+ tests green), `ruff`, `mypy algua`, `lint-imports` all
   clean.

## Deferred (Phase 3/#221)

- Effective-independent-trials recalibration
- Bootstrap nulls for shared-holdout dependence
- Dispersion floor and haircut retirement
- SAFFRON, per-family hierarchical budgets (Phase 4/#222)
