# Streaming-funnel FDR: Phase 2 — LORD++ online alpha-wealth ledger (issue #220)

**Status:** design — for GATE-1 review before coding.
**Date:** 2026-06-16
**Parent spec:** `docs/superpowers/specs/2026-06-15-streaming-funnel-multiple-testing-issue-211-design.md`
**Protected walls touched:** `algua/research/gates.py`, `algua/registry/promotion.py` (CODEOWNERS `@Lior-Nis`).

---

## Problem

Phase 1 (#211, merged) added a **calibrated per-strategy DSR confidence** check as a tighten-only
AND-check beside the deflated-Sharpe haircut. It deliberately exposes a *confidence*
`dsr_confidence ∈ [0,1]` (not a p-value).

Phase 1 is *per-strategy* — each hypothesis is evaluated in isolation against its own DSR threshold.
It does not account for the **multiplicity across the whole stream of auditions**. As the #211 umbrella
spec notes: "discoveries replenish the budget; dry spells tighten it." This is the job of an online
FDR procedure.

---

## Scope

This document designs **Phase 2** of the #211 umbrella spec: online hierarchical FDR accounting
over the strategy-research-promote stream, using **LORD++** (Javanmard & Montanari, 2018).

- **LORD++ first.** Conservative and valid for never-fixed-N streaming. Plain BH needs a fixed N
  so it does not fit the funnel. SAFFRON (Foster & Stine, 2019) has better power once p-value
  calibration is auditable — deferred to Phase 3 or a named follow-up.
- **Builds on Phase 1's calibrated evidence.** The LORD++ p-value is `p = 1 − dsr_confidence`.
  Phase 1 deliberately exposes a *confidence*; the conversion is explicit at the interface to avoid
  re-introducing the `≥`/`≤` inversion hazard GATE-1 caught in Phase 1.
- **Single global stream.** Per-thesis-family budgets + anti-gaming are Phase 4.
- **FDR here is an operating target, not a proof.** Every hypothesis reuses the same holdout/market
  regimes, so the trials are correlated; no textbook FDR guarantee holds cleanly. `candidate` is not
  capital; the paper/forward/live gates remain the hard guards downstream.

---

## LORD++ procedure

### Inputs

| Symbol | Meaning |
|--------|---------|
| `α` | Stream-wide FDR target (`LORD_FDR_ALPHA`). |
| `W₀` | Initial alpha-wealth (`LORD_W0`), `0 < W₀ ≤ α`. |
| `γ` | Non-increasing spending sequence `{γⱼ}`, `γⱼ ≥ 0`, `Σγⱼ = 1`. |
| `t` | 1-based stream index of the current test. |
| `τ₁ < τ₂ < …` | Indices of past rejections (discoveries) in the stream. |

### Test level at time `t`

```
α_t = γ_t · W₀
    + (α − W₀) · γ_{t−τ₁} · 𝟙[τ₁ < t]
    + α · Σ_{j≥2, τⱼ<t}  γ_{t−τⱼ}
```

The first two terms handle the **"++" correction**: because the initial `W₀` has already been
spent via the `γ_t·W₀` term, the first discovery only tops up by `α − W₀` (not `α`). All
subsequent discoveries earn the full `α`. This is the precise formula from Javanmard & Montanari
(2018) Theorem 2.

**Reject (= discovery)** iff `p_t ≤ α_t`. Rejection replenishes the budget (raises future `α`);
non-rejections let `γ` decay the future test level.

**Note on `t`:** `t` is the stream index of the *current* test (i.e., the next unoccupied slot in
the ledger, equal to the row count + 1). The gap terms `γ_{t−τⱼ}` use the 1-based offset from
each past discovery to the current test.

### Spending sequence

Polynomial decay:

```
γⱼ = C · j^{−p},   p = LORD_GAMMA_EXPONENT = 1.6,   C = 1 / ζ(1.6)
```

Where `ζ` is the Riemann zeta function (`scipy.special.zeta(1.6, 1)`). This gives
`C ≈ 0.4375`, ensuring `Σγⱼ = 1` exactly. The sequence is non-increasing (required by LORD++
validity) and decays faster than `j⁻¹` (required for `Σγⱼ` to converge). Pinned by a test:
`Σ_{j=1}^{10^6} γⱼ ≈ 1 ± 10⁻⁶` and `γ` is strictly decreasing.

Alternative (`onlineFDR` package default: `log(max(j,2))/(j·e^{√log j}`) has marginally better
empirical power but is opaque — deferred.

### Recommended constants

| Constant | Value | Rationale |
|----------|-------|-----------|
| `LORD_FDR_ALPHA` | `0.10` | Stream-wide FDR target. `candidate` is not capital; downstream paper/forward gates filter further. Distinct from Phase 1's per-test `DSR_ALPHA = 0.05`. |
| `LORD_W0` | `0.05` | Initial wealth = `0.5 · α`. Bootstrap: first test faces `α₁ = γ₁·W₀ = (1/ζ(1.6))·0.05 ≈ 0.0219` ⇒ `dsr_confidence ≥ 0.978`. Conservative on a cold ledger. |
| `LORD_GAMMA_EXPONENT` | `1.6` | Industry-standard polynomial exponent for LORD++ spending sequences. |

All three are **protected constants in `gates.py`**.

---

## p-value derivation from Phase 1

`p_t = 1 − dsr_confidence`

This conversion is explicit and co-located with the LORD++ computation in `evaluate_gate` (not
in the caller), to prevent the `≥`/`≤` inversion hazard. The conversion is only reached when
`dsr_confidence` is finite; `None` is handled by the binding/omit rules below.

---

## Relationship to Phase 1 (tighten-only invariant)

LORD++ is appended as **another binding AND-check** when binding. The tighten-only invariant:

```
new_pass == old_pass AND (NOT lord_binding OR lord_pass)
```

Phase 1's haircut and DSR checks are byte-for-byte unchanged. Note that LORD++'s `α_t` can
exceed `0.05` after many discoveries (budget replenishment), so the LORD++ check is not
individually stricter than the fixed Phase-1 `DSR_ALPHA` check — but ANDing it in is still
tighten-only relative to the existing gate.

---

## Binding rules (actor-independent, mirror Phase 1)

| Condition | `lord_binding` | Slot consumed? | `lord_skip_reason` |
|-----------|---------------|---------------|---------------------|
| DSR binds (`breadth.provenance == "measured"`) AND `dsr_confidence` is finite | `True` | Yes — one ledger row written | — |
| DSR omitted (no measured dispersion — human declared-breadth path) | `False` | No | `"no_dsr_pvalue"` |
| DSR binds but `dsr_confidence is None` (degenerate holdout / NULL-stats old rows) | `False` | No | `"dsr_failed_closed"` |

The third case deserves emphasis: when `dsr_confidence is None`, the DSR check already fails
closed (blocking promotion). LORD++ is **not** invoked and **no slot is consumed** — a degenerate
or missing-stats audition is not a valid trial in the FDR stream. There is no gaming benefit
(the gate already fails) and no distortion of the stream index.

**No agent override; no flag to relax a failing LORD++ check in Phase 2.** Any future human
escape hatch is an explicit audited flag — deferred, consistent with the gate philosophy.
(`candidate` is not capital and LORD++ is tighten-only; missing an override is safe.)

---

## Persistent alpha-wealth ledger (new durable state)

New table `alpha_wealth_ledger` in `algua/registry/db.py` `_SCHEMA`. The ledger is the
**immutable, auditable record** the issue calls for — each row freezes `α_t`, `p_t`, and the
`rejected` flag at evaluation time so accounting is stable even if constants are later changed.

```sql
CREATE TABLE IF NOT EXISTS alpha_wealth_ledger (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    stream_index        INTEGER NOT NULL UNIQUE,         -- t (1-based, dense, monotonic)
    strategy_id         INTEGER NOT NULL REFERENCES strategies(id),
    gate_evaluation_id  INTEGER NOT NULL REFERENCES gate_evaluations(id),
    p_value             REAL    NOT NULL,                -- 1 − dsr_confidence (finite)
    alpha_level         REAL    NOT NULL,                -- α_t the test faced
    rejected            INTEGER NOT NULL,                -- 1 if p_t ≤ α_t, else 0
    fdr_alpha           REAL    NOT NULL,                -- LORD_FDR_ALPHA at evaluation time (audit)
    w0                  REAL    NOT NULL,                -- LORD_W0 at evaluation time (audit)
    created_at          TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_alpha_wealth_ledger_rejected
    ON alpha_wealth_ledger (rejected);
```

**Key design decisions:**

- `stream_index UNIQUE` — the UNIQUE constraint is the concurrency guard (see below).
- `gate_evaluation_id NOT NULL REFERENCES gate_evaluations(id)` — every ledger row is linked
  to exactly one gate evaluation row, written atomically in the same transaction.
- `rejected` stores the binary outcome (`p_t ≤ α_t`), not the overall promotion decision.
  Promotions are a strict subset of LORD++ rejections (other gate checks may still fail).
- `fdr_alpha` and `w0` columns audit the constants in force, so future constant changes do
  not retroactively invalidate the accounting.

`SCHEMA_VERSION` bump **25 → 26** (`SCHEMA_VERSION=25` was claimed by #219's `factor_evaluations`
table). The new table is created by `executescript(_SCHEMA)` in `migrate()`; no
`_add_missing_columns` call needed.

---

## Atomicity and concurrency

The registry is multi-process (WAL + `busy_timeout=5000`, per #164). The ledger row and the
gate-evaluation row must be written together in one transaction; the `stream_index` must be
assigned exactly once, robust against concurrent `research promote` processes.

The holdout burns in `walk_forward` *before* the ledger section, in its own top-level
`BEGIN IMMEDIATE` (from #161). The ledger write is a **separate** top-level critical section —
no nesting.

### `run_gate` flow when `lord_binding=True`

(After `walk_forward`; the holdout is already finalized.)

1. **Read state:** `next_index, rejection_indices = repo.lord_state()` — reads
   `max(stream_index) + 1` (or `1` if the ledger is empty) and the ordered list of indices where
   `rejected = 1`.
2. **Compute level:** `α_t = gates.lord_alpha_level(next_index, rejection_indices, alpha, w0)` —
   pure function in `gates.py`.
3. **Derive p-value and outcome:** `p_t = 1.0 − dsr_confidence`; `lord_pass = (p_t ≤ α_t)`.
4. **Thread into `evaluate_gate`:** pass `lord_binding=True`, `lord_alpha=α_t`, `lord_p_value=p_t`
   → `evaluate_gate` appends
   `{"name":"fdr_lord","value":p_t,"threshold":α_t,"op":"<=","passed":lord_pass}` to the AND-set.
5. **Atomic write:** `repo.record_gate_evaluation_with_lord_trial(...)` — one transaction:
   a. `INSERT INTO gate_evaluations(...)` → get `gate_eval_id`.
   b. `INSERT INTO alpha_wealth_ledger(stream_index=next_index, ..., gate_evaluation_id=gate_eval_id)`.
   c. If the `INSERT` to `alpha_wealth_ledger` raises `UNIQUE` on `stream_index` (a concurrent
      promote claimed it first): **retry** from step 1 (bounded loop, e.g. ≤ 5 retries). The retry
      re-reads the ledger state, recomputes `α_t` for the new index, and may re-evaluate the gate
      (since `α_t` changes). The gate_evaluations INSERT from step 5a is NOT committed on a conflict;
      the whole transaction rolls back cleanly before the retry.
   d. `conn.commit()` on success.

This mirrors the `rowcount != 1` compare-and-swap pattern (Pattern A / #161) already used for
gate token consumption and allocation revokes.

### Why not a separate `BEGIN IMMEDIATE` read-check-write?

`BEGIN IMMEDIATE` (Pattern B) is the #161 holdout-reservation pattern, used when the read and write
must form an atomic predicate check on a *mutable state machine* (e.g., "no overlapping holdout
interval"). Here the invariant is structurally simpler: `stream_index UNIQUE` is a database-level
constraint; a concurrent writer that claims the same slot will fail the INSERT atomically at the DB
layer without explicit row locking. The retry-on-conflict approach (Pattern A variant) is sufficient
and simpler.

---

## Documented Phase-2 limitations

The same honest-caveat discipline as Phase 1:

1. **Crash window.** The holdout burns in `walk_forward` *before* the ledger write. A crash in
   between leaves an un-counted peek (a missing acceptance), making future `α_t` marginally more
   lenient (the stream index will be one lower than expected). Acceptable within the "operating
   target, not a proof" caveat. A two-phase reserve→finalize coupling (like the holdout) is deferred.
2. **Pure-LORD++ subset.** Promotions ⊆ LORD++ rejections (the other gate checks further restrict).
   There is no clean theorem that `FDR(promotions) ≤ α`, but the extra gate checks are
   truth-independent conservative filters — defensible and conservative; the shared-holdout disclaimer
   already stands.
3. **Single global stream.** Per-thesis-family budgets + anti-gaming (Phase 4) build on #137
   (breadth by family-id), #122 (family metadata), and the factor-lineage work (#140).

---

## Footprint (six files; two protected)

| File | Protected | Change |
|------|-----------|--------|
| `algua/research/gates.py` | **yes** | Pure `lord_alpha_level(t, rejection_indices, *, alpha, w0) -> float`; `_lord_gamma(j)` spending-sequence helper; protected constants `LORD_FDR_ALPHA = 0.10`, `LORD_W0 = 0.05`, `LORD_GAMMA_EXPONENT = 1.6`. `evaluate_gate` gains `lord_binding: bool = False`, `lord_alpha: float \| None = None`, `lord_p_value: float \| None = None` kwargs; when `lord_binding=True`, appends the `fdr_lord` check. New `GateDecision` audit fields (`lord_binding: bool`, `lord_alpha: float \| None`, `lord_p_value: float \| None`, `lord_stream_index: int \| None`, `lord_skip_reason: str \| None`); nulled-when-not-finite in `to_dict()`. |
| `algua/registry/promotion.py` | **yes** | `run_gate`: compute `p_t`; read `(next_index, rejection_indices)` from `repo.lord_state()` when `lord_binding=True`; compute `α_t` via `gates.lord_alpha_level`; thread into `evaluate_gate`; call `repo.record_gate_evaluation_with_lord_trial(...)` (replacing the existing `record_gate_evaluation` call when binding); include retry loop on `stream_index` conflict. Thread audit fields into the JSON payload. |
| `algua/registry/db.py` | no | `alpha_wealth_ledger` table + index in `_SCHEMA`; `SCHEMA_VERSION 25 → 26`; comment `# v26 (#220): alpha_wealth_ledger`. |
| `algua/registry/store.py` | no | `lord_state() -> tuple[int, list[int]]` (next index, rejection indices). `record_gate_evaluation_with_lord_trial(...)` — one `with self._conn:` transaction: INSERT gate row, INSERT ledger row, let `UNIQUE` conflict surface (caller handles retry). `record_gate_evaluation` stays for the non-binding case. |
| `algua/registry/repository.py` | no | Add `lord_state` and `record_gate_evaluation_with_lord_trial` to the `StrategyRepository` Protocol. |
| `algua/cli/research_cmd.py` | no | Surface `lord_*` audit fields in the `research promote` JSON payload. No new flags. |

`scipy.special.zeta` is needed for `C = 1/ζ(LORD_GAMMA_EXPONENT, 1)` (a one-time constant
computation; can be done at module import time). `scipy` is already an explicit dep (added by
Phase 1). `scipy.special` is importable; `zeta` is available (`scipy.special.zeta(s, a)` =
Hurwitz zeta, so `zeta(p, 1) = Σ_{n≥1} n^{-p}`). Pinned by a test.

---

## Edge cases (all fail-closed or handled)

| Case | Handling |
|------|---------|
| Cold ledger (no prior tests) | `lord_state()` returns `(1, [])` → `α₁ = γ₁·W₀` (only the first term) |
| `dsr_confidence is None` | LORD++ omitted (no slot), skip reason `"dsr_failed_closed"` — DSR already fails closed |
| `dsr_confidence = 1.0` | `p_t = 0.0` — always a rejection; ledger records correctly |
| `dsr_confidence = 0.0` | `p_t = 1.0` — never a rejection when `α < 1`; DSR check fails too |
| `t` and `τ_j` offsets: `t = τ_j` | Cannot happen — τ_j is a past rejection, all < current t |
| `γ_j` at very large j (j > 10^7) | Power-law decay → effectively 0; no overflow |
| Concurrent promote claims same slot | `UNIQUE` conflict → transaction rolls back → retry recomputes α_t for new slot |
| ≤ 5 retries exhausted | Raise `RuntimeError` (very unlikely with `busy_timeout=5000`; logged for ops) |
| `scipy.special.zeta` unavailable | Caught at import time with a clear error (scipy is a required dep) |

---

## Testing (for the TDD plan)

1. **Pure `lord_alpha_level` unit tests:**
   - Cold ledger: `t=1`, no rejections → `α₁ = γ₁·W₀ ≈ C·1^{-1.6}·0.05`.
   - First rejection at `τ₁=1`, then `t=2`: `α₂ = γ₂·W₀ + (α−W₀)·γ₁`.
   - Second+ rejections earn full `α`: `τ₂=2`, `t=3`: third term `α·γ₁`.
   - Dry spell lowers `α_t` (reference: after 50 non-rejections, `α₅₁ < α₁`).
   - Discovery raises next `α` vs dry-spell counterfactual.
   - Reference values pinned by hand (compute with `scipy.stats.norm.ppf`/`scipy.special.zeta`).

2. **Spending-sequence properties:**
   - `Σ_{j=1}^{10^6} γⱼ ≈ 1 ± 10⁻⁶`.
   - `γ` strictly decreasing.
   - `γ₁ = C ≈ 0.4375` (the annualized claim).

3. **`evaluate_gate` integration (tighten-only, strong form):**
   - Over a generated grid of gate decisions, assert
     `new_pass == old_pass AND (not lord_binding or lord_pass)`.
   - Binding: appends `fdr_lord` check; `passed = all(...)` includes it.
   - Non-binding (no measured dispersion): no `fdr_lord` check appended; `passed` unchanged.
   - `dsr_confidence None` + binding: → omit LORD++ (skip reason = `"dsr_failed_closed"`); no `fdr_lord` check.

4. **Ledger / store tests:**
   - Cold `lord_state()` → `(1, [])`.
   - After one rejection: `lord_state()` → `(2, [1])`.
   - `record_gate_evaluation_with_lord_trial` is atomic: both rows present on commit; neither on rollback.
   - Forced `UNIQUE` conflict on `stream_index` → `IntegrityError` (or equivalent) surfaces cleanly for the caller retry.
   - Concurrent two-process test (mirror #164): after N parallel promotes, `stream_index` values are dense (`{1, …, N}`) with no duplicates.

5. **Binding rule tests (promotion integration):**
   - Measured-dispersion path: `lord_binding=True`, slot consumed.
   - Declared-breadth (human) path: no slot consumed; `lord_skip_reason="no_dsr_pvalue"`.
   - `dsr_confidence=None` path: DSR fails closed, no slot; `lord_skip_reason="dsr_failed_closed"`.
   - Agent `research promote` end-to-end: second audition gets a different `α_t` than first;
     a discovery on the first raises the second's `α_t` above baseline.

6. **Schema migration (extend `tests/test_db_migrations.py`):**
   - Fresh DB → `PRAGMA user_version = 26`; `alpha_wealth_ledger` table present.
   - Second `migrate(conn)` is a no-op (idempotent).
   - Pre-v26 DB (with `factor_evaluations` but no `alpha_wealth_ledger`) → migrate → table added, version 26.

---

## Quality gate

`uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports` green.

---

## Deferred (Phase 3+)

- **Dependence-aware calibration (Phase 3):** effective independent trials from return-stream
  correlation (replaces raw-count `N` in the DSR benchmark); block/stationary bootstrap nulls;
  multi-regime robustness; dispersion floor from the `(count, mean, var)` triples Phase 1 records.
- **Hierarchical family budgets + anti-gaming (Phase 4):** global alpha budget above per-thesis-family
  budgets; empirical clustering by return-correlation / holdings / factor lineage (#140) / code
  ancestry. Builds on #137 (breadth by family-id), #122 (family metadata), #161/#192/#193/#205
  (holdout single-use + identity), #140 (factor lineage).
- **Retire the haircut (Phase 5/end-state):** once DSR + LORD++ calibration is audited under Phases
  2–3, retire the deflated-Sharpe haircut so the system does not carry two redundant multiplicity
  penalties forever.
- **SAFFRON (after Phase 3):** once p-value calibration is auditable, SAFFRON has better power than
  LORD++ for a stream where discoveries are expected. Switching is non-breaking (the same ledger
  shape is reusable).
- **Crash-window fix (deferred):** two-phase reserve→finalize for the ledger write (mirroring the
  holdout #161 pattern), so a crash between the holdout burn and the ledger INSERT does not leave an
  un-counted slot.
