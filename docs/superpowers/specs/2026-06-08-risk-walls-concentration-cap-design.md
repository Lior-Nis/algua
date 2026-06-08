# Risk Walls: Single-Name Concentration Cap + Explicit Long/Short Contract Field

**Issue:** #135
**Date:** 2026-06-08
**Status:** Design approved; GATE 1 (Codex design review) folded in.

## Problem

Strategy-design risk has a hard gap. The only weight-level rails today are **gross
exposure** and an **undeclared long-only convention**, both enforced on decision-time
weights with backtest↔live parity. Two holes follow:

1. **No single-name concentration cap.** Gross caps the *sum* of |weights|, so an
   agent's `compute_weights` can put **100% in one symbol** (gross = 1.0, long-only
   satisfied) and pass every check in backtest and live. Real capital risk, zero rail.
2. **Long-only is enforced but undeclared.** `check_long_only` raises unconditionally;
   there is no `ExecutionContract` field for it. `max_gross_exposure` is a declared,
   hashed field; long-only is an invisible convention. A long/short strategy is silently
   impossible, with no opt-in.

### Dividing principle (why this is code, not kb)

The agent authoring strategies is fallible/injectable. Any limit whose violation can
**lose money or corrupt research** must be a wall the agent cannot bypass by writing
clever weights — enforced in code, identically in backtest and live. Soft heuristics go
to the kb (sibling issue).

## Scope

**In scope**

- `max_weight_per_symbol: float = 1.0` — declared, hashed `ExecutionContract` field. A cap
  on `|weight|` per symbol (works long and short). Default `1.0` = no cap, preserving
  today's behavior.
- `allow_short: bool = False` — declared, hashed field gating the long-only check. Default
  `False` preserves today's long-only behavior.
- A single **shared decision-weight validation bundle** that all three decision sites call,
  replacing today's scattered per-site checks (folded in from the Codex design review).
- A **fail-closed finite-weight check** so non-finite values can't be silently flattened.
- Enforcement with backtest↔live parity (decide / per-bar loop / vectorized fast-path).
- Hash tests proving both new fields change `config_hash` (→ live re-approval).

**Out of scope (deferred, tracked)**

- **Turnover ceiling** — path-dependent (needs prior weights), doesn't fit the stateless
  per-bar weight-vector check and would complicate the fast-path parity guard.
- **Realized per-symbol cap in live** — drift / fractional rounding / partial fills can
  leave the realized broker book over a per-symbol cap. Live-only (no backtest parity),
  mirrors the existing `check_gross_exposure_realized` layer; separable. Follow-up.
- **Full (non-sampled) fast-path parity / restricting `panel_fn` in promotion paths** —
  `_assert_parity` samples ≤ 16 bars (`_PARITY_SAMPLE`), so a malicious `panel_fn` could
  hide an over-cap bar on an unsampled position. This is a **pre-existing** weakness that
  the concentration cap *inherits* (it rides the same guard as gross/long-only today);
  Issue 135 does not worsen it. Follow-up.

## Decisions (with rationale)

| Decision | Choice | Why |
|---|---|---|
| `max_weight_per_symbol` default | `1.0` (no cap) | No forced re-declaration / re-sign of existing strategies. Protection is that the limit is now **declared + hashed**; live go-live requires a human signature over the full contract hash, so the cap value is in what a human signs. Backtest has no capital risk. |
| Non-binding-cap go-live warning | **Not added** | The mandatory go-live signature already puts the cap value in front of a human. A separate warning is redundant for the threat model. |
| Cap semantics | `|weight| <= cap + WEIGHT_TOL` | One absolute-value rule covers long and short. A `-1.0` short with cap `1.0` passes (intentional 100% single-name short); `-0.6` with cap `0.5` breaches. |
| `allow_short` placement | Policy **inside** the shared bundle, not call-site `if` branches | A naked `if not allow_short:` at three sites is a parity surface that drifts. One bundle = one decision. |
| Three-site duplication | **Extract one shared bundle** all sites call | Adding a fourth check to three hand-maintained sites is how parity bugs reappear (Codex HIGH). One function = single source of truth. |
| `config_hash` | No change | `config_hash` serializes `asdict(strategy.execution)`; new fields fold in automatically. Locked by a test. **No compat shim** — existing live authorizations correctly invalidate and must re-sign. |

## Design

### 1. `ExecutionContract` (`algua/contracts/types.py`)

Two new fields on the frozen dataclass:

```python
max_weight_per_symbol: float = 1.0   # cap on |weight| per symbol; 1.0 = no cap
allow_short: bool = False            # False = long-only (today's behavior)
```

`__post_init__` gains: `max_weight_per_symbol > 0` (a zero/negative cap is nonsensical →
`ValueError`). `allow_short` needs no validation.

### 2. Risk checks (`algua/risk/limits.py`)

Keep small, individually-unit-testable check functions; compose them into one bundle that
is the **single call surface** for all decision paths.

```python
def check_finite_weights(weights, strategy_name) -> None:
    # Fail-closed. Reject non-finite VALUES (NaN or ±inf), non-numeric dtype, and
    # duplicate symbol indices → RiskBreach(kind="non_finite_weight").
    # See "Finite-weight semantics" below for the NaN/panel nuance.

def check_short_policy(weights, allow_short, strategy_name) -> None:
    # When allow_short is False, reject any negative weight (today's check_long_only).
    # When True, no-op. kind="long_only".

def check_max_weight_per_symbol(weights, max_per_symbol) -> None:
    # Reject any |weight| > max_per_symbol + WEIGHT_TOL.
    # kind="max_weight_per_symbol". Empty series = no-op (matches existing checks).

def validate_decision_weights(weights, contract, strategy_name) -> None:
    # The shared bundle, in order: finite -> short policy -> per-symbol cap -> gross.
    # The ONE function every decision path calls.
```

`check_gross_exposure` is unchanged and called last inside the bundle.
`check_short_policy` replaces the old `check_long_only` (renamed for what it now does;
no compat alias).

### 3. Finite-weight semantics (the parity nuance)

The two decision surfaces have **different NaN contracts**, so the finite check must be
applied with care:

- **Loop / live `decide()`** receive `strategy.target_weights(view)` — a Series naming
  symbols the strategy chose to weight. A `NaN`/`inf` there is an anomaly (the author
  named a symbol but gave it no real value) → **breach**. Today it is silently flattened
  by a downstream `fillna(0.0)` or skipped by `NaN < 0` / NaN-skipping `.sum()`.
- **Fast-path `panel_fn`** returns a dense matrix where **omitted cells are `NaN` by
  documented convention = flat** (`engine.py:162` `reindex(...).fillna(0.0)`). Here `NaN`
  is structural absence, not a bad value — it must stay "flat", not breach. But `inf` is
  **not** removed by `fillna` and survives into the weight matrix, so a non-finite *value*
  can still reach a position.

**Rule:** non-finite *values* (`±inf`), non-numeric dtype, and duplicate indices are a
breach in **all** paths. `NaN` is a breach where it represents a named-symbol value (loop /
decide) and is honored as flat where it is the panel's absence sentinel (fast-path, after
`fillna`). The bundle runs on the per-bar **decision** Series before any `fillna` in the
loop/decide paths; the fast-path applies the finite-value guard to its per-row non-zero
weights (catching `inf` that survived `fillna`) while preserving NaN-as-flat. A dedicated
test pins all three behaviors (target_weights NaN → breach; panel-omitted cell → flat; any
path producing inf → breach).

### 4. Enforcement sites (backtest↔live parity)

All three replace their two ad-hoc checks with a single `validate_decision_weights(...)`
call on the decision weights:

1. `algua/live/paper_loop.py::decide` — shared live + paper core.
2. `algua/backtest/engine.py::_decision_weights` — per-bar loop. `RiskBreach → BacktestError`
   wrapping unchanged.
3. `algua/backtest/engine.py::_decision_weights_fast` — vectorized fast-path. Same wrapping;
   covered by `_assert_parity` (within its sampled limitation, noted above).

`config_hash` (`algua/strategies/base.py`) needs **no change** — `asdict` picks up the new
fields. In live, `RiskBreach` (any new `.kind`) propagates through the existing
kill-switch/flatten path; verify the live CLI dispatches on `.kind` generically rather than
an allowlist (confirm during build).

## Testing

**Unit (`tests/test_risk_limits.py`)**
- per-symbol cap: pass at/under cap; breach over cap; for a **long** and a **short** position.
- `check_short_policy`: `allow_short=False` rejects a negative weight; `allow_short=True`
  permits it.
- `check_finite_weights`: NaN value → breach; `inf` → breach; duplicate index → breach;
  non-numeric → breach; clean series → pass.
- `validate_decision_weights`: ordering — a vector that breaches multiple rails surfaces the
  documented first one.
- `ExecutionContract.__post_init__`: `max_weight_per_symbol <= 0` → `ValueError`.

**Parity (`tests/test_decision_parity.py`)**
- a weight vector busting the per-symbol cap fails **identically** in backtest (loop) and
  paper.
- `allow_short=True` lets a short through in **both** paths; `False` rejects in both.
- finite semantics: `target_weights` NaN → breach in both loop and paper; a `panel_fn` that
  omits a cell → that symbol is flat (no breach); a path producing `inf` → breach.

**Identity (`tests/` — config_hash)**
- contracts differing only in `max_weight_per_symbol` produce different `config_hash`.
- contracts differing only in `allow_short` produce different `config_hash`.

## Quality gate

`uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports` green.
Enforcement touches `algua/backtest/engine.py` (CODEOWNERS-protected) + the shared decide
path → **human review required**.
