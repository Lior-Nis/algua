# Formalize portfolio construction as a distinct layer: signal → target portfolio (#141)

**Status:** design (GATE-1 round 1 folded) · **Date:** 2026-06-10 · **Issue:** #141

## Problem

A strategy today is a module exposing `CONFIG` + `compute_weights(view, params) → pd.Series`.
That one function does **two** jobs at once: generate the alpha (e.g. momentum scores) **and**
turn it into a risk-shaped target portfolio (top-k, equal-weight, gross-normalized). Established
quant practice separates three concerns — **signal → target portfolio → execution** — and algua
collapses the first two.

The cost shows up at agentic scale. The platform is built for an agent that authors *many*
strategies. With construction fused into every `compute_weights`, each strategy re-implements (and
re-tests, and can subtly get wrong) the same portfolio-construction logic, and #136's weight-space
risk conventions (vol-targeting, exposure caps, turnover control, risk budgeting) have nowhere to
live as a **shared, composable, individually-testable** layer. The execution/risk-overlay layer
already exists (the diff-to-target loop + the #135 hard walls); this issue formalizes the
**signal → portfolio-construction** split that the execution layer sits on top of.

## Decision

Split the authored contract into two layers with one shared evaluation pipeline:

1. **Signal (alpha)** — the strategy authors `signal(view, params) → pd.Series` of cross-sectional
   **scores** per symbol. Higher = more attractive. Scores are *not* weights and carry no gross/sign
   discipline.
2. **Portfolio construction** — a **named policy** selected in `CONFIG`, resolved from a new pure,
   import-enforced library `algua/portfolio/construction.py`. A policy maps `scores → target
   weights` under a reusable risk convention. Policies are shared across strategies and individually
   reviewable/testable.
3. **Execution / risk overlay (unchanged)** — `validate_decision_weights` (the #135 hard walls)
   runs *after* construction as the final rejection wall, single-sourced for backtest + paper + live.

A strategy is therefore **a signal + a chosen construction policy (+ params)**. Both layers fold
into the artifact identity, so each can be swapped/retested independently and a change to either
invalidates a dependent live authorization.

This is **shape A** of the three contract shapes considered (confirmed with the operator): a signal
fn + a config-named policy from a pure library. (B — two authored fns per module — rejected:
construction stops being reusable/shared. C — an injected `PortfolioConstruction` object — rejected:
more machinery than a named-policy dispatch needs.)

### Scope of this slice

Land the **structural seam** with a minimal-but-real construction library and full identity/parity
wiring. The construction contract this slice supports is **stateless, view-aware, cross-sectional**:
`construct(scores, view, params)`. The single-cross-sectional-score signal is a deliberate fit for
**rank/tilt** strategies — the platform's current shape — not a claim of universality (see
*Expressiveness ceiling* below). Explicitly deferred: vol-targeting/turnover/risk-budgeting policies
(#136), the signal *registry* and multi-output/long-short signals (#140), per-layer evaluation CLI
(#137), formal per-policy param schemas, a stateful `ConstructionContext`, and a vectorized
construction fast path for view-dependent policies.

## The signal layer

`algua/strategies/base.py` — type aliases and the authored contract.

- **Canonical signal:** a pure module-level `signal(view, params) → pd.Series` (scores indexed by
  symbol). `view` is the point-in-time bar-schema frame (history up to and including the decision
  bar `t`), exactly as `compute_weights` received it. A symbol may be omitted (no score); a missing
  or non-finite (`NaN`/`±inf`) score means **"no opinion — not selectable"** and is dropped by
  construction before ranking/normalization. It is *never* coerced to `0.0` (a real 0 score and "no
  score" must stay distinct).
- **Fundamentals lane (#132):** a strategy declaring `needs_fundamentals=True` authors
  `signal(view, params, fundamentals) → pd.Series`. The 3rd arg is the PIT-correct as-of frame the
  engine materializes for bar `t` (knowable_at ≤ t), unchanged from today. Distinct callable type so
  the 2-arg and 3-arg forms never silently overload.
- **Optional vectorized acceleration:** a pure module-level `signal_panel(bars, params) →
  pd.DataFrame` returning the **scores matrix** (index=timestamp, columns=symbol; PRE-lag) for the
  whole period in one shot. It is NOT a second definition: the engine uses it only behind the
  fail-closed **weight-level** parity guard (below) and raises on any disagreement.

These replace today's `compute_weights` / `compute_weights_panel`. **Clean break — no
`compute_weights` shim.** Both example strategies migrate.

`LoadedStrategy` stores the **raw** `ConstructFn` (never a params-bound `partial`) and reads
`config.construction_params` at call time, and exposes methods that close over `config.params`:
`signal(view, fund=None)`, `signal_panel(bars)` (or `None`),
`construct(scores, view) = construct_fn(scores, view, self.config.construction_params)`, and
`target_weights(view, fund=None) = construct(signal(view, fund), view)`. Storing the raw fn (not a
bound partial) is load-bearing twice over: a sweep that rebuilds the config with new
`construction_params` takes effect (no stale bound params), and `inspect.getmodule(construct_fn)`
resolves to the policy module for the identity hash (a `partial`/closure would resolve to `functools`).

### Expressiveness ceiling (acknowledged, deferred to #140)

A single cross-sectional score Series expresses **rank/tilt** strategies cleanly and nothing more.
It does *not* express pairs/spread (the leg pairing is lost in a per-symbol scalar), factor-aware
construction (needs per-factor scores), or optimizer/risk-parity strategies whose "signal" is
already a target/risk budget. Those need multi-output signals or a richer construction input and are
**out of scope for this slice** — they are #140's (signal registry / composable factors) territory.
Negative scores remain representable, so a future `long_short_*` policy can map them to ±weights;
the three starter policies are long-biased. This ceiling is documented in the `ConstructFn`/`signal`
contract so an agent hits a clear boundary rather than stuffing portfolio logic back into `signal`.

## The construction layer

New package `algua/portfolio/` with `construction.py`. **Pure** (no I/O; imports only `contracts` +
`features`; must NOT import `strategies` — would cycle), enforced by import-linter like `features`.

### Policy contract

`ConstructFn` is defined **in `algua/portfolio/construction.py`** (not in `strategies.base`, which
would force `portfolio` to import `strategies` and cycle):

```python
ConstructFn = Callable[[pd.Series, pd.DataFrame, dict[str, Any]], pd.Series]
# construct(scores, view, params) -> target weights (indexed by symbol)
```

- `scores` — the signal output for bar `t`. The policy **fails closed on a non-numeric score Series**
  (no silent string→NaN coercion that would hide a strategy bug), then **drops non-finite/missing
  scores**, then selects/normalizes over the finite remainder. Empty (or all-dropped) ⇒ empty weights
  (flat). Selection policies break ties **deterministically by symbol label** — sort by `(score
  descending, symbol ascending)` — so the per-bar `signal` Series order and the `signal_panel` column
  order yield the **identical** selected set (otherwise equal scores could silently diverge between
  the fast path and the loop on bars the bounded parity guard does not sample).
- `view` — the same PIT price frame the signal saw. Passed so a vol-targeting policy (#136), which
  is a pure function of prices, slots in without a contract change. The three starter policies ignore
  it. **NOTE (honest scope):** turnover control (needs prior target/current holdings) and risk
  budgeting (needs covariance + constraints/state) need *more* than `(scores, view)` — a future
  `ConstructionContext`. This slice does **not** claim those slot in for free.
- Returns target weights. The policy shapes sign/gross, but the #135 hard walls remain the
  authoritative rejection layer downstream — an out-of-bounds vector hard-breaches like any strategy.

### Dispatch + load-time validation

```python
_POLICIES: dict[str, _Policy] = { ... }      # id -> (fn, param_validator)
CONSTRUCTION_POLICIES = MappingProxyType(_POLICIES)   # immutable PUBLIC dispatch view

def get_construction_policy(policy_id: str) -> ConstructFn: ...      # ValueError on unknown id
def validate_construction_params(policy_id, params) -> None: ...     # per-policy, at LOAD time
```

The loader resolves `CONFIG.construction` and **validates params at load**, never first-decision.
Validation is **per-policy** (not just generic required-keys): reject an unknown policy id, then run
the policy's own `param_validator` — e.g. `top_k_equal_weight` requires `top_k` to be a **positive
int** (type + domain, short of a full typed schema), and every policy rejects unknown keys. Param
values are checked **recursively** for finite + JSON-serializable (so a nested `NaN` can't slip
through), and `config_hash` serializes with `allow_nan=False` so a non-canonical value fails closed
rather than producing a misleading hash. Policies are defined **statically** in the module (no
dynamic registration, no dynamic imports inside policy bodies); `MappingProxyType` makes the public
dispatch view read-only, and identity rests on the static module source (below) — not on runtime
dispatch state.

### Starter policies

| id | required params | maps scores → weights | replaces |
|---|---|---|---|
| `top_k_equal_weight` | `top_k: int` | drop non-finite; top-`k` by score; equal weight `1/k` | momentum's construction |
| `equal_weight_positive` | — | drop non-finite; equal-weight names with `score > 0` | earnings-tilt's construction |
| `score_proportional_long` | — | drop non-finite; clip negatives to 0; normalize positives to sum 1.0 | (new; a "raw" long-only mapping) |

All three ignore `view`. A weight-producing/bespoke strategy adds a **named** policy to this library
(additions-only) — there is deliberately no generic passthrough that returns scores as weights
(that would re-couple the layers).

## Composition and backtest↔live parity

`LoadedStrategy` (in `base.py`) binds a `StrategyConfig` + the authored fns + the **resolved**
construction callable & params, and is the single place the full pipeline is expressed. Both
consumers call `target_weights` unchanged:

- the backtest engine's per-bar loop (`_decision_weights`) and parity recompute (`_canonical_row`),
- the paper/live shared `decide()`.

Because the *same* `target_weights` composition runs in research and live, the signal→construction
pipeline is identical on both sides — **parity holds for free**. The engine reaches signal/construct
**only through the `LoadedStrategy` surface**; `algua.backtest` never imports `algua.portfolio`
directly. (`backtest → strategies.base → portfolio` is a legitimate indirect chain, so no
"backtest forbidden from portfolio" import rule is added — that would be unsatisfiable.) Parity
depends on construction using only inputs available live; the starter policies use only the `scores`
Series (computed from the same bar-schema `view` both sides have), so they cannot diverge.

## Fast path (CODEOWNERS-protected `engine.py`)

`signal_panel` is the vectorized **signal** twin — the speedup is computing the signal once instead
of recomputing it on the expanding view every bar. **Construction stays per-bar**, and the parity
guard stays at the **weight level** (it is NOT moved to signal-level — score equality within
`WEIGHT_TOL` does not bound the weight error for discontinuous policies like `top_k`, where a near
tie can flip the selected set). `_decision_weights_fast`:

1. call `strategy.signal_panel(bars)` once → the scores matrix (raise `BacktestError` if not a
   `DataFrame`); reindex onto the simulation grid **without filling NaN** (missing score ≠ 0);
2. per evaluated bar `t` (`i ≥ warmup`): build `view_t = bars_sorted.iloc[:stop]` using the **same**
   `bars_sorted` + `end_pos` slicing as the loop, then `w = strategy.construct(scores_row_t, view_t)`
   then `validate_decision_weights(w, ...)` — the same risk walls; warmup bars are held flat by
   **skipping construction** (weights stay 0), matching the loop's `if i < warmup: continue`;
3. the fail-closed **weight-level** parity guard (`_assert_parity`) compares, on a bounded
   deterministic sample of bars, the fast-path weights row against the canonical per-bar
   `strategy.target_weights(view_t)` (= `construct(signal(view_t), view_t)`) with tolerance
   `WEIGHT_TOL`. Any disagreement RAISES `BacktestError`; the fast path is never trusted without it
   and never silently falls back (invariant preserved exactly as today).

The fast path vectorizes only the **signal**; for a future view-dependent construction policy the
per-bar `construct` does real work and the speedup shrinks — a `construct_panel` vectorization is the
deferred answer there. PIT mode and the fundamentals lane still force the per-bar loop (unchanged).

## Identity / live-gate invalidation

A construction-policy change MUST fold into the artifact identity and invalidate dependent live
authorizations. Two independent axes, each covered automatically and **non-bypassably**:

- **`config_hash`** (`base.py`) folds `construction` (policy id) + `construction_params`. Swapping
  the policy or tuning its params — even with identical policy code — changes the config identity.
  (This changes `config_hash` for *every* strategy; any existing live approval is invalidated and
  must be re-recorded. The platform has no live strategies yet, so this is a non-issue in practice.)
- **`code_hash`** (`registry/approvals.py::compute_artifact_hashes`) hashes the first-party import
  closure rooted at the signal module. It is extended to **also root from the literal
  `algua.portfolio.construction` module object** (imported by name — *not* `_POLICIES[id].__module__`,
  whose value is a `_Policy` record, and *not* the params-bound callable, since `inspect.getmodule`
  on a `partial`/closure returns `functools`). Because every policy + the dispatch table + their
  first-party helpers live in that one module, hashing its whole source captures a policy-body edit,
  a helper edit, *and* an id-retargeting — any of which invalidates a prior approval. (Since
  `LoadedStrategy` now stores the raw `construct_fn`, `inspect.getmodule(construct_fn)` would also
  resolve correctly; rooting from the named module is the conservative belt-and-suspenders form.)

`dependency_hash` is orthogonal and unchanged — it pins the third-party `uv.lock` set; the expanded
*first-party* closure does not touch it.

## CONFIG, loader, and sweep changes

- **`StrategyConfig`** gains `construction: str` (required) and `construction_params: dict[str, Any]
  = {}`.
- **`load_strategy`** (`strategies/loader.py`): require `module.signal` (was `compute_weights`);
  detect optional `module.signal_panel`; resolve `CONFIG.construction` via
  `get_construction_policy` (unknown id ⇒ `StrategyNotFound`); `validate_construction_params` at
  load; arity-check the signal (2-arg, or 3-arg iff `needs_fundamentals`); reject `signal_panel` with
  `needs_fundamentals` (no vectorized fundamentals fast path yet). Build the `LoadedStrategy`.
- **`sweep.py`** (in-slice — otherwise sweeps silently break): `_override` must (a) **carry the
  construction policy** when rebuilding `LoadedStrategy` (today it drops everything but
  `fn`/`fundamentals_fn`/`panel_fn`; with the raw-`construct_fn` design it rebuilds from the updated
  config); (b) route a **namespaced** grid key `construction.<key>` into `construction_params`, with
  non-prefixed keys going to signal `params`; and (c) **re-run validation after routing** — a
  `construction.<key>` override is re-checked by the policy's `param_validator` (so a sweep can't
  inject `top_k=0`/`-1`/`"3"`/an unknown key past the load-time wall), and a non-prefixed key is
  rejected unless it **already exists in `CONFIG.params`** (the enforceable no-silent-no-op rule
  absent signal-param schemas). `config_hash` covers both namespaces, so records + ranking stay correct.

## Other surfaces touched

- **`strategy new` template** (`cli/strategy_cmd.py`): scaffold authors `signal` + declares a
  `construction` policy (default `top_k_equal_weight`, `construction_params={"top_k": 2}`), docstring
  points at the policy library. (Exact template is a PR deliverable.)
- **Example strategies** (clean break):
  - `cross_sectional_momentum` — `signal` = trailing return per symbol; `signal_panel` = vectorized
    trailing-return matrix; `construction="top_k_equal_weight", construction_params={"top_k": 3}`.
  - `fundamentals_earnings_tilt` — `signal(view, params, fundamentals)` = latest-known EPS per
    symbol; `construction="equal_weight_positive"`.
- **`Strategy` protocol** (`contracts/types.py`): docstring notes `target_weights` is now the
  composed `construct(signal(view), view)` (the protocol signature is unchanged).
- **`data/hindsight.py`**: update the stale `compute_weights` comment reference to `signal`.
- **Import-linter** (`pyproject.toml`): a new contract making `algua.portfolio` pure — forbidden
  from cli/registry/data/backtest/strategies/live/execution/tracking/research/knowledge; it may
  import only `contracts` + `features`.

## Testing

- **Construction library** (new `tests/test_portfolio_construction.py`): each starter policy maps
  representative score vectors to expected weights; **non-finite/missing scores are dropped**, not
  zeroed (a name with `NaN` is never selected; a real `0.0` is distinct); empty/all-negative ⇒ flat;
  `score_proportional_long` normalizes to gross 1.0; missing/unknown required params raise at the
  resolver; unknown policy id raises; the dispatch map is immutable.
- **Composition** (`tests/test_strategies_base*.py`): `target_weights = construct(signal(...))` for
  the 2-arg and fundamentals lanes; `LoadedStrategy` post-init still rejects the needs_fundamentals
  arity mismatch.
- **Weight-level parity** (`tests/test_fast_path.py`, `tests/test_decision_parity.py`): the fast path
  equals the per-bar loop end-to-end; a deliberately divergent `signal_panel` raises `BacktestError`;
  a discontinuous near-tie that would pass a (rejected) signal-level check is caught by the
  weight-level guard; warmup bars are flat in both paths; backtest↔paper decision parity holds.
- **Identity** (`tests/test_config_hash_fields.py`, `tests/test_registry_approvals.py`): `config_hash`
  changes on `construction`/`construction_params` change; `code_hash` changes when a policy's source
  changes and when an id is **retargeted** to another callable (module-by-name rooting, incl. the
  `partial` case); a prior approval no longer validates after either change.
- **Loader** (`tests/test_strategy_loader.py`): requires `signal`; resolves + validates the policy;
  arity + panel rejections.
- **Sweep** (`tests/test_sweep*.py`): `construction.<key>` tunes `construction_params` (distinct
  `config_hash` per combo); a non-prefixed key still tunes signal params; `_override` preserves the
  construction policy; a key targeting neither namespace is rejected.
- The full ~19-file contract test surface migrates from `compute_weights`/weights-shaped
  `target_weights` to the signal+policy contract.

## Build order (single all-or-nothing PR)

A phased rollout is rejected — a half-migrated state (loader expects `signal` while the template
emits `compute_weights`, or sweeps tune a dead namespace) is unsafe. One PR; the list below is
**implementation sequencing**, not a per-commit gate-green guarantee — the migration steps (5–7) make
`construction` required and rewrite the examples/tests together, so the **full gate is green at the PR
tip**, with dependent migrations grouped into green commits where practical:

1. `algua/portfolio/construction.py` (policies + dispatch + load-time validation) + its tests.
2. `StrategyConfig` fields + `config_hash` fold.
3. `LoadedStrategy` composition (signal/construct/target_weights/signal_panel) + base tests.
4. `loader.py` resolution + `approvals.py` module-by-name closure rooting + identity tests.
5. `engine.py` per-bar path, then the fast path with the weight-level parity guard.
6. `sweep.py` namespace routing + override fix; `strategy new` template; `contracts`/`hindsight`
   docstrings; import-linter contract.
7. Migrate both example strategies; migrate the remaining contract test surface.

## Deferred (not this slice)

- Vol-targeting / turnover / risk-budgeting construction policies, and the `ConstructionContext`
  (prior holdings / covariance / constraints) the stateful ones need — #136.
- Multi-output / multi-factor / long-short signals and the signal *registry* (factor IC/IR, lineage,
  discoverability) — #140; this slice gives it a formal consumer.
- Per-layer evaluation CLI (evaluate a signal or a policy individually) — #137 / #140.
- Formal per-policy `construction_params` schemas (this slice does required-keys + finite/JSON
  validation only).
- A vectorized `construct_panel` for view-dependent construction.

## Gate

`uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports` green.
