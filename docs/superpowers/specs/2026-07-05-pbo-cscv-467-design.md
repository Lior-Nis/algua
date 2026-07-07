# Advisory PBO/CSCV overfitting diagnostic over the sweep grid (#467)

## Problem
The gate stack (DSR / LORD++ FDR / deflated breadth / bootstrap / regime) never asks whether the
selection rule "pick the in-sample-best combo" generalizes OOS across a sweep. PBO via CSCV
(Bailey & Lopez de Prado) answers exactly that: `PBO = P(the IS-optimal trial lands below the OOS
median)`. Advisory/reporting-first — no gate changes.

## Design (simplest thing that works)
- **Periods = the walk-forward windows.** The sweep already computes per-combo `window_metrics`
  (per OOS window Sharpe) and the holdout is a SEPARATE segment. So the trials × periods matrix is
  the per-combo per-window Sharpe grid; the holdout is excluded BY CONSTRUCTION (the matrix has
  exactly `windows` columns, never windows+1) AND the holdout metric is never computed on the PBO
  path (see HIGH-2 / round-2 finding 1).
- **Pure `algua/research/cscv.py::pbo(matrix, *, rank_by)`** — numpy only, no new dependency.
  Implements CSCV with a BOUNDED sub-period count (see Gate-2 hardening), fails closed on
  degenerate/odd/tiny/non-finite input, and aligns its IS-best selection with sweep's `rank_by`.
- **Plumb the matrix out of `sweep.py` WITHOUT putting it on the public result.** `_evaluate_combo`
  already holds the walk_forward result; it emits its per-window Sharpes. A new internal entry point
  `sweep_with_matrix(...) -> tuple[SweepResult, list[list[float]]]` assembles the trials × windows
  matrix and returns it ALONGSIDE (not inside) the `SweepResult`. The public `sweep(...) ->
  SweepResult` calls `sweep_with_matrix` and returns ONLY the result — the matrix is dropped, so it
  never rides on any `SweepResult` instance an in-process caller could read (round-2 finding 2).
  sweep.py does NOT import cscv (respects the backtest→research import contract).
- **Advisory CLI `research pbo`** in `research_cmd.py` (the composition layer wires
  backtest→research): resolves inputs exactly like `backtest sweep` (shared `resolve_eval_inputs` /
  `resolve_universe_inputs` / `resolve_delisting_inputs` helpers), calls `sweep_with_matrix()` to get
  BOTH the result (for breadth recording) and the matrix (for cscv), RECORDS SEARCH BREADTH like a
  sweep, feeds the matrix to `cscv.pbo`, and emits an AGGREGATE-ONLY advisory JSON. Burns no holdout,
  transitions nothing, writes no gate/FDR ledger row.

## GATE-1 (round 2) blocking findings — resolutions

An adversarial Codex pass against the WORKTREE CODE (walkforward.py, sweep.py, search_breadth.py)
returned BLOCK with 4 HIGH findings that the round-1 design overstated. Resolutions below; the
round-1 sections (HIGH-1/2/3) are updated in place to stay consistent.

### R2-1 — the holdout claim is re-scoped: the STATISTIC is never computed/burned, but STRATEGY CODE EXECUTES over holdout bars
Codex read `walk_forward` and found the round-1 claim "holdout bars never fetched" is FALSE:
`walk_forward` calls `build_portfolio(strategy, provider, start, end, ...)` and
`_market_return_series(..., start, end, ...)` over the FULL `[start, end]` period, so the holdout
bars ARE read from the provider AND the strategy's signal/construction code IS executed and the
portfolio IS simulated over them regardless of any flag.

**`holdout_window()` was investigated as a true no-simulation seam and does NOT cleanly close the
gap.** `algua/backtest/engine.py::holdout_window(strategy, provider, start, end, *, holdout_frac)`
computes the exact in-sample/holdout boundary DATE from `adj_grid(bars).index` *without running the
strategy* — it proves the boundary can be derived without simulating the strategy over the tail. But
adopting it to bound `build_portfolio`'s execution to the in-sample span is rejected for two concrete
reasons the round-1 chicken-and-egg framing understated:
- (i) `holdout_window()` itself STILL fetches the full `[start, end]` bar range
  (`provider.get_bars(..., start, end, "1d")`) to read the index, so it does NOT avoid the bar-range
  READ residual — it only avoids running the strategy over the tail.
- (ii) Truncating `build_portfolio`'s simulation to `[start, in_sample_end]` would change the total
  `len(returns)` fed to `_segment_bounds`, which re-carves the K in-sample windows from a shorter
  grid and BREAKS the bit-identical window carving that makes "PBO-over-window-only ==
  PBO-over-a-real-sweep" true — the sweep-equivalence property is load-bearing for the diagnostic's
  meaning ("does the combo THIS sweep would pick generalize?"). Preserving it while skipping holdout
  simulation would require a new in-sample-only execution mode INSIDE `build_portfolio`, which lives
  in `algua/backtest/engine.py` (CODEOWNERS-protected). So we take **option (b): state precisely what
  is and isn't true, and enumerate the residual leak channels honestly.**

**What IS true on the `compute_holdout=False` path (the PBO path):**
- The holdout RETURN-SERIES STATISTIC is never computed: no `_segment_record` runs over the holdout
  slice, `holdout_metrics = {}`, `holdout_returns = None`.
- The single-use holdout burn never fires: `on_peek` is NEVER called.
- No holdout number is surfaced, recorded, ranked, or fed to any gate.
- The full-period benchmark read is SKIPPED (`market_returns = None`): `_market_return_series` is a
  regime-gate input the PBO path never consumes, so we avoid that second holdout-spanning read.
- `window_metrics` (hence the PBO matrix) is BIT-IDENTICAL to a normal sweep, because the window
  bounds are carved from the identical `_segment_bounds`.

**What is NOT true — the accepted residual, stated as EXECUTION not merely a read:** the underlying
holdout BAR RANGE is read AND the strategy's `signal`/`construction` code and the vectorbt simulation
RUN over those holdout bars as part of the single full-period `build_portfolio` call. This is NOT a
passive read-only snapshot access — arbitrary strategy code executes with the holdout bars in scope.
Because **strategy sandboxing is deferred** (there is no execution isolation around strategy code
today), that execution has residual leak channels a malicious or careless strategy could exploit:
- **Side effects:** a strategy that writes to files / a DB / a network endpoint / a global cache
  during `signal`/`construction` can persist holdout-period observations outside the return series.
- **Exceptions:** a holdout-bar-triggered exception can alter control flow or surface holdout-derived
  values through error messages / tracebacks.
- **Timing:** wall-clock execution time can correlate with holdout data characteristics (a timing
  side-channel), observable to a caller that measures it.
- **Logging:** strategy-emitted log lines over holdout bars can leak holdout-derived values into logs.
- **Mutable module state:** a strategy mutating module-level / class-level state during holdout
  execution leaves that state readable by a later in-process call (a cross-run channel).

**Why this residual is accepted for this advisory slice:** it is NOT new to PBO — every `backtest
sweep` / `walk_forward` today already executes the strategy over the full `[start, end]` including the
reserved holdout tail, so the PBO path is *strictly no-worse* than the existing sweep surface and adds
NO new scoring, ranking, recording, or burn of the holdout STATISTIC. The gate the holdout machinery
protects is *selecting on the holdout metric* — that metric is never computed here. Closing the
execution channel (a true in-sample-only strategy-execution mode, or strategy sandboxing) requires the
CODEOWNERS engine seam and is DEFERRED. The invariant language throughout this doc therefore claims
only "the holdout STATISTIC is never computed/burned"; it explicitly does NOT claim "the holdout is
never read" NOR "strategy code never executes over holdout bars".

### R2-2 — close the in-process SweepResult leak (matrix off the public result)
Codex flagged that round-1's plan (a `trial_window_sharpes` field on `SweepResult` + a `to_dict()`
that withholds it) still hands the raw matrix to any in-process Python caller of the public
`sweep()` via `result.trial_window_sharpes`; the `to_dict()` guard only covers the JSON/tracking
surface, not attribute access. **Resolution — option (a): the matrix is NEVER a field on
`SweepResult`.** The refactor:
- `sweep_with_matrix(...) -> tuple[SweepResult, list[list[float]]]` is the internal entry point that
  does today's sweep work AND assembles the matrix, returning them as two separate values.
- The public `sweep(...) -> SweepResult` is `return sweep_with_matrix(...)[0]` — it discards the
  matrix. Every existing caller (`backtest sweep`, `sweep_task`, `record_search_breadth`, tracking)
  keeps calling `sweep()` and gets a `SweepResult` that has NO matrix attribute at all.
- Only `research pbo` calls `sweep_with_matrix()` directly, holding the matrix as a local that flows
  straight into `cscv.pbo` and is never stored on any serializable object.

This is strictly stronger than a `to_dict()` override: there is no matrix attribute to read off a
`SweepResult`, so neither JSON, tracking, `--summary`, NOR a caller that holds a `SweepResult`
instance can reach the per-combo per-window Sharpes off that object.

**Scope of the claim (not overstated):** `sweep_with_matrix()` IS a real, importable in-process entry
point that returns the matrix by design — that is how `research pbo` obtains it. So the precise claim
is NOT "the matrix is unreachable to all in-process code"; any code can
`from algua.backtest.sweep import sweep_with_matrix` and call it. The claim is exactly: **no
`SweepResult` INSTANCE carries the matrix**, so nothing that merely holds a result object — tracking,
logging, `to_dict()`/JSON, `--summary`, `record_search_breadth`, or a caller of the public `sweep()`
— can reach it. This is the right boundary because importability grants an in-process caller no
capability it lacks anyway: any in-process code can already assemble the matrix itself by running its
own walk-forward per combo. What we prevent is the matrix RIDING on the recorded/serialized/tracked
result surface, where it would silently become an unmetered selection oracle. The threat model no
longer has to assume "callers only touch the JSON surface".

### R2-3 — the "no meta-search oracle" claim is softened to "no UNMETERED oracle"
Codex noted the residual that HIGH-3 (round 1) glossed: even aggregate-only, repeated `research pbo`
calls under different grids / `rank_by` are a REAL search channel — an agent can compare the scalar
PBO across many invocations. **We do not claim to eliminate this; we claim to METER it, and we accept
the metered residual explicitly.** Every `research pbo` invocation records its full grid's measured
breadth through `record_search_breadth` (round-1 HIGH-1), keyed by strategy name, so meta-searching
via repeated `pbo` runs INFLATES funnel-wide breadth exactly as repeated `backtest sweep` runs do —
which DEFLATES the eventual promotion Sharpe bar and TIGHTENS the LORD++ FDR ledger (over-counting is
the fail-safe direction). Combined with the aggregate-only output (no per-combo/per-split internals
to hill-climb on within a single call), the residual channel is limited to comparing a single
metered scalar across metered calls. **Documented residual risk:** breadth metering is the
mitigation, not elimination; a fully un-gameable design would require rate/count-limiting `pbo` per
strategy or persisting each PBO query's params to an audit ledger — both deferred (the latter needs a
schema/ledger this advisory slice deliberately avoids). The honest claim is therefore "`research pbo`
is a METERED search channel that self-penalizes at promotion via recorded breadth", NOT "no search
channel exists".

### R2-4 — provenance is fully reconstructable (base `config_hash` + untruncated `grid_hash` + delisting inputs)
Codex flagged that a recorded PBO figure was not reconstructable to the exact config+grid, and a
follow-up GATE-1 pass flagged that the delisting inputs — which CHANGE realized returns — were
missing from the block. **The `provenance` block adds:**
- the base strategy `config_hash` (`config_hash(base_strategy)` from `algua.strategies.base`, the
  same canonical hash the engine/registry use), and
- a FULL, untruncated `grid_hash` (`sha256(json.dumps(grid, sort_keys=True)).hexdigest()` — 64 hex
  chars, no `[:12]`), and
- the delisting inputs: `delisting_snapshot` (the ACTUAL resolved snapshot id from
  `resolve_delisting_inputs`, or `null`), `delistings_name` (the CLI `--delistings` handle/label the
  user passed, or `null`), and `assume_terminal_last_close` (bool). **These three are NOT carried on
  `SweepResult`** (it has no delisting fields), so `research pbo` captures them at the CLI layer from
  the `resolve_delisting_inputs(...)` return and the raw `--delistings` / `--assume-terminal-last-close`
  args, alongside the `SweepResult` meta.

With these added, EVERY return-affecting `backtest sweep` CLI input is now pinned in the block:
`period` (start/end), `data_source` + `snapshot_id` (demo/snapshot), `universe_name` +
`universe_snapshots`, `fundamentals_snapshot`, `news_snapshot`, `delisting_snapshot` +
`delistings_name` + `assume_terminal_last_close`, `windows`, `holdout_frac`, `rank_by`, `grid_hash`,
`seed`, `code_hash`, `dependency_hash`, and `config_hash`. (Backtest fees/slippage are fixed code
constants, not sweep-CLI-tunable, so they are pinned transitively by `code_hash`/`dependency_hash`.)
A recorded PBO number is now fully reconstructable to the exact base config, grid, data, and delisting
policy that produced it. The raw `grid` dict is still NOT emitted (only its full hash), keeping the
surface aggregate-only.

## Round-1 findings — resolutions (retained, reconciled with round 2)

### HIGH-1 — PBO search is METERED (route through `record_search_breadth()`)
`research pbo` runs a full, real grid sweep — genuine search — so it MUST be metered or it becomes an
unmetered search channel. **Resolution:** `research pbo` records this sweep's measured breadth
through the SAME `record_search_breadth(SqliteStrategyRepository(conn), name, result)` path
`backtest sweep`/`sweep_task` use — keyed by strategy NAME, inside a caller-owned `registry_conn()`.
The pbo sweep therefore counts toward funnel-wide breadth exactly like any other sweep (over-counting
only ever TIGHTENS downstream gates — fail-safe direction). The `result` passed to
`record_search_breadth` is the `SweepResult` from `sweep_with_matrix()[0]` (matrix already dropped).
`research pbo` **records search breadth**; it does NOT burn the holdout, does NOT transition, and does
NOT write any gate/FDR/holdout-burn ledger row.

### HIGH-2 — the holdout statistic is never computed on the PBO path (window-only path)
`walk_forward()` ALWAYS segments and evaluates `holdout_metrics` today. **Resolution:** add a
window-only evaluation path that never computes the holdout metric (see round-2 finding R2-1 for the
precise, honest scope — the bar RANGE is still read; the STATISTIC and burn are not).
- `walk_forward(..., compute_holdout: bool = True)` (default preserves today's behavior). When
  `False`: it carves the IDENTICAL `_segment_bounds` (same K in-sample windows, same purge/embargo
  gap, same holdout interval reserved-but-unscored), evaluates ONLY the K in-sample windows, sets
  `holdout_metrics = {}` and `holdout_returns = None`, NEVER slices/scores the holdout returns,
  NEVER calls `on_peek` (no burn), and SKIPS `_market_return_series` (`market_returns = None`) — the
  benchmark series is a regime-gate input the PBO path never uses, so skipping it avoids a redundant
  holdout-spanning read. `build_portfolio(start, end)` STILL runs over the full period (the accepted
  residual read, R2-1). `walkforward.py` is NOT a CODEOWNERS-protected file.
- `sweep_with_matrix(..., compute_holdout: bool = True)` threads the flag to `_evaluate_combo` →
  `walk_forward`. The public `sweep()` / `backtest sweep` / `sweep_task` keep the default (`True`) so
  their behavior is unchanged. `research pbo` passes `compute_holdout=False`.
- Because the window bounds are carved identically regardless of the flag, `window_metrics` (hence
  the matrix) is BIT-IDENTICAL to a normal sweep — PBO over the window-only path equals PBO over a
  full sweep, and it provably never computes the holdout statistic. `trial_sharpe_*` (the
  DSR/breadth evidence, derived from window Sharpes) is likewise unaffected, so breadth recording
  works unchanged with `compute_holdout=False`.

### HIGH-3 — CLI output is AGGREGATE-ONLY (no UNMETERED meta-search oracle surface)
`research pbo` emits a FIXED, curated dict — never `SweepResult.to_dict()`, never the `ranked` combo
list, never the raw Sharpe matrix, never per-split logits. Exposed fields: `pbo` (float | null),
`split_count`, `trial_count`, `window_count`, `subperiod_count`, `rank_by`, `warnings` (list[str]),
and a `provenance` block (see Gate-2 item 6). `cscv.pbo` returns a `PboResult` dataclass carrying
ONLY those aggregates; it computes per-split logits internally and discards them. The matrix lives
ONLY as a local in `research pbo` (returned by `sweep_with_matrix`, never stored on a `SweepResult` —
R2-2), so it never reaches tracking, `--summary`, or the recorded sweep payload. Result: no surface
an agent can read exposes the per-combo per-window Sharpes or the ranked selection within a single
call. The residual cross-call meta-search channel is METERED via recorded breadth (R2-3), not
eliminated.

## GATE-2 hardening (addressed pre-emptively in the design)

1. **Bounded C(S, S/2) split count — grouping partitions WINDOWS, it does NOT pre-average them.**
   `S = min(T, CSCV_MAX_SUBPERIODS)` with `CSCV_MAX_SUBPERIODS = 16` (even); if the resulting `S` is
   odd, `S -= 1` (C(S,S/2) needs even S). The T window-columns are partitioned into S CONTIGUOUS,
   balanced groups (sizes differ by ≤1; earlier groups absorb the remainder). A group is a BUNDLE OF
   ORIGINAL WINDOWS, purely to bound the split combinatorics; **there is no group-mean "cell".** Each
   of the `C(S, S/2)` splits assigns half the groups (hence their constituent windows) to the TRAIN
   half and half to the TEST half; the per-trial reductions in item 4 then run over those windows'
   TRUE per-window Sharpes (never a group mean). Splits = `C(S, S/2)`, capped at `C(16, 8) = 12_870`
   — cheap and bounded regardless of T. No subsampling needed at this cap, so PBO stays deterministic
   (no seeded split sampling).
2. **Fail-closed odd/tiny window counts.** `CSCV_MIN_WINDOWS = 4`. If `T < CSCV_MIN_WINDOWS`, or if
   there are `< 2` trials (a single combo is trivially always-selected → PBO undefined), `cscv.pbo`
   returns `PboResult(pbo=None, ..., warnings=[<why>])` — it does NOT raise and does NOT rank. Odd
   T ≥ 4 is handled by the `S -= 1` even-ing rule above (no even-T requirement imposed on the
   caller). The CLI surfaces the warning and emits `pbo: null`, exit 0 (advisory).
3. **NaN/inf → fail closed, never silently rank.** If ANY cell of the trials × windows matrix is
   non-finite, `cscv.pbo` returns `PboResult(pbo=None, warnings=["non-finite Sharpe in matrix; PBO
   not computed (fail closed)"])`. It refuses to produce a PBO number rather than ranking around the
   bad value (mirrors sweep's `_trial_sharpe_stats` fail-closed-on-non-finite discipline).
4. **IS-best selection aligned with sweep's `rank_by`, computed over TRUE per-window Sharpes (exact,
   no group-mean approximation).** `cscv.pbo(matrix, *, rank_by)` accepts
   `rank_by ∈ {"mean_sharpe","min_sharpe"}` (validated against the same `_RANK_KEYS` set sweep uses).
   Per split, the IS-best trial is the argmax over trials of the MATCHING reduction over the split's
   TRAIN WINDOWS — the original per-window Sharpes (matrix columns) whose windows fall in the
   train-half groups — NOT over group means: `mean` over the train windows when
   `rank_by=="mean_sharpe"`, `min` over the train windows when `rank_by=="min_sharpe"`. Reducing over
   the constituent windows (item 1) rather than pre-averaged group cells makes the `rank_by` alignment
   EXACT and, critically, closes the `min_sharpe` masking failure mode: a group mean could hide a
   single disastrous window inside an otherwise-fine group, letting a blow-up survive selection — the
   per-window `min` sees the true worst train window. Ties are broken by ascending train-Sharpe std
   **computed over those same train windows** then lowest trial index, mirroring sweep's
   `_rank_records` (score desc, std asc, stable). **No tie-break input comes from outside the split:**
   the std is over the split's train windows only, and the "trial index" is the FIXED positional row
   index of the matrix (the generated-combo order — see finding 1), a stable identity, NEVER sweep's
   global `ranked`/`score` (which would leak full-sample, out-of-split information into a per-split
   decision) and NEVER any TEST-window value. The OOS side then ranks that selected trial by MEAN
   Sharpe over the split's TEST WINDOWS (the generalization criterion, Bailey & LdP): relative rank
   `ω = avg_rank/(N+1)` (average ranks for ties; `1..N → ω ∈ (0,1)`, so the logit is always finite),
   `λ = ln(ω/(1-ω))`, `PBO = fraction of splits with λ ≤ 0`. The IS/OOS asymmetry (selection follows
   `rank_by`; generalization measured by mean) is intentional and documented in the docstring.
   `research pbo` passes the SAME `--rank-by` it hands `sweep_with_matrix()` so the diagnostic
   answers "does the combo THIS sweep would pick generalize?".
5. **Regression test: no matrix on the serialized OR in-process result surface.** Because the matrix
   is NEVER a field on `SweepResult` (R2-2), the regression test asserts: (a) `SweepResult` has no
   `trial_window_sharpes` / matrix attribute and `SweepResult(...).to_dict()` has no such key; (b) a
   real `sweep_task(...)` payload has no matrix key; (c) the `--summary` projection
   (`_SWEEP_SUMMARY_KEYS`) has no matrix key; (d) the dict `log_sweep` logs (via `result.to_dict()`)
   has no matrix key; and (e) `sweep_with_matrix(...)` returns the matrix as the SECOND tuple element
   while `sweep(...)` returns a `SweepResult` from which the matrix is unreachable. `SweepResult`
   keeps overriding `to_dict()` only as it does today (nothing new to withhold).
6. **Provenance in the advisory output (round-2 finding R2-4).** The `provenance` block is sourced
   from the `SweepResult` meta, the base strategy, AND the CLI-resolved delisting inputs:
   `config_hash` (`config_hash(base_strategy)`, the base strategy config identity — R2-4),
   `code_hash`, `dependency_hash`, `data_source`, `snapshot_id`, `timeframe`, `seed`, `period`,
   `universe_name`, `universe_snapshots`, `fundamentals_snapshot`, `news_snapshot`, `windows`,
   `holdout_frac`, `rank_by`, a `grid_hash` = FULL untruncated
   `sha256(json.dumps(grid, sort_keys=True)).hexdigest()` (64 hex chars — R2-4), and the delisting
   inputs `delisting_snapshot` (resolved snapshot id from `resolve_delisting_inputs`, or null),
   `delistings_name` (the `--delistings` CLI handle, or null), and `assume_terminal_last_close` (bool)
   — the last three captured at the CLI layer because `SweepResult` does not carry them (R2-4). A PBO
   number is thus fully reconstructable to the exact code/config/data/grid/delisting-policy that
   produced it. The raw `grid` dict is NOT emitted (only its hash), keeping the surface aggregate-only.

## Invariants held
- The holdout STATISTIC is never computed or burned on the PBO path (`compute_holdout=False`:
  `holdout_metrics={}`, `holdout_returns=None`, `on_peek` never called, `market_returns=None`). The
  underlying holdout bar RANGE is still read AND the strategy's signal/construction code plus the
  vectorbt simulation EXECUTE over those holdout bars as part of the single full-period
  `build_portfolio` call — this is NOT framed as a passive read-only snapshot access; arbitrary
  strategy code runs with the holdout bars in scope. Because strategy sandboxing is DEFERRED, that
  execution carries residual leak channels (side effects, exceptions, timing, logging, mutable module
  state — R2-1), accepted because it is strictly no-worse than every existing `backtest sweep` and
  surfaces/records/burns NO holdout statistic. "Holdout untouched" and "strategy never executes over
  holdout bars" are NOT claimed; "holdout STATISTIC never computed/burned" is.
- The per-combo per-window Sharpe matrix is NEVER a field on `SweepResult` and is unreachable from
  any `SweepResult` instance (public `sweep()` drops it; only `sweep_with_matrix()` returns it, as a
  separate tuple element that `research pbo` holds as a local and passes straight to `cscv.pbo`) —
  R2-2. Not on JSON, tracking, `--summary`, NOR attribute access off a result object. The claim is
  scoped to result-object surfaces: `sweep_with_matrix()` itself IS an importable in-process entry
  point that returns the matrix BY DESIGN (that is how `research pbo` gets it) — importability is not
  a leak, since any in-process caller could assemble the matrix itself anyway (R2-2).
- `research pbo` records search breadth (it performs real search); writes no gate/FDR/holdout-burn
  ledger row; transitions nothing. Repeated `pbo` runs are a METERED (not eliminated) search channel
  that self-penalizes at promotion via recorded breadth (R2-3).
- The CSCV IS-best reduction and tie-break run over TRUE per-window train Sharpes and use ONLY
  within-split information + the fixed positional (generated-combo-order) trial index — never
  cross-split, OOS, or sweep-global `ranked`/`score` values; `min_sharpe` sees the true worst train
  window (no group-mean masking) (Gate-2 items 1 & 4, findings 1 & 2).
- Aggregate-only CLI output: no raw matrix, no per-split logits, no ranked-combo detail on any
  agent-readable surface; provenance carries the base `config_hash`, a full-SHA-256 `grid_hash`, and
  the delisting inputs (`delisting_snapshot`, `delistings_name`, `assume_terminal_last_close`) so a
  PBO figure is fully reconstructable to the exact config/grid/data/delisting policy (R2-4).
- No new dependency (numpy only). No CODEOWNERS-protected file touched (`walkforward.py`, `sweep.py`,
  `cscv.py`, `research_cmd.py`, `search_breadth.py` are all unprotected). No gate change. No
  `SCHEMA_VERSION` bump (breadth recording reuses the existing `record_search_trial` schema).
- Import contract: `sweep.py`/`walkforward.py` do NOT import `algua.research`; `cscv.py`
  (`algua.research`) imports numpy only — no `algua.registry`, `algua.backtest`, or I/O. The
  `record_search_breadth` call and the `config_hash(base_strategy)` provenance call sit in the CLI
  layer (`algua.cli.research_cmd`), which may import both lanes.

## Task list (ordered)
1. **`walk_forward` window-only path** — add `compute_holdout: bool = True` to
   `algua/backtest/walkforward.py::walk_forward`. When `False`: carve the identical
   `_segment_bounds`, evaluate only the K in-sample windows, set `holdout_metrics = {}` and
   `holdout_returns = None`, never slice/score the holdout returns, never call `on_peek`, and SKIP
   `_market_return_series` (set `market_returns = None`). `build_portfolio(start, end)` still runs
   over the full period (accepted residual, R2-1). Default keeps current behavior. Tests
   (`tests/test_walkforward_window_only.py`): `window_metrics` is byte-identical to the
   `compute_holdout=True` path over the same inputs; `holdout_metrics == {}`, `holdout_returns is
   None`, `market_returns is None`; `on_peek` is NOT invoked (assert a passed spy is never called).
2. **`sweep` plumbs matrix via a separate internal entry point (matrix OFF the public result)** — in
   `algua/backtest/sweep.py`: (a) `_evaluate_combo` adds `"window_sharpes":
   [w["sharpe"] for w in wf.window_metrics]` to its returned dict; (b) rename today's `sweep` body
   into `sweep_with_matrix(..., compute_holdout: bool = True) -> tuple[SweepResult,
   list[list[float]]]` — it threads `compute_holdout` into `eval_kwargs`, assembles
   `matrix: list[list[float]]` (trials × windows, in GENERATED-COMBO ORDER — `[res["window_sharpes"]
   for res in results]`, where `results` is produced from `overridden`/`combos` in
   `_combos(grid)` generation order, so row `i` is combo `i`, aligned with `n_combos`), and returns
   `(SweepResult(...), matrix)`; (c) the public `sweep(...) -> SweepResult` becomes
   `return sweep_with_matrix(...)[0]` (drops the matrix). `SweepResult` gains NO new field. sweep.py
   must NOT import cscv. Tests extend `tests/test_sweep*.py`: matrix shape = `n_combos × windows`;
   **matrix rows align with the GENERATED-COMBO order (`combos`/`results` order — the order
   `_evaluate_combo` results are produced in), NOT `ranked` order** — assert this by constructing a
   grid whose `ranked` order provably DIFFERS from generation order and checking each matrix row `i`
   equals combo `i`'s per-window Sharpes (`results[i]["window_sharpes"]`), i.e. the matrix is NOT
   reordered by rank; matrix bit-identical with/without `compute_holdout`; `sweep()` return value has
   no matrix attribute.
3. **Pure `cscv.pbo`** — new `algua/research/cscv.py`: numpy-only `PboResult` dataclass
   (`pbo: float | None`, `split_count`, `trial_count`, `window_count`, `subperiod_count`,
   `rank_by`, `warnings: list[str]`) and `pbo(matrix, *, rank_by="mean_sharpe") -> PboResult`
   implementing the bounded-S CSCV of Gate-2 items 1–4 (constants `CSCV_MAX_SUBPERIODS=16`,
   `CSCV_MIN_WINDOWS=4`; fail-closed on `<2` trials, `T<4`, non-finite; `rank_by`-aligned IS-best
   reduction over the split's TRUE per-window TRAIN Sharpes — NOT group means (item 4); tie-break
   uses ONLY within-split train std + the fixed positional trial index, never cross-split/OOS/global
   info; mean-OOS logit crossing over the split's TEST windows). Imports numpy only — no
   registry/backtest/IO. Tests: `tests/test_cscv_pbo.py` covering a known-overfit matrix (PBO high),
   a generalizing matrix (PBO low), split-count bound at large T, odd/tiny/single-trial fail-closed,
   non-finite fail-closed, mean-vs-min `rank_by` divergence, AND a **`min_sharpe` masking regression**:
   a matrix where one trial has a single catastrophic train window buried in an otherwise-strong group
   must NOT be selected IS-best under `rank_by="min_sharpe"` (proving the reduction sees the true worst
   per-window value, not a group mean that would hide it).
4. **`research pbo` CLI** — new `@research_app.command("pbo")` in `algua/cli/research_cmd.py`:
   resolve inputs via the shared `resolve_eval_inputs`/`resolve_universe_inputs`/
   `resolve_delisting_inputs` + `parse_grid` (same as `sweep_task`); accept `--rank-by`,
   `--windows`, `--holdout-frac`, `--param`, universe/fundamentals/news/delisting/demo/snapshot
   flags; call `result, matrix = sweep_with_matrix(..., compute_holdout=False, rank_by=<rank_by>)`;
   capture the delisting provenance at the CLI layer — `resolve_delisting_inputs` returns the resolved
   `delisting_snapshot` id, and keep the raw `--delistings` handle and `--assume-terminal-last-close`
   flag (SweepResult carries NEITHER); RECORD BREADTH via `with registry_conn() as conn:
   record_search_breadth(SqliteStrategyRepository(conn), name, result)`; call
   `cscv.pbo(matrix, rank_by=<rank_by>)`; emit the AGGREGATE-ONLY payload (`pbo`, `split_count`,
   `trial_count`, `window_count`, `subperiod_count`, `rank_by`, `warnings`, `provenance` block with
   the item-6 fields including base `config_hash`, FULL `grid_hash`, and the delisting inputs
   `delisting_snapshot` + `delistings_name` + `assume_terminal_last_close`). Never emit `ranked`, the
   matrix, or per-split detail. `@json_errors(...)`. Tests: `tests/test_research_pbo_cli.py` (payload
   keys are exactly the aggregate+provenance set; NO matrix/ranked keys; `provenance.grid_hash` is 64
   hex chars; `provenance.config_hash` present and equals `config_hash(base)`; `provenance` carries
   `delisting_snapshot`, `delistings_name`, `assume_terminal_last_close`, and a `--delistings NAME` run
   stamps the resolved snapshot id while a run without it stamps null/false; breadth row IS recorded;
   no holdout burn / no transition; fail-closed grid yields `pbo: null` + warning, exit 0).
5. **Matrix-not-on-result regression test** — `tests/test_sweep_pbo_matrix_withheld.py` proving the
   matrix never reaches any `SweepResult` surface (Gate-2 item 5): (a) `SweepResult` has no matrix
   attribute and `SweepResult(...).to_dict()` has no matrix key; (b) a real `sweep_task` payload has
   none; (c) the `_SWEEP_SUMMARY_KEYS` `--summary` projection has none; (d) the dict `log_sweep`
   builds has none; (e) `sweep_with_matrix(...)` returns a non-empty `matrix` as its second element
   while `sweep(...)` exposes no way to reach it.
6. **CLAUDE.md command-surface entry** — add a `uv run algua research pbo ...` bullet to the
   command surface: advisory PBO/CSCV over the sweep grid; RECORDS breadth (real, metered search —
   repeated calls self-penalize at promotion); burns no holdout STATISTIC (holdout bars are read as
   part of the full-period sweep fetch, but never scored/burned), transitions nothing, writes no gate
   ledger; aggregate-only output (no matrix/ranked); a HIGH pbo (≳0.5) flags a selection rule that
   does not generalize OOS. Note the `interpret-results` skill link.
7. **FAST per-task check** during each task:
   `uv run ruff check . && uv run mypy algua && uv run lint-imports && uv run pytest -q <this
   task's test file(s)>`. **FULL gate** at integration/finish:
   `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`.
8. **PR** — branch pushed ALONE; `gh pr create` separately. No CODEOWNERS-protected path is touched
   (walkforward.py, sweep.py, cscv.py, research_cmd.py, search_breadth.py, tracking, CLAUDE.md are
   all unprotected) → eligible for auto-merge iff CI is green.

## Deferred / out of scope
- No gate wiring: PBO stays advisory (no promotion/forward gate consumes it). Wiring PBO as a
  tighten-only AND-check is a separate, gate-touching (CODEOWNERS) change.
- **No in-sample-only strategy EXECUTION (and no strategy sandboxing):** `holdout_window()` proves the
  in-sample/holdout boundary date is derivable without running the strategy, but (i) it still fetches
  the full bar range and (ii) truncating `build_portfolio`'s SIMULATION to the in-sample span needs a
  new in-sample-only execution mode inside CODEOWNERS `engine.py` AND would break bit-identical window
  carving (R2-1). So the strategy's signal/construction code and the simulation STILL execute over the
  holdout bars, with the residual leak channels enumerated in R2-1 (side effects, exceptions, timing,
  logging, mutable module state). This is accepted for the advisory slice (strictly no-worse than
  today's `backtest sweep`, no holdout statistic surfaced/burned); closing it via an in-sample-only
  execution mode or true strategy sandboxing is DEFERRED (CODEOWNERS engine seam).
- **No rate/count-limit or per-query audit ledger for `research pbo`:** the metered residual
  meta-search channel (R2-3) is mitigated by recorded breadth, not eliminated; a hard limiter or a
  PBO-query audit ledger would need a schema/ledger this advisory slice avoids — deferred.
- No seeded split SUBSAMPLING: the `S ≤ 16` cap keeps `C(S,S/2) ≤ 12_870`, so all splits are
  enumerated deterministically; a larger effective S (with sampling) is a future refinement.
- No persisted PBO/matrix artifact: the matrix is in-process only by design (R2-2). Recording a
  PBO time-series per strategy would need a schema/ledger and is deferred.
