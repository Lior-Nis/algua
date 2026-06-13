# Holdout peek/burn boundary — issue #193 design

**Issue:** #193 — "Holdout release frees a window whose holdout metrics were already computed in `walk_forward`'s tail."
**Surfaced by:** GATE-2 review of PR #191 (#161 atomic holdout reservation), Codex HIGH. Pre-existing; not a regression of #191.
**Base:** `origin/main` (contains PR #191's reserve→run→finalize/release reservation lifecycle).
**Touches:** `algua/backtest/walkforward.py`, `algua/cli/research_cmd.py`, tests. No CODEOWNERS-protected file.

## Problem

`research promote` burns a single-use out-of-sample holdout window via #161's lifecycle:

```
reservation_id = reserve_holdout(...)         # insert a PENDING row (committed_at = NULL)
try:
    wf = walk_forward(...)                     # compute the holdout
except BaseException:
    release_holdout_reservation(reservation_id)   # free the window
    raise
finalize_holdout_reservation(reservation_id, ...) # commit the burn
run_gate(...)
```

Inside `walk_forward`, the holdout metrics are computed (`_segment_record` on the holdout
slice), then ~25 more lines of fallible work run before the result is returned: stability
arithmetic, `runtime_stamps()`, `provenance()`, dataclass construction. A failure in that
tail — or a `KeyboardInterrupt`, which Python can deliver between **any** two bytecodes — reaches
the CLI's `except BaseException` and **releases a window whose holdout metrics were already
computed**.

This was deferred from #191 because the outcome matched `main` exactly (pre-#191, a `walk_forward`
that raised post-metrics also wrote no burn). It is a defense-in-depth gap, not a live exploit:
the post-peek tail is deterministic (no retry-until-pass attack), and on the failure path the
metrics are never returned or emitted, so no operator/agent observes them. The value of fixing it
is making the burn-on-peek invariant robust by construction — so future fallible code added to that
tail cannot silently reopen the gap.

## Authoritative definition of "peek"

**Computation == peek**, where the computation that counts is the **holdout summary-metric
evaluation** — the `_segment_record(returns, holdout[0], holdout[1])` call that produces
`holdout_metrics`.

Rationale (chosen over "observation == peek"):
- It is a **bright, enforceable line** — one call site, not "did anything ever observe the value"
  (which can't be proved across side channels: tracebacks, logs, timing).
- It is **robust to future code** — the wall holds by construction as long as the burn commits
  before that call, regardless of what tail code is added later.
- **Harms are asymmetric** — over-strict costs a recoverable window on a rare crash; over-lax costs
  silent, undetectable overfitting bias in a system that autonomously promotes toward live money.

### Why the upstream `build_portfolio` is NOT the peek

`walk_forward` runs the strategy **once** over the full `[start, end]` period via `build_portfolio`,
then segments the resulting returns series into in-sample windows + the holdout. The holdout
window's returns therefore exist in memory (as a slice of `returns`) before any metric is computed —
as do every in-sample window's returns.

That full-period simulation is **shared in-sample/holdout work, not the peek**. The holdout's
distinguishing event is not *when its bars are simulated* but *when its bars are evaluated into a
reported metric* — `_segment_record(holdout)`. On a failure before that evaluation, `walk_forward`
raises, the CLI releases, and **nothing is summarized or emitted** — the returns array is discarded.
No holdout information can flow to a subsequent promote attempt, so the overfitting threat
(read metric → tweak → re-run) is not enabled. Deferring holdout-bar *simulation* until after the
burn was considered and **rejected**: it would change the walk-forward "simulate once, then segment"
engine contract (requiring a stateful/resumable engine across the train/holdout seam) and buys no
real leak protection over the definition above.

## Design

### Invariant

> **No computed holdout metric is ever released.** The burn is committed the instant *before* the
> holdout metric is evaluated; everything from that point on fails closed (the window stays burned).

This is deliberately *not* "burn iff a metric was computed": the rare path where the burn commits
and then a crash/interrupt lands before `_segment_record` runs leaves a committed burn with no
returned metric. That **over-burning** is the accepted conservative direction.

### Mechanism — the idempotent-release linchpin

`release_holdout_reservation` is already a **no-op on a committed row**
(`DELETE FROM holdout_evaluations WHERE id = ? AND committed_at IS NULL`, store.py). So if the burn
is finalized *before* the peek, the CLI's existing release-on-failure becomes correct automatically:
after the commit, any release is a no-op and the burn survives.

### `algua/backtest/walkforward.py` — `walk_forward()`

Add an optional, repo-agnostic hook and fire it immediately before the holdout evaluation. All
holdout-independent fallible work is ordered before the hook so an infra failure there frees the
window (good UX) instead of spuriously burning it.

```python
def walk_forward(..., *, ..., on_peek: Callable[[str], None] | None = None) -> WalkForwardResult:
    pf, _weights = build_portfolio(strategy, provider, start, end, universe_by_date=universe_by_date)
    returns = pf.returns()
    bounds, holdout = _segment_bounds(len(returns), windows, holdout_frac)
    window_metrics = [{"index": i, **_segment_record(returns, s, e)} for i, (s, e) in enumerate(bounds)]

    # stability (from window_metrics), stamps, prov, cfg_hash — all holdout-independent, all first.
    sharpes = [w["sharpe"] for w in window_metrics]
    stability = {...}
    stamps = runtime_stamps()
    prov = provenance(provider, seed)
    cfg_hash = config_hash(strategy)

    if on_peek is not None:
        on_peek(cfg_hash)        # caller commits the single-use burn HERE, before the peek
    holdout_metrics = _segment_record(returns, holdout[0], holdout[1])   # THE PEEK

    return WalkForwardResult(
        strategy=strategy.name, config_hash=cfg_hash, timeframe="1d",
        code_hash=stamps["code_hash"], dependency_hash=stamps["dependency_hash"],
        period={...}, windows=windows, holdout_frac=holdout_frac,
        window_metrics=window_metrics, holdout_metrics=holdout_metrics,
        stability=stability, universe_name=universe_name, universe_snapshots=universe_snapshots,
        **prov,
    )
```

- `on_peek` is **opaque** to `walk_forward` (no repo import; module boundary preserved). It receives
  `cfg_hash` so the caller's burn records the exact evidentiary hash that goes into the result.
- Default `None` keeps every other caller and all existing tests **byte-identical**.
- The contract (docstring): `on_peek` is called exactly once, immediately before the holdout is
  evaluated. A caller using it to commit a single-use burn relies on nothing fallible-and-releasing
  happening after the call — which the CLI guarantees via idempotent release.

### `algua/cli/research_cmd.py` — `promote()`

Pass the finalize as the hook; remove the now-redundant post-return finalize; leave the existing
`except BaseException: release` unchanged.

```python
reservation_id, reused = repo.reserve_holdout(...)          # raises here = fail closed (pre-peek)
try:
    wf = walk_forward(
        strategy, provider, start_dt, end_dt, windows=windows, holdout_frac=holdout_frac,
        universe_by_date=universe_by_date, universe_name=universe, universe_snapshots=universe_prov,
        on_peek=lambda cfg: repo.finalize_holdout_reservation(reservation_id, config_hash=cfg),
    )
except BaseException:
    try:
        repo.release_holdout_reservation(reservation_id)   # no-op if on_peek already committed
    except Exception:
        pass
    raise
# (the previous post-return finalize call is REMOVED — the burn now happens via on_peek)
outcome = run_gate(repo, wf, ...)
```

### Correctness trace

- Failure **before** `on_peek` (build_portfolio, segmentation, window/stability/stamps/prov):
  row still pending → release `DELETE` removes it → freed. The holdout metric was never evaluated.
- `on_peek` commits → row committed.
- Failure **at or after** the peek — the holdout `_segment_record` raising, the constructor raising,
  a `KeyboardInterrupt` anywhere downstream, the `finalize` itself raising after it committed:
  CLI's `except` calls release → `DELETE ... WHERE committed_at IS NULL` matches 0 rows → no-op →
  **burn survives**.

⇒ "holdout metric computed but window released" is impossible, asynchronous exceptions included.

## Tests

1. **Pre-burn infra failure frees the window.** Monkeypatch `provenance` (runs before `on_peek`) to
   raise; drive `research promote`; assert the reservation row was deleted and a second promote on
   the same window is permitted (not "already consumed").
2. **#193 regression — post-peek failure keeps the burn.** Monkeypatch a post-`on_peek` step
   (e.g. `WalkForwardResult` construction) to raise; assert the row is **committed** (not deleted)
   and a second promote on the same window **fails** ("already consumed").
3. **Ordering lock.** Call `walk_forward` with an `on_peek` that raises; assert the holdout
   `_segment_record` was **not** called (the peek is strictly after `on_peek`). Guards a future
   refactor from moving the hook past the peek.
4. **No behavior drift.** Existing `walk_forward` success-path + determinism tests stay green
   (the `on_peek=None` path is unchanged).

Full gate after every commit: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`.

## Out of scope / deferred follow-ups

- **Audit surface for committed-burn-without-gate-row.** A crash between `on_peek` and `run_gate`
  leaves a burned window with no `gate_evaluations` row (fail-closed, conservative). Listing/auditing
  such rows is a separate feature — defer.
- **Stale pending-row hygiene.** The tiny async-exception window between `reserve_holdout` returning
  and the `try` can leave a *never-peeked* pending row blocking reuse. Not the #193 bug; pre-existing;
  irreducible residual. Left as-is.
- **Silent `except Exception: pass` around release.** Pre-existing #161 code; a best-effort warning
  is a possible later nicety but is not changed here (JSON stdout contract; not this bug).
- **#192** (holdout_frac re-burn) is independent and untouched.
