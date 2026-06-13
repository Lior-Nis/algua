# Exhaustive `signal_panel` Parity Gate at Promotion (#178) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A custom `signal_panel` that diverges from its per-bar `signal` twin on *any* bar cannot pass `research promote` (backtested→candidate), while ordinary backtests keep the cheap bounded 16-bar sample.

**Architecture:** Add a standalone engine verifier `verify_signal_panel_parity` that runs the panel in *static* mode over the promotion window and compares the full fast-path weight matrix to the canonical per-bar loop on every bar. Call it inside the protected `promotion_preflight`, before the holdout is touched. A behavior-preserving refactor first extracts the fast-path weight computation (`_fast_weights`) from its bounded runtime guard so the verifier gets a clean exhaustive error path.

**Tech Stack:** Python, pandas, pytest. Engine: `algua/backtest/engine.py`. Gate: `algua/registry/promotion.py`. CLI: `algua/cli/research_cmd.py`.

**Spec:** `docs/superpowers/specs/2026-06-13-exhaustive-signal-panel-parity-promotion-issue-178-design.md`

**Quality gate (run between tasks):** `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`

**Worktree:** create off `origin/main` at execution time (REQUIRED SUB-SKILL: superpowers:using-git-worktrees). Suggested branch: `178-exhaustive-signal-panel-parity`.

---

## Key facts the engineer needs (read once)

- `_decision_weights(strategy, bars, adj)` — the canonical per-bar loop; with default `universe_by_date=None, fundamentals=None` it computes `construct(signal(view), view)` for every evaluated bar (static mode). Holds the first `warmup_bars` rows flat. Returns a `pd.DataFrame(index=adj.index, columns=adj.columns)` zero-filled.
- `_decision_weights_fast(strategy, bars, adj)` — the vectorized fast path: calls `signal_panel(bars)` once, applies `construct` per bar, then calls `_assert_parity(...)` (bounded 16-bar sample). Same shape/zero-fill as the loop.
- `_assert_parity(strategy, bars_sorted, end_pos, weights, warmup)` — bounded guard; compares fast weights to `_canonical_row` on `_parity_sample_positions(warmup, n)` only.
- `_parity_sample_positions(warmup, n)` — deterministic bounded sample of evaluated-bar positions (`_PARITY_SAMPLE = 16`).
- `WEIGHT_TOL` (from `algua.risk.limits`) — comparison tolerance (rtol=0).
- `BacktestError(RuntimeError)` — raised by the engine on parity failure; the `promote` CLI is decorated `@json_errors(ValueError, LookupError, BacktestError)`, so a `BacktestError` propagating out of `promotion_preflight` renders as a JSON error **without** any wrapping. Do **not** wrap it.
- `promotion_preflight` already calls `load_strategy(name)` → `_loaded` for the `needs_fundamentals` wall. Reuse that object; do not reload.
- Fundamentals strategies never reach the verifier: the `needs_fundamentals` wall refuses them first, and the loader rejects a `signal_panel` on a fundamentals strategy.
- Test fixtures live in `tests/test_fast_path.py`: `SyntheticProvider(seed=...)` (from `algua.backtest._sample`), `_passthrough` construct, the `LoadedStrategy(config=..., signal_fn=..., signal_panel_fn=..., construct_fn=...)` pattern, and `START = datetime(2024,1,1,tzinfo=UTC)`, `END = datetime(2024,6,1,tzinfo=UTC)`.
- Loaded strategy modules must use a **registered** construction policy name (`equal_weight_positive`, `top_k_equal_weight`, `score_proportional_long`). `passthrough` is NOT registered — only usable via a direct `construct_fn=` on a hand-built `LoadedStrategy`.

---

## File Structure

- **Modify** `algua/backtest/engine.py` — extract `_fast_weights`; add `verify_signal_panel_parity`.
- **Modify** `algua/registry/promotion.py` — `promotion_preflight` gains `provider, start, end`; calls the verifier on `_loaded`.
- **Modify** `algua/cli/research_cmd.py` — pass `provider, start_dt, end_dt` into `promotion_preflight`.
- **Modify** `tests/test_fast_path.py` — unit tests for `_fast_weights` decoupling + `verify_signal_panel_parity`.
- **Modify** `tests/test_promotion.py` — update 4 existing `promotion_preflight` callers with the new args; add 2 integration tests (divergent refuses pre-holdout; faithful passes the gate).

---

## Task 1: Extract `_fast_weights` from `_decision_weights_fast` (behavior-preserving refactor)

**Files:**
- Modify: `algua/backtest/engine.py` (`_decision_weights_fast`, lines ~184-230)
- Test: `tests/test_fast_path.py`

- [ ] **Step 1: Write the failing test** (proves the bounded guard is decoupled — `_fast_weights` computes weights WITHOUT raising on a divergent panel; the bounded guard lives only in `_decision_weights_fast`)

Add to `tests/test_fast_path.py` (extend the existing `algua.backtest.engine` import to include `_fast_weights` and `_assert_parity`):

```python
def test_fast_weights_skips_bounded_guard() -> None:
    """`_fast_weights` returns the fast-path matrix WITHOUT the bounded parity guard, so a panel
    that diverges only where the bounded sample does not look does NOT raise here — the guard is
    `_decision_weights_fast`'s job, decoupled so the exhaustive verifier owns its own comparison."""
    def good_loop(view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        syms = sorted(view["symbol"].unique())
        return pd.Series(0.5, index=syms)

    bars, adj = _bars_adj(["AAA", "BBB"], seed=2)
    n = len(adj.index)
    sample = set(_parity_sample_positions(0, n))
    target = next(i for i in range(0, n) if i not in sample)  # an UNSAMPLED evaluated bar
    target_ts = adj.index[target]

    def sneaky_panel(bars_: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
        a = bars_.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
        out = pd.DataFrame(0.5, index=a.index, columns=a.columns)
        out.loc[target_ts, "AAA"] = 1.0  # diverge from equal-weight only here
        out.loc[target_ts, "BBB"] = 0.0
        return out

    cfg = StrategyConfig(
        name="sneaky", universe=["AAA", "BBB"],
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1, warmup_bars=0),
        params={}, construction="passthrough",
    )
    strat = LoadedStrategy(
        config=cfg, signal_fn=good_loop, signal_panel_fn=sneaky_panel, construct_fn=_passthrough
    )
    # `_fast_weights` does NOT raise (no bounded guard); it returns the divergent matrix as-is.
    fast = _fast_weights(strat, bars, adj)
    assert fast.loc[target_ts, "AAA"] == 1.0
    # The bounded guard at the unsampled bar also passes (documents the gap the verifier closes).
    bars_sorted = bars.sort_index()
    end_pos = bars_sorted.index.searchsorted(adj.index, side="right")
    _assert_parity(strat, bars_sorted, end_pos, fast, 0)  # no raise — sample misses target
```

Update the import block near the top of `tests/test_fast_path.py` (Task 1 adds the first four new
names; `verify_signal_panel_parity` is added by Task 2 — do NOT import it yet, it does not exist):

```python
from algua.backtest.engine import (
    BacktestError,
    _assert_parity,
    _decision_weights,
    _decision_weights_fast_or_loop,
    _fast_weights,
    _parity_sample_positions,
    simulate,
)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_fast_path.py::test_fast_weights_skips_bounded_guard -q`
Expected: FAIL with `ImportError: cannot import name '_fast_weights'` (and `verify_signal_panel_parity`).

- [ ] **Step 3: Refactor `_decision_weights_fast`** — move its body into `_fast_weights`, leave `_decision_weights_fast` as `_fast_weights` + the existing `_assert_parity` call.

Replace the existing `_decision_weights_fast` function in `algua/backtest/engine.py` with:

```python
def _fast_weights(
    strategy: LoadedStrategy, bars: pd.DataFrame, adj: pd.DataFrame
) -> pd.DataFrame:
    """Vectorized fast-path WEIGHTS, without the bounded runtime parity guard. Calls the strategy's
    `signal_panel` ONCE for the whole period to get the SCORES matrix, then applies the construction
    policy PER BAR (with the same expanding `view_t` the loop uses) + the shared risk walls. Static
    universe only; pre-lag, like the loop.

    The scores matrix is NOT NaN-filled before construction — a missing score means 'no opinion' and
    the policy drops it; only the FINAL weights are zero-filled to flat. The parity guard is applied
    by the caller (`_decision_weights_fast` for the bounded runtime check; `verify_signal_panel_
    parity` for the exhaustive promotion gate), so this function never falls back silently."""
    panel = strategy.signal_panel(bars)
    assert panel is not None  # caller guarantees signal_panel_fn is set
    if not isinstance(panel, pd.DataFrame):
        raise BacktestError(
            f"strategy {strategy.name!r} signal_panel returned "
            f"{type(panel).__name__}, expected a DataFrame"
        )
    columns = adj.columns
    warmup = strategy.execution.warmup_bars
    # Reindex the SCORES onto the simulation grid WITHOUT filling NaN (missing score != 0 score).
    scores = panel.reindex(index=adj.index, columns=columns)

    bars_sorted = bars.sort_index()
    end_pos = bars_sorted.index.searchsorted(adj.index, side="right")

    weights = pd.DataFrame(0.0, index=adj.index, columns=columns)
    for i, (t, stop) in enumerate(zip(adj.index, end_pos, strict=True)):
        if i < warmup:
            continue  # warm-up: held flat by SKIPPING construction (weights stay 0)
        view_t = bars_sorted.iloc[:stop]
        scores_row = scores.iloc[i].dropna()  # drop missing/NaN; policy also drops non-finite
        w = strategy.construct(scores_row, view_t)
        if len(w) == 0:
            continue
        try:
            validate_decision_weights(w, strategy.execution, strategy.name)
        except RiskBreach as breach:
            raise BacktestError(f"{breach.detail} at {t}") from breach
        row = w.reindex(columns).fillna(0.0)
        weights.loc[t, row.index] = row.to_numpy()
    return weights


def _decision_weights_fast(
    strategy: LoadedStrategy, bars: pd.DataFrame, adj: pd.DataFrame
) -> pd.DataFrame:
    """Vectorized fast path used by ordinary backtests: `_fast_weights` followed by the fail-closed
    WEIGHT-level parity guard on a bounded deterministic sample (`_assert_parity`). The fast path is
    never trusted without that guard and never silently falls back. The promotion gate uses
    `verify_signal_panel_parity` instead, which checks EVERY bar."""
    weights = _fast_weights(strategy, bars, adj)
    bars_sorted = bars.sort_index()
    end_pos = bars_sorted.index.searchsorted(adj.index, side="right")
    _assert_parity(strategy, bars_sorted, end_pos, weights, strategy.execution.warmup_bars)
    return weights
```

- [ ] **Step 4: Run the refactor test + the full fast-path suite (regression)**

Run: `uv run pytest tests/test_fast_path.py -q`
Expected: PASS (the new `test_fast_weights_skips_bounded_guard` passes; all existing fast-path tests stay green — behavior preserved). The module imports cleanly because Task 1's import block does NOT yet reference `verify_signal_panel_parity` (added in Task 2).

- [ ] **Step 5: Commit**

```bash
git add algua/backtest/engine.py tests/test_fast_path.py
git commit -m "refactor(178): extract _fast_weights from _decision_weights_fast (behavior-preserving)"
```

---

## Task 2: Add `verify_signal_panel_parity` (exhaustive static-mode verifier)

**Files:**
- Modify: `algua/backtest/engine.py` (add `verify_signal_panel_parity` after `_decision_weights_fast_or_loop`)
- Test: `tests/test_fast_path.py`

- [ ] **Step 1: Write the failing tests**

First add `verify_signal_panel_parity` to the `algua.backtest.engine` import block in
`tests/test_fast_path.py` (it now exists as of this task). Then add to `tests/test_fast_path.py`:

```python
# --- exhaustive parity gate (#178): every-bar panel-vs-loop check for promotion ---------------


def test_verifier_catches_divergence_on_unsampled_bar() -> None:
    """The crux: a panel diverging on a bar the bounded sample never inspects passes the runtime
    guard but MUST be caught by the exhaustive verifier."""
    def good_loop(view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        syms = sorted(view["symbol"].unique())
        return pd.Series(0.5, index=syms)

    _, adj = _bars_adj(["AAA", "BBB"], seed=2)
    n = len(adj.index)
    sample = set(_parity_sample_positions(0, n))
    target_ts = adj.index[next(i for i in range(0, n) if i not in sample)]

    def sneaky_panel(bars_: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
        a = bars_.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
        out = pd.DataFrame(0.5, index=a.index, columns=a.columns)
        out.loc[target_ts, "AAA"] = 1.0
        out.loc[target_ts, "BBB"] = 0.0
        return out

    cfg = StrategyConfig(
        name="sneaky", universe=["AAA", "BBB"],
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1, warmup_bars=0),
        params={}, construction="passthrough",
    )
    strat = LoadedStrategy(
        config=cfg, signal_fn=good_loop, signal_panel_fn=sneaky_panel, construct_fn=_passthrough
    )
    with pytest.raises(BacktestError, match="parity"):
        verify_signal_panel_parity(strat, SyntheticProvider(seed=2), START, END)


def test_verifier_passes_for_faithful_panel() -> None:
    """A panel equal to its per-bar twin everywhere passes (returns None, no raise)."""
    def equal_loop(view: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        syms = sorted(view["symbol"].unique())
        return pd.Series(0.5, index=syms)

    def faithful_panel(bars_: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
        a = bars_.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
        return pd.DataFrame(0.5, index=a.index, columns=a.columns)

    cfg = StrategyConfig(
        name="faithful", universe=["AAA", "BBB"],
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1, warmup_bars=0),
        params={}, construction="passthrough",
    )
    strat = LoadedStrategy(
        config=cfg, signal_fn=equal_loop, signal_panel_fn=faithful_panel, construct_fn=_passthrough
    )
    assert verify_signal_panel_parity(strat, SyntheticProvider(seed=2), START, END) is None


def test_verifier_noop_when_no_signal_panel_fn() -> None:
    """No signal_panel_fn -> nothing to verify; the verifier returns WITHOUT touching the provider."""
    class _BoomProvider:
        def get_bars(self, *a: Any, **k: Any) -> pd.DataFrame:
            raise AssertionError("provider must not be called when there is no signal_panel_fn")

    cfg = StrategyConfig(
        name="nopanel", universe=["AAA"],
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1, warmup_bars=0),
        params={}, construction="passthrough",
    )
    strat = LoadedStrategy(
        config=cfg, signal_fn=lambda v, p: pd.Series({"AAA": 1.0}),
        signal_panel_fn=None, construct_fn=_passthrough,
    )
    assert verify_signal_panel_parity(strat, _BoomProvider(), START, END) is None


def test_verifier_raises_on_empty_provider() -> None:
    """A provider with no bars for a panel strategy fails the gate (mirrors simulate's guard)."""
    class _EmptyProvider:
        def get_bars(self, *a: Any, **k: Any) -> pd.DataFrame:
            return pd.DataFrame()

    def faithful_panel(bars_: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
        a = bars_.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
        return pd.DataFrame(0.5, index=a.index, columns=a.columns)

    cfg = StrategyConfig(
        name="empty", universe=["AAA"],
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1, warmup_bars=0),
        params={}, construction="passthrough",
    )
    strat = LoadedStrategy(
        config=cfg, signal_fn=lambda v, p: pd.Series({"AAA": 1.0}),
        signal_panel_fn=faithful_panel, construct_fn=_passthrough,
    )
    with pytest.raises(BacktestError, match="no bars"):
        verify_signal_panel_parity(strat, _EmptyProvider(), START, END)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_fast_path.py -k verifier -q`
Expected: FAIL (`verify_signal_panel_parity` not defined / not importable).

- [ ] **Step 3: Implement `verify_signal_panel_parity`**

Insert into `algua/backtest/engine.py` immediately after `_decision_weights_fast_or_loop` (before `_fetch_symbols`):

```python
def verify_signal_panel_parity(
    strategy: LoadedStrategy, provider: DataProvider, start: datetime, end: datetime
) -> None:
    """EXHAUSTIVE fail-closed parity gate for promotion: assert `signal_panel` agrees with its
    per-bar `signal` twin on EVERY evaluated bar (not the bounded runtime sample).

    Verifies a CODE property — that the vectorized fast path equals the canonical per-bar
    `construct(signal(view), view)` — in STATIC mode over the strategy's declared universe. The
    agent promote backtest runs under PIT (which forces the loop and never exercises the panel) or,
    if `--universe` is omitted, may run the fast path; either way the panel must be checked here
    directly, where the fast path is the thing under test. No-op when `signal_panel_fn is None`.
    Raises `BacktestError` naming the first divergent bar + offending symbol(s)."""
    if strategy.signal_panel_fn is None:
        return  # nothing to verify

    try:
        bars = provider.get_bars(strategy.universe, start, end, "1d")
    except Exception as exc:
        raise BacktestError(f"provider error during parity check: {exc}") from exc
    if bars.empty:
        raise BacktestError("provider returned no bars for the signal_panel parity check")

    adj = bars.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
    adj = adj.sort_index()

    fast = _fast_weights(strategy, bars, adj)
    loop = _decision_weights(strategy, bars, adj)  # static: universe_by_date=None, fundamentals=None

    # Identical grid by construction (both built on adj.index/columns); assert it before comparing.
    if not (fast.index.equals(loop.index) and fast.columns.equals(loop.columns)):
        raise BacktestError(
            f"signal_panel parity check for {strategy.name!r}: fast/loop weight grids differ "
            f"(fast {fast.shape} vs loop {loop.shape})"
        )

    # NaN-safe, every-bar comparison. Both paths zero-fill final weights so a NaN cannot survive;
    # the isna() guard is defensive belt-and-suspenders against a future path that leaves one.
    nan_mismatch = fast.isna() != loop.isna()
    diff = (loop - fast).abs()
    bad = nan_mismatch | (diff > WEIGHT_TOL)
    if bool(bad.to_numpy().any()):
        first = next(t for t in fast.index if bool(bad.loc[t].any()))
        offenders = sorted(bad.columns[bad.loc[first].to_numpy()])
        raise BacktestError(
            f"signal_panel exhaustive parity check FAILED for strategy {strategy.name!r} at "
            f"{first}: signal_panel disagrees with the per-bar construct(signal(view), view) on "
            f"symbol(s) {offenders} "
            f"(per-bar={loop.loc[first, offenders].to_dict()}, "
            f"panel={fast.loc[first, offenders].to_dict()}, tol={WEIGHT_TOL})"
        )
```

(`DataProvider`, `datetime`, and `WEIGHT_TOL` are already imported at the top of `engine.py`.)

- [ ] **Step 4: Run the verifier tests + the full fast-path module**

Run: `uv run pytest tests/test_fast_path.py -q`
Expected: PASS (all verifier tests + the Task-1 refactor test + existing fast-path tests).

- [ ] **Step 5: Commit**

```bash
git add algua/backtest/engine.py tests/test_fast_path.py
git commit -m "feat(178): exhaustive signal_panel parity verifier (every-bar, static mode)"
```

---

## Task 3: Wire the verifier into `promotion_preflight` + CLI + tests

**Files:**
- Modify: `algua/registry/promotion.py` (`promotion_preflight` signature + call; imports)
- Modify: `algua/cli/research_cmd.py` (pass new args, line ~100-102)
- Test: `tests/test_promotion.py`

- [ ] **Step 1: Write the failing integration tests**

Add to `tests/test_promotion.py`. Extend the imports at the top:

```python
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

import algua.strategies.momentum as momentum_pkg
from algua.backtest._sample import SyntheticProvider
from algua.backtest.engine import BacktestError
```

Add `_START`/`_END` constants near the top of the file (after imports):

```python
_START = datetime(2024, 1, 1, tzinfo=UTC)
_END = datetime(2024, 6, 1, tzinfo=UTC)
```

Then add the two integration tests:

```python
def _write_tmp_strategy(filename: str, body: str) -> Path:
    """Write a temp strategy module into the momentum family dir so load_strategy can find it.
    Caller must unlink it in a finally block (mirrors tests/test_fast_path.py loader tests)."""
    p = Path(momentum_pkg.__path__[0]) / filename
    p.write_text(body)
    return p


def test_preflight_refuses_divergent_signal_panel_before_holdout(tmp_path):
    """A strategy whose signal_panel diverges from its per-bar signal is refused in preflight —
    before any holdout or gate row — by the exhaustive parity gate (#178)."""
    mod = _write_tmp_strategy(
        "tmp_divergent_panel.py",
        "from typing import Any\n"
        "import pandas as pd\n"
        "from algua.contracts.types import ExecutionContract\n"
        "from algua.strategies.base import StrategyConfig\n"
        "CONFIG = StrategyConfig(name='tmp_divergent_panel', universe=['AAA', 'BBB'],\n"
        "    execution=ExecutionContract(rebalance_frequency='1d', decision_lag_bars=1, warmup_bars=0),\n"
        "    construction='equal_weight_positive')\n"
        "def signal(view, params):\n"
        "    syms = sorted(view['symbol'].unique())\n"
        "    return pd.Series(1.0, index=syms)\n"  # both positive -> equal weight 0.5/0.5
        "def signal_panel(bars, params):\n"
        "    adj = bars.reset_index().pivot(index='timestamp', columns='symbol', values='adj_close')\n"
        "    out = pd.DataFrame(0.0, index=adj.index, columns=adj.columns)\n"
        "    out['AAA'] = 1.0\n"  # only AAA positive -> 100% AAA, diverges from 0.5/0.5
        "    return out\n",
    )
    try:
        repo = _repo(tmp_path)
        rec = repo.add("tmp_divergent_panel")
        repo.apply_transition(rec, Stage.BACKTESTED, Actor.AGENT, "bt")
        repo.record_search_trial("tmp_divergent_panel", 4, "{}")  # breadth present; isolate parity
        with pytest.raises(BacktestError, match="parity"):
            promotion_preflight(
                repo, "tmp_divergent_panel", actor=Actor.AGENT, declared_combos=None,
                allow_holdout_reuse=False, allow_non_pit=False,
                provider=SyntheticProvider(seed=2), start=_START, end=_END)
        # Refused before walk_forward -> no holdout/gate rows.
        assert repo._conn.execute("SELECT COUNT(*) c FROM gate_evaluations").fetchone()["c"] == 0
        assert repo._conn.execute("SELECT COUNT(*) c FROM holdout_evaluations").fetchone()["c"] == 0
    finally:
        mod.unlink()


def test_preflight_passes_parity_for_faithful_bundled_strategy(tmp_path):
    """A real bundled strategy with a faithful signal_panel passes the parity gate and preflight
    resolves breadth as usual (no false positive)."""
    repo = _repo(tmp_path)
    rec = repo.add("cross_sectional_momentum")
    repo.apply_transition(rec, Stage.BACKTESTED, Actor.AGENT, "bt")
    repo.record_search_trial("cross_sectional_momentum", 4, "{}")
    ctx = promotion_preflight(
        repo, "cross_sectional_momentum", actor=Actor.AGENT, declared_combos=None,
        allow_holdout_reuse=False, allow_non_pit=False,
        provider=SyntheticProvider(seed=7), start=_START, end=_END)
    assert ctx.own == 4 and ctx.provenance == "measured"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_promotion.py -k "divergent or faithful_bundled" -q`
Expected: FAIL — `promotion_preflight() got an unexpected keyword argument 'provider'`.

- [ ] **Step 3: Update `promotion_preflight` signature + call + imports**

In `algua/registry/promotion.py`:

3a. Change the date import line:
```python
from datetime import date
```
to:
```python
from datetime import date, datetime
```

3b. Add these imports after the existing `from algua.backtest.walkforward import WalkForwardResult` line:
```python
from algua.backtest.engine import verify_signal_panel_parity
from algua.contracts.types import DataProvider
```

3c. Extend the `promotion_preflight` signature (add three required keyword args):
```python
def promotion_preflight(
    repo: StrategyRepository,
    name: str,
    *,
    actor: Actor,
    declared_combos: int | None,
    allow_holdout_reuse: bool,
    allow_non_pit: bool,
    provider: DataProvider,
    start: datetime,
    end: datetime,
) -> BreadthContext:
```

3d. Insert the parity call immediately after the `needs_fundamentals` refusal block (after the `if _loaded is not None and _loaded.config.needs_fundamentals: raise ...` block, before `measured = repo.total_search_combos(name)`):
```python
    # Exhaustive signal_panel parity gate (#178): a panel that diverges from its per-bar signal on
    # ANY bar cannot pass promotion. Runs on the already-loaded strategy, in static mode over the
    # promotion window, BEFORE the holdout is touched. No-op when the strategy has no signal_panel.
    # Raises BacktestError on divergence (caught by the `promote` CLI's @json_errors).
    if _loaded is not None:
        verify_signal_panel_parity(_loaded, provider, start, end)
```

Also extend the `promotion_preflight` docstring's enumerated list to mention `(4) exhaustive signal_panel parity (no-op without a panel)`.

- [ ] **Step 4: Update the CLI caller**

In `algua/cli/research_cmd.py`, change the `promotion_preflight(...)` call (lines ~100-102) from:
```python
        breadth = promotion_preflight(
            repo, name, actor=actor_enum, declared_combos=n_combos,
            allow_holdout_reuse=allow_holdout_reuse, allow_non_pit=allow_non_pit)
```
to:
```python
        breadth = promotion_preflight(
            repo, name, actor=actor_enum, declared_combos=n_combos,
            allow_holdout_reuse=allow_holdout_reuse, allow_non_pit=allow_non_pit,
            provider=provider, start=start_dt, end=end_dt)
```

- [ ] **Step 5: Update the 4 existing `promotion_preflight` test callers**

The new args are required. Update each existing call in `tests/test_promotion.py` (and `tests/test_fundamentals_guards.py`) to pass `provider=SyntheticProvider(seed=0), start=_START, end=_END`. These tests use synthetic names that do NOT resolve via `load_strategy` (so `_loaded is None` → the verifier is a no-op and the provider is unused).

In `tests/test_promotion.py`, the calls at the existing tests `test_preflight_refuses_non_backtested_source` (line ~67), `test_preflight_refuses_system_actor_before_any_holdout_or_gate_row` (~82), `test_preflight_refuses_agent_without_measured_breadth` (~93), `test_preflight_resolves_measured_funnel_breadth` (~103): append `, provider=SyntheticProvider(seed=0), start=_START, end=_END` to each.

In `tests/test_fundamentals_guards.py::test_promotion_preflight_blocks_fundamentals` (call at ~line 36): this loads a real fundamentals strategy. The `needs_fundamentals` wall fires BEFORE the parity call, so the provider is still unused — append `, provider=SyntheticProvider(seed=0), start=datetime(2024,1,1,tzinfo=UTC), end=datetime(2024,6,1,tzinfo=UTC)` (add the imports `from datetime import UTC, datetime` and `from algua.backtest._sample import SyntheticProvider` to that file if absent).

- [ ] **Step 6: Run the full promotion + fundamentals-guard suites**

Run: `uv run pytest tests/test_promotion.py tests/test_fundamentals_guards.py tests/test_fast_path.py -q`
Expected: PASS (new integration tests pass; the 4 updated callers stay green).

- [ ] **Step 7: Commit**

```bash
git add algua/registry/promotion.py algua/cli/research_cmd.py tests/test_promotion.py tests/test_fundamentals_guards.py
git commit -m "feat(178): gate research promote on exhaustive signal_panel parity (preflight)"
```

---

## Task 4: Full quality gate + final verification

**Files:** none (verification only)

- [ ] **Step 1: Run the full quality gate**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: all green. In particular `lint-imports` must pass — `algua/registry/promotion.py` importing `algua.backtest.engine` is the same registry→backtest layer it already crosses via `walkforward`.

- [ ] **Step 2: Manual smoke (optional, documents the gate end-to-end)**

The unit + integration tests already prove the gate; no extra CLI smoke is required. If desired, confirm a faithful bundled strategy still backtests with the bounded sample:
Run: `uv run pytest tests/test_fast_path.py::test_cross_sectional_momentum_full_parity -q`
Expected: PASS (ordinary fast-path parity unchanged).

- [ ] **Step 3: Confirm clean tree + push readiness**

Run: `git status --short` (expect only intended changes) and `git log --oneline origin/main..HEAD` (expect the 3 feature commits).

---

## Self-review notes (coverage vs spec)

- Spec "standalone static verifier, every bar" → Task 2 (`verify_signal_panel_parity`).
- Spec "`_fast_weights` refactor, clean error path" → Task 1.
- Spec "called in `promotion_preflight` on `_loaded`, before holdout, no wrapping" → Task 3 Steps 3-4.
- Spec "NaN-safe + index/column equality + empty-bars guard" → Task 2 Step 3 + `test_verifier_raises_on_empty_provider`.
- Spec "ordinary backtests keep bounded sample" → Task 1 leaves `_decision_weights_fast` behavior identical; `test_cross_sectional_momentum_full_parity` + Task 4 Step 2.
- Spec "4 existing preflight callers updated; synthetic names → no-op" → Task 3 Step 5.
- Spec acceptance "divergent panel cannot pass promotion" → `test_verifier_catches_divergence_on_unsampled_bar` + `test_preflight_refuses_divergent_signal_panel_before_holdout`.
- Deferred (NOT in this plan): move agent PIT refusal into preflight; go-live parity.
