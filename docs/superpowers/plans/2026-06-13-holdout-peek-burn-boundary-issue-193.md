# Holdout peek/burn-boundary (#193) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Commit the single-use holdout burn the instant *before* the holdout metric is evaluated, so a failure (or `KeyboardInterrupt`) after the peek can never release a window whose metric was computed.

**Architecture:** `walk_forward` gains an opaque `on_peek` hook fired immediately before the holdout `_segment_record`. `research promote` passes `repo.finalize_holdout_reservation` as that hook and removes the now-redundant post-return finalize. The CLI's existing `except BaseException: release` becomes correct for free, because `release_holdout_reservation` is already a no-op on a committed row (`DELETE ... WHERE committed_at IS NULL`).

**Tech Stack:** Python 3.12, uv, pytest, Typer CliRunner, SQLite.

**Design spec:** `docs/superpowers/specs/2026-06-13-holdout-peek-burn-boundary-issue-193-design.md`

**Base:** worktree off `origin/main` (contains PR #191's reservation lifecycle).

**Quality gate after every commit:** `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`

---

## File structure

- `algua/backtest/walkforward.py` — add `on_peek` param to `walk_forward`; reorder so all
  holdout-independent fallible work precedes the hook; hoist `cfg_hash`. (Modify `walk_forward`,
  lines ~93–148.)
- `algua/cli/research_cmd.py` — pass `on_peek=` into `walk_forward`; delete the post-return
  `finalize_holdout_reservation` call. (Modify `promote`, lines ~116–131.)
- `tests/test_walkforward.py` — ordering/contract unit tests for the hook.
- `tests/test_cli_research.py` — pre-burn-frees and post-peek-keeps-burn CLI regression tests + a
  `committed_at` read helper.

---

## Task 1: `on_peek` hook in `walk_forward`

**Files:**
- Modify: `algua/backtest/walkforward.py` (`walk_forward`, lines ~93–148)
- Test: `tests/test_walkforward.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_walkforward.py` (it already imports `walk_forward`, `WalkForwardResult`,
`_equal_weight`, `SyntheticProvider`, `START`, `END`):

```python
import pytest

import algua.backtest.walkforward as wfmod


def test_on_peek_fires_before_holdout_eval(monkeypatch):
    # Spy on _segment_record to count holdout-metric evaluations.
    calls = []
    orig = wfmod._segment_record

    def spy(returns, s, e):
        calls.append((s, e))
        return orig(returns, s, e)

    monkeypatch.setattr(wfmod, "_segment_record", spy)

    def boom(_cfg_hash):
        raise RuntimeError("burn failed")

    with pytest.raises(RuntimeError, match="burn failed"):
        walk_forward(_equal_weight(), SyntheticProvider(seed=3), START, END,
                     windows=4, holdout_frac=0.2, on_peek=boom)

    # on_peek raised before the holdout was evaluated: only the 4 in-sample windows were recorded,
    # NOT a 5th (holdout) evaluation. Proves the burn point is strictly before the peek.
    assert len(calls) == 4


def test_on_peek_receives_config_hash_and_completes(monkeypatch):
    seen = []
    res = walk_forward(_equal_weight(), SyntheticProvider(seed=3), START, END,
                       windows=4, holdout_frac=0.2, on_peek=lambda cfg: seen.append(cfg))
    # Fired exactly once, with the same config_hash that lands in the result, and the run completed.
    assert seen == [res.config_hash]
    assert res.holdout_metrics  # the peek still happened on the success path


def test_walk_forward_without_on_peek_unchanged():
    # Default (on_peek=None) path is byte-identical to a second run.
    a = walk_forward(_equal_weight(), SyntheticProvider(seed=3), START, END)
    b = walk_forward(_equal_weight(), SyntheticProvider(seed=3), START, END, on_peek=None)
    assert a.to_dict() == b.to_dict()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_walkforward.py -q`
Expected: FAIL — `walk_forward() got an unexpected keyword argument 'on_peek'`.

- [ ] **Step 3: Implement the hook + reorder**

In `algua/backtest/walkforward.py`, add the import (top of file, with the other `typing` import):

```python
from collections.abc import Callable, Collection, Mapping
```

(`Collection` and `Mapping` are already imported from `collections.abc`; add `Callable` to that line.)

Change the `walk_forward` signature to add the keyword-only param (after `universe_snapshots`):

```python
    universe_snapshots: list[dict[str, str]] | None = None,
    on_peek: Callable[[str], None] | None = None,
) -> WalkForwardResult:
```

Extend the docstring with one sentence:

```
    ``on_peek`` (if given) is called exactly once, with the strategy ``config_hash``, immediately
    BEFORE the holdout window is evaluated. It is the burn point for a single-use holdout: a caller
    that commits a durable "burn" here can rely on nothing fallible-and-releasing running after it.
```

Replace the body from the `holdout_metrics = ...` line through the `return` with this reordered
version (all holdout-independent fallible work first, hook, then the peek, then construct):

```python
    sharpes = [w["sharpe"] for w in window_metrics]
    positive = sum(1 for w in window_metrics if w["total_return"] > 0)
    stability = {
        "mean_sharpe": float(np.mean(sharpes)),
        "std_sharpe": float(np.std(sharpes)),
        "min_sharpe": float(np.min(sharpes)),
        "pct_positive_windows": float(positive / len(window_metrics)),
    }
    stamps = runtime_stamps()
    prov = provenance(provider, seed)
    cfg_hash = config_hash(strategy)

    # Burn-on-peek boundary: the holdout metric is evaluated on the NEXT line, so any single-use
    # burn the caller commits in on_peek is durable before the peek. Nothing fallible-and-releasing
    # may be added between on_peek and the holdout evaluation.
    if on_peek is not None:
        on_peek(cfg_hash)
    holdout_metrics = _segment_record(returns, holdout[0], holdout[1])

    return WalkForwardResult(
        strategy=strategy.name,
        config_hash=cfg_hash,
        timeframe="1d",
        code_hash=stamps["code_hash"],
        dependency_hash=stamps["dependency_hash"],
        period={"start": start.date().isoformat(), "end": end.date().isoformat()},
        windows=windows,
        holdout_frac=holdout_frac,
        window_metrics=window_metrics,
        holdout_metrics=holdout_metrics,
        stability=stability,
        universe_name=universe_name,
        universe_snapshots=universe_snapshots,
        **prov,
    )
```

Note: this removes the inline `config_hash=config_hash(strategy)` (now the hoisted `cfg_hash`) and
moves the `holdout_metrics` computation down to just before the return. `window_metrics` and the
`sharpes`/`positive`/`stability` block are unchanged in value.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_walkforward.py tests/test_walkforward_segment.py tests/test_walkforward_metrics.py tests/test_tracking_walkforward.py -q`
Expected: PASS (new hook tests + all pre-existing walk_forward tests).

- [ ] **Step 5: Run the full gate**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: all green (no behavior change for any existing caller — `on_peek` defaults `None`).

- [ ] **Step 6: Commit**

```bash
git add algua/backtest/walkforward.py tests/test_walkforward.py
git commit -m "feat(193): on_peek burn-boundary hook in walk_forward

Fire an opaque on_peek(config_hash) immediately before the holdout metric is
evaluated; reorder so all holdout-independent work precedes it. Default None
keeps every existing caller byte-identical.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: wire the burn at the peek boundary in `research promote`

**Files:**
- Modify: `algua/cli/research_cmd.py` (`promote`, lines ~116–131)
- Test: `tests/test_cli_research.py`

- [ ] **Step 1: Write the failing regression tests**

Append to `tests/test_cli_research.py` (it already has `runner`, `_tmp`, `_backtest_to_backtested`,
`_stage`, `_PASS`, `_holdout_rows`, and imports `json`, `app`). Add a `committed_at` reader and two
tests:

```python
def _holdout_committed(tmp_path):
    """(id, committed_at) for every holdout_evaluations row — committed_at is None while pending."""
    import sqlite3
    conn = sqlite3.connect(tmp_path / "r.db")
    try:
        return conn.execute(
            "SELECT id, committed_at FROM holdout_evaluations ORDER BY id").fetchall()
    finally:
        conn.close()


def test_pre_burn_failure_frees_window(tmp_path, monkeypatch):
    # A failure BEFORE the burn boundary (provenance runs before on_peek) must release the
    # reservation, so the window is reusable: no row left behind, and a clean retry succeeds.
    assert _backtest_to_backtested().exit_code == 0
    import algua.backtest.walkforward as wfmod
    monkeypatch.setattr(wfmod, "provenance",
                        lambda *a, **k: (_ for _ in ()).throw(ValueError("provenance boom")))
    bad = runner.invoke(app, ["research", "promote", "cross_sectional_momentum", "--demo",
                              "--start", "2022-01-01", "--end", "2023-12-31",
                              *_PASS, "--n-combos", "9", "--allow-non-pit", "--actor", "human"])
    assert bad.exit_code == 1, bad.stdout
    assert json.loads(bad.stdout)["ok"] is False
    assert _holdout_committed(tmp_path) == []  # reservation released, window free
    assert _stage() == "backtested"

    # Undo the fault; the same window now promotes cleanly (proves it was genuinely freed).
    monkeypatch.undo()
    good = runner.invoke(app, ["research", "promote", "cross_sectional_momentum", "--demo",
                               "--start", "2022-01-01", "--end", "2023-12-31",
                               *_PASS, "--n-combos", "9", "--allow-non-pit", "--actor", "human"])
    assert good.exit_code == 0, good.stdout
    assert json.loads(good.stdout)["promoted"] is True
    assert _stage() == "candidate"


def test_post_peek_failure_keeps_burn(tmp_path, monkeypatch):
    # A failure AFTER the burn boundary (WalkForwardResult construction raises, post on_peek) must
    # KEEP the burn: the row stays committed and the same window is refused on retry (#193).
    assert _backtest_to_backtested().exit_code == 0
    import algua.backtest.walkforward as wfmod

    def boom(*a, **k):
        raise ValueError("post-peek boom")

    monkeypatch.setattr(wfmod, "WalkForwardResult", boom)
    bad = runner.invoke(app, ["research", "promote", "cross_sectional_momentum", "--demo",
                              "--start", "2022-01-01", "--end", "2023-12-31",
                              *_PASS, "--n-combos", "9", "--allow-non-pit", "--actor", "human"])
    assert bad.exit_code == 1, bad.stdout
    assert json.loads(bad.stdout)["ok"] is False
    rows = _holdout_committed(tmp_path)
    assert len(rows) == 1 and rows[0][1] is not None  # one row, COMMITTED (burn survived)
    assert _stage() == "backtested"

    # The burned window is refused on retry (single-use holdout held).
    monkeypatch.undo()
    retry = runner.invoke(app, ["research", "promote", "cross_sectional_momentum", "--demo",
                                "--start", "2022-01-01", "--end", "2023-12-31",
                                *_PASS, "--n-combos", "9", "--allow-non-pit", "--actor", "human"])
    assert retry.exit_code == 1, retry.stdout
    assert "already consumed" in json.loads(retry.stdout)["error"]
```

- [ ] **Step 2: Run the tests to verify the regression one fails**

Run: `uv run pytest tests/test_cli_research.py::test_post_peek_failure_keeps_burn tests/test_cli_research.py::test_pre_burn_failure_frees_window -q`
Expected: `test_post_peek_failure_keeps_burn` FAILS — with the current CLI the pre-peek burn is not
committed, so the `except BaseException` releases the pending row and `_holdout_committed` is `[]`
(or the row is uncommitted). (`test_pre_burn_failure_frees_window` may already pass — it guards that
the fix does not break pre-burn freeing.)

- [ ] **Step 3: Wire the burn at the peek boundary**

In `algua/cli/research_cmd.py`, replace the `try: wf = walk_forward(...) except BaseException: ...`
block and the post-return finalize (lines ~116–131) with:

```python
        try:
            wf = walk_forward(
                strategy, provider, start_dt, end_dt, windows=windows,
                holdout_frac=holdout_frac, universe_by_date=universe_by_date,
                universe_name=universe, universe_snapshots=universe_prov,
                # Burn-on-peek: commit the reservation into a burn the instant BEFORE walk_forward
                # evaluates the holdout metric. Because release_holdout_reservation no-ops on a
                # committed row, the except-release below is then correct for EVERY post-peek
                # failure (incl. KeyboardInterrupt) — a computed holdout can never be released.
                on_peek=lambda cfg: repo.finalize_holdout_reservation(
                    reservation_id, config_hash=cfg),
            )
        except BaseException:
            # Pre-peek failure: the row is still pending, so release frees the window. Post-peek
            # failure: on_peek already committed, so this DELETE matches 0 rows (harmless no-op) and
            # the burn survives. Swallow a release error so it never masks the original failure.
            try:
                repo.release_holdout_reservation(reservation_id)
            except Exception:
                pass
            raise
        outcome = run_gate(
```

(Delete the old `repo.finalize_holdout_reservation(reservation_id, config_hash=wf.config_hash)` line
and its two-line `# Burn-on-peek:` comment — the burn now happens via `on_peek`.)

- [ ] **Step 4: Run the regression tests to verify they pass**

Run: `uv run pytest tests/test_cli_research.py::test_post_peek_failure_keeps_burn tests/test_cli_research.py::test_pre_burn_failure_frees_window -q`
Expected: PASS — post-peek failure keeps the committed burn; pre-burn failure frees the window.

- [ ] **Step 5: Run the full gate**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: all green. In particular the existing single-use-holdout tests
(`test_first_promote_records_holdout_evaluation`, `test_second_promote_same_window_refused`,
`test_gate_row_written_on_both_pass_and_fail`) stay green — the burn is still committed on a
successful promote, just earlier.

- [ ] **Step 6: Commit**

```bash
git add algua/cli/research_cmd.py tests/test_cli_research.py
git commit -m "fix(193): commit the holdout burn at the peek boundary, not after walk_forward returns

Pass repo.finalize as walk_forward's on_peek hook (fires just before the holdout
metric is evaluated) and drop the post-return finalize. Idempotent release then
makes the existing except-BaseException release correct for every post-peek
failure, KeyboardInterrupt included: a computed holdout can no longer be released.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-review notes (already checked)

- **Spec coverage:** walk_forward hook + reorder (Task 1) ✓; CLI burn-at-peek + remove post-return
  finalize (Task 2) ✓; tests 1/2/3/4 from the spec map to
  `test_pre_burn_failure_frees_window` (1), `test_post_peek_failure_keeps_burn` (2),
  `test_on_peek_fires_before_holdout_eval` (3), `test_walk_forward_without_on_peek_unchanged` +
  `test_on_peek_receives_config_hash_and_completes` (4). ✓
- **Deferred items** (audit surface, pending-row hygiene, silent release-swallow) are intentionally
  NOT in this plan — see the spec's "Out of scope" section.
- **Type consistency:** `on_peek: Callable[[str], None] | None`; hook receives `config_hash` (str),
  same value passed to `finalize_holdout_reservation(config_hash=...)`. Consistent across both tasks.
