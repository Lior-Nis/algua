# Promotion Gates — Design

**Date:** 2026-05-31
**Branch:** `research-gates`. **Status:** Approved (design); plan to follow.
**Sub-project:** 3 (research core) — "research depth", slice 4 of 4 (the finale). Builds on
walk-forward (slice 1) + sweeps (slice 2). Completes the research engine and unlocks sub-project 4
(the autonomous agent research loop).

## 1. Goal

Gate the `backtested → shortlisted` lifecycle transition on quantitative evidence, so "shortlisted"
means a strategy cleared an out-of-sample bar — not just "a backtest ran".

```
algua research promote <strategy> --snapshot <id> [--min-holdout-sharpe 0.5 ...] [--n-combos 9]
```

## 2. Decisions (from brainstorm)
- **Orchestration `promote` command**, not a registry hard-wall. Shortlisting is a research-quality
  checkpoint; the only hard-enforced wall remains `paper→live` (human approval). The command runs a
  walk-forward, evaluates the gate, and advances the registry only on pass.
- **Holdout-as-defense against search breadth:** the gate requires the strategy to clear thresholds
  on the **untouched holdout** (never used during sweep selection) + stability thresholds; `n_combos`
  is recorded as evidence, not used to alter thresholds (an explicit breadth penalty is deferred).

## 3. Components

### 3.1 `algua/research/gates.py` (new `research/` package; pure)
- `GateCriteria` (dataclass, configurable, defaults):
  - `min_holdout_sharpe: float = 0.5`
  - `min_holdout_return: float = 0.0`  (strict `>`)
  - `min_pct_positive_windows: float = 0.6`
  - `min_window_sharpe: float = 0.0`   (the worst window's Sharpe, `wf.stability["min_sharpe"]`, ≥ this)
- `GateDecision` (dataclass): `passed: bool`, `checks: list[dict]` (each
  `{name, value, threshold, op, passed}`), `n_combos: int | None`; `to_dict()`.
- `evaluate_gate(wf: WalkForwardResult, criteria: GateCriteria, *, n_combos: int | None = None)
  -> GateDecision` — pure. Builds the check list from `wf.holdout_metrics["sharpe"]`,
  `wf.holdout_metrics["total_return"]`, `wf.stability["pct_positive_windows"]`,
  `wf.stability["min_sharpe"]`; `passed` = all checks pass. Records `n_combos`.

### 3.2 `algua/cli/research_cmd.py` (new `research` sub-app)
`research promote <name> (--demo | --snapshot <id>) [--start --end --windows --holdout-frac]
[--min-holdout-sharpe F --min-holdout-return F --min-pct-positive F --min-window-sharpe F]
[--n-combos N] [--actor agent]`:
1. `load_strategy(name)`; `_select_provider(demo, snapshot)` (reuse the helpers from
   `backtest_cmd` — import them, or factor shared helpers into a small `cli/_backtest_shared`
   module if cleaner; reuse, don't duplicate).
2. `wf = walk_forward(strategy, provider, _utc(start), _utc(end), windows=, holdout_frac=)`.
3. `decision = evaluate_gate(wf, GateCriteria(...from flags...), n_combos=n_combos)`.
4. **On pass:** open the registry (`connect`+`migrate`), and
   `store.transition(name, Stage.SHORTLISTED, Actor(actor), reason=<gate summary>,
   code_hash=wf.config_hash, config_hash=wf.config_hash)`; `promoted = True`.
   **On fail:** no transition; `promoted = False`.
5. Emit JSON: `decision.to_dict()` plus `"strategy"`, `"promoted"`, `"holdout"` + `"stability"`
   echoes, and `wf`'s `config_hash`/`snapshot_id`. **Exit 0 whenever the gate evaluates** (pass or
   fail — failing is a normal outcome). Real errors (no data; strategy not at `backtested` →
   `TransitionError`; bad flags) render as `{ok:false}` exit 1 via `@json_errors(ValueError,
   LookupError, BacktestError)`.

The reason string records the evidence, e.g.
`"gate pass: holdout_sharpe=0.71>=0.5, holdout_return=4.2%>0, pct_positive=0.75>=0.6,
min_window_sharpe=0.12>=0.0; n_combos=9"`, so `registry show` displays why it was shortlisted.

### 3.3 Registry integration
No registry change. `backtested → shortlisted` is an already-legal transition; the strategy must
be at `backtested` first (via `backtest run --register`). The gate is the orchestration that
performs that transition only on pass.

### 3.4 Boundary
New import-linter contract: `algua.backtest` must NOT import `algua.research` (engine stays pure).
`research/gates.py` imports only `algua.backtest.walkforward` result type + stdlib/dataclasses;
the CLI imports `research`, `walkforward`, `registry`. (5 → 6 contracts.)

## 4. Error handling
- Strategy not registered / not at `backtested` → `store.transition` raises `TransitionError`
  (a `ValueError`) → JSON error.
- No data / too-few-bars → `BacktestError` from `walk_forward` → JSON error.
- Bad threshold flag types → Typer/`ValueError` → JSON error.
- A gate **fail** is NOT an error: emit the decision with `passed:false`, `promoted:false`, exit 0.

## 5. Testing
- `evaluate_gate`: a WF result clearing all four thresholds → `passed=true`, four checks all
  `passed`; flip each threshold individually so exactly that check fails → `passed=false` with the
  right check flagged; `n_combos` recorded in the decision.
- CLI `research promote` (synthetic provider, strategy first advanced to `backtested` in-test via
  `registry add` + `registry transition`/`backtest run --register`):
  - lenient thresholds → `promoted:true` and `registry show` reports stage `shortlisted` with the
    gate-summary reason.
  - a threshold set impossibly high → `promoted:false`, no transition (stage stays `backtested`),
    exit 0.
  - promoting a strategy that is still at `idea` (not `backtested`) → `{ok:false}` exit 1.
- Full gate green: `pytest`, `ruff`, `mypy`, `lint-imports` (6 contracts; `backtest` off
  `research`).

## 6. Out of scope (later)
- An explicit search-breadth **penalty** (deflated-Sharpe / Bonferroni-style threshold inflation by
  `n_combos`) — v1 uses the untouched holdout as the breadth defense and records `n_combos`.
- Hard-enforcing the gate inside `store.transition` (kept an orchestration checkpoint).
- Multi-strategy ranking / auto-promoting the best of several; consuming a stored sweep's `best`
  automatically (v1 promotes one named strategy at its current config).
- The autonomous agent research loop (sub-project 4) that will *call* this gate.
