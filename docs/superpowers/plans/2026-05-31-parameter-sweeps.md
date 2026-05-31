# Parameter Sweeps Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `algua backtest sweep <strategy> --param KEY=v1,v2,…` — evaluate a strategy across a parameter grid with walk-forward (out-of-sample) scoring, rank the combos, and report the search-breadth count.

**Architecture:** A new `algua/backtest/sweep.py` parses repeatable `--param` flags into a grid, takes the Cartesian product (capped), evaluates each combo by overriding the strategy's params and running the existing `walk_forward`, then ranks by an out-of-sample window metric (`mean_sharpe` by default). The holdout is carried per combo but never used for ranking. A new CLI command emits the ranked `SweepResult` as JSON.

**Tech Stack:** Python 3.12, pandas, vectorbt, Typer, pytest. Builds on `algua/backtest/walkforward.py`.

**Key existing code (branch `research-sweeps`):**
- `algua/backtest/walkforward.py::walk_forward(strategy, provider, start, end, *, windows=4, holdout_frac=0.2) -> WalkForwardResult`. `WalkForwardResult` has: `.strategy`, `.config_hash`, `.data_source`, `.snapshot_id`, `.timeframe`, `.seed`, `.period`, `.windows`, `.holdout_frac`, `.window_metrics`, `.holdout_metrics` (dict with `n_bars`/`sharpe`/`total_return`/`max_drawdown`/…), `.stability` (dict: `mean_sharpe`, `std_sharpe`, `min_sharpe`, `pct_positive_windows`).
- `algua/backtest/engine.py::BacktestError`.
- `algua/strategies/base.py`: `StrategyConfig` (pydantic v2 BaseModel; has `.params: dict`), `LoadedStrategy` (dataclass: `.config`, `.fn`, `.name`).
- `algua/cli/backtest_cmd.py`: `backtest_app`, `_utc`, `_select_provider(demo, snapshot)`, `emit`, `load_strategy`, `@json_errors(ValueError, LookupError, BacktestError)`.

---

### Task 1: `_parse_grid`

**Files:**
- Create: `algua/backtest/sweep.py`
- Test: `tests/test_sweep_parse.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sweep_parse.py
import pytest
from algua.backtest.sweep import _parse_grid


def test_parses_ints_floats_strs():
    grid = _parse_grid(["lookback=20,40,60", "rate=0.1,0.2", "mode=fast,slow"])
    assert grid == {"lookback": [20, 40, 60], "rate": [0.1, 0.2], "mode": ["fast", "slow"]}


def test_empty_list_raises():
    with pytest.raises(ValueError):
        _parse_grid([])


def test_missing_equals_raises():
    with pytest.raises(ValueError):
        _parse_grid(["lookback"])


def test_empty_key_or_values_raises():
    with pytest.raises(ValueError):
        _parse_grid(["=1,2"])
    with pytest.raises(ValueError):
        _parse_grid(["k="])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_sweep_parse.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'algua.backtest.sweep'`.

- [ ] **Step 3: Write the implementation**

```python
# algua/backtest/sweep.py
from __future__ import annotations

from typing import Any


def _coerce(value: str) -> Any:
    """Coerce a grid value string to int, then float, else leave as str."""
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    return value


def _parse_grid(params: list[str]) -> dict[str, list[Any]]:
    """Parse repeatable `KEY=v1,v2,...` flags into a grid dict. Values coerced int->float->str."""
    if not params:
        raise ValueError("provide at least one --param KEY=v1,v2,...")
    grid: dict[str, list[Any]] = {}
    for item in params:
        if "=" not in item:
            raise ValueError(f"malformed --param {item!r}: expected KEY=v1,v2,...")
        key, _, raw = item.partition("=")
        key = key.strip()
        values = [v.strip() for v in raw.split(",") if v.strip() != ""]
        if not key or not values:
            raise ValueError(f"malformed --param {item!r}: empty key or value list")
        grid[key] = [_coerce(v) for v in values]
    return grid
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_sweep_parse.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add algua/backtest/sweep.py tests/test_sweep_parse.py
git commit -m "feat: add sweep grid parser"
```

---

### Task 2: `_combos` (Cartesian product + size guard)

**Files:**
- Modify: `algua/backtest/sweep.py`
- Test: `tests/test_sweep_combos.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sweep_combos.py
import pytest
from algua.backtest.engine import BacktestError
from algua.backtest.sweep import _combos


def test_cartesian_product():
    combos = _combos({"a": [1, 2], "b": [3, 4, 5]})
    assert len(combos) == 6
    assert {"a": 1, "b": 3} in combos
    assert {"a": 2, "b": 5} in combos


def test_single_param():
    assert _combos({"a": [1, 2, 3]}) == [{"a": 1}, {"a": 2}, {"a": 3}]


def test_too_many_combos_raises():
    with pytest.raises(BacktestError):
        _combos({"a": list(range(15)), "b": list(range(15))})  # 225 > 200
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_sweep_combos.py -v`
Expected: FAIL with `ImportError: cannot import name '_combos'`.

- [ ] **Step 3: Add to `algua/backtest/sweep.py`**

Add the import and helper:

```python
import itertools

from algua.backtest.engine import BacktestError

_MAX_COMBOS = 200


def _combos(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Cartesian product of the grid into a list of param dicts. Guarded by _MAX_COMBOS."""
    keys = list(grid.keys())
    value_lists = [grid[k] for k in keys]
    total = 1
    for values in value_lists:
        total *= len(values)
    if total > _MAX_COMBOS:
        raise BacktestError(
            f"grid too large: {total} combos > {_MAX_COMBOS}; narrow the grid"
        )
    return [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*value_lists)]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_sweep_combos.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add algua/backtest/sweep.py tests/test_sweep_combos.py
git commit -m "feat: add sweep Cartesian product with size guard"
```

---

### Task 3: `_override` (param override without mutation)

**Files:**
- Modify: `algua/backtest/sweep.py`
- Test: `tests/test_sweep_override.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sweep_override.py
import pandas as pd
from algua.backtest.sweep import _override
from algua.contracts.types import ExecutionContract
from algua.strategies.base import LoadedStrategy, StrategyConfig


def _base():
    cfg = StrategyConfig(
        name="m", universe=["AAA"],
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1),
        params={"lookback": 60, "top_k": 3},
    )
    return LoadedStrategy(config=cfg, fn=lambda v, p: pd.Series(dtype="float64"))


def test_override_merges_over_defaults():
    base = _base()
    out = _override(base, {"lookback": 20})
    assert out.config.params == {"lookback": 20, "top_k": 3}
    assert out.fn is base.fn
    assert out.name == "m"


def test_override_does_not_mutate_base():
    base = _base()
    _override(base, {"lookback": 20, "top_k": 1})
    assert base.config.params == {"lookback": 60, "top_k": 3}  # unchanged
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_sweep_override.py -v`
Expected: FAIL with `ImportError: cannot import name '_override'`.

- [ ] **Step 3: Add to `algua/backtest/sweep.py`**

Add the import and helper:

```python
from algua.strategies.base import LoadedStrategy


def _override(strategy: LoadedStrategy, combo: dict[str, Any]) -> LoadedStrategy:
    """Return a LoadedStrategy whose params are the base params with `combo` merged over them.
    Does not mutate the base strategy/config."""
    new_params = {**strategy.config.params, **combo}
    new_config = strategy.config.model_copy(update={"params": new_params})
    return LoadedStrategy(config=new_config, fn=strategy.fn)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_sweep_override.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add algua/backtest/sweep.py tests/test_sweep_override.py
git commit -m "feat: add sweep param override"
```

---

### Task 4: `SweepResult` + `sweep` driver

**Files:**
- Modify: `algua/backtest/sweep.py`
- Test: `tests/test_sweep.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sweep.py
from datetime import UTC, datetime

import pandas as pd
import pytest
from algua.backtest._sample import SyntheticProvider
from algua.backtest.sweep import SweepResult, sweep
from algua.contracts.types import ExecutionContract
from algua.strategies.base import LoadedStrategy, StrategyConfig

START = datetime(2022, 1, 1, tzinfo=UTC)
END = datetime(2023, 12, 31, tzinfo=UTC)


def _momentum():
    # mirror the bundled cross-sectional momentum so params actually affect results
    from algua.features.indicators import momentum

    cfg = StrategyConfig(
        name="m", universe=["AAA", "BBB", "CCC"],
        execution=ExecutionContract(rebalance_frequency="1d", decision_lag_bars=1),
        params={"lookback": 40, "top_k": 1},
    )

    def fn(view, params):
        wide = view.reset_index().pivot(index="timestamp", columns="symbol", values="adj_close")
        if len(wide) <= int(params["lookback"]):
            return pd.Series(dtype="float64")
        scores = momentum(wide, lookback=int(params["lookback"])).iloc[-1].dropna()
        winners = scores.sort_values(ascending=False).head(int(params["top_k"])).index
        if len(winners) == 0:
            return pd.Series(dtype="float64")
        return pd.Series(1.0 / len(winners), index=winners)

    return LoadedStrategy(config=cfg, fn=fn)


def test_sweep_ranks_and_counts():
    res = sweep(_momentum(), SyntheticProvider(seed=3), START, END,
                grid={"lookback": [20, 40], "top_k": [1, 2]}, windows=4, holdout_frac=0.2)
    assert isinstance(res, SweepResult)
    d = res.to_dict()
    assert d["n_combos"] == 4
    assert len(d["ranked"]) == 4
    scores = [r["score"] for r in d["ranked"]]
    assert scores == sorted(scores, reverse=True)              # ranked descending
    assert d["ranked"][0]["score"] == d["best"]["score"]
    # each combo carries holdout + stability; score == stability[rank_by]
    top = d["ranked"][0]
    assert "holdout" in top and "stability" in top
    assert top["score"] == top["stability"]["mean_sharpe"]
    assert set(top["params"]) == {"lookback", "top_k"}


def test_sweep_is_deterministic():
    kw = dict(grid={"lookback": [20, 40], "top_k": [1, 2]}, windows=4, holdout_frac=0.2)
    a = sweep(_momentum(), SyntheticProvider(seed=3), START, END, **kw)
    b = sweep(_momentum(), SyntheticProvider(seed=3), START, END, **kw)
    assert a.to_dict() == b.to_dict()


def test_sweep_rejects_bad_rank_by():
    with pytest.raises(ValueError):
        sweep(_momentum(), SyntheticProvider(seed=3), START, END,
              grid={"lookback": [20, 40]}, rank_by="holdout_sharpe")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_sweep.py -v`
Expected: FAIL with `ImportError: cannot import name 'SweepResult'`.

- [ ] **Step 3: Add to `algua/backtest/sweep.py`**

Add imports + the result type + driver:

```python
from dataclasses import dataclass
from datetime import datetime

from algua.backtest.walkforward import walk_forward
from algua.contracts.types import DataProvider

_RANK_KEYS = {"mean_sharpe", "min_sharpe"}


@dataclass
class SweepResult:
    strategy: str
    data_source: str
    snapshot_id: str | None
    timeframe: str
    seed: int | None
    period: dict[str, str]
    grid: dict[str, list[Any]]
    n_combos: int
    rank_by: str
    ranked: list[dict[str, Any]]
    best: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "data_source": self.data_source,
            "snapshot_id": self.snapshot_id,
            "timeframe": self.timeframe,
            "seed": self.seed,
            "period": self.period,
            "grid": self.grid,
            "n_combos": self.n_combos,
            "rank_by": self.rank_by,
            "ranked": self.ranked,
            "best": self.best,
        }


def sweep(
    strategy: LoadedStrategy,
    provider: DataProvider,
    start: datetime,
    end: datetime,
    *,
    grid: dict[str, list[Any]],
    windows: int = 4,
    holdout_frac: float = 0.2,
    rank_by: str = "mean_sharpe",
) -> SweepResult:
    """Evaluate every grid combo with walk_forward and rank by an out-of-sample window metric.

    The holdout is carried per combo but never used for ranking (reserved for promotion gates).
    """
    if rank_by not in _RANK_KEYS:
        raise ValueError(f"rank_by must be one of {sorted(_RANK_KEYS)}, got {rank_by!r}")
    combos = _combos(grid)

    records: list[dict[str, Any]] = []
    meta = None
    for combo in combos:
        wf = walk_forward(
            _override(strategy, combo), provider, start, end,
            windows=windows, holdout_frac=holdout_frac,
        )
        if meta is None:
            meta = wf  # sweep-level stamps are identical across combos
        h = wf.holdout_metrics
        records.append({
            "params": combo,
            "config_hash": wf.config_hash,
            "n_windows": wf.windows,
            "stability": wf.stability,
            "holdout": {
                "n_bars": h["n_bars"], "sharpe": h["sharpe"],
                "total_return": h["total_return"], "max_drawdown": h["max_drawdown"],
            },
            "score": wf.stability[rank_by],
        })

    records.sort(key=lambda r: r["score"], reverse=True)
    best = {"params": records[0]["params"], "score": records[0]["score"]}
    assert meta is not None  # combos is always non-empty (grid has >=1 key with >=1 value)
    return SweepResult(
        strategy=strategy.name,
        data_source=meta.data_source,
        snapshot_id=meta.snapshot_id,
        timeframe=meta.timeframe,
        seed=meta.seed,
        period=meta.period,
        grid=grid,
        n_combos=len(combos),
        rank_by=rank_by,
        ranked=records,
        best=best,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_sweep.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Gate + commit**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`
Expected: all green; 4 contracts kept (`sweep.py` imports only `algua.backtest`/`algua.contracts`/`algua.strategies`).
```bash
git add algua/backtest/sweep.py tests/test_sweep.py
git commit -m "feat: add sweep driver and SweepResult"
```

---

### Task 5: CLI `backtest sweep`

**Files:**
- Modify: `algua/cli/backtest_cmd.py`
- Test: `tests/test_cli_sweep.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli_sweep.py
import json
import pytest
from typer.testing import CliRunner
from algua.cli.main import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def _tmp(monkeypatch, tmp_path):
    monkeypatch.setenv("ALGUA_DB_PATH", str(tmp_path / "r.db"))
    monkeypatch.setenv("ALGUA_DATA_DIR", str(tmp_path))


def test_sweep_demo_emits_ranked():
    result = runner.invoke(app, ["backtest", "sweep", "cross_sectional_momentum", "--demo",
                                 "--start", "2022-01-01", "--end", "2023-12-31",
                                 "--param", "lookback=20,40", "--param", "top_k=1,3",
                                 "--top", "2"])
    assert result.exit_code == 0, result.stdout
    d = json.loads(result.stdout)
    assert d["n_combos"] == 4          # full count reported
    assert len(d["ranked"]) == 2       # --top limits printed rows
    assert d["best"]["params"]
    assert d["rank_by"] == "mean_sharpe"


def test_sweep_requires_param():
    result = runner.invoke(app, ["backtest", "sweep", "cross_sectional_momentum", "--demo"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["ok"] is False


def test_sweep_malformed_param_is_json_error():
    result = runner.invoke(app, ["backtest", "sweep", "cross_sectional_momentum", "--demo",
                                 "--param", "lookback"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["ok"] is False


def test_sweep_requires_data_source():
    result = runner.invoke(app, ["backtest", "sweep", "cross_sectional_momentum",
                                 "--param", "lookback=20,40"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["ok"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli_sweep.py -v`
Expected: FAIL — no `sweep` command.

- [ ] **Step 3: Edit `algua/cli/backtest_cmd.py`**

Add the import (with the other `algua.backtest` imports):
```python
from algua.backtest.sweep import _parse_grid, sweep
```
Add the command:
```python
@backtest_app.command("sweep")
@json_errors(ValueError, LookupError, BacktestError)
def sweep_cmd(
    name: str,
    start: str = typer.Option("2023-01-01", "--start"),
    end: str = typer.Option("2023-12-31", "--end"),
    demo: bool = typer.Option(False, "--demo", help="use the synthetic data provider"),
    snapshot: str = typer.Option(None, "--snapshot", help="backtest an ingested bars snapshot id"),
    windows: int = typer.Option(4, "--windows", help="walk-forward windows per combo"),
    holdout_frac: float = typer.Option(0.2, "--holdout-frac", help="fraction reserved as holdout"),
    param: list[str] = typer.Option(None, "--param", help="KEY=v1,v2,... (repeatable)"),
    rank_by: str = typer.Option("mean_sharpe", "--rank-by", help="mean_sharpe | min_sharpe"),
    top: int = typer.Option(20, "--top", help="max ranked rows to print"),
) -> None:
    """Sweep a strategy across a parameter grid; walk-forward score each combo and rank."""
    strategy = load_strategy(name)
    provider = _select_provider(demo, snapshot)
    grid = _parse_grid(param or [])
    result = sweep(strategy, provider, _utc(start), _utc(end),
                   grid=grid, windows=windows, holdout_frac=holdout_frac, rank_by=rank_by)
    payload = result.to_dict()
    payload["ranked"] = payload["ranked"][:top]
    emit(payload)
```
`BacktestError` is already imported in `backtest_cmd.py`. Typer makes a repeated `--param` a `list[str]`; `param or []` handles the none-given case so `_parse_grid` raises the "provide at least one" `ValueError` (rendered as JSON).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli_sweep.py tests/test_cli_backtest.py tests/test_cli_walkforward.py -q`
Expected: PASS (new sweep tests + existing run/walk-forward tests unaffected).

- [ ] **Step 5: Commit**

```bash
git add algua/cli/backtest_cmd.py tests/test_cli_sweep.py
git commit -m "feat: add 'backtest sweep' CLI command"
```

---

### Task 6: Full verification & smoke

**Files:** none (verification only)

- [ ] **Step 1: Full quality gate**

Run:
```bash
uv run pytest -q
uv run ruff check .
uv run mypy algua
uv run lint-imports
```
Expected: all tests pass; ruff clean; mypy `Success`; import-linter `4 kept, 0 broken` (`sweep`/`walkforward` stay off `algua.data`; `sweep` imports only within `backtest` + `contracts`/`strategies`).

- [ ] **Step 2: CLI smoke (synthetic)**

Run:
```bash
uv run algua backtest sweep cross_sectional_momentum --demo --start 2021-01-01 --end 2023-12-31 \
    --param lookback=20,40,60 --param top_k=1,2,3 --rank-by mean_sharpe --top 5
```
Expected: JSON with `n_combos: 9`, a `ranked` list of 5 rows sorted by `score` (each with `params`, `stability`, `holdout`, `score`), a `best` pointer, and `rank_by: "mean_sharpe"`.

- [ ] **Step 3: Final commit (if any verification fixes were needed)**

```bash
git add -A
git commit -m "test: verify parameter sweeps end to end" --allow-empty
```

---

## Self-Review Notes

- **Spec coverage:** `_parse_grid` (Task 1), `_combos` + `_MAX_COMBOS` (Task 2), `_override`
  without mutation (Task 3), `SweepResult` + `sweep` with walk-forward scoring/ranking + holdout
  carried-not-ranked + search-breadth `n_combos` (Task 4), CLI `sweep` with repeatable `--param`,
  `--rank-by`, `--top`, error paths (Task 5), verification (Task 6). Out-of-scope (MLflow,
  promotion gates, full WF optimization, random/Bayesian/parallel) intentionally absent.
- **Boundary:** all new code in `algua/backtest/sweep.py` + the CLI; `sweep` imports only
  `algua.backtest.walkforward`/`engine`, `algua.strategies.base`, `algua.contracts.types` — never
  `cli`/`registry`/`data`. lint-imports stays at 4 kept.
- **Type consistency:** `_parse_grid(list[str]) -> dict[str,list]`, `_combos(grid) -> list[dict]`,
  `_override(strategy, combo) -> LoadedStrategy`, `sweep(..., grid=, windows=, holdout_frac=,
  rank_by=) -> SweepResult`, `_RANK_KEYS`, `_MAX_COMBOS`, and `SweepResult.to_dict()` keys are used
  identically across tasks/tests. `score == stability[rank_by]`; ranking never uses holdout.
