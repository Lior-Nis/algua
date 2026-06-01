# Research lifecycle — the canonical command sequence

This is the end-to-end path an agent (or the human operator) drives to take a strategy from
**idea** to **shortlisted**, and the wall that stops it from going further unaided. Every step
is a `uv run algua ...` command that emits JSON on stdout. The flow below is the one exercised by
`tests/test_e2e_lifecycle.py`; if you change a command's surface, update both.

Stages: `idea -> backtested -> shortlisted -> paper -> live -> retired`
(allowed edges live in `algua/contracts/lifecycle.py`). An agent may operate the lifecycle
**up to and including paper**. The `paper -> live` transition is a hard wall — it requires a
human actor and a verified approval, enforced in `algua/registry/store.py`.

The examples use `--demo` (the synthetic provider) so the whole loop runs offline. Swap in real
bars with `--snapshot <id>` from `data ingest-bars` once you want real data.

## 1. Backtest and register (`idea -> backtested`)

A single backtest that also registers the strategy and advances it to `backtested`:

```bash
uv run algua backtest run cross_sectional_momentum --demo \
  --start 2022-01-01 --end 2023-12-31 --register
```

`--register` is what advances the stage; without it the run is exploratory and the registry is
untouched. Add `--track` to log the run to MLflow.

Confirm the stage:

```bash
uv run algua registry show cross_sectional_momentum   # -> "stage": "backtested"
```

## 2. Out-of-sample evidence (still `backtested`)

These do **not** change the stage — they generate the evidence the gate will judge.

Walk-forward (holdout + K equal out-of-sample windows + stability):

```bash
uv run algua backtest walk-forward cross_sectional_momentum --demo \
  --start 2022-01-01 --end 2023-12-31 --windows 4 --holdout-frac 0.2
```

Parameter sweep (optional; a bounded grid, each combo walk-forwarded and ranked). The holdout is
the search-breadth defense — the more you search here, the more the gate's holdout check matters:

```bash
uv run algua backtest sweep cross_sectional_momentum --demo \
  --param lookback=20,40,60 --param top_k=3,5 --rank-by mean_sharpe
```

## 3. Promotion gate (`backtested -> shortlisted`)

`research promote` runs the walk-forward, evaluates the gate, and transitions the strategy to
`shortlisted` **only on pass**. On failure it reports why and leaves the stage untouched.

```bash
uv run algua research promote cross_sectional_momentum --demo \
  --start 2022-01-01 --end 2023-12-31 \
  --min-holdout-sharpe 0.5 --min-holdout-return 0.0 \
  --min-pct-positive 0.6 --min-window-sharpe 0.0 \
  --n-combos 6
```

- `--n-combos` records how many combinations you searched in step 2 (evidence, must be `>= 1`).
- `--min-pct-positive` must be in `[0, 1]`.
- The default thresholds are the gate criteria in `algua/research/gates.py`.

Output carries `passed`, `promoted`, the per-check breakdown, the config hash, and the snapshot
id. After a pass, `registry show` reports `"stage": "shortlisted"`.

## 4. Toward paper (`shortlisted -> paper`)

An agent may take a shortlisted strategy to paper:

```bash
uv run algua registry transition cross_sectional_momentum \
  --to paper --actor agent --reason "passed gate; paper-trading next"
```

## 5. The live wall (`paper -> live`) — human only

An agent **cannot** put a strategy live. This fails as JSON with a non-zero exit, and the stage
stays at `paper`:

```bash
uv run algua registry transition cross_sectional_momentum --to live --actor agent
# -> {"ok": false, "error": ...}, exit 1
```

Going live requires a human actor plus a matching approval recorded via `registry approve`
(`--code-hash`, `--config-hash`, `--by`). This is the one boundary the system enforces and an
agent must never try to route around.
