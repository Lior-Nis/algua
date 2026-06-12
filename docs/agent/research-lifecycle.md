# Research lifecycle — the canonical command sequence

This is the end-to-end path an agent (or the human operator) drives to take a strategy from
**idea** to **candidate**, and the wall that stops it from going further unaided. Every step
is a `uv run algua ...` command that emits JSON on stdout. The flow below is the one exercised by
`tests/test_e2e_lifecycle.py`; if you change a command's surface, update both.

Stages: `idea -> backtested -> candidate -> paper -> forward_tested -> live -> retired`
(allowed edges live in `algua/contracts/lifecycle.py`). An agent may operate the lifecycle
**up to and including `forward_tested`**. The `forward_tested -> live` transition is a hard wall —
it requires a human actor, a verified approval, AND a fresh forward-test certificate, enforced in
`algua/registry/store.py` and `algua/registry/transitions.py`.

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

## 3. Promotion gate (`backtested -> candidate`)

`research promote` runs the walk-forward, evaluates the gate, and transitions the strategy to
`candidate` **only on pass**. On failure it reports why and leaves the stage untouched.

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
id. After a pass, `registry show` reports `"stage": "candidate"`.

## 4. Toward paper (`candidate -> paper`)

An agent may take a candidate strategy to paper:

```bash
uv run algua registry transition cross_sectional_momentum \
  --to paper --actor agent --reason "passed gate; paper-trading next"
```

## 5. Forward-test gate (`paper -> forward_tested`)

An agent advances from `paper` to `forward_tested` only via `paper promote`:

```bash
uv run algua paper promote cross_sectional_momentum
# -> {"ok": true, "stage": "forward_tested", "passed": true, ...} on pass
# -> {"ok": false, "passed": false, ...} on fail — stage stays at paper
```

The gate requires ≥63 broker-clocked daily return observations, ≥90% session coverage, realized
Sharpe ≥ max(0.5 × holdout_sharpe, 0.3), vol/drawdown bounds, clean integrity, and account
hygiene. Evidence must be ≤5 trading sessions stale. Relaxation flags are human-only.

Re-running `paper promote` on a strategy already at `forward_tested` refreshes the live-wall
certificate without a stage change (exit 0 on pass).

## 6. The live wall (`forward_tested -> live`) — human only

An agent **cannot** put a strategy live. This fails as JSON with a non-zero exit, and the stage
stays at `forward_tested`:

```bash
uv run algua registry transition cross_sectional_momentum --to live --actor agent
# -> {"ok": false, "error": ...}, exit 1
```

Going live is a two-step signed ceremony requiring a human actor AND a valid forward-test
certificate:

**Step 1** — get the go-live challenge (includes certificate summary):

```bash
uv run algua registry transition cross_sectional_momentum --to live --actor human
# -> {"ok": true, "action": "go_live_challenge", "challenge": "...", "certificate": {...}, ...}
```

**Step 2** — sign the challenge and re-run:

```bash
echo '<challenge value>' > challenge.txt
ssh-keygen -Y sign -n algua-go-live -f ~/.ssh/id_ed25519 challenge.txt
uv run algua registry transition cross_sectional_momentum --to live \
  --actor human --signature challenge.txt.sig
# -> {"ok": true, "stage": "live"}
```

The challenge is single-use and expires in 10 minutes. The forward certificate must be a PASS,
≤10 sessions old, with a clean record and matching identity. This is the boundary the system
enforces and an agent must never try to route around.
