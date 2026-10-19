# Acceptance dashboard

These scenarios expose a small number of deterministic, production-seam examples for human
inspection. They complement the broad unit and integration suite; they do not replace it.

## Generate the dashboard

Install the Python and report dependencies once:

```bash
uv sync
npm install
```

Then run:

```bash
scripts/acceptance-report.sh
```

The command runs only tests marked `acceptance`, replaces stale Allure results, and writes a fresh
static report to `.runs/acceptance/allure-report/index.html`. The generated `.runs/` content is
local evidence and is intentionally ignored by Git.

## Review protocol

For every scenario, inspect the named inputs, actual output, and expected-versus-actual attachment.
The dashboard contains:

- `DATA-01`: raw provider bars through `refresh_bars` and `DataStore` into a canonical snapshot.
- `STRAT-01`: obvious price paths through the bundled strategy's public decision methods.
- `BT-02`: the same strategy and an immutable snapshot through `simulate`, proving the `t` to
  `t+1` execution delay.
- `FWD-01`: one immutable snapshot through both `simulate` and the paper rehearsal, proving that
  the two lanes agree on target weights at the same information cutoff and that the paper intents
  fill no earlier than `t+1`.

Files under `fixtures/reference_market_v1/expected/` are approved examples, not snapshots to update
automatically. Change them only after a human has reviewed the attached input and actual output and
confirmed that the new behavior is intentional.
