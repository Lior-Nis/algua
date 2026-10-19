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

To verify the same deterministic scenarios inside the production image, without network access:

```bash
docker compose run --rm acceptance
```

The container gate runs the assertions; generate the Allure dashboard on the host for review.

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
- `RISK-01`: a legal proposed allocation through the public planner, showing the exact buy, sell,
  and dropped-holding intents it approves.
- `RISK-02`: concentration and gross-exposure violations through the same planner, showing their
  named rejection reasons and proving that neither produces an intent.
- `RISK-03`: a planner breach through the operational breach handler, showing the persistent kill
  switch and corresponding audit record.
- `BROKER-01`: a deterministic ordinary Alpaca rejection through the public paper-broker adapter,
  showing the exact submitted body and proving that no broker order ID is returned.
- `BROKER-02`: duplicate client-order IDs through the same adapter, showing recovery of an active
  venue order and the explicit skip of a duplicate whose venue order is already dead.
- `BROKER-03`: an ambiguous submit timeout followed by the venue's duplicate-ID response, proving
  that the adapter recovers the already-landed order instead of creating a second identity.
- `RECON-01`: partial and final paper fills through venue ingestion and account reconciliation,
  including a database restart and replay proving that fills are not double-counted.
- `RECON-02`: rejected and cancelled venue events through the same ingestion seam, proving that
  terminal non-fill events never create a portfolio position.
- `RECON-03`: a restart after venue acceptance but before broker-ID backfill, proving recovery by
  deterministic client order ID and correct attribution of the subsequent fill.
- `TICK-01`: canonical bars through the bundled strategy, planner/risk checks, paper submission,
  venue-fill ingestion and reconciliation, ending in an explicit persisted portfolio.
- `PROV-01`: the same strategy, immutable snapshot and explicit seed evaluated twice and written
  through the run ledger, proving equal results and persisted code/config/dependency identities.
- `OPS-01`: stale market data and a held symbol with no mark through the wall-clock planner,
  proving both conditions block every venue submission.
- `OPS-02`: a kill switch persisted across a process restart, proving the next tick halts before
  any venue submission.
- `OPS-03`: unexplained broker-versus-ledger drift across a restart and its grace window, showing
  trading deferred immediately and the mismatch eventually escalating to a halt.

Files under `fixtures/reference_market_v1/expected/` are approved examples, not snapshots to update
automatically. Change them only after a human has reviewed the attached input and actual output and
confirmed that the new behavior is intentional.
