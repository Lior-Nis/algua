# Readiness-review lane rubrics

Each finder reads ONLY its own `## <slug>` section below, plus the shared context brief.
Grounding contract (all lanes): cite `file:line`; read your KB anchor; run WebSearch for current
best practices and cite ≥1 URL; a finding without concrete evidence is not a finding. Every finding
must state its `north_star_link`: `ml_readiness`, `safe_scale`, `both`, or `none`.

## swe — Software engineering / architecture
KB anchor: `/home/liornisimov/KB/software-engineering/04-software-design`,
`.../07-backend-engineering`, `.../11-error-handling-resilience`,
`.../14-concurrency-async-distributed`, `.../06-database-design`
Audit for: module boundaries & coupling, error-handling/resilience, concurrency & transactional
correctness (BEGIN IMMEDIATE, TOCTOU), DB schema/migration discipline, dependency-boundary
violations (`lint-imports`), interface clarity. Also test-suite adequacy (folded in).
North-star focus: does the architecture admit an ML/DL strategy without breaking boundaries; do
concurrency/transaction patterns hold as strategy count scales?

## mle — ML engineering (infra / serving)
KB anchor: `/home/liornisimov/KB/machine-learning-engineering/`
Audit for: is there a training→artifact→serving path at all; model versioning/registry; reproducible
training; feature pipeline parity (train vs serve); inference in the decision/order path (latency,
determinism, failure modes); data provenance & reproducibility (folded in).
North-star focus: what MLE infra is MISSING for ML/DL strategies to be first-class; scaling of model
artifacts/inference.

## ds — Data science (stats / experimentation)
KB anchor: `/home/liornisimov/KB/data-science/`
Audit for: statistical validity of the gates (DSR, FDR/LORD++, deflation), experiment tracking,
metric definitions, sample-size/power floors, reproducibility of experiments, data leakage in
feature construction (folded in).
North-star focus: are the statistical gates sound enough to admit ML models (which overfit harder);
does experiment infra scale to many models?

## qf — Quant finance
KB anchor: `/home/liornisimov/KB/quant-trading/13-alpha-research`, `.../15-backtesting`,
`.../16-metrics-and-risk`, `.../09-leakage-and-bias`, `.../10-splitting-and-validation`,
`.../17-costs-slippage-execution`, `/home/liornisimov/KB/finance/`
Audit for: backtest realism (costs/slippage/fills/market impact), backtest↔live parity (folded in),
survivorship/look-ahead/PIT correctness, walk-forward/holdout discipline, position sizing & capital
allocation, corporate-action handling.
North-star focus: do cost/parity models hold for ML strategies (often higher turnover); does capital
allocation stay safe as strategies scale?

## clean-code — Clean code / maintainability
KB anchor: `/home/liornisimov/KB/software-engineering/17-code-quality-maintainability`,
`.../04-software-design`
Audit for: readability, naming, function/file size & responsibility, dead code / compat cruft /
dual paths, duplication, comment quality, cyclomatic complexity, test readability.
North-star focus: is the code maintainable enough that ML-strategy additions won't rot it; does
complexity scale sub-linearly with features?

## agentic — Agentic operation readiness
KB anchor: `/home/liornisimov/KB/agentic/`,
`/home/liornisimov/KB/software-engineering/24-agent-procedures`, `.../22-ai-coding-agent-risks`
Audit for: CLI/JSON seam completeness & stability, fail-closed defaults, agent-forgeable inputs
(`--actor` class), idempotency & crash-safety of agent operations, autonomy boundaries (the
never-go-live invariant), auditability of agent decisions (folded in), determinism of agent-facing
outputs.
North-star focus: can an autonomous operator drive ML strategies safely; do agent guardrails hold as
operation scales.

## ml-dl-integration — ML/DL integration readiness
KB anchor: `/home/liornisimov/KB/machine-learning-engineering/`,
`/home/liornisimov/KB/quant-trading/12-models`
Audit for: the concrete seams an ML/DL strategy would need — where a model would plug into the
`signal`/`construction` contract; feature availability & PIT-correctness for model inputs; how a
trained artifact would be versioned, gated, and lifecycle-managed; batch vs online inference;
GPU/CPU/latency budget in the tick loop; fallback when a model errors.
North-star focus: THIS lane's entire job is ml_readiness — enumerate the missing integration seams
concretely, each with file:line where the seam would attach.

## risk-safe-scaling — Risk & safe-scaling
KB anchor: `/home/liornisimov/KB/quant-trading/16-metrics-and-risk`,
`/home/liornisimov/KB/software-engineering/13-performance`, `.../14-concurrency-async-distributed`
Audit for: risk limits (position/sector/net/gross caps, drawdown, turnover), breach→flatten→halt
paths, allocation/reservation-pool correctness under concurrency, performance/throughput ceilings,
resource exhaustion, blast-radius containment (one strategy can't sink the book).
North-star focus: THIS lane's entire job is safe_scale — what breaks as strategies/capital/model
complexity scale; what risk wall is missing or fail-open.

## model-risk-management — Model risk management (drift / monitoring)
KB anchor: `/home/liornisimov/KB/quant-trading/12-models`, `.../10-splitting-and-validation`,
`/home/liornisimov/KB/machine-learning-engineering/` (monitoring/drift notes)
Audit for: model validation discipline (SR 11-7 style), concept/feature drift detection, live
model-performance monitoring, champion–challenger, automatic model rollback/kill on decay, model
governance & documentation, silent-decay detection.
North-star focus: an ML/DL strategy that silently decays in production is the core safe-scale +
ml_readiness failure — enumerate what monitoring/rollback machinery is missing.

## security — Security & trust boundaries
KB anchor: `/home/liornisimov/KB/software-engineering/12-security`, `.../22-ai-coding-agent-risks`
Audit for: broker API-key/secret handling, the go-live signature / trust-anchor integrity (re-verify
at trade time vs DB-as-record), injection (SQL/command/prompt), the agent↔human trust boundary &
forgeable inputs, supply-chain (deps, lockfile), authz fail-closed, audit-trail integrity.
North-star focus: real capital + autonomous operation raises the security bar as the system scales;
find the trust boundaries that don't hold.

## observability — Observability & operability
KB anchor: `/home/liornisimov/KB/software-engineering/15-observability`,
`.../11-error-handling-resilience`
Audit for: kill-switches / halt-all / flatten reachability, alerting on breach/error/decay,
structured logging & metrics, audit trail, incident-response runbooks, health checks, the
"can you SEE it and STOP it" axis, dead-letter/quarantine visibility.
North-star focus: as strategies/models scale, can an operator observe and halt unsafe behavior fast
enough.
