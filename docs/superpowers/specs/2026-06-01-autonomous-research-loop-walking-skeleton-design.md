# Autonomous Research Loop — Walking Skeleton (Sub-project 4, Slice 1)

**Date:** 2026-06-01
**Status:** Accepted (pending implementation)
**Scope:** The first slice of sub-project 4 (the agent operating layer). Make the autonomous
research loop **run unattended end-to-end**, authoring real strategy files and driving them
through `idea → backtested → shortlisted` — with **trivial, fixed ideation** (a hardcoded list of
parameter variants). De-risks the harness wiring and guardrails *before* the open-ended ideation
problem.

---

## 1. Context & non-goals

Sub-projects 1–3 built every research primitive (data, backtest, walk-forward, sweep, MLflow,
and the `research promote` gate). The harness decision is recorded in
`docs/superpowers/specs/2026-06-01-autonomous-operator-harness-design.md`: the operator runs on
the **Codex CLI** (`codex exec`), embedded as a subprocess, with OS-level sandbox write-scope and
a `--output-schema` result contract. This slice implements the **loop that drives it**.

algua already owns durable state (registry / MLflow / provenance) and the safety boundary (the
`paper → live` wall + gates, enforced *inside* the CLI). The loop is therefore thin: it decides
which CLI calls to make and bounds the run. It adds no new durable state and no new enforcement.

**Non-goals (deferred to later slices of sub-project 4):**
- Real ideation (novel/free-form strategy authoring, hypotheses). This slice uses a *fixed
  variant list* as a stand-in.
- Idea-ledger / learning across runs beyond registry-based dedup.
- Scheduling (cron/systemd). This slice ships a CLI command run on demand; wiring a timer is
  trivial and separate.
- Resuming a partially-completed variant (dedup treats it as "tried").
- Promotion past `shortlisted`.

---

## 2. What one iteration does

Author a **new** parameter-variant of the existing `cross_sectional_momentum` strategy and take
it through the lifecycle:

1. Author `algua/strategies/examples/<name>.py` — a new module exposing `CONFIG` (a variant of
   the momentum params, e.g. a different `lookback`) and `target_weights`, with a structured
   provenance header (`GENERATED_BY = "agent"` + a one-line comment). Authoritative provenance is
   the registry actor on the idea-creation transition; the header is human-friendly redundancy.
2. Drive the CLI: `registry add` → `backtest run --register` (→ `backtested`) →
   `research promote` (gate → `shortlisted` on pass, unchanged on fail).
3. Return a typed `IterationResult`.

Authoring into `examples/` (not a separate package) keeps the loader unchanged. The risk that
this lets the agent overwrite a curated strategy is handled by the **additions-only diff-gate**
(§5): only *new* files under `examples/` are permitted; modifying an existing file is a violation.

---

## 3. Components

Each is small, single-purpose, and independently testable. New package `algua/operator/`.

| Module | Responsibility | Depends on |
|---|---|---|
| `algua/operator/variants.py` | The fixed variant source: a deterministic list of momentum param-variants, each with a derived strategy name (e.g. `momentum_lb40`). Pure data + naming. | stdlib |
| `algua/operator/result.py` | The `IterationResult` model + the JSON schema file it serializes to (see §6). | pydantic |
| `algua/operator/gate.py` | Pure functions: the **additions-only diff-gate** (`changed_paths, allowed_prefix → ok\|violation`) and the **dedup** check (`variant_name, registered_names → tried?`). | stdlib |
| `algua/operator/runner.py` | The `AgentRunner` protocol (run one iteration for a variant → `IterationResult`) and the production `CodexRunner` (subprocess `codex exec …`). | subprocess |
| `algua/operator/loop.py` | The **controller**: dedup → pick next variant → run agent turn → diff-gate → record → repeat until a stop condition. Harness-agnostic; depends only on the `AgentRunner` protocol. | the modules above |
| `algua/cli/operator_cmd.py` | `algua operator run --max-iterations N [--demo]`; emits the run summary as JSON. | `loop` |

The test `FakeRunner` (a deterministic double that authors the file and drives the CLI in-process)
lives in `tests/`, not in the operator package, so the package stays free of `cli` imports.

### Boundary: the operator obeys the golden rule

`algua/operator/` talks to algua **only through the CLI** (subprocess `uv run algua …`) — the same
rule the agent and `CLAUDE.md` mandate ("never reach into modules to bypass the CLI"). It never
imports `cli` / `registry` / `backtest` / `research` / `data` internals. Dedup reads
`algua registry list --json`. A new import-linter contract enforces this.

---

## 4. Data & control flow per iteration

1. Controller shells `algua registry list --json` → registered names → subtract the variant list
   → **untried variants** (dedup; no new state).
2. Pick the next untried variant (deterministic order). Snapshot working-tree file state as the
   diff-gate baseline.
3. `AgentRunner.run(variant)`:
   - **`CodexRunner`** (production): `codex exec --json --output-schema
     docs/contracts/iteration-result.schema.json -o .algua/runs/iter-<N>.json --sandbox
     workspace-write --add-dir algua/strategies/examples --ask-for-approval never -C <repo>
     "<prompt referencing the operator skill>"`. Parses the schema-validated `IterationResult`
     from the `-o` file; captures the JSONL trace to the run log.
   - **`FakeRunner`** (tests): writes `examples/<name>.py` deterministically, then drives the
     algua CLI; builds the `IterationResult`.
4. Controller computes changed paths (git diff vs. the pre-turn snapshot) → **additions-only
   diff-gate**.
5. Append the `IterationResult` to the run log; loop.

**Stop conditions:** `--max-iterations` reached · no untried variants left · diff-gate violation
(fail-closed) · consecutive-failure circuit breaker tripped.

**Run summary (emitted JSON):** `{iterations: [IterationResult…], stopped_reason, n_shortlisted}`.

---

## 5. Guardrails & error handling

Defense in depth, three **distinct** failure classes with deliberately different responses:

- **Iteration failed** (codex non-zero exit; malformed/missing schema output; an `algua` command
  errored): record `status=failed` + error and **continue** to the next variant — one bad
  strategy must not kill the run. A **consecutive-failure circuit breaker** (stop after K in a
  row, K configurable, default 3) prevents a systemic fault from burning the whole budget.
- **Diff-gate violation** (the turn touched anything other than a *new* file under
  `algua/strategies/examples/` — including modifying a curated strategy): **fail-closed, stop the
  entire run.** This is a guardrail breach, not a routine failure.
- **Live wall:** unreachable by construction — the loop only drives `idea→backtested→shortlisted`;
  `promote` tops out at `shortlisted`, and algua blocks `→ live` regardless. The skill never
  instructs going past shortlist.

Layering: (1) the algua CLI is the authoritative wall (live gate + research gates); (2) the OS
sandbox (`--sandbox workspace-write --add-dir`) confines writes at the kernel; (3) the
additions-only diff-gate is the controller's belt-and-suspenders check; (4) the iteration/
circuit-breaker budget bounds spend.

`--ask-for-approval never` is safe **only** inside the sandbox; running the operator outside a
sandbox is disallowed.

---

## 6. The `IterationResult` contract

One model, serialized to `docs/contracts/iteration-result.schema.json`, used by both
`--output-schema` (production) and the `FakeRunner` (tests):

```json
{
  "iteration": 1,
  "strategy_name": "momentum_lb40",
  "stage_before": "idea",
  "stage_after": "shortlisted",
  "gate_passed": true,
  "promoted": true,
  "config_hash": "…",
  "status": "ok",
  "reason": "gate pass: holdout_sharpe=…",
  "error": null
}
```

- `status ∈ {ok, failed}`. `error` is null unless `status=failed`.
- `stage_before` / `stage_after` are lifecycle stages read back from the registry.
- The schema is the single source of truth; the pydantic model and the JSON schema must agree
  (a unit test validates a sample against the schema).

---

## 7. Testing

All green in CI with **no LLM**:

- **Unit:** variant naming/list determinism; the gate (new file ✓, modify-existing ✗,
  path-outside-`examples/` ✗); dedup (registered name skipped); `IterationResult` validates
  against `iteration-result.schema.json`; `CodexRunner` parses a *canned* `-o` schema file +
  JSONL with `subprocess` mocked (tests the parsing path without an LLM).
- **e2e with `FakeRunner`:** `operator run --max-iterations 2 --demo` → 2 new files authored, 2
  strategies registered **and** shortlisted, registry advanced, summary JSON correct.
- **Negative e2e:** `FakeRunner` configured to write outside `examples/` → gate violation → run
  stops fail-closed, nothing past the violation promoted.
- **`CodexRunner` vs. a real LLM:** documented manual smoke, **not** in CI.
- **Quality gates:** `pytest · ruff · mypy · lint-imports`, including the new contract that
  `algua.operator` imports none of `cli/registry/backtest/research/data`.

Determinism holds because backtest/walk-forward seeds are already fixed, so `FakeRunner` runs are
reproducible.

---

## 8. Consequences

- Proves the full operator machinery (codex-exec invocation, schema'd result, write-scope sandbox,
  diff-gate, budget, registry-driven dedup) against the existing strategy, so the next slice can
  focus purely on **ideation** without re-litigating plumbing.
- The operator dogfoods the golden rule (CLI-only), so the loop is itself a proof that the CLI
  surface is sufficient to operate the system.
- Generated files are uncommitted working-tree artifacts the human curates; nothing enters
  committed source without human action.
- Vendor coupling stays at the thin `CodexRunner` seam; swapping harnesses later is "add another
  `AgentRunner` impl," not a rewrite.
