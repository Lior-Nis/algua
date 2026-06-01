# Autonomous Operator Harness — Decision

**Date:** 2026-06-01
**Status:** Accepted (pending implementation in sub-project 4)
**Decision:** Run algua's autonomous research loop on the **Codex CLI** (`codex exec`),
embedded as a subprocess from a thin Python scheduler, with **OS-level sandboxing** for
write-scope and a **schema-enforced result contract** per iteration. **opencode** is the
documented escalation path if vendor-agnosticism becomes a hard requirement.

---

## 1. Context

algua is an agent-first platform. Sub-project 4 is the **autonomous research loop**: an
unattended, scheduled process that ideates a strategy, authors it, runs it through
backtest → walk-forward → sweep → the promotion gate, and shortlists or discards it — with no
human in the loop per iteration. The human re-enters only at the `paper → live` wall.

Before building that loop we must pick the **harness** it runs in. This document records that
decision and the reasoning, so the choice is auditable and the escalation paths are explicit.

### Two roles, decoupled

"Operate the repo" hides two different jobs with different best answers:

- **Co-developer harness** — the human-steered tool used to *build and maintain* algua (write
  code, run gates, open PRs). Today that is Claude Code. **Out of scope here** — low stakes to
  change, revisit independently.
- **Autonomous operator harness** — the unattended runtime that *runs* the research loop. This
  is the forced decision sub-project 4 demands, and the subject of this spec.

### The fact that shapes everything

algua's capability surface is **already the CLI**: every operation is `uv run algua … → JSON`.
And algua **already owns durable state and the safety boundary**:

- Durable state: the SQLite registry (lifecycle stages + transition history), MLflow runs,
  parquet snapshots, provenance manifests.
- Safety boundary: the `paper → live` wall (human actor + verified approval) and the research
  gates (walk-forward holdout/stability) are enforced **inside the CLI/registry**. An agent
  literally cannot promote to live, regardless of harness.

Therefore the harness is **thin and swappable**: it decides *which* CLI calls to make, holds
context, and manages the loop. It does **not** need to provide capabilities, durable state, or
the safety boundary — algua does. This collapses the decision onto a small set of harness
properties and makes the vendor choice reversible.

---

## 2. Decision drivers

Weighted for an **unattended, scheduled, CLI-driving** operator:

**High weight (these decide it):**
1. **Typed result contract per iteration** — the scheduler must reliably record *what each
   iteration decided* (which strategy, gate pass/fail, promoted or not, why). A
   schema-validated result object beats scraping a transcript.
2. **Write-scope guardrail** — a misbehaving agent must not edit anything outside the strategy
   authoring area or run anything but the algua CLI. This is a *second wall* behind algua's own
   enforcement, useful precisely because the operator is unattended.
3. **Embeddability in a Python scheduler** — lowest-plumbing way to wrap "bounded loop → CLI →
   read result → decide" with an iteration/cost budget guard.
4. **Structured failure isolation** — a partial iteration (authored but not backtested, etc.)
   must leave a clear, machine-readable trail; the scheduler logs "iteration N failed at step
   walk-forward, no promotion occurred."

**Low weight / discounted (algua or the scheduler already own these):**
- Durable node-level checkpointing, session resume — algua's registry/provenance is the durable
  record; iterations start fresh and read state from algua.
- Infinite-loop guards — the scheduler owns the bounded iteration/cost budget.
- "Richest permission API" — only matters if the agent's tool surface is broad; here it is
  effectively one tool (`bash → algua`), so OS sandboxing covers the threat more cheaply.

**Table stakes (all coding-harness candidates have these):** Agent Skills (`SKILL.md`),
`CLAUDE.md`/`AGENTS.md`, subagents, MCP, multi-tool orchestration.

---

## 3. Options considered

| Harness | Verdict | Why |
|---|---|---|
| **Codex CLI (`exec`)** | **Chosen** | `--output-schema` typed result; OS-level sandbox write-scope; subprocess embedding; well-documented JSONL trace. |
| **opencode** | Escalation path | Strong: fine-grained in-harness permission rules, provider-agnostic, reads `.claude/skills/`. Lacks a schema-enforced result; no Python SDK. Best if no-lock-in is a hard requirement. |
| **pi** | Escalation path | Minimal, transparent, fully-owned, provider-agnostic (RPC/SDK/print modes). Its value is the extension surface — unneeded when the agent's job is just to shell one CLI. Pick if "fully owned + no vendor" outranks everything. |
| **Claude Agent SDK** | Rejected (for now) | Richest permission/hook control surface — but that solves a broad-tool-surface problem we don't have; OS sandboxing covers our write-scope need more cheaply. Python-import coupling vs. zero-coupling subprocess. Becomes the choice only if the operator's tool surface broadens well beyond the CLI. |
| **Claude Managed Agents** | Rejected | Hosted/black-box loop, beta (`managed-agents-2026-04-01`), not ZDR/HIPAA-eligible, Anthropic lock-in. Offloads infra we don't need offloaded, at the cost of transparency. |
| **LangGraph** | Rejected (keep in pocket) | Its headline value (durable checkpointing, formal HITL) is **redundant** — algua owns durable state and the live wall. Would force hand-building the agent brain and lose portable skills. Revisit only if the loop grows complex multi-stage branching, where it would just *call the same CLI*. |

### The two real differentiators (Codex vs opencode)

Everything else is parity or already-owned-by-algua. The genuine trade:

- **Codex's edge:** a **typed, schema-enforced result per iteration** (`--output-schema`). No
  opencode equivalent — `--format json` emits a raw event stream but enforces no schema on the
  final answer. Combined with OS-level (kernel-enforced) sandboxing and subprocess embedding,
  Codex is the most robust *machine contract* of the two.
- **opencode's edge:** **fine-grained in-harness permission rules** (e.g. `bash:{"*":"deny",
  "uv run algua *":"allow"}`, `edit:{"*":"deny","strategies/**":"allow"}`) plus
  **provider-agnosticism** and native `.claude/skills/` reading.

We weight the schema-enforced result and OS sandbox above expressive permission rules, because
algua already owns the authoritative safety boundary and the operator's tool surface is narrow.

> Note on the Codex recommendation: a Codex consult independently picked Codex CLI. That carries
> obvious self-interest, so the decision here rests on a **documented, vendor-neutral
> differentiator** (`--output-schema`, which opencode lacks) rather than on that endorsement.

---

## 4. Integration architecture

A thin Python **scheduler** (cron / systemd timer / task runner) owns the loop budget and
invokes Codex once per iteration as a subprocess:

```bash
codex exec \
  --json \
  --output-schema docs/contracts/iteration-result.schema.json \
  -o .algua/runs/iteration-<N>.json \
  --sandbox workspace-write \
  --add-dir algua/strategies/examples \
  --ask-for-approval never \
  -C /path/to/algua \
  "<operator prompt: ideate→author→backtest/walk-forward/sweep→promote, per the skills>"
```

- **Result contract:** `--output-schema` validates the final answer against a JSON schema
  (`{strategy, stage_before, stage_after, gate_passed, promoted, reason, run_ids}` — exact shape
  defined in sub-project 4). `-o` writes it to a per-iteration file the scheduler reads.
- **Trace:** `--json` JSONL (`thread/turn/item.*` + per-turn token `usage`) is captured to the
  run log for observability and failure isolation.
- **Write-scope:** `--sandbox workspace-write` confines edits to the workspace; `--add-dir`
  scopes strategy authoring. The agent reaches the rest of the system only by *running the
  algua CLI*, never by editing engine/registry/contract files.
- **Approvals off** is safe **only** because it runs inside the sandbox.

### Guardrails — defense in depth

1. **algua CLI (authoritative):** live wall + research gates. The agent cannot go live or skip a
   gate no matter what the harness does.
2. **OS sandbox:** kernel-enforced write-scope (`workspace-write` + `--add-dir`).
3. **Scheduler budget:** hard cap on iterations / wall-clock / token spend per run.
4. **Optional git-diff scope gate (we own, harness-agnostic):** reject any iteration whose diff
   touches paths outside `algua/strategies/examples/`. Stronger than a prompt convention and
   portable across harnesses.

### Shared context layer — skills

The operating context lives as **Agent Skills** (`SKILL.md`), not baked into prompts:

- Content: the operating guide, the research-lifecycle walkthrough, gate criteria, bar schema.
- Location: `.claude/skills/` / `.agents/skills/` — read by Codex **and** opencode **and**
  Claude Code, so the *same* context serves the operator and the co-developer.
- This is what makes the harness swappable: changing harness = change the invocation, not the
  context.

---

## 5. Consequences

**Positive**
- Lowest-plumbing automation contract: subprocess + schema'd result + JSONL trace.
- Write-scope handled by an OS sandbox — no permission-callback layer to build or maintain.
- Vendor coupling is **reversible**: the loop rides on the algua CLI + portable skills, so
  swapping to opencode/pi is "change the invocation," not a rewrite.

**Negative / accepted**
- Coupled to the Codex CLI (and thus the OpenAI ecosystem) at the harness layer. Accepted
  because it is reversible (above) and the deciding feature (`--output-schema`) is concrete.
- `--ask-for-approval never` is only safe inside the sandbox; running the operator outside a
  sandbox is explicitly disallowed.

**Escalation paths (when to revisit)**
- **→ opencode:** vendor-agnosticism becomes a hard requirement, or we want the algua command
  whitelist expressed as declarative harness permission rules.
- **→ pi:** we want a minimal harness we fully own and read end-to-end.
- **→ LangGraph:** the loop grows genuine multi-stage branching (parallel strategy populations,
  multi-day human approval chains) needing formal orchestration — it would call the same CLI.
- **→ Claude Agent SDK:** the operator's tool surface broadens well beyond running the CLI.

---

## 6. Out of scope

- **The loop logic itself** (ideation strategy, authoring approach, stopping criteria, how
  results feed the next idea) — that is **sub-project 4's** design, brainstormed separately.
- **The co-developer harness** — stays Claude Code; revisit independently.
- The exact `iteration-result` schema and the operator skill contents — defined when
  sub-project 4 is designed.
