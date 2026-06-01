# Agent Operating Layer (Skills-First) — Sub-project 4

**Date:** 2026-06-01
**Status:** Accepted (pending implementation)
**Supersedes:** `2026-06-01-autonomous-research-loop-walking-skeleton-design.md` and its plan — we
pivoted away from an external Python enforcement harness (`run_loop`/gate/variants/adapters)
toward **skills + native Codex subagents + a thin launcher**, trusting a capable agent operating
inside a contained environment rather than caging it in code.

---

## 1. Intent

The architecture spec defines sub-project 4 as the *"agent operating layer — playbooks/skills for
the autonomous research loop (ideate→author→backtest→interpret→shortlist)."* The promotion gate
already shipped in sub-project 3, so this slice is the **operating layer itself**: the skills that
expose the codebase to an agent, the subagents it delegates to, and a launcher that runs a capable
agent **autonomously** through the research lifecycle to a shortlist — then hands back a branch for
human review.

**Design stance:** don't enforce the agent's behavior with a code harness — algua's CLI already is
the boundary (the live wall + research gates are enforced inside the CLI). Instead, give the agent
rich operating knowledge (skills) and **contain its environment** (an isolated worktree), so a run
is fully autonomous yet every change lands on a branch you review before it can reach `main`.

**Non-goals:** real-money execution (sub-projects 5–6); the TOTP live gate (sub-project 6 — see
the `algua-live-gate` design note); scheduling/cron (the launcher is run on demand; wrapping it in
a timer is trivial and separate).

---

## 2. Components

```
.codex/
  skills/
    operating-algua/SKILL.md        # orientation: golden rules, the CLI/JSON seam, lifecycle, where to look
    author-a-strategy/SKILL.md      # the strategy contract (CONFIG + target_weights), bar schema, features, GENERATED_BY marker
    run-the-research-loop/SKILL.md  # the loop: ideate→author→backtest/walk-forward/sweep→promote(gate)→shortlist/discard→report
    interpret-results/SKILL.md      # reading the JSON, holdout/stability semantics, overfitting & look-ahead pitfalls
  agents/
    author.toml                     # subagent: writes a strategy module from a hypothesis
    interpret.toml                  # subagent: judges results JSON, recommends promote/discard
  scripts/
    run-research-loop.sh            # the launcher — the only executable code in this slice
CODEOWNERS                          # human-only approval on integrity-critical files
AGENTS.md                           # += a short pointer routing the research-operator run to the skills
.claude/skills/<name>  -> ../../.codex/skills/<name>   # symlinks (Claude Code + opencode portability)
.agents/skills/<name>  -> ../../.codex/skills/<name>   # symlinks (.agents convention)
```

**Skills** are the operating knowledge (markdown). The primary agent reads `operating-algua` +
`run-the-research-loop`; it delegates to the `author` and `interpret` subagents, which read
`author-a-strategy` and `interpret-results` respectively. Skills are authored canonically in
`.codex/skills/` and symlinked into `.claude/skills/` and `.agents/skills/` so the same context
serves Codex (operator), Claude Code (co-dev), and opencode.

**Subagents** are the two roles: `author` (writes a strategy module; intended write scope =
`algua/strategies/examples/`) and `interpret` (read-only; judges results). The primary agent owns
ideation, orchestration, and the run report.

**Launcher** (`run-research-loop.sh`) is the only code. It creates an isolated git worktree on a
`research-run/<stamp>` branch and runs Codex non-interactively in yolo mode against that worktree
with a bounded goal, then prints the branch for review.

> **Codex skill discovery:** the always-read entry file is `AGENTS.md`, so the operating layer is
> routed through it (a short "operating the research loop" pointer) *and* the launcher's goal names
> the skills explicitly — the run works regardless of Codex's auto-discovery path. At build time,
> confirm whether Codex auto-loads `.codex/skills/` (or needs a `skills` entry in `.codex/config.toml`)
> and wire that too; the AGENTS.md pointer + explicit prompt are the robust fallback.

---

## 3. Run flow

```
operator (human or cron) runs:  .codex/scripts/run-research-loop.sh --hypotheses N --timeout 30m
  └─ git worktree add -b research-run/<stamp> ../algua-research-<stamp>
  └─ timeout <dur> codex exec --dangerously-bypass-approvals-and-sandbox -C <worktree> "<goal>"
        primary agent (per skills):
          repeat up to N hypotheses:
            ideate a hypothesis  →  delegate to `author` subagent (writes algua/strategies/examples/<name>.py)
            drive via CLI:  uv run algua backtest run <name> --demo --register
                            uv run algua backtest walk-forward <name> ...
                            (optionally) uv run algua backtest sweep <name> ...
                            uv run algua research promote <name> ...     # gate: backtested→shortlisted on pass
            delegate to `interpret` subagent  →  promote / discard decision + reasoning
          write run-report.md ; commit all authored files on the branch
  └─ launcher prints:  git diff main...research-run/<stamp>   (worktree path)
human reviews the branch (diff + run-report.md + registry/MLflow state) and merges what's good
```

The agent operates **only through `uv run algua …`** and never promotes past `shortlisted`
(`research promote` tops out there; the lifecycle blocks `→live` regardless). Authored strategy
modules go into `algua/strategies/examples/<name>.py` with a `GENERATED_BY = "agent"` header;
"only add new files, never edit existing strategies" is a **skill instruction** reinforced by human
branch review (no enforcement code).

---

## 4. Guardrails (defense in depth, no enforcement code)

1. **algua CLI = the boundary.** The live wall + research gates are enforced inside the CLI; the
   agent cannot promote to live or skip a gate no matter how it behaves.
2. **Worktree containment.** The run happens in a throwaway worktree on its own branch. A bad run
   (including any edit to safety code) is isolated to that branch and recoverable via git; nothing
   touches your working tree or `main`.
3. **CODEOWNERS + branch protection** on the integrity-critical files — `algua/registry/store.py`
   (live wall), `algua/contracts/lifecycle.py` (stages/transitions), `algua/backtest/engine.py`
   (anti-look-ahead `t→t+1`), `algua/research/gates.py` (gate criteria). The agent may edit these in
   its worktree, but a human must approve any merge to `main`. (Requires enabling "require review
   from Code Owners" in the GitHub branch-protection settings — a one-time repo setting.)
4. **Bounded run.** The goal caps hypotheses at `N` (intent), and the launcher wraps Codex in an
   OS-level `timeout <dur>` (hard kill) — the timeout holds even if the agent ignores the goal.
5. **No live path exists yet** (sub-projects 5–6 unbuilt), so a research run has no way to act on a
   `live` stage even in principle.

---

## 5. Verification

This slice is skills (markdown) + config + one bash script — little unit-testable code. So:

- **Launcher `--dry-run`** prints the worktree/branch it would create and the exact `codex exec`
  command (with `--dangerously-bypass-approvals-and-sandbox`, `-C <worktree>`, the `timeout`
  wrapper) **without** creating the worktree or invoking Codex. A pytest test asserts the dry-run
  output contains the expected flags, branch prefix, and bounded goal — so the wiring is tested
  without an LLM.
- **Skill frontmatter check** — a test asserts every `.codex/skills/*/SKILL.md` has valid YAML
  frontmatter with `name` + `description`.
- **Symlink check** — a test asserts each skill is reachable via `.claude/skills/` and
  `.agents/skills/`.
- **Existing quality gates** (`pytest · ruff · mypy · lint-imports`) stay green (this slice adds
  almost no Python, so they're largely unaffected).
- **Real-run smoke (manual, documented, not CI):** run the launcher for real (`--hypotheses 1`,
  short `--timeout`) and confirm it produces a branch with an authored strategy, a registry
  advanced to `shortlisted`, and a `run-report.md`. This is where Codex skill/subagent wiring is
  validated end-to-end.

---

## 6. Consequences

- The operating layer is **portable** (skills shared across Codex / Claude Code / opencode) and
  **swappable** (a different harness reuses the same skills + a different launcher).
- Far less code to maintain than the abandoned enforcement harness; the agent's autonomy is the
  point, not something to wrap in rails.
- Safety rests on layers that don't depend on the agent's goodwill: CLI enforcement, worktree
  containment, CODEOWNERS on integrity files, and (later) the TOTP live gate.
- The honest cost: a run's quality depends on the **skills' quality**. The skills are the real
  deliverable and will be iterated as runs reveal gaps.
