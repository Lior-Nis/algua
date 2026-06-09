# Web-research tooling for the autonomous operating agent (Issue 134)

**Status:** design — built (integrated with merged #126)
**Issue:** #134
**Author/operator:** Lior-Nis
**Design review:** GATE 1 + GATE 2 (Codex + Gemini + GLM panels) on the original two-phase design,
folded in. A capability spike (§3) and the **merge of #126 mid-flight** then reshaped scope (§2, §4).
**Relates to:** #126 (idea pool — now MERGED; this feeds it), #132 (news/fundamentals)

---

## 1. Context & premise

The autonomous research loop runs as `codex exec --dangerously-bypass-approvals-and-sandbox`
(`.codex/scripts/run-research-loop.sh`) inside an isolated worktree. Its tools are filesystem + shell
+ the algua CLI — **no web access**. So #126's `source-ideas` skill ("run deep-research, extract
hypotheses, `research idea add` the survivors") is **un-executable headlessly**: the agent has no
deep-research, no web_search, no fetch. This issue supplies that missing web tooling.

## 2. Scope after the spike + the #126 merge

Two facts reshaped the original design:

- **Capability spike (§3):** headless Codex MCP tool calls require full `--dangerously-bypass`; the
  `workspace-write` sandbox auto-cancels them. So the web-tooled agent runs under bypass. Operator
  decision (2026-06-08): **defer OS isolation to scale-out (VPS/container)** — don't sandbox now.
- **#126 merged first** (idea pool: `ideas` table + `research idea add`/`dedup-check`/`stats`,
  refuted-aware dedup, `DataCapability` parking, and a **collection-only `source-ideas` skill** that
  assumes interactive `deep-research`). That is the structured idea store this issue was going to
  improvise as `kb/research` markdown notes. **We integrate into #126's pool instead** — no parallel
  store, no bespoke validator.

**So #134's enduring contribution is exactly the web TOOLING** that lets the headless agent run
#126's skill: built-in `web_search` + paper-search MCP (arXiv/SSRN) + Firecrawl MCP (extraction).

**What this protects vs. does not:**
- **Protected:** money walls untouched (CLI boundary, signed live gate, CODEOWNERS); sourcing stops
  at `research idea add` (no authoring, no lifecycle, no promote); #126's refuted-aware dedup is the
  discipline wall; the operator's **main working tree is untouched** (sourcing runs in a throwaway
  worktree — important, the repo has concurrent sessions on main).
- **NOT protected (accepted, documented):** under bypass the sourcing agent has full local
  shell/net/fs and could read local secrets / exfil. Closed by OS isolation (VPS/container),
  deferred to scale-out by operator decision (§6).

## 3. Capability spike

Verified against `codex-cli 0.137.0`; full results in
`docs/superpowers/plans/2026-06-08-issue-134-spike-findings.md`. Load-bearing: MCP tool calls need
full bypass; `web_search=live`/`disabled` and per-invocation `-c mcp_servers.X={…}` +
`enabled_tools`/`env_vars`/`startup_timeout_sec` all valid under `--strict-config`; paper-search runs
as `uvx --from paper-search-mcp python -m paper_search_mcp.server` (tools `search_arxiv`/`search_ssrn`
/`search_papers`/`read_paper`).

## 4. Architecture — a dedicated sourcing launcher feeding #126's pool

`.codex/scripts/source-ideas.sh` (new; the research **loop** launcher is untouched):

```
in a throwaway worktree on source-ideas/<stamp>  (keeps the agent off your main working tree):
  uv sync; pre-install pinned MCP servers (warm caches)

  ALGUA_DB_PATH=<main checkout>/data/algua.db \      # ideas persist to the REAL pool, not the
                                                       #   throwaway worktree's DB (ideas are rows,
                                                       #   not files — they don't ride a branch)
  codex exec --dangerously-bypass-approvals-and-sandbox --ignore-user-config --strict-config
    -c web_search=live
    -c 'mcp_servers.papers={command="uvx",
          args=["--from","paper-search-mcp","python","-m","paper_search_mcp.server"],
          startup_timeout_sec=90,tool_timeout_sec=120,
          enabled_tools=["search_arxiv","search_ssrn","search_papers","read_paper"]}'
    [-c 'mcp_servers.firecrawl={command="npx",args=["-y","firecrawl-mcp"],
          env_vars=["FIRECRAWL_API_KEY"],startup_timeout_sec=90,
          enabled_tools=["firecrawl_search","firecrawl_scrape"]}']   # only if key set
    -C "$WORKTREE"
    "<follow source-ideas: research the thesis with the web tools; dedup-check; `research idea add`
      the survivors (up to --max-ideas); NEVER author/promote/touch state beyond `idea add`>"

  cleanup: remove the throwaway worktree (ideas already live in the pool)
```

Why a separate launcher, not a phase of the loop: ideas are DB rows that persist via `ALGUA_DB_PATH`,
not files carried back on a review branch — so sourcing has a different containment + persistence
model than the loop. And #126 is deliberately **collection-only**: the loop does **not** auto-consume
the pool yet (that waits on the gate counting idea breadth — a human/CODEOWNERS change). Keeping
sourcing a separate, deliberate command honors that and leaves the working loop untouched.

### 4.1 Toolset (sourcing run only)

| Tool | Transport | Key? | Role |
|---|---|---|---|
| built-in `web_search` | native | OpenAI acct | discovery (snippets + synthesis) |
| paper-search MCP | `uvx --from paper-search-mcp python -m paper_search_mcp.server` | none | arXiv/SSRN/Semantic Scholar |
| Firecrawl MCP | `npx -y firecrawl-mcp` | `FIRECRAWL_API_KEY` (MCP env only) | clean full-page extraction |

`FIRECRAWL_API_KEY` is forwarded only into the Firecrawl server's env via `env_vars` (no `${VAR}`
interpolation exists); never committed, never on a command line. **If unset, the launcher drops the
Firecrawl `-c` injection** (web_search + paper-search still run); a missing key never aborts a run.
MCP versions default to the package name (latest), overridable via `FIRECRAWL_MCP_VERSION` /
`PAPER_SEARCH_MCP_VERSION` — pin at deploy when supply-chain reproducibility matters (part of §6).

## 5. Skills

- **`.codex/skills/source-ideas/SKILL.md`** (merged with #126's): add an **"Autonomous (headless)
  sourcing"** section — when launched by `source-ideas.sh` the agent has web_search + paper-search +
  Firecrawl in place of interactive deep-research; same untrusted-content discipline; same steps 3–4
  (`dedup-check` → `research idea add`); never author or run the lifecycle.
- **`.claude/skills/source-ideas`** symlink (net-new; #126 added the skill but not the Claude-Code
  mirror) so the skill is reachable to both harnesses.

No `algua/` runtime code, no contracts/integrity files; `algua.knowledge` untouched. The #126 idea
pool + dedup CLI are reused as-is.

## 6. Deferred / out of scope

- **OS isolation of the sourcing agent (containerize / VPS).** Closes the §2 residual; deferred to
  scale-out by operator decision — the primary follow-up.
- **Wiring the pool into the loop as the default ideation step** — #126's deliberate non-default;
  waits on the promotion gate counting idea breadth (a human/CODEOWNERS change).
- **`interpret`/`author` subagents run under bypass despite declared sandboxes** — pre-existing,
  orthogonal; tracked separately.
- Deep-extraction consumers for error-analysis (#132); web/MCP API rate/cost caps; exact MCP version
  pinning (with OS isolation).

## 7. Verification

- `source-ideas.sh --dry-run` shows the bypass + web tools + `research idea` + `ALGUA_DB_PATH` (pool)
  + bounded timeout + isolated branch; `firecrawl: OFF` line when no key. (`tests/test_operator_layer.py`)
- `source-ideas` skill has valid frontmatter and is reachable via the `.claude/skills` symlink.
- Quality gate: `uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports`.

## 8. Quality gate

`uv run pytest -q && uv run ruff check . && uv run mypy algua && uv run lint-imports` at every commit.
