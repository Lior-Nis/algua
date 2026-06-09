# Issue 134 — Codex 0.137.0 capability spike findings

Run as a GO/NO-GO gate before building. Scratch dir `/tmp/algua-spike`. Model auto-selected `gpt-5.5`.

## Results

| # | Mechanic | Probe | Result |
|---|---|---|---|
| A | `web_search=disabled` key valid + bypass runs w/ `--ignore-user-config --strict-config` | bypass, "print CODEX_OK" | **PASS** — exit 0, ran, model auto-selected, no trust prompt |
| B | built-in `web_search=live` provides a usable web tool | sandboxed, do a search | **PASS** — `web search: …` → `WEB_OK` |
| E | `web_search=disabled` truly removes web **under bypass** | bypass, try to search | **PASS** — `NO_WEB_TOOL` |
| D | `workspace-write` blocks the agent's **shell** network | sandboxed `curl example.com` | **PASS** — `EXIT=6` (couldn't resolve host) |
| — | config keys parse under `--strict-config`: `mcp_servers.X`, `enabled_tools`, `env_vars`, `startup_timeout_sec`, `tool_timeout_sec`, `sandbox_workspace_write.network_access`, `approval_policy` | various | **PASS** — all accepted |
| C6 | paper-search MCP tool callable | **bypass** | **PASS** — `mcp: papers/search_arxiv (completed)` → `PAPERS_OK` |
| C4/C5/C7 | paper-search MCP tool callable **without full bypass** | `workspace-write`; `+network_access=true`; `+approval_policy=never` | **FAIL (all)** — `started` then `(failed) user cancelled MCP tool call` |

## The load-bearing finding

**Headless `codex exec` external MCP tool calls require `--dangerously-bypass-approvals-and-sandbox`.**
Under `workspace-write` they are auto-cancelled even with `network_access=true` *and*
`approval_policy=never` — the call needs an approver headless exec can't provide; only full-access
pre-grants it. Built-in `web_search` is the exception (server-side; works sandboxed). Sandbox modes
are exactly `read-only | workspace-write | danger-full-access` — no write-confined + MCP-OK middle ground.

**Consequence:** "sandboxed agent" and "MCP tools (Firecrawl/paper-search)" are mutually exclusive in
0.137.0. Operator decisions (2026-06-08): (1) **tools-first** — run the web-tooled agent under bypass;
(2) **defer OS isolation to scale-out (VPS/container)** — don't OS-sandbox now. Accepted residual: a
prompt-injected sourcing run has full local shell/net/fs (can read local secrets / exfil). Money walls
(CLI boundary, signed live gate, CODEOWNERS) untouched.

## Verified exact invocations

- `codex exec --dangerously-bypass-approvals-and-sandbox --ignore-user-config --strict-config
  -c web_search=live -c 'mcp_servers.papers={…}' [-c 'mcp_servers.firecrawl={…}']`
- **paper-search command:** `uvx --from paper-search-mcp python -m paper_search_mcp.server`
  (the bare `paper-search-mcp` "does not provide any executables"). Tools: `search_arxiv`,
  `search_ssrn`, `search_papers`, `read_paper`. Add `startup_timeout_sec=90`, `tool_timeout_sec=120`
  (cold `uvx` start is slow).
- `web_search` config key is top-level `-c web_search=live|disabled` (accepted under `--strict-config`).
