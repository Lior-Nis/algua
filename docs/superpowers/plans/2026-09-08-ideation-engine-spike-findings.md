# Ideation Engine — Codex 0.149 Sandbox Spike Findings (#626)

**Purpose:** The ideation-engine spec's privilege claims (§5, §6, §9) rest on the #134 spike
against codex 0.137. This re-verifies those claims on the currently-installed codex before any
launcher is written. Spike only — no code changes.

**Codex version installed:** `codex-cli 0.149.0` (`codex --version`), model `gpt-5.6-sol`,
provider `openai`.

## Results

| # | Probe | Command (see brief) | Expected | Observed | Verdict |
|---|-------|----------------------|----------|----------|---------|
| 1 | Built-in web search works sandboxed | `codex exec -s workspace-write -c approval_policy=never -c web_search=live '...arXiv 1706.03762...'` | `WEB_OK:Attention Is All You Need` | `WEB_OK:Attention Is All You Need` | **PASS** |
| 2 | Shell network off when told | `codex exec -s workspace-write -c approval_policy=never -c 'sandbox_workspace_write.network_access=false' 'curl ... example.com ...'` | `NET_OFF` | `NET_OFF` (curl: `Could not resolve host: example.com`) | **PASS** |
| 3 | Writes outside the workspace fail | `codex exec -s workspace-write -c approval_policy=never "touch /tmp/algua-spike-$$ ..."` | `BLOCKED` | `WROTE` | **FAIL** (`/tmp` is a declared writable root; see 3b) |
| 3b | Writes outside the workspace fail — real authority path (`$HOME`, not `/tmp`) | `codex exec -s workspace-write -c approval_policy=never "touch \$HOME/algua-spike-probe3b ..."` | `BLOCKED` | `BLOCKED` (`touch: cannot touch '/home/liornisimov/algua-spike-probe3b': Read-only file system`) | **PASS** |
| 4 | MCP tool call under workspace-write (opt-in path) | `codex exec -s workspace-write -c approval_policy=never --strict-config -c 'mcp_servers.papers={...}' 'Call search_arxiv ...'` | FAIL expected per spec (opt-in path stays bypass-only); PASS would be notable | Literal command: hard config-load error, exit 1, unrelated to sandbox (see below). Rerun without `--strict-config`: `MCP_FAIL:search_arxiv tool unavailable` | **FAIL** (matches spec's expected FAIL; see caveat) |

## Per-probe detail

**Probe 1 — web search.** Ran clean, exit 0, one line of stdout exactly as expected. The sandbox
banner for this run read `sandbox: workspace-write [workdir, /tmp, $TMPDIR] (network access
enabled)` — notable because `network_access` was **not** explicitly set in this invocation, yet
the banner already shows it enabled (see cross-cutting note below). Launcher takeaway: forage/leap
can rely on `-c web_search=live` for research without opening the shell network — the tool has its
own path independent of `sandbox_workspace_write.network_access`.

**Probe 2 — shell network off.** Ran clean, exit 0. With `network_access=false` explicitly set,
the banner drops the "(network access enabled)" suffix and the in-sandbox `curl` fails DNS
resolution (`Could not resolve host`), landing on the `NET_OFF` branch. Launcher takeaway:
`sandbox_workspace_write.network_access=false` is a real, working kill switch for the agent's
shell — set it explicitly on any launcher invocation that must not shell out to the network,
since the default (see below) is *not* to block it.

**Probe 3 — writes outside the workspace.** This did **not** match the brief's expectation. The
`touch /tmp/algua-spike-<pid>` succeeded (`WROTE`, exit 0) rather than being `BLOCKED`. The
sandbox banner explains why: codex 0.149's default `workspace-write` policy grants write access
to `[workdir, /tmp, $TMPDIR]` — `/tmp` is a standing writable root, not part of "the workspace."
This is a deterministic, repeatable result (not ambiguous — no rerun performed), so it is recorded
as observed. Launcher takeaway: assumption (c) — "writes outside the workspace are blocked" — is
**false as literally stated for `/tmp`**; it only holds for paths outside `{workdir, /tmp,
$TMPDIR}`. Any launcher that treats "outside the workspace" as a containment boundary must treat
`/tmp` as agent-writable too (e.g. don't rely on `/tmp` to hold anything the agent shouldn't be
able to touch/exfiltrate-via, and don't assume a stray `/tmp` write from a misbehaving probe would
be caught by this wall). Probe 3 alone doesn't test the property the launchers actually depend on
— writes to real authority paths (repo config, `$HOME`, credentials) outside the worktree — since
`/tmp` is a codex-declared exception, not "outside the sandbox" in the sense that matters. See
Probe 3b for that test.

**Probe 3b — writes outside the workspace, real authority path (`$HOME`).** Run from a fresh
throwaway `mktemp -d` workspace exactly like probe 3, but targeting `$HOME` instead of `/tmp`:
```
timeout 3m codex exec -s workspace-write -c approval_policy=never \
  "Run: touch \$HOME/algua-spike-probe3b && echo WROTE || echo BLOCKED. Print only that word." </dev/null
```
Exit 0, stdout `BLOCKED`. Transcript: `touch: cannot touch
'/home/liornisimov/algua-spike-probe3b': Read-only file system` → `BLOCKED`. Sandbox banner for
this run: `sandbox: workspace-write [workdir, /tmp, $TMPDIR] (network access enabled)` — identical
declared-roots list to probes 1 and 3; `$HOME` is conspicuously absent from it, and the write is
in fact refused. Confirmed no file was created (`ls $HOME/algua-spike-probe3b` → "No such file or
directory"), so no cleanup was needed. **Verdict: PASS.** Launcher takeaway: the containment wall
does hold for the paths that actually matter — repo config, `$HOME`, credentials, anything outside
`{workdir, /tmp, $TMPDIR}` — so assumption (c) is correct in the sense the launchers rely on
(no writes to real authority paths outside the worktree); the earlier probe-3 caveat is scoped
narrowly to `/tmp`/`$TMPDIR` themselves being a declared exception, not a hole in the wall generally.

**Probe 4 — MCP tool call under workspace-write.** The command exactly as given in the brief
(with `--strict-config`) failed before ever reaching the sandbox: `~/.codex/config.toml:101`
carries `features.experimental_use_rmcp_client = true`, a field codex 0.149 doesn't recognize
under `--strict-config`, so it aborts with a config-parse error (exit 1) unrelated to MCP-under-
sandbox. Per the brief's own instruction not to modify anything outside the findings file, the
global `~/.codex/config.toml` was left untouched. Re-running the identical probe without
`--strict-config` produced `MCP_FAIL:search_arxiv tool unavailable`, exit 0. Session-transcript
inspection (`~/.codex/sessions/.../rollout-*-<session-id>.jsonl`) shows the model ran a JS filter
over its own `ALL_TOOLS` list for anything matching `search_arxiv`/`arxiv`, found nothing, and
reported failure immediately — it never attempted to invoke or wait on the `papers` server, and
there is no trace anywhere in the transcript of an MCP server named `papers` starting, timing out,
or erroring. A `codex debug prompt-input` dump with the same `-c mcp_servers.papers=...` override
also shows no `papers`/`search_arxiv` tool registered. This leaves the *root cause* ambiguous
between two candidates — (a) an ad-hoc `mcp_servers.<name>={...}` table passed via `-c` on the
command line is not recognized the way a `[mcp_servers.<name>]` section in `config.toml` is, or
(b) MCP servers genuinely don't get exposed to the model under `workspace-write` — but the
**top-line observation is not ambiguous**: under the exact invocation style the brief specifies,
the MCP tool call did not succeed, either via the literal command (hard config error) or the
`--strict-config`-free variant (tool never appeared, call failed). That matches the spec's existing
assumption (d) and its own stated expectation ("expects FAIL → MCP stays opt-in with bypass").
Launcher takeaway: keep MCP tool calls off the default `workspace-write` launcher path; Task 8
(or a future spike) should confirm root cause (a) vs (b) with a proper `config.toml`-defined
server before ever considering enabling MCP under the sandbox — this run does not clear that bar,
package-spec note: `paper-search-mcp==0.1.3` was used as given in the brief and did not itself
error out (no "package not found" surfaced); the pin was not the blocker here.

## Cross-cutting note: default network access

Probes 1 and 3 both ran under the *default* `workspace-write` sandbox (no `network_access` flag
set) and both showed `(network access enabled)` in the banner. Probe 2 is the only run that
explicitly disabled it. This means on this install, **shell network access is on by default**
under `workspace-write` unless a launcher explicitly sets
`sandbox_workspace_write.network_access=false` — assumption (b) ("`network_access=false` cuts the
agent's shell network") is confirmed as a working *opt-out*, not a default. Any launcher that
wants a network-locked-down agent must set the flag explicitly; omitting it does not fail closed.

## Commit

`docs: ideation engine — codex 0.149 sandbox spike findings (#626)` — adds only this file.
