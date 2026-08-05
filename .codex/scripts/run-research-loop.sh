#!/usr/bin/env bash
#
# Launch an autonomous algua research cycle, driven by a SANDBOXED Codex agent, in the
# EXPLORE-ISOLATED topology.
#
# The agent ideates -> authors -> backtests / walk-forwards / sweeps -> gates (preview
# `research promote`) up to 'candidate'. Two independent walls keep it off the real funnel:
#
#   1. FILESYSTEM CONTAINMENT (the real wall). Codex runs under `-s workspace-write -a never`
#      with its working root at the throwaway worktree, so model-generated shell commands can
#      only WRITE inside the worktree. The authoritative registry (data/algua.db), snapshot
#      tree, kb vault, and main's .git all live OUTSIDE the worktree and are therefore not
#      writable by the agent — even though it is otherwise autonomous. Env-var path routing
#      alone is NOT containment (an unsandboxed agent can write any absolute path); the sandbox
#      is what makes "isolated" true.
#   2. NO AUTHORITATIVE PROMOTE PATH. Everything the agent touches is scratch: a per-run copy
#      of the registry (seeded from the real one via sqlite's consistent online backup) and a
#      per-run copy of the immutable snapshots, both INSIDE the worktree. So its `research
#      promote` is a realistic pass/fail PREVIEW (it sees real accumulated breadth/families/
#      burned holdouts) that can never mutate the real funnel or burn a real holdout. The
#      authoritative promote + code-merge is a separate, human-run `paper merge-back` (trusted
#      code; its diff-policy rejects gate-core edits before the merge) — the only real reconciler.
#
# The agent does NOT commit — it just authors files in the worktree; the DRIVER (trusted, after
# codex exits) commits them on research-run/<stamp>, so the agent needs no git-dir write access.
#
# Bounds: an OS-level `timeout` hard-kills the run; a repo-root flock serializes research cycles;
# the codex exit code propagates (a timeout/failure fails the systemd unit, never a false success).
# Safety: the agent CANNOT go live (human-signed wall) and CANNOT reach the real funnel.
#
# Usage:
#   .codex/scripts/run-research-loop.sh [--hypotheses N] [--timeout DUR] [--thesis TEXT] [--dry-run]
#
set -euo pipefail

N_HYPOTHESES="${N_HYPOTHESES:-3}"
TIMEOUT="${TIMEOUT:-45m}"
SYNC_TIMEOUT="${SYNC_TIMEOUT:-5m}"
THESIS="${THESIS:-a PIT-correct cross-sectional equity edge on the liquid universe}"
# A missing authoritative DB normally means a misconfigured deploy — FAIL CLOSED rather than
# silently preview against an empty funnel. Set ALGUA_ALLOW_EMPTY_FUNNEL=1 for a deliberate
# first-ever cold-start bootstrap.
ALLOW_EMPTY_FUNNEL="${ALGUA_ALLOW_EMPTY_FUNNEL:-0}"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hypotheses) N_HYPOTHESES="$2"; shift 2 ;;
    --timeout)    TIMEOUT="$2"; shift 2 ;;
    --thesis)     THESIS="$2"; shift 2 ;;
    --dry-run)    DRY_RUN=1; shift ;;
    -h|--help)    sed -n '2,32p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

REPO_ROOT="$(git rev-parse --show-toplevel)"
STAMP="$(date +%Y%m%d-%H%M%S)"
BRANCH="research-run/${STAMP}"
WORKTREE="${REPO_ROOT}/../algua-research-${STAMP}"

# The AUTHORITATIVE registry — read only by the DRIVER (to seed the scratch copy), never exposed
# writable to the agent. Resolve it BEFORE we point the agent's ALGUA_DB_PATH at scratch.
AUTH_DB="${ALGUA_DB_PATH:-${REPO_ROOT}/data/algua.db}"
AUTH_DATA_DIR="${ALGUA_DATA_DIR:-${REPO_ROOT}/data}"

# Everything the agent reads/writes lives INSIDE the worktree (so workspace-write contains it):
#   scratch registry (seeded copy) + scratch kb/mlruns + a per-run COPY of the snapshots.
SCRATCH="${WORKTREE}/.funnel-scratch"
export ALGUA_DB_PATH="${SCRATCH}/data/algua.db"
export ALGUA_DATA_DIR="${SCRATCH}/data"
export ALGUA_KNOWLEDGE_DIR="${SCRATCH}/kb"
export ALGUA_MLFLOW_TRACKING_URI="${SCRATCH}/mlruns"
# Keep uv's cache in-workspace so `uv run` needs no write outside the sandbox at agent runtime.
export UV_CACHE_DIR="${WORKTREE}/.uv-cache"

read -r -d '' GOAL <<EOF || true
You are operating the algua research platform autonomously. Use your skills:
operating-algua, run-the-research-loop, author-a-strategy, interpret-results.

Thesis to explore: ${THESIS}.

You are sandboxed and operating a THROWAWAY COPY of the funnel (a per-run scratch registry seeded
from the real one; scratch copies of the immutable snapshots). Nothing you do here touches the
authoritative funnel, so explore freely: 'research promote' on this scratch copy is a realistic
pass/fail PREVIEW, not the real promotion. The REAL authoritative promote + code-merge happens later
when a human runs 'paper merge-back' on a candidate you surface.

Hold the research discipline (it makes your preview trustworthy):
  - Use PIT-CORRECT ADJUSTED prices, never raw close/volume (raw is corporate-action contaminated).
  - Measure breadth with 'backtest sweep' before you promote (the gate requires it).
  - TRUST THE GATE; never pass a relaxation flag (human-only; they fail closed on your path).

Evaluate up to ${N_HYPOTHESES} strategy hypotheses. For each: form a concrete hypothesis, delegate
authoring to the 'author' subagent, then drive it via 'uv run algua ...' (registry add; registry
transition --to backtested; backtest walk-forward; backtest sweep; research promote --universe <name>
--snapshot <id> --start D --end D). Delegate results to the 'interpret' subagent for a promote/discard
call. Never go past 'candidate'; never put a strategy live. Author strategy files under
algua/strategies/<family>/ in this worktree; you do NOT need to 'git commit' — the launcher commits
your files afterward. Write run-report.md at the repo (worktree) root summarizing every hypothesis,
its walk-forward / sweep / preview-gate numbers and the promote/discard reason, and for any strategy
whose preview PASSED, the EXACT 'paper merge-back --branch ${BRANCH} --strategy <name> --universe <u>
--start D --end D' command a human should run for the real authoritative promote + merge.
EOF

# `-s workspace-write` confines model-generated writes to the worktree (verified: a write to any path
# outside it fails "read-only file system"); `approval_policy=never` runs headless (a blocked/failed
# command returns an error to the model, never a prompt). network_access=true lets the agent's shell
# commands use the network (uv, etc.); the threat model here is FUNNEL-WRITE CORRUPTION, closed by
# write-confinement — not network egress (codex's own API is unsandboxed anyway).
CODEX_CMD=(timeout "${TIMEOUT}" codex exec
  -s workspace-write
  -c approval_policy="never"
  -c 'sandbox_workspace_write.network_access=true'
  -C "${WORKTREE}"
  "${GOAL}")

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "DRY RUN — no worktree created, codex not invoked."
  echo "would create worktree:   ${WORKTREE}"
  echo "would create branch:     ${BRANCH}"
  echo "hypotheses: ${N_HYPOTHESES}   timeout: ${TIMEOUT}"
  echo "sandboxed explore-isolated wiring:"
  echo "  AUTH_DB (driver reads to seed scratch; NOT agent-writable): ${AUTH_DB}"
  echo "  AUTH_DATA_DIR (driver copies snapshots from):               ${AUTH_DATA_DIR}"
  echo "  ALGUA_DB_PATH (scratch, in-worktree):     ${ALGUA_DB_PATH}"
  echo "  ALGUA_DATA_DIR (scratch copy):            ${ALGUA_DATA_DIR}"
  echo "  ALGUA_KNOWLEDGE_DIR (scratch):            ${ALGUA_KNOWLEDGE_DIR}"
  echo "  ALGUA_MLFLOW_TRACKING_URI (scratch):      ${ALGUA_MLFLOW_TRACKING_URI}"
  echo "  UV_CACHE_DIR (in-worktree):               ${UV_CACHE_DIR}"
  echo "would run: ${CODEX_CMD[*]}"
  exit 0
fi

# Serialize overlapping research cycles (non-blocking: skip, don't queue). Not a funnel-write lock —
# exploration writes only scratch; the sole authoritative writer, `paper merge-back`, has its own lock.
LOCK="${REPO_ROOT}/data/research-loop.lock"
mkdir -p "$(dirname "${LOCK}")"
exec 9>"${LOCK}"
if ! flock -n 9; then
  echo "another research cycle holds ${LOCK}; skipping this firing." >&2
  exit 0
fi

# Clean up the worktree if we fail during SETUP (before codex runs). Cleared before codex so a real
# run's worktree is kept for review.
cleanup_setup() {
  git -C "${REPO_ROOT}" worktree remove --force "${WORKTREE}" 2>/dev/null || true
}
trap cleanup_setup EXIT

echo "Creating worktree ${WORKTREE} on branch ${BRANCH}..."
git -C "${REPO_ROOT}" worktree add -b "${BRANCH}" "${WORKTREE}" >/dev/null

echo "Building the scratch funnel inside the worktree..."
mkdir -p "${SCRATCH}/data" "${SCRATCH}/kb" "${SCRATCH}/mlruns"

# Seed the scratch registry from the authoritative one via sqlite's CONSISTENT online backup (safe
# even under a concurrent paper write; system python3 stdlib — no venv needed). Missing AUTH_DB FAILS
# CLOSED unless an explicit cold-start bootstrap is requested.
if [[ -f "${AUTH_DB}" ]]; then
  echo "  seeding scratch registry from ${AUTH_DB} (consistent sqlite backup)..."
  python3 - "${AUTH_DB}" "${ALGUA_DB_PATH}" <<'PY'
import sqlite3, sys
src = sqlite3.connect(sys.argv[1]); dst = sqlite3.connect(sys.argv[2])
with dst: src.backup(dst)
src.close(); dst.close()
PY
elif [[ "${ALLOW_EMPTY_FUNNEL}" == "1" ]]; then
  echo "  no authoritative DB at ${AUTH_DB}; ALGUA_ALLOW_EMPTY_FUNNEL=1 -> empty cold-start scratch."
else
  echo "authoritative DB not found at ${AUTH_DB}; refusing to preview against an empty funnel." >&2
  echo "fix the deploy (ALGUA_DB_PATH), or set ALGUA_ALLOW_EMPTY_FUNNEL=1 for a deliberate cold-start." >&2
  exit 1
fi

# Per-run COPY of the immutable snapshots (bars/universes/manifest) so the agent has data to work
# with WITHOUT the real snapshot tree being reachable. -L: copy symlink targets, not links out.
echo "  copying snapshots from ${AUTH_DATA_DIR} ..."
for item in snapshots manifest.jsonl; do
  [[ -e "${AUTH_DATA_DIR}/${item}" ]] && cp -RL "${AUTH_DATA_DIR}/${item}" "${SCRATCH}/data/" || true
done

# A fresh worktree has no .venv; algua installs editable -> the worktree. Build the env up front
# (outside the agent run) so `uv run` doesn't cold-sync mid-agent. Uses the in-worktree UV_CACHE_DIR.
echo "Pre-warming the worktree environment (uv sync, timeout ${SYNC_TIMEOUT})..."
( cd "${WORKTREE}" && timeout "${SYNC_TIMEOUT}" uv sync ) \
  || { echo "pre-warm (uv sync) failed or timed out after ${SYNC_TIMEOUT}; aborting." >&2; exit 1; }

# Setup succeeded — from here the worktree is KEPT for review regardless of the agent's outcome.
trap - EXIT

echo "Running research loop (timeout ${TIMEOUT}, up to ${N_HYPOTHESES} hypotheses), SANDBOXED..."
# stdin from /dev/null: an unattended run has no stdin; without this codex blocks. Capture the exit
# code so a timeout (124) or a codex auth/runtime failure PROPAGATES to systemd.
rc=0
"${CODEX_CMD[@]}" </dev/null || rc=$?
if [[ "${rc}" -ne 0 ]]; then
  echo "codex exec exited ${rc} (timeout=124, or an auth/runtime error) — review the branch anyway." >&2
fi

# The agent doesn't commit (it's sandboxed off the git dir); the DRIVER (trusted) commits its files.
# Scoped to the strategies tree + run-report so nothing stray is swept in. Best-effort: a failed
# commit (e.g. nothing authored) doesn't mask the codex exit code.
echo "Committing any authored strategies on ${BRANCH} (trusted driver)..."
git -C "${WORKTREE}" add algua/strategies run-report.md 2>/dev/null || true
git -C "${WORKTREE}" commit -q -m "research-run ${STAMP}: authored strategies + run report" 2>/dev/null \
  && echo "  committed." || echo "  nothing to commit."

echo
echo "Done. Review the run:"
echo "  git -C ${REPO_ROOT} diff main...${BRANCH}"
echo "  cat ${WORKTREE}/run-report.md"
echo "  # For a PASSED preview, do the REAL authoritative promote + merge (trusted, from main):"
echo "  uv run algua paper merge-back --branch ${BRANCH} --strategy <name> --universe <u> --start D --end D"
echo "When finished, remove the worktree:  git -C ${REPO_ROOT} worktree remove ${WORKTREE}"

# Propagate a failed/timed-out codex run so the systemd unit fails (alerts / no false 'success').
exit "${rc}"
