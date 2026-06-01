#!/usr/bin/env bash
#
# Launch an autonomous algua research run.
#
# A Codex agent operates the research lifecycle (ideate -> author -> backtest /
# walk-forward / sweep -> gate -> shortlist) per its skills, INSIDE an isolated git
# worktree on its own branch, then writes a run report. Nothing touches your working
# tree or main; you review the branch afterward and merge what's worth keeping.
#
# Containment + bounds:
#   - runs in a throwaway worktree on  research-run/<stamp>
#   - hard-bounded by an OS-level `timeout` (kills the process no matter what the agent does)
#   - the goal caps the number of hypotheses (intent-level bound)
# Safety: algua's CLI is the boundary (the agent cannot go live); integrity files are
# CODEOWNERS-protected (edits can't merge without a human).
#
# Usage:
#   .codex/scripts/run-research-loop.sh [--hypotheses N] [--timeout DUR] [--thesis TEXT] [--dry-run]
#
set -euo pipefail

N_HYPOTHESES="${N_HYPOTHESES:-3}"
TIMEOUT="${TIMEOUT:-30m}"
THESIS="${THESIS:-ride institutional/whale momentum}"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hypotheses) N_HYPOTHESES="$2"; shift 2 ;;
    --timeout)    TIMEOUT="$2"; shift 2 ;;
    --thesis)     THESIS="$2"; shift 2 ;;
    --dry-run)    DRY_RUN=1; shift ;;
    -h|--help)    sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

REPO_ROOT="$(git rev-parse --show-toplevel)"
STAMP="$(date +%Y%m%d-%H%M%S)"
BRANCH="research-run/${STAMP}"
WORKTREE="${REPO_ROOT}/../algua-research-${STAMP}"

read -r -d '' GOAL <<EOF || true
You are operating the algua research platform autonomously. Use your skills:
operating-algua, run-the-research-loop, author-a-strategy, interpret-results.

Thesis to explore: ${THESIS}.

Evaluate exactly ${N_HYPOTHESES} strategy hypotheses. For each one: form a concrete
hypothesis, delegate authoring to the 'author' subagent, then drive it through the
lifecycle ONLY via 'uv run algua ...' (backtest run --register, backtest walk-forward,
optionally sweep, then research promote to gate backtested->shortlisted). Delegate the
results to the 'interpret' subagent for a promote/discard recommendation. Trust the
gate; never lower its thresholds. Never go past 'shortlisted' and never edit the
human-owned safety files.

When you are done (or if you are running low on time), ensure every authored strategy
file is committed on this branch and write run-report.md at the repo root summarizing
every hypothesis, its results, the gate decision, and what you shortlisted or discarded
and why.
EOF

CODEX_CMD=(timeout "${TIMEOUT}" codex exec
  --dangerously-bypass-approvals-and-sandbox
  -C "${WORKTREE}"
  "${GOAL}")

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "DRY RUN — no worktree created, codex not invoked."
  echo "would create worktree: ${WORKTREE}"
  echo "would create branch:   ${BRANCH}"
  echo "hypotheses: ${N_HYPOTHESES}   timeout: ${TIMEOUT}"
  echo "would pre-warm env:    uv sync (in ${WORKTREE})"
  echo "would run: ${CODEX_CMD[*]}"
  exit 0
fi

echo "Creating worktree ${WORKTREE} on branch ${BRANCH}..."
git -C "${REPO_ROOT}" worktree add -b "${BRANCH}" "${WORKTREE}"

# A fresh worktree has no .venv, and algua installs editable -> the worktree path (so the agent's
# authored strategies are importable). Build the env once here, up front, instead of letting the
# first `uv run` cold-sync mid-agent-run. uv's global cache makes this fast after the first ever sync.
echo "Pre-warming the worktree environment (uv sync)..."
( cd "${WORKTREE}" && uv sync )

echo "Running research loop (timeout ${TIMEOUT}, ${N_HYPOTHESES} hypotheses)..."
# stdin from /dev/null: the goal is passed as an argument, and an unattended/cron run has no
# stdin — without this, codex blocks reading stdin and the run hangs.
"${CODEX_CMD[@]}" </dev/null || echo "codex exec exited non-zero (timeout or error) — review the branch anyway."

echo
echo "Done. Review the run:"
echo "  git -C ${REPO_ROOT} diff main...${BRANCH}"
echo "  cat ${WORKTREE}/run-report.md"
echo "  uv run algua registry list --stage shortlisted"
echo "When finished, remove the worktree:  git -C ${REPO_ROOT} worktree remove ${WORKTREE}"
