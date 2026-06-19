#!/usr/bin/env bash
# PreToolUse(Bash) guard: block a DIRECT `git push` to main so changes route through a PR.
# Reads the hook JSON on stdin, inspects .tool_input.command, and emits a PreToolUse "deny"
# decision when the command is a git push that targets main. Anything else exits 0 (allow).
# Not airtight (covers `git push [origin] main`, `git push -u origin main`, and bare
# `git push`/`git push origin` while the current branch is main) — it's a guardrail, not a wall.
set -euo pipefail

input="$(cat)"
cmd="$(printf '%s' "$input" | jq -r '.tool_input.command // ""')"

# Only consider git push invocations; let everything else through untouched.
printf '%s' "$cmd" | grep -Eq '(^|[;&|[:space:](])git[[:space:]]+push([[:space:]]|$)' || exit 0

deny() {
  reason='Direct pushes to main are disabled in this repo — route changes through a PR: push your feature branch, run `gh pr create`, then merge with `gh pr merge`. (guard: .claude/hooks/block-push-to-main.sh)'
  printf '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"deny","permissionDecisionReason":%s}}' \
    "$(printf '%s' "$reason" | jq -Rs .)"
  exit 0
}

# 1) Explicit ref to main: `git push origin main`, `git push origin +main`, `git push origin HEAD:main`.
if printf '%s' "$cmd" | grep -Eq '(^|[[:space:]:+])main([[:space:]"'\'':]|$)'; then
  deny
fi

# 2) Bare push (no explicit refspec) uses the branch upstream — dangerous only when on main.
args="$(printf '%s' "$cmd" | sed -E 's/.*git[[:space:]]+push[[:space:]]*//')"
nonflags="$(printf '%s' "$args" | tr ' ' '\n' | grep -Ev '^-' | grep -cE '.' || true)"
if [ "${nonflags:-0}" -le 1 ]; then
  branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '')"
  [ "$branch" = "main" ] && deny
fi

exit 0
