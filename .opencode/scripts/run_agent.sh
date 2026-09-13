#!/usr/bin/env bash
#
# THE AGENT-RUNTIME SEAM. The four loop drivers (research, leap, forage, and any future one) call
# this and nothing else; it is the only file in the repo that names the runtime or a model.
#
# Usage:
#   run_agent.sh --mode research|leap|forage --workdir DIR --prompt-file FILE
#                [--model PROVIDER/MODEL] [--variant high|max|minimal] [--timeout DUR]
#                [--log FILE] [--dry-run]
#
# Exit code is the agent's, except: 124 = timed out (the OS `timeout` convention the drivers
# already read), 2 = usage error, 3 = the runtime is missing or unusable.
#
# WHY A SEAM AT ALL. Under Codex each driver embedded its own `codex exec -s ... -c ...` invocation,
# so the runtime, the sandbox posture and the model choice were smeared across three scripts and a
# user-global config file that no driver passed. Moving to OpenCode with one seam means the next
# runtime change is one file, and `--dry-run` gives the tests a stable contract to assert instead of
# a vendor's flag spelling.
#
# ---------------------------------------------------------------------------------------------
# THREE THINGS THIS SEAM EXISTS TO HANDLE, all measured on opencode 1.18.30 (2026-09-13):
#
# 1. NO KERNEL SANDBOX. Codex's `-s workspace-write` was a real filesystem wall enforced below the
#    agent. OpenCode's permissions are TOOL-LEVEL only: a denied `edit` does not stop `bash` from
#    opening any absolute path, and the loops legitimately need bash for `uv run algua`. So the
#    write wall is re-imposed here with bwrap when it is available (ALGUA_AGENT_SANDBOX=bwrap|none,
#    default auto). This also closes the `/tmp` hole the Codex sandbox spike found.
#
# 2. A PROVIDER ERROR HANGS INSTEAD OF EXITING. A run against an exhausted subscription produced
#    ZERO bytes on stdout and stderr and had to be killed at the timeout. Codex exits immediately.
#    Left alone that turns a quota block into a full-length dead run every firing, so this seam
#    watches the log for a terminal provider error and kills the run early (exit 3), which the
#    drivers' existing rate-limit grep then classifies correctly.
#
# 3. `--pure` DOES NOT DISABLE SKILLS. It disables plugins only; a `--pure` run still loaded 73
#    skills from the user's global directories, with duplicate-name collisions. Hermetic runs need
#    XDG_CONFIG_HOME pointed at the repo AND the two skill-disabling env vars.
# ---------------------------------------------------------------------------------------------
set -uo pipefail

die() { echo "run_agent.sh: $*" >&2; exit 2; }

MODE=""; WORKDIR=""; PROMPT_FILE=""; MODEL=""; VARIANT=""; TIMEOUT="45m"; LOGFILE=""; DRY_RUN=0
while [ $# -gt 0 ]; do
  case "$1" in
    --mode)        MODE="${2:-}"; shift 2 ;;
    --workdir)     WORKDIR="${2:-}"; shift 2 ;;
    --prompt-file) PROMPT_FILE="${2:-}"; shift 2 ;;
    --model)       MODEL="${2:-}"; shift 2 ;;
    --variant)     VARIANT="${2:-}"; shift 2 ;;
    --timeout)     TIMEOUT="${2:-}"; shift 2 ;;
    --log)         LOGFILE="${2:-}"; shift 2 ;;
    --dry-run)     DRY_RUN=1; shift ;;
    *) die "unknown argument: $1" ;;
  esac
done

case "$MODE" in
  research|leap|forage) ;;
  "") die "--mode is required" ;;
  *)  die "unknown --mode: $MODE (expected research|leap|forage)" ;;
esac
[ -n "$WORKDIR" ] || die "--workdir is required"
[ -d "$WORKDIR" ] || die "--workdir does not exist: $WORKDIR"
[ -n "$PROMPT_FILE" ] || die "--prompt-file is required"
[ -f "$PROMPT_FILE" ] || die "--prompt-file does not exist: $PROMPT_FILE"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Model precedence: --model  >  ALGUA_AGENT_MODEL_<MODE>  >  ALGUA_AGENT_MODEL  >  opencode.json.
# Empty means "let opencode.json decide", which is the normal path: the config file is where model
# choices belong, so a driver never has to know one.
if [ -z "$MODEL" ]; then
  mode_upper="$(printf '%s' "$MODE" | tr '[:lower:]' '[:upper:]')"
  eval "MODEL=\"\${ALGUA_AGENT_MODEL_${mode_upper}:-}\""
  [ -n "$MODEL" ] || MODEL="${ALGUA_AGENT_MODEL:-}"
fi

# HERMETIC CONFIG. The user's global opencode config injects plugins, ~90 skills and a global
# AGENTS.md into every run. A cheap model does not need three `brainstorming` skills competing for
# its attention, and an unattended loop must not change behaviour because a human edited their
# personal config. XDG_DATA_HOME is deliberately NOT overridden — that is where auth.json lives.
XDG_DIR="${REPO_ROOT}/.opencode/xdg"
mkdir -p "${XDG_DIR}/opencode"

ENVV=(
  "XDG_CONFIG_HOME=${XDG_DIR}"
  "OPENCODE_DISABLE_EXTERNAL_SKILLS=1"
  "OPENCODE_DISABLE_CLAUDE_CODE_SKILLS=1"
)
# Forage is the only loop that may reach the web, and it does so through opencode's search tool.
[ "$MODE" = "forage" ] && ENVV+=("OPENCODE_ENABLE_EXA=1")

CMD=(opencode run --pure --agent "$MODE" --dir "$WORKDIR")
[ -n "$MODEL" ]   && CMD+=(-m "$MODEL")
[ -n "$VARIANT" ] && CMD+=(--variant "$VARIANT")
CMD+=(--print-logs --log-level INFO)
# The prompt is passed as a FILE, never as argv: Linux caps a single argument at 128 KiB and the
# leap prompt (inspirations + pool state + refuted list) is the one that gets close.
#
# ORDER MATTERS. `--file` is an ARRAY flag, so `-f FILE "message"` swallows the message as a
# second filename and the run dies with `File not found: message`. The message must come FIRST.
# Measured, not assumed: the first form failed every invocation until a live smoke test caught it.
CMD+=("Follow the instructions in the attached file." -f "$PROMPT_FILE")

# KERNEL WRITE WALL. Confine writes to the worktree plus the runtime's own state, so a model that
# talks its way past a tool permission still cannot reach the authoritative DB or the real checkout.
SANDBOX="${ALGUA_AGENT_SANDBOX:-auto}"
if [ "$SANDBOX" = "auto" ]; then
  SANDBOX=none
  command -v bwrap >/dev/null 2>&1 && SANDBOX=bwrap
fi
WRAP=()
if [ "$SANDBOX" = "bwrap" ]; then
  # The runtime needs its OWN state writable or it degrades in confusing ways: with
  # ~/.local/state/opencode read-only it logs "background dependency install failed ... EROFS"
  # and carries on half-initialised. Measured in the first live smoke run. These three paths are
  # the runtime's; everything else outside the worktree stays read-only.
  mkdir -p "${HOME}/.local/share/opencode" "${HOME}/.local/state/opencode" "${HOME}/.cache"
  WRAP=(bwrap --ro-bind / / --dev /dev --proc /proc --tmpfs /tmp
        --bind "$WORKDIR" "$WORKDIR"
        --bind "${HOME}/.local/share/opencode" "${HOME}/.local/share/opencode"
        --bind "${HOME}/.local/state/opencode" "${HOME}/.local/state/opencode"
        --bind "${HOME}/.cache" "${HOME}/.cache"
        --bind "$XDG_DIR" "$XDG_DIR"
        --die-with-parent)
elif [ "$SANDBOX" != "none" ]; then
  die "unknown ALGUA_AGENT_SANDBOX: $SANDBOX (expected auto|bwrap|none)"
fi

FULL=(timeout "$TIMEOUT" env "${ENVV[@]}" "${WRAP[@]}" "${CMD[@]}")

if [ "$DRY_RUN" = "1" ]; then
  printf 'would run:'
  printf ' %q' "${FULL[@]}"
  printf '\n'
  exit 0
fi

command -v opencode >/dev/null 2>&1 || { echo "run_agent.sh: opencode is not on PATH" >&2; exit 3; }

LOGFILE="${LOGFILE:-${WORKDIR}/agent.log}"

# Terminal provider failures. These never resolve by waiting, so the watchdog below kills the run
# the moment one appears rather than letting it burn the whole --timeout doing nothing.
FATAL_RE='Monthly usage limit reached|usage limit|quota exceeded|Token refresh failed|AI_RetryError|insufficient_quota|401 Unauthorized'

"${FULL[@]}" >"$LOGFILE" 2>&1 &
AGENT_PID=$!

(
  while kill -0 "$AGENT_PID" 2>/dev/null; do
    if grep -qE "$FATAL_RE" "$LOGFILE" 2>/dev/null; then
      echo "run_agent.sh: terminal provider error detected; killing the run early" >>"$LOGFILE"
      kill -TERM "$AGENT_PID" 2>/dev/null
      sleep 5
      kill -KILL "$AGENT_PID" 2>/dev/null
      exit 0
    fi
    sleep 10
  done
) &
WATCHDOG_PID=$!

wait "$AGENT_PID"; rc=$?
kill "$WATCHDOG_PID" 2>/dev/null; wait "$WATCHDOG_PID" 2>/dev/null

# A run the watchdog killed is a PROVIDER failure, not a timeout and not a model verdict. Report it
# as 3 so a driver can tell "the provider is down" from "the agent ran and produced nothing".
if [ "$rc" -ne 0 ] && grep -qE "$FATAL_RE" "$LOGFILE" 2>/dev/null; then
  rc=3
fi
cat "$LOGFILE"
exit "$rc"
