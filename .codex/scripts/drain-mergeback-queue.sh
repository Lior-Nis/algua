#!/usr/bin/env bash
#
# Drain the merge-back queue (factory slice 3): the trusted, unsandboxed, no-LLM consumer half of
# the auto merge-back pipeline. `run-research-loop.sh` (the producer) validates each hypothesis's
# merge_back trailer object and enqueues it to data/mergeback-queue.json; THIS script selects
# exactly ONE eligible item per firing and drives the real, authoritative
# `algua paper merge-back` through `algua operator lock-run` — sharing `operator.lock` with the
# daily paper tick, so the two can never mutate the shared checkout/registry concurrently.
#
# One item per cycle, on purpose: a single quality gate already runs ~9 minutes, and this script is
# meant to fire every 30 minutes (see deploy/systemd/algua-mergeback-drain.timer) — draining more
# than one item risks a cycle self-overlapping its own next firing.
#
# Selection RESERVES the item atomically (`select-and-reserve`, flips it to `in_progress` under the
# queue lock in the SAME operation as selection) rather than a read-only select followed by a later
# `record-attempt` — closing a TOCTOU window where a second, overlapping drainer invocation (a
# manual run overlapping the timer, or a wedged prior invocation re-fired by systemd) could select
# and act on the SAME pending item twice before the first one's result lands. A reservation stuck
# `in_progress` past `RESERVATION_STALE_SECONDS` (mergeback_queue.py) is reclaimed automatically.
#
# Usage:
#   .codex/scripts/drain-mergeback-queue.sh [--dry-run]
#
set -euo pipefail

DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

REPO_ROOT="$(git rev-parse --show-toplevel)"
DATA_DIR="${ALGUA_DATA_DIR:-${REPO_ROOT}/data}"
QUEUE_PATH="${ALGUA_MERGEBACK_QUEUE_PATH:-${DATA_DIR}/mergeback-queue.json}"
QUEUE_LOCK_PATH="${ALGUA_MERGEBACK_QUEUE_LOCK_PATH:-${DATA_DIR}/mergeback-queue.lock}"
MAX_MERGEBACK_ATTEMPTS="${MAX_MERGEBACK_ATTEMPTS:-3}"
BACKOFF_MINUTES="${MERGEBACK_BACKOFF_MINUTES_PER_ATTEMPT:-10}"
RESERVATION_STALE_SECONDS="${MERGEBACK_RESERVATION_STALE_SECONDS:-1800}"
QUEUE_MOD="${REPO_ROOT}/.codex/scripts/mergeback_queue.py"
# The repo venv's algua, NOT a bare `algua` from PATH: the systemd user unit's PATH
# (~/.local/bin:/usr/local/bin:...) never contains the project venv, so a bare default made every
# timer-fired drain exit 127 (`algua: command not found`) — classified transient_failure and
# retried silently forever. Still env-overridable for tests/alt checkouts.
ALGUA_BIN="${ALGUA_BIN:-${REPO_ROOT}/.venv/bin/algua}"

if [[ ! -f "${QUEUE_PATH}" ]]; then
  echo "no merge-back queue at ${QUEUE_PATH} yet; nothing to drain."
  exit 0
fi

if [[ "${DRY_RUN}" -eq 1 ]]; then
  # Read-only preview: `select` (never `select-and-reserve`) so a dry run cannot mutate the queue
  # — it must never flip a real item to `in_progress`.
  echo "selecting an eligible merge-back queue item (read-only; DRY RUN) from ${QUEUE_PATH}..."
  SELECTION="$(python3 "${QUEUE_MOD}" select --queue "${QUEUE_PATH}" --lock "${QUEUE_LOCK_PATH}" \
    --max-attempts "${MAX_MERGEBACK_ATTEMPTS}" --backoff-minutes "${BACKOFF_MINUTES}" \
    --format shell)"
  eval "${SELECTION}"
  if [[ "${MERGEBACK_SELECTED:-0}" -ne 1 ]]; then
    echo "no eligible merge-back queue item (all pending/exhausted/terminal); nothing to drain."
    exit 0
  fi
  echo "selected ${MERGEBACK_KEY}: strategy=${MERGEBACK_STRATEGY} universe=${MERGEBACK_UNIVERSE}" \
       "window=${MERGEBACK_START}..${MERGEBACK_END} branch=${MERGEBACK_BRANCH}"
  echo "DRY RUN — would invoke:"
  echo "  ${ALGUA_BIN} operator lock-run -- ${ALGUA_BIN} paper merge-back --branch ${MERGEBACK_BRANCH}" \
       "--strategy ${MERGEBACK_STRATEGY} --universe ${MERGEBACK_UNIVERSE} --start ${MERGEBACK_START}" \
       "--end ${MERGEBACK_END}"
  exit 0
fi

echo "selecting + reserving an eligible merge-back queue item from ${QUEUE_PATH}..."
SELECTION="$(python3 "${QUEUE_MOD}" select-and-reserve --queue "${QUEUE_PATH}" \
  --lock "${QUEUE_LOCK_PATH}" --max-attempts "${MAX_MERGEBACK_ATTEMPTS}" \
  --backoff-minutes "${BACKOFF_MINUTES}" \
  --stale-reservation-seconds "${RESERVATION_STALE_SECONDS}" --format shell)"
eval "${SELECTION}"

if [[ "${MERGEBACK_SELECTED:-0}" -ne 1 ]]; then
  echo "no eligible merge-back queue item (all pending/exhausted/terminal); nothing to drain."
  exit 0
fi

echo "selected ${MERGEBACK_KEY}: strategy=${MERGEBACK_STRATEGY} universe=${MERGEBACK_UNIVERSE}" \
     "window=${MERGEBACK_START}..${MERGEBACK_END} branch=${MERGEBACK_BRANCH}"

# Built as a bash ARRAY (never a shell string) so the pre-validated-but-still-untrusted-origin
# strategy/universe/date values can never be reinterpreted by a shell — argv, not concatenation.
CMD=("${ALGUA_BIN}" operator lock-run -- "${ALGUA_BIN}" paper merge-back
     --branch "${MERGEBACK_BRANCH}" --strategy "${MERGEBACK_STRATEGY}"
     --universe "${MERGEBACK_UNIVERSE}" --start "${MERGEBACK_START}" --end "${MERGEBACK_END}")

echo "invoking: ${CMD[*]}"
set +e
STDOUT_TEXT="$("${CMD[@]}")"
RC=$?
set -e
echo "lock-run exited ${RC}; stdout:"
echo "${STDOUT_TEXT}"

RESULT="$(printf '%s' "${STDOUT_TEXT}" | python3 "${QUEUE_MOD}" record-attempt \
  --queue "${QUEUE_PATH}" --lock "${QUEUE_LOCK_PATH}" --key "${MERGEBACK_KEY}" \
  --max-attempts "${MAX_MERGEBACK_ATTEMPTS}" --stdin)"
echo "queue update: ${RESULT}"

# The drainer itself always exits 0 on a completed selection/classification cycle — a
# gate_failed/terminal_failed/lock_contention/transient_failure outcome is DIGESTED into the queue,
# not a driver-script failure; the queue file (inspectable via the monitor/CLI) is the record of
# what happened. Only a bug in this script itself (set -e above) would abort earlier with non-zero.
exit 0
