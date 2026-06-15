#!/usr/bin/env bash
set -euo pipefail
RUN_ID="${1:-}"; shift || true
[[ "$RUN_ID" =~ ^[A-Za-z0-9_.-]+$ && "$RUN_ID" != *..* && ! "$RUN_ID" =~ ^\.+$ ]] \
  || { echo "invalid RUN_ID: ${RUN_ID:-<empty>}" >&2; echo "usage: scripts/run.sh <RUN_ID> <algua args...>" >&2; exit 2; }
mkdir -p ./runs
dir="./runs/$RUN_ID"
if [[ "${ALGUA_REUSE:-0}" == "1" ]]; then
  mkdir -p "$dir"                          # explicit reuse
else
  mkdir "$dir" 2>/dev/null \
    || { echo "run dir $dir exists; set ALGUA_REUSE=1 to reuse" >&2; exit 3; }
fi
HOST_UID="$(id -u)"; HOST_GID="$(id -g)"   # bash UID is readonly; GID is not a builtin — derive both
export RUN_ID HOST_UID HOST_GID
exec docker compose run --rm algua "$@"
