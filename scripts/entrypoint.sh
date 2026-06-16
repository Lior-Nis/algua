#!/usr/bin/env bash
set -euo pipefail
# Footgun guard (NOT a security wall, see spec §0): refuse the documented holdout-burning
# promote on an isolated per-run DB unless explicitly opted into authoritative mode.
# The real hard wall belongs in the CLI and is a deferred follow-up.
if [[ "${1:-}" == "research" && "${2:-}" == "promote" \
      && "${ALGUA_DB_PATH:-}" == /app/runs/* && "${ALGUA_ALLOW_PROMOTE:-0}" != "1" ]]; then
  echo "refusing 'research promote' on isolated per-run DB (${ALGUA_DB_PATH:-unset}):" >&2
  echo "point ALGUA_DB_PATH at the authoritative DB, or set ALGUA_ALLOW_PROMOTE=1." >&2
  exit 4
fi
exec uv run algua "$@"
