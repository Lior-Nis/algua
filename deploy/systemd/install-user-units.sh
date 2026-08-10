#!/usr/bin/env bash
#
# Idempotent installer: render the deploy/systemd unit TEMPLATES (written for a system-level
# /opt/algua deploy) into per-user units at ~/.config/systemd/user/, pointed at THIS checkout.
# Closes the template<->installed drift that appears when units are hand-edited into place.
#
# Rendering, per unit:
#   - every /opt/algua path        -> the ACTUAL repo root (git rev-parse from this script's
#                                     own location — works from any checkout/worktree);
#   - EnvironmentFile=/etc/algua/algua.env
#                                  -> DROPPED when that file does not exist (a unit referencing a
#                                     missing EnvironmentFile= fails to start); KEPT when present.
#                                     UnsetEnvironment= lines are ALWAYS kept — they scrub broker
#                                     creds regardless of where the environment came from;
#   - WantedBy=multi-user.target   -> default.target (user managers have no multi-user.target);
#                                     timers keep timers.target unchanged.
# Then `systemctl --user daemon-reload`. Enable commands are PRINTED, never executed — enabling
# a timer/service is a deliberate operator action, not a side effect of installing files.
#
# Usage:
#   deploy/systemd/install-user-units.sh [--dry-run]
#
#   --dry-run   print the rendered units to stdout; write nothing, no daemon-reload.
#
# Safe to re-run: rendering is deterministic and installs are atomic overwrites (mktemp + mv).
#
set -euo pipefail

DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) sed -n '2,26p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
UNIT_DIR="${HOME}/.config/systemd/user"
ENV_FILE="/etc/algua/algua.env"
UNITS=(
  algua-research.service
  algua-research.timer
  algua-paper.service
  algua-paper.timer
  algua-web.service
)

HAVE_ENV_FILE=1
[[ -f "${ENV_FILE}" ]] || HAVE_ENV_FILE=0

# Pure-bash line rendering: bash ${var//pat/repl} replacement is LITERAL, so a repo path containing
# sed/awk metacharacters (&, \, delimiters) can never corrupt the rendered unit.
render_unit() {
  local src="$1" line
  while IFS= read -r line; do
    if [[ "${line}" == "EnvironmentFile=${ENV_FILE}" && "${HAVE_ENV_FILE}" -eq 0 ]]; then
      continue
    fi
    if [[ "${line}" == "WantedBy=multi-user.target" ]]; then
      line="WantedBy=default.target"
    fi
    printf '%s\n' "${line//\/opt\/algua/${REPO_ROOT}}"
  done < "${src}"
}

# Fail closed up front if any template is missing (don't install a partial set).
for unit in "${UNITS[@]}"; do
  [[ -f "${SCRIPT_DIR}/${unit}" ]] || { echo "missing unit template: ${SCRIPT_DIR}/${unit}" >&2; exit 1; }
done

if [[ "${HAVE_ENV_FILE}" -eq 0 ]]; then
  echo "note: ${ENV_FILE} not found — EnvironmentFile= lines are dropped from the rendered units."
  echo "      Provide ALGUA_* config another way (create the env file and re-run this installer,"
  echo "      or 'systemctl --user set-environment'); algua-paper.service needs ALGUA_PAPER_SNAPSHOT."
fi

for unit in "${UNITS[@]}"; do
  src="${SCRIPT_DIR}/${unit}"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "==> ${unit}  (dry-run; would install to ${UNIT_DIR}/${unit})"
    render_unit "${src}"
    echo
  else
    mkdir -p "${UNIT_DIR}"
    tmp="$(mktemp "${UNIT_DIR}/.${unit}.tmp.XXXXXX")"
    render_unit "${src}" > "${tmp}"
    mv "${tmp}" "${UNIT_DIR}/${unit}"
    echo "installed ${UNIT_DIR}/${unit}"
  fi
done

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "dry-run complete — nothing written, no daemon-reload."
  exit 0
fi

systemctl --user daemon-reload
echo
echo "Units installed and user daemon reloaded. To enable (printed, NOT executed by this script):"
echo "  systemctl --user enable --now algua-research.timer   # research producer, every 2h"
echo "  systemctl --user enable --now algua-paper.timer      # paper tick, daily 21:30 UTC (calendar-gated)"
echo "  systemctl --user enable --now algua-web.service      # read-only monitor on 127.0.0.1:8787"
echo "(algua-research.service / algua-paper.service are oneshot units fired by their timers —"
echo " do not enable them directly.)"
echo "Consider 'loginctl enable-linger ${USER:-$(id -un)}' so user timers run without an active login."
