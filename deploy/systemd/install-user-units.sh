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
#                                  -> DROPPED when that file is missing or unreadable (a unit
#                                     referencing an unusable EnvironmentFile= fails to start);
#                                     KEPT when present AND readable.
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
# Safe to re-run: rendering is deterministic; all units render into a staging dir first (render
# failures abort before any install), then move into place (per-unit atomic renames, NOT
# cross-unit transactional) and daemon-reload ONCE. Recovery from any mid-run failure is simply
# re-running this idempotent installer.
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
# Fail closed on a repo root that plain substitution can't safely put into a unit file: systemd
# splits ExecStart= on whitespace and gives specifier/quote characters meaning, so anything
# outside [A-Za-z0-9/._-] would render a broken or subtly-wrong unit.
if [[ ! "${REPO_ROOT}" =~ ^[A-Za-z0-9/._-]+$ ]]; then
  echo "error: repo root '${REPO_ROOT}' contains whitespace or a systemd-unsafe character" >&2
  echo "       (allowed: A-Za-z0-9 / . _ -). Move the checkout to a safe path and re-run." >&2
  exit 1
fi
UNIT_DIR="${HOME}/.config/systemd/user"
ENV_FILE="/etc/algua/algua.env"
UNITS=(
  algua-research.service
  algua-research.timer
  algua-paper.service
  algua-paper.timer
  algua-mergeback-drain.service
  algua-mergeback-drain.timer
  algua-web.service
)

# Keep EnvironmentFile= only when the file exists AND is readable — a present-but-unreadable
# env file breaks the unit exactly like a missing one.
HAVE_ENV_FILE=1
[[ -f "${ENV_FILE}" && -r "${ENV_FILE}" ]] || HAVE_ENV_FILE=0

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
    # The user manager's default PATH omits ~/.local/bin, where uv (and codex, for the
    # research driver) live — the /opt system deploy gets PATH from the env file instead,
    # so the injection belongs HERE, not in the shared templates. Injected right after
    # [Service] so a template-provided Environment=PATH later in the section would win.
    if [[ "${line}" == "[Service]" ]]; then
      printf 'Environment=PATH=%%h/.local/bin:/usr/local/bin:/usr/bin:/bin\n'
    fi
  done < "${src}"
}

# Fail closed up front if any template is missing (don't install a partial set).
for unit in "${UNITS[@]}"; do
  [[ -f "${SCRIPT_DIR}/${unit}" ]] || { echo "missing unit template: ${SCRIPT_DIR}/${unit}" >&2; exit 1; }
done

if [[ "${HAVE_ENV_FILE}" -eq 0 ]]; then
  echo "note: ${ENV_FILE} missing or unreadable — EnvironmentFile= lines are dropped from the rendered units."
  echo "      Provide ALGUA_* config another way (create/fix the env file and re-run this installer,"
  echo "      or 'systemctl --user set-environment'); algua-paper.service needs ALGUA_PAPER_SNAPSHOT."
fi

if [[ "${DRY_RUN}" -eq 1 ]]; then
  for unit in "${UNITS[@]}"; do
    echo "==> ${unit}  (dry-run; would install to ${UNIT_DIR}/${unit})"
    render_unit "${SCRIPT_DIR}/${unit}"
    echo
  done
  echo "dry-run complete — nothing written, no daemon-reload."
  exit 0
fi

# Preflight: the user systemd manager must be reachable BEFORE any file is written — otherwise
# units could land on disk with no daemon-reload to pick them up.
systemctl --user show-environment >/dev/null \
  || { echo "error: 'systemctl --user' is not reachable (no user manager / DBus session?);" >&2
       echo "       aborting before touching any files." >&2; exit 1; }

# Stage ALL rendered units first, then move them ALL into place, THEN daemon-reload once.
# Render failures abort before any installed unit is replaced. The move loop itself is per-unit
# atomic (same-fs rename), NOT transactional across units: a mid-loop mv failure can leave a
# partially updated, un-reloaded set — the recovery is simply re-running this idempotent
# installer, which re-renders and re-moves everything. Staging inside UNIT_DIR keeps renames
# same-fs.
mkdir -p "${UNIT_DIR}"
STAGE_DIR="$(mktemp -d "${UNIT_DIR}/.stage.XXXXXX")"
trap 'rm -rf "${STAGE_DIR}"' EXIT
for unit in "${UNITS[@]}"; do
  render_unit "${SCRIPT_DIR}/${unit}" > "${STAGE_DIR}/${unit}"
done
for unit in "${UNITS[@]}"; do
  mv "${STAGE_DIR}/${unit}" "${UNIT_DIR}/${unit}"
  echo "installed ${UNIT_DIR}/${unit}"
done

systemctl --user daemon-reload
echo
echo "Units installed and user daemon reloaded. To enable (printed, NOT executed by this script):"
echo "  systemctl --user enable --now algua-research.timer          # research producer, every 2h"
echo "  systemctl --user enable --now algua-paper.timer             # paper tick, every 20m at :07 (session-gated)"
echo "  systemctl --user enable --now algua-mergeback-drain.timer   # merge-back consumer, every 30m"
echo "  systemctl --user enable --now algua-web.service             # read-only monitor on 127.0.0.1:8787"
echo "(algua-research.service / algua-paper.service / algua-mergeback-drain.service are oneshot"
echo " units fired by their timers — do not enable them directly.)"
echo "Consider 'loginctl enable-linger ${USER:-$(id -un)}' so user timers run without an active login."
