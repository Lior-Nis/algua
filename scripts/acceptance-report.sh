#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

report_root="${ALGUA_ACCEPTANCE_REPORT_ROOT:-.runs/acceptance}"
results_dir="$report_root/allure-results"
report_dir="$report_root/allure-report"

mkdir -p "$results_dir" "$report_dir"

uv run pytest -m acceptance \
  "--alluredir=$results_dir" \
  --clean-alluredir

revision="${GITHUB_SHA:-$(git rev-parse HEAD)}"
branch="${GITHUB_REF_NAME:-$(git branch --show-current)}"
generated_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf 'Revision=%s\nBranch=%s\nGenerated.UTC=%s\n' \
  "$revision" "$branch" "$generated_utc" \
  > "$results_dir/environment.properties"

npx --no-install allure generate "$results_dir" \
  --output "$report_dir" \
  --clean

printf 'Acceptance dashboard: %s/index.html\n' "$(cd "$report_dir" && pwd)"
