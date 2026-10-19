#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

results_dir=".runs/acceptance/allure-results"
report_dir=".runs/acceptance/allure-report"

uv run pytest -m acceptance \
  "--alluredir=$results_dir" \
  --clean-alluredir
npx --no-install allure generate "$results_dir" \
  --output "$report_dir" \
  --clean

printf 'Acceptance dashboard: %s/%s/index.html\n' "$repo_root" "$report_dir"
