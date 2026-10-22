from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "acceptance-dashboard.yml"


def test_acceptance_dashboard_is_published_as_a_persistent_pages_site() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    required_contract = (
        "push:\n    branches: [main]",
        "schedule:",
        "workflow_dispatch:",
        "permissions:\n  contents: read",
        "pages: write",
        "id-token: write",
        "environment:\n      name: github-pages",
        "scripts/acceptance-report.sh",
        "path: .runs/acceptance/allure-report",
        "uses: actions/deploy-pages@v4",
    )
    missing = [fragment for fragment in required_contract if fragment not in workflow]

    assert not missing, f"acceptance dashboard workflow is missing: {missing}"
