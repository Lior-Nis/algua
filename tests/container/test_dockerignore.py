# tests/container/test_dockerignore.py
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_dockerignore_excludes_secrets_and_state():
    lines = {
        ln.strip()
        for ln in (REPO_ROOT / ".dockerignore").read_text().splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    }
    # The CRITICAL finding: never bake secrets, host DBs, snapshots, or per-run state.
    for required in {".env", "data/", "runs/", "mlruns/", ".venv/", ".git", "*.db"}:
        assert required in lines, f"{required!r} must be in .dockerignore"
