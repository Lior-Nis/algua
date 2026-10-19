from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_compose_exposes_the_same_acceptance_suite_as_a_container_gate() -> None:
    compose = yaml.safe_load((REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8"))

    acceptance = compose["services"]["acceptance"]
    assert acceptance["build"] == "."
    assert acceptance["entrypoint"] == ["uv", "run", "pytest"]
    assert acceptance["command"] == ["-m", "acceptance", "-q"]
    assert acceptance["network_mode"] == "none"
