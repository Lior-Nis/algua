from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from algua.primitives.bounded_subprocess import BoundedCompletion
from algua.registry.frozen_source import FrozenFile
from algua.registry.planner_environment import (
    CREATE_FLAGS,
    SYNC_FLAGS,
    EnvironmentIncompatible,
    build_environment_key,
    current_interpreter_identity,
    inventory_environment,
    materialize_argv,
    provision_environment,
    scrubbed_environment,
    validate_lock,
    verify_environment,
)
from tests._venv_fixture import uv_like_venv
from tests._walk_faults import fail_scandir_once

WHEEL_HASH = "sha256:" + "a" * 64


def _inputs() -> tuple[FrozenFile, ...]:
    return (
        FrozenFile(".python-version", "100644", b"3.12\n"),
        FrozenFile("pyproject.toml", "100644", b"[project]\nname='algua'\nversion='0'\n"),
        FrozenFile(
            "uv.lock", "100644",
            (
                "version=1\n[[package]]\nname='x'\nversion='1'\n"
                "source={registry='https://pypi.org/simple'}\n"
                f"wheels=[{{url='https://files.pythonhosted.org/x.whl',hash='{WHEEL_HASH}'}}]\n"
            ).encode(),
        ),
    )


def test_environment_key_binds_inputs_interpreter_uv_and_exact_flags() -> None:
    key = build_environment_key(_inputs(), "a" * 64, uv_version="uv 0.9.26")
    assert key.interpreter == current_interpreter_identity()
    assert key.create_argv == CREATE_FLAGS
    assert key.sync_argv == SYNC_FLAGS
    assert key.digest != build_environment_key(
        _inputs(), "a" * 64, uv_version="uv 0.9.27").digest


def test_normative_flags_exclude_project_resolution_and_links() -> None:
    assert "--relocatable" in CREATE_FLAGS
    for flag in (
        "--locked", "--no-dev", "--no-default-groups", "--no-editable",
        "--no-install-project", "--no-install-workspace", "--no-install-local",
        "--no-build", "--no-python-downloads", "--no-config",
    ):
        assert flag in SYNC_FLAGS
    assert SYNC_FLAGS[SYNC_FLAGS.index("--link-mode") + 1] == "copy"


REPO = Path(__file__).resolve().parents[1]
SPEC = REPO / (
    "docs/development/specs/spec-story-1-3b-artifact-environment-contract/"
    "artifact-environment-contract.md")


def test_the_keyed_argv_is_exactly_the_normative_argv() -> None:
    section = SPEC.read_text(encoding="utf-8").split("## Environment construction", 1)[1]
    block = section.split("```text\n", 1)[1].split("```", 1)[0]
    create, sync = block.split("\nuv sync ", 1)
    assert tuple(create.replace("<env>", "<environment>").split()) == CREATE_FLAGS
    assert ("uv", "sync", *sync.split()) == SYNC_FLAGS


def test_the_installed_uv_runs_the_exact_keyed_argv_against_the_committed_lock(
    tmp_path: Path,
) -> None:
    # A real offline dry run of the production argv under the production environment: uv parses
    # every flag (conflicts included), checks the committed lock and plans the install, without
    # network or installing anything. An argv uv refuses can never provision an environment.
    uv = shutil.which("uv")
    assert uv is not None, "frozen environment construction requires uv"
    build_root, environment = tmp_path / "build", tmp_path / "environment"
    (build_root / ".home").mkdir(parents=True)
    for name in ("pyproject.toml", "uv.lock", ".python-version"):
        (build_root / name).write_bytes((REPO / name).read_bytes())
    env = scrubbed_environment(Path(uv).parent, home=build_root / ".home")
    places = {"<exact-current-interpreter>": Path(sys.executable), "<environment>": environment,
              "<private-build-input-root>": build_root}

    for argv, extra_env in (
        (CREATE_FLAGS, {}),
        ((*SYNC_FLAGS, "--dry-run", "--offline"), {"VIRTUAL_ENV": str(environment)}),
    ):
        ran = subprocess.run(
            [uv, *materialize_argv(argv, places)[1:]], cwd=build_root, env={**env, **extra_env},
            capture_output=True, text=True, timeout=120, check=False)
        assert ran.returncode == 0, ran.stderr
    assert not list(environment.glob("lib/python*/site-packages/*.dist-info"))


def test_scrubbed_environment_has_no_inherited_authority(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALGUA_DB_PATH", "/secret")
    monkeypatch.setenv("ALPACA_API_KEY", "secret")
    monkeypatch.setenv("PYTHONPATH", "/checkout")
    monkeypatch.setenv("HTTPS_PROXY", "http://secret")
    env = scrubbed_environment(Path("/tmp/bin"))
    assert env == {
        "HOME": "/nonexistent", "PATH": "/tmp/bin", "PYTHONDONTWRITEBYTECODE": "1",
        "LANG": env["LANG"], "LC_ALL": env["LC_ALL"],
    }


@pytest.mark.parametrize(
    "source",
    ["editable='.'", "path='../x'", "git='https://example/repo'", "url='file:///tmp/x.whl'"],
)
def test_lock_rejects_local_or_vcs_dependencies(source: str) -> None:
    lock = f"version=1\n[[package]]\nname='bad'\nversion='1'\nsource={{ {source} }}\n".encode()
    with pytest.raises(EnvironmentIncompatible):
        validate_lock(lock)


def test_lock_requires_a_wheel_for_every_registry_package() -> None:
    lock = (b"version=1\n[[package]]\nname='bad'\nversion='1'\n"
            b"source={registry='https://pypi.org/simple'}\nsdist={url='https://example/x.tgz'}\n")
    with pytest.raises(EnvironmentIncompatible, match="wheel"):
        validate_lock(lock)


@pytest.mark.parametrize(
    "wheel",
    [
        "{url='http://files.pythonhosted.org/x.whl',hash='sha256:" + "a" * 64 + "'}",
        "{url='https://files.pythonhosted.org/x.whl',hash='sha256:aa'}",
        "{url='https://files.pythonhosted.org/x.whl'}",
    ],
)
def test_lock_requires_canonical_registry_wheel_url_and_hash(wheel: str) -> None:
    lock = (
        "version=1\n[[package]]\nname='bad'\nversion='1'\n"
        "source={registry='https://pypi.org/simple'}\n"
        f"wheels=[{wheel}]\n"
    ).encode()
    with pytest.raises(EnvironmentIncompatible, match="wheel"):
        validate_lock(lock)


def test_python_pin_must_match_running_minor() -> None:
    wrong = list(_inputs())
    wrong[0] = FrozenFile(".python-version", "100644", b"9.9\n")
    with pytest.raises(EnvironmentIncompatible, match="Python pin"):
        build_environment_key(tuple(wrong), "a" * 64, uv_version="uv 0.9.26")
    assert sys.version_info[:2] == (3, 12)


def test_environment_inventory_and_isolated_probe(tmp_path: Path) -> None:
    env = uv_like_venv(tmp_path / "env")
    site = env / "lib/python3.12/site-packages"
    package = site / "numpy"
    dist = site / "numpy-2.3.3.dist-info"
    package.mkdir()
    dist.mkdir()
    (package / "__init__.py").write_text("__version__ = '2.3.3'\n")
    (dist / "METADATA").write_text("Name: numpy\nVersion: 2.3.3\n")
    inventory = inventory_environment(env)
    assert [(item.name, item.version) for item in inventory.distributions] == [("numpy", "2.3.3")]
    assert inventory.digest
    assert {link.path for link in inventory.interpreter_links} == {
        "bin/python", "bin/python3", "bin/python3.12",
    }
    verify_environment(env, current_interpreter_identity(), inventory.digest)

    (env / "bin/python3").unlink()
    with pytest.raises(EnvironmentIncompatible, match="interpreter link"):
        inventory_environment(env)


def test_environment_inventory_rejects_algua_hardlinks_and_unexpected_symlinks(
    tmp_path: Path,
) -> None:
    env = tmp_path / "env"
    site = env / "lib/python3.12/site-packages"
    dist = site / "algua-1.dist-info"
    dist.mkdir(parents=True)
    (dist / "METADATA").write_text("Name: algua\nVersion: 1\n")
    with pytest.raises(EnvironmentIncompatible, match="Algua"):
        inventory_environment(env)
    (dist / "METADATA").write_text("Name: safe\nVersion: 1\n")
    os.symlink("METADATA", dist / "bad-link")
    with pytest.raises(EnvironmentIncompatible, match="symlink"):
        inventory_environment(env)


def test_provision_uses_exact_uv_commands_and_private_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The key is built for a fixed uv version, so pin the installed version it is rechecked
    # against; otherwise any other installed uv reads as key drift.
    monkeypatch.setattr(
        "algua.registry.planner_environment.installer_version", lambda: "uv 0.9.26")
    calls: list[tuple[list[str], Path, dict[str, str], int]] = []
    environment = tmp_path / "environment"
    build_root = tmp_path / "inputs"

    def runner(argv, *, cwd, env, timeout, max_stdout, max_stderr):
        calls.append((argv, cwd, env, timeout))
        if argv[1] == "venv":
            uv_like_venv(environment)
            (environment / "lib64").symlink_to("lib")  # created by `uv venv`, then removed
        return BoundedCompletion(0, b"", b"")

    key = build_environment_key(_inputs(), "a" * 64, uv_version="uv 0.9.26")
    inventory = provision_environment(
        build_root, environment, _inputs(), key, runner=runner)

    assert [call[0] for call in calls] == [
        [item.replace("<exact-current-interpreter>", sys.executable).replace(
            "<environment>", str(environment)) for item in CREATE_FLAGS],
        [item.replace("<private-build-input-root>", str(build_root)) for item in SYNC_FLAGS],
    ]
    assert calls[1][2]["VIRTUAL_ENV"] == str(environment)
    assert calls[0][2]["HOME"] == str(build_root / ".home")
    assert calls[0][3] > 0 and calls[1][3] > 0
    assert not (environment / "lib64").exists()
    assert (build_root / "uv.lock").read_bytes() == _inputs()[2].data
    assert inventory.digest


def test_provision_rechecks_key_and_uv_before_running(tmp_path: Path, monkeypatch) -> None:
    key = build_environment_key(_inputs(), "a" * 64, uv_version="uv 0.9.26")
    monkeypatch.setattr(
        "algua.registry.planner_environment.installer_version", lambda: "uv 0.9.27",
    )
    called = False

    def runner(*_args, **_kwargs):
        nonlocal called
        called = True

    with pytest.raises(EnvironmentIncompatible, match="key"):
        provision_environment(
            tmp_path / "inputs", tmp_path / "environment", _inputs(), key, runner=runner,
        )
    assert called is False


def test_missing_uv_is_incompatible_not_retryable(monkeypatch) -> None:
    monkeypatch.setattr("algua.registry.planner_environment.shutil.which", lambda _name: None)
    from algua.registry.planner_environment import installer_version

    with pytest.raises(EnvironmentIncompatible, match="uv"):
        installer_version()


def test_distribution_identity_reads_only_the_metadata_header_block() -> None:
    from algua.registry.planner_environment_inventory import _metadata_identity

    raw = (
        "Metadata-Version: 2.1\nName: VectorBT_Pro\nVersion: 1.0.0\n\n"
        "Example output:\nName: (10, 20, ETH-USD), dtype: object\nVersion: nonsense value\n"
    )
    identity = _metadata_identity(raw)
    assert (identity.name, identity.version) == ("vectorbt-pro", "1.0.0")


@pytest.mark.parametrize(
    "raw",
    [
        "Name: numpy\nName: numpy\nVersion: 2.3.3\n",
        "Name: numpy\n\nVersion: 2.3.3\n",
        "Name: num py\nVersion: 2.3.3\n",
        "Name: numpy\nVersion: 2 .3\n",
    ],
)
def test_distribution_identity_rejects_ambiguous_or_malformed_headers(raw: str) -> None:
    from algua.registry.planner_environment_inventory import _metadata_identity

    with pytest.raises(EnvironmentIncompatible):
        _metadata_identity(raw)


@pytest.mark.parametrize(
    "declared,canonical",
    [("Zope.Interface", "zope-interface"), ("ruamel.yaml.clib", "ruamel-yaml-clib"),
     ("typing__extensions", "typing-extensions"), ("A-_.B", "a-b"), ("numpy", "numpy")],
)
def test_distribution_identity_collapses_separator_runs(declared: str, canonical: str) -> None:
    from algua.registry.planner_environment_inventory import _metadata_identity

    assert _metadata_identity(f"Name: {declared}\nVersion: 1.0\n").name == canonical


@pytest.mark.parametrize("declared", ["algua", "ALGUA", "Algua"])
def test_distribution_identity_forbids_algua_in_any_spelling(declared: str) -> None:
    from algua.registry.planner_environment_inventory import _metadata_identity

    with pytest.raises(EnvironmentIncompatible, match="Algua"):
        _metadata_identity(f"Name: {declared}\nVersion: 1\n")


def test_environment_inventory_rejects_distributions_equal_after_canonicalization(
    tmp_path: Path,
) -> None:
    env = tmp_path / "env"
    site = env / "lib/python3.12/site-packages"
    for directory, declared in (("zope.interface-1.dist-info", "zope.interface"),
                                ("zope_interface-2.dist-info", "Zope_Interface")):
        (site / directory).mkdir(parents=True)
        (site / directory / "METADATA").write_text(f"Name: {declared}\nVersion: 1\n")
    with pytest.raises(EnvironmentIncompatible, match="duplicate"):
        inventory_environment(env)


def test_environment_inventory_propagates_traversal_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = tmp_path / "env"
    uv_like_venv(env)
    hidden = env / "lib/python3.12/site-packages/hidden"
    hidden.mkdir()
    (hidden / "__init__.py").write_text("SMUGGLED = True\n")
    (env / "lib/python3.12/site-packages/visible.py").write_text("x = 1\n")
    failed = fail_scandir_once(monkeypatch, lambda path: path == hidden)

    with pytest.raises(PermissionError):
        inventory_environment(env)
    assert failed == [hidden]


def test_provisioning_the_same_key_twice_publishes_the_same_environment_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Real uv (and venv) write the absolute staging path into activation scripts; the frozen
    # environment is never activated, so those scripts must not enter its identity (Story 1.3c
    # found every admission publishing a new ~1 GB environment).
    monkeypatch.setattr(
        "algua.registry.planner_environment.installer_version", lambda: "uv 0.9.26")
    key = build_environment_key(_inputs(), "a" * 64, uv_version="uv 0.9.26")

    def provision(staging: Path) -> str:
        environment = staging / "environment"

        def runner(argv, *, cwd, env, timeout, max_stdout, max_stderr):
            if argv[1] == "venv":
                uv_like_venv(environment)
                cfg = environment / "pyvenv.cfg"  # like uv: no `command = <path>` line
                cfg.write_text("".join(line for line in cfg.read_text().splitlines(True)
                                       if not line.startswith("command")))
                (environment / "bin" / "activate.csh").write_text(
                    f"setenv VIRTUAL_ENV '{environment}'\n")
            return BoundedCompletion(0, b"", b"")

        inventory = provision_environment(
            staging / "inputs", environment, _inputs(), key, runner=runner)
        assert not list((environment / "bin").glob("activate*"))
        return inventory.digest

    assert provision(tmp_path / "one") == provision(tmp_path / "two")
