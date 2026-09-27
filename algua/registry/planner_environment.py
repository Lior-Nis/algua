"""Pinned interpreter and uv policy for shared frozen planner environments."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import sysconfig
import tomllib
from pathlib import Path
from typing import Any

from algua.registry.artifact_contract import (
    ArtifactFile,
    BuildInputs,
    EnvironmentKey,
    InstalledDistribution,
    InstalledInventory,
    InterpreterIdentity,
)
from algua.registry.frozen_source import FrozenFile

CREATE_FLAGS = (
    "uv", "venv", "--relocatable", "--python", "<exact-current-interpreter>",
    "--no-python-downloads", "<environment>",
)
SYNC_FLAGS = (
    "uv", "sync", "--project", "<private-build-input-root>", "--active", "--locked",
    "--no-dev", "--no-default-groups", "--no-editable", "--no-install-project",
    "--no-install-workspace", "--no-install-local", "--no-build", "--no-python-downloads",
    "--link-mode", "copy", "--no-env-file", "--no-config", "--no-progress",
)


class EnvironmentIncompatible(ValueError):
    """Committed inputs cannot produce the normative frozen environment."""


class EnvironmentUnavailable(RuntimeError):
    """A selected compatible locked wheel cannot currently be acquired."""


def installer_version() -> str:
    """Return the exact uv version used by the keyed provisioning policy."""
    uv = shutil.which("uv")
    if uv is None:
        raise EnvironmentUnavailable("uv is unavailable for frozen environment acquisition")
    try:
        result = subprocess.run(
            [uv, "--version"], env=scrubbed_environment(Path(uv).parent), check=True,
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise EnvironmentUnavailable("uv version could not be determined") from exc
    version = result.stdout.strip()
    if not version or len(version) > 128 or "\n" in version:
        raise EnvironmentIncompatible("uv returned an invalid version identity")
    return version


def current_interpreter_identity() -> InterpreterIdentity:
    return InterpreterIdentity(
        implementation=platform.python_implementation(),
        version=platform.python_version(),
        cache_tag=sys.implementation.cache_tag or "unknown",
        soabi=sysconfig.get_config_var("SOABI") or "unknown",
        platform_tag=sysconfig.get_platform(),
        os_name=platform.system().lower(),
        machine=platform.machine().lower(),
    )


def _by_path(inputs: tuple[FrozenFile, ...]) -> dict[str, FrozenFile]:
    result = {item.path: item for item in inputs}
    expected = {".python-version", "pyproject.toml", "uv.lock"}
    if set(result) != expected:
        raise EnvironmentIncompatible("environment build inputs are incomplete")
    return result


def validate_lock(raw: bytes) -> None:
    try:
        payload = tomllib.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise EnvironmentIncompatible("committed uv lock is invalid") from exc
    packages = payload.get("package")
    if not isinstance(packages, list):
        raise EnvironmentIncompatible("committed uv lock has no package inventory")
    for package in packages:
        if not isinstance(package, dict):
            raise EnvironmentIncompatible("committed uv lock package is invalid")
        source = package.get("source")
        name = package.get("name")
        if name == "algua" and source == {"editable": "."}:
            continue
        if not isinstance(source, dict) or set(source) != {"registry"}:
            raise EnvironmentIncompatible(
                "local, editable, URL and VCS dependencies are unsupported")
        wheels = package.get("wheels")
        if not isinstance(wheels, list) or not wheels:
            raise EnvironmentIncompatible("every locked registry package requires a wheel")


def build_environment_key(
    inputs: tuple[FrozenFile, ...], dependency_hash: str, *, uv_version: str,
) -> EnvironmentKey:
    by_path = _by_path(inputs)
    try:
        pin = by_path[".python-version"].data.decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        raise EnvironmentIncompatible("Python pin is not UTF-8") from exc
    current_minor = f"{sys.version_info.major}.{sys.version_info.minor}"
    if pin != current_minor:
        raise EnvironmentIncompatible("Python pin does not match the running interpreter")
    validate_lock(by_path["uv.lock"].data)
    entries = tuple(item.contract_entry for item in inputs)
    return EnvironmentKey(
        build_inputs_digest=BuildInputs(entries).digest,
        dependency_hash=dependency_hash,
        interpreter=current_interpreter_identity(),
        uv_version=uv_version,
        create_argv=CREATE_FLAGS,
        sync_argv=SYNC_FLAGS,
    )


def scrubbed_environment(binary_path: Path) -> dict[str, str]:
    """Return a replacement environment, never a filtered copy of inherited authority."""
    home = os.environ.get("HOME", "/nonexistent")
    locale = os.environ.get("LANG", "C.UTF-8")
    return {
        "HOME": home,
        "PATH": str(binary_path),
        "PYTHONDONTWRITEBYTECODE": "1",
        "LANG": locale,
        "LC_ALL": os.environ.get("LC_ALL", locale),
    }


def materialize_argv(template: tuple[str, ...], substitutions: dict[str, Path]) -> list[str]:
    result: list[str] = []
    for item in template:
        replacement: Any = substitutions.get(item)
        result.append(str(replacement) if replacement is not None else item)
    return result


def _metadata_identity(raw: str) -> InstalledDistribution:
    fields: dict[str, str] = {}
    for line in raw.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            if key in {"Name", "Version"}:
                fields[key] = value.strip()
    if set(fields) != {"Name", "Version"}:
        raise EnvironmentIncompatible("installed distribution metadata is incomplete")
    name = fields["Name"].lower().replace("_", "-")
    if name == "algua":
        raise EnvironmentIncompatible("installed Algua distribution is forbidden")
    return InstalledDistribution(name, fields["Version"])


def _permitted_interpreter_link(root: Path, path: Path) -> bool:
    if path.parent != root / "bin" or path.name not in {"python", "python3", "python3.12"}:
        return False
    resolved = path.resolve()
    base = Path(getattr(sys, "_base_executable", sys.executable)).resolve()
    return resolved == base or root.resolve() in resolved.parents


def inventory_environment(root: Path) -> InstalledInventory:
    if root.is_symlink() or not root.is_dir():
        raise EnvironmentIncompatible("environment root is not a real directory")
    files: list[ArtifactFile] = []
    distributions: list[InstalledDistribution] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        directory = Path(dirpath)
        for name in dirnames:
            if (directory / name).is_symlink():
                raise EnvironmentIncompatible("environment contains an unexpected symlink")
        for name in filenames:
            path = directory / name
            if path.is_symlink():
                if not _permitted_interpreter_link(root, path):
                    raise EnvironmentIncompatible("environment contains an unexpected symlink")
                continue
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise EnvironmentIncompatible(
                    "environment contains a non-regular or hardlinked file")
            if path.suffix in {".pyc", ".pyo"} or "__pycache__" in path.parts:
                raise EnvironmentIncompatible("environment contains generated bytecode")
            data = path.read_bytes()
            relative = path.relative_to(root).as_posix()
            mode = "100755" if info.st_mode & stat.S_IXUSR else "100644"
            files.append(ArtifactFile(
                relative, mode, len(data), hashlib.sha256(data).hexdigest()))
            if path.name == "METADATA" and path.parent.name.endswith(".dist-info"):
                distributions.append(_metadata_identity(data.decode("utf-8")))
    files.sort(key=lambda item: item.path.encode())
    distributions.sort(key=lambda item: (item.name, item.version))
    return InstalledInventory(tuple(distributions), tuple(files))


def verify_environment(
    root: Path, expected_interpreter: InterpreterIdentity, expected_inventory_digest: str,
) -> None:
    inventory = inventory_environment(root)
    if inventory.digest != expected_inventory_digest:
        raise EnvironmentIncompatible("installed environment inventory drifted")
    python = root / "bin/python"
    probe = (
        "import importlib.util,json,platform,sys,sysconfig;"
        "print(json.dumps({'implementation':platform.python_implementation(),"
        "'version':platform.python_version(),'cache_tag':sys.implementation.cache_tag or 'unknown',"
        "'soabi':sysconfig.get_config_var('SOABI') or 'unknown',"
        "'platform_tag':sysconfig.get_platform(),'os_name':platform.system().lower(),"
        "'machine':platform.machine().lower(),"
        "'algua':importlib.util.find_spec('algua') is not None}))"
    )
    try:
        result = subprocess.run(
            [str(python), "-I", "-c", probe], cwd=root, env=scrubbed_environment(python.parent),
            check=True, capture_output=True, text=True, timeout=30,
        )
        observed = json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        raise EnvironmentIncompatible(
            "published environment interpreter verification failed") from exc
    has_algua = observed.pop("algua", None)
    if has_algua is not False or observed != expected_interpreter.to_dict():
        raise EnvironmentIncompatible("published environment interpreter identity drifted")


def _expanded(template: tuple[str, ...], replacements: dict[str, str]) -> list[str]:
    return [replacements.get(item, item) for item in template]


def provision_environment(
    build_root: Path,
    environment: Path,
    inputs: tuple[FrozenFile, ...],
    key: EnvironmentKey,
    *,
    runner: Any = subprocess.run,
) -> InstalledInventory:
    """Create one private environment; publication remains a separate atomic step."""
    if build_root.exists() or environment.exists():
        raise EnvironmentIncompatible("environment staging paths must be absent")
    build_root.mkdir(parents=True, mode=0o700)
    for item in inputs:
        destination = build_root / item.path
        destination.write_bytes(item.data)
        destination.chmod(0o600)
    uv = shutil.which("uv")
    if uv is None:
        raise EnvironmentUnavailable("uv is unavailable for frozen environment acquisition")
    base_env = scrubbed_environment(Path(uv).parent)
    create = _expanded(CREATE_FLAGS, {
        "<exact-current-interpreter>": sys.executable,
        "<environment>": str(environment),
    })
    sync = _expanded(SYNC_FLAGS, {"<private-build-input-root>": str(build_root)})
    try:
        runner(create, cwd=build_root, env=base_env, check=True, capture_output=True, text=True)
        sync_env = {**base_env, "VIRTUAL_ENV": str(environment)}
        runner(sync, cwd=build_root, env=sync_env, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        diagnostic = (exc.stderr or "").lower()
        if any(token in diagnostic for token in ("download", "network", "timeout", "connection")):
            raise EnvironmentUnavailable(
                "a compatible locked wheel is temporarily unavailable") from exc
        raise EnvironmentIncompatible("locked environment provisioning failed") from exc
    inventory = inventory_environment(environment)
    verify_environment(environment, key.interpreter, inventory.digest)
    return inventory
