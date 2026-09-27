"""Pinned interpreter and uv policy for shared frozen planner environments."""
from __future__ import annotations

import platform
import re
import shutil
import subprocess
import sys
import sysconfig
import tomllib
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from algua.registry.artifact_contract import BuildInputs
from algua.registry.environment_contract import (
    EnvironmentKey,
    InstalledInventory,
    InterpreterIdentity,
)
from algua.registry.frozen_source import FrozenFile
from algua.registry.planner_environment_errors import (
    EnvironmentIncompatible,
    EnvironmentUnavailable,
)
from algua.registry.planner_environment_inventory import (
    inventory_environment,
    scrubbed_environment,
    verify_environment,
)

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


def installer_version() -> str:
    """Return the exact uv version used by the keyed provisioning policy."""
    uv = shutil.which("uv")
    if uv is None:
        raise EnvironmentIncompatible("uv is unavailable for frozen environment construction")
    try:
        result = subprocess.run(
            [uv, "--version"], env=scrubbed_environment(Path(uv).parent), check=True,
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise EnvironmentIncompatible("uv version could not be determined") from exc
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
        for wheel in wheels:
            if not isinstance(wheel, dict):
                raise EnvironmentIncompatible("locked wheel metadata is invalid")
            url = wheel.get("url")
            digest = wheel.get("hash")
            parsed = urlsplit(url) if isinstance(url, str) else None
            if (
                parsed is None or parsed.scheme != "https" or not parsed.netloc
                or parsed.username is not None or parsed.password is not None or parsed.fragment
                or not isinstance(digest, str)
                or re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None
            ):
                raise EnvironmentIncompatible("locked wheel URL or hash is not canonical")


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


def materialize_argv(template: tuple[str, ...], substitutions: dict[str, Path]) -> list[str]:
    result: list[str] = []
    for item in template:
        replacement: Any = substitutions.get(item)
        result.append(str(replacement) if replacement is not None else item)
    return result


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
    private_home = build_root / ".home"
    private_home.mkdir(mode=0o700)
    for item in inputs:
        destination = build_root / item.path
        destination.write_bytes(item.data)
        destination.chmod(0o600)
    uv = shutil.which("uv")
    if uv is None:
        raise EnvironmentIncompatible("uv is unavailable for frozen environment construction")
    observed_key = build_environment_key(
        inputs, key.dependency_hash, uv_version=installer_version(),
    )
    if observed_key != key:
        raise EnvironmentIncompatible("environment provisioning key drifted")
    base_env = scrubbed_environment(Path(uv).parent, home=private_home)
    create = _expanded(CREATE_FLAGS, {
        "<exact-current-interpreter>": sys.executable,
        "<environment>": str(environment),
    })
    sync = _expanded(SYNC_FLAGS, {"<private-build-input-root>": str(build_root)})
    try:
        runner(create, cwd=build_root, env=base_env, check=True, capture_output=True, text=True,
               timeout=900)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise EnvironmentIncompatible("locked environment creation failed") from exc
    lib64 = environment / "lib64"
    if lib64.is_symlink() and lib64.resolve() == (environment / "lib").resolve():
        lib64.unlink()
    try:
        sync_env = {**base_env, "VIRTUAL_ENV": str(environment)}
        runner(sync, cwd=build_root, env=sync_env, check=True, capture_output=True, text=True,
               timeout=900)
    except subprocess.TimeoutExpired as exc:
        raise EnvironmentUnavailable(
            "a compatible locked wheel is temporarily unavailable") from exc
    except subprocess.CalledProcessError as exc:
        diagnostic = (exc.stderr or "").lower()
        if any(token in diagnostic for token in ("download", "network", "timeout", "connection")):
            raise EnvironmentUnavailable(
                "a compatible locked wheel is temporarily unavailable") from exc
        raise EnvironmentIncompatible("locked environment provisioning failed") from exc
    inventory = inventory_environment(environment)
    verify_environment(environment, key.interpreter, inventory.digest)
    return inventory
