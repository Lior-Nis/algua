"""Strict inventory and isolated verification of frozen planner environments."""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from email.parser import HeaderParser
from pathlib import Path
from typing import Any

from algua.primitives.bounded_subprocess import OutputLimitExceeded, run_bounded
from algua.primitives.strict_walk import strict_walk
from algua.registry.artifact_contract import ArtifactFile, canonical_json, canonical_relative_path
from algua.registry.environment_contract import (
    BASE_INTERPRETER,
    MAX_DISTRIBUTION_METADATA_BYTES,
    MAX_ENVIRONMENT_BYTES,
    MAX_ENVIRONMENT_FILE_BYTES,
    MAX_ENVIRONMENT_FILES,
    InstalledDistribution,
    InstalledInventory,
    InterpreterIdentity,
    InterpreterLink,
)
from algua.registry.planner_environment_errors import EnvironmentIncompatible

_CHUNK_SIZE = 1024 * 1024
_SEPARATOR_RUN = re.compile(r"[-_.]+")
_PROBE_TIMEOUT_SECONDS = 30
_PROBE_OUTPUT_BYTES = 4096
_PROBE_IDENTITY_FIELDS = frozenset(
    {"implementation", "version", "cache_tag", "soabi", "platform_tag", "os_name", "machine"})
# `-I -S`: no site module, so no `.pth` line, sitecustomize or usercustomize executes and no
# bytecode is imported from the environment. Only the environment's direct import root (argv) is
# added, so `find_spec('algua')` sees installed top-level packages without processing any `.pth`.
_PROBE = (
    "import importlib.util,json,platform,sys,sysconfig\n"
    "sys.path.extend(sys.argv[1:])\n"
    "print(json.dumps({'implementation':platform.python_implementation(),"
    "'version':platform.python_version(),'cache_tag':sys.implementation.cache_tag or 'unknown',"
    "'soabi':sysconfig.get_config_var('SOABI') or 'unknown',"
    "'platform_tag':sysconfig.get_platform(),'os_name':platform.system().lower(),"
    "'machine':platform.machine().lower(),"
    "'algua':importlib.util.find_spec('algua') is not None},"
    "sort_keys=True,separators=(',',':'),ensure_ascii=False))\n"
)


def scrubbed_environment(binary_path: Path, *, home: Path | None = None) -> dict[str, str]:
    """Return a replacement environment, never a filtered copy of inherited authority."""
    locale = os.environ.get("LANG", "C.UTF-8")
    return {
        "HOME": str(home) if home is not None else "/nonexistent",
        "PATH": str(binary_path),
        "PYTHONDONTWRITEBYTECODE": "1",
        "LANG": locale,
        "LC_ALL": os.environ.get("LC_ALL", locale),
    }


def _metadata_identity(raw: str) -> InstalledDistribution:
    """Read exactly one Name and Version from the header block, never the description body."""
    headers = HeaderParser().parsestr(raw)
    names = headers.get_all("Name") or []
    versions = headers.get_all("Version") or []
    if len(names) != 1 or len(versions) != 1:
        raise EnvironmentIncompatible("installed distribution metadata is incomplete")
    name = _SEPARATOR_RUN.sub("-", str(names[0]).strip()).lower()  # PEP 503 canonical name
    if name == "algua":
        raise EnvironmentIncompatible("installed Algua distribution is forbidden")
    try:
        return InstalledDistribution(name, str(versions[0]).strip())
    except ValueError as exc:
        raise EnvironmentIncompatible("installed distribution identity is malformed") from exc


def _file_digest(path: Path, limit: int) -> tuple[int, str]:
    """Stream a digest, never reading more than one byte past ``limit`` even if the file grew."""
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(min(_CHUNK_SIZE, limit - size + 1)):
            size += len(chunk)
            if size > limit:
                raise EnvironmentIncompatible("environment file exceeds the per-file bound")
            digest.update(chunk)
    return size, digest.hexdigest()


def _read_metadata(path: Path, limit: int) -> str:
    with path.open("rb") as handle:
        raw = handle.read(limit + 1)
    if len(raw) > limit:
        raise EnvironmentIncompatible("installed distribution metadata exceeds its bound")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EnvironmentIncompatible("installed distribution metadata is not UTF-8") from exc


def _interpreter_link(root: Path, path: Path) -> InterpreterLink:
    required = {"python", "python3", f"python{sys.version_info.major}.{sys.version_info.minor}"}
    if path.parent != root / "bin" or path.name not in required:
        raise EnvironmentIncompatible("environment contains an unexpected symlink")
    resolved = path.resolve(strict=True)
    base = Path(getattr(sys, "_base_executable", sys.executable)).resolve()
    raw_target = Path(os.readlink(path))
    if raw_target.is_absolute():
        if resolved != base:
            raise EnvironmentIncompatible("interpreter link escapes its keyed base interpreter")
        target = BASE_INTERPRETER
    else:
        if resolved != base and root.resolve() not in resolved.parents:
            raise EnvironmentIncompatible("interpreter link escapes its environment")
        lexical_target = Path(os.path.normpath(path.parent / raw_target))
        try:
            target = lexical_target.relative_to(root).as_posix()
        except ValueError as exc:
            raise EnvironmentIncompatible("interpreter link escapes its environment") from exc
    return InterpreterLink(path.relative_to(root).as_posix(), target)


def _canonical(root: Path, path: Path) -> str:
    """The entry's canonical bounded relative path, refused before anything is read from it."""
    try:
        return canonical_relative_path(path.relative_to(root).as_posix(), "environment path")
    except ValueError as exc:
        raise EnvironmentIncompatible("environment path is not canonical and bounded") from exc


def inventory_environment(root: Path) -> InstalledInventory:
    """Inventory every entry; uv creates no empty directory, so any directory must be implied."""
    if root.is_symlink() or not root.is_dir():
        raise EnvironmentIncompatible("environment root is not a real directory")
    files: list[ArtifactFile] = []
    links: list[InterpreterLink] = []
    distributions: list[InstalledDistribution] = []
    directories: set[str] = set()
    entries = 0
    total = 0
    for dirpath, dirnames, filenames in strict_walk(root):
        directory = Path(dirpath)
        if directory != root:
            directories.add(directory.relative_to(root).as_posix())
        for name in dirnames:
            if (directory / name).is_symlink():
                raise EnvironmentIncompatible("environment contains an unexpected symlink")
        for name in filenames:
            if entries >= MAX_ENVIRONMENT_FILES:
                raise EnvironmentIncompatible("environment exceeds the file-count bound")
            entries += 1
            path = directory / name
            relative = _canonical(root, path)
            if path.is_symlink():
                links.append(_interpreter_link(root, path))
                continue
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise EnvironmentIncompatible(
                    "environment contains a non-regular or hardlinked file"
                )
            if path.suffix in {".pyc", ".pyo"} or "__pycache__" in path.parts:
                raise EnvironmentIncompatible("environment contains generated bytecode")
            if info.st_size > MAX_ENVIRONMENT_FILE_BYTES:
                raise EnvironmentIncompatible("environment file exceeds the per-file bound")
            if total + info.st_size > MAX_ENVIRONMENT_BYTES:
                raise EnvironmentIncompatible("environment exceeds the aggregate size bound")
            size, digest = _file_digest(path, MAX_ENVIRONMENT_FILE_BYTES)
            total += size
            if total > MAX_ENVIRONMENT_BYTES:
                raise EnvironmentIncompatible("environment exceeds the aggregate size bound")
            mode = "100755" if info.st_mode & stat.S_IXUSR else "100644"
            files.append(ArtifactFile(relative, mode, size, digest))
            if path.name == "METADATA" and path.parent.name.endswith(".dist-info"):
                raw = _read_metadata(path, MAX_DISTRIBUTION_METADATA_BYTES)
                distributions.append(_metadata_identity(raw))
    implied = {
        parent.as_posix()
        for entry in (*(item.path for item in files), *(link.path for link in links))
        for parent in Path(entry).parents
        if parent.as_posix() != "."
    }
    if directories != implied:
        raise EnvironmentIncompatible("environment contains an uninventoried directory")
    names = [item.name for item in distributions]
    if len(set(names)) != len(names):
        raise EnvironmentIncompatible("environment has duplicate installed distributions")
    versioned_python = f"bin/python{sys.version_info.major}.{sys.version_info.minor}"
    expected_links = {"bin/python", "bin/python3", versioned_python}
    if {link.path for link in links} != expected_links:
        raise EnvironmentIncompatible("environment interpreter link inventory is incomplete")
    files.sort(key=lambda item: item.path.encode())
    links.sort(key=lambda item: item.path.encode())
    distributions.sort(key=lambda item: (item.name, item.version))
    return InstalledInventory(tuple(distributions), tuple(files), tuple(links))


def _parse_probe(raw: bytes) -> tuple[dict[str, str], bool]:
    """Accept exactly one canonical identity object followed by one newline, nothing else.

    Canonical equality also refuses duplicate keys, whitespace and trailing output: none of them
    can round-trip to the canonical text of the decoded object.
    """
    try:
        text = raw.decode("utf-8")
        value: Any = json.loads(text)
        canonical = canonical_json(value) + "\n" if isinstance(value, dict) else None
    except ValueError as exc:
        raise EnvironmentIncompatible("environment interpreter probe output is malformed") from exc
    if (
        not isinstance(value, dict) or set(value) != {*_PROBE_IDENTITY_FIELDS, "algua"}
        or type(value["algua"]) is not bool
        or any(type(value[field]) is not str for field in _PROBE_IDENTITY_FIELDS)
        or text != canonical
    ):
        raise EnvironmentIncompatible(
            "environment interpreter probe output is not one canonical identity object")
    has_algua = value.pop("algua")
    return value, has_algua


def verify_environment(
    root: Path, expected_interpreter: InterpreterIdentity, expected_inventory_digest: str,
) -> None:
    inventory = inventory_environment(root)
    if inventory.digest != expected_inventory_digest:
        raise EnvironmentIncompatible("installed environment inventory drifted")
    python = root / "bin/python"
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    import_root = root / "lib" / version / "site-packages"
    try:
        result = run_bounded(
            [str(python), "-I", "-S", "-c", _PROBE, str(import_root)], cwd=root,
            env=scrubbed_environment(python.parent), timeout=_PROBE_TIMEOUT_SECONDS,
            max_stdout=_PROBE_OUTPUT_BYTES, max_stderr=_PROBE_OUTPUT_BYTES,
        )
    except (OSError, subprocess.SubprocessError, OutputLimitExceeded) as exc:
        raise EnvironmentIncompatible(
            "published environment interpreter verification failed") from exc
    if result.returncode != 0:
        raise EnvironmentIncompatible("published environment interpreter verification failed")
    observed, has_algua = _parse_probe(result.stdout)
    if has_algua or observed != expected_interpreter.to_dict():
        raise EnvironmentIncompatible("published environment interpreter identity drifted")
