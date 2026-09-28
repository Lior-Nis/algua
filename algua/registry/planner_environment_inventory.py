"""Strict, bounded inventory of frozen planner environments."""
from __future__ import annotations

import hashlib
import os
import re
import stat
import sys
from email.parser import HeaderParser
from pathlib import Path

from algua.primitives.bounded_walk import TraversalLimitExceeded, WalkCleanupError, scoped_walk
from algua.registry.artifact_contract import (
    MAX_PATH_BYTES,
    ArtifactFile,
    canonical_relative_path,
)
from algua.registry.environment_contract import (
    BASE_INTERPRETER,
    MAX_DISTRIBUTION_METADATA_BYTES,
    MAX_ENVIRONMENT_BYTES,
    MAX_ENVIRONMENT_DIRECTORIES,
    MAX_ENVIRONMENT_FILE_BYTES,
    MAX_ENVIRONMENT_FILES,
    InstalledDistribution,
    InstalledInventory,
    InterpreterLink,
)
from algua.registry.planner_environment_errors import EnvironmentIncompatible
from algua.registry.planner_environment_startup import (
    MAX_PYVENV_CFG_BYTES,
    PYVENV_CFG,
    require_isolated_site,
    require_pinned,
    startup_pin,
)

_CHUNK_SIZE = 1024 * 1024
_SEPARATOR_RUN = re.compile(r"[-_.]+")


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


def _read_bounded(path: Path, limit: int, label: str) -> bytes:
    """Read a small file once, never requesting more than one byte past its bound."""
    with path.open("rb") as handle:
        raw = handle.read(limit + 1)
    if len(raw) > limit:
        raise EnvironmentIncompatible(f"{label} exceeds its bound")
    return raw


def _decode(raw: bytes, label: str) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EnvironmentIncompatible(f"{label} is not UTF-8") from exc


def _small_text_bound(relative: str, path: Path) -> tuple[int, str] | None:
    """The explicit bound and label of a small text file whose bytes are both digested and
    parsed: it is size-checked against that bound before any read and read exactly once."""
    if relative == PYVENV_CFG:
        return MAX_PYVENV_CFG_BYTES, PYVENV_CFG
    if path.name == "METADATA" and path.parent.name.endswith(".dist-info"):
        return MAX_DISTRIBUTION_METADATA_BYTES, "installed distribution metadata"
    return None


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


def _canonical(relative: str) -> str:
    """The entry's canonical bounded relative path, refused before anything is read from it."""
    try:
        return canonical_relative_path(relative, "environment path")
    except ValueError as exc:
        raise EnvironmentIncompatible("environment path is not canonical and bounded") from exc


def inventory_environment(root: Path) -> InstalledInventory:
    """Inventory every entry; uv creates no empty directory, so any directory must be implied.

    The traversal streams and counts every file and directory against its protected bound before
    retaining it, so neither one huge directory nor empty-directory fanout can grow memory first.
    """
    if root.is_symlink() or not root.is_dir():
        raise EnvironmentIncompatible("environment root is not a real directory")
    files: list[ArtifactFile] = []
    links: list[InterpreterLink] = []
    distributions: list[InstalledDistribution] = []
    directories: set[str] = set()
    total = 0
    pyvenv: str | None = None
    try:
        with scoped_walk(
            root, max_files=MAX_ENVIRONMENT_FILES, max_directories=MAX_ENVIRONMENT_DIRECTORIES,
            max_path_bytes=MAX_PATH_BYTES,
        ) as tree:
            for entry in tree:
                # Every entry, directories included, is canonical before it is retained or used;
                # the walk descends only after this check, so a malformed directory's subtree is
                # never listed.
                relative = _canonical(entry.relative)
                if entry.is_dir:
                    directories.add(relative)
                    continue
                path = entry.path
                if entry.is_symlink:
                    try:
                        if path.is_dir():
                            raise EnvironmentIncompatible(
                                "environment contains an unexpected symlink")
                        links.append(_interpreter_link(root, path))
                    except (OSError, RuntimeError) as exc:  # dangling, looping or unreadable
                        raise EnvironmentIncompatible(
                            "environment interpreter link is dangling or unreadable") from exc
                    continue
                info = path.lstat()
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                    raise EnvironmentIncompatible(
                        "environment contains a non-regular or hardlinked file"
                    )
                if path.suffix in {".pyc", ".pyo"} or "__pycache__" in path.parts:
                    raise EnvironmentIncompatible("environment contains generated bytecode")
                pinned = startup_pin(relative)
                bound = _small_text_bound(relative, path)
                if bound is not None and info.st_size > bound[0]:
                    raise EnvironmentIncompatible(f"{bound[1]} exceeds its bound")
                if info.st_size > MAX_ENVIRONMENT_FILE_BYTES:
                    raise EnvironmentIncompatible("environment file exceeds the per-file bound")
                if total + info.st_size > MAX_ENVIRONMENT_BYTES:
                    raise EnvironmentIncompatible("environment exceeds the aggregate size bound")
                content: bytes | None = None
                if bound is None:
                    size, digest = _file_digest(path, MAX_ENVIRONMENT_FILE_BYTES)
                else:
                    content = _read_bounded(path, *bound)
                    size, digest = len(content), hashlib.sha256(content).hexdigest()
                require_pinned(relative, pinned, digest)
                total += size
                if total > MAX_ENVIRONMENT_BYTES:
                    raise EnvironmentIncompatible("environment exceeds the aggregate size bound")
                mode = "100755" if info.st_mode & stat.S_IXUSR else "100644"
                files.append(ArtifactFile(relative, mode, size, digest))
                if bound is not None and content is not None:
                    text = _decode(content, bound[1])
                    if relative == PYVENV_CFG:
                        pyvenv = text
                    else:
                        distributions.append(_metadata_identity(text))
    except TraversalLimitExceeded as exc:
        raise EnvironmentIncompatible(f"environment exceeds the {exc.kind} bound") from exc
    except WalkCleanupError as exc:
        raise EnvironmentIncompatible(
            "an environment directory listing could not be closed") from exc
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
    if pyvenv is None:
        raise EnvironmentIncompatible("environment has no root pyvenv.cfg")
    require_isolated_site(pyvenv)
    files.sort(key=lambda item: item.path.encode())
    links.sort(key=lambda item: item.path.encode())
    distributions.sort(key=lambda item: (item.name, item.version))
    return InstalledInventory(tuple(distributions), tuple(files), tuple(links))
