"""Exact Git-object export and source-path validation for frozen artifacts."""
from __future__ import annotations

import hashlib
import re
import subprocess
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from algua.registry.artifact_contract import (
    MAX_BUNDLE_BYTES,
    MAX_FILE_BYTES,
    ArtifactFile,
)

MAX_SOURCE_FILES = 10_000
MAX_PATH_BYTES = 1_024
_OID = re.compile(rb"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_MODES = {b"100644", b"100755"}
_BUILD_INPUTS = (".python-version", "pyproject.toml", "uv.lock")


class FrozenSourceError(ValueError):
    """The selected Git source cannot form a canonical frozen bundle."""


class FrozenAssetsUnsupported(FrozenSourceError):
    """The current frozen lane accepts source-only strategies."""


def require_source_only(model_handle: object | None) -> None:
    """Reject before inspecting any external model path or bytes."""
    if model_handle is not None:
        raise FrozenAssetsUnsupported("frozen planner assets are unsupported in this cycle")


@dataclass(frozen=True)
class GitTreeEntry:
    path: str
    mode: str
    oid: str


@dataclass(frozen=True)
class FrozenFile:
    path: str
    mode: str
    data: bytes

    @property
    def contract_entry(self) -> ArtifactFile:
        return ArtifactFile(
            path=self.path, mode=self.mode, size=len(self.data),
            sha256=hashlib.sha256(self.data).hexdigest(),
        )


def _git(repo_root: Path, *args: str, max_bytes: int) -> bytes:
    try:
        with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
            subprocess.run(
                ["git", *args], cwd=repo_root, check=True, stdout=stdout, stderr=stderr,
                timeout=60,
            )
            size = stdout.tell()
            if size > max_bytes:
                raise FrozenSourceError("Git output exceeds its protected bound")
            stdout.seek(0)
            return stdout.read(max_bytes + 1)
    except FrozenSourceError:
        raise
    except (OSError, subprocess.SubprocessError) as exc:
        raise FrozenSourceError("Git source could not be read") from exc


def _canonical_path(raw: bytes) -> str:
    if not raw or len(raw) > MAX_PATH_BYTES or b"\0" in raw or b"\\" in raw:
        raise FrozenSourceError("Git path is unsafe")
    try:
        path = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise FrozenSourceError("Git path is not valid UTF-8") from exc
    if unicodedata.normalize("NFC", path) != path:
        raise FrozenSourceError("Git path is not NFC-normalized")
    pure = PurePosixPath(path)
    segments = path.split("/")
    if (
        pure.is_absolute()
        or not segments
        or any(segment in {"", ".", ".."} for segment in segments)
        or re.match(r"^[A-Za-z]:", segments[0])
        or any(segment.endswith((".", " ")) for segment in segments)
    ):
        raise FrozenSourceError("Git path is unsafe")
    return path


def parse_tree(raw: bytes) -> tuple[GitTreeEntry, ...]:
    entries: list[GitTreeEntry] = []
    seen: set[str] = set()
    folded: set[str] = set()
    records = raw.split(b"\0")
    if records[-1] != b"":
        raise FrozenSourceError("Git tree output is not NUL terminated")
    for record in records[:-1]:
        try:
            header, raw_path = record.split(b"\t", 1)
            mode, kind, oid = header.split(b" ", 2)
        except ValueError as exc:
            raise FrozenSourceError("Git tree entry is malformed") from exc
        if mode not in _MODES or kind != b"blob" or _OID.fullmatch(oid) is None:
            raise FrozenSourceError("Git tree entry has unsupported type or mode")
        path = _canonical_path(raw_path)
        casefolded = path.casefold()
        if path in seen or casefolded in folded:
            raise FrozenSourceError("Git paths have a normalization or casefold collision")
        seen.add(path)
        folded.add(casefolded)
        entries.append(GitTreeEntry(path, mode.decode(), oid.decode()))
    paths = [entry.path.encode() for entry in entries]
    if paths != sorted(paths):
        raise FrozenSourceError("Git tree entries are not canonically ordered")
    return tuple(entries)


def _export(repo_root: Path, source_ref: str, paths: tuple[str, ...]) -> tuple[FrozenFile, ...]:
    tree_bound = (MAX_PATH_BYTES + 100) * (MAX_SOURCE_FILES + 1)
    raw = _git(
        repo_root, "ls-tree", "-r", "-z", "--full-tree", source_ref, "--", *paths,
        max_bytes=tree_bound,
    )
    entries = parse_tree(raw)
    if len(entries) > MAX_SOURCE_FILES:
        raise FrozenSourceError("Git source exceeds the file-count bound")
    files: list[FrozenFile] = []
    total = 0
    for entry in entries:
        size_raw = _git(repo_root, "cat-file", "-s", entry.oid, max_bytes=32)
        try:
            size = int(size_raw)
        except ValueError as exc:
            raise FrozenSourceError("Git blob size is invalid") from exc
        if size < 0 or size > MAX_FILE_BYTES:
            raise FrozenSourceError("Git blob exceeds the per-file bound")
        data = _git(repo_root, "cat-file", "blob", entry.oid, max_bytes=MAX_FILE_BYTES)
        if len(data) != size:
            raise FrozenSourceError("Git blob size changed while exporting")
        total += len(data)
        if total > MAX_BUNDLE_BYTES:
            raise FrozenSourceError("Git source exceeds the bundle-size bound")
        files.append(FrozenFile(entry.path, entry.mode, data))
    return tuple(files)


def export_source(repo_root: Path, source_ref: str) -> tuple[FrozenFile, ...]:
    files = _export(repo_root.resolve(), source_ref, ("algua",))
    if not files or any(not item.path.startswith("algua/") for item in files):
        raise FrozenSourceError("Git commit has no canonical algua source inventory")
    return files


def export_build_inputs(repo_root: Path, source_ref: str) -> tuple[FrozenFile, ...]:
    files = _export(repo_root.resolve(), source_ref, _BUILD_INPUTS)
    if tuple(item.path for item in files) != _BUILD_INPUTS:
        raise FrozenSourceError("Git commit is missing required environment build inputs")
    if any(item.mode != "100644" for item in files):
        raise FrozenSourceError("environment build inputs must be regular non-executable blobs")
    return files


def _generated_cache(path: str, tracked: set[str]) -> bool:
    pure = PurePosixPath(path)
    if pure.parent.name != "__pycache__" or pure.suffix not in {".pyc", ".pyo"}:
        return False
    module = pure.name.split(".", 1)[0]
    return str(pure.parent.parent / f"{module}.py") in tracked


def assert_clean_head(repo_root: Path) -> str:
    root = repo_root.resolve()
    head = _git(root, "rev-parse", "--verify", "HEAD^{commit}", max_bytes=129).decode().strip()
    if re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", head) is None:
        raise FrozenSourceError("Git HEAD is not a full commit object ID")
    listing_bound = (MAX_PATH_BYTES + 100) * (MAX_SOURCE_FILES + 1)
    if _git(
        root, "status", "--porcelain=v1", "-z", "--untracked-files=no",
        max_bytes=listing_bound,
    ):
        raise FrozenSourceError("working tree tracked files do not match HEAD")
    tracked_raw = _git(root, "ls-files", "-z", "--", "algua", max_bytes=listing_bound)
    tracked = {item.decode("utf-8") for item in tracked_raw.split(b"\0") if item}
    source_root = root / "algua"
    if not source_root.is_dir() or source_root.is_symlink():
        raise FrozenSourceError("working tree has no canonical algua source root")
    untracked: list[str] = []
    for item in source_root.rglob("*"):
        relative = item.relative_to(root).as_posix()
        if item.is_dir() and not item.is_symlink():
            if any(path.startswith(relative + "/") for path in tracked):
                continue
            descendants = [path for path in item.rglob("*") if path.is_file() or path.is_symlink()]
            if item.name == "__pycache__" and descendants and all(
                _generated_cache(path.relative_to(root).as_posix(), tracked)
                for path in descendants
            ):
                continue
            untracked.append(relative)
        elif (item.is_file() or item.is_symlink()) and relative not in tracked:
            if not _generated_cache(relative, tracked):
                untracked.append(relative)
    if untracked:
        raise FrozenSourceError("working tree contains untracked source/config shadowing")
    return head
