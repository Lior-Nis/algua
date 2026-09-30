"""Exact Git-object export and source-path validation for frozen artifacts."""
from __future__ import annotations

import hashlib
import os
import re
import selectors
import subprocess
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from algua.registry.artifact_contract import (
    MAX_BUNDLE_BYTES,
    MAX_FILE_BYTES,
    MAX_PATH_BYTES,
    MAX_SOURCE_FILES,
    ArtifactFile,
    canonical_relative_path,
)

_GIT_TIMEOUT_SECONDS = 60
_READ_CHUNK = 64 * 1024
_OID = re.compile(rb"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_MODES = {b"100644", b"100755"}
_BUILD_INPUTS = (".python-version", "pyproject.toml", "uv.lock")
# A well-formed ``__pycache__`` entry, ``<module>.<cache_tag>[.opt-<N>].pyc`` (PEP 3147/488). Python
# only loads one for an existing source, so a cache whose source is gone (a deleted module, a test's
# temporary module) cannot shadow anything either.
_CACHE_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\.[a-z]+-[0-9]+(?:\.opt-[0-9]+)?\.pyc")


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


def _stdout_chunks(process: subprocess.Popen[bytes], deadline: float) -> Iterator[bytes]:
    """Stream stdout chunk by chunk until EOF, failing once the deadline passes."""
    assert process.stdout is not None
    with selectors.DefaultSelector() as selector:
        selector.register(process.stdout, selectors.EVENT_READ)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(process.args, _GIT_TIMEOUT_SECONDS)
            if not selector.select(remaining):
                continue
            chunk = os.read(process.stdout.fileno(), _READ_CHUNK)
            if not chunk:
                return
            yield chunk


def _collect_bounded(chunks: Iterator[bytes], max_bytes: int) -> bytes:
    """Buffer output, stopping the moment it exceeds ``max_bytes``."""
    collected: list[bytes] = []
    received = 0
    for chunk in chunks:
        received += len(chunk)
        if received > max_bytes:
            raise FrozenSourceError("Git output exceeds its protected bound")
        collected.append(chunk)
    return b"".join(collected)


def _git_consume[T](
    repo_root: Path, args: tuple[str, ...], consume: Callable[[Iterator[bytes]], T],
) -> T:
    """Run Git, hand its streamed stdout to ``consume`` and kill it on any consumer failure."""
    deadline = time.monotonic() + _GIT_TIMEOUT_SECONDS
    try:
        with subprocess.Popen(
            ["git", *args], cwd=repo_root, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        ) as process:
            try:
                result = consume(_stdout_chunks(process, deadline))
                returncode = process.wait(timeout=max(deadline - time.monotonic(), 0))
            except BaseException:
                process.kill()
                process.wait()
                raise
    except FrozenSourceError:
        raise
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise FrozenSourceError("Git source could not be read") from exc
    if returncode != 0:
        raise FrozenSourceError("Git source could not be read")
    return result


def _git(repo_root: Path, *args: str, max_bytes: int) -> bytes:
    return _git_consume(repo_root, args, lambda chunks: _collect_bounded(chunks, max_bytes))


def _canonical_path(raw: bytes) -> str:
    try:
        path = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise FrozenSourceError("Git path is not valid UTF-8") from exc
    try:
        return canonical_relative_path(path, "Git path")
    except ValueError as exc:
        raise FrozenSourceError(str(exc)) from exc


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


def _untracked_kind(item: Path, relative: str) -> str:
    """The class of an offending path, named in the refusal."""
    pure = PurePosixPath(relative)
    if item.is_symlink() or item.is_dir():
        return "untracked symlink" if item.is_symlink() else "untracked directory"
    if pure.suffix != ".py" and pure.parent.name == "__pycache__":
        return "malformed bytecode cache"
    return {".py": "untracked Python source", ".pyc": "sourceless bytecode",
            ".pyo": "sourceless bytecode"}.get(pure.suffix, "untracked file")


def _admit_index_record(record: bytes, tracked: set[str]) -> None:
    if not record.startswith(b"H "):
        raise FrozenSourceError("Git index flags hide tracked working-tree drift")
    path = record[2:]
    if not path.startswith(b"algua/"):
        return
    if len(path) > MAX_PATH_BYTES:
        raise FrozenSourceError("Git index source path exceeds its path bound")
    try:
        tracked.add(path.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise FrozenSourceError("Git index path is not valid UTF-8") from exc
    if len(tracked) > MAX_SOURCE_FILES:
        raise FrozenSourceError("Git index source exceeds the file-count bound")


def _scan_index_records(chunks: Iterable[bytes]) -> set[str]:
    """Return tracked ``algua/`` paths from streamed ``git ls-files -z -v`` output.

    ``-v`` tags a plain cached entry ``H``; assume-unchanged entries are lowercase and
    skip-worktree entries ``S``. Either flag hides working-tree drift from the status check, so any
    tag other than ``H`` ANYWHERE in the repository index fails closed. The whole index is
    inspected without buffering it or imposing the source aggregate bound: only a bounded prefix of
    the current record (its tag plus one over-long source path) is ever retained.
    """
    limit = len(b"H ") + MAX_PATH_BYTES + 1
    tracked: set[str] = set()
    record = bytearray()
    for chunk in chunks:
        start = 0
        while (end := chunk.find(b"\0", start)) >= 0:
            record += chunk[start:min(end, start + max(limit - len(record), 0))]
            _admit_index_record(bytes(record), tracked)
            record.clear()
            start = end + 1
        record += chunk[start:start + max(limit - len(record), 0)]
    if record:
        raise FrozenSourceError("Git index listing is not NUL terminated")
    return tracked


def assert_clean_head(repo_root: Path) -> str:
    root = repo_root.resolve()
    head = _git(root, "rev-parse", "--verify", "HEAD^{commit}", max_bytes=129).decode().strip()
    if re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", head) is None:
        raise FrozenSourceError("Git HEAD is not a full commit object ID")
    listing_bound = (MAX_PATH_BYTES + 100) * (MAX_SOURCE_FILES + 1)
    tracked = _git_consume(root, ("ls-files", "-z", "-v"), _scan_index_records)
    if _git(
        root, "status", "--porcelain=v1", "-z", "--untracked-files=no",
        max_bytes=listing_bound,
    ):
        raise FrozenSourceError("working tree tracked files do not match HEAD")
    source_root = root / "algua"
    if not source_root.is_dir() or source_root.is_symlink():
        raise FrozenSourceError("working tree has no canonical algua source root")
    untracked: list[tuple[str, str]] = []
    for item in source_root.rglob("*"):
        relative = item.relative_to(root).as_posix()
        if item.is_dir() and not item.is_symlink():
            # Judged by its files; one holding none imports as a namespace HEAD does not have.
            if any(path.startswith(relative + "/") for path in tracked) or any(
                path.is_file() or path.is_symlink() for path in item.rglob("*")
            ):
                continue
            untracked.append((relative, _untracked_kind(item, relative)))
        elif (item.is_file() or item.is_symlink()) and relative not in tracked:
            if item.is_symlink() or item.parent.name != "__pycache__" \
                    or _CACHE_NAME.fullmatch(item.name) is None:
                untracked.append((relative, _untracked_kind(item, relative)))
    if untracked:
        path, kind = min(untracked)
        raise FrozenSourceError(f"working tree contains untracked source/config shadowing: "
                                f"{kind} {path!r} (1 of {len(untracked)})")
    return head
