"""One atomic/durable write implementation (spec §4.2). Linux-only threat model: single local
filesystem — see fsync notes on each helper."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path


def fsync_file(path: Path) -> None:
    """fsync a regular file's data to stable storage. Linux-only: a read-only fd still
    flushes the inode's dirty data pages. (Threat model is a single local Linux FS;
    macOS/NFS fsync semantics differ and are out of scope.)"""
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
    try:
        os.fsync(fd)
    except OSError as exc:
        raise OSError(f"fsync_file({path}) failed: {exc}") from exc
    finally:
        os.close(fd)


def fsync_dir(path: Path) -> None:
    """fsync a directory so a rename/creation entry within it becomes durable. O_DIRECTORY
    makes a non-directory path fail loudly (ENOTDIR) instead of silently fsyncing the wrong
    object. Linux-only (see `fsync_file`)."""
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(fd)
    except OSError as exc:
        raise OSError(f"fsync_dir({path}) failed: {exc}") from exc
    finally:
        os.close(fd)


def fsync_parents(path: Path, *, stop_at: Path) -> None:
    """fsync every directory from `path.parent` up to and including `stop_at` (the durable
    store root). Covers ancestor directories newly created by `mkdir(parents=True)`: fsyncing
    only the leaf parent leaves a freshly-created intermediate dir's own name un-durable in
    *its* parent. `path` must be at or under `stop_at` (a `ValueError` is raised otherwise so a
    miswired call fails loudly instead of silently fsyncing up to the filesystem root)."""
    stop_at = stop_at.resolve()
    resolved = path.resolve()
    if resolved == stop_at:
        fsync_dir(stop_at)
        return
    if stop_at not in resolved.parents:
        raise ValueError(f"{path} is not under stop_at {stop_at}")
    current = resolved.parent
    while True:
        fsync_dir(current)
        if current == stop_at:
            break
        current = current.parent


def fsync_tree(root: Path) -> None:
    """Bottom-up fsync of every regular file, then every subdirectory, then `root` itself
    (`os.walk(topdown=False)`), so child durability precedes the parent's. For partitioned
    trees whose part-files pyarrow wrote without exposing a handle, we reopen+fsync each."""
    def _raise(exc: OSError) -> None:
        raise exc
    for dirpath, _dirnames, filenames in os.walk(root, topdown=False, onerror=_raise):
        d = Path(dirpath)
        for name in filenames:
            fsync_file(d / name)
        fsync_dir(d)


def write_bytes_atomic(data: bytes, dest: Path) -> None:
    """Write `data` to `dest` atomically via a same-dir temp + `os.replace` (#181): a reader never
    sees a partially written file even if the process dies mid-write. No fsync — this is ephemeral
    (a plotting input), not durable (cf. `write_bytes_durable`)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix=".emit-")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, dest)
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass


def write_text_atomic(text: str, dest: Path) -> None:
    """Write `text` to `dest` atomically via a same-dir temp + `os.replace` (#181): a reader never
    sees a partially written file even if the process dies mid-write. No fsync — not power-loss
    durable, because the current consumer (the Obsidian knowledge-vault writer, see
    `algua/knowledge/sync.py`) is regenerable curation, not the binding audit trail (cf.
    `write_bytes_durable`)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix=".emit-")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(text)
        os.replace(tmp, dest)
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass


def write_bytes_durable(data: bytes, dest: Path, *, durable_root: Path | None = None) -> None:
    """Atomically publish `data` at `dest` via a same-dir temp +
    `os.replace` (#158): a reader never observes a partially written file, and a same-id
    concurrent re-publish is benign (content-addressed => identical bytes; readers see the
    old or new inode, byte-identical). Power-loss durable (#184): the temp's bytes are
    fsynced before the rename, and the target's parent-dir chain (up to `durable_root`, when
    given) after."""
    target_path = dest
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_fd, temp_name = tempfile.mkstemp(dir=target_path.parent, prefix=".publish-")
    try:
        with os.fdopen(temp_fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temp_name, target_path)
        if durable_root is not None:
            fsync_parents(target_path, stop_at=durable_root)
        else:
            fsync_dir(target_path.parent)
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
