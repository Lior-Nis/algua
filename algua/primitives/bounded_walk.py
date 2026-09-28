"""Streaming directory traversal that fails closed and bounds every entry before retaining it.

`os.walk` lists a whole directory into memory before yielding it and queues every subdirectory it
discovers, so one directory with an enormous child list -- or a tree of empty directories that no
file bound ever counts -- grows memory before any bound can refuse it, and without `onerror` it
silently skips a directory it cannot list. This traversal pulls one entry at a time from
`os.scandir`, counts every file and every directory against its bound and checks the entry's
relative path length before yielding or descending (which also bounds depth and the number of
open directory handles), never follows links, and propagates every listing, iteration and
type-check error. Order is depth-first pre-order: a directory is yielded before its contents.
"""
from __future__ import annotations

import os
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path


class TraversalLimitExceeded(ValueError):
    """A tree exceeded a traversal bound; ``kind`` names the bound."""

    def __init__(self, kind: str) -> None:
        super().__init__(f"tree exceeds its {kind} bound")
        self.kind = kind


@dataclass(frozen=True, slots=True)
class TreeEntry:
    path: Path
    relative: str
    is_dir: bool
    is_symlink: bool


def bounded_walk(
    root: Path, *, max_files: int, max_directories: int, max_path_bytes: int,
) -> Generator[TreeEntry, None, None]:
    """Yield every entry below ``root``; non-directories (links included) count as files."""
    files = 0
    directories = 0
    stack = [(os.scandir(root), "")]
    try:
        while stack:
            listing, prefix = stack[-1]
            entry = next(listing, None)
            if entry is None:
                stack.pop()[0].close()
                continue
            relative = prefix + entry.name
            if len(os.fsencode(relative)) > max_path_bytes:
                raise TraversalLimitExceeded("path-length")
            is_symlink = entry.is_symlink()
            is_dir = not is_symlink and entry.is_dir(follow_symlinks=False)
            if is_dir:
                directories += 1
                if directories > max_directories:
                    raise TraversalLimitExceeded("directory-count")
            else:
                files += 1
                if files > max_files:
                    raise TraversalLimitExceeded("file-count")
            yield TreeEntry(Path(entry.path), relative, is_dir, is_symlink)
            if is_dir:
                stack.append((os.scandir(entry.path), relative + "/"))
    finally:
        for listing, _prefix in stack:
            listing.close()
