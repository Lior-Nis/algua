"""Streaming directory traversal that fails closed and bounds every entry before retaining it.

`os.walk` lists a whole directory into memory before yielding it and queues every subdirectory it
discovers, so one directory with an enormous child list -- or a tree of empty directories that no
file bound ever counts -- grows memory before any bound can refuse it, and without `onerror` it
silently skips a directory it cannot list. This traversal pulls one entry at a time from
`os.scandir`, counts every file and every directory against its bound and checks the entry's
relative path length before yielding or descending (which also bounds depth and the number of
open directory handles), never follows links, and propagates every listing, iteration and
type-check error. Order is depth-first pre-order: a directory is yielded before its contents.

Production consumers walk through `scoped_walk`, which always closes the walk and keeps a
consumer's own error primary when closing also fails.
"""
from __future__ import annotations

import os
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class TraversalLimitExceeded(ValueError):
    """A tree exceeded a traversal bound; ``kind`` names the bound."""

    def __init__(self, kind: str) -> None:
        super().__init__(f"tree exceeds its {kind} bound")
        self.kind = kind


class WalkCleanupError(RuntimeError):
    """Closing a directory listing failed with an ordinary (non-interrupt) error.

    Raised from the walk in place of that error, which becomes its cause, so every consumer can
    translate a cleanup failure into its own typed error without also capturing listing or
    permission errors. A `GeneratorExit` from a close is reported the same way, because
    `generator.close()` would otherwise treat it as normal completion and silently discard it.
    Interrupts (`KeyboardInterrupt`, `SystemExit` and other non-`Exception` failures) are never
    wrapped.
    """


@dataclass(frozen=True, slots=True)
class TreeEntry:
    path: Path
    relative: str
    is_dir: bool
    is_symlink: bool


def _close(listing: Any) -> None:
    try:
        listing.close()
    except (Exception, GeneratorExit) as exc:
        raise WalkCleanupError("a directory listing could not be closed") from exc


def _close_all(stack: list[tuple[Any, str]]) -> BaseException | None:
    """Close every open listing, deepest first, whatever a close raises; return what to report.

    A failure of any kind is caught only so that every remaining listing is still closed. The
    report is the first (deepest) failure, except that an interrupt (a `BaseException` that is
    not an `Exception`, such as `KeyboardInterrupt` or `SystemExit`) is never dropped in favour
    of an ordinary failure. After every listing has been tried once, each listing whose close
    failed (possibly before releasing its handle) is retried exactly once, deepest first; a retry
    never displaces the first failure, but an interrupt it raises is still reported.
    """
    first: BaseException | None = None
    interrupt: BaseException | None = None
    failed: list[Any] = []
    while stack:
        listing, _prefix = stack.pop()
        try:
            _close(listing)
        except BaseException as exc:  # caught only to finish closing; reported below
            failed.append(listing)
            if first is None:
                first = exc
            if interrupt is None and not isinstance(exc, Exception):
                interrupt = exc
    for listing in failed:
        try:
            _close(listing)
        except BaseException as exc:  # a bounded retry; only an interrupt can change the report
            if interrupt is None and not isinstance(exc, Exception):
                interrupt = exc
    return interrupt if interrupt is not None else first


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
                # Unstack only after a successful close, so a failed close is retried in cleanup.
                _close(listing)
                stack.pop()
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
    except GeneratorExit:
        # An abandoned walk has no error of its own, so a cleanup failure is the one to report.
        failure = _close_all(stack)
        if failure is not None:
            raise failure from failure.__cause__  # hides only the abandonment signal
        raise
    except BaseException as active:
        # The active error stays primary over ordinary cleanup failures; every remaining handle
        # is still closed, and a cleanup interrupt is never swallowed by an ordinary error.
        failure = _close_all(stack)
        if isinstance(active, Exception) and failure is not None and not isinstance(
            failure, Exception,
        ):
            raise failure from active
        raise


@contextmanager
def scoped_walk(
    root: Path, *, max_files: int, max_directories: int, max_path_bytes: int,
) -> Iterator[Generator[TreeEntry, None, None]]:
    """A `bounded_walk` that is always closed when the consumer's block exits.

    If the block raised (a consumer's typed refusal, say), that error stays primary: closing the
    walk still closes every listing, and an ordinary failure while closing does not replace it;
    an interrupt while closing still propagates, caused by the block's error. If the block
    finished or stopped early, a failure while closing is reported, as for any abandoned walk.
    Errors raised by the traversal itself reach the block unchanged.
    """
    walk = bounded_walk(
        root, max_files=max_files, max_directories=max_directories,
        max_path_bytes=max_path_bytes,
    )
    try:
        yield walk
    except BaseException as active:
        try:
            walk.close()
        except BaseException as failure:  # every listing is closed; decide what to report
            if isinstance(active, Exception) and not isinstance(failure, Exception):
                raise failure from active
        raise
    walk.close()
