"""Directory traversal that fails closed instead of silently skipping a subtree.

`os.walk` ignores a directory it cannot list unless given `onerror`, so an inventory, seal or
verification walk could vouch for a tree it never saw. Every walk over an immutable artifact
tree uses this helper, which re-raises the first listing error and never follows links.
"""
from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path


def _raise(exc: OSError) -> None:
    raise exc


def strict_walk(root: Path, *, topdown: bool = True) -> Iterator[tuple[str, list[str], list[str]]]:
    """`os.walk` without link following that propagates every traversal error."""
    return os.walk(root, topdown=topdown, onerror=_raise, followlinks=False)
