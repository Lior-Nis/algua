"""Atomic rename that can never replace an existing destination.

`os.rename` silently replaces an empty destination directory (and a file with a file), so a
destination created after an existence check can be overwritten. Immutable publication instead
uses Linux `renameat2(RENAME_NOREPLACE)`: the kernel checks absence and renames in one step and
reports an occupied destination as `EEXIST`. When the primitive is missing or the filesystem does
not support the flag, publication fails closed; there is deliberately no fallback to a
replacement-capable rename.
"""
from __future__ import annotations

import ctypes
import errno
import os
import sys
from collections.abc import Callable
from pathlib import Path

_AT_FDCWD = -100
_RENAME_NOREPLACE = 1
_UNSUPPORTED = frozenset({errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP})


class NoReplaceUnsupported(OSError):
    """No atomic rename-without-replacement exists for this platform or filesystem."""


def _load_renameat2() -> Callable[[bytes, bytes], int] | None:
    """Return a call that renames without replacement and yields 0 or an errno, or None."""
    if not sys.platform.startswith("linux"):
        return None
    try:
        function = ctypes.CDLL(None, use_errno=True).renameat2
    except (AttributeError, OSError):
        return None
    function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
                         ctypes.c_uint]
    function.restype = ctypes.c_int

    def call(source: bytes, destination: bytes) -> int:
        if function(_AT_FDCWD, source, _AT_FDCWD, destination, _RENAME_NOREPLACE) == 0:
            return 0
        return ctypes.get_errno() or errno.EIO

    return call


def rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically rename ``source`` to an absent ``destination``.

    Raises `FileExistsError` (EEXIST) when anything already occupies ``destination`` and
    `NoReplaceUnsupported` when no atomic no-replace primitive is available; neither case
    modifies ``source`` or ``destination``.
    """
    call = _load_renameat2()
    if call is None:
        raise NoReplaceUnsupported(
            errno.ENOSYS, "atomic rename without replacement is unavailable")
    code = call(os.fsencode(source), os.fsencode(destination))
    if code == 0:
        return
    if code == errno.EEXIST:
        raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST))
    if code in _UNSUPPORTED:
        raise NoReplaceUnsupported(code, "atomic rename without replacement is unsupported")
    raise OSError(code, os.strerror(code))
