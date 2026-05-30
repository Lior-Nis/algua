from __future__ import annotations

import functools
from collections.abc import Callable

import typer

from algua.cli.app import emit


def json_errors(
    *error_types: type[Exception],
) -> Callable[[Callable[..., None]], Callable[..., None]]:
    handled = error_types or (ValueError, LookupError)

    def decorate(fn: Callable[..., None]) -> Callable[..., None]:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except handled as exc:
                emit({"ok": False, "error": str(exc)})
                raise typer.Exit(code=1) from exc

        return wrapper

    return decorate
