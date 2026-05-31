from __future__ import annotations

import sys

from typer.main import get_command

from algua.cli import (  # noqa: F401 - imports register subcommands
    backtest_cmd,
    data_cmd,
    registry_cmd,
    research_cmd,
    strategy_cmd,
)
from algua.cli.app import app, emit

try:  # Typer vendors its own Click fork; get_command() raises from that fork.
    from typer._click import exceptions as _click_exc
except ModuleNotFoundError:  # older Typer that defers to the real Click
    from click import exceptions as _click_exc  # type: ignore[no-redef]

__all__ = ["app", "main"]


def main() -> None:
    """Console-script entry point.

    Runs the Typer app with ``standalone_mode=False`` so Click argument-parse errors (bad option
    types, unknown options, missing arguments) are rendered as JSON ``{ok: false}`` instead of Rich
    usage text — honoring the platform-wide "every command emits JSON" contract. Command-body
    errors are already JSON-rendered by the ``json_errors`` decorator.
    """
    command = get_command(app)
    try:
        command(standalone_mode=False)
    except _click_exc.UsageError as exc:
        emit({"ok": False, "error": exc.format_message()})
        sys.exit(1)
    except _click_exc.Exit as exc:  # raised by typer.Exit / json_errors (a subclass of this)
        sys.exit(exc.exit_code)
    except _click_exc.Abort:
        emit({"ok": False, "error": "aborted"})
        sys.exit(1)
    sys.exit(0)
