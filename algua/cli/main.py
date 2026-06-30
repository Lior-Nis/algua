from __future__ import annotations

import sys

# Typer vendors a private Click fork (typer._click); its exception classes do NOT inherit from
# public click.exceptions, so we must import them from the private module.  We pin Typer tightly
# (see pyproject.toml) so this coupling is version-stable.  The smoke tests in
# tests/test_cli_main.py (test_bad_option_type_renders_json, test_unknown_option_renders_json)
# act as the contract-break detector if the Typer version is ever bumped.
from typer._click import exceptions as _click_exc
from typer.main import get_command

from algua.cli import (  # noqa: F401 - imports register subcommands
    backtest_cmd,
    data_cmd,
    factor_cmd,
    idea_cmd,
    live_cmd,
    paper_cmd,
    registry_cmd,
    research_cmd,
    strategy_cmd,
)
from algua.cli.app import app, emit

# Composition root: mount idea_app under research_app HERE (not inside idea_cmd) so no cli command
# module imports a sibling. Typer builds the command tree lazily at get_command(app) inside main(),
# so this only needs to run before that call. MUST stay after the `from algua.cli import (…)` block.
research_cmd.research_app.add_typer(idea_cmd.idea_app, name="idea")

__all__ = ["app", "main"]


def main(args: list[str] | None = None) -> None:
    """Console-script entry point.

    Runs the Typer app with ``standalone_mode=False`` so Click argument-parse errors (bad option
    types, unknown options, missing arguments) are rendered as JSON ``{ok: false}`` instead of Rich
    usage text. Command-body errors are already JSON-rendered by the ``json_errors`` decorator.
    """
    from algua.observability import configure_logging

    configure_logging()
    command = get_command(app)
    try:
        result = command.main(args=args, prog_name="algua", standalone_mode=False)
    except _click_exc.UsageError as exc:
        emit({"ok": False, "error": exc.format_message()})
        sys.exit(1)
    except _click_exc.Exit as exc:  # raised by typer.Exit / json_errors (a subclass of this)
        sys.exit(exc.exit_code)
    except _click_exc.Abort:
        emit({"ok": False, "error": "aborted"})
        sys.exit(1)
    if isinstance(result, int) and result != 0:
        sys.exit(result)
    sys.exit(0)
