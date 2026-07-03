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
    audit_cmd,
    backtest_cmd,
    data_cmd,
    eval_cmd,
    factor_cmd,
    fleet_cmd,
    governance_cmd,
    idea_cmd,
    live_cmd,
    monitoring_cmd,
    negative_cmd,
    paper_cmd,
    registry_cmd,
    research_batch_cmd,
    research_cmd,
    strategy_cmd,
)
from algua.cli.app import app, emit
from algua.cli.errors import error_code

# Composition root: mount idea_app under research_app HERE (not inside idea_cmd) so no cli command
# module imports a sibling. Typer builds the command tree lazily at get_command(app) inside main(),
# so this only needs to run before that call. MUST stay after the `from algua.cli import (…)` block.
# research_batch_cmd (the `run-all` batch worker, #326) is likewise mounted here — it imports the
# reusable task bodies from backtest_cmd/research_cmd, so mounting at the composition root keeps
# those listed command modules free of any sibling import.
research_cmd.research_app.add_typer(idea_cmd.idea_app, name="idea")
research_cmd.research_app.add_typer(negative_cmd.log_app, name="log")
research_cmd.research_app.add_typer(research_batch_cmd.run_all_app, name="run-all")

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
        emit({"ok": False, "error": exc.format_message(), "code": "usage_error"})
        sys.exit(1)
    except _click_exc.Exit as exc:  # raised by typer.Exit / json_errors (a subclass of this)
        sys.exit(exc.exit_code)
    except _click_exc.Abort:
        emit({"ok": False, "error": "aborted", "code": "aborted"})
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001 - last-resort net: any exception escaping an UNDECORATED
        # command body, the Typer callback, or arg parsing still renders as the JSON error envelope
        # (never a raw traceback), so an agent's stdout parser is never broken (issue #337). Exit/
        # UsageError/Abort are caught above (no double-emit); SystemExit/KeyboardInterrupt are
        # BaseException and pass straight through. `code` reuses the shared registry (DRY).
        emit({"ok": False, "error": str(exc), "code": error_code(exc)})
        sys.exit(1)
    if isinstance(result, int) and result != 0:
        sys.exit(result)
    sys.exit(0)
