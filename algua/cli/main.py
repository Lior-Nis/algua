from __future__ import annotations

from typing import Any

import click

from algua.cli import (  # noqa: F401 - imports register subcommands
    backtest_cmd,
    data_cmd,
    registry_cmd,
    strategy_cmd,
)
from algua.cli.app import app, emit


def main(args: list[str] | None = None) -> None:
    try:
        result = app(args=args, standalone_mode=False)
        if isinstance(result, int) and result != 0:
            raise SystemExit(result)
    except click.ClickException as exc:
        _emit_click_error(exc)
    except click.exceptions.Exit as exc:
        raise SystemExit(exc.exit_code) from exc
    except Exception as exc:
        if hasattr(exc, "format_message") and hasattr(exc, "exit_code"):
            _emit_click_error(exc)
        raise


def _emit_click_error(exc: Any) -> None:
    emit({"ok": False, "error": exc.format_message()})
    raise SystemExit(exc.exit_code) from exc


__all__ = ["app", "main"]
