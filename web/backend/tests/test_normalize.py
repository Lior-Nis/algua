"""Envelope-SHAPE classification of CLI stdout (never a real subprocess)."""

from __future__ import annotations

import json
from typing import Any

import pytest
from backend.algua_cli import CliError, run_cli


async def test_error_envelope_raises_cli_error(fake_cli: Any) -> None:
    fake_cli(json.dumps({"ok": False, "error": "no such strategy", "code": "not_found"}))
    with pytest.raises(CliError) as excinfo:
        await run_cli("registry", "show", "nope", ttl_s=10.0)
    assert excinfo.value.code == "not_found"
    assert excinfo.value.error == "no such strategy"


async def test_fleet_health_alerting_payload_is_data_not_error(fake_cli: Any) -> None:
    # fleet health exits 1 with ok: false while alerting BY DESIGN — that dict has
    # no "error"/"code" keys, so it is DATA, passed through with inner ok preserved.
    alerting = {
        "ok": False,
        "summary": {"alerting": 1},
        "rows": [{"strategy": "s1", "verdict": "stale"}],
        "alerting": ["s1"],
        "global_halt": False,
    }
    fake_cli(json.dumps(alerting), returncode=1)
    result = await run_cli("fleet", "health", ttl_s=10.0)
    assert result["ok"] is True
    assert result["stale"] is False
    assert result["data"] == alerting
    assert result["data"]["ok"] is False  # inner ok preserved verbatim


async def test_bare_list_is_wrapped(fake_cli: Any) -> None:
    fake_cli(json.dumps([{"name": "s1"}, {"name": "s2"}]))
    result = await run_cli("registry", "list", ttl_s=10.0)
    assert result["data"] == {"data": [{"name": "s1"}, {"name": "s2"}]}


async def test_ok_true_dict_is_data(fake_cli: Any) -> None:
    fake_cli(json.dumps({"ok": True, "version": "0.0.1"}))
    result = await run_cli("version", ttl_s=10.0)
    assert result["ok"] is True
    assert result["data"] == {"ok": True, "version": "0.0.1"}


async def test_non_json_stdout_is_bad_output(fake_cli: Any) -> None:
    fake_cli("Traceback (most recent call last):\n  boom\n")
    with pytest.raises(CliError) as excinfo:
        await run_cli("doctor", ttl_s=10.0)
    assert excinfo.value.code == "bad_output"
    assert "Traceback" in excinfo.value.error
    assert len(excinfo.value.error) <= 200


async def test_exit_code_1_with_valid_json_still_parsed(fake_cli: Any) -> None:
    fake_cli(json.dumps({"ok": True, "checks": []}), returncode=1)
    result = await run_cli("doctor", ttl_s=10.0)
    assert result["ok"] is True
    assert result["data"] == {"ok": True, "checks": []}
