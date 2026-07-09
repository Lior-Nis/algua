from __future__ import annotations

import json
import logging

import pytest

from algua.operator.alerts import emit_alert


class _CaptureHandler(logging.Handler):
    """Collect raw LogRecords off the `algua` logger.

    caplog can't see these records: the observability logger sets propagate=False (once any test
    calls configure_logging()), so records never reach the root logger caplog attaches to. Attach
    directly to the `algua` logger instead — the same pattern as test_observability_wiring.py.
    """

    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def alert_records():
    handler = _CaptureHandler()
    logger = logging.getLogger("algua")
    logger.addHandler(handler)
    try:
        yield handler.records
    finally:
        logger.removeHandler(handler)


def test_runner_invoked_once_with_json_payload_returns_true() -> None:
    calls: list[tuple[str, str]] = []

    def fake(cmd: str, payload: str) -> int:
        calls.append((cmd, payload))
        return 0

    result = emit_alert("breach", {"strategy": "a"}, alert_cmd="notify", runner=fake)

    assert result is True
    assert len(calls) == 1
    cmd, payload = calls[0]
    assert cmd == "notify"
    decoded = json.loads(payload)
    assert decoded == {"kind": "breach", "strategy": "a"}


def test_nonzero_returncode_returns_false() -> None:
    def fake(cmd: str, payload: str) -> int:
        return 3

    assert emit_alert("breach", {"strategy": "a"}, alert_cmd="notify", runner=fake) is False


def test_no_alert_cmd_does_not_invoke_runner_returns_false() -> None:
    calls: list[tuple[str, str]] = []

    def fake(cmd: str, payload: str) -> int:
        calls.append((cmd, payload))
        return 0

    assert emit_alert("breach", {"strategy": "a"}, runner=fake) is False
    assert emit_alert("breach", {"strategy": "a"}, alert_cmd=None, runner=fake) is False
    assert emit_alert("breach", {"strategy": "a"}, alert_cmd="", runner=fake) is False
    assert calls == []


def test_runner_exception_does_not_propagate_returns_false() -> None:
    def boom(cmd: str, payload: str) -> int:
        raise RuntimeError("delivery exploded")

    # Must not raise.
    assert emit_alert("breach", {"strategy": "a"}, alert_cmd="notify", runner=boom) is False


def test_structured_operator_alert_record_emitted(alert_records) -> None:
    emit_alert("breach", {"strategy": "a"}, alert_cmd=None)

    matched = [r for r in alert_records if r.getMessage() == "operator_alert"]
    assert len(matched) == 1
    fields = getattr(matched[0], "fields", None)
    assert fields == {"kind": "breach", "strategy": "a"}


def test_delivery_failure_logs_dedicated_record(alert_records) -> None:
    def boom(cmd: str, payload: str) -> int:
        raise RuntimeError("nope")

    emit_alert("breach", {"strategy": "a"}, alert_cmd="notify", runner=boom)

    messages = [r.getMessage() for r in alert_records]
    assert "operator_alert" in messages
    assert "operator_alert_delivery_failed" in messages


def test_default_runner_executes_command_shell_false(tmp_path) -> None:
    # Exercise the real subprocess-backed runner end-to-end (no monkeypatched runner). The runner is
    # shell=False, so redirection (`cat > file`) would NOT work — the payload is piped on stdin to a
    # script that copies stdin to a file. This proves the argv path, not a shell.
    out = tmp_path / "out.txt"
    script = tmp_path / "sink.sh"
    script.write_text(f"#!/bin/sh\ncat > {out}\n")
    script.chmod(0o755)

    result = emit_alert("breach", {"strategy": "a"}, alert_cmd=str(script))

    assert result is True
    assert json.loads(out.read_text()) == {"kind": "breach", "strategy": "a"}


def test_default_runner_uses_shlex_split_not_a_shell(monkeypatch) -> None:
    # A shell would honour the `;`; shell=False + shlex.split treats it as literal argv, so
    # subprocess gets ["echo", "hi;", "rm", "-rf", "/"] and no shell metacharacter is interpreted.
    import subprocess as sp

    seen: dict = {}

    def _fake_run(argv, **kwargs):
        seen["argv"] = argv
        seen["shell"] = kwargs.get("shell")
        seen["timeout"] = kwargs.get("timeout")
        return sp.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(sp, "run", _fake_run)
    emit_alert("breach", {"strategy": "a"}, alert_cmd="echo 'hi;' rm -rf /")

    assert seen["shell"] is False
    assert seen["argv"] == ["echo", "hi;", "rm", "-rf", "/"]  # shlex-split argv, not a shell string
    assert seen["timeout"] is not None and seen["timeout"] > 0


def test_default_runner_timeout_is_swallowed(monkeypatch) -> None:
    import subprocess as sp

    def _boom(argv, **kwargs):
        raise sp.TimeoutExpired(cmd=argv, timeout=kwargs.get("timeout"))

    monkeypatch.setattr(sp, "run", _boom)
    # A hung alert command times out; emit_alert swallows it and returns False (never crashes).
    assert emit_alert("breach", {"strategy": "a"}, alert_cmd="sleep 999") is False


def test_default_runner_nonzero_exit_returns_false() -> None:
    assert emit_alert("breach", {"strategy": "a"}, alert_cmd="sh -c 'exit 7'") is False
