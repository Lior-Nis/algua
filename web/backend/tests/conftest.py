"""Shared fixtures: fake the subprocess layer — no real CLI is ever spawned."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from backend import algua_cli


@pytest.fixture(autouse=True)
def reset_cli_state() -> Any:
    """Fresh cache/locks/semaphore per test (each test runs its own event loop)."""
    algua_cli._cache.clear()
    algua_cli._locks.clear()
    algua_cli._semaphore = asyncio.Semaphore(algua_cli._MAX_CONCURRENT_SUBPROCESSES)
    yield
    algua_cli._cache.clear()
    algua_cli._locks.clear()


class FakeProcess:
    """Stands in for the asyncio subprocess: canned stdout, optional hang."""

    def __init__(self, stdout: bytes, returncode: int = 0, hang_s: float = 0.0) -> None:
        self._stdout = stdout
        self._hang_s = hang_s
        self.returncode = returncode
        self.pid = 4242
        self.terminated = False
        self.killed = False

    async def communicate(self) -> tuple[bytes, bytes]:
        if self._hang_s:
            await asyncio.sleep(self._hang_s)
        return self._stdout, b""

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True

    async def wait(self) -> int:
        return self.returncode


@pytest.fixture
def group_signals(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, int]]:
    """Record (pgid, signal) pairs sent via os.killpg — the group-kill contract."""
    sent: list[tuple[int, int]] = []

    def fake_killpg(pgid: int, sig: int) -> None:
        sent.append((pgid, int(sig)))

    monkeypatch.setattr(algua_cli.os, "killpg", fake_killpg)
    return sent


@pytest.fixture
def fake_cli(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Install a fake subprocess spawner; returns a call-recorder.

    Usage: ``calls = fake_cli(stdout, returncode=0, hang_s=0.0)``. The recorder
    exposes ``calls["count"]`` (spawn invocations) and ``calls["procs"]``.
    """

    def install(stdout: bytes | str, returncode: int = 0, hang_s: float = 0.0) -> dict[str, Any]:
        raw = stdout.encode() if isinstance(stdout, str) else stdout
        calls: dict[str, Any] = {"count": 0, "cmds": [], "procs": []}

        async def fake_spawn(*cmd: str, **kwargs: Any) -> FakeProcess:
            calls["count"] += 1
            calls["cmds"].append(cmd)
            proc = FakeProcess(raw, returncode=returncode, hang_s=hang_s)
            calls["procs"].append(proc)
            return proc

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_spawn)
        return calls

    return install
