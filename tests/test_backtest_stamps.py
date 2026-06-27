from __future__ import annotations

from algua.backtest import stamps


def test_code_hash_includes_dirty_workspace_hash(monkeypatch, tmp_path):
    head = "abc123"
    untracked = tmp_path / "strategy.py"
    untracked.write_text("changed result\n", encoding="utf-8")
    monkeypatch.setattr(stamps, "_ROOT", tmp_path)

    def git_bytes(args: list[str]) -> bytes | None:
        if args == ["rev-parse", "HEAD"]:
            return f"{head}\n".encode()
        if args == ["status", "--porcelain=v1", "-z", "--untracked-files=all"]:
            return b" M algua/backtest/engine.py\0?? strategy.py\0"
        if args == ["diff", "HEAD", "--binary"]:
            return b"diff --git a/algua/backtest/engine.py b/algua/backtest/engine.py\n"
        if args == ["ls-files", "--others", "--exclude-standard", "-z"]:
            return b"strategy.py\0"
        raise AssertionError(args)

    monkeypatch.setattr(stamps, "_git_bytes", git_bytes)

    code_hash = stamps.runtime_stamps()["code_hash"]

    assert code_hash is not None
    assert code_hash.startswith(f"{head}-dirty-")
    assert code_hash != head


def test_code_hash_is_head_when_workspace_is_clean(monkeypatch):
    head = "abc123"

    def git_bytes(args: list[str]) -> bytes | None:
        if args == ["rev-parse", "HEAD"]:
            return f"{head}\n".encode()
        if args == ["status", "--porcelain=v1", "-z", "--untracked-files=all"]:
            return b""
        raise AssertionError(args)

    monkeypatch.setattr(stamps, "_git_bytes", git_bytes)

    assert stamps.runtime_stamps()["code_hash"] == head


def test_code_hash_none_when_git_status_unavailable(monkeypatch):
    """A git status timeout/error (None) must NOT be equated with a clean workspace (#256).

    rev-parse HEAD succeeds but status is unavailable -> the stamp degrades to None
    (unknown), never to the bare clean HEAD that would mislabel a dirty tree as clean.
    """
    head = "abc123"

    def git_bytes(args: list[str]) -> bytes | None:
        if args == ["rev-parse", "HEAD"]:
            return f"{head}\n".encode()
        if args == ["status", "--porcelain=v1", "-z", "--untracked-files=all"]:
            return None  # subprocess timed out / git unavailable
        raise AssertionError(args)

    monkeypatch.setattr(stamps, "_git_bytes", git_bytes)

    assert stamps.runtime_stamps()["code_hash"] is None
