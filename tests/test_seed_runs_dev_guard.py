"""The `scripts/seed_runs_dev.py` write guard.

The seed script drives `reserve_holdout`/`record_holdout_returns` against whatever `--db` names.
Pointed at the operator's real registry that BURNS single-use holdout reservations — the one
resource in this system that cannot be recreated. Two review rounds shipped a guard that did not
hold because NOTHING covered it: the guard was a denylist of cwd-derived guesses
(`Settings.db_path`, `Path.cwd()/'data'/'algua.db'`, the script-relative path), and run from a
git worktree all three collapse onto that worktree's own `data/algua.db` while
`--db ~/Projects/algua/data/algua.db` matched none of them.

The guard is now the positive rule "the target must not resolve inside any git working tree", and
this module is its test. Nothing here opens, creates, or writes ANY database: every case asserts
the refusal happens before `connect()`, and the one accepting case stops at the `--yes` gate.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "seed_runs_dev.py"


def _load_script():
    """Import `scripts/seed_runs_dev.py` by path — `scripts/` is a bare directory of entrypoints,
    not an importable package, so there is no module path to import it under."""
    spec = importlib.util.spec_from_file_location("seed_runs_dev_under_test", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


seed_runs_dev = _load_script()


def _make_checkout(root: Path) -> Path:
    """A directory that looks like an ordinary git CHECKOUT: `.git` is a directory."""
    (root / ".git").mkdir(parents=True)
    (root / "data").mkdir(parents=True)
    return root


def _make_worktree(root: Path, gitdir: Path) -> Path:
    """A directory that looks like a linked git WORKTREE: `.git` is a FILE holding `gitdir: ...`.
    This is the case a `.git`-is-a-directory check waves straight through, and worktrees are this
    repo's dominant workflow — so it gets its own fixture rather than riding on the checkout one."""
    root.mkdir(parents=True, exist_ok=True)
    (root / ".git").write_text(f"gitdir: {gitdir}\n", encoding="utf-8")
    (root / "data").mkdir(parents=True, exist_ok=True)
    return root


# ---------------------------------------------------------------------------------------------
# `_enclosing_git_root` — the rule itself
# ---------------------------------------------------------------------------------------------


def test_enclosing_git_root_finds_a_checkout_dot_git_directory(tmp_path: Path) -> None:
    checkout = _make_checkout(tmp_path / "algua")
    assert seed_runs_dev._enclosing_git_root(checkout / "data") == checkout


def test_enclosing_git_root_finds_a_worktree_dot_git_file(tmp_path: Path) -> None:
    """A worktree's `.git` is a FILE, not a directory."""
    worktree = _make_worktree(tmp_path / "wt", tmp_path / "algua" / ".git" / "worktrees" / "wt")
    assert (worktree / ".git").is_file()
    assert seed_runs_dev._enclosing_git_root(worktree / "data") == worktree


def test_enclosing_git_root_walks_up_through_nested_directories(tmp_path: Path) -> None:
    checkout = _make_checkout(tmp_path / "algua")
    nested = checkout / "a" / "b" / "c"
    nested.mkdir(parents=True)
    assert seed_runs_dev._enclosing_git_root(nested) == checkout


def test_enclosing_git_root_is_none_outside_any_repo(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    assert seed_runs_dev._enclosing_git_root(scratch) is None


# ---------------------------------------------------------------------------------------------
# `main()` — the refusals, all of them before anything touches disk
# ---------------------------------------------------------------------------------------------


def test_main_refuses_a_target_inside_a_checkouts_data_dir(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    checkout = _make_checkout(tmp_path / "algua")
    target = checkout / "data" / "algua.db"

    assert seed_runs_dev.main(["--db", str(target), "--yes"]) == 1

    err = capsys.readouterr().err
    assert "REFUSING" in err
    assert "git working tree" in err
    assert str(checkout) in err  # the message names the repo root it found
    assert not target.exists()  # `connect()` was never reached — it would have created the file


def test_main_refuses_a_target_inside_a_worktrees_data_dir(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    worktree = _make_worktree(tmp_path / "wt", tmp_path / "algua" / ".git" / "worktrees" / "wt")
    target = worktree / "data" / "algua.db"

    assert seed_runs_dev.main(["--db", str(target), "--yes"]) == 1

    err = capsys.readouterr().err
    assert "REFUSING" in err
    assert str(worktree) in err
    assert not target.exists()


def test_main_refuses_another_checkouts_registry_while_cwd_is_a_worktree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """THE regression. This is the exact invocation the previous guard let through.

    Stood in a worktree, with no `ALGUA_DB_PATH` set, all three denylist candidates resolve to
    `<worktree>/data/algua.db` — so the operator's real registry in a DIFFERENT checkout matched
    none of them and the script proceeded to burn holdouts in it. The candidates still collapse
    exactly the same way here (asserted below, so this test keeps its teeth if the defaults ever
    change); the git-working-tree rule is what refuses."""
    real_checkout = _make_checkout(tmp_path / "algua")
    worktree = _make_worktree(
        tmp_path / "algua" / ".claude" / "worktrees" / "slice3",
        tmp_path / "algua" / ".git" / "worktrees" / "slice3",
    )
    monkeypatch.delenv("ALGUA_DB_PATH", raising=False)
    monkeypatch.chdir(worktree)
    real_registry = real_checkout / "data" / "algua.db"

    # Precondition: the belt-and-braces denylist genuinely does NOT cover this target.
    assert real_registry.resolve() not in seed_runs_dev._real_registry_candidates()

    assert seed_runs_dev.main(["--db", str(real_registry), "--yes"]) == 1

    err = capsys.readouterr().err
    assert "REFUSING" in err
    assert "git working tree" in err
    assert not real_registry.exists()


def test_main_still_refuses_a_configured_registry_outside_any_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The belt-and-braces layer's own job: a registry configured to live outside every checkout
    is invisible to the git rule, and the `Settings.db_path` candidate is what catches it."""
    configured = tmp_path / "elsewhere" / "algua.db"
    configured.parent.mkdir(parents=True)
    monkeypatch.setenv("ALGUA_DB_PATH", str(configured))
    assert seed_runs_dev._enclosing_git_root(configured.parent) is None  # git rule can't see it

    assert seed_runs_dev.main(["--db", str(configured), "--yes"]) == 1

    err = capsys.readouterr().err
    assert "REFUSING" in err
    assert "real-registry candidate" in err
    assert not configured.exists()


def test_main_accepts_a_scratch_target_outside_any_repo(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The guard must not be a brick wall: a genuine scratch path clears it.

    Stopped at the `--yes` gate rather than allowed to seed, so this test never opens a database
    — reaching that gate IS the proof the git rule and the denylist both passed the target."""
    assert seed_runs_dev._enclosing_git_root(tmp_path) is None
    target = tmp_path / "slice3-dev.db"

    assert seed_runs_dev.main(["--db", str(target)]) == 1  # no --yes

    err = capsys.readouterr().err
    assert "Refusing to write without --yes" in err
    assert "REFUSING" not in err  # not the guard — the guard passed
    assert not target.exists()
