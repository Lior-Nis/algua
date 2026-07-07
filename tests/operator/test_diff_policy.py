"""Tests for the CODEOWNERS-aware merge-back diff policy (#485, Task 2 — finding #3, C5, R5)."""

from __future__ import annotations

import pytest

from algua.operator.diff_policy import DiffEntry, evaluate_diff

_CODEOWNERS = (
    "# comment\n"
    "/algua/registry/store.py        @Lior-Nis\n"
    "/algua/contracts/lifecycle.py   @Lior-Nis\n"
    "/approvers/                     @Lior-Nis\n"
    "/algua/cli/paper_cmd.py         @Lior-Nis\n"
)


def _add(path: str, mode: str = "100644") -> DiffEntry:
    return DiffEntry(mode, "A", None, path)


def test_strategy_only_diff_passes() -> None:
    entries = [_add("algua/strategies/momentum/xsec.py"), _add("kb/reports/xsec.md")]
    assert evaluate_diff(entries, _CODEOWNERS).ok


def test_top_level_strategy_file_without_family_rejected() -> None:
    # No <family> segment — reject-by-default (only algua/strategies/<family>/**.py is allowed).
    res = evaluate_diff([_add("algua/strategies/loader.py")], _CODEOWNERS)
    assert not res.ok


def test_codeowners_protected_path_rejected() -> None:
    res = evaluate_diff([DiffEntry("100644", "M", None, "algua/registry/store.py")], _CODEOWNERS)
    assert not res.ok
    assert any("denylist" in r for _, r in res.rejected)


def test_codeowners_directory_prefix_rejected() -> None:
    res = evaluate_diff([_add("approvers/new_key.pub")], _CODEOWNERS)
    assert not res.ok


def test_static_extras_tests_config_operator_rejected() -> None:
    for path in ("tests/test_x.py", "pyproject.toml", "algua/operator/mergeback.py"):
        res = evaluate_diff([DiffEntry("100644", "M", None, path)], _CODEOWNERS)
        assert not res.ok, path


def test_symlink_under_strategies_rejected() -> None:
    res = evaluate_diff([DiffEntry("120000", "A", None, "algua/strategies/foo/link.py")],
                        _CODEOWNERS)
    assert not res.ok
    assert any("non-regular" in r for _, r in res.rejected)


def test_gitlink_rejected() -> None:
    res = evaluate_diff([DiffEntry("160000", "A", None, "algua/strategies/foo/sub")], _CODEOWNERS)
    assert not res.ok


def test_exec_bit_on_py_strategy_rejected() -> None:
    res = evaluate_diff([DiffEntry("100755", "A", None, "algua/strategies/foo/bar.py")],
                        _CODEOWNERS)
    assert not res.ok
    assert any("executable" in r for _, r in res.rejected)


def test_rename_of_denylisted_source_into_allowlisted_dest_rejected() -> None:
    # git mv algua/registry/store.py algua/strategies/foo/store.py — source is denylisted.
    res = evaluate_diff(
        [DiffEntry("100644", "R100", "algua/registry/store.py", "algua/strategies/foo/store.py")],
        _CODEOWNERS)
    assert not res.ok


def test_copy_of_protected_code_into_allowlisted_dest_rejected() -> None:
    res = evaluate_diff(
        [DiffEntry("100644", "C100", "algua/registry/store.py", "algua/strategies/foo/copy.py")],
        _CODEOWNERS)
    assert not res.ok


def test_deletion_of_allowlisted_path_rejected() -> None:
    res = evaluate_diff([DiffEntry("000000", "D", "algua/strategies/foo/bar.py",
                                   "algua/strategies/foo/bar.py")], _CODEOWNERS)
    assert not res.ok
    assert any("deletion" in r for _, r in res.rejected)


def test_case_folded_denied_path_rejected() -> None:
    # A case-variant of a denied path must still be denied (case-fold matching).
    res = evaluate_diff([DiffEntry("100644", "M", None, "ALGUA/Registry/Store.py")], _CODEOWNERS)
    assert not res.ok


def test_traversal_path_rejected() -> None:
    res = evaluate_diff([_add("algua/strategies/foo/../../registry/store.py")], _CODEOWNERS)
    assert not res.ok


def test_malformed_codeowners_fails_closed() -> None:
    with pytest.raises(ValueError, match="failing closed"):
        evaluate_diff([_add("algua/strategies/foo/bar.py")], "# only comments\n\n")
