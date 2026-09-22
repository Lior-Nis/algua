"""`code_hash` must track BEHAVIOUR, not typography.

It hashes the strategy's first-party source closure. Hashing raw text meant a comment edit, a
reflow or a docstring rewrite invalidated every prior approval and reset every strategy's
forward-evidence clock -- against a gate that needs 250-500 observations under one unchanged
identity. A cosmetic edit cannot change a decision, so it must not move the identity.

The opposite error is worse: normalising away something that DOES change behaviour would let a
real code change slip past the live gate. The second half of this file is that direction.
"""
from __future__ import annotations

from algua.registry.approvals import _strip_cosmetics


def test_a_comment_does_not_change_the_normalized_source():
    assert _strip_cosmetics("x = 1  # why\n") == _strip_cosmetics("x = 1\n")


def test_reformatting_does_not_change_it():
    assert _strip_cosmetics("def f(a,b):\n    return a+b\n") == _strip_cosmetics(
        "def f(\n    a,\n    b,\n):\n    return a + b\n")


def test_a_docstring_rewrite_does_not_change_it():
    a = 'def f():\n    """One thing."""\n    return 1\n'
    b = 'def f():\n    """Something else entirely, at length."""\n    return 1\n'
    assert _strip_cosmetics(a) == _strip_cosmetics(b)


def test_a_module_docstring_rewrite_does_not_change_it():
    assert _strip_cosmetics('"""A."""\nx = 1\n') == _strip_cosmetics('"""B."""\nx = 1\n')


def test_a_docstring_only_function_still_parses():
    """Removing the docstring must not leave an empty body."""
    out = _strip_cosmetics('def f():\n    """Only a docstring."""\n')
    assert "pass" in out


def test_a_changed_literal_DOES_change_it():
    assert _strip_cosmetics("x = 1\n") != _strip_cosmetics("x = 2\n")


def test_a_changed_operator_DOES_change_it():
    assert _strip_cosmetics("y = a + b\n") != _strip_cosmetics("y = a - b\n")


def test_a_renamed_local_DOES_change_it():
    assert _strip_cosmetics("def f(a):\n    return a\n") != _strip_cosmetics(
        "def f(b):\n    return b\n")


def test_a_reordered_statement_DOES_change_it():
    assert _strip_cosmetics("a = 1\nb = 2\n") != _strip_cosmetics("b = 2\na = 1\n")


def test_unparseable_source_is_hashed_raw_rather_than_collapsing():
    """A module that cannot be parsed must not normalize to the empty string -- that would make
    every broken module share one identity."""
    broken = "def f(:\n"
    assert _strip_cosmetics(broken) == broken


def test_deeply_nested_source_falls_back_raw_instead_of_raising(monkeypatch):
    """A source string parseable by `ast.parse` can still blow the C recursion limit inside
    `ast.walk`/`ast.unparse` on deeply nested expressions -- how deep is stack-dependent, so this
    forces the failure directly rather than relying on a fragile literal depth. That must fall
    back to the raw source (same as a SyntaxError), not raise -- an uncaught RecursionError here
    would take down `compute_artifact_hashes`, and with it `fleet status` and the live gate."""
    import ast as ast_module

    def blow_up(_tree):
        raise RecursionError("maximum recursion depth exceeded")

    monkeypatch.setattr(ast_module, "unparse", blow_up)
    source = "x = 1\n"
    assert _strip_cosmetics(source) == source
