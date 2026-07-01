"""CLI tests for `algua eval gate` (#347)."""
from __future__ import annotations

import contextlib
import io
import json

import pytest

from algua.cli.main import main


def _run(argv: list[str]) -> dict:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.suppress(SystemExit):
        main(argv)
    return json.loads(buf.getvalue())


def test_eval_gate_emits_headline_fields():
    d = _run(["eval", "gate", "--k", "4"])
    assert d["ok"] is True
    for key in ("false_promote_rate", "crash_rate", "pass_at_k", "pass_pow_k",
                "any_seed_correct", "all_seed_correct", "failure_histogram", "provenance"):
        assert key in d, key
    assert d["false_promote_rate"] == 0.0
    assert d["crash_rate"] == 0.0
    assert len(d["scenarios"]) == 4


def test_eval_gate_scenario_subsetting():
    d = _run(["eval", "gate", "--k", "3", "--scenario", "no_edge"])
    assert d["ok"] is True
    assert [s["scenario"] for s in d["scenarios"]] == ["no_edge"]


def test_eval_gate_multiple_scenarios_preserve_bank_order():
    d = _run(["eval", "gate", "--k", "2", "--scenario", "negative_edge",
              "--scenario", "obvious_edge"])
    assert [s["scenario"] for s in d["scenarios"]] == ["obvious_edge", "negative_edge"]


def test_eval_gate_unknown_scenario_fails_closed():
    d = _run(["eval", "gate", "--scenario", "bogus"])
    assert d["ok"] is False
    assert "bogus" in json.dumps(d)


def test_eval_gate_is_mounted():
    # `eval` group + `gate` command resolve through the composition root.
    d = _run(["eval", "gate", "--k", "2"])
    assert d["ok"] is True


@pytest.mark.parametrize("bad_k", ["0", "-1"])
def test_eval_gate_rejects_nonpositive_k(bad_k):
    # typer min=1 → parse error rendered as the JSON error envelope.
    d = _run(["eval", "gate", "--k", bad_k])
    assert d["ok"] is False
