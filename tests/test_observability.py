from __future__ import annotations

import json
import logging
import sys

import pytest

from algua.observability import (
    CycleCounters,
    JsonFormatter,
    configure_logging,
    correlation_context,
    current_correlation_id,
    get_logger,
)


def _record(**kw) -> logging.LogRecord:
    base = dict(
        name="algua.x", level=logging.INFO, pathname=__file__, lineno=1,
        msg="hello", args=(), exc_info=None,
    )
    base.update(kw)
    return logging.LogRecord(**base)


def test_format_is_one_line_json_with_core_keys():
    out = JsonFormatter().format(_record())
    assert "\n" not in out
    d = json.loads(out)
    assert d["msg"] == "hello" and d["level"] == "INFO" and d["logger"] == "algua.x"
    assert "ts" in d


def test_fields_merged_but_core_keys_win():
    rec = _record()
    rec.fields = {"strategy": "alpha", "msg": "SPOOFED", "level": "SPOOFED"}
    d = json.loads(JsonFormatter().format(rec))
    assert d["strategy"] == "alpha"
    assert d["msg"] == "hello" and d["level"] == "INFO"  # core wins


def test_exc_info_rendered_to_strings_not_dropped():
    try:
        raise ValueError("boom")
    except ValueError:
        rec = _record(exc_info=sys.exc_info())
    d = json.loads(JsonFormatter().format(rec))
    assert d["exc_type"] == "ValueError" and d["exc_message"] == "boom"
    assert "ValueError" in d["stacktrace"]


def test_exc_fields_cannot_be_spoofed_by_caller_fields():
    try:
        raise ValueError("real")
    except ValueError:
        rec = _record(exc_info=sys.exc_info())
    rec.fields = {"exc_type": "Spoofed", "exc_message": "fake"}
    d = json.loads(JsonFormatter().format(rec))
    assert d["exc_type"] == "ValueError" and d["exc_message"] == "real"


def test_nonserializable_field_does_not_drop_record():
    rec = _record()
    rec.fields = {"obj": object()}
    d = json.loads(JsonFormatter().format(rec))  # must not raise
    assert d["msg"] == "hello"


def test_correlation_id_present_only_inside_context():
    rec = _record()
    assert "correlation_id" not in json.loads(JsonFormatter().format(rec))
    with correlation_context("CID123") as cid:
        assert cid == "CID123"
        rec2 = _record()
        assert json.loads(JsonFormatter().format(rec2))["correlation_id"] == "CID123"
    assert current_correlation_id() is None


def test_correlation_context_resets_on_exception():
    with pytest.raises(RuntimeError):
        with correlation_context("X"):
            raise RuntimeError("nope")
    assert current_correlation_id() is None


def test_correlation_context_generates_id_when_none():
    with correlation_context() as cid:
        assert isinstance(cid, str) and len(cid) > 0


def test_configure_logging_idempotent_and_stderr(capfd, monkeypatch):
    monkeypatch.setenv("ALGUA_LOG_LEVEL", "DEBUG")
    configure_logging()
    configure_logging()
    logger = logging.getLogger("algua")
    marked = [h for h in logger.handlers if getattr(h, "_algua_observability", False)]
    assert len(marked) == 1
    assert logger.level == logging.DEBUG
    assert logger.propagate is False
    get_logger("algua.test").info("ping", extra={"fields": {"k": 1}})
    out, err = capfd.readouterr()
    assert out == ""  # nothing on stdout
    assert json.loads(err.strip().splitlines()[-1])["msg"] == "ping"


def test_configure_logging_unknown_level_falls_back_to_info(monkeypatch):
    monkeypatch.setenv("ALGUA_LOG_LEVEL", "NOPE")
    configure_logging()
    assert logging.getLogger("algua").level == logging.INFO


def test_configure_logging_preserves_foreign_handler():
    logger = logging.getLogger("algua")
    foreign = logging.NullHandler()
    logger.addHandler(foreign)
    try:
        configure_logging()
        assert foreign in logger.handlers
    finally:
        logger.removeHandler(foreign)


def test_get_logger_namespaces_under_algua():
    assert get_logger("foo").name == "algua.foo"
    assert get_logger("algua.bar").name == "algua.bar"
    assert get_logger().name == "algua"


def test_cycle_counters_as_fields():
    c = CycleCounters()
    c.ticks += 2
    c.breaches += 1
    f = c.as_fields()
    assert f["ticks"] == 2 and f["breaches"] == 1 and f["flatten_failures"] == 0
    assert set(f) == {
        "ticks", "breaches", "flatten_failures", "reconcile_deferred", "reconcile_halted",
    }
