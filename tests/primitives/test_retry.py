"""algua.primitives.retry — one bounded-exponential-backoff loop (spec §4.3)."""
from __future__ import annotations

import pytest

from algua.primitives.retry import RetriesExhausted, call_with_backoff


class _Boom(Exception):
    pass


def test_returns_first_success_no_sleep():
    sleeps: list[float] = []
    result = call_with_backoff(
        lambda: "ok", attempts=3, backoff_base=0.5,
        retryable_exceptions=(_Boom,), sleep=sleeps.append,
    )
    assert result == "ok"
    assert sleeps == []


def test_retries_exceptions_with_exponential_schedule():
    sleeps: list[float] = []
    calls = {"n": 0}

    def send():
        calls["n"] += 1
        if calls["n"] < 3:
            raise _Boom(f"attempt {calls['n']}")
        return "ok"

    assert call_with_backoff(
        send, attempts=3, backoff_base=0.5,
        retryable_exceptions=(_Boom,), sleep=sleeps.append,
    ) == "ok"
    assert sleeps == [0.5, 1.0]  # base * 2**0, base * 2**1


def test_exhausted_exceptions_raise_with_last_exception():
    with pytest.raises(RetriesExhausted) as exc_info:
        call_with_backoff(
            lambda: (_ for _ in ()).throw(_Boom("always")), attempts=2, backoff_base=0.0,
            retryable_exceptions=(_Boom,), sleep=lambda _s: None,
        )
    assert exc_info.value.attempts == 2
    assert isinstance(exc_info.value.last_exception, _Boom)


def test_retryable_result_returned_on_last_attempt():
    results = iter([503, 503, 503])
    out = call_with_backoff(
        lambda: next(results), attempts=3, backoff_base=0.0,
        retryable_exceptions=(_Boom,), retry_result=lambda r: r == 503,
        sleep=lambda _s: None,
    )
    assert out == 503  # final response returned for caller inspection, not raised


def test_non_retryable_exception_propagates_immediately():
    with pytest.raises(ValueError):
        call_with_backoff(
            lambda: (_ for _ in ()).throw(ValueError("no")), attempts=3, backoff_base=0.0,
            retryable_exceptions=(_Boom,), sleep=lambda _s: None,
        )
