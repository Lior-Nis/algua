"""One bounded-exponential-backoff retry loop (spec §4.3), unifying the two HTTP clones
(execution/alpaca_broker, data/providers/alpaca). The caller supplies error mapping and
retryability; safety-specific handling (e.g. the #394 redirect refusal) stays at the call
site — this module knows nothing about HTTP."""
from __future__ import annotations

import time
from collections.abc import Callable


class RetriesExhausted(Exception):
    """Every attempt failed with a retryable exception. Carries the last one so the caller
    can wrap it in its domain error (BrokerError / ProviderError) with full context."""

    def __init__(self, attempts: int, last_exception: BaseException) -> None:
        super().__init__(f"failed after {attempts} attempts: {last_exception}")
        self.attempts = attempts
        self.last_exception = last_exception


def call_with_backoff[T](
    send: Callable[[], T],
    *,
    attempts: int,
    backoff_base: float,
    retryable_exceptions: tuple[type[BaseException], ...],
    retry_result: Callable[[T], bool] = lambda _r: False,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call `send` up to `attempts` times, sleeping `backoff_base * 2**i` between attempts.

    A retryable exception on the final attempt raises RetriesExhausted. A retryable RESULT
    (e.g. an HTTP 429/5xx response) on the final attempt is RETURNED — the caller inspects
    the final response and decides (both Alpaca sites' documented semantics). A
    non-retryable exception propagates immediately."""
    last_exc: BaseException | None = None
    for attempt in range(attempts):
        try:
            result = send()
        except retryable_exceptions as exc:
            last_exc = exc
        else:
            if not retry_result(result) or attempt == attempts - 1:
                return result
        if attempt < attempts - 1:
            sleep(backoff_base * (2**attempt))
    assert last_exc is not None  # result path always returned above on the final attempt
    raise RetriesExhausted(attempts, last_exc)
