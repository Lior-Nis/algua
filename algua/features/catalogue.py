# algua/features/catalogue.py
from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from algua.contracts.idea import DataCapability


class FactorKind(StrEnum):
    """Controlled, deliberately minimal factor categories (OTHER is the escape hatch). Extend as
    real factors demand, not speculatively."""

    MOMENTUM = "momentum"
    REVERSION = "reversion"
    VALUE = "value"
    SENTIMENT = "sentiment"
    VOLATILITY = "volatility"
    QUALITY = "quality"
    OTHER = "other"


class FactorNotFound(LookupError):
    pass


@dataclass(frozen=True)
class FactorSpec:
    """Catalogued metadata for one pure factor. Collection fields are tuples so a registered spec
    cannot be mutated in place (a frozen dataclass does not stop list mutation)."""

    name: str
    summary: str
    kind: FactorKind
    tags: tuple[str, ...]
    data_needs: tuple[DataCapability, ...]
    import_path: str
    module: str
    signature: str
    doc: str | None


_SPEC_ATTR = "__factor_spec__"


def _first_nonempty_line(doc: str | None) -> str | None:
    if not doc:
        return None
    for line in doc.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return None


def factor(
    *,
    name: str | None = None,
    summary: str | None = None,
    tags: Iterable[str] = (),
    kind: FactorKind = FactorKind.OTHER,
    data_needs: Iterable[DataCapability] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Annotate a pure factor function with discoverability metadata. PURE: it stamps a FactorSpec
    on the function as ``__factor_spec__`` and returns the function UNCHANGED (no wrapper) so
    call semantics, ``inspect.getsource`` and the live-gate ``code_hash`` see the real function.
    It mutates no module global at import time — discovery (``load_all_factors``) scans for the
    stamp. ``data_needs`` states the factor's INPUT requirement, not current platform availability.
    """

    def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        resolved_name = name or fn.__name__
        doc = inspect.getdoc(fn)
        resolved_summary = summary or _first_nonempty_line(doc)
        if not resolved_summary:
            raise ValueError(
                f"factor {resolved_name!r} needs a summary (pass summary= or add a docstring)"
            )
        spec = FactorSpec(
            name=resolved_name,
            summary=resolved_summary,
            kind=kind,
            tags=tuple(tags),
            data_needs=tuple(data_needs) if data_needs is not None else (DataCapability.OHLCV,),
            import_path=f"{fn.__module__}:{fn.__qualname__}",
            module=fn.__module__,
            signature=str(inspect.signature(fn)),
            doc=doc,
        )
        setattr(fn, _SPEC_ATTR, spec)
        return fn

    return decorate
