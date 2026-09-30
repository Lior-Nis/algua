"""Store an int written into a float-typed contract field as the equal float.

``ExecutionContract`` and ``CapacityLimit`` float fields reach two serializers: ``asdict`` feeds
``config_hash`` and keeps an int ``1``, while ``model_dump(mode="json")`` records a frozen
deployment's config and writes ``1.0``. Normalising at construction makes ``1`` and ``1.0`` one
value, so the strict frozen decoder's re-hash of the recorded config matches the descriptor
(Story 1.3c). A bool is never converted, so every bool rail (#344) still sees and rejects it; an int
too large for a float is refused as a ``ValueError`` like every other rail. Pure: stdlib only.
"""
from __future__ import annotations

from dataclasses import fields
from typing import Any


def store_ints_as_floats(instance: Any) -> None:
    """Call first in ``__post_init__``: every field annotated ``float`` holding an int (exactly an
    int, never a bool) is replaced, on the frozen instance, by the equal float."""
    for field in fields(instance):
        value = getattr(instance, field.name)
        if field.type in ("float", float) and type(value) is int:
            try:
                coerced = float(value)
            except OverflowError:
                raise ValueError(f"{field.name} must be a finite number") from None
            object.__setattr__(instance, field.name, coerced)
