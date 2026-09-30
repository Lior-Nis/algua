"""The canonical JSON encoder pins the bytes every frozen artifact digest is taken over."""
from __future__ import annotations

import pytest

from algua.contracts.canonical import (
    FROZEN_WIRE,
    FROZEN_WIRE_NAME,
    FROZEN_WIRE_VERSION,
    canonical_json,
)


def test_golden_bytes() -> None:
    value = {
        "z": [1, 2.5, True, None],
        "a": {"café": "ё", "b": (0.1, -0.0)},
        "Ä": "x",
    }

    assert canonical_json(value).encode("utf-8") == (
        b'{"a":{"b":[0.1,-0.0],"caf\xc3\xa9":"\xd1\x91"},"z":[1,2.5,true,null],"\xc3\x84":"x"}'
    )


def test_keys_are_sorted_by_code_point_at_every_depth() -> None:
    assert canonical_json({"b": {"y": 1, "x": 2}, "a": 0, "B": 3}) == (
        '{"B":3,"a":0,"b":{"x":2,"y":1}}'
    )


def test_separators_are_compact() -> None:
    assert canonical_json({"a": [1, {"b": "c d"}]}) == '{"a":[1,{"b":"c d"}]}'


def test_keys_and_strings_are_nfc_normalised_before_sorting() -> None:
    # Decomposed "A" + combining diaeresis sorts before "a"; its NFC form (U+00C4) sorts after "z".
    assert canonical_json({"Ä": "é", "z": ["ö"], "a": 1}) == (
        '{"a":1,"z":["ö"],"Ä":"é"}'
    )


def test_keys_colliding_after_normalisation_are_rejected() -> None:
    with pytest.raises(ValueError, match="collide after normalization"):
        canonical_json({"café": 1, "café": 2})


def test_non_ascii_text_is_emitted_raw_not_escaped() -> None:
    encoded = canonical_json({"k": "日本 \U0001f600"})

    assert encoded == '{"k":"日本 \U0001f600"}'
    assert "\\u" not in encoded


@pytest.mark.parametrize("number", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_numbers_are_rejected(number: float) -> None:
    with pytest.raises(ValueError, match="must be finite"):
        canonical_json({"n": [number]})


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ({1: "x"}, "object keys must be strings"),
        ({"s": {1, 2}}, "unsupported canonical JSON value: set"),
        ({"b": b"x"}, "unsupported canonical JSON value: bytes"),
        ({"k": "\ud800"}, "must be valid UTF-8 text"),
        ({"\udfff": 1}, "must be valid UTF-8 text"),
    ],
)
def test_unencodable_values_are_rejected(value: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        canonical_json(value)


def test_frozen_wire_identity() -> None:
    assert (FROZEN_WIRE_NAME, FROZEN_WIRE_VERSION) == ("frozen-planner", 1)
    assert canonical_json(FROZEN_WIRE) == '{"name":"frozen-planner","version":1}'
