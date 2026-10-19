"""Small Allure attachment helpers for human-reviewable acceptance evidence."""
from __future__ import annotations

import json
from typing import Any

import allure
import pandas as pd


def attach_json(name: str, value: Any) -> None:
    """Attach stable structured evidence with readable formatting."""
    allure.attach(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str),
        name=name,
        attachment_type=allure.attachment_type.JSON,
    )


def attach_comparison(name: str, expected: Any, actual: Any) -> None:
    """Attach expected and actual values together so the report remains reviewable on success."""
    attach_json(name, {"expected": expected, "actual": actual})


def attach_frame(name: str, frame: pd.DataFrame) -> None:
    """Attach a bounded scenario DataFrame as inspectable CSV."""
    allure.attach(
        frame.to_csv(index=False),
        name=name,
        attachment_type=allure.attachment_type.CSV,
    )
