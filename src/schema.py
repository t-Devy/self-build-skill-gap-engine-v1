
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from src.constants import FEATURE_COLS, TARGET_COL, ALLOWED_LABELS, RAW_DATA_PATH
from src.data import load_raw_data


@dataclass
class SchemaReport:
    ok: bool
    errors: list[str]


def _require_columns(df: pd.DataFrame, required: Iterable[str]) -> list[str]:
    missing = [c for c in required if c not in df.columns]
    return [f"missing column {c}" for c in missing]

def validate_schema(df: pd.DataFrame) -> SchemaReport:
    errors: list[str] = []

    required = FEATURE_COLS + [TARGET_COL]
    errors += _require_columns(df, required)

    if not errors:
        errors += _validate_bounds(df, FEATURE_COLS)
        errors += validate_labels(df, TARGET_COL, ALLOWED_LABELS)

    if errors:
        return SchemaReport(False, errors)
    return SchemaReport(len(errors) == 0, errors)

def _validate_bounds(df: pd.DataFrame, cols: Iterable[str], lo: int = 0, hi: int = 3) -> list[str]:
    errors: list[str] = []

    for col in cols:
        numeric_col = pd.to_numeric(df[col], errors='coerce')
        if not numeric_col.between(lo, hi).all():
            errors.append(f"{col} must be in [{lo}, {hi}]")
    return errors

def validate_labels(df: pd.DataFrame, target_col: str, allowed_labels: list[str]) -> list[str]:
    errors: list[str] = []

    unique_labels = df[target_col].unique()

    for label in unique_labels:
        if label not in allowed_labels:
            errors.append(f"{label} not allowed in labels")
    return errors










