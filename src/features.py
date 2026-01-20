
from __future__ import annotations

import torch

import numpy as np
import pandas as pd
import json

from typing import Any
from dataclasses import dataclass
from src.constants import FEATURE_COLS, TARGET_COL, META_PATH

@dataclass(frozen=True)
class FeaturePackage:
    X: torch.Tensor
    y: torch.Tensor
    meta: dict[str, Any]

def build_X(df: pd.DataFrame) -> torch.Tensor:
    X_df = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    X = torch.tensor(X_df, dtype=torch.float32)
    return X

def build_label_mapping(df) -> tuple[dict, dict]:
    labels = df[TARGET_COL].unique()
    label_to_idx = {lid: i for i, lid in enumerate(labels)}
    idx_to_label = {i: lid for lid, i in label_to_idx.items()}
    return label_to_idx, idx_to_label

def build_y(df, label_to_index) -> torch.Tensor:
    y_df = df[TARGET_COL].astype(str).map(label_to_index).to_numpy()
    y = torch.tensor(y_df, dtype=torch.long)
    return y

def build_features(df: pd.DataFrame) -> FeaturePackage:
    X = build_X(df)
    label_to_idx, idx_to_label = build_label_mapping(df)
    y = build_y(df, label_to_idx)

    meta: dict[str, Any] = {
        "feature_columns": df.columns.tolist(),
        "label_to_index": label_to_idx,
        "index_to_label": idx_to_label,
        "num_features": len(df[FEATURE_COLS].columns),
        "num_classes": len(df[TARGET_COL].unique()),
    }

    # Save meta, Python to JSON
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return FeaturePackage(X=X, y=y, meta=meta)




