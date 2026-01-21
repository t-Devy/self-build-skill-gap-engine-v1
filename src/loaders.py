

from __future__ import annotations

from src.data import load_raw_data
from src.schema import validate_schema, SchemaValidationError
from src.features import build_features
from src.constants import RAW_DATA_PATH
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass

import pandas as pd

@dataclass(frozen=True)
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    meta: dict


def make_loaders(csv_path: Path = RAW_DATA_PATH, batch_size=32, val_split=0.2, seed=42, shuffle_train=True):
    df = load_raw_data(csv_path)

    report = validate_schema(df)
    if not report.ok:
        raise SchemaValidationError(report.errors)

    features = build_features(df)
    X = features.X
    y = features.y
    ds = TensorDataset(X, y)

    n = len(ds)
    val_size = int(n * val_split)
    train_size = n - val_size

    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = torch.utils.data.random_split(ds, [train_size, val_size], generator=gen)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_train)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return DataBundle(train_loader=train_loader, val_loader=val_loader, meta=features.meta)

