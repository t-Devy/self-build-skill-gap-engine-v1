
from __future__ import annotations


import torch


def _accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds == y).float().mean()
    return float(correct.item())

def _topk_accuracy(logits: torch.Tensor, y: torch.Tensor, k: int = 1) -> float:
    """Prediction is correct if true class is in model's topk logits"""
    k = min(k, logits.size(1))
    topk_idx = logits.topk(k=k, dim=1).indices
    correct = (topk_idx == y.unsqueeze(1)).any(dim=1)
    return float(correct.float().mean().item())






