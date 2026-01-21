

from __future__ import annotations

import torch

from src.metrics import _accuracy, _topk_accuracy

def main():

    logits = torch.tensor([[3, 0, 0, 0, 0],
                          [0, 3, 0, 0, 0],
                          [0, 0, 3, 0, 0],
                          [0, 0, 0, 3, 0]])
    y = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    print(logits.shape)
    print(logits.size(1))
    print(y.shape)
    preds = logits.argmax(dim=1)
    print(preds)
    print(preds == y)
    print(f"y.unsqueeze value: {y.unsqueeze(1)}")

    print(_accuracy(logits, y))
    print(_topk_accuracy(logits, y, k=1))
    print(_topk_accuracy(logits, y, k=3))


if __name__ == "__main__":
    main()
