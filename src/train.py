
from __future__ import annotations


import torch.nn as nn
import torch
import json

from pathlib import Path
from src.metrics import _accuracy, _topk_accuracy
from src.loaders import make_loaders
from src.model import SkillPriorityNet
from src.constants import MODEL_PATH, META_PATH, ARTIFACTS_DIR





def evaluate(model: nn.Module, loader, loss_fn, device) -> dict[str, float]:
    model.eval()

    total_loss = 0.0
    # total batches
    total_examples = 0

    overall_acc = 0.0
    top1_sum = 0.0
    top3_sum = 0.0

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = loss_fn(logits, yb)

            # .item() retrieves scalar loss from loss_fn as a float
            bs = xb.size(0)
            total_loss += loss.item() * bs
            total_examples += bs

            # multiply accuracy by batch size to get total counts - weighted by bs
            # (e.g. 0.5 rate * 32 batch = 16 correct)
            overall_acc += _accuracy(logits, yb) * bs
            top1_sum += _topk_accuracy(logits, yb, k=1) * bs
            top3_sum += _topk_accuracy(logits, yb, k=3) * bs

    # keep total from dividing by 0 when batches/examples are empty
    # take (count / total) to convert back to percent/weighted average
    return {
            "loss": total_loss / max(total_examples, 1),
            "overall_acc": overall_acc / max(total_examples, 1),
            "top1_acc": top1_sum / max(total_examples, 1),
            "top3_acc": top3_sum / max(total_examples, 1),
        }

def train():
    # prepare device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # prepare data
    bundle = make_loaders(batch_size=32)

    # prepare model
    input_dim = bundle.meta['num_features']
    num_classes = bundle.meta['num_classes']

    model = SkillPriorityNet(input_dim=input_dim, num_classes=num_classes, hidden_dim=64, dropout=0.1).to(device)

    # prepare loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # training loop
    epochs = 30
    for epoch in range(1, epochs + 1):
        model.train()

        total_loss = 0.0
        total_examples = 0
        overall_acc = 0.0
        top1_sum = 0.0
        top3_sum = 0.0

        for xb, yb in bundle.train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            # clear previous grads
            optimizer.zero_grad()

            # forward pass
            logits = model(xb)

            # compute loss
            loss = loss_fn(logits, yb)
            loss.backward()

            # update gradients
            optimizer.step()

            # accumulate loss
            bs = xb.size(0)
            total_loss += loss.item() * bs
            total_examples += bs

            overall_acc += _accuracy(logits, yb) * bs
            top1_sum += _topk_accuracy(logits, yb, k=1) * bs
            top3_sum += _topk_accuracy(logits, yb, k=3) * bs

        train_metrics = {
            "loss": total_loss / max(total_examples, 1),
            "overall_acc": overall_acc / max(total_examples, 1),
            "top1_acc": top1_sum / max(total_examples, 1),
            "top3_acc": top3_sum / max(total_examples, 1),
        }

        val_metrics = evaluate(model, bundle.val_loader, loss_fn, device)

        if epoch == 1 or epoch % 5 == 0:
            print(
                f"Epoch {epoch:02d}/{epochs} | "
                f" train loss = {train_metrics['loss']:.4f} "
                f" train_overall_acc = {train_metrics['overall_acc']:.3f}"
                f" train_top1 = {train_metrics['top1_acc']:.3f}"
                f" train_top3 = {train_metrics['top3_acc']:.3f} | "
                f" val_loss = {val_metrics['loss']:.4f} "
                f" val_overall_acc = {val_metrics['overall_acc']:.3f}"
                f" val_top1 = {val_metrics['top1_acc']:.3f} "
                f" val_top3 = {val_metrics['top3_acc']:.3f} "
            )


    # save artifacts
    Path(ARTIFACTS_DIR).mkdir(parents=True, exist_ok=True)

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(bundle.meta, f, indent=2)

    torch.save(model.state_dict(), str(MODEL_PATH))


    print(f"\nSaved meta -> {META_PATH}")
    print(f"Saved model -> {MODEL_PATH}")




