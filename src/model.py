
from __future__ import annotations

import torch
import torch.nn as nn

class SkillPriorityNet(nn.Module):
    def __init__(self, input_dim: int=5, num_classes: int=5, hidden_dim: int=32, dropout: float=0.2):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

