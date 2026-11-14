"""Local temporal convolution heads over frozen layer-6 frame windows."""
from __future__ import annotations

import torch
from torch import nn


class TemporalEventHead(nn.Module):
    def __init__(self, ref_dim: int, n_classes: int, hidden: int = 128):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv1d(768, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.attention = nn.Linear(hidden, 1)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2 + ref_dim, 256), nn.GELU(), nn.Dropout(0.20),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(128, n_classes),
        )

    def forward(self, frames: torch.Tensor, refs: torch.Tensor) -> torch.Tensor:
        # frames [batch,time,768], preserving alignment-centered order.
        z = self.temporal(frames.transpose(1, 2)).transpose(1, 2)
        weights = torch.softmax(self.attention(z).squeeze(-1), dim=1)
        attended = torch.sum(z * weights.unsqueeze(-1), dim=1)
        mean = z.mean(dim=1)
        return self.head(torch.cat([attended, mean, refs], dim=-1))

