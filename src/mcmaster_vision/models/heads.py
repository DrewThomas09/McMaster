"""Trainable heads (torch). Imported only by the trainer and checkpoint loader."""

from __future__ import annotations

import math

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

if nn is not None:

    class ProjectionHead(nn.Module):
        """MLP that maps backbone features into the retrieval embedding space."""

        def __init__(
            self, in_dim: int, out_dim: int = 512, hidden: int | None = None, dropout: float = 0.0
        ):
            super().__init__()
            hidden = hidden or in_dim
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, out_dim),
            )

        def forward(self, x):
            return F.normalize(self.net(x), dim=-1)

    class ArcFaceHead(nn.Module):
        """Additive angular margin classifier over SKUs (or families).

        With 700k classes the weight matrix is 700k x 512 floats (~1.4 GB); use the
        ``family`` label space or partial-FC sampling for full-catalog training.
        """

        def __init__(self, dim: int, n_classes: int, scale: float = 30.0, margin: float = 0.3):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(n_classes, dim))
            nn.init.xavier_uniform_(self.weight)
            self.scale, self.margin = scale, margin

        def forward(self, emb, labels):
            cos = F.linear(F.normalize(emb), F.normalize(self.weight)).clamp(-1 + 1e-7, 1 - 1e-7)
            theta = torch.acos(cos)
            target = torch.cos(theta + self.margin)
            one_hot = F.one_hot(labels, cos.shape[1]).bool()
            logits = torch.where(one_hot, target, cos) * self.scale
            return logits

    class CategoryHead(nn.Module):
        """Auxiliary coarse-category classifier; regularises the embedding and yields
        a category prior at inference time."""

        def __init__(self, dim: int, n_categories: int):
            super().__init__()
            self.fc = nn.Linear(dim, n_categories)

        def forward(self, emb):
            return self.fc(emb)

    def arcface_init_like(dim: int, n_classes: int) -> float:
        return 1 / math.sqrt(dim * n_classes)
