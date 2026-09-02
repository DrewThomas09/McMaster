"""Metric-learning losses (torch)."""

from __future__ import annotations

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def supcon_loss(features, labels, temperature: float = 0.07):
    """Supervised contrastive loss (Khosla et al. 2020).

    features: (B, V, D) L2-normalised, V views per sample. labels: (B,)
    """
    b, v, _ = features.shape
    feats = features.reshape(b * v, -1)
    labels = labels.repeat_interleave(v)
    sim = feats @ feats.T / temperature
    self_mask = torch.eye(b * v, device=feats.device, dtype=torch.bool)
    sim = sim.masked_fill(self_mask, -1e9)
    pos_mask = (labels[:, None] == labels[None, :]) & ~self_mask
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    mean_log_prob_pos = (log_prob * pos_mask).sum(1) / pos_mask.sum(1).clamp(min=1)
    return -mean_log_prob_pos.mean()


def triplet_loss(anchor, positive, negative, margin: float = 0.2):
    d_ap = 1 - (anchor * positive).sum(-1)
    d_an = 1 - (anchor * negative).sum(-1)
    return F.relu(d_ap - d_an + margin).mean()


def arcface_loss(logits, labels):
    return F.cross_entropy(logits, labels)
