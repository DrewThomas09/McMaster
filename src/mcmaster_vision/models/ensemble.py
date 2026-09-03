"""EnsembleBackbone: concatenate several backbones' embeddings.

Cosine similarity of the concatenation of L2-normalised, weight-scaled vectors
equals the weighted sum of the per-backbone cosine similarities, so a single
vector index over the concatenation performs late fusion for free. Useful to
combine a learned model with the hand-crafted descriptor, or two learned
models with complementary failure modes.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from PIL import Image

from mcmaster_vision.models.backbone import Backbone, l2_normalize


class EnsembleBackbone(Backbone):
    name = "ensemble"

    def __init__(self, members: Sequence[Backbone], weights: Sequence[float] | None = None):
        if not members:
            raise ValueError("ensemble needs at least one backbone")
        self.members = list(members)
        w = np.asarray(weights if weights is not None else [1.0] * len(members), dtype=np.float32)
        if len(w) != len(self.members):
            raise ValueError("one weight per member")
        self.weights = np.sqrt(w / w.sum())  # sqrt so that cosine = weighted mean of member cosines
        self.dim = sum(m.dim for m in self.members)
        self.name = "ensemble(" + "+".join(m.name for m in self.members) + ")"

    @property
    def version(self) -> str:
        return "ensemble(" + "+".join(m.version for m in self.members) + ")"

    def embed(self, images: Sequence[Image.Image]) -> np.ndarray:
        if not images:
            return np.zeros((0, self.dim), dtype=np.float32)
        parts = [
            l2_normalize(m.embed(images)) * w
            for m, w in zip(self.members, self.weights, strict=True)
        ]
        return l2_normalize(np.concatenate(parts, axis=1))
