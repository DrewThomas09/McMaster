"""PartEmbedder: the single object the pipeline uses to turn images into vectors.

It wraps a backbone and adds test-time augmentation (TTA): the query photo is
embedded at several rotations / flips. The retriever searches with every variant
and keeps each part's best score ("multi-query retrieval"), which makes retrieval
robust to how the user held the phone even for rotation-sensitive descriptors.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from PIL import Image

from mcmaster_vision.models.backbone import Backbone, l2_normalize


class PartEmbedder:
    def __init__(
        self,
        backbone: Backbone,
        tta_rotations: Sequence[int] = (0, 90, 180, 270),
        tta_flip: bool = True,
    ):
        self.backbone = backbone
        self.tta_rotations = tuple(tta_rotations)
        self.tta_flip = tta_flip

    @property
    def dim(self) -> int:
        return self.backbone.dim

    @property
    def version(self) -> str:
        return self.backbone.version

    def embed_catalog(self, images: Sequence[Image.Image]) -> np.ndarray:
        """Gallery side: no TTA (catalog images are canonical)."""
        return self.backbone.embed(images)

    def query_variants(self, image: Image.Image, mode: str = "full") -> list[Image.Image]:
        """``full``: all rotations x flip (8 views); ``fast``: 0/90 degrees, no flip (2 views);
        ``none``: the image as is."""
        if mode == "none":
            return [image]
        rotations = self.tta_rotations if mode == "full" else tuple(self.tta_rotations[:2])
        variants = [image.rotate(r, expand=True) for r in rotations]
        if self.tta_flip and mode == "full":
            variants += [v.transpose(Image.Transpose.FLIP_LEFT_RIGHT) for v in variants]
        return variants

    def embed_query(self, image: Image.Image, tta: bool | str = True) -> np.ndarray:
        """Return an (V, D) matrix of L2-normalised query vectors (V = 1 without TTA)."""
        mode = tta if isinstance(tta, str) else ("full" if tta else "none")
        return l2_normalize(self.backbone.embed(self.query_variants(image, mode)))

    @staticmethod
    def pooled(query: np.ndarray) -> np.ndarray:
        """Mean-pooled single vector (used for the category prior)."""
        q = np.asarray(query, dtype=np.float32)
        return l2_normalize(q.mean(axis=0)) if q.ndim == 2 else l2_normalize(q)
