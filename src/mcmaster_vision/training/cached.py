"""Pre-augmented view cache for fast from-scratch training.

Photo-style augmentation in Pillow costs ~40 ms per image, which caps a CPU
training loop at a few steps per second. Rendering ``K`` augmented views per
catalog image *once* (in a process pool) and training on the cached uint8 tensors
gives 10-20x more optimisation steps per minute. The cache is regenerated every
few epochs so the model still sees fresh augmentations over a run.
"""

from __future__ import annotations

from collections.abc import Sequence
from multiprocessing import get_context

import numpy as np
from PIL import Image

from mcmaster_vision.data.augment import AugmentConfig, PhotoAugmenter
from mcmaster_vision.pipeline.preprocess import preprocess
from mcmaster_vision.schemas import Part


def _to_u8(img: Image.Image, size: int) -> np.ndarray:
    return np.asarray(
        img.convert("RGB").resize((size, size), Image.Resampling.LANCZOS), dtype=np.uint8
    )


def _job(args: tuple[str, int, int, int, dict]) -> np.ndarray:
    path, seed, k, size, cfg_kwargs = args
    aug = PhotoAugmenter(AugmentConfig(**cfg_kwargs), seed=seed)
    img = Image.open(path)
    return np.stack(
        [_to_u8(preprocess(aug(img, out_size=max(160, size)), size), size) for _ in range(k)]
    )


def build_view_cache(
    parts: Sequence[Part],
    k: int,
    image_size: int,
    *,
    cfg: AugmentConfig | None = None,
    seed: int = 0,
    workers: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(views uint8 (N*k, S, S, 3), part_index int64 (N*k,))`` where part_index
    refers to the position of the part in ``parts``."""
    cfg = cfg or AugmentConfig()
    cfg_kwargs = {f: getattr(cfg, f) for f in cfg.__dataclass_fields__}
    jobs, owners = [], []
    for i, p in enumerate(parts):
        for j, path in enumerate(p.image_paths):
            jobs.append((path, seed + 1000 * i + j, k, image_size, cfg_kwargs))
            owners.append(i)
    if not jobs:
        return np.zeros((0, image_size, image_size, 3), np.uint8), np.zeros((0,), np.int64)
    if workers > 1:
        # "spawn" (not fork): forking a process that already initialised torch's
        # thread pool can deadlock the workers.
        with get_context("spawn").Pool(workers) as pool:
            views = pool.map(_job, jobs, chunksize=8)
    else:
        views = [_job(j) for j in jobs]
    x = np.concatenate(views)
    y = np.repeat(np.asarray(owners, dtype=np.int64), k)
    return x, y


def to_tensor(u8: np.ndarray):
    """uint8 (N, S, S, 3) -> float tensor (N, 3, S, S) normalised to [-1, 1]."""
    import torch

    return (
        torch.from_numpy(np.ascontiguousarray(u8)).permute(0, 3, 1, 2).float() / 255.0 - 0.5
    ) / 0.5
