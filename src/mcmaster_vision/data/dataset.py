"""PyTorch datasets (imported lazily; the rest of the package does not need torch)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from PIL import Image

from mcmaster_vision.data.augment import PhotoAugmenter
from mcmaster_vision.schemas import Part


def _require_torch():
    try:
        import torch  # noqa: F401
        from torch.utils.data import Dataset
    except ImportError as e:  # pragma: no cover - exercised only without torch
        raise ImportError("Training datasets need torch: pip install 'mcmaster-vision[ml]'") from e
    return Dataset


def build_label_map(parts: Sequence[Part]) -> dict[str, int]:
    return {p.part_number: i for i, p in enumerate(sorted(parts, key=lambda p: p.part_number))}


def worker_init_fn(worker_id: int) -> None:
    """Reseed the dataset's augmenter inside each DataLoader worker.

    Forked workers inherit identical RNG state, so without this every worker (and
    every epoch) would replay the same augmentations."""
    import torch

    info = torch.utils.data.get_worker_info()
    if info is None:
        return
    ds = info.dataset
    aug = getattr(ds, "augmenter", None)
    if aug is not None:
        aug.reseed(int(torch.initial_seed()) % (2**31) + worker_id)


def make_contrastive_dataset(
    parts: Sequence[Part],
    transform: Callable[[Image.Image], Any],
    augmenter: PhotoAugmenter | None = None,
    views: int = 2,
    image_size: int = 224,
):
    """Each item returns ``views`` augmented tensors of the same SKU plus its label.

    Used by supervised-contrastive and ArcFace training alike.
    """
    Dataset = _require_torch()
    import torch

    augmenter = augmenter or PhotoAugmenter()
    label_map = build_label_map(parts)
    items = [(p, path) for p in parts for path in p.image_paths]

    class ContrastiveDataset(Dataset):  # type: ignore[misc,valid-type]
        def __init__(self) -> None:
            self.augmenter = augmenter

        def __len__(self) -> int:
            return len(items)

        def __getitem__(self, idx: int):
            part, path = items[idx]
            img = Image.open(path).convert("RGB")
            tensors = [transform(augmenter(img, out_size=image_size)) for _ in range(views)]
            return torch.stack(tensors), label_map[part.part_number]

    return ContrastiveDataset(), label_map


def make_catalog_dataset(parts: Sequence[Part], transform: Callable[[Image.Image], Any]):
    """Clean catalog images (no augmentation) for building the gallery / index."""
    Dataset = _require_torch()
    items = [(p.part_number, path) for p in parts for path in p.image_paths]

    class CatalogDataset(Dataset):  # type: ignore[misc,valid-type]
        def __len__(self) -> int:
            return len(items)

        def __getitem__(self, idx: int):
            pn, path = items[idx]
            return transform(Image.open(path).convert("RGB")), pn

    return CatalogDataset()
