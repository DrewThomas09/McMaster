"""PyTorch datasets (imported lazily; the rest of the package does not need torch)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from PIL import Image

from mcmaster_vision.data.augment import PhotoAugmenter
from mcmaster_vision.schemas import Part, spec_key


def _require_torch():
    try:
        import torch  # noqa: F401
        from torch.utils.data import Dataset
    except ImportError as e:  # pragma: no cover - exercised only without torch
        raise ImportError("Training datasets need torch: pip install 'mcmaster-vision[ml]'") from e
    return Dataset


def build_label_map(parts: Sequence[Part]) -> dict[str, int]:
    """Contrastive labels: one per distinct spec, so twins are positives of each other
    rather than negatives the loss can never separate."""
    labels: dict[tuple, int] = {}
    out: dict[str, int] = {}
    for p in sorted(parts, key=lambda p: p.part_number):
        key = spec_key(p)
        if key not in labels:
            labels[key] = len(labels)
        out[p.part_number] = labels[key]
    return out


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
    _require_torch()  # torch must be importable here even though the class is module-level

    augmenter = augmenter or PhotoAugmenter()
    label_map = build_label_map(parts)
    items = [(p.part_number, path) for p in parts for path in p.image_paths]
    return ContrastiveDataset(items, label_map, transform, augmenter, views, image_size), label_map


class _DatasetBase:
    pass


def _dataset_base():
    try:
        return _require_torch()
    except Exception:  # pragma: no cover - torch missing
        return _DatasetBase


class ContrastiveDataset(_dataset_base()):  # type: ignore[misc]
    """Module-level (picklable) so DataLoader workers start under "spawn" too."""

    def __init__(self, items, label_map, transform, augmenter, views, image_size):
        self.items = items
        self.label_map = label_map
        self.transform = transform
        self.augmenter = augmenter
        self.views = views
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        import torch

        pn, path = self.items[idx]
        img = Image.open(path).convert("RGB")
        tensors = [
            self.transform(self.augmenter(img, out_size=self.image_size)) for _ in range(self.views)
        ]
        return torch.stack(tensors), self.label_map[pn]


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
