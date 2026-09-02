"""Datasets, augmentation, and synthetic data for development."""

from mcmaster_vision.data.augment import PhotoAugmenter
from mcmaster_vision.data.splits import split_by_family
from mcmaster_vision.data.synthetic import SyntheticCatalog

__all__ = ["PhotoAugmenter", "SyntheticCatalog", "split_by_family"]
