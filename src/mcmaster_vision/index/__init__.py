"""Vector index: maps embeddings to part numbers at catalog scale."""

from mcmaster_vision.index.base import VectorIndex, load_index, open_index
from mcmaster_vision.index.builder import add_photos, build_index
from mcmaster_vision.index.numpy_index import NumpyIndex

__all__ = ["NumpyIndex", "VectorIndex", "add_photos", "build_index", "load_index", "open_index"]
