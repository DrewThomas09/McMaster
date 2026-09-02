"""Exact brute-force cosine index. Simple, dependency-free, fine up to ~100k rows
(700k x 512 float32 = 1.4 GB and ~50 ms per query on a modern CPU, so it even
works at full catalog scale on a beefy box; use FAISS for latency/memory)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from mcmaster_vision.index.base import VectorIndex


class NumpyIndex(VectorIndex):
    backend = "numpy"

    def __init__(self, dim: int):
        super().__init__(dim)
        self._chunks: list[np.ndarray] = []
        self._matrix: np.ndarray | None = None

    def add(self, ids: Sequence[str], vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"expected (N, {self.dim}) vectors, got {vectors.shape}")
        if len(ids) != len(vectors):
            raise ValueError("ids and vectors length mismatch")
        self._chunks.append(vectors)
        self.ids.extend(ids)
        self._matrix = None

    @property
    def matrix(self) -> np.ndarray:
        if self._matrix is None:
            self._matrix = (
                np.concatenate(self._chunks)
                if self._chunks
                else np.zeros((0, self.dim), np.float32)
            )
            self._chunks = [self._matrix] if len(self._matrix) else []
        return self._matrix

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        q = np.asarray(queries, dtype=np.float32)
        m = self.matrix
        if len(m) == 0 or k <= 0:
            return np.zeros((len(q), 0), np.float32), np.zeros((len(q), 0), np.int64)
        k = min(k, len(m))
        sims = q @ m.T
        part = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        part_scores = np.take_along_axis(sims, part, axis=1)
        order = np.argsort(-part_scores, axis=1)
        return np.take_along_axis(part_scores, order, axis=1), np.take_along_axis(
            part, order, axis=1
        )

    def _save_vectors(self, path: Path) -> None:
        np.save(path / "vectors.npy", self.matrix)

    def _load_vectors(self, path: Path) -> None:
        self._chunks = [np.load(path / "vectors.npy")]
        self._matrix = None

    def __len__(self) -> int:
        return len(self.ids)
