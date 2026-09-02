"""FAISS index for full-catalog scale.

Uses HNSW for < 1M vectors (best recall/latency trade-off, no training step) and
can be switched to IVF-PQ for memory-constrained deployments.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from mcmaster_vision.index.base import VectorIndex


class FaissIndex(VectorIndex):
    backend = "faiss"

    def __init__(
        self,
        dim: int,
        kind: str = "hnsw",
        hnsw_m: int = 32,
        ef_search: int = 128,
        nlist: int = 4096,
        pq_m: int = 64,
    ):
        super().__init__(dim)
        try:
            import faiss
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install 'mcmaster-vision[faiss]'") from e
        self._faiss = faiss
        self.kind = kind
        if kind == "hnsw":
            self.index = faiss.IndexHNSWFlat(dim, hnsw_m, faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efSearch = ef_search
            self.index.hnsw.efConstruction = 200
        elif kind == "ivfpq":
            quant = faiss.IndexFlatIP(dim)
            self.index = faiss.IndexIVFPQ(quant, dim, nlist, pq_m, 8, faiss.METRIC_INNER_PRODUCT)
            self.index.nprobe = 32
        elif kind == "flat":
            self.index = faiss.IndexFlatIP(dim)
        else:
            raise ValueError(f"unknown faiss kind {kind}")
        self.meta["faiss_kind"] = kind

    def add(self, ids: Sequence[str], vectors: np.ndarray) -> None:
        vectors = np.ascontiguousarray(np.asarray(vectors, dtype=np.float32))
        if self.kind == "ivfpq" and not self.index.is_trained:
            self.index.train(vectors)
        self.index.add(vectors)
        self.ids.extend(ids)

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        q = np.ascontiguousarray(np.asarray(queries, dtype=np.float32))
        scores, rows = self.index.search(q, k)
        return scores.astype(np.float32), rows.astype(np.int64)

    def _save_vectors(self, path: Path) -> None:
        self._faiss.write_index(self.index, str(path / "index.faiss"))

    def _load_vectors(self, path: Path) -> None:
        self.index = self._faiss.read_index(str(path / "index.faiss"))

    def __len__(self) -> int:
        return len(self.ids)
