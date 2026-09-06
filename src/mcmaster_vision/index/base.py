"""Index interface and on-disk layout.

An index stores one vector per *catalog image* (a SKU may have several) and maps
each row to a part number. Search returns per-image hits; the retriever collapses
them to per-part scores.

On disk an index is a directory:
    <path>/meta.json        backend, dim, backbone version, timestamps
    <path>/ids.json         row -> part number
    <path>/vectors.npy      (numpy backend)  or  index.faiss (faiss backend)
    <path>/categories.npz   category centroids used for the coarse prior
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from mcmaster_vision.schemas import IndexStats


class VectorIndex(ABC):
    backend: str = "base"

    def __init__(self, dim: int):
        self.dim = dim
        self.ids: list[str] = []
        self.meta: dict = {}
        self.category_names: list[str] = []
        self.category_centroids: np.ndarray | None = None

    # ---------------------------------------------------------- abstract
    @abstractmethod
    def add(self, ids: Sequence[str], vectors: np.ndarray) -> None: ...

    @abstractmethod
    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (scores, row_indices), each (Q, k). Scores are cosine similarities."""

    @abstractmethod
    def _save_vectors(self, path: Path) -> None: ...

    @abstractmethod
    def _load_vectors(self, path: Path) -> None: ...

    @abstractmethod
    def __len__(self) -> int: ...

    # ------------------------------------------------------------ common
    def search_ids(self, query: np.ndarray, k: int) -> list[tuple[str, float]]:
        q = np.asarray(query, dtype=np.float32).reshape(1, -1)
        scores, rows = self.search(q, min(k, len(self)))
        return [
            (self.ids[int(r)], float(s)) for s, r in zip(scores[0], rows[0], strict=True) if r >= 0
        ]

    def set_categories(self, names: Sequence[str], centroids: np.ndarray) -> None:
        self.category_names = list(names)
        self.category_centroids = np.asarray(centroids, dtype=np.float32)

    def category_scores(self, query: np.ndarray) -> dict[str, float]:
        if self.category_centroids is None or not len(self.category_names):
            return {}
        sims = self.category_centroids @ np.asarray(query, dtype=np.float32)
        return dict(zip(self.category_names, sims.tolist(), strict=True))

    def stats(self) -> IndexStats:
        built = self.meta.get("built_at")
        return IndexStats(
            backend=self.backend,
            vectors=len(self),
            dim=self.dim,
            parts=len(set(self.ids)),
            built_at=datetime.fromisoformat(built) if built else None,
            backbone=self.meta.get("backbone", "unknown"),
        )

    def save(self, path: str | Path) -> None:
        """Write the index atomically: build in a sibling temp directory, then swap it
        into place, so a server reloading mid-write never sees a partial index."""
        import shutil
        import tempfile

        final = Path(path)
        final.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix=final.name + ".tmp-", dir=final.parent))
        try:
            self._write_all(tmp)
            if final.exists():
                old = final.with_name(final.name + ".old")
                if old.exists():
                    shutil.rmtree(old)
                final.rename(old)
                tmp.rename(final)
                shutil.rmtree(old, ignore_errors=True)
            else:
                tmp.rename(final)
        finally:
            if tmp.exists():
                shutil.rmtree(tmp, ignore_errors=True)

    def _write_all(self, p: Path) -> None:
        p.mkdir(parents=True, exist_ok=True)
        self.meta.update(
            {
                "backend": self.backend,
                "dim": self.dim,
                "vectors": len(self),
                "built_at": self.meta.get("built_at") or datetime.now(timezone.utc).isoformat(),
            }
        )
        (p / "meta.json").write_text(json.dumps(self.meta, indent=2), encoding="utf-8")
        (p / "ids.json").write_text(json.dumps(self.ids), encoding="utf-8")
        if self.category_centroids is not None:
            np.savez(
                p / "categories.npz",
                names=np.array(self.category_names),
                centroids=self.category_centroids,
            )
        self._save_vectors(p)

    @classmethod
    def load(cls, path: str | Path) -> VectorIndex:
        p = Path(path)
        meta = json.loads((p / "meta.json").read_text(encoding="utf-8"))
        idx = cls(int(meta["dim"]))
        idx.meta = meta
        idx.ids = json.loads((p / "ids.json").read_text(encoding="utf-8"))
        cats = p / "categories.npz"
        if cats.exists():
            z = np.load(cats, allow_pickle=False)
            idx.set_categories([str(n) for n in z["names"]], z["centroids"])
        idx._load_vectors(p)
        return idx


def open_index(backend: str, dim: int) -> VectorIndex:
    if backend == "numpy":
        from mcmaster_vision.index.numpy_index import NumpyIndex

        return NumpyIndex(dim)
    if backend == "faiss":
        from mcmaster_vision.index.faiss_index import FaissIndex

        return FaissIndex(dim)
    raise ValueError(f"unknown index backend {backend}")


def load_index(path: str | Path) -> VectorIndex:
    """Load whichever backend the directory was saved with."""
    meta = json.loads((Path(path) / "meta.json").read_text(encoding="utf-8"))
    backend = meta.get("backend", "numpy")
    if backend == "numpy":
        from mcmaster_vision.index.numpy_index import NumpyIndex

        return NumpyIndex.load(path)
    if backend == "faiss":
        from mcmaster_vision.index.faiss_index import FaissIndex

        return FaissIndex.load(path)
    raise ValueError(f"unknown index backend in meta.json: {backend}")
