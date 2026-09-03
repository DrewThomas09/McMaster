"""Retrieval: embedding -> per-part candidate list with a category prior."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mcmaster_vision.catalog.taxonomy import Taxonomy
from mcmaster_vision.index.base import VectorIndex


@dataclass
class Hit:
    part_number: str
    similarity: float  # best image-level cosine similarity
    hits: int  # how many (variant, image) pairs of this part landed in the top-K
    category_prior: float = 0.0


def _softmax(x: np.ndarray, temp: float) -> np.ndarray:
    z = (x - x.max()) / temp
    e = np.exp(z)
    return e / e.sum()


class Retriever:
    def __init__(
        self,
        index: VectorIndex,
        top_k: int = 50,
        category_weight: float = 0.15,
        category_temp: float = 0.05,
        qe_k: int = 0,
        qe_alpha: float = 3.0,
    ):
        self.index = index
        self.top_k = top_k
        # alpha-weighted query expansion: re-query with the mean of the query and
        # its top-``qe_k`` gallery neighbours (weights = similarity ** alpha).
        self.qe_k = qe_k
        self.qe_alpha = qe_alpha
        self.category_weight = category_weight
        self.category_temp = category_temp
        self.category_depth = int(index.meta.get("category_depth", 2))

    def category_prior(self, query: np.ndarray) -> dict[str, float]:
        scores = self.index.category_scores(_pool(query))
        if not scores:
            return {}
        names = list(scores)
        probs = _softmax(np.array([scores[n] for n in names], np.float32), self.category_temp)
        return dict(zip(names, probs.tolist(), strict=True))

    def retrieve(
        self, query: np.ndarray, top_k: int | None = None, oversample: int = 3
    ) -> list[Hit]:
        """``query`` is (D,) or (V, D) for V test-time-augmented variants.

        Each variant is searched separately and a part keeps its best similarity
        across variants and across its catalog images.
        """
        k = top_k or self.top_k
        q = np.asarray(query, dtype=np.float32)
        if q.ndim == 1:
            q = q[None, :]
        if self.qe_k > 0:
            q = self.expand_queries(q)
        scores, rows = self.index.search(q, min(k * oversample, len(self.index)))
        best: dict[str, Hit] = {}
        for v in range(q.shape[0]):
            seen_this_variant: set[str] = set()
            for s, r in zip(scores[v], rows[v], strict=True):
                if r < 0:
                    continue
                pn = self.index.ids[int(r)]
                h = best.get(pn)
                if h is None:
                    best[pn] = Hit(pn, float(s), 1)
                else:
                    h.similarity = max(h.similarity, float(s))
                    if pn not in seen_this_variant:
                        h.hits += 1
                seen_this_variant.add(pn)
        return sorted(best.values(), key=lambda h: -h.similarity)[:k]

    def expand_queries(self, q: np.ndarray) -> np.ndarray:
        """Alpha query expansion (Radenovic et al.): each variant is replaced by a
        similarity-weighted average of itself and its nearest gallery vectors.
        Requires a backend that exposes vectors (numpy index); others are left as is."""
        vectors = getattr(self.index, "matrix", None)
        if vectors is None or len(vectors) == 0:
            return q
        scores, rows = self.index.search(q, min(self.qe_k, len(self.index)))
        out = []
        for v in range(q.shape[0]):
            valid = rows[v] >= 0
            neigh = vectors[rows[v][valid]]
            w = np.clip(scores[v][valid], 0, None) ** self.qe_alpha
            expanded = q[v] + (w[:, None] * neigh).sum(axis=0)
            out.append(expanded / (np.linalg.norm(expanded) + 1e-8))
        return np.stack(out).astype(np.float32)

    def apply_category_prior(
        self, hits: list[Hit], query: np.ndarray, categories: dict[str, list[str]]
    ) -> list[Hit]:
        """``categories`` maps part_number -> category_path. Adds the prior in place."""
        prior = self.category_prior(query)
        if not prior:
            return hits
        for h in hits:
            key = Taxonomy.key(categories.get(h.part_number, []), self.category_depth)
            h.category_prior = prior.get(key, 0.0)
        return hits


def _pool(query: np.ndarray) -> np.ndarray:
    q = np.asarray(query, dtype=np.float32)
    if q.ndim == 2:
        q = q.mean(axis=0)
    return q / (np.linalg.norm(q) + 1e-8)
