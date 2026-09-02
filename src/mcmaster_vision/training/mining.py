"""Hard-negative mining.

After each epoch, embed the training gallery, find for every SKU the nearest SKUs
from *other* families, and use them to build batches where confusable parts appear
together. Contrastive losses learn far more from those than from random batches.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

import numpy as np

from mcmaster_vision.schemas import Part


def mine_hard_negatives(
    parts: Sequence[Part],
    embeddings: np.ndarray,
    *,
    per_part: int = 4,
    exclude_same_family: bool = True,
) -> dict[str, list[str]]:
    """``embeddings[i]`` is the mean L2-normalised embedding of ``parts[i]``.
    Returns part_number -> list of confusable part numbers."""
    if len(parts) == 0:
        return {}
    sims = embeddings @ embeddings.T
    np.fill_diagonal(sims, -np.inf)
    fam = [p.family_id for p in parts]
    out: dict[str, list[str]] = {}
    for i, p in enumerate(parts):
        order = np.argsort(-sims[i])
        negs: list[str] = []
        for j in order:
            if exclude_same_family and fam[j] is not None and fam[j] == fam[i]:
                continue
            negs.append(parts[j].part_number)
            if len(negs) >= per_part:
                break
        out[p.part_number] = negs
    return out


def hard_batch_sampler(
    parts: Sequence[Part], hard_negatives: dict[str, list[str]], batch_parts: int, seed: int = 0
):
    """Yield lists of part numbers: each batch = random anchors + their hard negatives."""
    rng = random.Random(seed)
    pns = [p.part_number for p in parts]
    idx = {pn: i for i, pn in enumerate(pns)}
    while True:
        rng.shuffle(pns)
        batch: list[int] = []
        seen: set[int] = set()
        for pn in pns:
            for cand in (pn, *hard_negatives.get(pn, [])):
                i = idx.get(cand)
                if i is None or i in seen:
                    continue
                batch.append(i)
                seen.add(i)
                if len(batch) >= batch_parts:
                    yield batch
                    batch, seen = [], set()
