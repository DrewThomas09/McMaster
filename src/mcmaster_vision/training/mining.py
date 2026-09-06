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
    chunk: int = 2048,
) -> dict[str, list[str]]:
    """``embeddings[i]`` is the mean L2-normalised embedding of ``parts[i]``.
    Returns part_number -> list of confusable part numbers.

    Similarities are computed in row chunks against the gallery and only the top
    candidates are partially sorted, so memory is O(chunk x N), not O(N^2)."""
    n = len(parts)
    if n == 0:
        return {}
    emb = np.asarray(embeddings, dtype=np.float32)
    fam = np.array([p.family_id or "" for p in parts])
    out: dict[str, list[str]] = {}
    # ask for more than per_part so same-family rows can be skipped without a full sort
    k = min(n - 1, per_part * 8 + 8) if n > 1 else 0
    for start in range(0, n, chunk):
        stop = min(n, start + chunk)
        sims = emb[start:stop] @ emb.T  # (chunk, N)
        for r, i in enumerate(range(start, stop)):
            sims[r, i] = -np.inf
            negs: list[str] = []
            if k > 0:
                cand = np.argpartition(-sims[r], k - 1)[:k]
                cand = cand[np.argsort(-sims[r][cand])]
                for j in cand:
                    if exclude_same_family and fam[j] and fam[j] == fam[i]:
                        continue
                    negs.append(parts[j].part_number)
                    if len(negs) >= per_part:
                        break
            out[parts[i].part_number] = negs
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
        if batch:  # trailing partial batch: never leave the consumer waiting forever
            yield batch
