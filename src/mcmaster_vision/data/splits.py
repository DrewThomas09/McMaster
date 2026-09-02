"""Train / validation splits.

Splitting by *family* rather than by SKU is essential: SKUs in the same family are
near-duplicates, so a random SKU split leaks and inflates validation recall.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

from mcmaster_vision.schemas import Part


def _bucket(key: str, n: int = 1000) -> int:
    return int(hashlib.sha1(key.encode()).hexdigest(), 16) % n


def split_by_family(parts: Iterable[Part], val_frac: float = 0.1) -> tuple[list[Part], list[Part]]:
    train: list[Part] = []
    val: list[Part] = []
    cutoff = int(val_frac * 1000)
    for p in parts:
        key = p.family_id or p.part_number
        (val if _bucket(key) < cutoff else train).append(p)
    return train, val
