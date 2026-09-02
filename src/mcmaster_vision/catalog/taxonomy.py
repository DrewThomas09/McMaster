"""McMaster-Carr product taxonomy.

The live catalog is organised into ~20 top-level categories, each a tree several
levels deep. The taxonomy drives two things in this system:

* a coarse category prior during retrieval (category centroids in embedding space)
* the "family" grouping that lets us answer "this is a 1/4"-20 socket head cap screw,
  probably 1" long" when length is not visually distinguishable.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

# The publicly listed top-level categories of the McMaster-Carr catalog.
TOP_LEVEL_CATEGORIES: tuple[str, ...] = (
    "Abrading & Polishing",
    "Adhesives, Sealants & Tape",
    "Building & Grounds",
    "Electrical & Lighting",
    "Fabricating & Machining",
    "Fastening & Joining",
    "Filtering",
    "Flow & Level Control",
    "Furniture & Storage",
    "Hand Tools",
    "Hardware",
    "Heating & Cooling",
    "Lubricating",
    "Material Handling",
    "Measuring & Inspecting",
    "Office Supplies & Signs",
    "Pipe, Tubing, Hose & Fittings",
    "Plumbing & Janitorial",
    "Power Transmission",
    "Pressure & Temperature Control",
    "Pulling & Lifting",
    "Raw Materials",
    "Safety Supplies",
    "Sawing & Cutting",
    "Sealing",
    "Suspending",
)


class Taxonomy:
    """In-memory category tree built from the ``category_path`` of ingested parts."""

    def __init__(self) -> None:
        self._children: dict[tuple[str, ...], set[str]] = defaultdict(set)
        self._counts: dict[tuple[str, ...], int] = defaultdict(int)

    def add(self, path: Iterable[str]) -> None:
        parts = tuple(path)
        for i in range(len(parts)):
            prefix = parts[:i]
            self._children[prefix].add(parts[i])
            self._counts[parts[: i + 1]] += 1

    def children(self, path: Iterable[str] = ()) -> list[str]:
        return sorted(self._children.get(tuple(path), ()))

    def count(self, path: Iterable[str]) -> int:
        return self._counts.get(tuple(path), 0)

    def leaves(self) -> list[tuple[str, ...]]:
        return sorted(p for p in self._counts if not self._children.get(p))

    @staticmethod
    def key(path: Iterable[str], depth: int | None = None) -> str:
        """Stable string key for a (possibly truncated) category path."""
        parts = list(path)
        if depth is not None:
            parts = parts[:depth]
        return " > ".join(parts)

    def __len__(self) -> int:
        return len(self._counts)
