"""Ingestion: source -> validation -> store."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from mcmaster_vision.catalog.sources import CatalogSource, open_source
from mcmaster_vision.catalog.store import CatalogStore
from mcmaster_vision.schemas import Part

log = logging.getLogger(__name__)


def _validated(parts: CatalogSource, strict: bool, stats: dict[str, int]):
    for part in parts:
        missing = [p for p in part.image_paths if not Path(p).exists()]
        if missing:
            stats["missing_images"] += len(missing)
            if strict:
                raise FileNotFoundError(f"{part.part_number}: missing images {missing[:3]}")
            part = part.model_copy(
                update={"image_paths": [p for p in part.image_paths if p not in missing]}
            )
        if not part.image_paths:
            stats["without_images"] += 1
        stats["parts"] += 1
        yield part


def ingest(
    source: str | Path | CatalogSource,
    store: CatalogStore,
    *,
    strict: bool = False,
    progress: Callable[[int], None] | None = None,
) -> dict[str, int]:
    """Load every part from ``source`` into ``store``. Returns ingestion statistics."""
    src = source if isinstance(source, CatalogSource) else open_source(source)
    stats = {"parts": 0, "missing_images": 0, "without_images": 0}

    def gen():
        for i, part in enumerate(_validated(src, strict, stats), 1):
            if progress and i % 1000 == 0:
                progress(i)
            yield part

    written = store.upsert(gen())
    stats["written"] = written
    log.info("ingested %s parts (%s without images)", written, stats["without_images"])
    return stats


def ingest_parts(parts: list[Part], store: CatalogStore) -> int:
    return store.upsert(parts)
