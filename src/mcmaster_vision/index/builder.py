"""Build the gallery index from the catalog store.

For every part with images: load, embed, add to the index. Also computes one
centroid per category (at ``category_depth``) for the coarse prior used during
retrieval. Embedding 700k parts x 3 images on one GPU at ~1000 img/s takes
~35 minutes; the loop below is streaming so memory stays flat.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from mcmaster_vision.catalog.store import CatalogStore
from mcmaster_vision.catalog.taxonomy import Taxonomy
from mcmaster_vision.index.base import VectorIndex, open_index
from mcmaster_vision.models.backbone import l2_normalize
from mcmaster_vision.models.embedder import PartEmbedder
from mcmaster_vision.pipeline.preprocess import preprocess_catalog

log = logging.getLogger(__name__)


def build_index(
    store: CatalogStore,
    embedder: PartEmbedder,
    backend: str = "numpy",
    *,
    batch_size: int = 256,
    category_depth: int = 2,
    image_size: int = 224,
    progress: Callable[[int, int], None] | None = None,
    out_path: str | Path | None = None,
) -> VectorIndex:
    index = open_index(backend, embedder.dim)
    cat_sums: dict[str, np.ndarray] = {}
    cat_counts: dict[str, int] = {}

    ids: list[str] = []
    images: list[Image.Image] = []
    cats: list[str] = []
    done = 0
    total = store.count()

    def flush() -> None:
        nonlocal ids, images, cats
        if not images:
            return
        vecs = embedder.embed_catalog(images)
        index.add(ids, vecs)
        for cat, v in zip(cats, vecs, strict=True):
            cat_sums[cat] = cat_sums.get(cat, np.zeros(embedder.dim, np.float32)) + v
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
        ids, images, cats = [], [], []

    for part in store.iter_parts(with_images_only=True):
        cat_key = Taxonomy.key(part.category_path, category_depth)
        for path in part.image_paths:
            try:
                img = preprocess_catalog(Image.open(path), size=image_size)
            except (OSError, FileNotFoundError):
                log.warning("skipping unreadable image %s (%s)", path, part.part_number)
                continue
            ids.append(part.part_number)
            images.append(img)
            cats.append(cat_key)
            if len(images) >= batch_size:
                flush()
        done += 1
        if progress and done % 500 == 0:
            progress(done, total)
    flush()

    if cat_sums:
        names = sorted(cat_sums)
        centroids = l2_normalize(np.stack([cat_sums[n] / cat_counts[n] for n in names]))
        index.set_categories(names, centroids)

    index.meta.update(
        {
            "backbone": embedder.version,
            "category_depth": category_depth,
            "image_size": image_size,
            "built_at": datetime.now(timezone.utc).isoformat(),
            "parts": len(set(index.ids)),
        }
    )
    if out_path:
        index.save(out_path)
        store.set_meta("index_backbone", embedder.version)
    return index
