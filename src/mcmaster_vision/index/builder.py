"""Build (and incrementally extend) the gallery index from the catalog store.

* streaming: images are loaded and embedded in batches, memory stays flat
* parallel: ``workers > 1`` embeds part shards in spawned processes (CPU boxes)
* incremental: ``only_new=True`` embeds only parts missing from an existing index
* gallery augmentation: ``gallery_augment`` extra photo-style rows per image
* category centroids for the coarse prior are (re)computed every build

At 700k parts x 3 images, TinyCNN on one CPU core embeds ~100 img/s (6 h) and
4 workers bring that to ~1.5 h; a single GPU with CLIP does it in ~35 min.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from mcmaster_vision.catalog.store import CatalogStore
from mcmaster_vision.catalog.taxonomy import Taxonomy
from mcmaster_vision.data.augment import AugmentConfig, PhotoAugmenter
from mcmaster_vision.index.base import VectorIndex, load_index, open_index
from mcmaster_vision.models.backbone import l2_normalize
from mcmaster_vision.models.embedder import PartEmbedder
from mcmaster_vision.pipeline.preprocess import preprocess_catalog
from mcmaster_vision.schemas import Part

log = logging.getLogger(__name__)

FAISS_THRESHOLD = 50_000  # vectors; above this prefer FAISS when it is installed


def choose_backend(n_vectors: int, requested: str | None = None) -> str:
    """'numpy' below the threshold, 'faiss' above it when available."""
    if requested and requested != "auto":
        return requested
    if n_vectors >= FAISS_THRESHOLD:
        try:
            import faiss  # noqa: F401

            return "faiss"
        except ImportError:
            log.warning("%d vectors: install faiss-cpu for a faster, smaller index", n_vectors)
    return "numpy"


def _rows_for_part(
    part: Part,
    embedder: PartEmbedder,
    image_size: int,
    augmenter: PhotoAugmenter | None,
    gallery_augment: int,
    category_depth: int,
):
    """(ids, images, category keys) for one part."""
    cat_key = Taxonomy.key(part.category_path, category_depth)
    ids, images, cats = [], [], []
    for path in part.image_paths:
        try:
            raw = Image.open(path).convert("RGB")
        except (OSError, FileNotFoundError):
            log.warning("skipping unreadable image %s (%s)", path, part.part_number)
            continue
        variants = [preprocess_catalog(raw, size=image_size)]
        if augmenter is not None:
            variants += [
                preprocess_catalog(augmenter(raw, out_size=image_size), size=image_size)
                for _ in range(gallery_augment)
            ]
        for v in variants:
            ids.append(part.part_number)
            images.append(v)
            cats.append(cat_key)
    return ids, images, cats


def embed_parts(
    parts: Iterable[Part],
    embedder: PartEmbedder,
    *,
    batch_size: int = 256,
    image_size: int = 224,
    gallery_augment: int = 0,
    category_depth: int = 2,
    seed: int = 0,
    progress: Callable[[int], None] | None = None,
) -> tuple[list[str], np.ndarray, list[str]]:
    """Embed every image of ``parts``. Returns (ids, vectors, category keys)."""
    augmenter = PhotoAugmenter(AugmentConfig.gallery(), seed=seed) if gallery_augment > 0 else None
    all_ids: list[str] = []
    all_cats: list[str] = []
    chunks: list[np.ndarray] = []
    ids: list[str] = []
    images: list[Image.Image] = []
    cats: list[str] = []

    def flush() -> None:
        nonlocal ids, images, cats
        if images:
            chunks.append(embedder.embed_catalog(images))
            all_ids.extend(ids)
            all_cats.extend(cats)
        ids, images, cats = [], [], []

    done = 0
    for part in parts:
        i, im, c = _rows_for_part(
            part, embedder, image_size, augmenter, gallery_augment, category_depth
        )
        ids += i
        images += im
        cats += c
        if len(images) >= batch_size:
            flush()
        done += 1
        if progress and done % 500 == 0:
            progress(done)
    flush()
    vectors = np.concatenate(chunks) if chunks else np.zeros((0, embedder.dim), np.float32)
    return all_ids, vectors.astype(np.float32), all_cats


# ---------------------------------------------------------------------------
# multiprocess sharding
# ---------------------------------------------------------------------------
_WORKER_EMBEDDER: PartEmbedder | None = None


def _worker_init(settings_dump: dict) -> None:
    global _WORKER_EMBEDDER
    from mcmaster_vision.config import Settings
    from mcmaster_vision.models.backbone import load_backbone

    try:  # torch is an optional extra; the hash backbone needs none of it
        import torch

        torch.set_num_threads(1)
    except ImportError:
        pass
    _WORKER_EMBEDDER = PartEmbedder(load_backbone(Settings(**settings_dump)))


def _worker_embed(args: tuple) -> tuple[list[str], np.ndarray, list[str]]:
    parts_json, kw = args
    parts = [Part.model_validate_json(j) for j in parts_json]
    assert _WORKER_EMBEDDER is not None
    return embed_parts(parts, _WORKER_EMBEDDER, **kw)


def embed_parts_parallel(
    parts: list[Part], settings_dump: dict, workers: int, shard: int = 200, **kw
) -> tuple[list[str], np.ndarray, list[str]]:
    """Embed ``parts`` across ``workers`` spawned processes, each loading its own backbone
    from ``settings_dump`` (a ``Settings.model_dump(mode='json')``)."""
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    shards = [
        [p.model_dump_json() for p in parts[i : i + shard]] for i in range(0, len(parts), shard)
    ]
    ids: list[str] = []
    cats: list[str] = []
    chunks: list[np.ndarray] = []
    with ctx.Pool(workers, initializer=_worker_init, initargs=(settings_dump,)) as pool:
        for i, (sid, vec, scat) in enumerate(
            pool.imap(_worker_embed, [(s, kw) for s in shards]), 1
        ):
            ids.extend(sid)
            cats.extend(scat)
            if len(vec):
                chunks.append(vec)
            log.info("embedded shard %d/%d", i, len(shards))
    dim = chunks[0].shape[1] if chunks else 0
    return ids, (np.concatenate(chunks) if chunks else np.zeros((0, dim), np.float32)), cats


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
def _centroids(
    index: VectorIndex, vectors: np.ndarray, cats: list[str], *, merge: bool = False
) -> None:
    """(Re)compute per-category centroids. With ``merge=True`` the new rows are blended
    into the existing centroids weighted by the per-category counts kept in
    ``index.meta['category_counts']``, so an incremental build of 5 screws does not
    replace a centroid that summarised 10,000."""
    sums: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    for c, v in zip(cats, vectors, strict=True):
        sums[c] = sums.get(c, 0) + v
        counts[c] = counts.get(c, 0) + 1
    if merge and index.category_centroids is not None:
        old_counts: dict[str, int] = dict(index.meta.get("category_counts", {}))
        for name, cen in zip(index.category_names, index.category_centroids, strict=True):
            n_old = int(old_counts.get(name, 1))
            sums[name] = sums.get(name, 0) + cen * n_old
            counts[name] = counts.get(name, 0) + n_old
    if sums:
        names = sorted(sums)
        index.set_categories(names, l2_normalize(np.stack([sums[n] / counts[n] for n in names])))
        index.meta["category_counts"] = {n: int(counts[n]) for n in names}


def build_index(
    store: CatalogStore,
    embedder: PartEmbedder,
    backend: str = "numpy",
    *,
    batch_size: int = 256,
    category_depth: int = 2,
    image_size: int = 224,
    gallery_augment: int = 0,
    seed: int = 0,
    progress: Callable[[int, int], None] | None = None,
    out_path: str | Path | None = None,
    only_new: bool = False,
    workers: int = 1,
    settings_dump: dict | None = None,
) -> VectorIndex:
    parts = list(store.iter_parts(with_images_only=True))
    total = len(parts)
    existing: VectorIndex | None = None
    if only_new and out_path and (Path(out_path) / "meta.json").exists():
        existing = load_index(out_path)
        if (
            existing.meta.get("backbone") != embedder.version
            or int(existing.meta.get("gallery_augment", 0)) != gallery_augment
            or int(existing.meta.get("image_size", image_size)) != image_size
            or int(existing.meta.get("category_depth", category_depth)) != category_depth
        ):
            log.warning("existing index was built differently; rebuilding from scratch")
            existing = None
        else:
            have = set(existing.ids)
            parts = [p for p in parts if p.part_number not in have]
            log.info("incremental build: %d new parts (index holds %d)", len(parts), len(have))

    kw = dict(
        batch_size=batch_size,
        image_size=image_size,
        gallery_augment=gallery_augment,
        category_depth=category_depth,
        seed=seed,
    )
    if workers > 1 and settings_dump is not None and len(parts) > 2 * 200:
        ids, vectors, cats = embed_parts_parallel(parts, settings_dump, workers, **kw)
    else:
        ids, vectors, cats = embed_parts(
            parts, embedder, progress=(lambda d: progress(d, total)) if progress else None, **kw
        )

    if existing is not None:
        index = existing
        if len(ids):
            index.add(ids, vectors)
        if len(ids):
            _centroids(index, vectors, cats, merge=True)
    else:
        backend = choose_backend(len(ids), backend)
        index = open_index(backend, embedder.dim)
        if len(ids):
            index.add(ids, vectors)
        _centroids(index, vectors, cats)

    index.meta.update(
        {
            "backbone": embedder.version,
            "category_depth": category_depth,
            "image_size": image_size,
            "gallery_augment": gallery_augment,
            "built_at": datetime.now(timezone.utc).isoformat(),
            "parts": len(set(index.ids)),
        }
    )
    if out_path:
        index.save(out_path)
        store.set_meta("index_backbone", embedder.version)
    return index
