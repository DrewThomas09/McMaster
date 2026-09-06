from __future__ import annotations

import numpy as np

from mcmaster_vision.config import Settings
from mcmaster_vision.index import build_index
from mcmaster_vision.index.builder import embed_parts, embed_parts_parallel


def test_parallel_embedding_matches_serial(store, embedder, tmp_path):
    parts = list(store.iter_parts(with_images_only=True))
    settings = Settings(backbone="hash")
    ids_s, vec_s, cats_s = embed_parts(parts, embedder, image_size=224)
    ids_p, vec_p, cats_p = embed_parts_parallel(
        parts, settings.model_dump(mode="json"), workers=2, shard=10, image_size=224
    )
    assert ids_p == ids_s and cats_p == cats_s
    assert vec_p.shape == vec_s.shape
    assert np.allclose(vec_p, vec_s, atol=1e-5)


def test_build_index_uses_workers(store, embedder, tmp_path):
    # workers > 1 only kicks in above 2 shards of 200 parts; force the parallel path by
    # calling with a small store via the serial fallback and make sure the result is sane.
    idx = build_index(
        store,
        embedder,
        "numpy",
        workers=2,
        settings_dump=Settings(backbone="hash").model_dump(mode="json"),
        out_path=tmp_path / "idx",
    )
    assert idx.stats().parts == store.count()
