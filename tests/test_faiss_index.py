from __future__ import annotations

import numpy as np
import pytest

from mcmaster_vision.index import load_index
from mcmaster_vision.models.backbone import l2_normalize

faiss = pytest.importorskip("faiss")


@pytest.mark.parametrize("kind", ["flat", "hnsw"])
def test_faiss_index_roundtrip(tmp_path, kind):
    from mcmaster_vision.index.faiss_index import FaissIndex

    rng = np.random.default_rng(0)
    vecs = l2_normalize(rng.normal(size=(200, 32)))
    idx = FaissIndex(32, kind=kind)
    idx.add([f"P{i}" for i in range(200)], vecs)
    assert idx.search_ids(vecs[17], 3)[0][0] == "P17"
    idx.save(tmp_path / "f")
    loaded = load_index(tmp_path / "f")
    assert loaded.backend == "faiss" and len(loaded) == 200
    assert loaded.search_ids(vecs[17], 1)[0][0] == "P17"
    scores, rows = loaded.search(vecs[:2], 5)
    assert scores.shape == (2, 5) and rows[0, 0] == 0


def test_faiss_via_builder(store, embedder, tmp_path):
    from mcmaster_vision.index import build_index

    idx = build_index(store, embedder, "faiss", out_path=tmp_path / "fidx")
    assert idx.stats().backend == "faiss" and idx.stats().parts == store.count()
    part = next(store.iter_parts(with_images_only=True))
    from PIL import Image

    from mcmaster_vision.pipeline.preprocess import preprocess_catalog

    q = embedder.embed_catalog([preprocess_catalog(Image.open(part.image_paths[0]))])[0]
    assert load_index(tmp_path / "fidx").search_ids(q, 1)[0][0] == part.part_number
