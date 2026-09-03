from __future__ import annotations

import numpy as np

from mcmaster_vision.index import NumpyIndex, build_index
from mcmaster_vision.models.backbone import l2_normalize
from mcmaster_vision.pipeline.retrieve import Retriever


def _clustered_index(rng, clusters=3, per=10, dim=16):
    centres = l2_normalize(rng.normal(size=(clusters, dim)))
    vecs, ids = [], []
    for c in range(clusters):
        for i in range(per):
            vecs.append(centres[c] + 0.15 * rng.normal(size=dim))
            ids.append(f"C{c}_{i}")
    idx = NumpyIndex(dim)
    idx.add(ids, l2_normalize(np.stack(vecs)))
    return idx, centres


def test_query_expansion_keeps_query_in_its_cluster():
    rng = np.random.default_rng(1)
    idx, centres = _clustered_index(rng)
    q = l2_normalize(centres[0] + 0.3 * rng.normal(size=16))
    r = Retriever(idx, top_k=5, qe_k=3)
    expanded = r.expand_queries(q[None, :])
    assert expanded.shape == (1, 16)
    assert np.allclose(np.linalg.norm(expanded, axis=1), 1.0, atol=1e-5)
    assert not np.allclose(expanded[0], q)
    # the expanded query is at least as close to the cluster centre as the raw one
    assert float(expanded[0] @ centres[0]) >= float(q @ centres[0]) - 1e-3
    assert all(h.part_number.startswith("C0_") for h in r.retrieve(q))
    assert Retriever(idx, top_k=5, qe_k=0).retrieve(q)[0].part_number.startswith("C0_")


def test_gallery_augmentation_multiplies_index_rows(store, embedder, tmp_path):
    base = build_index(store, embedder, "numpy")
    aug = build_index(store, embedder, "numpy", gallery_augment=1, out_path=tmp_path / "ga")
    assert len(aug) == 2 * len(base)
    assert aug.stats().parts == base.stats().parts
    assert aug.meta["gallery_augment"] == 1
