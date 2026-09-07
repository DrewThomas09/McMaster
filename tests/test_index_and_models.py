from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from mcmaster_vision.index import NumpyIndex, load_index
from mcmaster_vision.models import HashBackbone, PartEmbedder
from mcmaster_vision.models.backbone import l2_normalize


def test_hash_backbone_is_normalised_and_deterministic():
    bb = HashBackbone()
    img = Image.new("RGB", (100, 80), (200, 30, 30))
    v1, v2 = bb.embed([img, img])
    assert v1.shape == (bb.dim,)
    assert np.allclose(np.linalg.norm(v1), 1.0, atol=1e-5)
    assert np.allclose(v1, v2)
    assert bb.embed([]).shape == (0, bb.dim)


def test_embedder_tta_query_is_normalised():
    emb = PartEmbedder(HashBackbone())
    img = Image.new("RGB", (64, 64), (10, 200, 10))
    img.paste(Image.new("RGB", (20, 30), (200, 20, 20)), (10, 10))
    q = emb.embed_query(img)
    assert q.shape == (8, emb.dim)
    assert np.allclose(np.linalg.norm(q, axis=1), 1.0, atol=1e-5)
    assert np.allclose(np.linalg.norm(PartEmbedder.pooled(q)), 1.0, atol=1e-5)
    assert emb.embed_query(img, tta=False).shape == (1, emb.dim)


def test_numpy_index_search_and_persistence(tmp_path):
    rng = np.random.default_rng(0)
    vecs = l2_normalize(rng.normal(size=(50, 16)))
    idx = NumpyIndex(16)
    idx.add([f"P{i}" for i in range(50)], vecs)
    scores, rows = idx.search(vecs[:3], 5)
    assert rows[:, 0].tolist() == [0, 1, 2]
    assert np.allclose(scores[:, 0], 1.0, atol=1e-5)
    hits = idx.search_ids(vecs[7], 3)
    assert hits[0][0] == "P7"

    idx.set_categories(["a", "b"], l2_normalize(rng.normal(size=(2, 16))))
    idx.save(tmp_path / "idx")
    loaded = load_index(tmp_path / "idx")
    assert len(loaded) == 50 and loaded.dim == 16
    assert loaded.search_ids(vecs[7], 1)[0][0] == "P7"
    assert set(loaded.category_scores(vecs[0])) == {"a", "b"}
    assert loaded.stats().parts == 50


def test_numpy_index_rejects_bad_shapes():
    idx = NumpyIndex(4)
    with pytest.raises(ValueError):
        idx.add(["a"], np.zeros((1, 5), np.float32))
    with pytest.raises(ValueError):
        idx.add(["a", "b"], np.zeros((1, 4), np.float32))
    assert idx.search(np.zeros((1, 4), np.float32), 3)[1].shape == (1, 0)


def test_built_index_has_category_centroids(index, store):
    st = index.stats()
    assert st.parts == store.count()
    assert st.vectors == 80
    assert index.category_centroids is not None and len(index.category_names) >= 2


def test_index_save_is_atomic_swap(tmp_path):
    rng = np.random.default_rng(0)
    idx = NumpyIndex(4)
    idx.add(["a", "b"], l2_normalize(rng.normal(size=(2, 4))))
    target = tmp_path / "parts"
    idx.save(target)
    assert (target / "meta.json").exists()
    idx.add(["c"], l2_normalize(rng.normal(size=(1, 4))))
    idx.save(target)  # overwrite via swap
    assert len(load_index(target)) == 3
    assert not list(tmp_path.glob("parts.tmp-*")) and not (tmp_path / "parts.old").exists()


def test_hash_thumbnail_weight_actually_applies():
    """The anisotropy weight scales the oriented thumbnails after normalisation."""
    import numpy as np
    from PIL import Image, ImageDraw

    from mcmaster_vision.models import HashBackbone

    hb = HashBackbone()
    disc = Image.new("RGB", (200, 200), "white")
    ImageDraw.Draw(disc).ellipse((40, 40, 160, 160), fill=(80, 80, 85))
    rod = Image.new("RGB", (200, 200), "white")
    ImageDraw.Draw(rod).rectangle((20, 90, 180, 110), fill=(80, 80, 85))
    sizes = {}
    for name in hb.GROUP_ORDER:
        sizes[name] = len(hb.feature_groups(disc)[name])
    off = 0
    span = {}
    for name in hb.GROUP_ORDER:
        span[name] = (off, off + sizes[name])
        off += sizes[name]
    v_disc = hb.embed([disc])[0]
    v_rod = hb.embed([rod])[0]
    a, b = span["thumb_gray"]
    # a round part's thumbnail block must carry clearly less energy than a rod's
    assert np.linalg.norm(v_disc[a:b]) < 0.6 * np.linalg.norm(v_rod[a:b])
