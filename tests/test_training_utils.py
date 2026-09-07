from __future__ import annotations

import numpy as np
import pytest

from mcmaster_vision.data import split_by_family
from mcmaster_vision.models.backbone import l2_normalize
from mcmaster_vision.schemas import Part
from mcmaster_vision.training.mining import hard_batch_sampler, mine_hard_negatives
from mcmaster_vision.training.train import DEFAULTS, load_train_config


def _parts(n):
    return [Part(part_number=f"P{i}", name="x", family_id=f"F{i % 3}") for i in range(n)]


def test_split_by_family_is_disjoint_and_stable():
    parts = _parts(60)
    tr, va = split_by_family(parts, 0.34)
    assert len(tr) + len(va) == 60
    assert not ({p.family_id for p in tr} & {p.family_id for p in va})
    tr2, _ = split_by_family(parts, 0.34)
    assert [p.part_number for p in tr] == [p.part_number for p in tr2]


def test_hard_negatives_exclude_same_family():
    parts = _parts(12)
    emb = l2_normalize(np.random.default_rng(0).normal(size=(12, 8)))
    negs = mine_hard_negatives(parts, emb, per_part=3)
    fam = {p.part_number: p.family_id for p in parts}
    for pn, ns in negs.items():
        assert len(ns) == 3 and all(fam[n] != fam[pn] for n in ns)
    batch = next(hard_batch_sampler(parts, negs, batch_parts=8))
    assert len(batch) == 8 and len(set(batch)) == 8
    assert mine_hard_negatives([], np.zeros((0, 8))) == {}


def test_train_config_defaults(tmp_path):
    cfg = load_train_config(None)
    assert cfg == DEFAULTS
    p = tmp_path / "t.yaml"
    p.write_text("epochs: 2\nloss: arcface\n")
    cfg = load_train_config(p)
    assert (
        cfg["epochs"] == 2
        and cfg["loss"] == "arcface"
        and cfg["batch_size"] == DEFAULTS["batch_size"]
    )


def test_hard_negatives_chunked_matches_dense_and_sampler_yields_tail():
    parts = _parts(50)
    emb = l2_normalize(np.random.default_rng(3).normal(size=(50, 16)))
    a = mine_hard_negatives(parts, emb, per_part=3, chunk=7)
    b = mine_hard_negatives(parts, emb, per_part=3, chunk=1000)
    assert a == b and all(len(v) == 3 for v in a.values())
    # fewer parts than the batch size: the sampler must still yield (the trailing batch)
    batches = hard_batch_sampler(parts[:5], {}, batch_parts=64)
    assert len(next(batches)) == 5


def test_supcon_loss_is_fp16_safe():
    torch = pytest.importorskip("torch")
    from mcmaster_vision.models.losses import supcon_loss

    feats = torch.nn.functional.normalize(torch.randn(8, 2, 16), dim=-1).half()
    loss = supcon_loss(feats, torch.arange(8), 0.07)
    assert torch.isfinite(loss)


def test_evaluation_breaks_down_by_category_and_lists_misses(identifier, store):
    from mcmaster_vision.training import evaluate_retrieval

    rep = evaluate_retrieval(identifier, store, max_queries=12)
    assert rep.queries == 12
    assert rep.by_category and all(
        0 <= v["recall_1"] <= v["recall_5"] <= 1 for v in rep.by_category.values()
    )
    assert sum(v["queries"] for v in rep.by_category.values()) == 12
    # weakest category first
    r1 = [v["recall_1"] for v in rep.by_category.values()]
    assert r1 == sorted(r1)
    for m in rep.hardest:
        assert m["truth"] and m["rank"] != 1
    import json

    d = json.loads(rep.to_json())
    assert "by_category" in d and "hardest" in d and "score_lists" not in d


def test_training_review_regressions(tmp_path):
    import pickle

    from PIL import Image

    from mcmaster_vision.catalog.web import family_key
    from mcmaster_vision.cli import _split_feedback
    from mcmaster_vision.data.augment import PhotoAugmenter
    from mcmaster_vision.data.splits import split_by_family
    from mcmaster_vision.schemas import Part

    # a transparent PNG composites on white, never on the hidden black RGB
    rgba = Image.new("RGBA", (120, 120), (0, 0, 0, 0))
    for x in range(40, 80):
        for y in range(40, 80):
            rgba.putpixel((x, y), (60, 60, 65, 255))
    out = PhotoAugmenter(seed=0)(rgba, out_size=160).convert("RGB")
    corner = out.getpixel((3, 3))
    assert min(corner) > 120, corner  # workbench-light, not black

    # web imports share a family across sizes; splits fall back to the category
    a = family_key(["Pipe Fittings", "Elbows"], 'Type 304 Stainless Steel 90° Elbow, 3/8" NPT')
    b = family_key(["Pipe Fittings", "Elbows"], 'Type 304 Stainless Steel 90° Elbow, 1/2" NPT')
    c = family_key(["Pipe Fittings", "Elbows"], "Type 316 Stainless Steel 90° Elbow, M6 x 1")
    assert a == b and a != c
    parts = [
        Part(part_number=f"P{i}", name="x", category_path=["Cat", "Sub"], family_id=None)
        for i in range(30)
    ]
    train, val = split_by_family(parts, 0.3)
    assert not train or not val  # one category -> one side, never leaked across

    # hold-out: one photo from every part with 2+, none from singletons
    extra, held = _split_feedback(
        {"A": ["a1"], "B": ["b1", "b2"], "C": [f"c{i}" for i in range(7)]}
    )
    assert extra["A"] == ["a1"] and "A" not in dict(held)
    assert dict(held)["B"] == "b2" and extra["B"] == ["b1"]
    held_c = [p for pn, p in held if pn == "C"]
    assert "c6" in held_c and len(held_c) == 2 and len(extra["C"]) == 5

    # the dataset is picklable for spawn workers (when torch is present)
    torch = pytest.importorskip("torch")
    from mcmaster_vision.data.dataset import make_contrastive_dataset

    img = tmp_path / "p.png"
    Image.new("RGB", (32, 32), "gray").save(img)
    ds, _ = make_contrastive_dataset(
        [Part(part_number="Q", name="q", category_path=["c"], image_paths=[str(img)])],
        transform=lambda im: torch.zeros(3, 8, 8),
        views=2,
        image_size=32,
    )
    assert len(pickle.dumps(ds)) > 0 and ds[0][0].shape == (2, 3, 8, 8)


def test_view_cache_drops_unreadable_images(tmp_path):
    from PIL import Image

    from mcmaster_vision.schemas import Part
    from mcmaster_vision.training.cached import build_view_cache

    good = tmp_path / "good.png"
    Image.new("RGB", (64, 64), "gray").save(good)
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")
    parts = [
        Part(part_number="A", name="a", category_path=["c"], image_paths=[str(good), str(bad)]),
        Part(
            part_number="B", name="b", category_path=["c"], image_paths=[str(tmp_path / "gone.jpg")]
        ),
    ]
    x, owners = build_view_cache(parts, 2, 32, workers=1)
    assert x.shape == (2, 32, 32, 3) and owners.tolist() == [0, 0]  # only the readable image
