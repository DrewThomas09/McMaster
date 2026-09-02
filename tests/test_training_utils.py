from __future__ import annotations

import numpy as np

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
