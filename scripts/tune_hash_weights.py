"""Tune HashBackbone feature-group weights on synthetic photo-style queries.

Usage: python scripts/tune_hash_weights.py [iterations]
Prints per-group retrieval power and the tuned weight dict to paste into
HashBackbone.DEFAULT_WEIGHTS.
"""

import json
import pathlib
import sys
import tempfile

import numpy as np
from PIL import Image

from mcmaster_vision.data import SyntheticCatalog
from mcmaster_vision.data.augment import AugmentConfig, PhotoAugmenter
from mcmaster_vision.models import HashBackbone
from mcmaster_vision.pipeline.preprocess import preprocess, preprocess_catalog

d = pathlib.Path(tempfile.mkdtemp())
parts = list(SyntheticCatalog(120, 2, seed=11).generate(d))
bb = HashBackbone()
gal = [
    bb.feature_groups(preprocess_catalog(Image.open(p.image_paths[i])))
    for p in parts
    for i in range(2)
]
gal_pn = [p.part_number for p in parts for i in range(2)]
off = dict(
    background_prob=0,
    shadow_prob=0,
    occlusion_prob=0,
    blur_prob=0,
    noise_prob=0,
    jpeg_prob=0,
    perspective=0,
    brightness=(1, 1),
    contrast=(1, 1),
    color_temp=0,
    rotate_deg=0,
    scale_range=(1, 1),
    translate_frac=0,
    grayscale_prob=0,
)
conds = {
    "none": AugmentConfig(**off),
    "rotate": AugmentConfig(**{**off, "rotate_deg": 180}),
    "persp": AugmentConfig(**{**off, "perspective": 0.12}),
    "color": AugmentConfig(
        **{**off, "brightness": (0.6, 1.4), "contrast": (0.7, 1.3), "color_temp": 0.12}
    ),
    "bg+shadow": AugmentConfig(**{**off, "background_prob": 1.0, "shadow_prob": 1.0}),
    "EVAL": AugmentConfig.evaluation(),
}
qparts = parts[:60]
queries = {}
for cname, cfg in conds.items():
    aug = PhotoAugmenter(cfg, seed=5)
    qs = []
    for p in qparts:
        img = aug(Image.open(p.image_paths[0]), out_size=224)
        variants = [img.rotate(r, expand=True) for r in (0, 90, 180, 270)]
        variants += [v.transpose(Image.Transpose.FLIP_LEFT_RIGHT) for v in variants]
        qs.append([bb.feature_groups(preprocess(v)) for v in variants])
    queries[cname] = qs


def norm(v):
    return v / (np.linalg.norm(v) + 1e-6)


def vec(g, w):
    return norm(np.concatenate([w.get(n, 0) * norm(g[n]) for n in bb.GROUP_ORDER]))


def eval_weights(w):
    G = np.stack([vec(g, w) for g in gal])
    out = {}
    for cname, qs in queries.items():
        r1 = r10 = 0
        for p, variants in zip(qparts, qs):
            Q = np.stack([vec(g, w) for g in variants])
            sims = (Q @ G.T).max(axis=0)
            best = {}
            for pn, sv in zip(gal_pn, sims):
                best[pn] = max(best.get(pn, -9), sv)
            ranked = sorted(best, key=lambda k: -best[k])
            r1 += ranked[0] == p.part_number
            r10 += p.part_number in ranked[:10]
        out[cname] = (r1 / len(qparts), r10 / len(qparts))
    return out


print("per-group (R@1/R@10):")
for n in bb.GROUP_ORDER:
    res = eval_weights({n: 1.0})
    print(f"  {n:12s} " + "  ".join(f"{c}={a:.2f}/{b:.2f}" for c, (a, b) in res.items()))
print("default:", {c: f"{a:.2f}/{b:.2f}" for c, (a, b) in eval_weights(bb.DEFAULT_WEIGHTS).items()})
w = dict(bb.DEFAULT_WEIGHTS)


def objective(w):
    r = eval_weights(w)
    return sum(0.6 * a + 0.4 * b for a, b in r.values()) / len(r)


best = objective(w)
for it in range(int(sys.argv[1]) if len(sys.argv) > 1 else 2):
    for n in bb.GROUP_ORDER:
        for cand in (0.0, 0.25, 0.5, 1.0, 1.5, 2.5):
            w2 = dict(w)
            w2[n] = cand
            o = objective(w2)
            if o > best + 1e-6:
                best, w = o, w2
    print("iter", it, "obj", round(best, 3))
print("tuned:", {c: f"{a:.2f}/{b:.2f}" for c, (a, b) in eval_weights(w).items()})
print("WEIGHTS", json.dumps(w))
