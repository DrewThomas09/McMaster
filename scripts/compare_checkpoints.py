"""Compare TinyCNN checkpoints on a fresh held-out synthetic catalog.

Usage: python scripts/compare_checkpoints.py CKPT [CKPT ...] [--parts 800] [--queries 400] [--seed 4242]

Builds one catalog, indexes it once per checkpoint, evaluates photo-style queries and
prints Recall@K, family Recall@K and MRR side by side, so a retrained model is only
shipped when it beats the current one on parts neither has seen.
"""

from __future__ import annotations

import argparse
import pathlib
import tempfile

from mcmaster_vision.catalog import CatalogStore, ingest
from mcmaster_vision.config import Settings
from mcmaster_vision.data import SyntheticCatalog
from mcmaster_vision.index import build_index
from mcmaster_vision.models import PartEmbedder, load_backbone
from mcmaster_vision.pipeline import Identifier
from mcmaster_vision.training import evaluate_retrieval


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--parts", type=int, default=800)
    ap.add_argument("--queries", type=int, default=400)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--image-size", type=int, default=96)
    ap.add_argument("--gallery-augment", type=int, default=2)
    args = ap.parse_args()
    d = pathlib.Path(tempfile.mkdtemp(prefix="mcv-compare-"))
    SyntheticCatalog(n_parts=args.parts, images_per_part=3, seed=args.seed).write_jsonl(
        d / "img", d / "p.jsonl"
    )
    store = CatalogStore(d / "c.sqlite")
    ingest(d / "p.jsonl", store)
    rows = []
    for ckpt in args.checkpoints:
        s = Settings(
            backbone="tinycnn",
            backbone_checkpoint=pathlib.Path(ckpt),
            backbone_pretrained="none",
            image_size=args.image_size,
        )
        emb = PartEmbedder(load_backbone(s))
        idx = build_index(
            store, emb, "numpy", image_size=args.image_size, gallery_augment=args.gallery_augment
        )
        rep = evaluate_retrieval(
            Identifier(store, idx, emb, top_k=50, image_size=args.image_size),
            store,
            max_queries=args.queries,
        )
        rows.append((ckpt, rep))
        print(
            f"{pathlib.Path(ckpt).name:28s} R@1 {rep.recall_at[1]:.3f} R@5 {rep.recall_at[5]:.3f} "
            f"R@10 {rep.recall_at[10]:.3f} famR@1 {rep.family_recall_at[1]:.3f} MRR {rep.mrr:.3f}",
            flush=True,
        )
    best = max(rows, key=lambda r: (r[1].recall_at[1], r[1].mrr))
    print(f"best: {best[0]}")


if __name__ == "__main__":
    main()
