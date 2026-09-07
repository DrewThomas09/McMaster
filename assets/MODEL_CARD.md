# Model card: `tinycnn_synthetic.pt`

**What it is.** A 1.6M-parameter residual CNN (`models/tinycnn.py`: width 24,
GeM pooling, 128-d embedding, 96 px input) that maps a part photo to a vector
for nearest-neighbour retrieval against catalog images.

**Training data.** 8,000 synthetic parts (the first 8,000 of a 20,000-part
render from `data/synthetic.py`, seed 7; 39 hardware families, 14 materials),
3 rendered views each, with photo-style augmentation (backgrounds, shadows,
rotation, perspective, colour shifts, blur, noise, JPEG). No McMaster-Carr
imagery and no real photographs were used.

**Recipe.** `configs/train_tinycnn.yaml`: cached augmented views (3 per image,
refreshed every 8 epochs), supervised-contrastive loss over SKU labels plus a
classification head over parts, hard-negative batches, AdamW 5e-4 with warmup
and cosine decay, 24 epochs on 4 CPU cores (2.2 h, ~3 GB RAM).

**Measured.** 800-part catalog, queries from held-out families:
Recall@1 0.33, Recall@5 0.72, Recall@10 0.88, Recall@50 1.00, MRR 0.50 (alone;
the previous 800-part model scored 0.29 / 0.74 / 0.86 / 1.00 / 0.48).
20k-part catalog (~40 look-alikes per family), parts never seen in training:
SKU Recall@10 0.20 / @50 0.57, family Recall@1 0.42 / @10 0.67.

**Intended use.** Bootstrapping and demonstrating the pipeline offline; a
starting point to fine-tune on real catalog images and confirmed photos
(`mcv train --query-dir data/queries`, `mcv retrain`).

**Limitations.** Knows rendered shapes, not real materials, lighting, or wear;
cannot separate SKUs that differ only in a non-visual dimension (use the
family answer and attribute constraints); trained on synthetic renders of 39
families, so it does not cover the breadth of a 700k-SKU catalog without
retraining on real imagery.


## Re-evaluation with the per-category breakdown (2026-09-07)

Fresh 800-part synthetic catalog (seed 4242, 3 renders per part, gallery augmentation 2),
400 photo-style queries, TinyCNN checkpoint `tinycnn_synthetic.pt`, default calibration:

| metric | value |
|---|---|
| Recall@1 / @5 / @10 / @50 | 0.28 / 0.73 / 0.87 / 1.00 |
| Family Recall@1 / @5 | 0.49 / 0.79 |
| MRR | 0.48 |

By top-level category (Recall@1 / Recall@5 / queries): Sealing 0.11 / 0.56 / 9;
Pipe, Tubing, Hose & Fittings 0.22 / 1.00 / 37; Fastening & Joining 0.25 / 0.65 / 220;
Hardware 0.29 / 0.60 / 48; Power Transmission 0.37 / 0.91 / 67; Hand Tools 0.55 / 0.73 / 11.
Fasteners (the largest group and the one with the most look-alikes) are where the
size question, the coin hint and thread pitch matter most; the earlier 0.33 Recall@1
came from a different seed and is within the spread of these small held-out sets.

Same catalog seed after the renderer fix (every kind's second view is now a distinct
rotation instead of a duplicate of the first; long parts stay on the canvas): Recall@1 /
@5 / @10 = 0.30 / 0.73 / 0.87, Family Recall@1 0.50, MRR 0.49. Weakest categories:
Sealing 0.11 (9 queries), Fastening & Joining 0.24 (220), Hardware 0.31 (48).


## Retrained on the fixed renderer (2026-09-07, shipped)

Same recipe (`configs/train_tinycnn.yaml`: lr 5e-4, 24 epochs, 8,000 parts of seed 7,
3 views each) on the renderer after the second-view fix (every kind's second view is a
distinct rotation or top view; long parts stay on the canvas; 44 families including
the pipe nipples, couplings, flanges, caps and bushings). Best epoch 18 (validation
Recall@1 0.204 on the 8k-part split); 4 CPU cores, 7.6 h with concurrent load.

Head to head on the fresh held-out 800-part catalog (seed 4242, 400 photo-style queries,
`scripts/compare_checkpoints.py`):

| checkpoint | R@1 | R@5 | R@10 | family R@1 | MRR |
|---|---|---|---|---|---|
| previous `tinycnn_synthetic.pt` | 0.302 | 0.698 | 0.858 | 0.490 | 0.479 |
| **this one (shipped)** | **0.468** | **0.863** | **0.968** | **0.645** | **0.628** |

The gain comes from training views that finally differ per part (the duplicate second
view had been teaching the model that two identical images are two views), not from a
recipe change.
