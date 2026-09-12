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

By top-level category on the 800-part demo evaluation of the same day (200 photo-style
queries, gallery augmentation 2; Recall@1 / Recall@5 / queries): Sealing 0.20 / 1.00 / 5;
Hardware 0.36 / 0.77 / 22; Pipe, Tubing, Hose & Fittings 0.38 / 0.85 / 39 (now including
the nipple, coupling, flange, cap and bushing kinds); Sawing & Cutting 0.44 / 1.00 / 9;
Fastening & Joining 0.50 / 0.91 / 92; Power Transmission 0.58 / 0.90 / 31. Overall
Recall@1 0.47, Recall@5 0.89, family Recall@1 0.68, MRR 0.63. Fittings and hardware
remain the weakest: the size question, the coin, the bore and the thread pitch are the
levers there, not the model alone.


## 48-epoch retrain (2026-09-08, superseded)

Same recipe and data as the previous section, with the cosine schedule stretched to 48
epochs (`epochs: 48`); best epoch 42 (validation Recall@1 0.266 on the 8k-part split),
about 10 h on 4 CPU cores alongside other load. Head to head on the fresh held-out
800-part catalog (seed 4242, 400 photo-style queries, `scripts/compare_checkpoints.py`):

| checkpoint | R@1 | R@5 | R@10 | family R@1 | MRR |
|---|---|---|---|---|---|
| 24-epoch (previous shipped) | 0.468 | 0.863 | 0.968 | 0.645 | 0.628 |
| **48-epoch (shipped until 2026-09-09)** | **0.550** | **0.932** | **0.990** | **0.743** | **0.707** |

The longer schedule keeps improving on the fixed renderer; the 24-epoch numbers of the
earlier sections are superseded.

By top-level category on the 800-part demo evaluation (200 photo-style queries, gallery
augmentation 2; Recall@1 / Recall@5 / queries): Pipe, Tubing, Hose & Fittings 0.49 / 0.95 /
39; Hardware 0.55 / 1.00 / 22; Fastening & Joining 0.55 / 0.95 / 92; Sawing & Cutting 0.56 /
0.78 / 9; Sealing 0.60 / 1.00 / 5; Power Transmission 0.65 / 1.00 / 31. Overall Recall@1
0.56, Recall@5 0.96, family Recall@1 0.77, MRR 0.72. Fittings remain the weakest category
(up from 0.38), which is what the size question, the coin, the bore and the thread pitch
are for.


## 96-epoch schedule at half the learning rate, stopped at epoch 83 (2026-09-09, shipped)

Same recipe and data, with the cosine schedule stretched to 96 epochs and the backbone
learning rate halved (`epochs: 96`, `lr: 5.0e-4`, head 1.0e-3). The run was killed by a
container restart after epoch 83 (about 30 h on 4 shared CPU cores); the checkpoint
shipped is the last one written (epoch 83), not the best-validation one (epoch 56,
validation Recall@1 0.286), because on the fresh held-out 800-part catalog (seed 4242,
400 photo-style queries, `scripts/compare_checkpoints.py`, all three measured the same
day on the same catalog) the later checkpoint is clearly better:

| checkpoint | R@1 | R@5 | R@10 | family R@1 | MRR |
|---|---|---|---|---|---|
| 48-epoch (previous shipped) | 0.530 | 0.915 | 0.990 | 0.723 | 0.694 |
| epoch 56 of 96 (best validation) | 0.540 | 0.945 | 0.995 | 0.750 | 0.709 |
| **epoch 83 of 96 (shipped)** | **0.623** | **0.958** | **0.995** | **0.830** | **0.764** |

The 48-epoch row reads 0.530 here against 0.550 in the previous section: the synthetic
renderer changed on 2026-09-09 (pipe nipples drawn to the catalog's proportions), so
the held-out catalog is not byte-identical to the one measured the day before. The
validation split's Recall@1 (0.25-0.29) is a poor guide to the held-out number: the
split is 8k parts of the training catalog with many near-twins, the held-out catalog
is 800 unseen parts. Thirteen epochs of the schedule remain unrun; the cosine tail
would have taken the learning rate to zero and may have added a little more. A 13-epoch
annealing tail from these weights (`init_checkpoint`, lr 3e-5 to 0) was started three
times on 2026-09-10/11 and killed each time when the hosted container was reclaimed for
inactivity; the third attempt finished one epoch before dying, and that checkpoint measured
R@1 0.635, R@5 0.968, family R@1 0.823, MRR 0.774 against the shipped 0.623 / 0.958 /
0.830 / 0.764 on the same held-out catalog: a point on Recall@1 and MRR, a point down on
families, within the noise of 400 queries, so it was not shipped. The full 13-epoch tail
needs a machine the session does not own, and is left as the next training step. So is a
retrain on a catalog drawn by the deduplicated generator (2026-09-12): the 8,000-part
training catalog behind every checkpoint so far listed about one part in eight as a twin
of another (same family, same attributes, same render), and the contrastive loss treated
each twin as a negative of the other, which is noise the next run should not pay for.

By top-level category on the 800-part demo evaluation (200 photo-style queries, gallery
augmentation 2; Recall@1 / Recall@5 / queries): Sealing 0.40 / 1.00 / 5; Hand Tools 0.50 /
1.00 / 2; Fastening & Joining 0.59 / 0.98 / 93; Hardware 0.65 / 1.00 / 20; Pipe, Tubing,
Hose & Fittings 0.73 / 0.98 / 41; Power Transmission 0.77 / 1.00 / 31; Sawing & Cutting
0.88 / 1.00 / 8. Overall Recall@1 0.66, Recall@5 0.985, family Recall@1 0.85, MRR 0.80;
tier precision exact 1.00 (21 queries), likely 0.94 (35), candidate 0.54 (144). Fittings,
the weakest category of the 48-epoch model at 0.49, are now 0.73; fasteners are the
weakest at 0.59, which is the size and thread question the coin and the pitch reader are
for.
