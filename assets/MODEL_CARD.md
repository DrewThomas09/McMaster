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
