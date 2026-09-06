# Model card: `tinycnn_synthetic.pt`

**What it is.** A 1.6M-parameter residual CNN (`models/tinycnn.py`: width 24,
GeM pooling, 128-d embedding, 96 px input) that maps a part photo to a vector
for nearest-neighbour retrieval against catalog images.

**Training data.** 800 synthetic parts from `data/synthetic.py` (39 hardware
families, 14 materials, seed 2026), 3 rendered views each, with photo-style
augmentation (backgrounds, shadows, rotation, perspective, colour shifts, blur,
noise, JPEG). No McMaster-Carr imagery and no real photographs were used.

**Recipe.** `configs/train_tinycnn.yaml`: cached augmented views (10 per image,
refreshed every 8 epochs), supervised-contrastive loss over SKU labels plus a
classification head over parts, hard-negative batches, AdamW 5e-4 with warmup
and cosine decay, 42 epochs on 4 CPU cores (~2 h).

**Measured.** 800-part catalog, queries from held-out families:
Recall@1 0.29, Recall@5 0.74, Recall@10 0.86, Recall@50 1.00, MRR 0.48 (alone);
Recall@1 0.32 / MRR 0.50 in the 1:0.3 ensemble with the hash descriptor.
Fresh 200-part catalog: Recall@1 0.58, Recall@5 0.97. 20k-part catalog
(~40 look-alikes per family): SKU Recall@10 0.20, family Recall@10 0.58.

**Intended use.** Bootstrapping and demonstrating the pipeline offline; a
starting point to fine-tune on real catalog images and confirmed photos
(`mcv train --query-dir data/queries`, `mcv retrain`).

**Limitations.** Knows rendered shapes, not real materials, lighting, or wear;
cannot separate SKUs that differ only in a non-visual dimension (use the
family answer and attribute constraints); trained on 800 parts, so it does not
generalise to the full breadth of a 700k-SKU catalog without retraining.
