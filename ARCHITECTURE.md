# Architecture

## Why retrieval, not classification

A 700,000-class softmax is the wrong tool: most SKUs have one to three catalog
images, new SKUs appear weekly, and thousands of parts are visually identical
except for a dimension. An **embedding + nearest-neighbour** design handles all
three: a new SKU is one more row in the index (no retraining), the model only has
to learn *what makes two parts look alike*, and identical-looking parts collapse
into a *family* that is reported honestly instead of guessed.

## Pipeline

| Stage | Module | What happens |
|---|---|---|
| 1. Preprocess | `pipeline/preprocess.py` | decode, EXIF orientation, optional background removal (rembg), saliency crop (plane-fit foreground mask), pad to square with the photo's own border colour, resize. The **same** normalisation is applied to catalog images at index time so gallery and query vectors share a distribution. |
| 2. OCR | `pipeline/ocr.py` | easyocr reads text; a regex extracts McMaster-style part numbers (`91251A537`). A hit that exists in the catalog is injected into the candidate pool and yields an `exact` tier. |
| 3. Embed | `models/` | backbone → L2-normalised vector. Query side uses TTA (4 rotations × flip = 8 vectors). |
| 4. Retrieve | `index/`, `pipeline/retrieve.py` | every TTA vector is searched; a part keeps its best score across variants and across its catalog images (multi-query retrieval). Category centroids stored in the index give a coarse prior. |
| 5. Rerank | `pipeline/rerank.py`, `pipeline/attributes.py` | **FusionReranker** combines similarity, category prior, multi-hit bonus, OCR evidence, attribute consistency, and (optionally) **ClaudeVisionReranker** output. The LLM sees the query photo plus the top-K catalog images/specs and returns a structured ranking and the attributes it can read from the photo. |
| 6. Calibrate | `pipeline/calibration.py` | softmax over fused scores with a temperature fitted on validation queries; rules map probability + margin + raw similarity to `exact / likely / candidate / unknown`. |

Latency budget on one CPU core with FAISS-HNSW over 700k × 512-d vectors: embed
~30 ms (ViT-B/16, batched TTA on GPU is ~5 ms), search < 5 ms, fusion < 1 ms. The
LLM reranker adds 2–6 s and is meant for the "confirm before ordering" path.

## Models

* `HashBackbone` – numpy descriptor. Foreground mask from a plane-fit background
  model (fitted on corner patches with a robust re-fit, morphological closing,
  largest blob), rotation-invariant polar-FFT ring signatures of the grayscale
  image and silhouette, anisotropy-weighted oriented thumbnails, chromaticity
  histogram, Hu moments, and a gradient-orientation spectrum. Feature-group
  weights were tuned with `scripts/tune_hash_weights.py`. Dev/CI only; its known
  weak spots are cast shadows merging into the silhouette and low-contrast parts
  on similar backgrounds.
* `OpenCLIPBackbone` – CLIP / SigLIP image tower. Best zero-shot starting point;
  its text tower can be used later for text-to-part search.
* `DINOv2Backbone` – strongest off-the-shelf instance-retrieval features.

All three implement `embed(images) -> (N, dim)`; nothing downstream knows which
one is loaded. A fine-tuned checkpoint adds a `ProjectionHead` (512-d).

## Measured retrieval quality (synthetic catalog, photo-style queries)

800-part, 39-family synthetic catalog. Queries = evaluation-preset augmentations
(backgrounds, shadows, perspective, colour shifts, noise, JPEG) of 73 parts from
*held-out families* plus 100 training parts; the gallery holds all 800 parts.
`ga` = extra photo-style rows indexed per catalog image.

| backbone | ga | Recall@1 | Recall@5 | Recall@10 | Recall@50 | MRR | ms/query |
|---|---|---|---|---|---|---|---|
| hash | 0 | 0.13 | 0.34 | 0.39 | 0.62 | 0.22 | 135 |
| hash | 2 | 0.17 | 0.39 | 0.47 | 0.70 | 0.27 | 140 |
| tinycnn (24 epochs) | 0 | 0.12 | 0.37 | 0.59 | 0.95 | 0.26 | 75 |
| tinycnn (24 epochs) | 2 | 0.13 | 0.46 | 0.71 | 0.97 | 0.29 | 78 |
| ensemble tinycnn(24ep)+hash (1:1) | 2 | 0.20 | 0.57 | 0.71 | 0.98 | 0.36 | 203 |
| tinycnn (42 epochs, shipped `assets/tinycnn_synthetic.pt`) | 2 | **0.29** | **0.74** | 0.86 | **1.00** | **0.48** | 92 |
| tinycnn (42 epochs) + query expansion k=3 | 2 | 0.28 | 0.72 | **0.87** | 1.00 | 0.47 | 92 |
| ensemble tinycnn(42ep)+hash (1:1) | 2 | 0.27 | 0.73 | 0.87 | 1.00 | 0.47 | 200 |
| ensemble tinycnn(42ep)+hash (1:0.5) | 2 | 0.31 | 0.74 | **0.88** | 1.00 | 0.49 | 200 |
| **ensemble tinycnn(42ep)+hash (1:0.3), default** | 2 | **0.32** | **0.74** | **0.88** | **1.00** | **0.50** | 200 |

On a *fresh* 200-part synthetic catalog (parts never seen in training, queries =
augmented photos of every part) the shipped model reaches Recall@1 0.58,
Recall@5 0.97, Recall@10 1.0, and the calibrated tiers are usable: `exact`
precision 1.0, `likely` 0.94 (`mcv demo --parts 200 --backbone tinycnn`).

The learned model was trained from scratch on 4 CPU cores (`configs/train_tinycnn.yaml`:
cached views, SupCon + classification, hard negatives); 24 epochs took ~35 min,
42 epochs ~2 h. Longer training kept improving held-out recall, so a GPU run or
the CLIP/DINOv2 backbones are the next step for real photos. Once the learned
model is strong the hand-crafted descriptor only helps at a small weight (1:0.3).

## Training (`training/train.py`)

* **Objective**: supervised contrastive (SupCon) over SKU labels with two
  augmented views per catalog image; optional ArcFace over *family* labels as an
  auxiliary head (family, not SKU, keeps the classifier matrix small).
* **Augmentation** (`data/augment.py`): random background texture, shadow,
  rotation, perspective, scale, colour temperature, blur, sensor noise, JPEG,
  occlusion. Closes the studio-image → phone-photo gap.
* **Hard negatives** (`training/mining.py`): after each epoch the gallery is
  re-embedded, each SKU's nearest *other-family* SKUs are found, and batches are
  built from anchor + confusers.
* **Curriculum**: augmentation strength is blended from the mild evaluation
  preset to the full training preset over the first epochs.
* **Learning rate**: from-scratch nets stall at chance with AdamW above ~1e-3
  (embeddings stay collapsed); `configs/train_tinycnn.yaml` uses 5e-4.
* **Split** (`data/splits.py`): by family hash, so near-duplicates never leak.
* **Validation**: Recall@1 with augmented queries against a held-out gallery.

## Intake and operations

```
drop (images / folder tree / JSONL / CSV / URL list)
  -> mcv validate      (report: missing, corrupt, duplicate, tiny, coverage)
  -> mcv bootstrap     ingest + normalise (EXIF, RGB, <=1024 px, de-dupe)
                       -> embed (N worker processes) -> index (numpy | FAISS auto)
                       -> evaluate + calibrate -> data/manifest.json
  -> mcv serve         GET /status, POST /admin/reload after rebuilds
  -> mcv enrich        names / categories / specs from McMaster pages (image-only drops)
  -> mcv build-index --only-new   incremental additions
```

`RUNBOOK.md` is the operator's guide. Everything in this diagram is exercised by
the test suite against the synthetic catalog; the only missing input is real
imagery.

## Scale test (20k synthetic parts, 60k images, 4 CPU cores)

| stage | measured |
|---|---|
| render 60k images | 260 s |
| ingest 20k parts into SQLite + FTS | 42 s; keyword query 2.5 ms |
| embed + exact index, TinyCNN, one process | 839 s (72 img/s incl. crop/normalise) |
| FAISS HNSW build from the vectors | 3.3 s; 48 MB on disk |
| query latency, exact numpy (60k x 128-d) | p50 111 ms / p95 130 ms |
| query latency, HNSW | p50 34 ms / p95 41 ms, identical Recall@K, 88% top-50 overlap |

Retrieval at this size: SKU Recall@1 0.01 / @10 0.20 / @50 0.59, **family**
Recall@1 0.33 / @10 0.58 / @50 0.89. The synthetic generator only has 495
kind x material families, so 20k parts means ~40 visually identical renders per
family; SKU-level recall is bounded by that ambiguity, not by the pipeline. Real
catalogs have the same structure for length / thread variants, which is exactly
why the result carries a family answer with the attributes that resolve it. The
shipped model also saw only 800 parts in training; retrain on the full catalog
(`mcv train`) before judging SKU-level numbers at scale.

## Catalog scale

* SQLite store, JSON attributes, FTS5 keyword search; 700k rows ≈ 300 MB.
* Index: FAISS HNSW (M=32) for ≤ 2M vectors – ~3 GB RAM for 2.1M × 512 float32;
  switch `FaissIndex(kind="ivfpq")` to shrink to ~150 MB at a small recall cost.
* Index build is streaming (batches of 256 images) so memory is flat; embedding
  2.1M images at 1,000 img/s on one GPU takes ~35 min.

## Data sourcing

McMaster-Carr's catalog is proprietary and scraping violates their terms. The
`catalog/sources.py` adapters expect exports you are licensed to use:
`JSONLSource`, `CSVSource`, `DirectorySource`, and a `McMasterApiSource` stub for
the account-holder Product Information API. The synthetic renderer in
`data/synthetic.py` exists so the whole system runs end to end without any of it.

## Feedback loop (how accuracy improves in use)

```
photo -> /identify -> user taps "This is it" -> /feedback
      -> data/queries/<part_number>/<request_id>.jpg  (+ feedback.jsonl)
      -> mcv evaluate --query-dir data/queries      (real-photo Recall@K, calibration)
      -> mcv train --query-dir data/queries         (real photos become training views)
      -> mcv build-index                            (new checkpoint, same catalog)
```

The synthetic renderer bootstraps the model; confirmed photos are the only data
that closes the synthetic-to-real gap, so the UI makes confirming a one-tap
action and "None of these" photos are kept under `_unknown/` for labelling.
`/feedback/stats` exposes the confirmed top-1 rate as the live accuracy metric.

## Multi-photo queries and family answers

Several photos of one part (different angles) are embedded independently and
their TTA variants are stacked into one multi-query; a catalog part keeps its
best similarity over every (photo, variant, catalog image) triple. When the top
candidates are members of one family (same geometry, different length / thread
/ size), the result carries a `family` block listing the members and the
attributes whose values differ, so the UI can ask exactly the one question that
resolves the SKU instead of guessing.

## Answer semantics

| tier | meaning | suggested action |
|---|---|---|
| `exact` | OCR read the part number, or one candidate dominates with a strong visual match | auto-fill |
| `likely` | confident top candidate, some ambiguity | show top-3, preselect best |
| `candidate` | plausible matches, needs a human | show top-5 with reasons |
| `unknown` | nothing in the catalog resembles the photo | ask for another angle / manual search |

Within a *family* (same geometry, different length/thread pitch) the visual model
cannot tell SKUs apart; the reranker's extracted attributes and the family grouping
let the UI ask exactly the one question that resolves it ("what length?").
