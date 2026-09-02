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
| 1. Preprocess | `pipeline/preprocess.py` | decode, EXIF orientation, optional background removal (rembg), saliency crop, pad to square, resize. The **same** normalisation is applied to catalog images at index time so gallery and query vectors share a distribution. |
| 2. OCR | `pipeline/ocr.py` | easyocr reads text; a regex extracts McMaster-style part numbers (`91251A537`). A hit that exists in the catalog is injected into the candidate pool and yields an `exact` tier. |
| 3. Embed | `models/` | backbone → L2-normalised vector. Query side uses TTA (4 rotations × flip = 8 vectors). |
| 4. Retrieve | `index/`, `pipeline/retrieve.py` | every TTA vector is searched; a part keeps its best score across variants and across its catalog images (multi-query retrieval). Category centroids stored in the index give a coarse prior. |
| 5. Rerank | `pipeline/rerank.py`, `pipeline/attributes.py` | **FusionReranker** combines similarity, category prior, multi-hit bonus, OCR evidence, attribute consistency, and (optionally) **ClaudeVisionReranker** output. The LLM sees the query photo plus the top-K catalog images/specs and returns a structured ranking and the attributes it can read from the photo. |
| 6. Calibrate | `pipeline/calibration.py` | softmax over fused scores with a temperature fitted on validation queries; rules map probability + margin + raw similarity to `exact / likely / candidate / unknown`. |

Latency budget on one CPU core with FAISS-HNSW over 700k × 512-d vectors: embed
~30 ms (ViT-B/16, batched TTA on GPU is ~5 ms), search < 5 ms, fusion < 1 ms. The
LLM reranker adds 2–6 s and is meant for the "confirm before ordering" path.

## Models

* `HashBackbone` – numpy descriptor (orientation-normalised silhouette, colour,
  Hu moments, radial profiles, HOG-lite). Dev/CI only.
* `OpenCLIPBackbone` – CLIP / SigLIP image tower. Best zero-shot starting point;
  its text tower can be used later for text-to-part search.
* `DINOv2Backbone` – strongest off-the-shelf instance-retrieval features.

All three implement `embed(images) -> (N, dim)`; nothing downstream knows which
one is loaded. A fine-tuned checkpoint adds a `ProjectionHead` (512-d).

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
* **Split** (`data/splits.py`): by family hash, so near-duplicates never leak.
* **Validation**: Recall@1 with augmented queries against a held-out gallery.

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
