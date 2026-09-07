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
| tinycnn (42 epochs on 800 parts) | 2 | 0.29 | 0.74 | 0.86 | 1.00 | 0.48 | 92 |
| tinycnn (24 epochs on 8,000 parts, previous shipped checkpoint) | 2 | 0.33 | 0.72 | 0.88 | **1.00** | 0.50 | 92 |
| **tinycnn retrained on the fixed renderer (shipped `assets/tinycnn_synthetic.pt`, 2026-09-07; seed 4242, 400 queries)** | 2 | **0.47** | **0.86** | **0.97** | **1.00** | **0.63** | 92 |
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

Query expansion (`MCV_QUERY_EXPANSION_K`) stays off: on the 800-part held-out
catalog (TinyCNN, 300 queries) k=3 left Recall@1 unchanged and cost 3 points of
family Recall@1, k=8 cost 3 points of Recall@1; raising `top_k` from 50 to 200
changed nothing (2026-09-07).

Hash descriptor: the anisotropy weighting of the oriented thumbnails was a
no-op (undone by per-group normalisation) until 2026-09-07; applied properly it
is worth +1.4 points Recall@1 and +2 points family Recall@1 on a 300-part
catalog (0.237 vs 0.223, 0.293 vs 0.273). The synthetic renderer now gives every
kind a distinct second view and keeps long parts on the canvas, which moved the
shipped TinyCNN's held-out Recall@1 from 0.28 to 0.30 on the same seed, and retraining
on the fixed renderer took it to 0.47 (family Recall@1 0.65, MRR 0.63; see the model
card).

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

Retrieval at this size with the shipped (800-part) model: SKU Recall@1 0.02 /
@10 0.18 / @50 0.54, **family** Recall@1 0.36 / @10 0.64 on parts from families
it never saw. A TinyCNN trained on 8,000 of these parts (24 epochs, 2.2 h on
4 cores) reaches SKU @10 0.20 / @50 0.57 and family Recall@1 0.42 / @10 0.67 on
the same held-out parts: more training data helps family-level recall, while
SKU-level recall stays capped by the catalog's look-alikes. The synthetic generator only has 495
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

Ingest is index-bound: 20k parts load in 0.5 s and re-load (delete + insert through the
FTS index) in 2.7 s on one core, so a 700k-part catalog is minutes, not hours.

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
      -> mcv build-index --with-feedback            (no training: the photo itself joins the gallery)
```

Three things use a confirmation, in order of cost: the reranker's usage prior
(log-scaled confirmation count, a tie-breaker, immediate), the gallery
(`--with-feedback` / `retrain` embed the real photo next to the renders, so the
same part photographed again from that angle is a near-exact hit), and
training (`mcv train --query-dir`, `mcv retrain`).

The synthetic renderer bootstraps the model; confirmed photos are the only data
that closes the synthetic-to-real gap, so the UI makes confirming a one-tap
action and "None of these" photos are kept under `_unknown/` for labelling.
`/feedback/stats` exposes the confirmed top-1 rate as the live accuracy metric.

## Purchase loop (the demo storefront that teaches the model)

```
photo -> /identify -> "Add to cart" (POST /cart, tied to the request_id)
      -> POST /checkout -> orders.jsonl + one `checkout` confirmation per photographed item
      -> events.jsonl: identify / cart_add / cart_remove / checkout / feedback / error
      -> GET /analytics: funnel, predicted-vs-bought confusions, tier precision when bought,
         confidence when right vs wrong, latency p95, errors -> plain-language issues
      -> mcv learn: photos into the index (incremental, seconds); full retrain once
         `learn_retrain_after` new confirmations arrived (purchase-weighted)
      -> mcv simulate [--learn]: synthetic customers walk the journey in-process and the
         same analytics say what is wrong, before and after learning
```

A purchase is the strongest label there is (the customer paid for it), so a
`checkout` confirmation weighs 3, a tap 2, a cart add 1 (`FEEDBACK_WEIGHTS`).
The weight repeats the photo for training and scales the usage prior; the index
holds each photo once because retrieval max-pools over a part's rows.
`learn_index` appends only the photos the index does not hold yet
(`meta.learned_paths`) with the deployment's own embedder, so the running API
accepts and picks up the result within seconds; a foreign or older index is
rebuilt once. Before a new photo joins the gallery it is identified once against the
current index and its candidate scores are kept as a calibration sample; from
30 samples on, `mcv learn` refits the temperature and the exact / likely
thresholds on those real outcomes (`calibration_from_purchases` in the
manifest), so the tiers track what customers actually bought instead of
synthetic renders. A learned photo scores ~1.0 against its own gallery row, and
the reranker's near-identical bonus (`w_exact`, ramping above similarity 0.99)
makes sure no category or usage prior can overturn it. The manifest records
`learned_at` / `retrained_at`, and the dashboard's "Learning loop" panel shows
the funnel, the confusion pairs, the issue list and how far the next retrain is.

Measured with `mcv simulate --customers 120 --learn` on a clean 200-part
synthetic demo (hash backbone, gallery augment 2, baseline scored without the
usage prior, 2026-09-07):

| | before learning | after `mcv learn` |
|---|---|---|
| bought photos that were the top answer | 42% | 99% (same photos, index reloaded) |
| top-1 over all 120 customers | 31% | 73% |
| top-1 on *new* photos of the same parts | 31% | 44% |
| part found in the top 5 (new photos) | 73% | 72% |

With the shipped TinyCNN checkpoint (retrained on the fixed renderer) on the
same demo, same protocol:

| | before learning | after `mcv learn` |
|---|---|---|
| bought photos that were the top answer | 74% | 100% |
| top-1 over all 120 customers | 71% | 98% |
| top-1 on *new* photos of the same parts | 71% | 84% |
| part found in the top 5 (new photos) | 98% | 99% |

(The previous checkpoint on the same protocol: 61% -> 100%, 59% -> 96%,
59% -> 69%, 96% -> 97%.) Its 118 calibration samples (34 wrong) raised the
exact threshold from 0.90 to 0.97 and the likely one from 0.60 to 0.78, because
on real outcomes "likely" answers had been right 86% of the time when bought and
"candidate" ones 57%; the tighter tiers are what the customer sees as
confidence. The two confusion pairs it flagged differ by pipe size and by
length / thread size, which the issue list turns into "use the coin and
Measure", and identification ran at 28 ms p50.

The retrain half of the loop was exercised on the same TinyCNN data: 232 new
confirmations tripped the threshold, `mcv learn` ran a purchase-weighted
`mcv retrain` (1 epoch, for the check), held out 110 purchase photos, and
measured Recall@1 0.68 / Recall@5 0.74 / MRR 0.71 on them; because the new
checkpoint's version differs from the served one, the index and calibration went
to `index-tinycnn/` and `models-tinycnn/` and the served index's `learned_at`
was left alone.

The hash learn step was incremental (88 photos in 24 s, index rows 1976 -> 2064). The
88 photos gave 88 calibration samples (51 wrong); the refit kept the temperature
and the exact threshold and raised the "likely" threshold from 0.60 to 0.75,
because no threshold reached 90% precision and 0.75 was the most precise one
with support. The gallery photo makes the exact angle a near-sure hit and lifts
new angles of the same part by almost half; the rest is what the full retrain is
for. The analytics also flagged four confusion pairs and that "likely" answers
were right only half the time when bought, which is the hash backbone's ceiling.

## Durability (nothing learned at run time is lost)

Every run-time artefact is a file under `data/` and is written before the
response goes out: the query photo to `cache/recent/` (so `/feedback` works
after a restart or on another worker), the feedback line and labelled photo,
the request-log line (reloaded into the `/metrics` window at boot). Index
writes are atomic (temp dir + swap) and the API polls the index `meta.json`
mtime every 15 s, so `build-index`, `retrain` and `restore` need neither a
restart nor a token. `mcv backup` folds the SQLite WAL, tars catalog, index,
calibration, photos, logs, manifest (and optionally the checkpoint) with a
`BACKUP.json` inventory; `mcv restore` extracts to a temp dir and swaps each
component in. On the phone, a confirmation made without a connection waits in
a localStorage outbox and syncs on `online` and every minute.

## Size matching (`pipeline/measure.py`)

McMaster-Carr often uses one image for a whole family, so no visual model can
tell a 1" screw from a 1-1/4" one. With the photo's scale (the user taps the
two ends of a coin, a card edge or an inch on a ruler; `mm_per_px` on
`/identify`) the foreground mask's extent along its principal axes gives the
object's long and short dimensions; the blob under the line the user drew is the
reference itself and is removed first, and a scale given in uploaded pixels is
corrected when the server decoded the JPEG at reduced size. Each candidate's
`length` / `od` / diameter is parsed to millimetres (fractions, inches, mm, `M6`, `#8-32`) and scored
+1 inside a tolerance band (a screw's length excludes the head) falling to -1
one catalog size away; the fusion reranker weights it like 0.2 of cosine
similarity, enough to reorder look-alikes but not to overturn a clear visual
match. Per 1280 px photo on one core (hash backbone, fast TTA): identify alone 29 ms;
with a scale and reference (size + thread pitch) 67 ms; with the coin hint
(no scale yet) 42 ms. Nothing is measured when no scale is supplied.

## Pipe sizing and thread pitch (`pipeline/pipe.py`, `pipeline/threads.py`)

The catalog's own measuring pages are encoded as tables: nominal pipe size to
male OD and schedule 40 ID, threads per inch for NPT and BSP, and the thread
compatibility matrix. Size matching uses them so a "3/8" fitting is checked
against 17.1 mm (male) or 12.5 mm (female), not 9.5 mm. Thread pitch comes from
the photo: the mean intensity along the part's major axis inside the foreground
mask is detrended and its dominant period (FFT, fundamental preferred over the
crest harmonic) times the scale is the pitch. On synthetic threads with noise,
shading and rotation the pitch is recovered within 8% in 9 of 9 cases: enough
to separate coarse from fine fastener threads (a 1.3x step) with a ±7% band.
NPT and BSP differ by only 3.6–5.5% at a given size, so the pipe-thread band
is ±1.5% and a measurement between the two stays neutral rather than endorsing
both. The reference coin is excluded from the profile, profiles shorter than
32 samples are rejected, and the peak must exceed the largest other spectral
feature by 2.5x so a washer's hole edges never read as a thread.

## Multi-photo queries and family answers

Several photos of one part (different angles) are embedded independently and
their TTA variants are stacked into one multi-query; a catalog part keeps its
best similarity over every (photo, variant, catalog image) triple. When the top
candidates are members of one family (same geometry, different length / thread
/ size), the result carries a `family` block listing the members and the
attributes whose values differ, so the UI can ask exactly the one question that
resolves the SKU instead of guessing.

## Serving details

* `/identify` accepts `constraints` (attributes the user already knows); when no
  retrieved candidate satisfies them the result says so in `notes` rather than
  pretending the filter applied. `category_guess` always carries the top coarse
  categories from the embedding prior.
* CPU-bound work runs in the threadpool so health checks and static files never
  stall behind a 6-photo query or a 200-photo batch; batch uploads are size-
  checked per file. `MCV_RATE_LIMIT_PER_MINUTE` caps per-client calls,
  `MCV_API_TOKEN` guards `/admin/*`.
* Phone performance: candidate images are served as cached 200 px JPEG
  thumbnails with long cache headers, JSON is gzipped, the client downscales
  photos to 1280 px before upload, and `tta=fast` (2 views) cuts embedding
  time versus `full` (8 views). Measured per query on 2 CPU threads (30-part
  demo catalog, gallery augmentation 1): TinyCNN 25 ms full / 15 ms fast;
  ensemble 103 ms full / 36 ms fast, with the same top-1 on that sample.
* The API warms up (catalog, index, backbone) at startup; `mcv serve --workers N`
  runs N processes via the `get_app` factory (each holds its own index copy).
* Every identification is appended to `data/logs/requests.jsonl`; `GET /metrics`
  joins it with feedback for the confirmed top-1 rate.
* Incremental index builds (`--only-new`) blend category centroids by stored
  per-category counts and refuse to mix a different image size / category depth
  with an existing index.

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
