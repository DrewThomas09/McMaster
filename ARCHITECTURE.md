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
| 6. Calibrate | `pipeline/calibration.py` | softmax over fused scores with a temperature fitted on validation queries; rules map probability + margin + raw similarity to `exact / likely / candidate / unknown`. Two listings of one spec (same family, every attribute equal) count as one answer: their probabilities add up and the margin is taken against the first different thing; the twins come back as `also_sold_as`. |

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
| tinycnn retrained on the fixed renderer, 24 epochs (2026-09-07; seed 4242, 400 queries) | 2 | 0.47 | 0.86 | 0.97 | **1.00** | 0.63 | 92 |
| tinycnn 48 epochs (shipped 2026-09-08 to 2026-09-09; seed 4242, 400 queries) | 2 | 0.55 | 0.93 | 0.99 | 1.00 | 0.71 | 92 |
| **tinycnn 83 of a 96-epoch schedule at half the learning rate (shipped `assets/tinycnn_synthetic.pt`, 2026-09-09; same held-out catalog and queries, re-rendered nipples)** | 2 | **0.62** | **0.96** | **1.00** | **1.00** | **0.76** | 92 |
| ensemble tinycnn(48ep)+hash (1:0.3) on the 800-part demo, 200 queries (2026-09-08) | 2 | 0.56 | 0.95 | 1.00 | 1.00 | 0.72 | 200 |
| tinycnn(48ep) alone, same demo and queries | 2 | 0.56 | 0.96 | 0.99 | 1.00 | 0.72 | 92 |
| tinycnn (83 of 96 epochs, shipped) alone on the 800-part demo, 200 queries (2026-09-09) | 2 | 0.66 | 0.99 | 1.00 | 1.00 | 0.80 | 81 |
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
shipped TinyCNN's held-out Recall@1 from 0.28 to 0.30 on the same seed, retraining
on the fixed renderer took it to 0.47, a 48-epoch schedule to 0.55, and 83 epochs of
a 96-epoch schedule at half the learning rate to 0.62 (family Recall@1 0.83, MRR 0.76;
the 48-epoch model's family
Recall@1 0.74, MRR 0.71; see the model card).

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
         confidence when right vs wrong, latency p95, errors -> plain-language issues;
         `found_by` (cart adds by door: photo / search / for_you / order_again) and the
         search funnel (typed searches, share narrowed by a facet chip, share with no
         result, share that led to a cart add)
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

With the shipped TinyCNN checkpoint (48 epochs on the fixed renderer,
2026-09-08) on the same demo, same protocol, half the customers using a coin:

| | before learning | after `mcv learn` |
|---|---|---|
| bought photos that were the top answer | 82% | 97% |
| top-1 over all 120 customers | 78% | 94% |
| top-1 on *new* photos of the same parts | 78% | 84% |
| part found in the top 5 (new photos) | 97% | 98% |

Purchases whose photo carried a coin measurement were the top answer 23/23
times before learning, against 76% without one; "exact" answers were right 54
of 55 times when bought. (The 24-epoch checkpoint on the same protocol without
coins: 74% -> 100%, 71% -> 98%, 71% -> 84%, 98% -> 99%; the first TinyCNN
checkpoint: 61% -> 100%, 59% -> 96%, 59% -> 69%, 96% -> 97%.) Its 118 calibration samples (34 wrong) raised the
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

## Marketplace: customers, segments and personalised ranking

```
orders.jsonl -> CustomerBook: per-customer histograms (categories, families, materials,
                sizes, parts), k-means segments over damped category shares (top level,
                plus the second level at half weight: an industry proxy), co-purchase pairs
             -> boosts(customer, candidates): log-odds of the customer's blended category
                prior (personal + segment + global, weighted by history) plus family /
                size / material nudges (soft, scaled by history) and "bought this before"
                (strong, from the first order); clipped to [-1, 1]
             -> /search?client_id re-sorts text hits inside bm25 tiers: the history
                decides among hits that match the words equally well (the size, material
                and finish variants of one name) and never lifts a weaker text match over
                a stronger one; part-number matches stay first
                /identify?client_id adds w_customer x max(0, boost) in the fusion
                /recommend: parts due again, staples, recent buys, complements of the
                last order, segment favourites; /me, /segments (five customers or more);
                the dashboard's "Customers and segments" panel
```

`mcv simulate-market --shops N --min-orders 10 --max-orders 20` builds a demo
marketplace: shops drawn from six industries (plumbing, machine shop,
maintenance, cabinetry, fluid systems, general) with jittered category mixes,
two preferred materials and a few staples they re-order; each finds its items by
text search or by photo, buys them, and comes back. Every lookup is asked with
and without the shop's id, so the lift is measured on the same query (the same
photo pose for both arms, only the personalised arm logged), the customer model
is rebuilt once per order round (a nightly rebuild, so the run is reproducible),
and the report splits the lift by how many orders the shop has placed and by
whether it had bought that part before. 60 shops, 870 orders (10-20 each) on the
shipped 48-epoch TinyCNN (2026-09-09, 1,215 searches and 790 photos):

| | plain | personalised | bought before | never bought | order 1-3 | order 4-8 | order 9+ |
|---|---|---|---|---|---|---|---|
| search top-1 | 53.3% | 69.6% | 55% -> 84% (n=675) | 51% -> 52% (n=540) | 54% -> 61% | 52% -> 72% | 54% -> 72% |
| search MRR | 0.699 | 0.810 | | | | | |
| photo top-1 | 81.0% | 83.9% | 83% -> 90% (n=393) | 79% -> 78% (n=397) | 78% -> 80% | 81% -> 82% | 83% -> 87% |
| photo MRR | 0.896 | 0.912 | | | | | |

At 1000 shops and 14,947 orders (10-20 each) on the final code of the day
(2026-09-09, 20,579 searches and 13,562 photos, 30% of the photos next to a
quarter; segments recovered the six industries with purity 0.90):

| | plain | personalised | bought before | never bought | order 1-3 | order 4-8 | order 9+ |
|---|---|---|---|---|---|---|---|
| search top-1 | 48.7% | 65.7% | 50% -> 82% (n=10900) | 47% -> 48% (n=9679) | 49% -> 57% | 49% -> 68% | 49% -> 68% |
| search MRR | 0.666 | 0.784 | | | | | |
| photo top-1 | 81.5% | 84.7% | 82% -> 89% (n=7238) | 81% -> 79% (n=6324) | 82% -> 83% | 81% -> 85% | 82% -> 85% |
| photo top-1, with a coin | 84.9% | 86.4% | | 84% -> 83% (n=1521) | | | |
| photo top-1, no coin | 80.4% | 84.2% | | 80% -> 78% (n=4803) | | | |
| photo MRR | 0.898 | 0.916 | | | | | |

A recommendation was in the next order 74.9% of the time against the order-again
baseline's 76.5%, 3.3% of orders taking a never-bought part.

The same marketplace on a catalog four times the size (800 parts, so four times
the same-name variants a search or a photo must choose among), 300 shops and
4,488 orders, on the epoch-83 model shipped that evening (2026-09-09, 6,359
searches, 4,085 photos, 30% with a coin; purity 0.91):

| 800-part catalog | plain | personalised | bought before | never bought |
|---|---|---|---|---|
| search top-1 | 23.1% | 44.4% | 23% -> 72% (n=2682) | 23% -> 24% (n=3677) |
| search MRR | 0.407 | 0.591 | | |
| photo top-1 | 64.4% | 69.0% | 67% -> 79% (n=1782) | 62% -> 61% (n=2303) |
| photo top-1, with a coin | 72.7% | 76.6% | | 71% -> 71% (n=524) |
| photo top-1, no coin | 61.8% | 66.7% | | 60% -> 59% (n=1779) |

A third of that run's wrong photo answers were the same spec under another part
number: the synthetic generator had listed 13% of the 800 parts as twins with an
identical family and attributes, which no photo, coin or prior can separate
(the analytics now report that share, `confusions_equivalent_share`, and raise
an issue when it is large, since real catalogs carry such twins too). The
generator no longer mints them (2.6% remain, where a family has few sizes); on
a fresh 800-part catalog drawn that way the held-out evaluation reads Recall@1
0.64 against 0.66 on the old draw, within the noise of 200 queries, so the
twins were a marketplace artefact (staples repeat) more than a model ceiling.

At 1000 shops and 15,015 orders on a fresh 800-part catalog drawn by the
deduplicated generator, final code of 2026-09-12 (20,890 searches, 13,892 photos,
30% with a coin; purity 0.94):

| 800-part catalog, 1000 shops | plain | personalised | bought before | never bought |
|---|---|---|---|---|
| search top-1 | 24.7% | 45.4% | 28% -> 74% (n=9105) | 22% -> 23% (n=11785) |
| search MRR | 0.437 | 0.611 | | |
| photo top-1 | 64.6% | 70.1% | 64% -> 78% (n=5914) | 65% -> 64% (n=7978) |
| photo top-1, with a coin | 74.4% | 78.5% | | 75% -> 75% (n=1870) |
| photo top-1, no coin | 61.5% | 67.4% | | 62% -> 61% (n=6108) |

A recommendation was in the next order 65.8% of the time against the order-again
baseline's 67.1%, 1.1% of orders taking a never-bought part.

The bigger the catalog, the more a shop's history is worth: a text search that
lands the right part first 23% of the time on its own lands it 72% of the time
for a part the shop has bought before, and the coin is worth 11 points on a
photo. The 200-part numbers above are the demo; this row is closer to a real
catalog. What the whole marketplace buys now also breaks ties among a name's
variants for a stranger (`popularity_boosts`, at most 0.3 inside a bm25 tier):
the same run with it reads plain search 26.1% top-1 (MRR 0.442), rising from
24% in a shop's first three orders to 27% from its ninth as purchases pile up,
with personalised search unchanged at 44.5% (a shop's own history outranks it).

When neither the history nor popularity can tell the variants apart, the phone
shows what does: `GET /search/facets?q=` takes the bm25 top tier of the query
(`top_tier`: hits within 10% of the leader, the variants of one name) and
returns the attributes that vary across it with their value counts (`facets`:
keys at least half the tier carries, with at least two values, largest first),
and the phone renders the first two as chips under the results (from two
variants up). One tap appends the value to the query. A spec value typed as it is
written is then an exact match, not two more words for bm25: `1-1/2"` mentions
`1` twice and would outscore `1/2"`, so `search_text_scored` strengthens the
text score of a hit whose attribute value appears in the query as a whole
whitespace-delimited token by 25% per value (`_promote_exact_values`,
`EXACT_VALUE_BONUS`). The exact variants form the leading tier, which the
chips can narrow again (one key per tap, most widely carried first). A
stranger's never-bought searches sit where the text alone puts them in every
run above (46% top-1 on the 200-part catalog, 22-24% on the 800-part one) and
no re-ranking moves them; the chips are the lever for that half of the
traffic. `mcv simulate-market` plays the tap as a customer who knows the spec
they want: when the wanted part is not on the page the phone shows (10 rows)
and chips are offered, the shop taps the first of the two shown whose
attribute its part carries, up to twice (`_tap_a_chip`), never a wrong chip,
so the reading is the ceiling for a customer who knows the size. The report
carries `search.chips` (how many searches had the part off the page, how many
were offered chips, tapped, and made worse) and a `narrowed_top1` next to every
search split. Runs dated before 2026-09-12 predate the exact-value promotion,
which also moves the plain arm when a query word is itself an attribute value
(`Brass Hex Nut`), so their plain columns are not directly comparable.

Measured with the exact-value promotion and the chips (300 shops, 10-20 orders
each, seed 0, 30% of photos with a coin, 2026-09-12):

| | searches | plain | personalised | + chips | never bought: plain -> pers. -> chips | part off the page | chips tapped | ended first | worse |
|---|---|---|---|---|---|---|---|---|---|
| 200-part catalog, 4,466 orders | 6,090 | 51.9% (MRR 0.689) | 66.3% (0.788) | 66.3% (0.788) | 49.0% -> 48.1% -> 48.1% | 0 | 0 | - | 0 |
| 800-part catalog, 4,513 orders | 6,301 | 28.2% (MRR 0.469) | 44.8% (0.606) | 46.6% (0.639) | 22.8% -> 22.4% -> 25.2% | 461 (7.3%) | 459 | 25% | 0 |

On the small catalog every variant of a name fits on the page, so the honest
simulation never taps (the chips still show; a customer may prefer a tap to a
scroll, which this does not count). On the 800-part catalog one search in
fourteen has the wanted part past the ten rows shown; chips were offered on
every one of them, one tap put a quarter of them first and none lower, and
never-bought search top-1 moved for the first time (22.4% -> 25.2%). The plain
arm itself rose from 22-23% to 28% top-1 on this catalog against the runs
above, the exact-value promotion's own effect on queries that name a material.
Photos in the same runs: 84.2% -> 86.8% (200 parts) and 65.0% -> 69.1% (800
parts) top-1 plain -> personalised, 78.5% with a coin against 66.2% without on
the 800-part catalog; segments recovered the six industries with purity 0.91
in both; the For-you strip was bought from in 74.1% / 66.7% of the orders that
followed it against an order-again baseline of 75.5% / 68.0%, with a
never-bought part in 4.0% / 1.7%.

An earlier run at 300 shops and 4,481 orders (seed 3, 2026-09-09, 6,147
searches and 4,141 photos, before the eraser fixes and the segment vector
change):

| | plain | personalised | bought before | never bought | order 1-3 | order 4-8 | order 9+ |
|---|---|---|---|---|---|---|---|
| search top-1 | 47.6% | 62.9% | 49% -> 78% (n=3255) | 46% -> 46% (n=2892) | 47% -> 55% | 48% -> 65% | 47% -> 65% |
| search MRR | 0.658 | 0.766 | | | | | |
| photo top-1 | 79.9% | 82.6% | 81% -> 88% (n=2219) | 78% -> 77% (n=1922) | 78% -> 80% | 81% -> 83% | 81% -> 84% |
| photo MRR | 0.889 | 0.905 | | | | | |

Segment purity 0.55 at k = 8 (second-level vectors, since replaced); a recommendation in the next order 74.6% against an
order-again baseline of 76.1%, 2.3% of orders taking a never-bought part (this run
predates the material-aware browse of the new-thing slot).

Read the split, not the headline: nearly all of the search lift is the shop's
own re-orders being put first among the variants of a name, which is what a
customer expects and the text rank alone cannot do; on a part the shop has
never bought the category and material prior is worth one point on search and
costs one on photos (the fusion term pulls toward what the shop usually buys,
and a new part is by definition not that). The lift grows with a shop's history
and settles after about four orders.

Segments: clustering the raw second-level category shares recovered the six
industries with purity 0.55-0.58 (k = 8), because a shop's staple sub-category,
bought every week, dominated its vector. Square-rooted shares of the top level
plus the second level at half weight, unit length, give 0.85 offline on the
300-shop orders (six seeds, 0.77-0.88) and 0.77 in a fresh 60-shop run, where
the one merge left is plumbing with fluid systems, which the personas define
over the same categories. The ranking lift did not move with it (search top-1
+16 points either way, photos +3).

The photo term's weight (`w_customer`) was swept on the same 60 shops, photos
only (2,005 lookups, 1,068 of parts bought before, 937 never bought; 2026-09-09):

| w_customer | photo top-1 | MRR | bought before | never bought |
|---|---|---|---|---|
| 0 (off) | 80.8% | 0.895 | 83.2% | 78.0% |
| 0.03 | 82.7% | 0.905 | 87.3% | 77.5% |
| **0.06 (shipped)** | 83.5% | 0.910 | 89.5% | 76.7% |
| 0.10 | 83.9% | 0.912 | 90.9% | 76.0% |

Every step buys re-orders and sells new parts: the prior pulls a photo of a new
part toward the look-alike the shop bought before. 0.06 stays: most of the gain
for a third of the loss. The pairs the prior confuses are size siblings (O-rings
one OD apart, elbows 1/2" against 1", nuts #6 against 5/16"), which a coin in the
frame should settle, and on those pairs it does (a 1/2" elbow photographed next
to a quarter rules the 1" out, a 5/16" nut rules #6 out). Measured on the same
60 shops and photos with `--coin-rate 1.0` (1,586 of 1,979 photos staged with a
coin, after the demo learned to stage nuts and fittings and the size rules
gained a nut vote):

| photos only, 1,979 lookups | plain | personalised | MRR (plain) | bought before | never bought |
|---|---|---|---|---|---|
| no coin | 80.2% | 83.0% | 0.893 | 84% -> 91% | 76% -> 74% |
| every photo next to a quarter | 85.6% | 87.2% | 0.914 | 87% -> 91% | 84% -> 83% |
| the same, before the eraser fixes | 78.2% | 79.9% | 0.844 | 79% -> 83% | 78% -> 77% |

The first all-coin run said the coin cost 2 points overall, and a sweep of
every part in three poses found why, in three parts. One pose in three puts the
part on a dark bench close to the coin's colour, and with the coin filling 43%
of the close-up the eraser took its ruler path, grew the coin's colour across
the whole bench and painted the part over (that pose: 81% -> 63% with a coin).
The fill's noise was sampled from the coin's anti-aliased rim, so the painted
disc was speckled on a smooth bench and read as foreground, spoiling the crop.
And a 1/4" screw next to a quarter is under the crop's minimum blob, so it
stayed an eighth of the frame. Coin or ruler is now decided by shape (the
colour region must fill the disc on both sides of the segment), a coin on a
bench of its own colour erases only its disc, the bench is sampled clear of
the rim with a robust noise level, and the crop trusts a smaller blob once a
reference has been erased. The short axis is now the narrowest width over all
directions, so a turned nut still measures across flats. The sweep after all
of it, no customer prior: top-1 81.3% without a coin, 91.5% with one, every
pose gaining. What still lost was a size vote that was right about the render
and wrong about the part: the synthetic nipple was drawn 2.2:1 where a real
3/8" x 2-1/2" one is 3.7:1, so the vote preferred the 1/2" sibling; the renderer
now draws nipples to the catalog's length over OD, and on a catalog rendered
that way nipples with a coin go 75% -> 100% (the sweep overall 83.0% -> 91.5%).
The rest are near-ties where both siblings fit the measurement (a bearing
measured between 1" and 1-1/4"). In the marketplace the coin lifts
never-bought parts by 8 points and leaves the prior costing one there, and
re-orders keep their gain. The coin is worth asking for; the prior stays a
tie-breaker. The hint that asks for it (`find_coin`, the ring drawn on the
photo) was measured the same way: on 322 staged photos it lands on the coin
99.7% of the time and never on another blob, with two false hints in 400
photos that had no coin, both round brass parts (2026-09-11; it was 68% before
the "something else in the frame" test stopped demanding a companion 15% the
coin's size, and 94% before a finer second pass for a small coin beside a long
part).

Segments recovered the six industries with purity 0.58 at k = 8 with the original second-level vectors (see below for the
fix): plumbing and fluid systems, and machine shop, maintenance and cabinetry,
overlap in what they buy, which is honest, since the boost works off the shared
category mix either way.

Recommendations are scored against the dumbest baseline, the shop's six
most-bought parts (an order-again list). On the same 60 shops, six slots:

| recommendation list | in the next order | of which a part never bought before |
|---|---|---|
| order-again baseline (six most-bought parts) | 76.3% | 0% by construction |
| staples only (bought twice or more) | 65.1% | 0.3% |
| due, staples, one-off buys by recency | 78.0% | 0.3% |
| the same with one slot kept for something new | 76.3% | 3.5% |

A shop's own history fills six slots after three orders, so without a reserved
slot the discovery half of the list (complements, segment favourites, unbought
parts in the shop's usual aisle and material) never showed. Which guess earns
the slot was then measured by kind (2026-09-12, 60 shops): an unbought part from
the usual aisle and material was bought 5.7% of the times it was shown, a
complement 1.8%, a segment favourite never; the guess tiers now rank them in
that order, and the same run reads 4.7% of orders taking a never-bought part
(from 4.0%) at 74.0% overall. Keeping one slot
costs 1.7 points of re-order hits and buys a tenfold rise in new parts bought
from the strip. The synthetic shops pick their non-staple parts at random
within their category mix, which is close to the ceiling for one guess; the
number to watch on real customers is that last column.

`scripts/replay_recommendations.py` replays a market's order log through the
recommender round by round (the book built from the rounds before, as the
nightly rebuild does) and scores variants in a minute instead of a two-hour
run. On the two 300-shop logs of 2026-09-12 it reproduces the simulation
(66.7% / 73.9% against 68.1% / 75.4% for the order-again list at six slots) and
shows where the gap sits: the sixth re-order slot is bought 5-8% of the time,
the guess that replaces it 1.7-3.6%, and the re-order ordering itself beats the
baseline (68.4% / 76.2% with no new slot). The phone's strip scrolls and asks
for eight, where the eighth re-order is worth 3-5% and one new slot costs
nothing against the baseline:

| eight slots, one kept for something new | 800-part catalog | 200-part catalog |
|---|---|---|
| order-again baseline (eight most-bought parts) | 71.2% | 78.6% |
| the strip as shipped | 70.6% (2.2% never bought) | 79.2% (4.8% never bought) |
| no new slot | 71.7% (1.1%) | 80.1% (2.9%) |
| two new slots | 69.7% (3.7%) | 78.0% (6.5%) |

A new slot only when the re-order it displaces was bought once and not recently
scored the same as always keeping one (70.6% / 79.4%); the simpler rule stays.
`mcv simulate-market` now scores the strip at the eight slots the phone shows,
against an eight-part order-again list.

The first 1000-shop run of the day (same shops and orders) used a search
re-rank that could lift any hit in a 50-row window by up to 0.3 of the position
score; it showed search top-1 48.6% -> 59.2% and photo top-1 79.8% -> 83.6%.
The review found the weight could overturn a text match; the tiered re-rank
above gains more (65.7%) while never lifting a weaker text match over a
stronger one.

### The learning loop at marketplace scale

`mcv simulate-market --learn-every 3` runs `mcv learn` (purchased photos into the
gallery, tiers refitted on outcomes) after every third order round and serves the
result, as a nightly job would. 60 shops, photos only, 20 rounds, six learns
(2,029 lookups, 2026-09-12, same shops and photos as the no-learning run):

| photos only | plain | personalised | bought before | never bought |
|---|---|---|---|---|
| no learning | 84.4% | 86.1% | 86.5% -> 91.1% | 82.0% -> 80.6% |
| learning every 3 rounds | 84.6% | 86.5% | 87.7% -> 92.5% | 81.2% -> 79.8% |

A gain on the parts the marketplace has photographed and bought, a smaller loss
on the ones it has not: learned rows of a bought part attract the photos of its
look-alikes. Two rules keep that in check, and both were set by measurement:
a confirmed photo joins as one gallery row (a real photo is already in the photo
domain; the catalog's three augmented rows per image would triple its pull), and
a part keeps at most `MCV_LEARN_MAX_PHOTOS_PER_PART` (8) of them, newest first.
The gain is small here by construction: a learned photo of a synthetic render
adds little that the render lacked, where a real photo adds the whole domain
gap. The first run of this measurement read four points worse across the board
and turned out to be a bug worth the exercise: the simulation's worker had no
`MCV_INDEX_GALLERY_AUGMENT`, and `mcv learn` rebuilt the augmented 1,800-row
gallery as 600 unaugmented rows before adding photos. Learning now keeps the
index's own augmentation.

Could the learned rows count for less, so a neighbour's photo does not pull a
look-alike's query away from its own catalog rows? The index now records which
rows came from photos (`meta.photo_rows`) and `learned_row_weight` multiplies
their similarity in retrieval. Measured with `mcv simulate --learn` (300
customers, seed 1, 200-part catalog, 2026-09-12): at 1.0 the bought photos come
back 99.3% top-1 and new photos of the same parts 85.7% (from 80.3% before
learning); at 0.9 the bought photos fall to 83.7% and new photos stay at
85.7%; at 0.75, 82.7% and 85.7%. The catalog rows of look-alikes sit above 0.9
cosine in this space, so any discount on the exact photo match loses the
"seen it before" promise and buys nothing on new photos. The weight stays at
1.0; the knob is kept for a backbone with a wider spread.

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

The reference object must not vote on looks: when a segment is given, the disc it
spans is filled with the surrounding bench (median colour, matched noise) before
the photo is embedded (`erase_reference`: the region grown through the
reference's own colour from the marked segment, so a card or ruler is removed
without touching the part beside it), and only the measurement sees the coin. On
52 synthetic parts of every kind with the shipped TinyCNN (2026-09-07), photos
with a quarter beside the part were top-1 37/52 with the erase and the size
votes, against 35/52 for the same photos without a coin and about 12/52 when the
coin was left in the embedding; the part in the top 5 in 47/52 against 52/52,
the cost of the part being smaller in a frame it shares with a coin. In the purchase-loop simulation with half the customers
using a coin (`mcv simulate --customers 120 --coin-rate 0.5 --learn`, same
model and demo, re-run on the colour-grown erase), the bought part had been the
top answer 83% of the time when the photo carried a measurement and 71% when it
did not, and on new photos after learning 80% against 76%; the dashboard shows
the same split live.

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

### Wall schedules and the bore (from the butt-weld and nipple pages)

The catalog's unthreaded (butt-weld) and thick-wall pages add a second
dimension: wall thickness by schedule (10 thin-wall, 40 standard, 80 thick-wall;
`PIPE_WALL_IN`), so a female or unthreaded fitting's bore is OD minus two walls
(`pipe_id_mm(size, schedule | wall)`). The Measure tool now also reads the
largest hole through the part (`bore_px`: the foreground's enclosed holes, from
a border flood fill) and `size_consistency` compares it with the candidate's
bore when the part names a wall or schedule, so an end-on photo of a coupling
separates thin-wall from thick-wall look-alikes of the same nominal size.

The page importer reads the layouts these pages use: two pipe sizes per row
(reducing bushings and couplings, `pipe_size_b`), a max-psi cell in front of
every column, wall thickness and the (C) dimension, flange OD and bolt columns,
thread adapters (`thread_a`, `thread_b_type`, `thread_b` such as `M10 x 1.0`),
pipe sold by the foot (`length: 3 ft`), and the section's "Connections:" line.

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
