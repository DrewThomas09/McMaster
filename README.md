# McMaster-Vision

Photo in, McMaster-Carr part number out.

Upload a picture of a screw, bearing, fitting, or any of the ~700,000 SKUs in the
McMaster-Carr catalog and the system returns the most likely part numbers with a
calibrated confidence and the evidence behind each candidate.

```
photo ──▶ preprocess ──▶ embed ──▶ vector search (top-K) ──▶ rerank ──▶ calibrate ──▶ 91251A537 (94%)
                (crop, EXIF)   (CLIP/DINOv2)   (FAISS, 700k)   (fusion + Claude vision)
```

The problem is treated as **fine-grained image retrieval**, not a 700,000-way
classifier: every catalog image is embedded once into a vector index, a query photo
is embedded with the same model, nearest neighbours become candidates, and a
reranking stage resolves the look-alikes. See [ARCHITECTURE.md](ARCHITECTURE.md).

## Quick start (no GPU, no proprietary data)

```bash
pip install -e ".[dev]"
mcv demo --parts 300           # synthetic catalog -> index -> evaluation -> UI on :8000
```

`mcv demo` renders a synthetic hardware catalog, ingests it, builds the index,
reports Recall@K on photo-style augmented queries, identifies a sample query, and
serves the upload UI at http://127.0.0.1:8000.

The synthetic catalog covers 39 hardware families (screws, nuts, washers, pins,
gears, bearings, fittings, springs, brackets, tools) in 14 materials with
canonical, top-down, and rotated views.

Two backbones run fully offline:

| backbone | what it is | when to use |
|---|---|---|
| `hash` | hand-crafted numpy descriptor | CI, plumbing, no torch |
| `tinycnn` | 1.6M-parameter residual net trained from scratch with `mcv train --config configs/train_tinycnn.yaml` | offline demos of the full learning loop |
| `ensemble` | weighted fusion of `tinycnn` + `hash` (best offline accuracy) | offline serving |

```bash
mcv demo --parts 800 --backbone ensemble                    # uses the shipped assets/tinycnn_synthetic.pt
mcv demo --parts 800 --backbone ensemble --train-epochs 24  # ...or train tinycnn first (~1 h on 4 CPU cores)
```

`assets/tinycnn_synthetic.pt` (6.5 MB) is a TinyCNN trained for 42 epochs on
the synthetic catalog; it is the default for `tinycnn` / `ensemble` in the demo
when no `--checkpoint` is given. It knows synthetic renders, not real photos:
retrain on your own images with `mcv train`.

Real accuracy on real photos comes from the CLIP / DINOv2 backbones plus
fine-tuning (below). See ARCHITECTURE.md for measured numbers.

## Using McMaster-Carr's own images

The most useful gallery is McMaster-Carr's own product imagery. Three ways in:

1. **Import product pages by part number or URL** (fetches the page, its images,
   name, category breadcrumb, and spec table; polite: one request at a time,
   `robots.txt` honoured, cached on disk):

   ```bash
   mcv import-web 91251A537 9452K21 https://www.mcmaster.com/3164T1/
   mcv import-web --file my_parts.txt        # one part number / URL per line
   mcv build-index
   ```

   This is meant for the parts you care about (an order history, a BOM, a shelf),
   not for crawling the whole catalog. McMaster-Carr's terms of use restrict
   automated bulk access; for the full 700k-SKU catalog use a licensed export or
   their account-holder Product Information API (`catalog/sources.py` has the
   adapter stub). The page parser was written against McMaster's page structure
   (JSON-LD product data, Open Graph tags, `ImageCache` images) and falls back to
   generic heuristics; verify it on a live page and adjust `McMasterParser` if
   their markup changes.

2. **Screenshots.** Save screenshots of product images into a folder named by
   part number (`91251A537.png`, `91251A537_2.png`, ...), optionally with a
   `meta.jsonl` of names/categories, then `mcv ingest that_folder`.

3. **Bulk exports** you already hold:

| Format | Shape |
|---|---|
| JSONL | one object per line: `part_number, name, category_path[], description, attributes{}, image_paths[], family_id` |
| CSV | same columns; `category_path` joined with `>`, `image_paths` with `;`; unknown columns become attributes |
| Directory | `<root>/<part_number>/{meta.json, *.jpg}` |

Optional extras: `ml` (torch, open_clip, timm), `faiss`, `ocr` (easyocr),
`llm` (anthropic), `segment` (rembg), `export` (onnx, onnxscript, onnxruntime),
`dev`, or `all`.

```bash
cp .env.example .env                         # set MCV_BACKBONE=openclip for real models
pip install -e ".[ml,faiss]"                 # torch + open_clip + faiss
mcv ingest data/catalog/parts.jsonl          # -> data/catalog/catalog.sqlite
mcv build-index --backend faiss              # embed every image -> data/index/parts/
mcv evaluate --fit-calibration               # Recall@K on augmented queries, fit confidence temperature
mcv identify photo.jpg                       # JSON result
mcv serve                                    # http://localhost:8000  (POST /identify)
```

## One photo, one answer (or several angles)

`mcv serve` hosts a phone-friendly page: **Take a photo** opens the camera on a
phone, desktop users can drop an image or paste a screenshot, the image is
downscaled client-side, and the answer comes back as a verdict card
(`exact / likely / candidate / unknown`) with the top candidates, their catalog
images, specs, evidence, and a link to the part on mcmaster.com. Open it from a
phone on the same network (`mcv serve --host 0.0.0.0`) or behind HTTPS for
camera access on iOS.

* **Add another angle** sends up to six photos of the same part in one query;
  every photo's views are searched and each catalog part keeps its best score.
* **Look-alike families.** When the probability mass lands on several SKUs that
  differ only in a non-visual spec, the answer says so and lists what tells them
  apart (`family.distinguishing_attributes`, e.g. length 1/2" / 3/4" / 1").
* **"This is it" / "None of these"** posts to `/feedback`, which files the photo
  under `data/queries/<part_number>/` (or `_unknown/`). Those labelled real
  photos are exactly what `mcv evaluate --query-dir data/queries` measures on
  and what `mcv train --query-dir data/queries` adds as training views, so
  accuracy on *your* parts improves with use. `/feedback/stats` reports how
  often the top-1 was confirmed.
* **Text search** (`/search?q=`) is the fallback when nothing matches.

### Improving accuracy

1. **Fine-tune the backbone.** `mcv train --config configs/train_openclip.yaml`
   runs supervised-contrastive training with photo-style augmentation and hard
   negative mining, splitting validation by *family* so look-alike SKUs never leak.
   Point `MCV_BACKBONE_CHECKPOINT` at `best.pt` and rebuild the index.
2. **Turn on the vision reranker.** `MCV_RERANK_LLM_ENABLED=true` (needs
   `ANTHROPIC_API_KEY`, `pip install -e ".[llm]"`). The top candidates and their
   catalog images are shown to Claude, which ranks them and extracts attributes
   (material, drive style, visible text) that the fusion scorer then uses.
3. **Index photo-style variants.** `MCV_INDEX_GALLERY_AUGMENT=2` embeds two
   augmented renders per catalog image alongside the clean one (database-side
   augmentation). It lifts the hash backbone's Recall@1 by ~50% relative and
   helps any backbone that was not fine-tuned on photos.
4. **Query expansion.** `MCV_QUERY_EXPANSION_K=3` re-queries with the mean of
   the query and its nearest gallery vectors. Helps learned embeddings; leave it
   off for the hash descriptor.
5. **Turn on OCR.** `MCV_OCR_ENABLED=true` (`pip install -e ".[ocr]"`). A readable
   part number on a bag or a stamped marking short-circuits to an exact match.
6. **Collect real photos.** Put labelled phone photos under
   `data/queries/<part_number>/*.jpg` and run `mcv evaluate --query-dir data/queries`
   to measure and calibrate on the real distribution.
7. **Export for serving.** `training/export.py` writes the fine-tuned embedder to
   ONNX or TorchScript so the API container needs no training dependencies.

Set `MCV_BACKBONE_PRETRAINED=none` to start from random weights (offline smoke
tests only).

## API

```
POST /identify?top_n=5&use_llm=false     multipart "file"   -> IdentificationResult
GET  /parts/{part_number}                                    -> Part
GET  /parts/{part_number}/image
GET  /search?q=socket+head+screw
GET  /stats  /health  /docs
```

`IdentificationResult.tier` is one of `exact`, `likely`, `candidate`, `unknown`
and is the field a downstream workflow should branch on.

## Layout

```
src/mcmaster_vision/
  config.py        settings (env MCV_*, YAML)
  schemas.py       Part, Candidate, IdentificationResult ...
  catalog/         taxonomy, sources (JSONL/CSV/dir/API), SQLite store, ingest
  data/            photo-style augmentation, synthetic renderer, splits, torch datasets
  models/          backbones (hash | open_clip | DINOv2), embedder + TTA, heads, losses
  index/           VectorIndex (numpy exact | FAISS HNSW/IVF-PQ), builder
  pipeline/        preprocess, OCR, retrieve, attributes, rerank (fusion + Claude), calibrate, identify
  training/        train loop, hard-negative mining, evaluation, ONNX/TorchScript export
  api/             FastAPI app + upload UI
  cli.py           mcv ingest | build-index | identify | serve | train | evaluate | demo
```

## Development

```bash
make lint      # ruff
make test      # pytest (uses the synthetic catalog; no network, no GPU)
docker compose up api                          # serve
docker compose --profile jobs run indexer      # ingest + index inside a container
```
