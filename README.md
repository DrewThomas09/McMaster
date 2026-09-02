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

## Using your own catalog data

McMaster-Carr does not offer a public bulk product API and prohibits scraping, so
this repository ships **no** McMaster data. Bring an export you are licensed to
use (their Product Information API for account holders, an internal dataset, or a
partner feed). Three input formats are supported:

| Format | Shape |
|---|---|
| JSONL | one object per line: `part_number, name, category_path[], description, attributes{}, image_paths[], family_id` |
| CSV | same columns; `category_path` joined with `>`, `image_paths` with `;`; unknown columns become attributes |
| Directory | `<root>/<part_number>/{meta.json, *.jpg}` |

```bash
cp .env.example .env                         # set MCV_BACKBONE=openclip for real models
pip install -e ".[ml,faiss]"                 # torch + open_clip + faiss
mcv ingest data/catalog/parts.jsonl          # -> data/catalog/catalog.sqlite
mcv build-index --backend faiss              # embed every image -> data/index/parts/
mcv evaluate --fit-calibration               # Recall@K on augmented queries, fit confidence temperature
mcv identify photo.jpg                       # JSON result
mcv serve                                    # http://localhost:8000  (POST /identify)
```

### Improving accuracy

1. **Fine-tune the backbone.** `mcv train --config configs/train_openclip.yaml`
   runs supervised-contrastive training with photo-style augmentation and hard
   negative mining, splitting validation by *family* so look-alike SKUs never leak.
   Point `MCV_BACKBONE_CHECKPOINT` at `best.pt` and rebuild the index.
2. **Turn on the vision reranker.** `MCV_RERANK_LLM_ENABLED=true` (needs
   `ANTHROPIC_API_KEY`, `pip install -e ".[llm]"`). The top candidates and their
   catalog images are shown to Claude, which ranks them and extracts attributes
   (material, drive style, visible text) that the fusion scorer then uses.
3. **Turn on OCR.** `MCV_OCR_ENABLED=true` (`pip install -e ".[ocr]"`). A readable
   part number on a bag or a stamped marking short-circuits to an exact match.
4. **Collect real photos.** Put labelled phone photos under
   `data/queries/<part_number>/*.jpg` and run `mcv evaluate --query-dir data/queries`
   to measure and calibrate on the real distribution.

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
