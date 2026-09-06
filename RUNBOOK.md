# Runbook: from a folder of McMaster-Carr images to a running identifier

Everything below is built and tested against the synthetic catalog. The only
input this system still needs is the real imagery (plus whatever metadata comes
with it). Times are for 700k parts x 3 images unless stated.

## 0. What the images can look like

| you have | do this |
|---|---|
| a folder of images named by part number (`91251A537.jpg`, `91251A537_2.png`, screenshots ...) | `mcv bootstrap that_folder` — part numbers come from file names; then `mcv enrich` fills names / categories / specs from McMaster pages |
| a folder tree `<part_number>/{meta.json, *.jpg}` | `mcv bootstrap that_folder` |
| a spreadsheet / JSONL with `part_number, name, category_path, ... , image_paths` | `mcv bootstrap parts.jsonl` (or `.csv`) |
| a spreadsheet with **image URLs** instead of files | `mcv fetch-images parts.csv --out data/catalog/parts_with_images.jsonl` then `mcv bootstrap data/catalog/parts_with_images.jsonl` |
| just part numbers | `mcv import-web --file part_numbers.txt` (fetches pages + images politely), then `mcv build-index` |

`mcv validate <source>` first if the drop is large: it reports missing or
unreadable files, duplicate part numbers / images, parts without images, tiny
images, and category coverage, without embedding anything.

## 1. Bootstrap (one command)

```bash
cp .env.example .env            # choose backbone, index backend, gallery augmentation
mcv bootstrap /path/to/drop --workers 4
```

Stages, all resumable by re-running the individual commands:

1. **validate** (seconds) - stops on duplicate part numbers or zero usable images.
2. **ingest + normalise** (~20 min) - images are copied to `data/images/catalog/<pn>/`
   as EXIF-corrected RGB JPEG at most 1024 px, de-duplicated by content hash;
   metadata goes to `data/catalog/catalog.sqlite` (with full-text search).
3. **embed + index** - `--workers N` runs N embedding processes.
   TinyCNN CPU: ~100 img/s per core -> 2.1M images in ~1.5 h with 4 cores.
   CLIP/DINOv2 on one GPU: ~1000 img/s -> ~35 min. Backend `auto` picks
   FAISS HNSW above 50k vectors (needs `pip install -e ".[faiss]"`).
   Gallery augmentation (`MCV_INDEX_GALLERY_AUGMENT=2`) triples the rows and
   the time; measured +25-50% Recall@1 for un-finetuned backbones.
4. **evaluate + calibrate** (minutes) - Recall@K on photo-style augmented
   queries, softmax temperature fitted and saved to `data/models/calibration.json`.

`data/manifest.json` records what was built; `mcv status` and `GET /status` show it.

## 2. Serve

```bash
mcv serve --host 0.0.0.0 --port 8000       # or: docker compose up api
```

Open `http://<host>:8000/` on a phone (camera needs HTTPS off-localhost - put it
behind any TLS proxy). `POST /identify` takes 1-6 photos; `POST /feedback`
stores confirmations; `POST /admin/reload` (header `X-API-Token` when
`MCV_API_TOKEN` is set) swaps in a rebuilt index without downtime.

## 3. Keep it accurate

* **Confirmations are training data.** Every "This is it" lands in
  `data/queries/<part_number>/`. Measure on them: `mcv evaluate --query-dir data/queries --fit-calibration`.
  Train on them: `mcv train -c configs/train_tinycnn.yaml --query-dir data/queries`
  (or `configs/train_openclip.yaml` on a GPU), then `mcv build-index` and `POST /admin/reload`.
* **New SKUs**: `mcv ingest new_parts.jsonl && mcv build-index --only-new` embeds only the additions.
  `GET /status` reports `index_stale: true` whenever the catalog changed after the
  index was built. Index writes are atomic (temp dir + swap), so rebuilding while
  serving and then `POST /admin/reload` is safe.
* **Removed or changed SKUs**: `mcv build-index` (full rebuild, same command).
* **Better model**: set `MCV_BACKBONE=openclip` (or `dinov2`), `MCV_BACKBONE_CHECKPOINT=...`, rebuild the index.
* **Hard cases**: `MCV_RERANK_LLM_ENABLED=true` sends the top candidates and the
  photo to Claude for a structured verdict (needs `ANTHROPIC_API_KEY`); `MCV_OCR_ENABLED=true`
  reads part numbers printed on bags and parts.

## 4. Sizing (700k parts, 2.1M images, 512-d)

Measured on 20k parts / 60k images with 4 CPU cores: 72 img/s embedding per
process (TinyCNN, including crop and normalisation), HNSW build 3 s per 60k
vectors, query p50 34 ms with HNSW vs 111 ms exact. Extrapolated to 2.1M images:
~8 h of embedding per process (2 h with 4 workers, ~35 min on a GPU with CLIP),
HNSW build ~2 min, index ~2 GB at 128-d or ~5 GB at 512-d.

| item | size |
|---|---|
| normalised images at <= 1024 px | ~150-250 GB (JPEG q92) |
| SQLite catalog + FTS | ~1 GB |
| numpy exact index (float32) | 4.3 GB RAM, ~50 ms/query |
| FAISS HNSW (M=32) | ~5 GB on disk / RAM, < 5 ms/query |
| FAISS IVF-PQ (`FaissIndex(kind="ivfpq")`) | ~0.5 GB, small recall loss |
| API container | index size + ~1 GB |

## 4b. Operator commands

| command | purpose |
|---|---|
| `mcv doctor` (`--json`) | optional deps, GPU, checkpoint, index/backbone match, index freshness, calibration, disk |
| `mcv status` / `GET /status` | what is built, from what, and how well it measured |
| `GET /metrics` | request volume, tier mix, latency p50/p95, confirmed top-1 rate |
| `mcv review-unknowns` | HTML contact sheet of "none of these" photos + current candidates, for labelling |
| `mcv retrain --reload-url http://localhost:8000` | train on catalog + confirmed photos, rebuild, refit, hot-reload |
| `mcv identify-dir photos/ --out results.csv` | batch identification of a bin / drawer / BOM shoot |
| `MCV_RATE_LIMIT_PER_MINUTE=60` | per-client cap on `/identify`; `MCV_API_TOKEN` protects `/admin/*` |

Nightly refresh (cron):

```
0 3 * * *  cd /srv/mcmaster-vision && mcv retrain --epochs 8 --reload-url http://localhost:8000 >> data/logs/retrain.log 2>&1
```

## 5. Checks before going live

```bash
mcv status                                  # ready: true, index backbone == settings backbone
mcv evaluate --max-queries 500              # synthetic-photo recall on the real gallery
mcv identify some_real_photo.jpg            # end-to-end on one photo
pytest                                      # 79 tests incl. a real browser run of the UI
```
