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

`mcv up` is the shortcut: serves the built catalog on the network with a QR
code and demo mode on (`/demo/*`: sample parts, printable sheet, `/connect`).
Set `MCV_DEMO_MODE=false` (the default for `mcv serve`) in production.

Phones: `mcv serve --host 0.0.0.0 --qr` on the same network, `--https` for the
installable app and live camera, or `deploy/` for tunnels and a real domain
(see `deploy/README.md`).

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
* **Purchases are the strongest confirmations.** The storefront's `POST /checkout` files a
  `checkout` confirmation for every item that came from a photo (weight 3 vs 2 for a tap).
  `mcv learn` folds new confirmations into the index incrementally (seconds; the API picks
  it up by itself) and runs a full `mcv retrain` once `MCV_LEARN_RETRAIN_AFTER` (default 50)
  new confirmations arrived. Cron it hourly: `0 * * * * mcv learn` (`--index-only` never
  retrains; `--retrain-after N` and `--epochs N` override the recipe for a quick check). The dashboard's **Learning loop** panel and `GET /analytics` show the funnel
  (identify -> cart -> checkout), predicted-vs-bought confusion pairs, tier precision on
  what was bought, and a plain-language issues list with the command that fixes each.
  Backend tracking lives in `data/logs/events.jsonl` (every identify, cart, checkout,
  feedback and error) and `data/logs/orders.jsonl`.
* **Self-run the demo before customers do.** `mcv simulate --customers 40 --learn` walks
  synthetic customers through photo -> cart -> checkout in-process, prints the issues the
  analytics found, learns from the purchases and reports before/after top-1 on the bought
  photos and on new photos of the same parts. It runs on a scratch copy of the index and
  calibration (the catalog is shared read-only), so synthetic purchases never become real
  evidence; `--live` writes into the deployment on purpose. Run it after any change to the
  reranker, calibration or index settings; `--json` for a machine-readable report.
* **What a purchase is worth.** A checkout files the photo with weight 3, a "This is it"
  tap 2, an add-to-cart 1 (an abandoned cart still teaches a little; the checkout upgrades
  the same photo). The usage prior, `mcv learn` and `mcv retrain` all read those weights;
  the gallery holds each photo once. Before a photo joins the gallery it is scored once
  against the current index; from 30 such samples with at least 5 wrong answers among
  them, `mcv learn` refits the tiers on real outcomes (`models/calibration_samples.jsonl`).
* **Carts and orders.** Carts are one small file per phone under `data/logs/carts/`
  (every worker sees them, pruned after 14 days); orders append to `data/logs/orders.jsonl`.
  `GET /orders?client_id=...` shows a phone its own orders; the full list needs the API
  token. A request id is a random token only the phone that took the photo holds, which
  is what lets a purchase label that photo. Two learns cannot overlap (`data/learn.lock`;
  the dashboard button answers 409 while cron's `mcv learn` runs).
* **New SKUs**: `mcv ingest new_parts.jsonl && mcv build-index --only-new` embeds only the additions.
  `GET /status` reports `index_stale: true` whenever the catalog changed after the
  index was built. Index writes are atomic (temp dir + swap), so rebuilding while
  serving and then `POST /admin/reload` is safe.
* **Removed or changed SKUs**: `mcv build-index` (full rebuild, same command).
* **Better model**: set `MCV_BACKBONE=openclip` (or `dinov2`), `MCV_BACKBONE_CHECKPOINT=...`, rebuild the index.
  The API refuses to serve an index built with a different backbone than it is configured
  for (it would fail on every photo); `mcv retrain` with a recipe for another backbone writes
  its index and calibration to sibling directories (`index-<backbone>`, `models-<backbone>`)
  and prints the environment to switch to. "Different" means the full model version,
  checkpoint included: a retrain from the shipped checkpoint is a switch too.
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
| `mcv selfcheck` | one command on a fresh machine: environment, build, identify, coin + measure, feedback, backup, restore; PASS/FAIL per step |
| `mcv doctor` (`--json`) | optional deps, GPU, checkpoint, index/backbone match, index freshness, calibration, disk |
| `mcv status` / `GET /status` | what is built, from what, and how well it measured |
| `GET /metrics` | request volume, tier mix, latency p50/p95, confirmed top-1 rate |
| `mcv review-unknowns` | HTML contact sheet of "none of these" photos + current candidates, for labelling |
| `mcv retrain --reload-url http://localhost:8000` | train on catalog + confirmed photos, rebuild, refit, hot-reload |
| `mcv identify-dir photos/ --out results.csv --coin "US quarter"` | batch identification of a bin / drawer / BOM shoot; with a coin in each photo, sizes and thread pitch are matched and written to the CSV |
| `mcv import-pages catalog.txt --first-page 4` | parts from the OCR text of printed catalog pages (sizes, materials, fitting types, prices) |
| `MCV_RATE_LIMIT_PER_MINUTE=60` | per-client cap on `/identify`; `MCV_API_TOKEN` protects `/admin/*` |
| `MCV_MAX_CONCURRENCY=4` | simultaneous identifications (default: CPU count); live previews queue behind real photos |
| `mcv build-index --with-feedback` | confirmed photos become gallery entries (no training needed) |
| `MCV_FORWARDED_ALLOW_IPS="*"` | trust X-Forwarded-For from the reverse proxy so rate limits are per phone, not per proxy |
| `mcv backup` / `mcv restore <tar.gz>` | one archive of everything learned at run time; see section 4c |
| `POST /admin/backup`, `GET /admin/backups` | the same from the API (token-protected) |

Nightly refresh (cron):

```
0 2 * * *  cd /srv/mcmaster-vision && mcv backup >> data/logs/backup.log 2>&1
0 3 * * *  cd /srv/mcmaster-vision && mcv retrain --epochs 8 --reload-url http://localhost:8000 >> data/logs/retrain.log 2>&1
```

## 4c. Nothing gets lost

Everything the system learns while running is on disk and covered by one command:

| state | where | written by |
|---|---|---|
| catalog | `data/catalog.sqlite` (WAL) | `mcv ingest` / `import-web` / `bootstrap` |
| vector index | `data/index/parts/` (atomic swap) | `mcv build-index` / `retrain` |
| calibration | `data/models/calibration.json` | `mcv evaluate --fit-calibration` / `retrain` |
| confirmed photos | `data/queries/<part>/`, `feedback.jsonl` | `POST /feedback` (a re-confirmation replaces the earlier answer) |
| request log | `data/logs/requests.jsonl` | `POST /identify`; reloaded at boot so `/metrics` keeps its history |
| recent query photos | `data/cache/recent/` (7 days / 500) | `POST /identify`; lets a confirmation land after a restart or on another worker |
| manifest | `data/manifest.json` | every build |

```bash
mcv backup                         # data/backups/mcv-<timestamp>.tar.gz (+ BACKUP.json inventory)
mcv backup --out /mnt/nas/mcv/     # elsewhere
mcv restore data/backups/mcv-20260906T120000Z.tar.gz          # everything
mcv restore backup.tar.gz --only queries --only calibration    # just some of it
```

Restores are atomic per component (extract, then swap). The running API notices a
restored or rebuilt index within 15 s (`MCV_AUTO_RELOAD`, on by default) so no restart
and no token are needed. Add `mcv backup` before the nightly `retrain` in cron.

## 5. Checks before going live

```bash
mcv status                                  # ready: true, index backbone == settings backbone
mcv evaluate --max-queries 500              # synthetic-photo recall on the real gallery
mcv identify some_real_photo.jpg            # end-to-end on one photo
python3 -m pytest -q                        # 100+ tests incl. real browser runs of the UI
```
