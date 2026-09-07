# Changelog

## 0.3.3

Bug-hunt release: four independent reviews plus API fuzzing and concurrency probes; every
finding below has a regression test.

- Security: quotes in catalog strings could break out of inline handlers (stored XSS);
  `/feedback` accepted unlimited uploads for any id; 500s leaked paths; thumbnails wrote
  unbounded cache files; part numbers are now URL-encoded everywhere.
- Serving: the CPU gate no longer exhausts the thread pool (health checks and pages stayed
  responsive under load); logging and photo storage run off the event loop; batch requests
  are charged per photo by the rate limiter; IPv6 clients are bucketed per /64; concurrent
  backups serialise; a rebuilt or restored index with a different backbone is refused
  instead of failing on every photo; `mcv retrain` with a different backbone writes a
  sibling index and prints the switch, never clobbering the live one; `mcv up` reuses a
  demo index with the backbone it was built with.
- Catalog: FTS maintenance was quadratic (20k re-ingest took 77 s; now index-backed); a
  quote-only search crashed; Excel "CSV UTF-8" (BOM) and ragged rows crashed ingest;
  `__MACOSX` / `.cache` clutter turned a photo folder into a folder-per-part source;
  `import-web` wiped existing images (now merges); robots.txt wildcards and stacked
  user-agents; JSON-LD `ImageObject` and breadcrumb shapes; unsafe page-supplied part
  numbers could write outside the image directory; redirected pages resolve links
  against their final URL.
- Index and training: FAISS asserted on an empty index; ids/vector counts are checked on
  load; ensemble members no longer all load one checkpoint; calibration keeps LIKELY
  below EXACT; the cached trainer crashed on the sampler's tail batch; grayscale
  confirmation photos and lower-case feedback folders are handled; the Claude reranker's
  schema was being rejected by structured outputs (now closed, no bounds, no refs).

## 0.3.2

- Size matching: a "Measure" tool in the phone UI (tap the two ends of a coin, a card edge
  or 1 inch on a ruler) gives the photo's scale; the object's extent is measured along its
  principal axes and compared with each candidate's catalog dimensions (length, OD, thread
  size), which separates look-alike SKUs that share one catalog image. The marked
  reference is excluded from the measurement. API: `mm_per_px` and `ref`; CLI `mcv identify --mm-per-px`.
- Service worker is network-first: phones no longer keep an old UI or a stale dashboard.
- `mcv validate` / `bootstrap` report how many parts carry a parseable dimension.

## 0.3.1

- Nothing gets lost: query photos wait on disk so a confirmation lands after a restart or
  on another worker; the request log reloads at boot; feedback re-confirmations replace
  instead of double-counting; `mcv backup` / `mcv restore` (+ `POST /admin/backup`,
  `GET /admin/backups`) bundle catalog, index, calibration, photos, logs and manifest.
- `mcv build-index --with-feedback` and `mcv retrain` embed confirmed photos as gallery
  entries, so a part photographed once is found again from that angle without training.
- The API picks up a rebuilt or restored index automatically (`MCV_AUTO_RELOAD`).
- Phone UI: confirmations made offline wait in an outbox and sync when the network is back.
- Reranker: a small usage prior from confirmation counts breaks ties toward parts people
  actually confirm (log-scaled; never overrides visual evidence).
- Dashboard: storage panel with sizes, ages, last backup and a "Back up now" button;
  `mcv doctor` flags a missing or stale backup.
- Uploads: decompression-bomb guard and reduced-scale JPEG decoding (a 12 MP JPEG decodes in
  116 ms instead of 165 ms on one CPU core);
  `/health` reports uptime, requests served and index/backbone mismatch.

## 0.3.0

- McMaster-inspired theme (dark-green header, yellow selection, dense spec tables) across the app and all pages; clearly labelled unofficial.
- New pages: `/browse` (taxonomy + part grid), `/part/{pn}` (gallery, specs, look-alike family), `/dashboard` (stats, tiers, recent identifications, build info).
- `/search` category filter and offset; Live ID; concurrency gate; `/categories`.

## 0.2.2

- Live ID: continuous identification overlay on the live camera; preview frames skip the request log.
- Concurrency gate (MCV_MAX_CONCURRENCY) and `/categories`.
- `mcv up`: one-command demo/serve with QR code; demo mode (sample parts identified live, printable sheet, `/connect`); install button; CORS setting; DEMO.md.
- Query-embedding cache; `mcv up` defaults to TinyCNN for a sub-minute first build.

## 0.2.1

- Shipped checkpoint retrained on 8,000 synthetic parts (held-out Recall@1 0.33 vs 0.29).
- Phone interface v3: live camera, compare lightbox, image strips, spec table, history, Fast/Accurate.
- Thumbnails, gzip, TTA modes, query-embedding cache; `mcv serve --qr/--https`; Caddy deployment.

## 0.2.0

Groundwork complete: everything except real McMaster-Carr imagery.

- Intake: `mcv validate`, normalisation and de-duplication, `mcv fetch-images`
  (URL lists), `mcv import-web` (product pages, polite), screenshot folders,
  `mcv enrich` (metadata for image-only drops), `mcv bootstrap` one-command
  pipeline with `data/manifest.json`.
- Retrieval: TTA multi-query, gallery augmentation, alpha query expansion,
  category prior, FAISS auto-selection above 50k vectors, parallel and
  incremental index builds with count-weighted centroids, atomic index swaps.
- Models: TinyCNN trained from scratch (shipped `assets/tinycnn_synthetic.pt`
  + model card), cached-view trainer (SupCon + classification, hard negatives,
  curriculum), ensemble backbone, CLIP / DINOv2 adapters, ONNX / TorchScript.
- Answers: calibrated tiers with precision-targeted thresholds, family answers
  with distinguishing attributes, attribute constraints (loose matching),
  category guesses, honest `notes`.
- Interface: camera-first PWA, several angles per query, one-tap confirmation
  feedback, text search, batch endpoint and `mcv identify-dir`, HEIC support.
- Operations: `doctor`, `status` (incl. `index_stale`), `metrics`, request log,
  `retrain` (cron) with held-out evaluation, hot reload, rate limit, API token,
  multi-worker serving, `export-dataset`, `review-unknowns`, RUNBOOK.md.
- Quality: 79 tests including a real-browser UI run; two code-review passes
  with all findings fixed.

## 0.1.0

Initial skeleton: catalog store, synthetic renderer, hash / CLIP / DINOv2
backbones, numpy / FAISS index, identification pipeline with Claude vision
reranker, FastAPI service and upload UI, training loop, CLI, tests, CI.
