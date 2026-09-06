# Changelog

## 0.2.2

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
