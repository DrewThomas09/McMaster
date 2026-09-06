# McMaster-Vision

Photo -> McMaster-Carr part number. Retrieval pipeline (embed -> vector search -> rerank -> calibrate), not a classifier.
Read `ARCHITECTURE.md` for design and measured numbers, `RUNBOOK.md` for operating it.

## Commands
- `pip install -e ".[dev]"` then `python3 -m pytest -q` (56+ tests; browser test skips without Playwright)
- `ruff check src tests && ruff format src tests`
- `mcv demo --parts 300 --no-serve` end-to-end smoke on the synthetic catalog; `mcv --help` for the rest

## Layout
`src/mcmaster_vision/`: `catalog/` (sources, web importer, intake, SQLite store), `data/` (augmentation, synthetic renderer),
`models/` (hash | tinycnn | openclip | dinov2 | ensemble backbones), `index/` (numpy | FAISS, builder),
`pipeline/` (preprocess, retrieve, rerank, calibrate, identify, feedback), `training/` (cached trainer, eval),
`api/` (FastAPI + one-photo UI), `cli.py` (`mcv`). Shipped model: `assets/tinycnn_synthetic.pt`.

## Conventions
- No proprietary McMaster data in the repo; the synthetic renderer (`data/synthetic.py`) is the test catalog.
- Every backbone implements `embed(images) -> (N, dim)` L2-normalised; gallery and query go through the same preprocessing.
- Heavy deps (torch, faiss, easyocr, anthropic) are optional extras and imported lazily; tests skip when missing.
- Training uses spawned worker processes; entry points must be importable scripts, not heredocs.
- Recall numbers in docs come from `scripts/`-style runs on the synthetic catalog; keep them honest and dated.
