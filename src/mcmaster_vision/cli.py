"""``mcv`` command-line interface."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer

from mcmaster_vision.config import Settings, load_settings

app = typer.Typer(
    help="McMaster-Vision: identify McMaster-Carr parts from photos.", no_args_is_help=True
)

_config_opt = typer.Option(None, "--config", "-c", help="YAML config file (env MCV_* overrides).")


def _settings(config: Path | None, **overrides) -> Settings:
    s = load_settings(config, **overrides)
    s.ensure_dirs()
    return s


@app.callback()
def _init(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )


@app.command()
def ingest(
    source: Path = typer.Argument(
        ..., help="parts.jsonl / parts.csv / directory of <part_number>/ folders"
    ),
    config: Path | None = _config_opt,
    strict: bool = typer.Option(False, help="Fail on missing image files instead of skipping them"),
) -> None:
    """Load a catalog export into the SQLite store."""
    from mcmaster_vision.catalog import CatalogStore
    from mcmaster_vision.catalog import ingest as _ingest

    s = _settings(config)
    with CatalogStore(s.catalog_db) as store:
        stats = _ingest(
            source, store, strict=strict, progress=lambda i: typer.echo(f"  {i} parts...")
        )
    typer.echo(json.dumps(stats))


@app.command()
def validate(
    source: Path = typer.Argument(..., help="parts.jsonl / parts.csv / image folder"),
    max_parts: int | None = typer.Option(None),
    no_image_check: bool = typer.Option(False, help="Skip decoding every image (faster)"),
) -> None:
    """Check a catalog drop before spending hours indexing it."""
    from mcmaster_vision.catalog.intake import validate_source

    rep = validate_source(source, check_images=not no_image_check, max_parts=max_parts)
    typer.echo(rep.to_json())
    if not rep.ok():
        typer.echo("PROBLEMS FOUND (see above)", err=True)
        raise typer.Exit(code=1)


@app.command("fetch-images")
def fetch_images(
    source: Path = typer.Argument(
        ..., help="JSONL/CSV whose rows carry image_urls (list or ';'-joined)"
    ),
    out: Path = typer.Option(
        Path("./data/catalog/parts_with_images.jsonl"), help="JSONL with image_paths to ingest next"
    ),
    config: Path | None = _config_opt,
    delay: float = typer.Option(0.2, help="Seconds between downloads"),
) -> None:
    """Download the images referenced by a spreadsheet export; resumable."""
    from mcmaster_vision.catalog.intake import fetch_image_urls, read_records, write_jsonl

    s = _settings(config)
    n = write_jsonl(
        fetch_image_urls(
            read_records(source),
            s.data_dir / "images" / "fetched",
            delay_s=delay,
            progress=lambda i: typer.echo(f"  {i} parts..."),
        ),
        out,
    )
    typer.echo(f"{n} records -> {out}. Now run: mcv ingest {out}")


@app.command()
def enrich(
    config: Path | None = _config_opt,
    only_missing: bool = typer.Option(True, help="Only parts without a name/category"),
    delay: float = typer.Option(1.5),
    limit: int | None = typer.Option(None),
) -> None:
    """Fill in names, categories and specs from McMaster-Carr product pages for parts that only have images."""
    from mcmaster_vision.catalog import CatalogStore
    from mcmaster_vision.catalog.web import WebImporter

    s = _settings(config)
    importer = WebImporter(s.data_dir / "images" / "web", delay_s=delay)
    updated = 0
    with CatalogStore(s.catalog_db) as store:
        todo = [
            p
            for p in store.iter_parts()
            if not only_missing or not p.category_path or p.name == p.part_number
        ]
        for i, part in enumerate(todo[:limit]):
            data = importer.fetch_page(part.part_number)
            if data is None:
                continue
            store.upsert(
                [
                    part.model_copy(
                        update={
                            "name": data.name or part.name,
                            "category_path": data.category_path or part.category_path,
                            "description": data.description or part.description,
                            "attributes": {**data.attributes, **part.attributes},
                            "url": data.url,
                        }
                    )
                ]
            )
            updated += 1
            if (i + 1) % 50 == 0:
                typer.echo(f"  {i + 1}/{len(todo)} ...")
    typer.echo(f"enriched {updated} of {len(todo)} parts")


@app.command()
def bootstrap(
    source: Path = typer.Argument(
        ..., help="Catalog drop: parts.jsonl / parts.csv / folder of images"
    ),
    config: Path | None = _config_opt,
    workers: int = typer.Option(1, help="Embedding processes"),
    normalize: bool = typer.Option(
        True, help="Copy images into data/images normalised (EXIF, RGB, <=1024 px, de-duplicated)"
    ),
    evaluate_: bool = typer.Option(
        True,
        "--evaluate/--no-evaluate",
        help="Measure Recall@K on augmented queries and fit calibration",
    ),
    max_eval_queries: int = typer.Option(300),
    skip_validate: bool = typer.Option(False),
) -> None:
    """Everything from a folder of images to a served, calibrated system: validate -> ingest -> index -> evaluate."""
    import time

    from mcmaster_vision.catalog import CatalogStore, open_source
    from mcmaster_vision.catalog.intake import normalise_parts, validate_source
    from mcmaster_vision.index import build_index
    from mcmaster_vision.models import PartEmbedder, load_backbone
    from mcmaster_vision.pipeline import Identifier
    from mcmaster_vision.pipeline.calibration import Calibration
    from mcmaster_vision.pipeline.manifest import update_manifest
    from mcmaster_vision.training import evaluate_retrieval

    s = _settings(config)
    t0 = time.time()
    if not skip_validate:
        typer.echo("1/4 validating ...")
        rep = validate_source(source, check_images=False)
        typer.echo(
            f"  {rep.parts} parts, {rep.with_images} with images, {rep.missing_files} missing files, {rep.duplicate_part_numbers} duplicate part numbers"
        )
        if not rep.ok():
            typer.echo(rep.to_json(), err=True)
            raise typer.Exit(code=1)
    typer.echo("2/4 ingesting ...")
    src = open_source(source)
    store = CatalogStore(s.catalog_db)
    parts_iter = (
        normalise_parts(
            src,
            s.data_dir / "images" / "catalog",
            progress=lambda i: typer.echo(f"  {i} parts normalised"),
        )
        if normalize
        else src
    )
    n = store.upsert(parts_iter)
    typer.echo(f"  {n} parts in {s.catalog_db}")
    typer.echo("3/4 embedding + indexing ...")
    embedder = PartEmbedder(load_backbone(s))
    t = time.time()
    idx = build_index(
        store,
        embedder,
        "auto",
        out_path=s.index_path,
        image_size=s.image_size,
        gallery_augment=s.index_gallery_augment,
        workers=workers,
        settings_dump=s.model_dump(mode="json"),
        progress=lambda d, tot: typer.echo(f"  {d}/{tot} parts embedded"),
    )
    typer.echo(f"  {idx.stats().vectors} vectors ({idx.backend}) in {time.time() - t:.0f}s")
    report = None
    if evaluate_:
        typer.echo("4/4 evaluating + calibrating ...")
        ident = Identifier(
            store,
            idx,
            embedder,
            top_k=s.index_top_k,
            qe_k=s.query_expansion_k,
            image_size=s.image_size,
        )
        report = evaluate_retrieval(ident, store, max_queries=max_eval_queries)
        cal = Calibration.fit_temperature(report.score_lists, report.correct_idx).fit_thresholds(
            report.score_lists, report.correct_idx
        )
        cal.save(s.model_dir / "calibration.json")
        typer.echo(
            f"  Recall@1 {report.recall_at.get(1)}  Recall@10 {report.recall_at.get(10)}  MRR {report.mrr}  calibration T={cal.temperature}"
        )
    update_manifest(
        s,
        source=str(source),
        parts=n,
        index=idx.stats().model_dump(mode="json"),
        index_path=str(s.index_path),
        backbone=embedder.version,
        evaluation=(report.to_json() and __import__("json").loads(report.to_json()))
        if report
        else None,
        bootstrap_seconds=round(time.time() - t0, 1),
    )
    store.close()
    typer.echo(f"done in {time.time() - t0:.0f}s. Serve with: mcv serve")


@app.command()
def status(config: Path | None = _config_opt) -> None:
    """What is built: catalog, index, calibration, feedback, manifest."""
    from mcmaster_vision.pipeline.manifest import status as _status

    typer.echo(json.dumps(_status(_settings(config)), indent=2, default=str))


@app.command()
def doctor(config: Path | None = _config_opt) -> None:
    """Check the environment: optional dependencies, checkpoint, index/backbone match, disk, GPU."""
    import importlib
    import shutil

    s = _settings(config)
    checks: list[tuple[str, bool, str]] = []

    def dep(name: str, extra: str) -> None:
        try:
            importlib.import_module(name)
            checks.append((f"{name}", True, "installed"))
        except ImportError:
            checks.append((f"{name}", False, f"pip install -e '.[{extra}]'"))

    for name, extra in (
        ("torch", "ml"),
        ("open_clip", "ml"),
        ("timm", "ml"),
        ("faiss", "faiss"),
        ("easyocr", "ocr"),
        ("anthropic", "llm"),
        ("rembg", "segment"),
        ("pillow_heif", "heic"),
    ):
        dep(name, extra)
    try:
        import torch

        checks.append(
            (
                "cuda",
                torch.cuda.is_available(),
                f"{torch.cuda.device_count()} GPU(s)" if torch.cuda.is_available() else "CPU only",
            )
        )
    except ImportError:
        pass
    if s.backbone in ("tinycnn", "ensemble", "openclip", "dinov2"):
        ok = s.backbone_checkpoint is not None and Path(s.backbone_checkpoint).exists()
        checks.append(
            (
                "checkpoint",
                ok,
                str(s.backbone_checkpoint)
                if s.backbone_checkpoint
                else "MCV_BACKBONE_CHECKPOINT unset (pretrained / random weights)",
            )
        )
    checks.append(("catalog", s.catalog_db.exists(), str(s.catalog_db)))
    meta = s.index_path / "meta.json"
    checks.append(("index", meta.exists(), str(s.index_path)))
    if meta.exists() and s.catalog_db.exists():
        built_with = json.loads(meta.read_text()).get("backbone")
        try:
            from mcmaster_vision.models import load_backbone

            current = load_backbone(s).version
            checks.append(
                (
                    "index/backbone match",
                    built_with == current,
                    f"index={built_with} settings={current}",
                )
            )
        except Exception as e:  # noqa: BLE001
            checks.append(("backbone loads", False, str(e)[:120]))
    checks.append(
        (
            "calibration",
            (s.model_dir / "calibration.json").exists(),
            "run mcv evaluate --fit-calibration"
            if not (s.model_dir / "calibration.json").exists()
            else "fitted",
        )
    )
    free_gb = shutil.disk_usage(s.data_dir if s.data_dir.exists() else Path(".")).free / 1e9
    checks.append(("disk", free_gb > 5, f"{free_gb:.1f} GB free under {s.data_dir}"))
    if s.rerank_llm_enabled:
        import os

        checks.append(
            (
                "ANTHROPIC_API_KEY",
                bool(os.environ.get("ANTHROPIC_API_KEY")),
                "needed for the vision reranker",
            )
        )
    width = max(len(c[0]) for c in checks)
    for name, ok, detail in checks:
        typer.echo(f"{'OK  ' if ok else 'MISS'} {name.ljust(width)}  {detail}")
    typer.echo(
        "ready"
        if all(ok for n, ok, _ in checks if n in ("catalog", "index"))
        else "not ready: build the catalog and index (mcv bootstrap)"
    )


@app.command()
def retrain(
    config: Path | None = _config_opt,
    train_config: Path = typer.Option(Path("configs/train_tinycnn.yaml"), help="Training recipe"),
    epochs: int | None = typer.Option(None),
    reload_url: str | None = typer.Option(
        None, help="e.g. http://localhost:8000 - POST /admin/reload after rebuilding"
    ),
) -> None:
    """Scheduled refresh: train on catalog + confirmed photos, rebuild the index, refit calibration, reload the API."""
    from mcmaster_vision.catalog import CatalogStore
    from mcmaster_vision.index import build_index
    from mcmaster_vision.models import PartEmbedder, load_backbone
    from mcmaster_vision.pipeline import Identifier
    from mcmaster_vision.pipeline.calibration import Calibration
    from mcmaster_vision.pipeline.feedback import FeedbackStore
    from mcmaster_vision.pipeline.manifest import update_manifest
    from mcmaster_vision.training import evaluate_retrieval
    from mcmaster_vision.training.train import load_train_config
    from mcmaster_vision.training.train import train as _train

    s = _settings(config)
    cfg = load_train_config(train_config)
    if epochs:
        cfg["epochs"] = epochs
    cfg["output_dir"] = str(s.model_dir / "retrain")
    extra = FeedbackStore(s.queries_dir).labelled_images()
    typer.echo(
        f"1/3 training on catalog + {sum(len(v) for v in extra.values())} confirmed photos ..."
    )
    with CatalogStore(s.catalog_db) as store:
        ckpt = _train(store, cfg, extra_images=extra)
        s = s.model_copy(
            update={
                "backbone_checkpoint": ckpt,
                "backbone": cfg["backbone"],
                "backbone_pretrained": cfg.get("backbone_pretrained", s.backbone_pretrained),
            }
        )
        typer.echo("2/3 rebuilding index ...")
        embedder = PartEmbedder(load_backbone(s))
        idx = build_index(
            store,
            embedder,
            "auto",
            out_path=s.index_path,
            image_size=s.image_size,
            gallery_augment=s.index_gallery_augment,
        )
        typer.echo("3/3 evaluating + calibrating ...")
        ident = Identifier(store, idx, embedder, top_k=s.index_top_k, image_size=s.image_size)
        rep = evaluate_retrieval(
            ident, store, query_dir=s.queries_dir if extra else None, max_queries=500
        )
        cal = Calibration.fit_temperature(rep.score_lists, rep.correct_idx).fit_thresholds(
            rep.score_lists, rep.correct_idx
        )
        cal.save(s.model_dir / "calibration.json")
    update_manifest(
        s,
        checkpoint=str(ckpt),
        index=idx.stats().model_dump(mode="json"),
        retrain_eval=json.loads(rep.to_json()),
    )
    typer.echo(
        f"checkpoint {ckpt}; set MCV_BACKBONE_CHECKPOINT={ckpt}. Recall@1 {rep.recall_at.get(1)} on {rep.queries} queries"
    )
    if reload_url:
        import httpx

        r = httpx.post(
            reload_url.rstrip("/") + "/admin/reload",
            headers={"X-API-Token": s.api_token or ""},
            timeout=120,
        )
        typer.echo(f"reload -> {r.status_code}")


@app.command("review-unknowns")
def review_unknowns(
    config: Path | None = _config_opt,
    out: Path = typer.Option(Path("unknowns_review.html")),
    top_n: int = typer.Option(5),
) -> None:
    """Contact sheet of 'none of these' photos with their current top candidates, for labelling."""
    import base64
    import html

    from mcmaster_vision.pipeline import load_identifier
    from mcmaster_vision.pipeline.feedback import UNKNOWN_DIR

    s = _settings(config)
    folder = s.queries_dir / UNKNOWN_DIR
    photos = (
        sorted(
            p for p in folder.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
        )
        if folder.exists()
        else []
    )
    if not photos:
        typer.echo("no unknown photos to review")
        return
    ident = load_identifier(s)
    rows = []
    for photo in photos:
        res = ident.identify_path(photo, top_n=top_n)
        b64 = base64.b64encode(photo.read_bytes()).decode()
        cands = "".join(
            f"<li><b>{html.escape(c.part_number)}</b> {html.escape(c.name)} ({c.confidence:.0%})</li>"
            for c in res.candidates
        )
        rows.append(
            f"<tr><td><img src='data:image/jpeg;base64,{b64}' width='160'><br>{html.escape(photo.name)}</td><td>{html.escape(res.tier.value)}<ol>{cands}</ol>"
            f"<p>label: <code>mv {html.escape(str(photo))} {html.escape(str(s.queries_dir))}/&lt;PART_NUMBER&gt;/</code></p></td></tr>"
        )
    out.write_text(
        "<html><body><h1>Unlabelled photos</h1><table border=1 cellpadding=8>"
        + "".join(rows)
        + "</table></body></html>",
        encoding="utf-8",
    )
    typer.echo(f"{len(photos)} photos -> {out}")


@app.command("export-dataset")
def export_dataset(
    out: Path = typer.Argument(..., help="Destination folder"),
    config: Path | None = _config_opt,
    include_feedback: bool = typer.Option(True, help="Add confirmed real photos as extra images"),
    query_set: int = typer.Option(
        0, help="Also write N photo-style augmented queries per part under queries/"
    ),
    image_size: int = typer.Option(0, help="Resize exported images to this side (0 = keep)"),
) -> None:
    """Export the catalog as a plain image-folder dataset (images/<part>/..., labels.csv, parts.jsonl)
    for training on another machine or service."""
    import csv
    import shutil

    from PIL import Image

    from mcmaster_vision.catalog import CatalogStore
    from mcmaster_vision.data.augment import AugmentConfig, PhotoAugmenter
    from mcmaster_vision.pipeline.feedback import FeedbackStore

    s = _settings(config)
    out.mkdir(parents=True, exist_ok=True)
    (out / "images").mkdir(exist_ok=True)
    extra = FeedbackStore(s.queries_dir).labelled_images() if include_feedback else {}
    aug = PhotoAugmenter(AugmentConfig.evaluation(), seed=0) if query_set else None
    n_img = n_parts = 0
    with (
        CatalogStore(s.catalog_db) as store,
        open(out / "labels.csv", "w", newline="", encoding="utf-8") as lf,
        open(out / "parts.jsonl", "w", encoding="utf-8") as pf,
    ):
        w = csv.writer(lf)
        w.writerow(["path", "part_number", "family_id", "category", "source"])
        for part in store.iter_parts(with_images_only=True):
            n_parts += 1
            folder = out / "images" / part.part_number
            folder.mkdir(exist_ok=True)
            sources = [(p, "catalog") for p in part.image_paths] + [
                (p, "photo") for p in extra.get(part.part_number, [])
            ]
            rel_paths = []
            for i, (src, kind) in enumerate(sources):
                dst = folder / f"{part.part_number}_{kind}_{i}.jpg"
                try:
                    if image_size:
                        im = Image.open(src).convert("RGB")
                        im.thumbnail((image_size, image_size))
                        im.save(dst, quality=92)
                    else:
                        shutil.copyfile(src, dst)
                except OSError:
                    continue
                rel = dst.relative_to(out).as_posix()
                rel_paths.append(rel)
                w.writerow(
                    [
                        rel,
                        part.part_number,
                        part.family_id or "",
                        " > ".join(part.category_path),
                        kind,
                    ]
                )
                n_img += 1
            if aug is not None and part.image_paths:
                qdir = out / "queries" / part.part_number
                qdir.mkdir(parents=True, exist_ok=True)
                for q in range(query_set):
                    try:
                        aug(
                            Image.open(part.image_paths[q % len(part.image_paths)]),
                            out_size=image_size or 224,
                        ).save(qdir / f"q{q}.jpg", quality=90)
                    except OSError:
                        pass
            pf.write(part.model_copy(update={"image_paths": rel_paths}).model_dump_json() + "\n")
    typer.echo(
        f"{n_parts} parts, {n_img} images -> {out} (labels.csv, parts.jsonl{', queries/' if query_set else ''})"
    )


@app.command("import-web")
def import_web(
    items: list[str] = typer.Argument(None, help="Part numbers or product URLs"),
    file: Path | None = typer.Option(
        None, "--file", "-f", help="Text file with one part number / URL per line"
    ),
    config: Path | None = _config_opt,
    delay: float = typer.Option(1.5, help="Seconds between requests"),
    max_images: int = typer.Option(4),
    no_robots: bool = typer.Option(False, help="Do not consult robots.txt"),
) -> None:
    """Fetch McMaster-Carr product pages, download their images, and add the parts to the store."""
    from mcmaster_vision.catalog import CatalogStore
    from mcmaster_vision.catalog import ingest as _ingest
    from mcmaster_vision.catalog.web import WebImporter, WebSource, read_items

    s = _settings(config)
    todo = list(items or []) + (read_items(file) if file else [])
    if not todo:
        raise typer.BadParameter("give part numbers / URLs or --file")
    importer = WebImporter(
        s.data_dir / "images" / "web",
        delay_s=delay,
        max_images=max_images,
        respect_robots=not no_robots,
    )
    with CatalogStore(s.catalog_db) as store:
        stats = _ingest(WebSource(importer, todo), store)
    typer.echo(json.dumps(stats))
    typer.echo("Now run: mcv build-index")


@app.command("build-index")
def build_index_cmd(
    config: Path | None = _config_opt,
    backend: str | None = typer.Option(
        None, help="numpy | faiss | auto (faiss above 50k vectors when installed)"
    ),
    batch_size: int = typer.Option(256),
    workers: int = typer.Option(1, help="Embedding processes (CPU boxes: one per core)"),
    only_new: bool = typer.Option(
        False, help="Add parts missing from the existing index instead of rebuilding"
    ),
    gallery_augment: int | None = typer.Option(
        None, help="Photo-style variants per catalog image (default from config)"
    ),
) -> None:
    """Embed every catalog image and write the vector index."""
    import time

    from mcmaster_vision.catalog import CatalogStore
    from mcmaster_vision.index import build_index
    from mcmaster_vision.models import PartEmbedder, load_backbone
    from mcmaster_vision.pipeline.manifest import update_manifest

    s = _settings(config, index_backend=None if backend in (None, "auto") else backend)
    ga = s.index_gallery_augment if gallery_augment is None else gallery_augment
    embedder = PartEmbedder(load_backbone(s))
    t = time.time()
    with CatalogStore(s.catalog_db) as store:
        idx = build_index(
            store,
            embedder,
            backend or s.index_backend,
            batch_size=batch_size,
            out_path=s.index_path,
            image_size=s.image_size,
            gallery_augment=ga,
            only_new=only_new,
            workers=workers,
            settings_dump=s.model_dump(mode="json"),
            progress=lambda d, t: typer.echo(f"  {d}/{t} parts embedded"),
        )
    stats = idx.stats()
    update_manifest(
        s,
        index=stats.model_dump(mode="json"),
        index_build_seconds=round(time.time() - t, 1),
        index_path=str(s.index_path),
    )
    typer.echo(stats.model_dump_json(indent=2))


@app.command()
def identify(
    image: Path = typer.Argument(..., exists=True),
    config: Path | None = _config_opt,
    top_n: int = typer.Option(5),
    llm: bool = typer.Option(False, help="Use the Claude vision reranker"),
) -> None:
    """Identify the part in a photo."""
    from mcmaster_vision.pipeline import load_identifier

    s = _settings(config, rerank_llm_enabled=llm or None)
    result = load_identifier(s).identify_path(image, top_n=top_n, use_llm=llm or None)
    typer.echo(result.model_dump_json(indent=2))


@app.command("identify-dir")
def identify_dir(
    folder: Path = typer.Argument(
        ..., exists=True, file_okay=False, help="Folder of photos, one part per photo"
    ),
    out: Path = typer.Option(Path("identify_results.csv"), help="CSV of results"),
    config: Path | None = _config_opt,
    top_n: int = typer.Option(3),
) -> None:
    """Identify every photo in a folder (a bin, a drawer, a BOM shoot) and write a CSV."""
    import csv

    from mcmaster_vision.pipeline import load_identifier

    ident = load_identifier(_settings(config))
    exts = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".bmp"}
    files = sorted(f for f in folder.iterdir() if f.suffix.lower() in exts)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "tier", "best", "confidence", "family", "candidates", "error"])
        for f in files:
            try:
                res = ident.identify_path(f, top_n=top_n)
            except OSError as e:
                w.writerow([f.name, "", "", "", "", "", str(e)])
                continue
            w.writerow(
                [
                    f.name,
                    res.tier.value,
                    res.best.part_number if res.best else "",
                    res.best.confidence if res.best else "",
                    res.family.family_id if res.family else "",
                    " ".join(c.part_number for c in res.candidates),
                    "",
                ]
            )
            typer.echo(f"{f.name}: {res.tier.value} {res.best.part_number if res.best else '-'}")
    typer.echo(f"{len(files)} photos -> {out}")


@app.command()
def serve(
    config: Path | None = _config_opt,
    host: str | None = typer.Option(None),
    port: int | None = typer.Option(None),
    workers: int = typer.Option(1, help="Uvicorn worker processes (each loads the index)"),
) -> None:
    """Run the HTTP API + upload UI."""
    from mcmaster_vision.api.app import run

    run(_settings(config), host=host, port=port, workers=workers)


@app.command()
def train(
    config: Path = typer.Option(Path("configs/train_openclip.yaml"), "--config", "-c"),
    runtime_config: Path | None = typer.Option(None, help="Runtime YAML for catalog location"),
    query_dir: Path | None = typer.Option(
        None,
        help="Labelled real photos (<dir>/<part_number>/*.jpg, e.g. the feedback store) to add as training views",
    ),
) -> None:
    """Fine-tune a backbone on the catalog (requires the [ml] extra)."""
    from mcmaster_vision.catalog import CatalogStore
    from mcmaster_vision.pipeline.feedback import FeedbackStore
    from mcmaster_vision.training.train import load_train_config
    from mcmaster_vision.training.train import train as _train

    s = _settings(runtime_config)
    cfg = load_train_config(config)
    extra = FeedbackStore(query_dir or s.queries_dir).labelled_images()
    if extra:
        typer.echo(
            f"adding {sum(len(v) for v in extra.values())} real photos for {len(extra)} parts as training views"
        )
    with CatalogStore(s.catalog_db) as store:
        ckpt = _train(store, cfg, extra_images=extra)
    typer.echo(f"best checkpoint: {ckpt}")


@app.command()
def evaluate(
    config: Path | None = _config_opt,
    query_dir: Path | None = typer.Option(
        None, help="Real labelled photos: <dir>/<part_number>/*.jpg"
    ),
    max_queries: int | None = typer.Option(None),
    fit_calibration: bool = typer.Option(
        False, help="Fit softmax temperature and save calibration.json"
    ),
    out: Path | None = typer.Option(None, help="Write the JSON report here"),
) -> None:
    """Measure Recall@K / MRR (synthetic photo-style queries unless --query-dir)."""
    from mcmaster_vision.catalog import CatalogStore
    from mcmaster_vision.pipeline import load_identifier
    from mcmaster_vision.pipeline.calibration import Calibration
    from mcmaster_vision.training import evaluate_retrieval

    s = _settings(config)
    ident = load_identifier(s)
    with CatalogStore(s.catalog_db) as store:
        report = evaluate_retrieval(ident, store, query_dir=query_dir, max_queries=max_queries)
    typer.echo(report.to_json())
    if fit_calibration:
        cal = Calibration.fit_temperature(report.score_lists, report.correct_idx)
        cal = cal.fit_thresholds(report.score_lists, report.correct_idx)
        cal.save(s.model_dir / "calibration.json")
        typer.echo(
            f"calibration temperature={cal.temperature} saved to {s.model_dir / 'calibration.json'}"
        )
    if out:
        out.write_text(report.to_json(), encoding="utf-8")


@app.command()
def demo(
    parts: int = typer.Option(300, help="Synthetic parts to generate"),
    images_per_part: int = typer.Option(3),
    data_dir: Path = typer.Option(Path("./data/demo")),
    serve_: bool = typer.Option(True, "--serve/--no-serve", help="Start the API afterwards"),
    port: int = typer.Option(8000),
    gallery_augment: int = typer.Option(2, help="Photo-style variants indexed per catalog image"),
    backbone: str = typer.Option(
        "hash", help="hash | tinycnn | ensemble (tinycnn+hash) | openclip | dinov2"
    ),
    checkpoint: Path | None = typer.Option(
        None, help="Fine-tuned checkpoint (.pt) for the backbone"
    ),
    train_epochs: int = typer.Option(
        0, help="Train the backbone on the synthetic catalog first (needs torch)"
    ),
) -> None:
    """Generate a synthetic catalog, index it, evaluate, and (optionally) serve it."""
    from PIL import Image

    from mcmaster_vision.catalog import CatalogStore
    from mcmaster_vision.catalog import ingest as _ingest
    from mcmaster_vision.data import PhotoAugmenter, SyntheticCatalog
    from mcmaster_vision.index import build_index
    from mcmaster_vision.models import PartEmbedder, load_backbone
    from mcmaster_vision.pipeline import Identifier
    from mcmaster_vision.training import evaluate_retrieval

    if checkpoint is None and train_epochs == 0 and backbone in ("tinycnn", "ensemble"):
        shipped = Path(__file__).resolve().parents[2] / "assets" / "tinycnn_synthetic.pt"
        if shipped.exists():
            checkpoint = shipped
            typer.echo(f"using shipped checkpoint {shipped}")
    s = Settings(
        data_dir=data_dir,
        catalog_db=data_dir / "catalog.sqlite",
        index_dir=data_dir / "index",
        model_dir=data_dir / "models",
        backbone=backbone,  # type: ignore[arg-type]
        backbone_pretrained="none" if backbone in ("tinycnn", "ensemble") else None,
        backbone_checkpoint=checkpoint,
        index_backend="numpy",
    )
    if backbone == "openclip":
        s = s.model_copy(update={"backbone_pretrained": Settings().backbone_pretrained})
    s.ensure_dirs()
    typer.echo(f"1/4 generating {parts} synthetic parts ...")
    jsonl = data_dir / "parts.jsonl"
    SyntheticCatalog(parts, images_per_part, seed=0).write_jsonl(data_dir / "images", jsonl)

    typer.echo("2/4 ingesting ...")
    store = CatalogStore(s.catalog_db)
    _ingest(jsonl, store)

    if train_epochs > 0:
        from mcmaster_vision.training.train import load_train_config
        from mcmaster_vision.training.train import train as _train

        # "ensemble" = learned tinycnn + hash: train the learned member, then fuse.
        trainable = "tinycnn" if backbone == "ensemble" else backbone
        typer.echo(f"2b/4 training {trainable} for {train_epochs} epochs ...")
        cfg_path = Path("configs") / f"train_{trainable}.yaml"
        cfg = load_train_config(cfg_path if cfg_path.exists() else None)
        cfg.update(
            {
                "backbone": trainable,
                "epochs": train_epochs,
                "output_dir": str(s.model_dir / trainable),
            }
        )
        s = s.model_copy(update={"backbone_checkpoint": _train(store, cfg)})

    typer.echo("3/4 embedding + indexing ...")
    embedder = PartEmbedder(load_backbone(s))
    index = build_index(
        store, embedder, "numpy", out_path=s.index_path, gallery_augment=gallery_augment
    )
    ident = Identifier(store, index, embedder, qe_k=s.query_expansion_k)

    typer.echo("4/4 evaluating on photo-style augmented queries + fitting calibration ...")
    report = evaluate_retrieval(ident, store, max_queries=min(parts, 200))
    typer.echo(report.to_json())
    from mcmaster_vision.pipeline.calibration import Calibration

    cal = Calibration.fit_temperature(report.score_lists, report.correct_idx)
    cal.save(s.model_dir / "calibration.json")
    ident.calibration = cal
    typer.echo(f"calibration temperature={cal.temperature} -> {s.model_dir / 'calibration.json'}")

    sample = next(store.iter_parts(with_images_only=True))
    q = PhotoAugmenter(seed=7)(Image.open(sample.image_paths[0]))
    q_path = data_dir / "sample_query.jpg"
    q.save(q_path)
    res = ident.identify(q)
    typer.echo(
        f"\nsample query {q_path} (truth {sample.part_number}) -> {res.tier.value} "
        f"{res.best.part_number if res.best else None}"
    )
    typer.echo(
        f"\nRe-run against this catalog with:\n  MCV_CATALOG_DB={s.catalog_db} MCV_INDEX_DIR={s.index_dir} mcv identify {q_path}"
    )

    if serve_:
        import uvicorn

        from mcmaster_vision.api.app import create_app

        typer.echo(f"\nserving UI at http://127.0.0.1:{port}")
        uvicorn.run(create_app(s, ident), host="127.0.0.1", port=port)


if __name__ == "__main__":
    app()
