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
) -> None:
    """Run the HTTP API + upload UI."""
    from mcmaster_vision.api.app import run

    run(_settings(config), host=host, port=port)


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
