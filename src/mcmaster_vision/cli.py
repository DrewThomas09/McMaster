"""``mcv`` command-line interface."""

from __future__ import annotations

import json
import logging
import re
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
            f"  {rep.parts} parts, {rep.with_images} with images, {rep.with_dimensions} with a "
            f"length/OD/thread spec (size matching), {rep.missing_files} missing files, "
            f"{rep.duplicate_part_numbers} duplicate part numbers"
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
        retrain_eval=None,  # a fresh build supersedes an older retrain's numbers
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
def doctor(
    config: Path | None = _config_opt,
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output"),
) -> None:
    """Check the environment: optional dependencies, checkpoint, index/backbone match, disk, GPU."""

    s = _settings(config)
    checks = _doctor_checks(s)
    if as_json:
        typer.echo(
            json.dumps(
                {
                    "checks": [{"name": n, "ok": ok, "detail": d} for n, ok, d in checks],
                    "ready": all(ok for n, ok, _ in checks if n in ("catalog", "index")),
                },
                indent=2,
            )
        )
        return
    width = max(len(c[0]) for c in checks)
    for name, ok, detail in checks:
        typer.echo(f"{'OK  ' if ok else 'MISS'} {name.ljust(width)}  {detail}")
    typer.echo(
        "ready"
        if all(ok for n, ok, _ in checks if n in ("catalog", "index"))
        else "not ready: build the catalog and index (mcv bootstrap)"
    )


def _doctor_checks(s: Settings) -> list[tuple[str, bool, str]]:
    import importlib
    import shutil

    checks: list[tuple[str, bool, str]] = []

    def dep(name: str, extra: str) -> None:
        try:
            importlib.import_module(name)
            checks.append((f"{name}", True, "installed"))
        except ImportError:
            checks.append((f"{name}", False, f"pip install -e '.[{extra}]'"))
        except Exception as e:  # installed but broken (missing native library)
            checks.append((f"{name}", False, f"import failed: {str(e)[:80]}"))

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
    except Exception:  # no torch, or a torch that cannot load
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
    if meta.exists() and s.catalog_db.exists():
        from mcmaster_vision.catalog import CatalogStore

        with CatalogStore(s.catalog_db) as store:
            updated = store.get_meta("updated_at")
        built = json.loads(meta.read_text()).get("built_at")
        stale = bool(updated and built and updated > built)
        checks.append(
            (
                "index up to date",
                not stale,
                "catalog changed after the index was built: run mcv build-index --only-new"
                if stale
                else "index newer than catalog",
            )
        )
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
    from mcmaster_vision.pipeline.backup import storage_status

    sto = storage_status(s)
    last = sto["last_backup"]
    checks.append(
        (
            "backup",
            bool(last) and not sto["backup_stale"],
            (
                f"{last['created_at']} ({last['bytes'] / 1e6:.1f} MB)"
                + ("; state changed since -> mcv backup" if sto["backup_stale"] else "")
            )
            if last
            else f"none yet: mcv backup ({sto['bytes_total'] / 1e6:.1f} MB of state)",
        )
    )
    if s.rerank_llm_enabled:
        import os

        checks.append(
            (
                "ANTHROPIC_API_KEY",
                bool(os.environ.get("ANTHROPIC_API_KEY")),
                "needed for the vision reranker",
            )
        )
    return checks


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
    from mcmaster_vision.training import evaluate_retrieval
    from mcmaster_vision.training.train import load_train_config
    from mcmaster_vision.training.train import train as _train

    s = _settings(config)
    cfg = load_train_config(train_config)
    if epochs:
        cfg["epochs"] = epochs
    cfg["output_dir"] = str(s.model_dir / "retrain")
    # purchases count more than taps (weighted photos repeat); the held-out split sees
    # each photo once so the evaluation is honest
    fstore = FeedbackStore(s.queries_dir)
    extra, held_out = _split_feedback(fstore.labelled_images())
    weights = {
        str(Path(x.image_path).resolve()): x.weight for x in fstore.entries() if x.image_path
    }
    extra = {
        pn: [x for x in v for _ in range(max(1, weights.get(x, 2)))] for pn, v in extra.items()
    }
    n_train_photos = sum(len(set(v)) for v in extra.values())
    typer.echo(
        f"1/3 training on catalog + {n_train_photos} confirmed photos, purchase-weighted "
        f"({len(held_out)} held out) ..."
    )
    live = s
    with CatalogStore(s.catalog_db) as store:
        ckpt = _train(store, cfg, extra_images=extra)
        s = s.model_copy(
            update={
                "backbone_checkpoint": ckpt,
                "backbone": cfg["backbone"],
                "backbone_pretrained": cfg.get("backbone_pretrained", s.backbone_pretrained),
            }
        )
        embedder = PartEmbedder(load_backbone(s))
        # the running API serves with *its* backbone and checkpoint and picks up a rebuilt
        # index within seconds, then refuses one whose version differs. So unless the new
        # embedder is exactly what the deployment serves, everything (index, calibration,
        # manifest) goes to sibling directories: switching is an explicit env change and
        # restart, never a cron side effect
        switched = _retrain_switches(live, embedder.version)
        if switched:
            tag = cfg["backbone"]
            s = s.model_copy(
                update={
                    "index_dir": live.index_dir.with_name(f"{live.index_dir.name}-{tag}"),
                    "model_dir": live.model_dir.with_name(f"{live.model_dir.name}-{tag}"),
                }
            )
            s.ensure_dirs()
            typer.echo(
                f"the deployment serves {live.backbone!r} ({live.backbone_checkpoint or 'no checkpoint'}); "
                f"the new model is {embedder.version!r}: index -> {s.index_path}, calibration -> "
                f"{s.model_dir} (the live ones are untouched)"
            )
        typer.echo("2/3 rebuilding index (catalog renders + confirmed photos) ...")
        idx = build_index(
            store,
            embedder,
            "auto",
            out_path=s.index_path,
            image_size=s.image_size,
            gallery_augment=s.index_gallery_augment,
            extra_images=extra or None,
        )
        typer.echo("3/3 evaluating + calibrating ...")
        ident = Identifier(
            store,
            idx,
            embedder,
            top_k=s.index_top_k,
            qe_k=s.query_expansion_k,
            image_size=s.image_size,
        )
        if not held_out:
            typer.echo(
                "no confirmed photo could be held out (parts need 2+): evaluating and "
                "calibrating on synthetic photo-style queries instead"
            )
        rep = evaluate_retrieval(ident, store, query_items=held_out or None, max_queries=500)
        cal = Calibration.fit_temperature(rep.score_lists, rep.correct_idx).fit_thresholds(
            rep.score_lists, rep.correct_idx
        )
        cal.save(s.model_dir / "calibration.json")
    from mcmaster_vision.pipeline.learn import mark_retrained

    mark_retrained(
        s,
        checkpoint=str(ckpt),
        index=idx.stats().model_dump(mode="json"),
        index_with_feedback=bool(extra),
        retrain_eval=json.loads(rep.to_json()),
        retrain_eval_source="held-out confirmed photos" if held_out else "synthetic renders",
        evaluation=None,  # the held-out real photos are the number that matters now
    )
    typer.echo(
        f"checkpoint {ckpt}; set MCV_BACKBONE_CHECKPOINT={ckpt}. Recall@1 {rep.recall_at.get(1)} on {rep.queries} queries"
    )
    if switched:
        typer.echo(
            f"to serve it: MCV_BACKBONE={cfg['backbone']} MCV_BACKBONE_CHECKPOINT={ckpt} "
            f"MCV_INDEX_DIR={s.index_dir} MCV_MODEL_DIR={s.model_dir} mcv serve "
            "(then future retrains update it in place)"
        )
        if reload_url:
            typer.echo("skipping --reload-url: the running API uses a different backbone")
            reload_url = None
    if reload_url:
        import httpx

        r = httpx.post(
            reload_url.rstrip("/") + "/admin/reload",
            headers={"X-API-Token": s.api_token or ""},
            timeout=120,
        )
        typer.echo(f"reload -> {r.status_code}")


@app.command()
def backup(
    config: Path | None = _config_opt,
    out: Path | None = typer.Option(
        None, help="Archive path or directory (default data/backups/mcv-<timestamp>.tar.gz)"
    ),
    no_checkpoint: bool = typer.Option(False, help="Leave the model checkpoint out"),
) -> None:
    """Bundle catalog, index, calibration, confirmed photos, logs and manifest into one tar.gz."""
    from mcmaster_vision.pipeline.backup import create_backup, read_inventory

    s = _settings(config)
    path = create_backup(s, out, include_checkpoint=not no_checkpoint)
    inv = read_inventory(path)
    total = sum(c["bytes"] for c in inv["components"].values())
    for name, c in inv["components"].items():
        typer.echo(f"  {name:12s} {c['bytes'] / 1e6:8.1f} MB  {c['source']}")
    typer.echo(f"{path}  ({path.stat().st_size / 1e6:.1f} MB compressed, {total / 1e6:.1f} MB raw)")


@app.command()
def restore(
    archive: Path = typer.Argument(..., exists=True, help="A backup written by `mcv backup`"),
    config: Path | None = _config_opt,
    only: list[str] | None = typer.Option(
        None, help="Restore just these components (catalog, index, calibration, queries, ...)"
    ),
    list_only: bool = typer.Option(False, "--list", help="Show what the archive holds and exit"),
) -> None:
    """Put a backup back in place (the running API picks the index up automatically)."""
    from mcmaster_vision.pipeline.backup import read_inventory, restore_backup

    inv = read_inventory(archive)
    typer.echo(f"backup from {inv.get('created_at')} (version {inv.get('version')}):")
    for name, c in inv["components"].items():
        typer.echo(f"  {name:12s} {c['bytes'] / 1e6:8.1f} MB")
    if list_only:
        return
    s = _settings(config)
    res = restore_backup(s, archive, components=only or None)
    for name, dest in res["restored"].items():
        typer.echo(f"restored {name} -> {dest}")


def _split_feedback(
    labelled: dict[str, list[str]],
) -> tuple[dict[str, list[str]], list[tuple[str, str]]]:
    """Training photos and held-out (part, path) pairs: one photo from every part with at
    least two, plus every 5th beyond that, so evaluation never sees training data and a
    shop with a few confirmations per part still gets a real-photo number."""
    extra: dict[str, list[str]] = {}
    held: list[tuple[str, str]] = []
    for pn, paths in labelled.items():
        if len(paths) < 2:
            extra[pn] = list(paths)
            continue
        out = {paths[-1]} | {p for i, p in enumerate(paths[:-1]) if i % 5 == 4}
        held += [(pn, p) for p in paths if p in out]
        keep = [p for p in paths if p not in out]
        if keep:
            extra[pn] = keep
    return extra, held


def _retrain_switches(live: Settings, new_version: str) -> bool:
    """Would serving ``new_version`` need different settings than ``live`` has? True when
    the deployment's own embedder (backbone + checkpoint) has a different version string,
    which is exactly what ``load_identifier`` compares the index against."""
    from mcmaster_vision.models import PartEmbedder, load_backbone

    try:
        current = PartEmbedder(load_backbone(live)).version
    except Exception:  # the live backbone cannot even load: anything new is a switch
        return True
    return current != new_version


@app.command()
def learn(
    config: Path | None = _config_opt,
    index_only: bool = typer.Option(
        False, "--index-only", help="Never retrain, only fold confirmed photos into the index"
    ),
    retrain_after: int | None = typer.Option(
        None, help="New confirmations that trigger a full retrain (default from config)"
    ),
    force: bool = typer.Option(False, help="Rebuild the index even with nothing new"),
    train_config: Path = typer.Option(Path("configs/train_tinycnn.yaml"), help="Training recipe"),
    reload_url: str | None = typer.Option(None, help="POST /admin/reload after a retrain"),
) -> None:
    """Close the loop: confirmed and bought photos into the index now, a full retrain once
    enough new ones arrived. Safe to run from cron every hour."""
    from mcmaster_vision.pipeline.learn import learn_index, learning_state

    s = _settings(config)
    threshold = s.learn_retrain_after if retrain_after is None else retrain_after
    state = learning_state(s)
    typer.echo(
        f"{state['new_confirmations']} new confirmations ({state['new_purchases']} purchases) "
        f"since the last learn; {state['since_retrain']}/{threshold} towards a retrain"
    )
    if not index_only and state["since_retrain"] >= threshold and state["since_retrain"] > 0:
        typer.echo("enough new evidence: full retrain")
        retrain(config=config, train_config=train_config, epochs=None, reload_url=reload_url)
        return
    res = learn_index(s, force=force)
    if res["action"] == "none":
        typer.echo("nothing new to learn")
        return
    typer.echo(
        f"index rebuilt with {res['photos']} confirmed photos of {res['parts']} parts in "
        f"{res['seconds']}s; a running API picks it up within seconds"
    )


@app.command()
def simulate(
    config: Path | None = _config_opt,
    customers: int = typer.Option(40, help="Synthetic customers to run through the journey"),
    seed: int = typer.Option(0),
    learn: bool = typer.Option(
        False, "--learn", help="Then learn from the purchases and run the customers again"
    ),
    top_n: int = typer.Option(5),
    tta: str = typer.Option("fast"),
    as_json: bool = typer.Option(False, "--json", help="Print the full report as JSON"),
) -> None:
    """Self-run the demo: customers identify, add to the cart and check out in-process,
    then the analytics name what went wrong. With --learn, shows before/after."""
    from mcmaster_vision.pipeline.simulate import simulate as _simulate

    s = _settings(config)
    out = _simulate(
        s,
        customers=customers,
        seed=seed,
        learn=learn,
        top_n=top_n,
        tta=tta,
        echo=None if as_json else typer.echo,
    )
    if as_json:
        typer.echo(json.dumps(out, indent=2, default=str))
    elif not out["issues"]:
        typer.echo("  no issues found at this traffic level")


@app.command()
def selfcheck(
    data_dir: Path = typer.Option(
        None, help="Scratch directory for the check (default: a temporary one, removed after)"
    ),
    parts: int = typer.Option(60, help="Synthetic parts to build the check catalog from"),
) -> None:
    """One command that proves this machine can run the whole thing: environment, build,
    identify, measure with a coin, feedback, backup, restore. Prints PASS or FAIL per step."""
    import shutil
    import tempfile
    import time

    from PIL import Image, ImageDraw

    from mcmaster_vision.pipeline.backup import create_backup, read_inventory, restore_backup
    from mcmaster_vision.pipeline.feedback import FeedbackStore

    tmp = None
    if data_dir is None:
        tmp = tempfile.mkdtemp(prefix="mcv-selfcheck-")
        data_dir = Path(tmp)
    results: list[tuple[str, bool, str]] = []

    def step(name: str, fn):
        t = time.perf_counter()
        try:
            detail = fn() or ""
            results.append((name, True, f"{detail} ({time.perf_counter() - t:.1f}s)"))
        except Exception as e:  # report, do not abort: every step is informative
            results.append((name, False, f"{type(e).__name__}: {str(e)[:120]}"))
        typer.echo(f"{'PASS' if results[-1][1] else 'FAIL'}  {name:28s} {results[-1][2]}")

    state: dict = {}

    def _env():
        checks = _doctor_checks(_settings(None))
        missing = [n for n, ok, _ in checks if not ok and n in ("catalog", "index")]
        return f"{sum(ok for _, ok, _ in checks)}/{len(checks)} checks ok" + (
            "" if not missing else " (no catalog built yet: fine for a check)"
        )

    def _build():
        s = _demo_settings(
            data_dir, backbone=_best_offline_backbone(), checkpoint=None, train_epochs=0
        )
        state["s"] = s
        state["ident"] = _build_demo(s, parts=parts, images_per_part=2, gallery_augment=1)
        return f"{parts} parts, backbone {state['ident'].embedder.version}"

    def _identify():
        ident = state["ident"]
        part = next(ident.store.iter_parts(with_images_only=True))
        img = Image.open(part.image_paths[0]).convert("RGB")
        res = ident.identify(img, tta="fast")
        state["part"], state["img"], state["res"] = part, img, res
        ranked = [c.part_number for c in res.candidates]
        rank = ranked.index(part.part_number) + 1 if part.part_number in ranked else None
        if rank is None:
            raise RuntimeError("the catalog image of a part did not retrieve itself")
        return f"tier {res.tier.value}, truth ranked #{rank}, {res.timings_ms.get('total')} ms"

    def _measure():
        ident = state["ident"]
        render = state["img"].resize((256, 256))
        canvas = Image.new("RGB", (512, 256), (255, 255, 255))
        ImageDraw.Draw(canvas).ellipse((40, 48, 200, 208), fill=(184, 172, 120))
        canvas.paste(render, (256, 0))
        res = ident.identify(canvas, tta="fast", suggest_reference=True)
        if not res.coin_hint:
            raise RuntimeError("no coin hint on a photo with a coin")
        d = res.coin_hint["diameter_px"]
        ref = (
            res.coin_hint["cx"] - d / 2,
            res.coin_hint["cy"],
            res.coin_hint["cx"] + d / 2,
            res.coin_hint["cy"],
        )
        res2 = ident.identify(canvas, tta="fast", mm_per_px=24.26 / d, reference=ref)
        if not res2.measured:
            raise RuntimeError("no measurement with a scale")
        return f"coin {d:.0f} px, part {res2.measured['long_mm']} x {res2.measured['short_mm']} mm"

    def _feedback():
        s = state["s"]
        fs = FeedbackStore(s.queries_dir)
        fb = fs.record(b"x", state["res"].request_id, state["part"].part_number)
        n = fs.stats()["confirmed"]
        return f"{n} confirmed, stored at {Path(fb.image_path).parent.name}/"

    def _backup():
        s = state["s"]
        archive = create_backup(s, data_dir / "backups")
        inv = read_inventory(archive)
        state["archive"] = archive
        return f"{len(inv['components'])} components, {archive.stat().st_size / 1e6:.1f} MB"

    def _restore():
        s = state["s"]
        other = data_dir / "restored"
        s2 = s.model_copy(
            update={
                "data_dir": other,
                "catalog_db": other / "catalog.sqlite",
                "index_dir": other / "index",
                "queries_dir": other / "queries",
                "model_dir": other / "models",
            }
        )
        res = restore_backup(s2, state["archive"])
        from mcmaster_vision.pipeline import load_identifier

        ident2 = load_identifier(s2)
        return f"{len(res['restored'])} components back, {len(ident2.index.ids)} index rows"

    for name, fn in (
        ("environment", _env),
        ("build catalog + index", _build),
        ("identify", _identify),
        ("coin hint + measure", _measure),
        ("feedback", _feedback),
        ("backup", _backup),
        ("restore + load", _restore),
    ):
        step(name, fn)
    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)
    failed = [n for n, ok, _ in results if not ok]
    typer.echo("ALL PASS" if not failed else f"FAILED: {', '.join(failed)}")
    if failed:
        raise typer.Exit(code=1)


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
    from mcmaster_vision.api.pages import layout

    out.write_text(
        layout(
            "Unlabelled photos",
            '<h1 class="page">Unlabelled photos</h1><div class="card" style="padding:8px 12px"><table class="spec">'
            + "".join(rows)
            + "</table></div>",
            head="<style>img{border:1px solid var(--rule);border-radius:6px} ol{padding-left:18px}</style>",
        ),
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
                # re-encoded copies are JPEG; verbatim copies keep their real format
                ext = ".jpg" if image_size else (Path(src).suffix.lower() or ".jpg")
                dst = folder / f"{part.part_number}_{kind}_{i}{ext}"
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
        # merge: a part that already has catalog images keeps them (and its name) when the
        # page adds specs or fails to yield images
        stats = _ingest(WebSource(importer, todo), store, merge=True)
    typer.echo(json.dumps(stats))
    typer.echo("Now run: mcv build-index")


@app.command("import-pages")
def import_pages(
    files: list[Path] = typer.Argument(..., exists=True, help="OCR text of catalog pages"),
    config: Path | None = _config_opt,
    first_page: int = typer.Option(1, help="Catalog page number of the first page in the text"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Parse and summarise (sections, materials, sizes) without writing"
    ),
) -> None:
    """Add the parts listed on printed catalog pages (an OCR text dump) to the store: part
    numbers, pipe sizes, materials, fitting types and prices. Images come later
    (`mcv fetch-images` / `mcv import-web`). Use --dry-run first to check the parse."""
    from collections import Counter

    from mcmaster_vision.catalog import CatalogStore
    from mcmaster_vision.catalog import ingest as _ingest
    from mcmaster_vision.catalog.pages import read_pages
    from mcmaster_vision.catalog.sources import CatalogSource

    class _Pages(CatalogSource):
        def __iter__(self):
            yield from read_pages(files, first_page=first_page)

        def __len__(self) -> int:
            return sum(1 for _ in self)

    if dry_run:
        parts = list(_Pages())
        sections = Counter(" > ".join(p.category_path) for p in parts)
        materials = Counter(p.attributes.get("material", "?") for p in parts)
        sizes = Counter(p.attributes.get("pipe_size", "?") for p in parts)
        families = len({p.family_id for p in parts if p.family_id})
        typer.echo(f"{len(parts)} parts, {families} families, {len(sections)} sections")
        for name, n in sections.most_common(15):
            typer.echo(f"  {n:5d}  {name}")
        typer.echo("materials: " + ", ".join(f"{m} ({n})" for m, n in materials.most_common(8)))
        typer.echo("sizes: " + ", ".join(f"{sz} ({n})" for sz, n in sizes.most_common(12)))
        for p in parts[:5]:
            typer.echo(f"  {p.part_number}: {p.name} {p.attributes}")
        unsized = [p.part_number for p in parts if p.attributes.get("pipe_size", "?") == "?"]
        if unsized:
            typer.echo(f"{len(unsized)} parts without a size, e.g. {unsized[:5]}")
        return

    s = _settings(config)
    with CatalogStore(s.catalog_db) as store:
        stats = _ingest(_Pages(), store, merge=True)
    typer.echo(f"{stats['parts']} parts from {len(files)} file(s) ({stats})")
    typer.echo("Next: add images (mcv fetch-images / mcv import-web), then mcv build-index")


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
    with_feedback: bool = typer.Option(
        False,
        "--with-feedback",
        help="Also embed confirmed photos from the feedback store as gallery images",
    ),
) -> None:
    """Embed every catalog image and write the vector index."""
    import time

    from mcmaster_vision.catalog import CatalogStore
    from mcmaster_vision.index import build_index
    from mcmaster_vision.models import PartEmbedder, load_backbone
    from mcmaster_vision.pipeline.feedback import FeedbackStore
    from mcmaster_vision.pipeline.manifest import update_manifest

    s = _settings(config, index_backend=None if backend in (None, "auto") else backend)
    ga = s.index_gallery_augment if gallery_augment is None else gallery_augment
    extra = FeedbackStore(s.queries_dir).labelled_images() if with_feedback else None
    if extra:
        typer.echo(
            f"including {sum(len(v) for v in extra.values())} confirmed photos of {len(extra)} parts"
        )
        if only_new:
            typer.echo(
                "note: --only-new skips parts already indexed; photos for those are not added"
            )
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
            extra_images=extra,
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
    mm_per_px: float | None = typer.Option(
        None, help="Scale of the photo (mm per pixel) to match candidates by size"
    ),
    ref: str | None = typer.Option(
        None, help="x1,y1,x2,y2 of a line across the reference object (coin, card), in pixels"
    ),
) -> None:
    """Identify the part in a photo."""
    from mcmaster_vision.pipeline import load_identifier
    from mcmaster_vision.pipeline.preprocess import decode_image

    s = _settings(config, rerank_llm_enabled=llm or None)
    reference = tuple(float(v) for v in ref.split(",")) if ref else None
    if reference is not None and len(reference) != 4:
        raise typer.BadParameter("--ref needs four numbers: x1,y1,x2,y2")
    result = load_identifier(s).identify(
        decode_image(image.read_bytes()),
        top_n=top_n,
        use_llm=llm or None,
        mm_per_px=mm_per_px,
        reference=reference,  # type: ignore[arg-type]
    )
    typer.echo(result.model_dump_json(indent=2))


@app.command("identify-dir")
def identify_dir(
    folder: Path = typer.Argument(
        ..., exists=True, file_okay=False, help="Folder of photos, one part per photo"
    ),
    out: Path = typer.Option(Path("identify_results.csv"), help="CSV of results"),
    config: Path | None = _config_opt,
    top_n: int = typer.Option(3),
    mm_per_px: float | None = typer.Option(
        None, help="Scale of every photo (mm per pixel): sizes and thread pitch are matched"
    ),
    coin: str | None = typer.Option(
        None,
        help="A coin photographed next to each part ('US quarter', '1 euro', ...): it is found "
        "in the photo and sets the scale; see pipeline/reference.py for the names",
    ),
) -> None:
    """Identify every photo in a folder (a bin, a drawer, a BOM shoot) and write a CSV."""
    import csv

    from mcmaster_vision.pipeline import load_identifier
    from mcmaster_vision.pipeline.preprocess import decode_image
    from mcmaster_vision.pipeline.reference import COINS_MM, find_coin

    coin_mm = None
    if coin:
        key = next((k for k in COINS_MM if k.lower() == coin.strip().lower()), None)
        if key is None:
            raise typer.BadParameter(f"unknown coin {coin!r}; choose from {', '.join(COINS_MM)}")
        coin_mm = COINS_MM[key]
    ident = load_identifier(_settings(config))
    exts = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".bmp"}
    files = sorted(f for f in folder.iterdir() if f.suffix.lower() in exts)
    cols = ["file", "tier", "best", "confidence", "family", "candidates", "error"]
    if mm_per_px or coin_mm:
        cols += ["long_mm", "short_mm", "pitch_mm", "scale_note"]
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for f in files:
            try:
                img = decode_image(f.read_bytes())
                scale, ref, note = mm_per_px, None, ""
                if coin_mm:
                    found = find_coin(img)
                    if found is not None:
                        k = float(img.info.get("upload_scale", 1.0) or 1.0)
                        scale = found.mm_per_px(coin_mm) / k
                        ref = tuple(v * k for v in found.segment())
                        note = f"coin {found.diameter_px * k:.0f} px"
                    elif scale is None:
                        note = "no coin found"
                res = ident.identify(img, top_n=top_n, mm_per_px=scale, reference=ref)
            except (OSError, ValueError) as e:  # unreadable, or refused as too large
                w.writerow([f.name, "", "", "", "", "", str(e)] + [""] * (len(cols) - 7))
                continue
            row = [
                f.name,
                res.tier.value,
                res.best.part_number if res.best else "",
                res.best.confidence if res.best else "",
                res.family.family_id if res.family else "",
                " ".join(c.part_number for c in res.candidates),
                "",
            ]
            if len(cols) > 7:
                m = res.measured or {}
                row += [m.get("long_mm", ""), m.get("short_mm", ""), m.get("pitch_mm", ""), note]
            w.writerow(row)
            typer.echo(f"{f.name}: {res.tier.value} {res.best.part_number if res.best else '-'}")
    typer.echo(f"{len(files)} photos -> {out}")


@app.command()
def serve(
    config: Path | None = _config_opt,
    host: str | None = typer.Option(None),
    port: int | None = typer.Option(None),
    workers: int = typer.Option(1, help="Uvicorn worker processes (each loads the index)"),
    https: bool = typer.Option(
        False,
        help="Serve HTTPS with a self-signed certificate (PWA install + live camera on phones)",
    ),
    qr: bool = typer.Option(False, help="Print a QR code of the LAN URL to scan with a phone"),
) -> None:
    """Run the HTTP API + phone UI. Use --host 0.0.0.0 --qr to open it on a phone on the same network."""
    from mcmaster_vision.api.app import run

    run(_settings(config), host=host, port=port, workers=workers, https=https, qr=qr)


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
    ident.feedback = None  # the usage prior would reward exactly the parts being evaluated
    if query_dir and int(ident.index.meta.get("extra_images", 0)):
        typer.echo(
            "warning: the index was built --with-feedback; photos from that store are gallery "
            "entries, so evaluating on them measures memorisation, not recall",
            err=True,
        )
    with CatalogStore(s.catalog_db) as store:
        report = evaluate_retrieval(ident, store, query_dir=query_dir, max_queries=max_queries)
    typer.echo(report.to_json())
    if report.by_category:
        typer.echo("\nweakest categories first (Recall@1 / Recall@5 / queries):")
        for cat, st in list(report.by_category.items())[:12]:
            typer.echo(f"  {st['recall_1']:.2f}  {st['recall_5']:.2f}  {st['queries']:4d}  {cat}")
    if report.hardest:
        typer.echo("hardest queries (truth -> predicted, rank):")
        for m in report.hardest[:8]:
            typer.echo(f"  {m['truth']} -> {m['predicted']} (rank {m['rank'] or 'not retrieved'})")
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
def up(
    config: Path | None = _config_opt,
    port: int = typer.Option(8000),
    https: bool = typer.Option(
        False, help="Self-signed HTTPS (installable app + live camera on phones)"
    ),
    parts: int = typer.Option(300, help="Synthetic parts to generate when nothing is built yet"),
    demo_dir: Path = typer.Option(Path("./data/demo")),
    backbone: str = typer.Option(
        "auto", help="auto | hash | tinycnn | ensemble (demo catalog only)"
    ),
) -> None:
    """Serve on the network with a QR code. Uses your built catalog if there is one, otherwise
    builds (and reuses) a synthetic demo catalog with the shipped model. The seamless demo entry point."""
    from mcmaster_vision.api.app import run

    s = _settings(config)
    if (s.index_path / "meta.json").exists() and s.catalog_db.exists():
        typer.echo(f"serving the built catalog at {s.catalog_db}")
        s = s.model_copy(update={"demo_mode": True})
    else:
        typer.echo(
            "no catalog built yet -> using a synthetic demo catalog (built once, then reused)"
        )
        chosen = _best_offline_backbone() if backbone == "auto" else backbone
        s = _demo_settings(demo_dir, backbone=chosen, checkpoint=None, train_epochs=0)
        s = s.model_copy(update={"demo_mode": True})
        meta = s.index_path / "meta.json"
        if meta.exists():
            # an earlier `mcv demo --backbone hash` built this index: serve it with the
            # backbone it was built with (a mismatch would fail on every photo)
            built_with = str(json.loads(meta.read_text(encoding="utf-8")).get("backbone", ""))
            built_name = re.split(r"[@:]", built_with)[0]  # "tinycnn:w24:d128@ckpt" -> "tinycnn"
            if built_name and built_name != chosen and backbone == "auto":
                typer.echo(f"reusing the demo index built with {built_name}")
                s = _demo_settings(demo_dir, backbone=built_name, checkpoint=None, train_epochs=0)
                s = s.model_copy(update={"demo_mode": True})
            # the *full* version (checkpoint included) must match what will be served
            from mcmaster_vision.models import PartEmbedder, load_backbone

            try:
                serving = PartEmbedder(load_backbone(s)).version
            except Exception:
                serving = ""
            if built_with != serving:
                typer.echo(
                    f"demo index was built with {built_with!r}; rebuilding for {serving or chosen!r}"
                )
                _build_demo(s, parts=parts, images_per_part=3, gallery_augment=2)
        else:
            _build_demo(s, parts=parts, images_per_part=3, gallery_augment=2)
    run(s, host="0.0.0.0", port=port, https=https, qr=True)


def _best_offline_backbone() -> str:
    """TinyCNN when torch and the shipped checkpoint are available (fast to index and query),
    otherwise the dependency-free hash descriptor."""
    try:
        import torch  # noqa: F401

        if (Path(__file__).resolve().parents[2] / "assets" / "tinycnn_synthetic.pt").exists():
            return "tinycnn"
    except ImportError:
        pass
    return "hash"


def _demo_settings(
    data_dir: Path, backbone: str, checkpoint: Path | None, train_epochs: int
) -> Settings:
    if checkpoint is None and train_epochs == 0 and backbone in ("tinycnn", "ensemble"):
        shipped = Path(__file__).resolve().parents[2] / "assets" / "tinycnn_synthetic.pt"
        if shipped.exists():
            checkpoint = shipped
    s = Settings(
        data_dir=data_dir,
        catalog_db=data_dir / "catalog.sqlite",
        index_dir=data_dir / "index",
        model_dir=data_dir / "models",
        queries_dir=data_dir / "queries",
        backbone=backbone,  # type: ignore[arg-type]
        backbone_pretrained="none" if backbone in ("tinycnn", "ensemble") else None,
        backbone_checkpoint=checkpoint,
        index_backend="numpy",
    )
    if backbone == "openclip":
        s = s.model_copy(update={"backbone_pretrained": Settings().backbone_pretrained})
    s.ensure_dirs()
    return s


def _build_demo(
    s: Settings, *, parts: int, images_per_part: int, gallery_augment: int, seed: int = 0
):
    """Generate + ingest + index + calibrate a synthetic catalog into ``s`` (returns the Identifier)."""
    from mcmaster_vision.catalog import CatalogStore
    from mcmaster_vision.catalog import ingest as _ingest
    from mcmaster_vision.data import SyntheticCatalog
    from mcmaster_vision.index import build_index
    from mcmaster_vision.models import PartEmbedder, load_backbone
    from mcmaster_vision.pipeline import Identifier
    from mcmaster_vision.pipeline.calibration import Calibration
    from mcmaster_vision.training import evaluate_retrieval

    data_dir = s.data_dir
    jsonl = data_dir / "parts.jsonl"
    if not jsonl.exists():
        typer.echo(f"1/4 generating {parts} synthetic parts ...")
        SyntheticCatalog(parts, images_per_part, seed=seed).write_jsonl(data_dir / "images", jsonl)
    typer.echo("2/4 ingesting ...")
    store = CatalogStore(s.catalog_db)
    _ingest(jsonl, store)
    typer.echo(f"3/4 embedding + indexing with {s.backbone} ...")
    embedder = PartEmbedder(load_backbone(s))
    index = build_index(
        store,
        embedder,
        "numpy",
        out_path=s.index_path,
        gallery_augment=gallery_augment,
        image_size=s.image_size,
    )
    ident = Identifier(store, index, embedder, qe_k=s.query_expansion_k, image_size=s.image_size)
    typer.echo("4/4 evaluating + calibrating ...")
    report = evaluate_retrieval(ident, store, max_queries=min(parts, 120))
    cal = Calibration.fit_temperature(report.score_lists, report.correct_idx).fit_thresholds(
        report.score_lists, report.correct_idx
    )
    cal.save(s.model_dir / "calibration.json")
    ident.calibration = cal
    typer.echo(
        f"   Recall@1 {report.recall_at.get(1)}  Recall@5 {report.recall_at.get(5)}  Recall@10 {report.recall_at.get(10)}"
    )
    return ident


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
    phone: bool = typer.Option(
        False, help="Serve on the network with a QR code (and demo mode) so a phone can open it"
    ),
    https: bool = typer.Option(
        False, help="Self-signed HTTPS (with --phone): installable app + live camera"
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

    if backbone == "hash" and checkpoint is None:
        try:
            import torch  # noqa: F401

            if (Path(__file__).resolve().parents[2] / "assets" / "tinycnn_synthetic.pt").exists():
                typer.echo(
                    "hint: torch is installed; --backbone ensemble uses the shipped learned model (much higher recall)"
                )
        except ImportError:
            pass
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
        from mcmaster_vision.api.app import run

        s = s.model_copy(update={"demo_mode": True})
        if phone:
            run(s, host="0.0.0.0", port=port, https=https, qr=True)
        else:
            typer.echo(
                f"\nserving UI at http://127.0.0.1:{port}  (add --phone to open it on a phone)"
            )
            run(s, host="127.0.0.1", port=port)


if __name__ == "__main__":
    app()
