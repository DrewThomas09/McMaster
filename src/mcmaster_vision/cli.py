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


@app.command("build-index")
def build_index_cmd(
    config: Path | None = _config_opt,
    backend: str | None = typer.Option(None, help="numpy | faiss"),
    batch_size: int = typer.Option(256),
) -> None:
    """Embed every catalog image and write the vector index."""
    from mcmaster_vision.catalog import CatalogStore
    from mcmaster_vision.index import build_index
    from mcmaster_vision.models import PartEmbedder, load_backbone

    s = _settings(config, index_backend=backend)
    embedder = PartEmbedder(load_backbone(s))
    with CatalogStore(s.catalog_db) as store:
        idx = build_index(
            store,
            embedder,
            s.index_backend,
            batch_size=batch_size,
            out_path=s.index_path,
            image_size=s.image_size,
            gallery_augment=s.index_gallery_augment,
            progress=lambda d, t: typer.echo(f"  {d}/{t} parts embedded"),
        )
    typer.echo(idx.stats().model_dump_json(indent=2))


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
) -> None:
    """Fine-tune a backbone on the catalog (requires the [ml] extra)."""
    from mcmaster_vision.catalog import CatalogStore
    from mcmaster_vision.training.train import load_train_config
    from mcmaster_vision.training.train import train as _train

    s = _settings(runtime_config)
    cfg = load_train_config(config)
    with CatalogStore(s.catalog_db) as store:
        ckpt = _train(store, cfg)
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
    backbone: str = typer.Option("hash", help="hash | tinycnn | openclip | dinov2"),
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

    s = Settings(
        data_dir=data_dir,
        catalog_db=data_dir / "catalog.sqlite",
        index_dir=data_dir / "index",
        model_dir=data_dir / "models",
        backbone=backbone,  # type: ignore[arg-type]
        backbone_pretrained="none" if backbone == "tinycnn" else None,
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

        typer.echo(f"2b/4 training {backbone} for {train_epochs} epochs ...")
        cfg = load_train_config(
            Path("configs") / f"train_{backbone}.yaml"
            if (Path("configs") / f"train_{backbone}.yaml").exists()
            else None
        )
        cfg.update(
            {
                "backbone": backbone,
                "epochs": train_epochs,
                "output_dir": str(s.model_dir / backbone),
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
