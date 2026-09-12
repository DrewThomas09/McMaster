from __future__ import annotations

from typer.testing import CliRunner

from mcmaster_vision.cli import app


def test_demo_end_to_end(tmp_path):
    runner = CliRunner()
    r = runner.invoke(
        app,
        [
            "demo",
            "--parts",
            "25",
            "--images-per-part",
            "2",
            "--data-dir",
            str(tmp_path),
            "--no-serve",
        ],
    )
    assert r.exit_code == 0, r.output
    assert "recall_at" in r.output
    assert (tmp_path / "index" / "parts" / "meta.json").exists()
    assert (tmp_path / "sample_query.jpg").exists()

    q = tmp_path / "sample_query.jpg"
    r = runner.invoke(
        app,
        ["identify", str(q), "--top-n", "3"],
        env={
            "MCV_CATALOG_DB": str(tmp_path / "catalog.sqlite"),
            "MCV_INDEX_DIR": str(tmp_path / "index"),
            "MCV_MODEL_DIR": str(tmp_path / "models"),
        },
    )
    assert r.exit_code == 0, r.output
    assert '"candidates"' in r.output


def test_identify_dir_writes_csv(tmp_path, store, index, embedder, demo_dir):
    import csv

    from PIL import Image

    index.save(tmp_path / "index" / "parts")
    photos = tmp_path / "photos"
    photos.mkdir()
    parts = list(store.iter_parts(with_images_only=True))[:3]
    for p in parts:
        Image.open(p.image_paths[0]).convert("RGB").save(photos / f"{p.part_number}.jpg")
    (photos / "junk.jpg").write_bytes(b"not an image")
    env = {
        "MCV_CATALOG_DB": str(demo_dir / "catalog.sqlite"),
        "MCV_INDEX_DIR": str(tmp_path / "index"),
        "MCV_MODEL_DIR": str(tmp_path / "m"),
        "MCV_DATA_DIR": str(tmp_path),
        "MCV_QUERIES_DIR": str(tmp_path / "q"),
    }
    r = CliRunner().invoke(
        app, ["identify-dir", str(photos), "--out", str(tmp_path / "res.csv")], env=env
    )
    assert r.exit_code == 0, r.output
    rows = list(csv.DictReader(open(tmp_path / "res.csv")))
    assert len(rows) == 4
    assert all(row["best"] == row["file"].split(".")[0] for row in rows if not row["error"])
    assert any(row["error"] for row in rows)


def test_export_dataset(tmp_path, demo_dir):
    import csv

    env = {
        "MCV_CATALOG_DB": str(demo_dir / "catalog.sqlite"),
        "MCV_DATA_DIR": str(tmp_path),
        "MCV_INDEX_DIR": str(tmp_path / "i"),
        "MCV_MODEL_DIR": str(tmp_path / "m"),
        "MCV_QUERIES_DIR": str(tmp_path / "q"),
    }
    r = CliRunner().invoke(
        app,
        ["export-dataset", str(tmp_path / "ds"), "--query-set", "1", "--image-size", "128"],
        env=env,
    )
    assert r.exit_code == 0, r.output
    rows = list(csv.DictReader(open(tmp_path / "ds" / "labels.csv")))
    assert len(rows) == 80 and all((tmp_path / "ds" / row["path"]).exists() for row in rows[:5])
    assert len(list((tmp_path / "ds" / "queries").iterdir())) == 40
    assert sum(1 for _ in open(tmp_path / "ds" / "parts.jsonl")) == 40


def test_up_builds_demo_once_then_reuses(tmp_path, monkeypatch):
    """`mcv up` with nothing built: builds a synthetic demo catalog, then serves; second run reuses it."""
    import mcmaster_vision.api.app as app_mod

    calls = []
    monkeypatch.setattr(app_mod, "run", lambda s, **kw: calls.append((s, kw)))
    env = {
        "MCV_DATA_DIR": str(tmp_path / "data"),
        "MCV_CATALOG_DB": str(tmp_path / "data" / "c.sqlite"),
        "MCV_INDEX_DIR": str(tmp_path / "data" / "i"),
        "MCV_MODEL_DIR": str(tmp_path / "data" / "m"),
        "MCV_QUERIES_DIR": str(tmp_path / "data" / "q"),
        "MCV_BACKBONE": "hash",
    }
    r = CliRunner().invoke(
        app, ["up", "--parts", "20", "--demo-dir", str(tmp_path / "demo")], env=env
    )
    assert r.exit_code == 0, r.output
    assert (
        "synthetic demo catalog" in r.output
        and (tmp_path / "demo" / "index" / "parts" / "meta.json").exists()
    )
    s, kw = calls[-1]
    assert s.demo_mode is True and kw["host"] == "0.0.0.0" and kw["qr"] is True
    r = CliRunner().invoke(
        app, ["up", "--parts", "20", "--demo-dir", str(tmp_path / "demo")], env=env
    )
    assert r.exit_code == 0 and "generating" not in r.output and len(calls) == 2


def test_identify_dir_with_a_coin_sets_the_scale(tmp_path, demo_dir, store):
    """--coin finds the coin in each photo, measures the part and writes sizes to the CSV."""
    import csv

    from PIL import Image, ImageDraw

    photos = tmp_path / "shots"
    photos.mkdir()
    part = next(store.iter_parts(with_images_only=True))
    render = Image.open(part.image_paths[0]).convert("RGB").resize((256, 256))
    canvas = Image.new("RGB", (512, 256), (255, 255, 255))
    ImageDraw.Draw(canvas).ellipse((40, 48, 200, 208), fill=(184, 172, 120))
    canvas.paste(render, (256, 0))
    canvas.save(photos / "with_coin.jpg", quality=92)
    render.save(photos / "alone.jpg", quality=92)
    import shutil

    shutil.copytree(demo_dir / "index", tmp_path / "idx" / "parts")  # index_dir/parts layout
    env = {
        "MCV_CATALOG_DB": str(demo_dir / "catalog.sqlite"),
        "MCV_INDEX_DIR": str(tmp_path / "idx"),
        "MCV_DATA_DIR": str(tmp_path),
        "MCV_BACKBONE": "hash",
    }
    out = tmp_path / "res.csv"
    r = CliRunner().invoke(
        app, ["identify-dir", str(photos), "--out", str(out), "--coin", "US quarter"], env=env
    )
    assert r.exit_code == 0, r.output
    rows = {row["file"]: row for row in csv.DictReader(out.open())}
    assert (
        rows["with_coin.jpg"]["scale_note"].startswith("coin")
        and float(rows["with_coin.jpg"]["long_mm"]) > 0
    )
    assert rows["alone.jpg"]["scale_note"] == "no coin found" and rows["alone.jpg"]["long_mm"] == ""
    bad = CliRunner().invoke(app, ["identify-dir", str(photos), "--coin", "doubloon"], env=env)
    assert bad.exit_code != 0 and "unknown coin" in bad.output


def test_report_lists_customers_and_the_strip(identifier, store, tmp_path):
    """`mcv report` reads the orders and events a deployment wrote and prints the customer
    model and the For-you strip's take rate next to the funnel."""
    from fastapi.testclient import TestClient

    from mcmaster_vision.api import create_app
    from mcmaster_vision.cli import app as cli
    from mcmaster_vision.config import Settings

    s = Settings(
        data_dir=tmp_path, catalog_db=store.path, queries_dir=tmp_path / "q", demo_mode=True
    )
    parts = list(store.iter_parts(with_images_only=True))[:3]
    with TestClient(create_app(s, identifier=identifier)) as client:
        for i in range(3):
            client.get("/recommend?client_id=shop-x&n=4")
            client.post(
                "/cart", json={"client_id": "shop-x", "part_number": parts[i % 3].part_number}
            )
            assert client.post("/checkout", json={"client_id": "shop-x"}).status_code == 200
    cfg = tmp_path / "mcv.yaml"
    cfg.write_text(
        f"data_dir: {tmp_path}\ncatalog_db: {s.catalog_db}\nqueries_dir: {tmp_path / 'q'}\n"
    )
    r = CliRunner().invoke(cli, ["report", "--config", str(cfg)])
    assert r.exit_code == 0, r.output
    assert "For-you strip" in r.output and "customers 1" in r.output
    assert "parts added by" in r.output and "searches" in r.output
