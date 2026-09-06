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
