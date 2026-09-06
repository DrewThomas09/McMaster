from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from typer.testing import CliRunner

from mcmaster_vision.api import create_app
from mcmaster_vision.api.ratelimit import RateLimiter
from mcmaster_vision.cli import app
from mcmaster_vision.config import Settings


def test_rate_limiter():
    rl = RateLimiter(2)
    assert rl.allow("a") and rl.allow("a") and not rl.allow("a") and rl.allow("b")
    assert RateLimiter(0).allow("x")


def test_identify_rate_limited(identifier, store, tmp_path):
    client = TestClient(
        create_app(
            Settings(data_dir=tmp_path, queries_dir=tmp_path / "q", rate_limit_per_minute=1),
            identifier=identifier,
        )
    )
    part = next(store.iter_parts(with_images_only=True))
    buf = io.BytesIO()
    Image.open(part.image_paths[0]).save(buf, format="PNG")
    assert (
        client.post("/identify", files={"file": ("a.png", buf.getvalue(), "image/png")}).status_code
        == 200
    )
    assert (
        client.post("/identify", files={"file": ("a.png", buf.getvalue(), "image/png")}).status_code
        == 429
    )


def test_doctor_and_review_unknowns(tmp_path, index, demo_dir):
    index.save(tmp_path / "index" / "parts")
    env = {
        "MCV_CATALOG_DB": str(demo_dir / "catalog.sqlite"),
        "MCV_INDEX_DIR": str(tmp_path / "index"),
        "MCV_MODEL_DIR": str(tmp_path / "m"),
        "MCV_DATA_DIR": str(tmp_path),
        "MCV_QUERIES_DIR": str(tmp_path / "q"),
        "MCV_BACKBONE": "hash",
    }
    r = CliRunner().invoke(app, ["doctor"], env=env)
    assert r.exit_code == 0, r.output
    assert "index/backbone match" in r.output and "ready" in r.output
    r = CliRunner().invoke(app, ["review-unknowns", "--out", str(tmp_path / "rev.html")], env=env)
    assert r.exit_code == 0 and "no unknown photos" in r.output
    unk = tmp_path / "q" / "_unknown"
    unk.mkdir(parents=True)
    from mcmaster_vision.catalog import CatalogStore

    with CatalogStore(demo_dir / "catalog.sqlite") as st:
        part = next(st.iter_parts(with_images_only=True))
    Image.open(part.image_paths[0]).convert("RGB").save(unk / "abc.jpg")
    r = CliRunner().invoke(app, ["review-unknowns", "--out", str(tmp_path / "rev.html")], env=env)
    assert r.exit_code == 0, r.output
    assert part.part_number in (tmp_path / "rev.html").read_text()


def test_retrain_end_to_end(tmp_path, demo_dir, index, store):
    """Cron path: train briefly on catalog + confirmed photos, rebuild, calibrate, write manifest."""
    pytest.importorskip("torch")
    import json

    from PIL import Image

    index.save(tmp_path / "index" / "parts")
    # two confirmed photos for one part -> one held out for evaluation
    part = next(store.iter_parts(with_images_only=True))
    qdir = tmp_path / "q" / part.part_number
    qdir.mkdir(parents=True)
    for i in range(5):
        Image.open(part.image_paths[0]).convert("RGB").save(qdir / f"r{i}.jpg")
    cfg = tmp_path / "train.yaml"
    cfg.write_text(
        "backbone: tinycnn\nbackbone_pretrained: none\nimage_size: 96\nembedding_dim: 32\nepochs: 1\nbatch_size: 8\nmax_parts: 12\ncache_views: 1\ncache_workers: 1\ntorch_threads: 2\nwarmup_steps: 1\nval_frac: 0.2\n"
    )
    env = {
        "MCV_CATALOG_DB": str(demo_dir / "catalog.sqlite"),
        "MCV_INDEX_DIR": str(tmp_path / "index"),
        "MCV_MODEL_DIR": str(tmp_path / "m"),
        "MCV_DATA_DIR": str(tmp_path),
        "MCV_QUERIES_DIR": str(tmp_path / "q"),
        "MCV_BACKBONE": "hash",
        "MCV_INDEX_GALLERY_AUGMENT": "0",
    }
    r = CliRunner().invoke(app, ["retrain", "--train-config", str(cfg)], env=env)
    assert r.exit_code == 0, r.output
    assert "1 held out" in r.output and "checkpoint" in r.output
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["checkpoint"].endswith("best.pt") and manifest["retrain_eval"]["queries"] == 1
    assert (tmp_path / "m" / "calibration.json").exists()
