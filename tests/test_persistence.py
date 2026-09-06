"""Durable state: recent photos on disk, request log reload, feedback dedupe, backup/restore,
feedback photos in the index, auto-reload, decode guards."""

from __future__ import annotations

import io
import json
import os
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from typer.testing import CliRunner

from mcmaster_vision.api.app import create_app
from mcmaster_vision.cli import app as cli
from mcmaster_vision.config import Settings
from mcmaster_vision.index import build_index
from mcmaster_vision.index.base import load_index
from mcmaster_vision.pipeline.backup import create_backup, read_inventory, restore_backup
from mcmaster_vision.pipeline.feedback import FeedbackStore, RecentPhotos
from mcmaster_vision.pipeline.preprocess import decode_image
from mcmaster_vision.pipeline.requestlog import RequestLog
from mcmaster_vision.schemas import IdentificationResult, MatchTier


def _jpeg(size=(64, 64), color=(200, 30, 30)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="JPEG")
    return buf.getvalue()


def test_recent_photos_roundtrip_and_prune(tmp_path):
    rp = RecentPhotos(tmp_path / "recent", keep=3, max_age_s=3600)
    for i in range(5):
        rp.put(f"req{i}", b"x" * (i + 1))
        os.utime(rp._path(f"req{i}"), (time.time() - 100 + i, time.time() - 100 + i))
    assert rp.get("req4") == b"xxxxx"
    assert rp.get("../etc/passwd") is None
    rp.prune()
    assert len(rp) == 3
    assert rp.get("req0") is None and rp.get("req4") is not None
    rp2 = RecentPhotos(tmp_path / "recent", keep=3, max_age_s=1)
    rp2.prune()
    assert len(rp2) == 0


def _result(tier=MatchTier.CANDIDATE) -> IdentificationResult:
    return IdentificationResult(
        request_id="abc123", tier=tier, best=None, candidates=[], timings_ms={"total": 12.0}
    )


def test_request_log_survives_restart(tmp_path):
    path = tmp_path / "logs" / "requests.jsonl"
    rl = RequestLog(path, keep=10)
    for _ in range(12):
        rl.log(_result())
    assert rl.total == 12
    path.open("a").write("not json\n")  # a torn line must not break boot
    rl2 = RequestLog(path, keep=10)
    assert rl2.total == 12 and rl2.summary()["requests_window"] == 10
    assert rl2.recent(3)[0]["request_id"] == "abc123"


def test_feedback_dedupes_reconfirmation(tmp_path):
    fs = FeedbackStore(tmp_path / "q")
    fs.record(b"a", "r1", "91251A537", predicted="91251A537")
    fs.record(b"a", "r1", "91251A540", predicted="91251A537")  # corrected tap
    fs.record(b"b", "r2", None)
    e = {x.request_id: x for x in fs.entries()}
    assert len(e) == 2 and e["r1"].part_number == "91251A540"
    st = fs.stats()
    assert st["total"] == 2 and st["confirmed"] == 1 and st["correct_top1"] == 0


def test_feedback_survives_api_restart(identifier, store, tmp_path):
    settings = Settings(data_dir=tmp_path, queries_dir=tmp_path / "queries", warm_up=False)
    pn = next(store.iter_parts()).part_number
    with TestClient(create_app(settings, identifier)) as c:
        rid = c.post("/identify", files={"file": ("a.jpg", _jpeg(), "image/jpeg")}).json()[
            "request_id"
        ]
    # new process / new app object: the photo is on disk, not in memory
    with TestClient(create_app(settings, identifier)) as c:
        r = c.post("/feedback", data={"request_id": rid, "part_number": pn})
        assert r.status_code == 200, r.text
        assert (tmp_path / "queries" / pn / f"{rid}.jpg").exists()
        h = c.get("/health").json()
        assert h["requests_total"] == 1 and h["uptime_s"] >= 0


def test_backup_restore_roundtrip(tmp_path, demo_dir, index):
    data = tmp_path / "data"
    settings = Settings(
        data_dir=data,
        catalog_db=demo_dir / "catalog.sqlite",
        index_dir=demo_dir / "index_root",
        queries_dir=data / "queries",
        model_dir=data / "models",
    )
    import shutil

    shutil.copytree(demo_dir / "index", settings.index_path)
    FeedbackStore(settings.queries_dir).record(_jpeg(), "r1", "ABC")
    (data / "manifest.json").write_text('{"x": 1}')
    archive = create_backup(settings, tmp_path / "b")
    inv = read_inventory(archive)
    assert {"catalog", "index", "queries", "manifest"} <= set(inv["components"])
    assert archive.parent == tmp_path / "b"

    # restore into a fresh location
    other = tmp_path / "restored"
    s2 = Settings(
        data_dir=other,
        catalog_db=other / "catalog.sqlite",
        index_dir=other / "index",
        queries_dir=other / "queries",
        model_dir=other / "models",
    )
    res = restore_backup(s2, archive)
    assert set(res["restored"]) == set(inv["components"])
    assert s2.catalog_db.exists() and (s2.index_path / "meta.json").exists()
    assert (s2.queries_dir / "ABC" / "r1.jpg").exists()
    assert json.loads((other / "manifest.json").read_text()) == {"x": 1}
    assert len(load_index(s2.index_path).ids) == len(index.ids)
    # restoring over an existing install replaces, never merges
    FeedbackStore(s2.queries_dir).record(_jpeg(), "r9", "ZZZ")
    restore_backup(s2, archive, components=["queries"])
    assert not (s2.queries_dir / "ZZZ").exists() and (s2.queries_dir / "ABC").exists()
    with pytest.raises(ValueError):
        restore_backup(s2, archive, components=["nope"])


def test_backup_cli_and_bad_archive(tmp_path, demo_dir):
    runner = CliRunner()
    env = {
        "MCV_DATA_DIR": str(tmp_path / "d"),
        "MCV_CATALOG_DB": str(demo_dir / "catalog.sqlite"),
        "MCV_INDEX_DIR": str(demo_dir),
        "MCV_QUERIES_DIR": str(tmp_path / "d" / "queries"),
        "MCV_MODEL_DIR": str(tmp_path / "d" / "models"),
    }
    r = runner.invoke(cli, ["backup", "--out", str(tmp_path / "out")], env=env)
    assert r.exit_code == 0, r.output
    archives = list((tmp_path / "out").glob("mcv-*.tar.gz"))
    assert len(archives) == 1
    r = runner.invoke(cli, ["restore", str(archives[0]), "--list"], env=env)
    assert r.exit_code == 0 and "catalog" in r.output
    bad = tmp_path / "bad.tar.gz"
    import tarfile

    with tarfile.open(bad, "w:gz") as t:
        t.add(str(demo_dir / "parts.jsonl"), arcname="parts.jsonl")
    with pytest.raises(ValueError):
        read_inventory(bad)


def test_index_includes_feedback_photos(store, embedder, tmp_path):
    parts = list(store.iter_parts(with_images_only=True))
    pn = parts[0].part_number
    photo = tmp_path / "real.jpg"
    Image.open(parts[0].image_paths[0]).convert("RGB").rotate(15, fillcolor="white").save(photo)
    base = build_index(store, embedder, "numpy")
    with_fb = build_index(
        store, embedder, "numpy", extra_images={pn: [str(photo)], "NOPE": [str(photo)]}
    )
    assert len(with_fb.ids) == len(base.ids) + 1
    assert with_fb.meta["extra_images"] == 1
    hits = with_fb.search_ids(embedder.embed_query(Image.open(photo), tta="none")[0], 1)
    assert hits[0][0] == pn and hits[0][1] > 0.99


def test_api_auto_reloads_rebuilt_index(store, embedder, tmp_path, monkeypatch):
    idx_dir = tmp_path / "index"
    build_index(store, embedder, "numpy", out_path=idx_dir / "parts")
    settings = Settings(
        data_dir=tmp_path, index_dir=idx_dir, catalog_db=store.path, backbone="hash"
    )
    with TestClient(create_app(settings)) as c:
        first = c.get("/stats").json()["vectors"]
        # rebuild with gallery augmentation -> more vectors, newer meta.json
        build_index(store, embedder, "numpy", out_path=idx_dir / "parts", gallery_augment=1)
        meta = idx_dir / "parts" / "meta.json"
        os.utime(meta, (time.time() + 5, time.time() + 5))
        c.app.state.last_index_check = 0.0
        assert c.get("/stats").json()["vectors"] > first


def test_decode_guards():
    big = Image.new("RGB", (5000, 3000), "white")
    buf = io.BytesIO()
    big.save(buf, format="JPEG", quality=30)
    img = decode_image(buf.getvalue())
    assert max(img.size) <= 2500  # JPEG draft decoded at reduced scale
    # a header claiming an absurd size is rejected before any pixels are decoded
    huge = io.BytesIO()
    Image.new("L", (1, 1)).save(huge, format="PNG")
    raw = bytearray(huge.getvalue())
    raw[16:24] = (60000).to_bytes(4, "big") + (60000).to_bytes(4, "big")
    with pytest.raises((ValueError, OSError)):
        decode_image(bytes(raw))
