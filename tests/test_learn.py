"""Recursive learning: confirmed photos into the index (incrementally) and the retrain trigger."""

from __future__ import annotations

import io

from PIL import Image

from mcmaster_vision.config import Settings
from mcmaster_vision.index import load_index
from mcmaster_vision.pipeline.feedback import FeedbackStore
from mcmaster_vision.pipeline.learn import learn_index, learning_state, mark_retrained
from mcmaster_vision.pipeline.manifest import read_manifest


def _settings(tmp_path, demo_dir, index):
    s = Settings(
        data_dir=tmp_path,
        catalog_db=demo_dir / "catalog.sqlite",
        index_dir=tmp_path / "index",
        model_dir=tmp_path / "models",
        queries_dir=tmp_path / "q",
        backbone="hash",
        index_gallery_augment=0,
        learn_retrain_after=3,
    )
    s.ensure_dirs()
    index.save(s.index_path)
    return s


def _photo(part) -> bytes:
    buf = io.BytesIO()
    Image.open(part.image_paths[0]).convert("RGB").resize((300, 300)).save(buf, "JPEG")
    return buf.getvalue()


def test_learn_index_incremental_then_nothing_new(tmp_path, demo_dir, store, index):
    s = _settings(tmp_path, demo_dir, index)
    parts = list(store.iter_parts(with_images_only=True))[:2]
    fb = FeedbackStore(s.queries_dir)
    assert learn_index(s)["action"] == "none"
    fb.record(_photo(parts[0]), "r1", parts[0].part_number, source="checkout")
    fb.record(_photo(parts[1]), "r2", parts[1].part_number, source="tap")
    before = len(load_index(s.index_path))
    res = learn_index(s)
    assert res["action"] == "index" and res["how"] == "incremental" and res["added"] == 2
    idx = load_index(s.index_path)
    assert len(idx) == before + 2 and len(idx.meta["learned_paths"]) == 2
    assert idx.meta["extra_images"] == 2
    m = read_manifest(s)
    assert m["learned_at"] == res["learned_at"] and m["learned_how"] == "incremental"
    # the same photo is never added twice; a new one is
    assert learn_index(s)["action"] == "none"
    fb.record(_photo(parts[0]), "r3", parts[0].part_number, source="checkout")
    res = learn_index(s)
    assert res["added"] == 1 and len(load_index(s.index_path)) == before + 3
    # the learned photo is found again from itself
    from mcmaster_vision.models import HashBackbone, PartEmbedder
    from mcmaster_vision.pipeline import Identifier

    ident = Identifier(store, load_index(s.index_path), PartEmbedder(HashBackbone()), top_k=10)
    res_id = ident.identify(Image.open(io.BytesIO(_photo(parts[0]))), top_n=3, tta="none")
    assert res_id.candidates[0].part_number == parts[0].part_number
    # retrain trigger
    st = learning_state(s)
    assert st["since_retrain"] == 3 and st["retrain_due"] is True
    mark_retrained(s, checkpoint="x.pt")
    st = learning_state(s)
    assert st["since_retrain"] == 0 and st["retrain_due"] is False and st["last_retrain"]


def test_learn_index_rebuilds_when_index_is_foreign(tmp_path, demo_dir, store, index):
    s = _settings(tmp_path, demo_dir, index)
    parts = list(store.iter_parts(with_images_only=True))[:1]
    FeedbackStore(s.queries_dir).record(_photo(parts[0]), "r1", parts[0].part_number)
    # an older --with-feedback index that does not list its photos: rebuilt once
    idx = load_index(s.index_path)
    idx.meta["extra_images"] = 5
    idx.meta.pop("learned_paths", None)
    idx.save(s.index_path)
    res = learn_index(s)
    assert res["how"] == "rebuild" and res["photos"] == 1
    assert len(load_index(s.index_path).meta["learned_paths"]) == 1
    assert learn_index(s, force=True)["how"] == "incremental"


def test_simulate_reports_and_learns(tmp_path, demo_dir, store, index):
    from mcmaster_vision.pipeline.simulate import make_customers, simulate

    s = _settings(tmp_path, demo_dir, index)
    crowd = make_customers(["A", "B"], 5, seed=1)
    assert len({c.client_id for c in crowd}) == 5 and all(c.part_number in "AB" for c in crowd)
    out = simulate(s, customers=8, seed=3, learn=True, tta="none")
    b = out["before"]
    assert b["identified"] == 8 and b["carts"] >= b["checkouts"]
    assert b["found_in_list"] <= b["identified"] and not b["errors"], b["errors"]
    assert isinstance(out["issues"], list) and out["analytics"]["window"]["identify"] == 8
    if out["learn"]["action"] == "index":
        same = out["after_same_photos"]
        assert same["bought_top1_after"] >= same["bought_top1_before"]
        assert "after_new_photos" in out
    # everything a real deployment leaves behind is there
    assert (tmp_path / "logs" / "events.jsonl").exists()
    assert learning_state(s)["learned_at"] or out["learn"]["action"] == "none"


def test_cli_simulate_and_learn(tmp_path, demo_dir, index):
    from typer.testing import CliRunner

    from mcmaster_vision.cli import app

    s = _settings(tmp_path, demo_dir, index)
    env = {
        "MCV_DATA_DIR": str(tmp_path),
        "MCV_CATALOG_DB": str(s.catalog_db),
        "MCV_INDEX_DIR": str(s.index_dir),
        "MCV_MODEL_DIR": str(s.model_dir),
        "MCV_QUERIES_DIR": str(s.queries_dir),
        "MCV_BACKBONE": "hash",
        "MCV_INDEX_GALLERY_AUGMENT": "0",
    }
    r = CliRunner().invoke(app, ["simulate", "--customers", "6", "--tta", "none"], env=env)
    assert r.exit_code == 0, r.output
    assert "customers on" in r.output and "found in list" in r.output
    r = CliRunner().invoke(app, ["learn", "--index-only"], env=env)
    assert r.exit_code == 0, r.output
    assert "new confirmations" in r.output
    r = CliRunner().invoke(
        app, ["simulate", "--customers", "3", "--tta", "none", "--json"], env=env
    )
    assert r.exit_code == 0, r.output
    assert '"before"' in r.output
