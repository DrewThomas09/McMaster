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
    out = simulate(s, customers=8, seed=3, learn=True, tta="none", scratch=tmp_path / "sim")
    b = out["before"]
    assert b["identified"] == 8 and b["carts"] >= b["checkouts"]
    assert b["found_in_list"] <= b["identified"] and not b["errors"], b["errors"]
    assert isinstance(out["issues"], list) and out["analytics"]["window"]["identify"] == 8
    if out["learn"]["action"] == "index":
        same = out["after_same_photos"]
        assert same["bought_top1_after"] >= same["bought_top1_before"]
        assert "after_new_photos" in out
    # everything a real deployment leaves behind is in the scratch copy, not the live data
    assert (tmp_path / "sim" / "logs" / "events.jsonl").exists()
    assert not (tmp_path / "logs" / "events.jsonl").exists()
    assert learning_state(s)["learned_at"] is None and not FeedbackStore(s.queries_dir).entries()
    if out["learn"]["action"] == "index":
        assert out["served_rows_after_learn"] == len(
            load_index(tmp_path / "sim" / "index" / "parts")
        )
        assert out["served_rows_after_learn"] > len(index)


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


def test_exact_gallery_hit_beats_priors():
    from mcmaster_vision.pipeline.rerank import FusionReranker
    from mcmaster_vision.pipeline.retrieve import Hit
    from mcmaster_vision.schemas import Part

    parts = {
        "A": Part(part_number="A", name="a", category_path=["x", "y"]),
        "B": Part(part_number="B", name="b", category_path=["x", "z"]),
    }
    # B is the learned photo itself (similarity 1.0); A has a strong category prior,
    # more hits and a long purchase history
    hits = [Hit("A", 0.96, 4, 0.9), Hit("B", 1.0, 1, 0.0)]
    out = FusionReranker().rerank(hits, parts, popularity={"A": 50})
    assert out[0].part.part_number == "B"
    assert any("near-identical" in r for r in out[0].reasons)
    # below the ramp nothing changes
    out = FusionReranker().rerank([Hit("A", 0.96, 4, 0.9), Hit("B", 0.98, 1, 0.0)], parts)
    assert not any("near-identical" in r for s in out for r in s.reasons)


def test_learn_calibrates_from_confirmed_photos(tmp_path, demo_dir, store, index, monkeypatch):
    from mcmaster_vision.pipeline import learn as learn_mod
    from mcmaster_vision.pipeline.calibration import Calibration

    monkeypatch.setattr(learn_mod, "MIN_CALIBRATION_SAMPLES", 4)
    monkeypatch.setattr(learn_mod, "MIN_WRONG_SAMPLES", 1)
    s = _settings(tmp_path, demo_dir, index)
    parts = list(store.iter_parts(with_images_only=True))[:5]
    fb = FeedbackStore(s.queries_dir)
    for i, part in enumerate(parts[:3]):
        fb.record(_photo(part), f"r{i}", part.part_number, source="checkout")
    res = learn_index(s)
    cal = res["calibration"]
    assert cal["new_samples"] == 3 and cal["samples"] == 3 and cal["refit"] is False
    samples = (s.model_dir / "calibration_samples.jsonl").read_text().splitlines()
    assert len(samples) == 3 and not (s.model_dir / "calibration.json").exists()
    # a wrong label: the photo of one part filed under another (a mistaken tap)
    fb.record(_photo(parts[3]), "q0", parts[4].part_number, source="tap")
    fb.record(_photo(parts[4]), "q1", parts[4].part_number, source="tap")
    res = learn_index(s)
    cal = res["calibration"]
    assert cal["samples"] == 5 and cal["wrong"] >= 1
    if cal["refit"]:
        fitted = Calibration.load(s.model_dir / "calibration.json")
        assert fitted.likely_threshold <= fitted.exact_threshold
        assert read_manifest(s)["calibration_from_purchases"]["samples"] == 5
    # already-learned photos are never scored again
    assert learn_index(s, force=True)["calibration"] is None


def test_all_correct_samples_do_not_refit(tmp_path, demo_dir, store, index, monkeypatch):
    from mcmaster_vision.pipeline import learn as learn_mod

    monkeypatch.setattr(learn_mod, "MIN_CALIBRATION_SAMPLES", 2)
    s = _settings(tmp_path, demo_dir, index)
    fb = FeedbackStore(s.queries_dir)
    for i, part in enumerate(list(store.iter_parts(with_images_only=True))[:3]):
        fb.record(_photo(part), f"r{i}", part.part_number, source="checkout")
    cal = learn_index(s)["calibration"]
    assert cal["samples"] == 3 and cal["refit"] is False  # nothing wrong to learn from
    assert not (s.model_dir / "calibration.json").exists()


def test_learn_lock_and_rebuild_triggers(tmp_path, demo_dir, store, index):
    import pytest

    from mcmaster_vision.pipeline.learn import LearnBusy, _Lock

    s = _settings(tmp_path, demo_dir, index)
    parts = list(store.iter_parts(with_images_only=True))[:2]
    fb = FeedbackStore(s.queries_dir)
    fb.record(_photo(parts[0]), "r1", parts[0].part_number, source="checkout")
    with _Lock(s), pytest.raises(LearnBusy):
        learn_index(s)
    assert learn_index(s)["how"] == "incremental"
    from mcmaster_vision.pipeline.events import EventLog

    assert [r["how"] for r in EventLog(s.data_dir / "logs" / "events.jsonl").rows("learn")] == [
        "incremental"
    ]
    # a corrected label removed the photo under the old part: only a rebuild drops the row
    fb.record(_photo(parts[0]), "r1", parts[1].part_number, source="tap")
    assert not (s.queries_dir / parts[0].part_number / "r1.jpg").exists()
    res = learn_index(s)
    assert res["how"] == "rebuild"
    idx = load_index(s.index_path)
    wanted = str((s.queries_dir / parts[1].part_number / "r1.jpg").resolve())
    assert idx.meta["learned_paths"] == [wanted]
    # photos a retrain held out for its evaluation never join the gallery
    r2 = str((s.queries_dir / parts[1].part_number / "r2.jpg").resolve())
    mark_retrained(s, held_out_paths=[r2])
    fb.record(_photo(parts[1]), "r2", parts[1].part_number, source="checkout")
    res = learn_index(s)
    assert res["action"] == "index" and res["photos"] == 1 and res["added"] == 0
    assert load_index(s.index_path).meta["learned_paths"] == [wanted]


def test_thresholds_fall_back_to_best_achievable():
    from mcmaster_vision.pipeline.calibration import Calibration

    cal = Calibration(temperature=0.05, likely_threshold=0.6, exact_threshold=0.99)
    # 40 queries: confident-and-right ones score high; a band of 0.6-0.7 answers is a coin
    # toss, so no threshold reaches 90% precision with support 10 until p >= 0.8
    score_lists, correct = [], []
    for i in range(40):
        if i < 12:
            score_lists.append([1.0, 0.8]), correct.append(0)  # p0 ~ 0.98
        elif i < 15:
            score_lists.append([1.0, 0.8]), correct.append(1)
        else:
            score_lists.append([1.0, 0.97]), correct.append(0 if i % 2 else 1)  # p0 ~ 0.65
    fitted = cal.fit_thresholds(score_lists, correct, likely_precision=0.9)
    thr = fitted.likely_threshold
    assert thr > 0.6
    kept = [
        ok
        for sc, ci in zip(score_lists, correct, strict=True)
        for ok in [ci == 0]
        if cal.probabilities(sc)[0] >= thr
    ]
    assert len(kept) == 15 and abs(sum(kept) / len(kept) - 0.8) < 1e-9  # the coin-toss band is out
    # when the current threshold is already the best achievable, nothing changes
    same = Calibration(temperature=0.05, likely_threshold=thr, exact_threshold=0.99)
    assert same.fit_thresholds(score_lists, correct).likely_threshold == thr
    # a threshold is never lowered by the fallback, and exact never sits below likely
    high = Calibration(temperature=0.05, likely_threshold=0.995, exact_threshold=0.999)
    assert high.fit_thresholds(score_lists, correct).likely_threshold == 0.995
    low_exact = Calibration(temperature=0.05, likely_threshold=0.6, exact_threshold=0.5)
    f2 = low_exact.fit_thresholds(score_lists, correct)
    assert f2.exact_threshold >= f2.likely_threshold >= 0.6


def test_calibration_samples_follow_the_model(tmp_path, demo_dir, store, index, monkeypatch):
    import json

    from mcmaster_vision.pipeline import learn as learn_mod

    monkeypatch.setattr(learn_mod, "MIN_CALIBRATION_SAMPLES", 2)
    monkeypatch.setattr(learn_mod, "MIN_WRONG_SAMPLES", 0)
    s = _settings(tmp_path, demo_dir, index)
    path = s.model_dir / "calibration_samples.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    # samples an older model left behind never refit the current one
    path.write_text(
        "".join(
            json.dumps({"scores": [1.0, 0.5], "correct": 1, "model": "old"}) + "\n"
            for _ in range(50)
        )
    )
    part = next(store.iter_parts(with_images_only=True))
    FeedbackStore(s.queries_dir).record(_photo(part), "r1", part.part_number, source="checkout")
    cal = learn_index(s)["calibration"]
    assert cal["samples"] == 1 and cal["new_samples"] == 1
    rows = [json.loads(ln) for ln in path.read_text().splitlines()]
    assert rows[-1]["model"] == index.meta["backbone"]


def test_retrain_refuses_under_a_running_learn(tmp_path, demo_dir, index):
    from typer.testing import CliRunner

    from mcmaster_vision.cli import app
    from mcmaster_vision.pipeline.learn import _Lock

    s = _settings(tmp_path, demo_dir, index)
    env = {
        "MCV_DATA_DIR": str(tmp_path),
        "MCV_CATALOG_DB": str(s.catalog_db),
        "MCV_INDEX_DIR": str(s.index_dir),
        "MCV_MODEL_DIR": str(s.model_dir),
        "MCV_QUERIES_DIR": str(s.queries_dir),
    }
    with _Lock(s):
        r = CliRunner().invoke(app, ["retrain"], env=env)
    assert r.exit_code == 1 and "already running" in r.output


def test_event_log_compaction_is_locked_and_keeps_late_rows(tmp_path):
    from mcmaster_vision.pipeline.events import EventLog

    path = tmp_path / "e.jsonl"
    log = EventLog(path, keep=5)
    for i in range(20):
        log.log("identify", request_id=f"x{i}")
    # rows appended after the boot read (another worker) survive the compaction
    with open(path, "a") as fh:
        fh.write('{"kind": "identify", "request_id": "late"}\n')
    again = EventLog(path, keep=5)
    assert again.identify_row("late") is not None
    assert len(path.read_text().splitlines()) == 5 and not list(tmp_path.glob("*.tmp"))


def test_switched_retrain_does_not_claim_the_live_index_learned(tmp_path, demo_dir, index):
    s = _settings(tmp_path, demo_dir, index)
    mark_retrained(s, started_at="2026-01-01T00:00:00+00:00", learned=False, checkpoint="x")
    m = read_manifest(s)
    assert m["retrained_at"] == "2026-01-01T00:00:00+00:00" and "learned_at" not in m
    mark_retrained(s, started_at="2026-01-02T00:00:00+00:00", checkpoint="y")
    assert read_manifest(s)["learned_at"] == "2026-01-02T00:00:00+00:00"
