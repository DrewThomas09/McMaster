from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from mcmaster_vision.api import create_app
from mcmaster_vision.config import Settings
from mcmaster_vision.pipeline.feedback import FeedbackStore
from mcmaster_vision.pipeline.identify import Identifier
from mcmaster_vision.pipeline.rerank import Scored
from mcmaster_vision.schemas import Part
from mcmaster_vision.training.train import with_extra_images


def _png(part) -> bytes:
    buf = io.BytesIO()
    Image.open(part.image_paths[0]).save(buf, format="PNG")
    return buf.getvalue()


def test_family_hint_lists_distinguishing_attributes():
    fam = [
        Part(
            part_number="A1",
            name="Zinc Hex Bolt",
            family_id="hex:zinc",
            attributes={"thread_size": "M6", "length": '1"'},
        ),
        Part(
            part_number="A2",
            name="Zinc Hex Bolt",
            family_id="hex:zinc",
            attributes={"thread_size": "M6", "length": '2"'},
        ),
        Part(
            part_number="B1",
            name="Brass Nut",
            family_id="nut:brass",
            attributes={"thread_size": "M6"},
        ),
    ]
    scored = [Scored(p, 0.9, 0.9) for p in fam]
    hint = Identifier._family_hint(scored, [0.5, 0.3, 0.2])
    assert hint is not None and hint.family_id == "hex:zinc"
    assert hint.part_numbers == ["A1", "A2"] and abs(hint.probability - 0.8) < 1e-6
    assert hint.distinguishing_attributes == {"length": ['1"', '2"']}
    assert Identifier._family_hint(scored, [0.4, 0.1, 0.5]) is None  # no family dominates
    assert (
        Identifier._family_hint([scored[2]], [1.0]) is None
    )  # single SKU is not a family question


def test_multi_photo_identify_matches_single(identifier, store):
    part = next(store.iter_parts(with_images_only=True))
    imgs = [Image.open(p) for p in part.image_paths[:2]]
    res = identifier.identify(imgs, top_n=3)
    assert res.photos == 2 and res.candidates[0].part_number == part.part_number
    with pytest.raises(ValueError):
        identifier.identify([], top_n=3)


def test_feedback_store_roundtrip(tmp_path):
    fs = FeedbackStore(tmp_path / "q")
    fb = fs.record(b"img", "req1", "91251a537", predicted="91251A537", tier="likely", ext="png")
    fs.record(b"img", "req2", None, predicted="X", tier="unknown", ext="png")
    assert fb.part_number == "91251A537" and (tmp_path / "q" / "91251A537" / "req1.png").exists()
    assert fs.stats() == {
        "total": 2,
        "confirmed": 1,
        "unknown": 1,
        "correct_top1": 1,
        "parts_with_photos": 1,
    }
    labelled = fs.labelled_images()
    assert list(labelled) == ["91251A537"] and len(labelled["91251A537"]) == 1
    parts = with_extra_images(
        [
            Part(part_number="91251A537", name="x", image_paths=["a.png"]),
            Part(part_number="Z", name="z"),
        ],
        labelled,
    )
    assert len(parts[0].image_paths) == 2 and parts[1].image_paths == []


def test_api_multi_file_and_feedback(identifier, store, tmp_path):
    app = create_app(Settings(queries_dir=tmp_path / "queries"), identifier=identifier)
    client = TestClient(app)
    part = next(store.iter_parts(with_images_only=True))
    files = [
        ("files", ("a.png", _png(part), "image/png")),
        ("files", ("b.png", _png(part), "image/png")),
    ]
    r = client.post("/identify?top_n=3", files=files)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["photos"] == 2 and body["candidates"][0]["part_number"] == part.part_number
    assert client.post("/identify").status_code == 400

    r = client.post(
        "/feedback",
        data={
            "request_id": body["request_id"],
            "part_number": part.part_number,
            "predicted": body["candidates"][0]["part_number"],
            "tier": body["tier"],
        },
    )
    assert r.status_code == 200, r.text
    assert (tmp_path / "queries" / part.part_number / f"{body['request_id']}.jpg").exists()
    assert client.get("/feedback/stats").json()["confirmed"] == 1
    assert client.post("/feedback", data={"request_id": "nope"}).status_code == 404
    assert (
        client.post(
            "/feedback", data={"request_id": body["request_id"], "part_number": "NOPE1"}
        ).status_code
        == 404
    )
    # "none of these" with the photo re-sent
    r = client.post(
        "/feedback",
        data={"request_id": "later"},
        files={"file": ("q.png", _png(part), "image/png")},
    )
    assert r.status_code == 200 and r.json()["part_number"] is None
