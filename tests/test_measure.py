from __future__ import annotations

import io

import pytest
from PIL import Image, ImageDraw

from mcmaster_vision.pipeline.measure import (
    Measurement,
    measure,
    object_extent_px,
    parse_length_mm,
    size_consistency,
)
from mcmaster_vision.schemas import Part


@pytest.mark.parametrize(
    "text,mm",
    [
        ('1/2"', 12.7),
        ('1-1/4"', 31.75),
        ("1 1/4 in", 31.75),
        ("20 mm", 20.0),
        ("2.5 cm", 25.0),
        ("M6x1.0", 6.0),
        ("m8", 8.0),
        ("#8-32", 4.166),
        ('1/4"-20', 6.35),
        ("3/4 inch", 19.05),
        ("1 ft", 304.8),
        ("18-8 Stainless", None),
        ("", None),
        ("long", None),
    ],
)
def test_parse_length_mm(text, mm):
    got = parse_length_mm(text)
    if mm is None:
        assert got is None
    else:
        assert got == pytest.approx(mm, abs=0.01)


def _photo(w=400, h=300, box=(60, 120, 300, 180), angle=0):
    im = Image.new("RGB", (w, h), (236, 236, 232))
    ImageDraw.Draw(im).rectangle(box, fill=(70, 70, 75))
    if angle:
        im = im.rotate(angle, fillcolor=(236, 236, 232), resample=Image.Resampling.BICUBIC)
    return im


def test_extent_measures_principal_axes():
    long_px, short_px = object_extent_px(_photo())
    assert long_px == pytest.approx(241, rel=0.08)
    assert short_px == pytest.approx(61, rel=0.15)
    # a tilted part is measured along its own axes, not the bounding box
    long_r, short_r = object_extent_px(_photo(angle=30))
    assert long_r == pytest.approx(241, rel=0.1) and short_r == pytest.approx(61, rel=0.25)
    assert object_extent_px(Image.new("RGB", (100, 100), "white")) is None


def test_measure_and_consistency():
    m = measure(_photo(), mm_per_px=0.1)  # 241 px -> ~24 mm long, ~6 mm wide
    assert m is not None and m.long_mm == pytest.approx(24.1, rel=0.1)
    one_inch = Part(part_number="A", name="screw", category_path=["x"], attributes={"length": '1"'})
    two_inch = Part(part_number="B", name="screw", category_path=["x"], attributes={"length": '2"'})
    half = Part(part_number="C", name="screw", category_path=["x"], attributes={"length": '1/2"'})
    washer = Part(part_number="D", name="washer", category_path=["x"], attributes={"od": "1 in"})
    nothing = Part(part_number="E", name="?", category_path=["x"], attributes={"material": "brass"})
    assert size_consistency(m, one_inch)[0] == 1.0
    assert size_consistency(m, two_inch)[0] < 0
    assert size_consistency(m, half)[0] < 0
    assert size_consistency(m, washer)[0] == 1.0
    assert size_consistency(m, nothing) == (0.0, [])
    assert size_consistency(None, one_inch) == (0.0, [])
    assert "consistent" in size_consistency(m, one_inch)[1][0]
    assert Measurement(24.1, 6.0, 0.1).as_dict()["long_mm"] == 24.1


def test_identify_with_scale_prefers_the_right_length(store, index, embedder, monkeypatch):
    """Two look-alikes that differ only in the length attribute: the photo's measured
    size decides."""
    from mcmaster_vision.pipeline.identify import Identifier

    ident = Identifier(store, index, embedder, top_k=20)
    part = next(p for p in store.iter_parts(with_images_only=True))
    img = Image.open(part.image_paths[0]).convert("RGB")
    base = ident.identify(img, tta="none", top_n=5)
    assert base.measured is None
    ext = object_extent_px(img)
    assert ext is not None
    # pretend the pictured part is 50 mm long; give the top candidates conflicting lengths
    scale = 50.0 / ext[0]
    top = [c.part_number for c in base.candidates[:2]]
    a, b = store.get(top[0]), store.get(top[1])
    a2 = a.model_copy(update={"attributes": {"length": '1/2"'}})  # 12.7 mm: wrong
    b2 = b.model_copy(update={"attributes": {"length": "2 in"}})  # 50.8 mm: right
    real_get_many = store.get_many

    def fake_get_many(pns):
        out = real_get_many(pns)
        if a.part_number in out:
            out[a.part_number] = a2
        if b.part_number in out:
            out[b.part_number] = b2
        return out

    monkeypatch.setattr(store, "get_many", fake_get_many)
    res = ident.identify(img, tta="none", top_n=5, mm_per_px=scale)
    assert res.measured and res.measured["long_mm"] == pytest.approx(50, rel=0.05)
    score = {c.part_number: c.score for c in res.candidates}
    before = {c.part_number: c.score for c in base.candidates}
    # the right length gains, the wrong one loses: the gap moves by the full size weight
    gain = (score[b.part_number] - score[a.part_number]) - (
        before[b.part_number] - before[a.part_number]
    )
    assert gain > 0.3
    reasons = {c.part_number: " ".join(c.reasons) for c in res.candidates}
    assert "consistent" in reasons[b.part_number] and "off" in reasons[a.part_number]


def test_api_accepts_scale(identifier, store):
    from fastapi.testclient import TestClient

    from mcmaster_vision.api import create_app
    from mcmaster_vision.config import Settings

    client = TestClient(create_app(Settings(), identifier=identifier))
    part = next(store.iter_parts(with_images_only=True))
    buf = io.BytesIO()
    Image.open(part.image_paths[0]).convert("RGB").save(buf, format="JPEG")
    r = client.post(
        "/identify?tta=none&mm_per_px=0.2", files={"file": ("a.jpg", buf.getvalue(), "image/jpeg")}
    )
    assert r.status_code == 200, r.text
    assert r.json()["measured"]["long_mm"] > 0
    assert (
        client.post(
            "/identify?mm_per_px=-1", files={"file": ("a.jpg", buf.getvalue(), "image/jpeg")}
        ).status_code
        == 422
    )
