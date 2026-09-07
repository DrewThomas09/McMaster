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
        ('.75"', 19.05),
        ('1/4"-20 x 1"', 6.35),
        ("10 mm x 1.5 mm", 10.0),
        ('3/8" to 1/2"', None),
        ('1/2" - 3/4"', None),
        ("12 mm to 15 mm", None),
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


def test_reference_object_is_not_measured_as_the_part():
    """A quarter next to a small screw is the largest blob; the line the user drew across
    it tells the server which blob to ignore."""
    im = Image.new("RGB", (600, 400), (240, 240, 236))
    d = ImageDraw.Draw(im)
    d.ellipse((60, 100, 260, 300), fill=(180, 170, 120))  # coin: 200 px across
    d.rectangle((340, 190, 540, 214), fill=(60, 60, 65))  # screw: 200 x 24 px
    naive = object_extent_px(im)
    assert naive is not None and naive[1] > 150  # the coin: round, ~200 px both ways
    ext = object_extent_px(im, exclude=(60, 200, 260, 200))
    assert ext is not None
    assert ext[0] == pytest.approx(201, rel=0.08) and ext[1] == pytest.approx(25, rel=0.4)
    # a quarter is 24.26 mm across 200 px -> the screw measures ~24 x 3 mm
    m = measure(im, 24.26 / 200, reference=(60, 200, 260, 200))
    assert m is not None and m.long_mm == pytest.approx(24.4, rel=0.1)
    # everything excluded -> no measurement, not a bogus one
    assert object_extent_px(_photo(), exclude=(60, 150, 300, 150)) is None


def test_scale_survives_reduced_jpeg_decoding():
    """A 4000 px upload is decoded at half size; a scale given in uploaded pixels must
    still yield the true dimensions, and the reference line must land on the coin."""
    from mcmaster_vision.pipeline.preprocess import decode_image

    im = Image.new("RGB", (4000, 3000), (240, 240, 236))
    d = ImageDraw.Draw(im)
    d.ellipse((400, 1200, 1400, 2200), fill=(180, 170, 120))  # coin 1000 px
    d.rectangle((2000, 1450, 3600, 1550), fill=(60, 60, 65))  # part 1600 x 100 px
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=85)
    dec = decode_image(buf.getvalue())
    assert max(dec.size) == 2000 and dec.info["upload_scale"] == pytest.approx(2.0)
    m = measure(dec, 24.26 / 1000, reference=(400, 1700, 1400, 1700))
    assert m is not None
    assert m.long_mm == pytest.approx(1600 * 24.26 / 1000, rel=0.1)  # 38.8 mm
    assert m.short_mm == pytest.approx(100 * 24.26 / 1000, rel=0.5)


def test_diameter_uses_short_axis_when_part_has_a_length():
    m = Measurement(long_mm=25.4, short_mm=6.35, mm_per_px=0.1)
    pin = Part(
        part_number="P",
        name="dowel pin",
        category_path=["x"],
        attributes={"Diameter": '1/4"', "Length": '1"'},
    )
    fat_pin = pin.model_copy(update={"attributes": {"Diameter": '3/8"', "Length": '1"'}})
    washer = Part(part_number="W", name="washer", category_path=["x"], attributes={"OD": "1 in"})
    assert size_consistency(m, pin)[0] == 1.0
    assert size_consistency(m, fat_pin)[0] < 0.5
    assert size_consistency(m, washer)[0] == 1.0  # no length: OD is the long axis


def test_coin_hint_offered_and_usable(identifier, store):
    """A coin next to the part is reported in uploaded pixels; using it as the scale gives
    a measurement without any tapping."""
    from fastapi.testclient import TestClient
    from PIL import ImageDraw

    from mcmaster_vision.api import create_app
    from mcmaster_vision.config import Settings

    part = next(store.iter_parts(with_images_only=True))
    render = Image.open(part.image_paths[0]).convert("RGB").resize((256, 256))
    canvas = Image.new("RGB", (512, 256), (255, 255, 255))
    ImageDraw.Draw(canvas).ellipse((40, 48, 200, 208), fill=(184, 172, 120))
    canvas.paste(render, (256, 0))
    buf = io.BytesIO()
    canvas.save(buf, format="JPEG", quality=92)
    client = TestClient(create_app(Settings(), identifier=identifier))
    r = client.post("/identify?tta=none", files={"file": ("a.jpg", buf.getvalue(), "image/jpeg")})
    hint = r.json()["coin_hint"]
    assert hint and abs(hint["cx"] - 120) < 10 and abs(hint["diameter_px"] - 160) < 16
    # live-preview frames do not pay for the hint
    r2 = client.post(
        "/identify?tta=none&log=false", files={"file": ("a.jpg", buf.getvalue(), "image/jpeg")}
    )
    assert r2.json()["coin_hint"] is None
    d = hint["diameter_px"]
    ref = f"{hint['cx'] - d / 2},{hint['cy']},{hint['cx'] + d / 2},{hint['cy']}"
    r3 = client.post(
        f"/identify?tta=none&mm_per_px={24.26 / d}&ref={ref}",
        files={"file": ("a.jpg", buf.getvalue(), "image/jpeg")},
    )
    m = r3.json()["measured"]
    assert m and 20 < m["long_mm"] < 45 and r3.json()["coin_hint"] is None


def test_bore_measured_and_matched_to_schedule():
    from PIL import Image, ImageDraw

    from mcmaster_vision.pipeline.measure import Measurement, measure, size_consistency
    from mcmaster_vision.schemas import Part

    img = Image.new("RGB", (400, 400), "white")
    d = ImageDraw.Draw(img)
    d.ellipse((40, 40, 360, 360), fill=(120, 120, 130))  # a 1/2 sch-10 coupling end-on:
    d.ellipse((115, 115, 285, 285), fill="white")  # OD ~ 32 mm, bore ~ 17 mm
    m = measure(img, 0.1)
    assert m.bore_mm and abs(m.bore_mm - 17.0) < 1.0
    sch10 = Part(
        part_number="A",
        name="Thin-Wall Butt-Weld Coupling",
        category_path=["x"],
        attributes={"pipe_size": "1/2", "wall_thickness": '0.083"'},
    )
    sch80_small = Part(
        part_number="B",
        name="Thick-Wall Coupling",
        category_path=["x"],
        attributes={"pipe_size": "1/4", "schedule": "80"},
    )
    ok, reasons = size_consistency(m, sch10)
    bad, reasons_b = size_consistency(m, sch80_small)
    assert any("bore" in r and "consistent" in r for r in reasons), reasons
    assert any("bore" in r and "off" in r for r in reasons_b), reasons_b
    assert ok > bad
    # no bore measured: the bore rule stays silent
    flat = Measurement(30.0, 30.0, 0.1)
    assert not any("bore" in r for r in size_consistency(flat, sch10)[1])


def test_bore_is_the_largest_hole_and_threaded_females_are_spared():
    from PIL import Image, ImageDraw

    from mcmaster_vision.pipeline.measure import Measurement, measure, size_consistency
    from mcmaster_vision.schemas import Part

    img = Image.new("RGB", (400, 400), "white")
    d = ImageDraw.Draw(img)
    d.ellipse((20, 20, 380, 380), fill=(120, 120, 130))  # a flange: centre bore + 4 bolt holes
    d.ellipse((160, 160, 240, 240), fill="white")
    for cx, cy in ((80, 200), (320, 200), (200, 80), (200, 320)):
        d.ellipse((cx - 15, cy - 15, cx + 15, cy + 15), fill="white")
    m = measure(img, 0.1)
    assert m.bore_mm and abs(m.bore_mm - 8.0) < 0.6  # the centre bore, not all holes summed
    threaded = Part(
        part_number="T",
        name="Thick-Wall Female Coupling NPT",
        category_path=["x"],
        attributes={"pipe_size": "1/4", "schedule": "80"},
    )
    meas = Measurement(30.0, 30.0, 0.1, bore_mm=11.0)
    # a threaded female is judged by its thread's minor diameter, never by OD - 2 walls
    reasons_t = size_consistency(meas, threaded)[1]
    assert not any("pipe ID" in r for r in reasons_t)
    assert any("female thread ID" in r and "consistent" in r for r in reasons_t), reasons_t
    welded = Part(
        part_number="W",
        name="Thick-Wall Butt-Weld Coupling",
        category_path=["x"],
        attributes={"pipe_size": "1/4", "schedule": "80"},
    )
    assert any("bore" in r for r in size_consistency(meas, welded)[1])


def test_female_thread_bore_rule():
    from mcmaster_vision.pipeline.measure import Measurement, size_consistency
    from mcmaster_vision.pipeline.pipe import female_thread_id_mm
    from mcmaster_vision.schemas import Part

    assert abs(female_thread_id_mm("1/2") - 0.719 * 25.4) < 0.1 and female_thread_id_mm("x") is None
    half = Part(
        part_number="H",
        name="Female Coupling, NPT",
        category_path=["x"],
        attributes={"pipe_size": "1/2"},
    )
    quarter = Part(
        part_number="Q",
        name="Female Coupling, NPT",
        category_path=["x"],
        attributes={"pipe_size": "1/4"},
    )
    meas = Measurement(30.0, 30.0, 0.1, bore_mm=18.0)  # 1/2 NPT female bore ~ 18.3 mm
    ok, r1 = size_consistency(meas, half)
    bad, r2 = size_consistency(meas, quarter)
    assert any("female thread ID" in r and "consistent" in r for r in r1), r1
    assert any("female thread ID" in r and "off" in r for r in r2), r2
    assert ok > bad
