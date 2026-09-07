from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw

from mcmaster_vision.pipeline.pipe import thread_family_from_pitch
from mcmaster_vision.pipeline.threads import dominant_period, measure_thread_pitch


def _threaded_rod(period_px: float, length=600, width=90, angle=0.0, canvas=(800, 400)):
    """A dark rod with lighter crests every ``period_px`` pixels along its axis."""
    im = Image.new("RGB", canvas, (240, 238, 232))
    rod = Image.new("RGB", (length, width), (70, 70, 74))
    d = ImageDraw.Draw(rod)
    x = 0.0
    while x < length:
        d.line([(x, 0), (x, width)], fill=(160, 160, 165), width=max(1, int(period_px * 0.3)))
        x += period_px
    if angle:
        rod = rod.rotate(angle, expand=True, fillcolor=(240, 238, 232))
    im.paste(rod, ((canvas[0] - rod.width) // 2, (canvas[1] - rod.height) // 2))
    return im


@pytest.mark.parametrize("period", [12.0, 20.0, 31.0])
def test_pitch_recovered_from_synthetic_threads(period):
    im = _threaded_rod(period)
    scale = 1.0 / 20  # 20 px per mm
    tp = measure_thread_pitch(im, scale)
    assert tp is not None, "no pitch found"
    assert tp.pitch_mm == pytest.approx(period * scale, rel=0.06)
    assert tp.strength >= 3


def test_pitch_is_axis_aligned_not_image_aligned():
    tp = measure_thread_pitch(_threaded_rod(18.0, angle=25), 1.0 / 20)
    assert tp is not None and tp.pitch_mm == pytest.approx(0.9, rel=0.1)


def test_no_threads_no_pitch():
    plain = Image.new("RGB", (800, 400), (240, 238, 232))
    ImageDraw.Draw(plain).rectangle((100, 150, 700, 250), fill=(70, 70, 74))
    assert measure_thread_pitch(plain, 1.0 / 20) is None
    assert dominant_period(np.zeros(10)) is None
    assert measure_thread_pitch(_threaded_rod(20.0), 0) is None


def test_pitch_maps_to_pipe_thread_family():
    # 1/8 NPT is 27 tpi (0.941 mm); BSP is 28 tpi (0.907 mm): a 3.6% difference
    tp = measure_thread_pitch(_threaded_rod(0.941 * 25), 1.0 / 25)
    assert tp is not None and thread_family_from_pitch("1/8", tp.threads_per_inch) == ["NPT"]


def test_catalog_pitch_parsing_and_consistency():
    from mcmaster_vision.pipeline.measure import Measurement, catalog_pitch_mm, size_consistency
    from mcmaster_vision.schemas import Part

    assert catalog_pitch_mm({"thread_size": '1/4"-20'})[0] == pytest.approx(1.27)
    assert catalog_pitch_mm({"thread_size": "M6 x 1"})[0] == pytest.approx(1.0)
    assert catalog_pitch_mm({"thread_size": "#8-32"})[0] == pytest.approx(0.794, abs=0.01)
    assert catalog_pitch_mm({"thread_size": "M6"}) is None
    assert catalog_pitch_mm({"pipe_size": "1/8"}, "Elbow, NPT")[0] == pytest.approx(0.941, abs=0.01)
    assert catalog_pitch_mm({"pipe_size": "1/8"}, "Elbow, BSPT")[0] == pytest.approx(
        0.907, abs=0.01
    )
    coarse = Part(
        part_number="A", name="screw", category_path=["x"], attributes={"thread_size": '1/4"-20'}
    )
    fine = Part(
        part_number="B", name="screw", category_path=["x"], attributes={"thread_size": '1/4"-28'}
    )
    m = Measurement(25.0, 6.3, 0.1, pitch_mm=1.25)
    assert size_consistency(m, coarse)[0] > 0.5 and size_consistency(m, fine)[0] < 0
    assert any("pitch" in r for r in size_consistency(m, coarse)[1])
    assert m.as_dict()["threads_per_inch"] == pytest.approx(20.3, abs=0.1)


def test_identify_reports_pitch_from_the_photo(store, index, embedder):
    from mcmaster_vision.pipeline.identify import Identifier

    ident = Identifier(store, index, embedder, top_k=10)
    res = ident.identify(_threaded_rod(20.0), tta="none", mm_per_px=1.0 / 20)
    assert res.measured and res.measured.get("pitch_mm") == pytest.approx(1.0, rel=0.08)
    assert res.measured["threads_per_inch"] == pytest.approx(25.4, rel=0.08)
