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
