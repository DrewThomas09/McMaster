from __future__ import annotations

import pytest
from PIL import Image, ImageDraw

from mcmaster_vision.pipeline.measure import measure
from mcmaster_vision.pipeline.reference import COINS_MM, find_coin


def _scene(coin=True, nut=False, washer=False, screw=True):
    im = Image.new("RGB", (640, 480), (238, 236, 230))
    d = ImageDraw.Draw(im)
    if coin:
        d.ellipse((80, 160, 240, 320), fill=(186, 170, 118))  # 160 px coin
    if nut:
        d.regular_polygon((160, 240, 80), n_sides=6, fill=(90, 90, 95))
    if washer:
        d.ellipse((80, 160, 240, 320), fill=(150, 150, 155))
        d.ellipse((130, 210, 190, 270), fill=(238, 236, 230))  # a large hole
    if screw:
        d.rectangle((340, 225, 600, 255), fill=(60, 60, 65))
    return im


def test_finds_the_coin_not_the_screw():
    c = find_coin(_scene())
    assert c is not None
    assert c.cx == pytest.approx(160, abs=8) and c.cy == pytest.approx(240, abs=8)
    assert c.diameter_px == pytest.approx(160, rel=0.08)
    seg = c.segment()
    assert seg[0] < 100 and seg[2] > 220
    # using it as a quarter measures the screw at about 260 px * (24.26 / 160) = 39 mm
    m = measure(_scene(), c.mm_per_px(COINS_MM["US quarter"]), reference=seg)
    assert m is not None and m.long_mm == pytest.approx(39.4, rel=0.1)


def test_no_coin_no_suggestion():
    assert find_coin(_scene(coin=False)) is None  # only the screw
    assert find_coin(Image.new("RGB", (400, 300), "white")) is None
    assert find_coin(_scene(coin=False, nut=True)) is None  # a hex nut is not round enough
    assert find_coin(_scene(coin=False, washer=True)) is None  # a washer's hole gives it away


def test_coin_alone_and_tiny_images():
    assert find_coin(_scene(screw=False)) is not None
    assert find_coin(Image.new("RGB", (3, 3), "white")) is None
