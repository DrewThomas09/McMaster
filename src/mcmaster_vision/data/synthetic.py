"""Synthetic catalog generator (v2).

Renders ~35 families of hardware (screws, nuts, washers, pins, gears, fittings,
springs, brackets, tools ...) with simple shading using Pillow only, so that the
entire pipeline - ingest, embed, index, identify, train, evaluate - can be
exercised end to end without proprietary McMaster-Carr data. Each SKU gets a
canonical view, an alternate view (top-down where it makes sense, otherwise a
different rotation), and further rotated views. Part numbers follow the
McMaster style ``NNNNNANNN``.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Callable, Iterator
from pathlib import Path

from PIL import Image, ImageDraw

from mcmaster_vision.schemas import Part

Color = tuple[int, int, int]

_MATERIALS: dict[str, Color] = {
    "Zinc-Plated Steel": (178, 181, 188),
    "Yellow-Zinc-Plated Steel": (205, 180, 90),
    "Black-Oxide Steel": (54, 54, 58),
    "18-8 Stainless Steel": (208, 211, 215),
    "316 Stainless Steel": (198, 202, 208),
    "Brass": (196, 160, 80),
    "Bronze": (170, 120, 70),
    "Aluminum": (216, 219, 223),
    "Nylon": (236, 232, 216),
    "Black Nylon": (40, 40, 42),
    "Copper": (184, 115, 51),
    "Titanium": (150, 150, 158),
    "Galvanized Steel": (160, 165, 170),
    "Buna-N Rubber": (30, 30, 32),
}

_THREADS = [
    "#4-40",
    "#6-32",
    "#8-32",
    "#10-24",
    "#10-32",
    '1/4"-20',
    '5/16"-18',
    '3/8"-16',
    '1/2"-13',
    "M3",
    "M4",
    "M5",
    "M6",
    "M8",
    "M10",
]
_LENGTHS = ['1/4"', '3/8"', '1/2"', '5/8"', '3/4"', '1"', '1-1/4"', '1-1/2"', '2"', '2-1/2"', '3"']
_ODS = ['1/4"', '3/8"', '1/2"', '5/8"', '3/4"', '1"', '1-1/4"', '1-1/2"', '2"']

# kind -> (category path, display name, has_top_view)
_FAMILIES: dict[str, tuple[list[str], str, bool]] = {
    "socket_head_screw": (
        ["Fastening & Joining", "Screws & Bolts", "Socket Head Screws"],
        "Socket Head Screw",
        True,
    ),
    "hex_bolt": (
        ["Fastening & Joining", "Screws & Bolts", "Hex Head Screws"],
        "Hex Head Screw",
        True,
    ),
    "pan_head_screw": (
        ["Fastening & Joining", "Screws & Bolts", "Pan Head Screws"],
        "Phillips Pan Head Screw",
        True,
    ),
    "flat_head_screw": (
        ["Fastening & Joining", "Screws & Bolts", "Flat Head Screws"],
        "Flat Head Screw",
        True,
    ),
    "button_head_screw": (
        ["Fastening & Joining", "Screws & Bolts", "Button Head Screws"],
        "Button Head Screw",
        True,
    ),
    "set_screw": (
        ["Fastening & Joining", "Screws & Bolts", "Set Screws"],
        "Cup-Point Set Screw",
        True,
    ),
    "thumb_screw": (
        ["Fastening & Joining", "Screws & Bolts", "Thumb Screws"],
        "Knurled Thumb Screw",
        True,
    ),
    "eye_bolt": (["Fastening & Joining", "Screws & Bolts", "Eyebolts"], "Eyebolt", False),
    "u_bolt": (["Fastening & Joining", "Screws & Bolts", "U-Bolts"], "U-Bolt", False),
    "threaded_rod": (
        ["Fastening & Joining", "Threaded Rods", "Fully Threaded Rods"],
        "Threaded Rod",
        False,
    ),
    "hex_nut": (["Fastening & Joining", "Nuts", "Hex Nuts"], "Hex Nut", True),
    "nylon_lock_nut": (["Fastening & Joining", "Nuts", "Locknuts"], "Nylon-Insert Locknut", True),
    "cap_nut": (["Fastening & Joining", "Nuts", "Cap Nuts"], "Acorn Cap Nut", True),
    "wing_nut": (["Fastening & Joining", "Nuts", "Wing Nuts"], "Wing Nut", True),
    "square_nut": (["Fastening & Joining", "Nuts", "Square Nuts"], "Square Nut", True),
    "flat_washer": (["Fastening & Joining", "Washers", "Flat Washers"], "Flat Washer", False),
    "fender_washer": (["Fastening & Joining", "Washers", "Fender Washers"], "Fender Washer", False),
    "lock_washer": (
        ["Fastening & Joining", "Washers", "Split Lock Washers"],
        "Split Lock Washer",
        False,
    ),
    "dowel_pin": (["Fastening & Joining", "Pins", "Dowel Pins"], "Dowel Pin", True),
    "clevis_pin": (["Fastening & Joining", "Pins", "Clevis Pins"], "Clevis Pin", True),
    "cotter_pin": (["Fastening & Joining", "Pins", "Cotter Pins"], "Cotter Pin", False),
    "rivet": (["Fastening & Joining", "Rivets", "Blind Rivets"], "Blind Rivet", True),
    "spur_gear": (["Power Transmission", "Gears", "Spur Gears"], "Spur Gear", False),
    "sprocket": (
        ["Power Transmission", "Sprockets", "Roller Chain Sprockets"],
        "Roller Chain Sprocket",
        False,
    ),
    "bearing": (["Power Transmission", "Bearings", "Ball Bearings"], "Ball Bearing", False),
    "shaft_collar": (
        ["Power Transmission", "Shaft Collars", "Set Screw Shaft Collars"],
        "Set Screw Shaft Collar",
        False,
    ),
    "pulley": (["Power Transmission", "Pulleys", "V-Belt Pulleys"], "V-Belt Pulley", False),
    "key_stock": (["Power Transmission", "Keys", "Key Stock"], "Machine Key", False),
    "compression_spring": (
        ["Fastening & Joining", "Springs", "Compression Springs"],
        "Compression Spring",
        True,
    ),
    "o_ring": (["Sealing", "O-Rings", "Round O-Rings"], "O-Ring", False),
    "hose_clamp": (
        ["Pipe, Tubing, Hose & Fittings", "Clamps", "Worm-Drive Hose Clamps"],
        "Worm-Drive Hose Clamp",
        False,
    ),
    "pipe_elbow": (
        ["Pipe, Tubing, Hose & Fittings", "Pipe Fittings", "Elbows"],
        "90 Degree Pipe Elbow",
        False,
    ),
    "pipe_tee": (["Pipe, Tubing, Hose & Fittings", "Pipe Fittings", "Tees"], "Pipe Tee", False),
    "l_bracket": (["Hardware", "Brackets", "Corner Brackets"], "Corner Bracket", False),
    "flat_bracket": (["Hardware", "Brackets", "Mending Plates"], "Mending Plate", False),
    "hinge": (["Hardware", "Hinges", "Butt Hinges"], "Butt Hinge", False),
    "knob": (["Hardware", "Knobs", "Fluted Knobs"], "Fluted Knob", False),
    "hex_key": (["Hand Tools", "Hex Keys", "L-Keys"], "Hex L-Key", False),
    "drill_bit": (
        ["Sawing & Cutting", "Drill Bits", "Jobbers' Drill Bits"],
        "Jobbers' Drill Bit",
        True,
    ),
}


# ---------------------------------------------------------------------------
# drawing helpers
# ---------------------------------------------------------------------------
def _mix(a: Color, b: Color, t: float) -> Color:
    return tuple(int(a[i] * (1 - t) + b[i] * t) for i in range(3))  # type: ignore[return-value]


def _dark(c: Color, k: float = 0.55) -> Color:
    return _mix(c, (0, 0, 0), 1 - k) if k < 1 else c


def _light(c: Color, k: float = 0.3) -> Color:
    return _mix(c, (245, 245, 245), k)


def _shade_rect(
    d: ImageDraw.ImageDraw, box, color: Color, axis: str = "x", outline: Color | None = None
) -> None:
    """Cylinder-like shading across ``axis`` ('x' = shade varies left->right)."""
    x0, y0, x1, y1 = (int(v) for v in box)
    n = (x1 - x0) if axis == "x" else (y1 - y0)
    for i in range(max(n, 1)):
        t = i / max(n - 1, 1)
        k = 1 - abs(t - 0.4) * 1.6  # brightest ~40% across
        c = _mix(_dark(color, 0.6), _light(color, 0.35), max(0.0, min(1.0, k)))
        if axis == "x":
            d.line([x0 + i, y0, x0 + i, y1], fill=c)
        else:
            d.line([x0, y0 + i, x1, y0 + i], fill=c)
    d.rectangle([x0, y0, x1, y1], outline=outline or _dark(color, 0.45), width=2)


def _shade_ellipse(d: ImageDraw.ImageDraw, box, color: Color, steps: int = 12) -> None:
    x0, y0, x1, y1 = (float(v) for v in box)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    rx, ry = (x1 - x0) / 2, (y1 - y0) / 2
    for i in range(steps):
        t = i / steps
        c = _mix(_dark(color, 0.7), _light(color, 0.35), t)
        ox, oy = cx - rx * 0.18 * t, cy - ry * 0.18 * t
        d.ellipse(
            [
                ox - rx * (1 - t * 0.85),
                oy - ry * (1 - t * 0.85),
                ox + rx * (1 - t * 0.85),
                oy + ry * (1 - t * 0.85),
            ],
            fill=c,
        )
    d.ellipse([x0, y0, x1, y1], outline=_dark(color, 0.45), width=2)


def _threads(d: ImageDraw.ImageDraw, box, pitch: int, color: Color, slant: float = 0.5) -> None:
    x0, y0, x1, y1 = (int(v) for v in box)
    dk = _dark(color, 0.5)
    for y in range(y0 + pitch, y1, max(pitch, 2)):
        d.line([x0, y, x1, y - int(pitch * slant)], fill=dk, width=1)
        d.line([x0, y + 1, x1, y + 1 - int(pitch * slant)], fill=_light(color, 0.5), width=1)


def _polar(cx: float, cy: float, r: float, deg: float) -> tuple[int, int]:
    return int(cx + r * math.cos(math.radians(deg))), int(cy + r * math.sin(math.radians(deg)))


def _polygon(cx: float, cy: float, r: float, n: int, rot: float = 0) -> list[tuple[int, int]]:
    return [_polar(cx, cy, r, rot + 360 * i / n) for i in range(n)]


def _hex_top(d, c, r, color, rot=0):
    d.polygon(_polygon(c, c, r, 6, rot), fill=color, outline=_dark(color, 0.45))
    d.ellipse(
        [c - r * 0.55, c - r * 0.55, c + r * 0.55, c + r * 0.55], outline=_dark(color, 0.6), width=2
    )


# ---------------------------------------------------------------------------
# part renderers: f(d, c, r, color, params, view) ; c = centre, r = size radius
# ---------------------------------------------------------------------------
def _screw_side(d, c, r, color, p, head: str):
    head_h = int(r * (0.22 if head in ("flat", "button", "set") else 0.34))
    shaft_w = int(r * 0.3)
    length = int(r * p["length_frac"])
    top = c - (head_h + length) // 2
    head_w = int(r * (0.9 if head != "set" else 0.3))
    if head == "socket":
        _shade_rect(d, [c - head_w // 2, top, c + head_w // 2, top + head_h], color, "x")
        d.ellipse([c - r * 0.15, top + 3, c + r * 0.15, top + head_h - 3], fill=_dark(color, 0.35))
    elif head == "hex":
        _shade_rect(d, [c - head_w // 2, top, c + head_w // 2, top + head_h], color, "x")
        d.line(
            [c - head_w // 6, top, c - head_w // 6, top + head_h], fill=_dark(color, 0.5), width=2
        )
        d.line(
            [c + head_w // 6, top, c + head_w // 6, top + head_h], fill=_light(color, 0.5), width=2
        )
    elif head == "pan":
        d.chord(
            [c - head_w // 2, top, c + head_w // 2, top + 2 * head_h],
            180,
            360,
            fill=_light(color, 0.15),
            outline=_dark(color, 0.45),
            width=2,
        )
        d.line([c, top + 2, c, top + head_h - 2], fill=_dark(color, 0.35), width=3)
    elif head == "flat":
        d.polygon(
            [
                (c - head_w // 2, top),
                (c + head_w // 2, top),
                (c + shaft_w // 2, top + head_h),
                (c - shaft_w // 2, top + head_h),
            ],
            fill=_light(color, 0.15),
            outline=_dark(color, 0.45),
        )
        d.line(
            [c - head_w // 4, top + 2, c + head_w // 4, top + 2], fill=_dark(color, 0.35), width=3
        )
    elif head == "button":
        d.chord(
            [c - head_w // 2, top, c + head_w // 2, top + 2 * head_h],
            180,
            360,
            fill=_light(color, 0.2),
            outline=_dark(color, 0.45),
            width=2,
        )
        d.rectangle([c - r * 0.1, top + 3, c + r * 0.1, top + head_h - 1], fill=_dark(color, 0.35))
    elif head == "thumb":
        _shade_rect(d, [c - head_w // 2, top, c + head_w // 2, top + head_h], color, "x")
        for x in range(c - head_w // 2 + 3, c + head_w // 2 - 2, 4):
            d.line([x, top + 2, x, top + head_h - 2], fill=_dark(color, 0.5))
    elif head == "set":
        head_h = 0
    _shade_rect(
        d, [c - shaft_w // 2, top + head_h, c + shaft_w // 2, top + head_h + length], color, "x"
    )
    _threads(
        d,
        [c - shaft_w // 2 + 1, top + head_h, c + shaft_w // 2 - 1, top + head_h + length],
        max(3, int(p["pitch"])),
        color,
    )
    if head == "set":
        d.ellipse([c - r * 0.08, top + 3, c + r * 0.08, top + 10], fill=_dark(color, 0.35))


def _screw_top(d, c, r, color, p, head: str):
    rr = r * 0.45
    if head == "hex":
        _hex_top(d, c, rr, color, p["rot"])
    elif head == "set":
        _shade_ellipse(d, [c - rr * 0.5, c - rr * 0.5, c + rr * 0.5, c + rr * 0.5], color)
        d.polygon(_polygon(c, c, rr * 0.25, 6, p["rot"]), fill=_dark(color, 0.35))
    else:
        _shade_ellipse(d, [c - rr, c - rr, c + rr, c + rr], color)
        if head in ("socket", "button", "thumb"):
            d.polygon(_polygon(c, c, rr * 0.4, 6, p["rot"]), fill=_dark(color, 0.35))
        elif head == "pan":
            d.line([c - rr * 0.5, c, c + rr * 0.5, c], fill=_dark(color, 0.35), width=4)
            d.line([c, c - rr * 0.5, c, c + rr * 0.5], fill=_dark(color, 0.35), width=4)
        elif head == "flat":
            d.line([c - rr * 0.6, c, c + rr * 0.6, c], fill=_dark(color, 0.35), width=4)
        if head == "thumb":
            for a in range(0, 360, 15):
                d.line(
                    [_polar(c, c, rr * 0.85, a), _polar(c, c, rr, a)],
                    fill=_dark(color, 0.5),
                    width=2,
                )


def _nut_side(d, c, r, color, p, kind):
    h = int(r * 0.4)
    w = int(r * 0.9)
    _shade_rect(d, [c - w // 2, c - h // 2, c + w // 2, c + h // 2], color, "x")
    d.line([c - w // 6, c - h // 2, c - w // 6, c + h // 2], fill=_dark(color, 0.5), width=2)
    d.line([c + w // 6, c - h // 2, c + w // 6, c + h // 2], fill=_light(color, 0.5), width=2)
    if kind == "nylon_lock_nut":
        d.rectangle(
            [c - w // 2 + 2, c - h // 2 - h // 3, c + w // 2 - 2, c - h // 2],
            fill=(60, 90, 180),
            outline=(30, 40, 90),
        )
    if kind == "cap_nut":
        d.chord(
            [c - w // 2, c - h // 2 - h, c + w // 2, c + h // 2],
            180,
            360,
            fill=_light(color, 0.2),
            outline=_dark(color, 0.45),
            width=2,
        )
    if kind == "wing_nut":
        for sgn in (-1, 1):
            d.polygon(
                [
                    (c + sgn * w // 2, c),
                    (c + sgn * (w // 2 + int(r * 0.35)), c - int(r * 0.55)),
                    (c + sgn * (w // 2 + int(r * 0.15)), c - int(r * 0.6)),
                    (c + sgn * w // 3, c - h // 2),
                ],
                fill=color,
                outline=_dark(color, 0.45),
            )


def _nut_top(d, c, r, color, p, kind):
    rr = r * 0.45
    if kind == "square_nut":
        d.polygon(_polygon(c, c, rr, 4, 45 + p["rot"]), fill=color, outline=_dark(color, 0.45))
    else:
        _hex_top(d, c, rr, color, p["rot"])
    hole = rr * (0.3 if kind == "cap_nut" else 0.42)
    if kind == "cap_nut":
        _shade_ellipse(d, [c - rr * 0.6, c - rr * 0.6, c + rr * 0.6, c + rr * 0.6], color)
    else:
        d.ellipse(
            [c - hole, c - hole, c + hole, c + hole],
            fill=(250, 250, 250),
            outline=_dark(color, 0.45),
            width=2,
        )
        if kind == "nylon_lock_nut":
            d.ellipse(
                [c - hole * 1.25, c - hole * 1.25, c + hole * 1.25, c + hole * 1.25],
                outline=(60, 90, 180),
                width=4,
            )
    if kind == "wing_nut":
        for sgn in (-1, 1):
            d.polygon(
                [
                    (c + sgn * rr, c - rr * 0.2),
                    (c + sgn * rr * 1.7, c - rr * 0.25),
                    (c + sgn * rr * 1.7, c + rr * 0.25),
                    (c + sgn * rr, c + rr * 0.2),
                ],
                fill=color,
                outline=_dark(color, 0.45),
            )


def _washer(d, c, r, color, p, kind):
    ro = r * (0.5 if kind != "fender_washer" else 0.5)
    ri = ro * (p["inner"] * (0.5 if kind == "fender_washer" else 1.0) + 0.1)
    _shade_ellipse(d, [c - ro, c - ro, c + ro, c + ro], color)
    d.ellipse(
        [c - ri, c - ri, c + ri, c + ri], fill=(250, 250, 250), outline=_dark(color, 0.45), width=2
    )
    if kind == "lock_washer":
        d.pieslice(
            [c - ro - 1, c - ro - 1, c + ro + 1, c + ro + 1],
            350 + p["rot"],
            12 + p["rot"],
            fill=(250, 250, 250),
        )
        d.line(
            [_polar(c, c, ri, 355 + p["rot"]), _polar(c, c, ro, 355 + p["rot"])],
            fill=_dark(color, 0.45),
            width=2,
        )


def _gear(d, c, r, color, p, kind):
    teeth = p["teeth"]
    pts = []
    for i in range(teeth * 2):
        if kind == "sprocket":
            rad = r * 0.5 if i % 2 == 0 else r * 0.4
        else:
            rad = r * 0.5 if i % 2 == 0 else r * 0.42
        pts.append(_polar(c, c, rad, 360 * i / (teeth * 2) + p["rot"]))
    d.polygon(pts, fill=color, outline=_dark(color, 0.45))
    d.ellipse(
        [c - r * 0.3, c - r * 0.3, c + r * 0.3, c + r * 0.3], outline=_dark(color, 0.5), width=2
    )
    hole = r * 0.12
    d.ellipse(
        [c - hole, c - hole, c + hole, c + hole],
        fill=(250, 250, 250),
        outline=_dark(color, 0.45),
        width=2,
    )
    if kind == "sprocket":
        for k in range(p["holes"] + 2):
            hx, hy = _polar(c, c, r * 0.22, 360 * k / (p["holes"] + 2) + p["rot"])
            d.ellipse(
                [hx - 5, hy - 5, hx + 5, hy + 5], fill=(250, 250, 250), outline=_dark(color, 0.45)
            )
    else:
        d.rectangle([c - hole, c - hole - 4, c + hole, c - hole + 2], fill=(250, 250, 250))


def _bearing(d, c, r, color, p, view):
    ro = r * 0.5
    _shade_ellipse(d, [c - ro, c - ro, c + ro, c + ro], color)
    ri = ro * 0.62
    d.ellipse(
        [c - ri, c - ri, c + ri, c + ri],
        fill=_light(color, 0.15),
        outline=_dark(color, 0.45),
        width=2,
    )
    rb = ro * 0.4
    d.ellipse(
        [c - rb, c - rb, c + rb, c + rb], fill=(250, 250, 250), outline=_dark(color, 0.45), width=2
    )
    for i in range(p["teeth"]):
        bx, by = _polar(c, c, ro * 0.8, 360 * i / p["teeth"] + view * 11)
        d.ellipse(
            [bx - 4, by - 4, bx + 4, by + 4], fill=(240, 240, 240), outline=_dark(color, 0.45)
        )


def _pin(d, c, r, color, p, kind):
    w = int(r * (0.22 + p["inner"] * 0.4))
    L = int(r * p["length_frac"] * 0.6)
    box = [c - L // 2, c - w // 2, c + L // 2, c + w // 2]
    if kind == "dowel_pin":
        d.rounded_rectangle(
            box, radius=w // 3, fill=_light(color, 0.1), outline=_dark(color, 0.45), width=2
        )
        _shade_rect(
            d,
            [box[0] + 2, box[1] + 2, box[2] - 2, box[3] - 2],
            color,
            "y",
            outline=_light(color, 0.1),
        )
    elif kind == "clevis_pin":
        _shade_rect(d, box, color, "y")
        d.rectangle(
            [box[0] - w // 2, c - w * 0.8, box[0], c + w * 0.8],
            fill=color,
            outline=_dark(color, 0.45),
            width=2,
        )
        d.ellipse(
            [box[2] - w, c - 3, box[2] - w + 6, c + 3],
            fill=(250, 250, 250),
            outline=_dark(color, 0.45),
        )
    elif kind == "rivet":
        d.chord(
            [box[0] - w, c - w, box[0] + w, c + w],
            90,
            270,
            fill=_light(color, 0.15),
            outline=_dark(color, 0.45),
            width=2,
        )
        _shade_rect(d, [box[0], c - w // 3, box[2], c + w // 3], color, "y")
        _shade_rect(
            d,
            [box[0] - w // 2, c - w // 2, box[0] + int(L * 0.25), c + w // 2],
            _dark(color, 0.8),
            "y",
        )
    elif kind == "threaded_rod":
        _shade_rect(d, box, color, "y")
        for x in range(box[0] + 3, box[2], max(3, int(p["pitch"]))):
            d.line([x, box[1], x - 2, box[3]], fill=_dark(color, 0.5))


def _cotter_pin(d, c, r, color, p):
    w = max(3, int(r * 0.09))
    L = int(r * 0.9)
    d.ellipse(
        [c - L // 2 - w * 3, c - w * 3, c - L // 2 + w * 3, c + w * 3],
        outline=_dark(color, 0.6),
        width=w,
    )
    d.line(
        [c - L // 2 + w * 2, c - w // 2, c + L // 2, c - w // 2], fill=_dark(color, 0.6), width=w
    )
    d.line(
        [c - L // 2 + w * 2, c + w // 2 + 1, c + L // 2 - w * 3, c + w * 2],
        fill=_dark(color, 0.6),
        width=w,
    )


def _spring(d, c, r, color, p, view):
    w = int(r * 0.5)
    L = int(r * p["length_frac"] * 0.55)
    coils = p["teeth"]
    dk = _dark(color, 0.5)
    if view == 1:
        for i in range(coils):
            rr = w // 2 - (i % 2) * 2
            d.ellipse([c - rr, c - rr, c + rr, c + rr], outline=dk, width=3)
        return
    y0 = c - L // 2
    step = max(4, L // coils)
    for i in range(coils):
        y = y0 + i * step
        d.line([c - w // 2, y + step, c + w // 2, y], fill=dk, width=3)
        d.line([c + w // 2, y, c - w // 2, y + step // 2], fill=_light(color, 0.2), width=2)
    d.rectangle([c - w // 2, y0, c + w // 2, y0 + coils * step], outline=None)


def _o_ring(d, c, r, color, p):
    ro = r * 0.5
    wdt = max(3, int(r * (1 - p["inner"]) * 0.18))
    d.ellipse([c - ro, c - ro, c + ro, c + ro], outline=_dark(color, 0.7), width=wdt)
    d.arc(
        [c - ro + wdt // 2, c - ro + wdt // 2, c + ro - wdt // 2, c + ro - wdt // 2],
        200,
        300,
        fill=_light(color, 0.35),
        width=max(1, wdt // 3),
    )


def _bracket(d, c, r, color, p, kind):
    t = int(r * 0.18)
    if kind == "l_bracket":
        _shade_rect(d, [c - r // 2, c - r // 2, c - r // 2 + t, c + r // 2], color, "x")
        _shade_rect(d, [c - r // 2, c + r // 2 - t, c + r // 2, c + r // 2], color, "y")
        for k in range(p["holes"]):
            y = c - r // 2 + t + (k + 1) * (r - 2 * t) // (p["holes"] + 1)
            d.ellipse(
                [c - r // 2 + t // 4, y - t // 4, c - r // 2 + 3 * t // 4, y + t // 4],
                fill=(250, 250, 250),
                outline=_dark(color, 0.45),
            )
            x = c - r // 2 + t + (k + 1) * (r - 2 * t) // (p["holes"] + 1)
            d.ellipse(
                [x - t // 4, c + r // 2 - 3 * t // 4, x + t // 4, c + r // 2 - t // 4],
                fill=(250, 250, 250),
                outline=_dark(color, 0.45),
            )
    elif kind == "flat_bracket":
        L = int(r * 0.95)
        _shade_rect(d, [c - L // 2, c - t // 2, c + L // 2, c + t // 2], color, "y")
        for k in range(p["holes"] + 1):
            x = c - L // 2 + (k + 1) * L // (p["holes"] + 2)
            d.ellipse(
                [x - t // 4, c - t // 4, x + t // 4, c + t // 4],
                fill=(250, 250, 250),
                outline=_dark(color, 0.45),
            )
    elif kind == "hinge":
        L = int(r * 0.9)
        h = int(r * 0.5)
        for sgn in (-1, 1):
            x0, x1 = (c + sgn * 4, c + sgn * L // 2)
            _shade_rect(d, [min(x0, x1), c - h // 2, max(x0, x1), c + h // 2], color, "x")
            for k in range(p["holes"]):
                y = c - h // 2 + (k + 1) * h // (p["holes"] + 1)
                x = c + sgn * L // 4
                d.ellipse(
                    [x - 4, y - 4, x + 4, y + 4], fill=(250, 250, 250), outline=_dark(color, 0.45)
                )
        _shade_rect(d, [c - 5, c - h // 2 - 3, c + 5, c + h // 2 + 3], color, "x")
        for y in range(c - h // 2, c + h // 2, max(6, h // 4)):
            d.line([c - 5, y, c + 5, y], fill=_dark(color, 0.45))


def _collar(d, c, r, color, p):
    ro = r * 0.5
    _shade_ellipse(d, [c - ro, c - ro, c + ro, c + ro], color)
    ri = ro * (0.45 + p["inner"] * 0.6)
    d.ellipse(
        [c - ri, c - ri, c + ri, c + ri], fill=(250, 250, 250), outline=_dark(color, 0.45), width=2
    )
    d.rectangle([c - 4, c - ro - 2, c + 4, c - ri + 2], fill=_dark(color, 0.5))
    d.ellipse([c - 3, c - ro + 2, c + 3, c - ro + 8], fill=_dark(color, 0.3))


def _pulley(d, c, r, color, p):
    ro = r * 0.5
    _shade_ellipse(d, [c - ro, c - ro, c + ro, c + ro], color)
    d.ellipse(
        [c - ro * 0.9, c - ro * 0.9, c + ro * 0.9, c + ro * 0.9], outline=_dark(color, 0.5), width=3
    )
    d.ellipse(
        [c - ro * 0.55, c - ro * 0.55, c + ro * 0.55, c + ro * 0.55],
        fill=_light(color, 0.15),
        outline=_dark(color, 0.45),
        width=2,
    )
    hole = ro * 0.15
    d.ellipse(
        [c - hole, c - hole, c + hole, c + hole],
        fill=(250, 250, 250),
        outline=_dark(color, 0.45),
        width=2,
    )
    d.rectangle([c - hole * 0.6, c - hole - 4, c + hole * 0.6, c - hole + 2], fill=(250, 250, 250))


def _key_stock(d, c, r, color, p):
    L = int(r * 0.9)
    w = int(r * (0.12 + p["inner"] * 0.4))
    _shade_rect(d, [c - L // 2, c - w // 2, c + L // 2, c + w // 2], color, "y")


def _hose_clamp(d, c, r, color, p):
    ro = r * 0.5
    wdt = max(4, int(r * 0.1))
    d.ellipse([c - ro, c - ro, c + ro, c + ro], outline=color, width=wdt)
    d.ellipse([c - ro, c - ro, c + ro, c + ro], outline=_dark(color, 0.5), width=1)
    for a in range(0, 360, 12):
        d.line(
            [_polar(c, c, ro - wdt + 1, a), _polar(c, c, ro - 1, a)],
            fill=_dark(color, 0.55),
            width=1,
        )
    hx, hy = _polar(c, c, ro, 300 + p["rot"] % 60)
    d.rectangle(
        [hx - 10, hy - 8, hx + 10, hy + 8],
        fill=_light(color, 0.1),
        outline=_dark(color, 0.45),
        width=2,
    )
    d.rectangle(
        [hx + 8, hy - 4, hx + 18, hy + 4], fill=_dark(color, 0.6), outline=_dark(color, 0.45)
    )


def _pipe(d, c, r, color, p, kind):
    w = int(r * 0.32)
    L = int(r * 0.5)
    _shade_rect(
        d,
        [c - L, c - w // 2, c + (L if kind == "pipe_tee" else 0) + w // 2, c + w // 2],
        color,
        "y",
    )
    if kind == "pipe_elbow":
        _shade_rect(d, [c - w // 2, c - L, c + w // 2, c + w // 2], color, "x")
        d.rectangle(
            [c - w // 2, c - L, c + w // 2, c - L + w // 4],
            fill=_dark(color, 0.75),
            outline=_dark(color, 0.45),
        )
        d.rectangle(
            [c - L, c - w // 2, c - L + w // 4, c + w // 2],
            fill=_dark(color, 0.75),
            outline=_dark(color, 0.45),
        )
    else:
        _shade_rect(d, [c - w // 2, c - L, c + w // 2, c], color, "x")
        for box in (
            [c - L, c - w // 2, c - L + w // 4, c + w // 2],
            [c + L - w // 4 + w // 2, c - w // 2, c + L + w // 2, c + w // 2],
            [c - w // 2, c - L, c + w // 2, c - L + w // 4],
        ):
            d.rectangle(box, fill=_dark(color, 0.75), outline=_dark(color, 0.45))


def _knob(d, c, r, color, p):
    ro = r * 0.5
    for a in range(0, 360, 360 // max(5, p["teeth"])):
        x, y = _polar(c, c, ro * 0.85, a + p["rot"])
        d.ellipse(
            [x - ro * 0.25, y - ro * 0.25, x + ro * 0.25, y + ro * 0.25],
            fill=color,
            outline=_dark(color, 0.45),
        )
    _shade_ellipse(d, [c - ro * 0.85, c - ro * 0.85, c + ro * 0.85, c + ro * 0.85], color)
    d.ellipse(
        [c - ro * 0.3, c - ro * 0.3, c + ro * 0.3, c + ro * 0.3],
        fill=_light(color, 0.2),
        outline=_dark(color, 0.45),
        width=2,
    )


def _hex_key(d, c, r, color, p):
    w = max(4, int(r * (0.08 + p["inner"] * 0.2)))
    L = int(r * 0.9)
    s = int(r * 0.35)
    _shade_rect(d, [c - L // 2, c - s // 2, c - L // 2 + w, c + s // 2 + L // 3], color, "x")
    _shade_rect(d, [c - L // 2, c - s // 2, c + L // 2, c - s // 2 + w], color, "y")


def _drill_bit(d, c, r, color, p, view):
    w = int(r * (0.12 + p["inner"] * 0.3))
    L = int(r * p["length_frac"] * 0.6)
    if view == 1:
        _shade_ellipse(d, [c - w, c - w, c + w, c + w], color)
        d.line(
            [c - w * 0.8, c - w * 0.3, c + w * 0.8, c + w * 0.3], fill=_dark(color, 0.3), width=3
        )
        return
    box = [c - L // 2, c - w // 2, c + L // 2, c + w // 2]
    _shade_rect(d, box, color, "y")
    for x in range(box[0] + int(L * 0.25), box[2], max(4, w)):
        d.line([x, box[1], x + w, box[3]], fill=_dark(color, 0.4), width=2)
    d.polygon(
        [(box[2], box[1]), (box[2] + w // 2, c), (box[2], box[3])],
        fill=_light(color, 0.2),
        outline=_dark(color, 0.45),
    )


# ---------------------------------------------------------------------------
class SyntheticCatalog:
    def __init__(
        self,
        n_parts: int = 300,
        images_per_part: int = 3,
        size: int = 256,
        seed: int = 0,
        kinds: list[str] | None = None,
    ):
        self.n_parts = n_parts
        self.images_per_part = images_per_part
        self.size = size
        self.rng = random.Random(seed)
        self.kinds = kinds or list(_FAMILIES)

    # --------------------------------------------------------- rendering
    def _render(self, kind: str, color: Color, p: dict, view: int) -> Image.Image:
        s = self.size
        img = Image.new("RGB", (s, s), (255, 255, 255))
        d = ImageDraw.Draw(img)
        c = s // 2
        r = int(s * p["scale"])
        top_view = view == 1 and _FAMILIES[kind][2]
        screws = {
            "socket_head_screw": "socket",
            "hex_bolt": "hex",
            "pan_head_screw": "pan",
            "flat_head_screw": "flat",
            "button_head_screw": "button",
            "set_screw": "set",
            "thumb_screw": "thumb",
        }
        nuts = {"hex_nut", "nylon_lock_nut", "cap_nut", "wing_nut", "square_nut"}

        if kind in screws:
            (_screw_top if top_view else _screw_side)(d, c, r, color, p, screws[kind])
        elif kind in nuts:
            (_nut_top if not top_view else _nut_side)(d, c, r, color, p, kind)
        elif kind in ("flat_washer", "fender_washer", "lock_washer"):
            _washer(d, c, r, color, p, kind)
        elif kind in ("spur_gear", "sprocket"):
            _gear(d, c, r, color, p, kind)
        elif kind == "bearing":
            _bearing(d, c, r, color, p, view)
        elif kind in ("dowel_pin", "clevis_pin", "rivet", "threaded_rod"):
            if top_view:
                rr = r * (0.11 + p["inner"] * 0.2)
                _shade_ellipse(d, [c - rr, c - rr, c + rr, c + rr], color)
            else:
                _pin(d, c, r, color, p, kind)
        elif kind == "cotter_pin":
            _cotter_pin(d, c, r, color, p)
        elif kind == "compression_spring":
            _spring(d, c, r, color, p, 1 if top_view else 0)
        elif kind == "o_ring":
            _o_ring(d, c, r, color, p)
        elif kind in ("l_bracket", "flat_bracket", "hinge"):
            _bracket(d, c, r, color, p, kind)
        elif kind == "shaft_collar":
            _collar(d, c, r, color, p)
        elif kind == "pulley":
            _pulley(d, c, r, color, p)
        elif kind == "key_stock":
            _key_stock(d, c, r, color, p)
        elif kind == "hose_clamp":
            _hose_clamp(d, c, r, color, p)
        elif kind in ("pipe_elbow", "pipe_tee"):
            _pipe(d, c, r, color, p, kind)
        elif kind == "knob":
            _knob(d, c, r, color, p)
        elif kind == "hex_key":
            _hex_key(d, c, r, color, p)
        elif kind == "drill_bit":
            _drill_bit(d, c, r, color, p, 1 if top_view else 0)
        elif kind == "eye_bolt":
            ring = r * 0.22
            d.ellipse(
                [c - ring, c - r // 2, c + ring, c - r // 2 + 2 * ring],
                outline=color,
                width=max(4, int(ring * 0.45)),
            )
            d.ellipse(
                [c - ring, c - r // 2, c + ring, c - r // 2 + 2 * ring],
                outline=_dark(color, 0.45),
                width=1,
            )
            _shade_rect(
                d, [c - r * 0.12, c - r // 2 + 2 * ring, c + r * 0.12, c + r // 2], color, "x"
            )
            _threads(
                d,
                [c - r * 0.12, c - r // 2 + 2 * ring, c + r * 0.12, c + r // 2],
                max(3, int(p["pitch"])),
                color,
            )
        elif kind == "u_bolt":
            w = int(r * 0.12)
            gap = int(r * (0.3 + p["inner"] * 0.6))
            d.arc(
                [c - gap // 2 - w // 2, c - r // 2, c + gap // 2 + w // 2, c - r // 2 + gap + w],
                180,
                360,
                fill=color,
                width=w,
            )
            for sgn in (-1, 1):
                x = c + sgn * gap // 2
                _shade_rect(
                    d,
                    [x - w // 2, c - r // 2 + gap // 2 + w // 2, x + w // 2, c + r // 2],
                    color,
                    "x",
                )
                _threads(d, [x - w // 2, c, x + w // 2, c + r // 2], max(3, int(p["pitch"])), color)

        angle = 0 if view <= 1 else 25 * view + p["rot"] * 0.3
        if angle:
            img = img.rotate(angle, fillcolor=(255, 255, 255), resample=Image.Resampling.BICUBIC)
        return img

    # --------------------------------------------------------- generation
    def _part_number(self, i: int) -> str:
        return f"{90000 + i:05d}A{self.rng.randint(100, 999)}"

    def _params(self) -> dict:
        return {
            "scale": self.rng.uniform(0.5, 0.8),
            "length_frac": self.rng.uniform(0.9, 2.2),
            "pitch": self.rng.uniform(3, 9),
            "inner": self.rng.uniform(0.15, 0.6),
            "teeth": self.rng.randint(6, 16),
            "holes": self.rng.randint(1, 3),
            "rot": self.rng.uniform(0, 60),
        }

    def _attributes(self, kind: str, material: str, p: dict) -> dict:
        a: dict = {"material": material}
        if (
            kind.endswith("screw")
            or kind in ("hex_bolt", "eye_bolt", "u_bolt", "threaded_rod")
            or kind.endswith("nut")
        ):
            a["thread_size"] = self.rng.choice(_THREADS)
        if kind.endswith("screw") or kind in (
            "hex_bolt",
            "dowel_pin",
            "clevis_pin",
            "threaded_rod",
            "compression_spring",
            "drill_bit",
            "key_stock",
            "rivet",
            "cotter_pin",
        ):
            a["length"] = self.rng.choice(_LENGTHS)
        if kind in (
            "flat_washer",
            "fender_washer",
            "lock_washer",
            "o_ring",
            "bearing",
            "shaft_collar",
            "pulley",
            "hose_clamp",
            "knob",
        ):
            a["od"] = self.rng.choice(_ODS)
        if kind in ("spur_gear", "sprocket"):
            a["teeth"] = p["teeth"]
        if kind in ("pipe_elbow", "pipe_tee"):
            a["pipe_size"] = self.rng.choice(['1/8"', '1/4"', '3/8"', '1/2"', '3/4"', '1"'])
        if kind in ("l_bracket", "flat_bracket", "hinge"):
            a["holes"] = p["holes"] * (2 if kind != "flat_bracket" else 1) + (
                1 if kind == "flat_bracket" else 0
            )
        return a

    def generate(self, out_dir: str | Path) -> Iterator[Part]:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        materials = list(_MATERIALS.items())
        for i in range(self.n_parts):
            kind = self.rng.choice(self.kinds)
            cat, name, _ = _FAMILIES[kind]
            if kind == "o_ring":
                material, color = "Buna-N Rubber", _MATERIALS["Buna-N Rubber"]
            else:
                material, color = self.rng.choice([m for m in materials if m[0] != "Buna-N Rubber"])
            p = self._params()
            pn = self._part_number(i)
            attrs = self._attributes(kind, material, p)
            images: list[str] = []
            for v in range(self.images_per_part):
                img = self._render(kind, color, p, v)
                path = out / f"{pn}_{v}.png"
                img.save(path)
                images.append(str(path.resolve()))
            yield Part(
                part_number=pn,
                name=f"{material} {name}",
                category_path=cat,
                description=f"{name} made of {material}.",
                attributes=attrs,
                image_paths=images,
                family_id=f"{kind}:{material}",
            )

    def write_jsonl(self, out_dir: str | Path, jsonl_path: str | Path) -> int:
        n = 0
        with open(jsonl_path, "w", encoding="utf-8") as fh:
            for part in self.generate(out_dir):
                fh.write(json.dumps(part.model_dump(), default=str) + "\n")
                n += 1
        return n


FAMILY_KINDS: tuple[str, ...] = tuple(_FAMILIES)
RENDERERS: dict[str, Callable] = {}  # reserved for user-registered extra kinds
