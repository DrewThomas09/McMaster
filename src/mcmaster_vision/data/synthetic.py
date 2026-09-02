"""Synthetic catalog generator.

Renders simple but visually distinct hardware (screws, nuts, washers, gears, pins,
brackets, ...) with Pillow so that the entire pipeline - ingest, embed, index,
identify, train, evaluate - can be exercised end to end without any proprietary
McMaster-Carr data. Part numbers follow the McMaster style ``NNNNNANNN``.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterator
from pathlib import Path

from PIL import Image, ImageDraw

from mcmaster_vision.schemas import Part

_MATERIALS = {
    "Zinc-Plated Steel": (175, 178, 185),
    "Black-Oxide Steel": (52, 52, 56),
    "18-8 Stainless Steel": (205, 208, 212),
    "Brass": (196, 160, 80),
    "Aluminum": (215, 218, 222),
    "Nylon": (240, 236, 220),
    "Copper": (184, 115, 51),
}

_FAMILIES: list[tuple[str, list[str], str]] = [
    (
        "socket_head_screw",
        ["Fastening & Joining", "Screws & Bolts", "Socket Head Screws"],
        "Socket Head Screw",
    ),
    ("hex_bolt", ["Fastening & Joining", "Screws & Bolts", "Hex Head Screws"], "Hex Head Screw"),
    ("hex_nut", ["Fastening & Joining", "Nuts", "Hex Nuts"], "Hex Nut"),
    ("flat_washer", ["Fastening & Joining", "Washers", "Flat Washers"], "Flat Washer"),
    ("lock_washer", ["Fastening & Joining", "Washers", "Split Lock Washers"], "Split Lock Washer"),
    ("spur_gear", ["Power Transmission", "Gears", "Spur Gears"], "Spur Gear"),
    ("dowel_pin", ["Fastening & Joining", "Pins", "Dowel Pins"], "Dowel Pin"),
    ("l_bracket", ["Hardware", "Brackets", "Corner Brackets"], "Corner Bracket"),
    ("o_ring", ["Sealing", "O-Rings", "Round O-Rings"], "O-Ring"),
    ("bearing", ["Power Transmission", "Bearings", "Ball Bearings"], "Ball Bearing"),
]

_THREADS = [
    "#4-40",
    "#6-32",
    "#8-32",
    "#10-24",
    '1/4"-20',
    '5/16"-18',
    '3/8"-16',
    "M3",
    "M4",
    "M5",
    "M6",
    "M8",
]
_LENGTHS = ['1/4"', '3/8"', '1/2"', '3/4"', '1"', '1-1/2"', '2"']


class SyntheticCatalog:
    def __init__(
        self, n_parts: int = 300, images_per_part: int = 3, size: int = 256, seed: int = 0
    ):
        self.n_parts = n_parts
        self.images_per_part = images_per_part
        self.size = size
        self.rng = random.Random(seed)

    # --------------------------------------------------------- rendering
    def _draw_part(
        self, kind: str, color: tuple[int, int, int], params: dict, view: int
    ) -> Image.Image:
        s = self.size
        img = Image.new("RGB", (s, s), (255, 255, 255))
        d = ImageDraw.Draw(img)
        dark = tuple(max(0, c - 60) for c in color)
        light = tuple(min(238, c + 35) for c in color)
        c = s // 2
        r = int(s * params["scale"])

        if kind in ("socket_head_screw", "hex_bolt"):
            head_h = int(r * 0.35)
            shaft_w = int(r * 0.32)
            length = int(r * params["length_frac"])
            top = c - (head_h + length) // 2
            d.rectangle(
                [c - r // 2, top, c + r // 2, top + head_h], fill=color, outline=dark, width=2
            )
            if kind == "socket_head_screw":
                d.rectangle(
                    [c - r // 2, top, c + r // 2, top + head_h], fill=color, outline=dark, width=2
                )
                d.ellipse([c - r // 5, top + 3, c + r // 5, top + head_h - 3], fill=dark)
            else:
                pts = [
                    (c - r // 2, top + head_h),
                    (c - r // 4, top),
                    (c + r // 4, top),
                    (c + r // 2, top + head_h),
                ]
                d.polygon(pts, fill=color, outline=dark)
            d.rectangle(
                [c - shaft_w // 2, top + head_h, c + shaft_w // 2, top + head_h + length],
                fill=light,
                outline=dark,
                width=2,
            )
            pitch = max(3, int(params["pitch"]))
            for y in range(top + head_h + pitch, top + head_h + length, pitch):
                d.line([c - shaft_w // 2, y, c + shaft_w // 2, y - pitch // 2], fill=dark, width=1)
        elif kind == "hex_nut":
            pts = _polygon(c, c, r // 2, 6, view * 7)
            d.polygon(pts, fill=color, outline=dark)
            d.ellipse(
                [c - r // 5, c - r // 5, c + r // 5, c + r // 5],
                fill=(255, 255, 255),
                outline=dark,
                width=2,
            )
        elif kind == "flat_washer":
            d.ellipse(
                [c - r // 2, c - r // 2, c + r // 2, c + r // 2], fill=color, outline=dark, width=2
            )
            ir = int(r * params["inner"])
            d.ellipse([c - ir, c - ir, c + ir, c + ir], fill=(255, 255, 255), outline=dark, width=2)
        elif kind == "lock_washer":
            d.ellipse(
                [c - r // 2, c - r // 2, c + r // 2, c + r // 2], fill=color, outline=dark, width=2
            )
            ir = int(r * params["inner"])
            d.ellipse([c - ir, c - ir, c + ir, c + ir], fill=(255, 255, 255), outline=dark, width=2)
            d.pieslice(
                [c - r // 2, c - r // 2, c + r // 2, c + r // 2],
                350 + view * 5,
                10 + view * 5,
                fill=(255, 255, 255),
            )
        elif kind == "spur_gear":
            teeth = params["teeth"]
            pts = []
            for i in range(teeth * 2):
                rad = r // 2 if i % 2 == 0 else int(r * 0.42)
                pts.append(_polar(c, c, rad, 360 * i / (teeth * 2) + view * 3))
            d.polygon(pts, fill=color, outline=dark)
            d.ellipse(
                [c - r // 6, c - r // 6, c + r // 6, c + r // 6],
                fill=(255, 255, 255),
                outline=dark,
                width=2,
            )
        elif kind == "dowel_pin":
            w = int(r * params["inner"] * 1.2)
            d.rounded_rectangle(
                [c - r // 2, c - w // 2, c + r // 2, c + w // 2],
                radius=w // 3,
                fill=light,
                outline=dark,
                width=2,
            )
            d.line(
                [c - r // 2 + 4, c - w // 4, c + r // 2 - 4, c - w // 4],
                fill=(255, 255, 255),
                width=2,
            )
        elif kind == "l_bracket":
            t = int(r * 0.18)
            d.rectangle(
                [c - r // 2, c - r // 2, c - r // 2 + t, c + r // 2],
                fill=color,
                outline=dark,
                width=2,
            )
            d.rectangle(
                [c - r // 2, c + r // 2 - t, c + r // 2, c + r // 2],
                fill=color,
                outline=dark,
                width=2,
            )
            for k in range(params["holes"]):
                y = c - r // 2 + t + (k + 1) * (r - 2 * t) // (params["holes"] + 1)
                d.ellipse(
                    [c - r // 2 + t // 4, y - t // 4, c - r // 2 + 3 * t // 4, y + t // 4],
                    fill=(255, 255, 255),
                    outline=dark,
                )
        elif kind == "o_ring":
            w = max(3, int(r * (1 - params["inner"]) * 0.5))
            d.ellipse(
                [c - r // 2, c - r // 2, c + r // 2, c + r // 2], outline=(30, 30, 30), width=w
            )
        elif kind == "bearing":
            d.ellipse(
                [c - r // 2, c - r // 2, c + r // 2, c + r // 2], fill=color, outline=dark, width=2
            )
            ir = int(r * 0.33)
            d.ellipse([c - ir, c - ir, c + ir, c + ir], fill=light, outline=dark, width=2)
            br = int(r * 0.2)
            d.ellipse([c - br, c - br, c + br, c + br], fill=(255, 255, 255), outline=dark, width=2)
            for i in range(params["teeth"]):
                bx, by = _polar(c, c, int(r * 0.41), 360 * i / params["teeth"] + view * 11)
                d.ellipse([bx - 4, by - 4, bx + 4, by + 4], fill=(240, 240, 240), outline=dark)
        return img.rotate(view * 15, fillcolor=(255, 255, 255), resample=Image.Resampling.BICUBIC)

    # --------------------------------------------------------- generation
    def _part_number(self, i: int) -> str:
        return f"{90000 + i:05d}A{self.rng.randint(100, 999)}"

    def generate(self, out_dir: str | Path) -> Iterator[Part]:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for i in range(self.n_parts):
            kind, cat, name = self.rng.choice(_FAMILIES)
            material, color = self.rng.choice(list(_MATERIALS.items()))
            params = {
                "scale": self.rng.uniform(0.45, 0.8),
                "length_frac": self.rng.uniform(0.8, 2.2),
                "pitch": self.rng.uniform(3, 9),
                "inner": self.rng.uniform(0.15, 0.3),
                "teeth": self.rng.randint(6, 14),
                "holes": self.rng.randint(1, 3),
            }
            pn = self._part_number(i)
            attrs = {"material": material}
            if kind in ("socket_head_screw", "hex_bolt", "hex_nut"):
                attrs["thread_size"] = self.rng.choice(_THREADS)
            if kind in ("socket_head_screw", "hex_bolt", "dowel_pin"):
                attrs["length"] = self.rng.choice(_LENGTHS)
            if kind == "spur_gear":
                attrs["teeth"] = params["teeth"]
            images: list[str] = []
            for v in range(self.images_per_part):
                img = self._draw_part(kind, color, params, v)
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


def _polar(cx: int, cy: int, r: int, deg: float) -> tuple[int, int]:
    import math

    return int(cx + r * math.cos(math.radians(deg))), int(cy + r * math.sin(math.radians(deg)))


def _polygon(cx: int, cy: int, r: int, n: int, rot: float = 0) -> list[tuple[int, int]]:
    return [_polar(cx, cy, r, rot + 360 * i / n) for i in range(n)]
