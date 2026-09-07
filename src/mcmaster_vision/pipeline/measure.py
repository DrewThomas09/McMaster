"""Size from a photo with a known scale.

Look-alike SKUs in a family mostly differ by a dimension the camera cannot judge
without a reference: length, outside diameter, thread size. McMaster-Carr often
uses one image for a whole family, so no amount of visual retrieval can tell a
1" from a 1-1/4" screw. Given the scale of the photo (``mm_per_px``: the user
marks a coin, a card or a ruler in the app) the object's extent along its
principal axes is measured from the foreground mask and compared with each
candidate's catalog dimensions.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from PIL import Image

from mcmaster_vision.pipeline.pipe import pipe_id_mm, pipe_od_mm
from mcmaster_vision.pipeline.preprocess import foreground_mask
from mcmaster_vision.schemas import Part

INCH = 25.4
# ANSI numbered screw sizes -> major diameter (inches)
_GAUGE_IN = {
    0: 0.060,
    1: 0.073,
    2: 0.086,
    3: 0.099,
    4: 0.112,
    5: 0.125,
    6: 0.138,
    8: 0.164,
    10: 0.190,
    12: 0.216,
    14: 0.242,
}
_UNIT_MM = {
    "mm": 1.0,
    "millimeter": 1.0,
    "millimeters": 1.0,
    "cm": 10.0,
    "m": 1000.0,
    "in": INCH,
    "inch": INCH,
    "inches": INCH,
    '"': INCH,
    "″": INCH,
    "”": INCH,
    "ft": 12 * INCH,
}
_NUM = re.compile(
    r"(?<![\d.])(?P<whole>\d*\.\d+|\d+)?\s*(?:[- ]\s*)?(?P<frac>\d+/\d+)?\s*"
    r"(?P<unit>mm|millimeters?|cm|m\b|in\b|inch(?:es)?|\"|″|”|ft\b)",
    re.IGNORECASE,
)
_RANGE = re.compile(r"^\s*(?:-|–|—|to)\s*$", re.IGNORECASE)


def parse_length_mm(text) -> float | None:
    """``1/2"`` -> 12.7, ``1-1/4"`` -> 31.75, ``20 mm`` -> 20, ``M6x1`` -> 6 (diameter),
    ``#8-32`` -> 4.17 (diameter), ``1/4"-20`` -> 6.35. None when nothing parses."""
    t = str(text).strip().lower()
    if not t:
        return None
    m = re.match(r"^m(\d+(?:\.\d+)?)", t)
    if m:
        return float(m.group(1))
    m = re.match(r"^#(\d+)", t)
    if m:
        g = _GAUGE_IN.get(int(m.group(1)))
        return round(g * INCH, 3) if g else None
    matches = [m for m in _NUM.finditer(t) if m.group("whole") or m.group("frac")]
    if not matches:
        return None
    m = matches[0]
    if len(matches) > 1:
        m2 = matches[1]
        num_start = m2.start("whole") if m2.group("whole") else m2.start("frac")
        if _RANGE.match(t[m.end() : num_start]):
            return None  # a range ("3/8\" to 1/2\"", "12 mm - 15 mm") is not one dimension
    value = float(m.group("whole") or 0)
    if m.group("frac"):
        num, den = m.group("frac").split("/")
        if int(den) == 0:
            return None
        value += float(Fraction(int(num), int(den)))
    unit = m.group("unit").lower()
    return round(value * _UNIT_MM.get(unit, INCH), 3)


@dataclass
class Measurement:
    long_mm: float
    short_mm: float
    mm_per_px: float

    def as_dict(self) -> dict[str, float]:
        return {
            "long_mm": round(self.long_mm, 1),
            "short_mm": round(self.short_mm, 1),
            "mm_per_px": round(self.mm_per_px, 5),
        }


Segment = tuple[float, float, float, float]  # x1, y1, x2, y2 in image pixels


def _segment_mask(shape: tuple[int, int], seg: Segment, s: float, pad: int) -> np.ndarray:
    """Rasterise a segment (given in full-image pixels) into the working mask, dilated."""
    h, w = shape
    x1, y1, x2, y2 = (v * s for v in seg)
    n = int(max(abs(x2 - x1), abs(y2 - y1))) + 1
    xs = np.clip(np.round(np.linspace(x1, x2, n)).astype(int), 0, w - 1)
    ys = np.clip(np.round(np.linspace(y1, y2, n)).astype(int), 0, h - 1)
    m = np.zeros((h, w), dtype=bool)
    m[ys, xs] = True
    for _ in range(pad):
        d = m.copy()
        d[1:] |= m[:-1]
        d[:-1] |= m[1:]
        d[:, 1:] |= m[:, :-1]
        d[:, :-1] |= m[:, 1:]
        m = d
    return m


def object_extent_px(
    image: Image.Image, work: int = 160, exclude: Segment | None = None
) -> tuple[float, float] | None:
    """(long, short) extent of the foreground object in *image* pixels, measured along
    its principal axes so a tilted part is not measured by its bounding box. ``exclude``
    is the segment the user drew across the reference object (coin, card, ruler): the
    blob under it is the reference, not the part, and is removed first."""
    w, h = image.size
    s = min(1.0, work / max(w, h))
    small = image.convert("RGB").resize((max(1, round(w * s)), max(1, round(h * s))))
    excl = _segment_mask(small.size[::-1], exclude, s, pad=2) if exclude else None
    mask = foreground_mask(np.asarray(small, dtype=np.float32), exclude=excl)
    if mask is None or mask.sum() < 8:
        return None
    ys, xs = np.nonzero(mask)
    pts = np.stack([xs, ys], axis=1).astype(np.float64)
    pts -= pts.mean(axis=0)
    cov = pts.T @ pts / len(pts)
    _, vecs = np.linalg.eigh(cov)  # ascending: last column = major axis
    proj = pts @ vecs
    # the mask includes the anti-aliased edge (about half a pixel each side at the
    # working size), so the raw extent over-reads by roughly one pixel
    extents = np.maximum(proj.max(axis=0) - proj.min(axis=0) - 1.0, 1.0)
    long_px, short_px = float(extents[1]) / s, float(extents[0]) / s
    return max(long_px, short_px), min(long_px, short_px)


def measure(
    image: Image.Image, mm_per_px: float, reference: Segment | None = None
) -> Measurement | None:
    """``mm_per_px`` and ``reference`` are in *uploaded* pixels; a JPEG the server decoded
    at reduced size carries the factor in ``image.info["upload_scale"]``."""
    if mm_per_px <= 0:
        return None
    k = float(image.info.get("upload_scale", 1.0) or 1.0)
    seg = tuple(v / k for v in reference) if reference else None
    ext = object_extent_px(image, exclude=seg)  # type: ignore[arg-type]
    if ext is None:
        return None
    scale = mm_per_px * k  # mm per *decoded* pixel
    return Measurement(ext[0] * scale, ext[1] * scale, mm_per_px)


def _ratio_score(ratio: float, lo_ok: float, hi_ok: float, falloff: float = 1.6) -> float:
    """+1 inside [lo_ok, hi_ok]; outside, falls with the log-distance from the band,
    reaching -1 when ``falloff`` times beyond it (the next catalog size up or down)."""
    if ratio <= 0:
        return -1.0
    if lo_ok <= ratio <= hi_ok:
        return 1.0
    excess = math.log(lo_ok / ratio) if ratio < lo_ok else math.log(ratio / hi_ok)
    return max(-1.0, 1.0 - 2.0 * excess / math.log(falloff))


# catalog attribute -> tolerated ratio band. A screw's "length" excludes the head, so
# the measured long axis may exceed it. A diameter / width is the long axis of a round
# flat part (washer, ring) but the *short* axis of anything that also has a length
# (pin, standoff, shaft).
_LENGTH_KEYS = ("length", "overall_length")
_DIAMETER_KEYS = ("od", "outside_diameter", "diameter", "width")
_PIPE_KEYS = ("pipe_size", "pipe", "nominal_pipe_size")
# (keys, band low, band high, falloff): lengths step 1/2" -> 3/4" -> 1" (x1.5), diameters
# step 1/4" -> 5/16" -> 3/8" (x1.25), so a diameter one size off must already score -1
_RULES: list[tuple[tuple[str, ...], float, float, float]] = [
    (_LENGTH_KEYS, 0.85, 1.4, 1.6),
    (_DIAMETER_KEYS, 0.85, 1.15, 1.35),
]


def size_consistency(meas: Measurement | None, part: Part) -> tuple[float, list[str]]:
    """(score in [-1, 1], reasons): does the measured object fit this part's dimensions?"""
    if meas is None:
        return 0.0, []
    attrs = {
        k.lower().replace("-", "_").replace(" ", "_"): str(v) for k, v in part.attributes.items()
    }
    has_length = any(k in attrs and parse_length_mm(attrs[k]) for k in _LENGTH_KEYS)
    votes: list[float] = []
    reasons: list[str] = []
    # pipe size is nominal: a "3/8" fitting is 0.675" across male threads and 0.49" inside
    # female ones. Compare the measured short axis with whichever the part's gender implies
    # (either when unknown: a fitting body is at least the pipe OD wide).
    pipe = next((attrs[k] for k in _PIPE_KEYS if k in attrs), None)
    if pipe and (od := pipe_od_mm(pipe)):
        text = " ".join([part.name.lower(), *attrs.values()]).lower()
        female = "female" in text or "fpt" in text
        male = "male" in text or "mpt" in text or "nipple" in text
        targets = []
        if male or not female:
            targets.append(("pipe OD", od))
        if female or not male:
            pid = pipe_id_mm(pipe)
            if pid:
                targets.append(("pipe ID", pid))
        # a fitting body (hex, elbow) is wider than its thread but never narrower: allow up
        # to 1.5x the OD, and fall off fast below it (the next size down is 0.8x)
        scores = []
        for label, mm in targets:
            r = meas.short_mm / mm
            scores.append((_ratio_score(r, 0.9, 1.5, 1.25 if r < 0.9 else 1.4), label, mm))
        if scores:
            sc, label, mm = max(scores)
            votes.append(sc)
            verdict = "consistent" if sc > 0.5 else ("off" if sc < -0.5 else "close")
            reasons.append(
                f"measured {meas.short_mm:.0f} mm vs pipe size {pipe} ({label} {mm:.1f} mm): {verdict}"
            )
    for keys, lo, hi, falloff in _RULES:
        for key in keys:
            if key not in attrs:
                continue
            mm = parse_length_mm(attrs[key])
            if not mm:
                continue
            short_axis = keys is _DIAMETER_KEYS and has_length
            measured = meas.short_mm if short_axis else meas.long_mm
            sc = _ratio_score(measured / mm, lo, hi, falloff)
            votes.append(sc)
            verdict = "consistent" if sc > 0.5 else ("off" if sc < -0.5 else "close")
            reasons.append(
                f"measured {measured:.0f} mm vs {key} {attrs[key]} ({mm:.1f} mm): {verdict}"
            )
            break
    if not votes:
        return 0.0, []
    return float(np.clip(np.mean(votes), -1.0, 1.0)), reasons
