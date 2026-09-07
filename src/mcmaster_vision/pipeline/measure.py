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

from mcmaster_vision.pipeline.pipe import (
    THREADS_PER_INCH,
    normalise_pipe_size,
    pipe_id_mm,
    pipe_od_mm,
    schedule_from_text,
)
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
    pitch_mm: float | None = None  # thread pitch read from the photo, when there is one
    bore_mm: float | None = None  # the largest hole through the part (end-on fittings)

    def as_dict(self) -> dict[str, float]:
        out = {
            "long_mm": round(self.long_mm, 1),
            "short_mm": round(self.short_mm, 1),
            "mm_per_px": round(self.mm_per_px, 5),
        }
        if self.bore_mm:
            out["bore_mm"] = round(self.bore_mm, 1)
        if self.pitch_mm:
            out["pitch_mm"] = round(self.pitch_mm, 3)
            out["threads_per_inch"] = round(INCH / self.pitch_mm, 1)
        return out


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


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Background pixels not reachable from the image border are holes."""
    try:
        from scipy import ndimage

        return ndimage.binary_fill_holes(mask)
    except ImportError:  # iterative propagation from the border, vectorised
        bg = ~mask
        reach = np.zeros_like(mask, dtype=bool)
        reach[0, :] = bg[0, :]
        reach[-1, :] = bg[-1, :]
        reach[:, 0] = bg[:, 0]
        reach[:, -1] = bg[:, -1]
        for _ in range(max(mask.shape)):
            grown = reach.copy()
            grown[1:, :] |= reach[:-1, :]
            grown[:-1, :] |= reach[1:, :]
            grown[:, 1:] |= reach[:, :-1]
            grown[:, :-1] |= reach[:, 1:]
            grown &= bg
            if (grown == reach).all():
                break
            reach = grown
        return mask | (bg & ~reach)


def bore_px(image: Image.Image, work: int = 160, exclude: Segment | None = None) -> float | None:
    """Diameter (equivalent circle, image pixels) of the largest hole enclosed by the
    foreground object: the bore of a fitting, washer or bearing photographed end-on.
    None when the object has no hole worth the name (under 4% of its area)."""
    w, h = image.size
    s = min(1.0, work / max(w, h))
    small = image.convert("RGB").resize((max(1, round(w * s)), max(1, round(h * s))))
    excl = _segment_mask(small.size[::-1], exclude, s, pad=2) if exclude else None
    mask = foreground_mask(np.asarray(small, dtype=np.float32), exclude=excl)
    if mask is None or mask.sum() < 8:
        return None
    mask = mask.astype(bool)
    holes = _fill_holes(mask) & ~mask
    if holes.sum() < 0.04 * mask.sum():
        return None
    try:
        from scipy import ndimage

        labels, n = ndimage.label(holes)
        if n == 0:
            return None
        area = max(int((labels == i).sum()) for i in range(1, n + 1))
    except ImportError:
        area = int(holes.sum())  # one hole is the common case
    return 2.0 * float(np.sqrt(area / np.pi)) / s


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
    bore = bore_px(image, exclude=seg)  # type: ignore[arg-type]
    return Measurement(
        ext[0] * scale, ext[1] * scale, mm_per_px, bore_mm=bore * scale if bore else None
    )


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


_TPI = re.compile(r"(?:^|[\s\-x×])(\d{1,2}(?:\.\d)?)\s*(?:tpi|threads?\s*per\s*inch)?\s*$", re.I)
_METRIC_PITCH = re.compile(r"^m\d+(?:\.\d+)?\s*[x×]\s*(\d+(?:\.\d+)?)", re.I)
_INCH_THREAD = re.compile(r"^(?:#?\d+(?:-\d+/\d+)?|\d+/\d+)\s*(?:\"|″|”)?\s*-\s*(\d{1,2})\b")
_METRIC_DASH = re.compile(r"^m\d+(?:\.\d+)?\s*-\s*(\d+\.\d+)", re.I)  # M6-1.0


def catalog_pitch_mm(attrs: dict[str, str], name: str = "") -> tuple[float, str] | None:
    """The thread pitch a catalog entry implies: ``1/4"-20`` -> 1.27 mm, ``M6 x 1`` -> 1 mm,
    ``#8-32`` -> 0.79 mm, or a pipe size with NPT / BSP -> its threads per inch."""
    ts = attrs.get("thread_size") or attrs.get("thread") or ""
    m = _METRIC_PITCH.match(ts.strip()) or _METRIC_DASH.match(ts.strip())
    if m:
        return float(m.group(1)), f"thread {ts}"
    m = _INCH_THREAD.match(ts.strip())
    if m:
        return INCH / float(m.group(1)), f"thread {ts}"
    tpi = attrs.get("threads_per_inch") or attrs.get("tpi")
    if tpi:
        try:
            return INCH / float(str(tpi).split()[0]), f"{tpi} tpi"
        except ValueError:
            pass
    pitch = attrs.get("thread_pitch") or attrs.get("pitch")
    if pitch:
        p = parse_length_mm(pitch)
        if p:
            return p, f"pitch {pitch}"
        m = _TPI.search(pitch)
        if m:
            return INCH / float(m.group(1)), f"pitch {pitch}"
    text = (name + " " + " ".join(attrs.values())).upper()
    pipe = next((attrs[k] for k in _PIPE_KEYS if k in attrs), None)
    if not pipe and re.search(r"\b(NPT|NPTF|NPS[MHLC]?|BSP[TP]?|PIPE)\b", text):
        pipe = ts  # a bare "1/4" is a pipe thread only when the entry says so
    key = normalise_pipe_size(pipe) if pipe else None
    if key in THREADS_PER_INCH:
        npt, bsp = THREADS_PER_INCH[key]
        if "BSP" in text and bsp:
            return INCH / bsp, f"{key} BSP {bsp} tpi"
        if npt:
            return INCH / npt, f"{key} NPT {npt} tpi"
    return None


def pitch_consistency(meas: Measurement | None, part: Part) -> tuple[float, str | None]:
    """+1 when the measured pitch matches the catalog thread, -1 one thread standard away."""
    if meas is None or not meas.pitch_mm:
        return 0.0, None
    attrs = {
        k.lower().replace("-", "_").replace(" ", "_"): str(v) for k, v in part.attributes.items()
    }
    cat = catalog_pitch_mm(attrs, part.name)
    if cat is None:
        return 0.0, None
    expected, label = cat
    if " tpi" in label and ("NPT" in label or "BSP" in label):
        # NPT vs BSP differ by only 3.6-5.5% at a given size: a tight band, and a measured
        # pitch that lands between them stays neutral rather than endorsing both
        sc = _ratio_score(meas.pitch_mm / expected, 0.985, 1.015, 1.06)
    else:
        # coarse vs fine steps are ~1.3-1.4x (20 vs 28 tpi, M6x1 vs 0.75): -1 at that distance
        sc = _ratio_score(meas.pitch_mm / expected, 0.93, 1.07, 1.3)
    verdict = "consistent" if sc > 0.5 else ("off" if sc < -0.5 else "close")
    return sc, f"pitch {meas.pitch_mm:.2f} mm vs {label} ({expected:.2f} mm): {verdict}"


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
    # pipe size is nominal: a "3/8" fitting is 0.675" across its male threads. What a
    # photo shows is the silhouette, so the short axis is compared with the pipe OD: a
    # male thread is the OD itself (a hex body up to 1.5x), a female fitting's body wraps
    # the pipe and must be wider (1.1x to 1.6x). Female look-alikes one size apart are
    # genuinely ambiguous from a silhouette; two sizes apart are not.
    pipe = next((attrs[k] for k in _PIPE_KEYS if k in attrs), None)
    if pipe and (od := pipe_od_mm(pipe)):
        text = " ".join([part.name.lower(), *attrs.values()]).lower()
        female = "female" in text or "fpt" in text
        male = bool(re.search(r"(?<!fe)male\b", text)) or "mpt" in text or "nipple" in text
        if female and not male:
            lo, hi = 1.1, 1.6
        elif male and not female:
            lo, hi = 0.9, 1.5
        else:
            lo, hi = 0.9, 1.6
        r = meas.short_mm / od
        sc = _ratio_score(r, lo, hi, 1.25 if r < lo else 1.4)
        votes.append(sc)
        verdict = "consistent" if sc > 0.5 else ("off" if sc < -0.5 else "close")
        reasons.append(
            f"measured {meas.short_mm:.0f} mm vs pipe size {pipe} (pipe OD {od:.1f} mm, "
            f"{'female' if female and not male else 'male' if male and not female else 'fitting'} body): {verdict}"
        )
        # an unthreaded (butt-weld) fitting or plain pipe is the pipe itself: the catalog
        # says to measure its OD, and its wall / schedule fixes the bore. A measured bore
        # (the smaller of the two axes on an end-on photo) is compared with that ID
        wall_txt = attrs.get("wall_thickness") or attrs.get("wall")
        schedule = attrs.get("schedule") or schedule_from_text(text)
        try:
            wall_in = float(str(wall_txt).replace('"', "").strip()) if wall_txt else None
        except ValueError:
            wall_in = None
        if (wall_in or schedule) and meas.bore_mm:
            inner = pipe_id_mm(pipe, schedule, wall_in)
            if inner:
                rb = meas.bore_mm / inner
                sb = _ratio_score(rb, 0.85, 1.15, 1.3)
                votes.append(sb)
                reasons.append(
                    f"measured bore {meas.bore_mm:.0f} mm vs pipe ID {inner:.1f} mm "
                    f"({'wall ' + str(wall_txt) if wall_txt else 'schedule ' + str(schedule)}): "
                    f"{'consistent' if sb > 0.5 else 'off' if sb < -0.5 else 'close'}"
                )
    p_sc, p_reason = pitch_consistency(meas, part)
    if p_reason:
        votes.append(p_sc)
        reasons.append(p_reason)
    if not votes:
        return 0.0, []
    return float(np.clip(np.mean(votes), -1.0, 1.0)), reasons
