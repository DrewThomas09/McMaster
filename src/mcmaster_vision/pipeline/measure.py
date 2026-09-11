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
    female_thread_id_mm,
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
    long_px = max(1.0, float(proj[:, 1].max() - proj[:, 1].min()) - 1.0)
    # the short axis is the narrowest width over all directions (rotating calipers): a
    # nut's width across flats whatever its turn, where the covariance of a regular
    # polygon is isotropic and its minor axis would be arbitrary
    angles = np.deg2rad(np.arange(0.0, 180.0, 3.0))
    dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    widths = (pts @ dirs.T).max(axis=0) - (pts @ dirs.T).min(axis=0)
    short_px = max(1.0, float(widths.min()) - 1.0)
    long_px, short_px = long_px / s, short_px / s
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
    area = _largest_component(holes)
    if area == 0:
        return None
    return 2.0 * float(np.sqrt(area / np.pi)) / s


def _largest_component(mask: np.ndarray) -> int:
    """Pixel count of the largest 4-connected component (a flange's centre bore, not its
    bolt holes summed together)."""
    from mcmaster_vision.pipeline.preprocess import label_components

    _, sizes = label_components(mask.astype(bool))
    return max(sizes.values(), default=0)


def _grow(seed: np.ndarray, alike: np.ndarray, limit: np.ndarray | None) -> np.ndarray:
    """Flood ``seed`` through ``alike`` pixels (4-connected), inside ``limit`` if given."""
    grown = seed & alike
    if limit is not None:
        grown &= limit
    for _ in range(max(grown.shape)):
        d = grown.copy()
        d[1:, :] |= grown[:-1, :]
        d[:-1, :] |= grown[1:, :]
        d[:, 1:] |= grown[:, :-1]
        d[:, :-1] |= grown[:, 1:]
        d &= alike
        if limit is not None:
            d &= limit
        if (d == grown).all():
            break
        grown = d
    return grown


def erase_reference(image: Image.Image, reference: Segment, work: int = 160) -> Image.Image:
    """The photo with the reference object (the coin, card or ruler the user drew the
    segment across) painted over in the surrounding bench colour, so the embedding sees
    the part alone: the reference sets the scale, it must not vote on looks.

    A coin-sized segment erases the disc it spans (a coin the colour of the bench is
    invisible to the foreground mask, but still there for the embedding). A long segment
    (a card edge, a ruler) erases the foreground blobs it crosses plus a thin band along
    it, never a disc that would swallow the part beside it. Foreground blobs the segment
    does not touch are spared either way."""
    w, h = image.size
    k = float(image.info.get("upload_scale", 1.0) or 1.0)
    seg = tuple(v / k for v in reference)
    s = min(1.0, work / max(w, h))
    small = image.convert("RGB").resize((max(1, round(w * s)), max(1, round(h * s))))
    arr = np.asarray(small, dtype=np.float32)
    mask = foreground_mask(arr)
    mask = mask.astype(bool) if mask is not None else np.zeros(arr.shape[:2], dtype=bool)
    x1, y1, x2, y2 = (v * s for v in seg)
    half = 0.5 * float(np.hypot(x2 - x1, y2 - y1))
    if half < 1.5:
        return image
    hh, ww = mask.shape
    yy, xx = np.mgrid[0:hh, 0:ww]
    # the reference is what the segment lies on, grown through pixels of its own colour:
    # a coin, a card or a ruler is fairly uniform, and a part beside it is not the same
    # colour, so the growth stops at the part even when the foreground mask merges them
    probe = _segment_mask(mask.shape, seg, s, pad=1)
    ref_rgb = np.median(arr[probe], axis=0) if probe.any() else None
    if ref_rgb is None:
        return image
    dist = np.sqrt(((arr - ref_rgb) ** 2).sum(axis=-1))
    alike = dist <= 48.0
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    # coin or ruler is a question of shape, not of size (a coin fills half a close-up):
    # grown inside 1.5 radii of the segment's centre, a coin's colour fills the disc the
    # segment spans; a card edge or a ruler leaves most of that disc as bench
    limit = d2 <= (1.5 * half) ** 2
    grown = _grow(probe & alike, alike, limit)
    inner = d2 <= (0.9 * half) ** 2
    # a coin fills the disc on both sides of the segment; a card edge or a ruler fills
    # at most the one side the card is on
    side = (x2 - x1) * (yy - y1) - (y2 - y1) * (xx - x1)
    halves = [inner & (side > 0), inner & (side < 0)]
    fill = min(float((grown & hf).sum()) / max(1.0, float(hf.sum())) for hf in halves)
    protect = np.zeros_like(mask)
    if fill >= 0.5:  # a coin: the disc it spans, plus whatever of its colour it grew to
        disc = d2 <= (1.12 * half) ** 2
        if grown.sum() > 1.6 * disc.sum():
            # the bench is the coin's colour too: the growth says nothing, the disc is
            # the coin, and foreground beyond its rim (a part beside it) is kept
            grown = grown & disc
            protect = mask & ~alike & (d2 > (1.12 * half + 3) ** 2)
        comp = disc | grown
        core = comp
    else:  # a card or ruler: its own colour region, plus a thin band along the segment
        grown = _grow(probe & alike, alike, None)
        bench = ~mask
        if (grown & bench).sum() > 0.6 * max(1, int(bench.sum())):
            # the reference is the colour of the bench: its region is the bench itself,
            # so only the band the user drew is safe to erase
            grown = probe
        core = grown  # the band along the segment may graze the part: per pixel only
        comp = grown | _segment_mask(mask.shape, seg, s, pad=max(2, int(0.03 * max(hh, ww))))
    # foreground blobs the reference's own region reaches are the reference (its rim,
    # its shadow), erased whole; what only the band touches is erased pixel by pixel;
    # blobs neither reaches are the part, spared whole
    touched = _grow(core & mask, mask, None) | (comp & mask)
    for _ in range(3):  # a little beyond the edge, for the anti-aliased rim and shadow
        d = comp.copy()
        d[1:, :] |= comp[:-1, :]
        d[:-1, :] |= comp[1:, :]
        d[:, 1:] |= comp[:, :-1]
        d[:, :-1] |= comp[:, 1:]
        comp = d
    spared = mask & ~touched
    comp &= ~spared  # never paint over foreground the reference's colour did not reach
    comp &= ~protect
    if not comp.any():
        return image

    # the bench around the erased region: median colour and noise level of a ring just
    # outside it (never the part's pixels), sampled at working resolution
    # a ring 3-8 px outside the erased region: the 2 px nearest it still carry the
    # reference's anti-aliased rim and shadow, and would tint and speckle the fill (a
    # speckled disc on a smooth bench reads as foreground and spoils the crop)
    def _dilate(m: np.ndarray, n: int) -> np.ndarray:
        for _ in range(n):
            d = m.copy()
            d[1:, :] |= m[:-1, :]
            d[:-1, :] |= m[1:, :]
            d[:, 1:] |= m[:, :-1]
            d[:, :-1] |= m[:, 1:]
            m = d
        return m

    near = _dilate(comp, 2)
    ring = _dilate(near, 6) & ~near & ~mask
    if ring.sum() < 20:
        ring = ~comp & ~mask
    if ring.sum() < 20:
        return image
    samples = arr[ring]
    base = np.median(samples, axis=0)
    # a robust noise level: the median absolute deviation, so a few stray rim pixels
    # cannot make a smooth bench look grainy
    sigma = np.clip(1.4826 * np.median(np.abs(samples - base), axis=0), 0, 12)
    full = (
        np.asarray(
            Image.fromarray((comp * 255).astype(np.uint8)).resize((w, h), Image.Resampling.BILINEAR)
        )
        > 127
    )
    out = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    rng = np.random.default_rng(int(x1 * 7 + y1 * 13))
    out[full] = base + rng.normal(0, 1, (int(full.sum()), 3)) * sigma
    result = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
    result.info.update(image.info)
    result.info["reference_erased"] = True  # the crop may trust a small remaining blob
    return result


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
_THREAD_KEYS = ("thread_size", "thread", "size")
# width across flats of a standard hex nut by nominal thread diameter (ASME B18.2.2
# inch series, ISO 4032 metric), in mm; square nuts and nylon-insert locknuts match
NUT_WIDTH_MM: dict[float, float] = {
    **{
        d * INCH: w * INCH
        for d, w in (
            (0.112, 0.25),
            (0.125, 0.3125),
            (0.138, 0.3125),
            (0.164, 0.34375),
            (0.19, 0.375),
            (0.216, 0.4375),
            (0.25, 0.4375),
            (0.3125, 0.5),
            (0.375, 0.5625),
            (0.4375, 0.6875),
            (0.5, 0.75),
            (0.5625, 0.875),
            (0.625, 0.9375),
            (0.75, 1.125),
            (0.875, 1.3125),
            (1.0, 1.5),
        )
    },
    **{
        3.0: 5.5,
        4.0: 7.0,
        5.0: 8.0,
        6.0: 10.0,
        8.0: 13.0,
        10.0: 16.0,
        12.0: 18.0,
        16.0: 24.0,
        20.0: 30.0,
    },
}


def nut_width_mm(thread_size: str) -> float | None:
    """Across-flats width of a standard nut for a thread size (``#10-24``, ``5/16"-18``,
    ``M8 x 1.25``): the table's nearest nominal within 4%, else 1.6x the nominal."""
    d = parse_length_mm(thread_size)
    if not d:
        return None
    best = min(NUT_WIDTH_MM, key=lambda k: abs(k - d) / k)
    if abs(best - d) / best <= 0.04:
        return NUT_WIDTH_MM[best]
    return 1.6 * d


_NOT_A_NUT = re.compile(
    r"driver|wrench|flange|\bt-nut|speed|push|panel|peanut|\bnut plate|coconut|walnut|"
    r"donut|doughnut|chestnut|hazelnut",
    re.I,
)
_A_NUT = re.compile(r"\b(?:lock|jam|wing|cap|acorn|nyloc|castle|coupling|thumb)?nuts?\b", re.I)


def is_nut(part: Part) -> bool:
    """A nut whose silhouette is its width across flats: hex, square, jam, wing, lock
    and nylon-insert nuts; not a nut driver, a flange nut or a T-nut."""
    text = " ".join([part.name, *part.category_path]).lower()
    return bool(_A_NUT.search(text)) and not _NOT_A_NUT.search(text)


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
    # a nut's silhouette is its width across flats (the short axis face-on): a standard
    # width per thread size, one size step being 1.15-1.25x, so one size off scores -1
    if is_nut(part) and not has_length:
        ts = next((attrs[k] for k in _THREAD_KEYS if attrs.get(k)), None)
        width = nut_width_mm(ts) if ts else None
        if width:
            # face on, the narrowest width is across flats; on edge (long over short
            # above 1.5) the short axis is the thickness and the long one is across
            # flats to across corners, so the long axis over 1.08 stands in
            across = meas.short_mm if meas.long_mm < 1.5 * meas.short_mm else meas.long_mm / 1.08
            sc = _ratio_score(across / width, 0.85, 1.15, 1.3)
            votes.append(sc)
            verdict = "consistent" if sc > 0.5 else ("off" if sc < -0.5 else "close")
            reasons.append(
                f"measured {across:.0f} mm across vs a {ts} nut ({width:.1f} mm): {verdict}"
            )
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
        wall_mm = parse_length_mm(str(wall_txt)) if wall_txt else None
        wall_in = wall_mm / INCH if wall_mm else None
        # only where the hole *is* the pipe bore: plain pipe, unthreaded (butt-weld)
        # fittings, or a part whose spec gives the wall. A threaded female fitting's hole
        # is the thread's minor diameter, not OD minus two walls
        unthreaded = bool(re.search(r"butt.?weld|unthreaded|\bweld", text))
        if meas.bore_mm and female and not male and not unthreaded:
            # the catalog's female scale: measure the ID of the fitting, which is the
            # thread's minor diameter for that nominal size
            fid = female_thread_id_mm(pipe)
            if fid:
                rf = meas.bore_mm / fid
                sf = _ratio_score(rf, 0.85, 1.15, 1.3)
                votes.append(sf)
                reasons.append(
                    f"measured bore {meas.bore_mm:.0f} mm vs female thread ID {fid:.1f} mm: "
                    f"{'consistent' if sf > 0.5 else 'off' if sf < -0.5 else 'close'}"
                )
        elif meas.bore_mm and (wall_in or (schedule and (unthreaded or not (female and not male)))):
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
