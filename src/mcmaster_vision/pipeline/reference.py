"""Find a likely reference coin in the photo so the scale can be offered in one tap.

A coin next to the part is a solid, near-circular blob. Among the foreground
components we look for one whose principal-axis extents are equal (within a
tolerance: a hex nut is 0.87 across flats vs corners, a coin tilted 25 degrees
is 0.9), whose area fills its bounding ellipse (a washer with a visible hole
does not), and that is neither a speck nor the whole
frame. The result is only a *suggestion*: the app shows "coin found, tap to use
as a US quarter" and the user confirms which coin it is. Everything else in
size matching (``measure.py``) then treats it exactly like a hand-drawn line.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image

from mcmaster_vision.pipeline.preprocess import _close, label_components

# Common reference objects, mm across
COINS_MM: dict[str, float] = {
    "US quarter": 24.26,
    "US nickel": 21.21,
    "US penny": 19.05,
    "US dime": 17.91,
    "1 euro": 23.25,
    "2 euro": 25.75,
    "1 pound": 23.43,
    "1 CAD": 26.5,
}


@dataclass
class CoinCandidate:
    cx: float  # centre, image pixels
    cy: float
    diameter_px: float
    circularity: float  # 1.0 = perfect disc

    def segment(self) -> tuple[float, float, float, float]:
        """A horizontal line across the coin (what the user would have drawn)."""
        r = self.diameter_px / 2
        return (self.cx - r, self.cy, self.cx + r, self.cy)

    def mm_per_px(self, coin_mm: float = COINS_MM["US quarter"]) -> float:
        return coin_mm / self.diameter_px


def _foreground(arr: np.ndarray) -> np.ndarray:
    """Plain background-difference mask without the largest-blob selection."""
    h, w, _ = arr.shape
    cw, ch = max(2, int(w * 0.12)), max(2, int(h * 0.12))
    border = np.zeros((h, w), dtype=bool)
    border[:ch, :cw] = border[:ch, -cw:] = border[-ch:, :cw] = border[-ch:, -cw:] = True
    border[0], border[-1], border[:, 0], border[:, -1] = True, True, True, True
    bg = np.median(arr[border], axis=0)
    res = np.linalg.norm(arr - bg, axis=-1)
    spread = float(np.std(res[border]))
    return _close(res > max(12.0, 4.0 * spread), iterations=2)


def find_coin(
    image: Image.Image,
    work: int = 200,
    *,
    min_frac: float = 0.03,
    max_frac: float = 0.6,
    min_circularity: float = 0.9,
) -> CoinCandidate | None:
    """The most coin-like blob, or None. ``min_frac``/``max_frac`` bound the coin's
    diameter as a fraction of the image's long side."""
    w, h = image.size
    s = min(1.0, work / max(w, h))
    small = image.convert("RGB").resize((max(1, round(w * s)), max(1, round(h * s))))
    arr = np.asarray(small, dtype=np.float32)
    mask = _foreground(arr)
    if mask.mean() < 0.001 or mask.mean() > 0.95:
        return None
    labels, sizes = label_components(mask)
    long_side = max(small.size)
    best: CoinCandidate | None = None
    best_area = 0
    best_score = -1.0
    for lab, area in sizes.items():
        if area < 30:
            continue
        ys, xs = np.nonzero(labels == lab)
        pts = np.stack([xs, ys], axis=1).astype(np.float64)
        centre = pts.mean(axis=0)
        pts -= centre
        cov = pts.T @ pts / len(pts)
        vals, vecs = np.linalg.eigh(cov)
        proj = pts @ vecs
        raw = proj.max(axis=0) - proj.min(axis=0) + 1.0  # includes the anti-aliased halo
        ext = np.maximum(raw - 2.0, 1.0)
        long_px, short_px = float(ext[1]), float(ext[0])
        if not (min_frac * long_side <= long_px <= max_frac * long_side):
            continue
        aspect = short_px / long_px
        # fill against the halo-inclusive disc: a solid coin is ~1.0, a washer's hole
        # or a nut's corners bring it down
        fill = area / (math.pi * (raw[1] / 2) * (raw[0] / 2))
        circ = min(aspect, min(fill, 1.0))
        if (
            aspect < 0.9 or fill < 0.92 or circ < min_circularity
        ):  # hex nut: aspect 0.87; washer: fill < 0.9
            continue
        # a coin is flat: its face is fairly even in brightness. A screw head with a
        # socket, a knob with a hub or a pulley with a bore is round too, but marked by
        # a dark centre or strong shading; prefer the flatter of two round blobs
        lum = arr[labels == lab].mean(axis=1)
        flatness = 1.0 - min(1.0, float(lum.std()) / 80.0)  # relief and highlights allowed
        cand = CoinCandidate(
            cx=float(centre[0]) / s,
            cy=float(centre[1]) / s,
            diameter_px=(long_px + short_px) / 2 / s,
            circularity=round(circ, 3),
        )
        score = 0.7 * circ + 0.3 * flatness
        if best is None or score > best_score:
            best, best_area, best_score = cand, area, score
    if best is not None:
        # a coin is a *reference next to* a part: with nothing else in the frame the round
        # blob is the part itself (a washer), and offering it as the coin would exclude it
        # anything else of substance counts: a 1/4" screw beside a quarter is 3% of it
        floor = max(30.0, 0.01 * best_area)
        others = [a for a in sizes.values() if a != best_area and a >= floor]
        if not others:
            return None
    if best is None and work < 400 and max(w, h) > work:
        # a small coin beside a long part (a six-inch nipple, a drill bit) is a dozen
        # pixels across at the coarse size and fails the roundness test: look again finer
        return find_coin(
            image,
            work=400,
            min_frac=min_frac / 2,
            max_frac=max_frac,
            min_circularity=min_circularity,
        )
    return best
