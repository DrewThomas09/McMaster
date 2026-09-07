"""Thread pitch from a photo with a known scale.

Thread crests are a periodic light/dark pattern along the part's axis. Inside the
foreground mask we sample the intensity along the major axis (averaged across the
minor axis), remove the slow trend, and take the dominant period of the profile
(FFT). With ``mm_per_px`` that period is the pitch; threads per inch = 25.4 / pitch.
It reads a fastener's pitch (M6x1 vs M6x0.75, 1/4-20 vs 1/4-28) and, when the
measurement is within a couple of percent, helps separate NPT from BSP at a pipe size.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from mcmaster_vision.pipeline.measure import Segment, _segment_mask
from mcmaster_vision.pipeline.preprocess import foreground_mask

INCH = 25.4
MIN_SAMPLES = 32  # shorter profiles cannot hold enough crests to trust


@dataclass
class ThreadPitch:
    pitch_mm: float
    threads_per_inch: float
    period_px: float  # in the (possibly reduced) analysed image
    strength: float  # dominant peak vs the rest of the spectrum (>= 3 is a clear thread)

    def as_dict(self) -> dict[str, float]:
        return {
            "pitch_mm": round(self.pitch_mm, 3),
            "threads_per_inch": round(self.threads_per_inch, 1),
            "strength": round(self.strength, 2),
        }


def axis_profile(
    image: Image.Image, work: int = 512, exclude: Segment | None = None
) -> tuple[np.ndarray, float] | None:
    """Mean intensity along the object's major axis (one sample per pixel) and the
    scale factor from the analysed image back to ``image`` pixels. ``exclude`` (image
    pixels) marks the reference coin so its face is never mistaken for the part."""
    w, h = image.size
    s = min(1.0, work / max(w, h))
    small = image.convert("L").resize((max(1, round(w * s)), max(1, round(h * s))))
    gray = np.asarray(small, dtype=np.float32)
    # the mask is estimated at low resolution for speed, then scaled up
    ms = min(1.0, 160 / max(small.size))
    tiny = image.convert("RGB").resize(
        (max(1, round(small.size[0] * ms)), max(1, round(small.size[1] * ms)))
    )
    excl = _segment_mask(tiny.size[::-1], exclude, s * ms, pad=2) if exclude else None
    mask_small = foreground_mask(np.asarray(tiny, dtype=np.float32), exclude=excl)
    if mask_small is None or mask_small.sum() < 16:
        return None
    mask = (
        np.asarray(
            Image.fromarray(mask_small.astype(np.uint8) * 255).resize(
                small.size, Image.Resampling.NEAREST
            )
        )
        > 127
    )
    ys, xs = np.nonzero(mask)
    pts = np.stack([xs, ys], axis=1).astype(np.float64)
    centre = pts.mean(axis=0)
    pts -= centre
    _, vecs = np.linalg.eigh(pts.T @ pts / len(pts))
    major, minor = vecs[:, 1], vecs[:, 0]
    along = pts @ major
    across = pts @ minor
    # keep the central band across the minor axis (crests are cleanest there)
    band = np.abs(across) <= max(2.0, 0.35 * (across.max() - across.min()) / 2)
    along_b = along[band]
    vals = gray[ys[band], xs[band]]
    lo, hi = int(np.floor(along_b.min())), int(np.ceil(along_b.max()))
    n = hi - lo + 1
    if n < MIN_SAMPLES:
        return None
    sums = np.zeros(n)
    counts = np.zeros(n)
    idx = (np.round(along_b) - lo).astype(int)
    np.add.at(sums, idx, vals)
    np.add.at(counts, idx, 1)
    prof = sums / np.maximum(counts, 1)
    prof[counts == 0] = np.interp(
        np.nonzero(counts == 0)[0], np.nonzero(counts)[0], prof[counts > 0]
    )
    return prof, 1.0 / s


def dominant_period(
    profile: np.ndarray, min_period: float = 3.0, max_frac: float = 0.15
) -> tuple[float, float] | None:
    """(period in samples, peak strength) of the strongest periodic component after
    detrending, or None when the profile has no clear periodicity. A real thread shows
    at least ~7 crests (``max_frac``); strength is the peak against the *largest* other
    spectral feature, so a washer's two hole edges (a couple of broad low bins) do not
    pass as a thread the way they would against the median."""
    n = len(profile)
    if n < MIN_SAMPLES:
        return None
    # remove the slow trend (shading along the part) with a window far wider than any
    # thread period, so the fundamental of a coarse thread survives
    win = max(31, n // 5) | 1
    x = profile - np.convolve(profile, np.ones(win) / win, mode="same")
    x = x * np.hanning(n)
    spec = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(n)
    valid = (
        (freqs > 0)
        & (1.0 / np.maximum(freqs, 1e-9) >= min_period)
        & (1.0 / np.maximum(freqs, 1e-9) <= n * max_frac)
    )
    if not valid.any():
        return None
    spec_v = np.where(valid, spec, 0.0)
    k = int(np.argmax(spec_v))
    peak = spec_v[k]
    if peak <= 0:
        return None
    # a narrow crest has strong harmonics: if half the frequency (double the period) also
    # stands out, that is the fundamental
    half = k // 2
    med = np.median(spec[valid]) + 1e-9
    if half >= 1 and valid[half] and spec[half] >= 0.35 * peak and spec[half] > 3 * med:
        k, peak = half, spec[half]
    others = spec_v.copy()
    for centre in (k, 2 * k):
        others[max(0, centre - 2) : centre + 3] = 0.0
    rest = float(others.max()) + 1e-9
    # refine the peak with a parabolic fit over neighbours
    if 0 < k < len(spec) - 1:
        a, b, c = spec[k - 1], spec[k], spec[k + 1]
        denom = a - 2 * b + c
        shift = 0.5 * (a - c) / denom if abs(denom) > 1e-9 else 0.0
        f = freqs[k] + shift * (freqs[1] - freqs[0])
    else:
        f = freqs[k]
    return 1.0 / f, float(peak / rest)


def measure_thread_pitch(
    image: Image.Image,
    mm_per_px: float,
    reference: Segment | None = None,
    *,
    min_strength: float = 2.5,
) -> ThreadPitch | None:
    """``mm_per_px`` and ``reference`` are in *uploaded* pixels (``image.info['upload_scale']``
    corrects for a reduced-size decode, as in ``measure.py``)."""
    if mm_per_px <= 0:
        return None
    k = float(image.info.get("upload_scale", 1.0) or 1.0)
    seg = tuple(v / k for v in reference) if reference else None
    out = axis_profile(image, exclude=seg)  # type: ignore[arg-type]
    if out is None:
        return None
    prof, back = out
    dp = dominant_period(prof)
    if dp is None:
        return None
    period, strength = dp
    if strength < min_strength:
        return None
    pitch_mm = period * back * mm_per_px * k
    if not (0.2 <= pitch_mm <= 6.0):  # outside any thread standard: not a thread
        return None
    return ThreadPitch(pitch_mm, INCH / pitch_mm, period, strength)
