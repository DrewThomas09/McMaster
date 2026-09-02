"""Query-photo preprocessing.

1. Decode + EXIF orientation fix.
2. Optional background removal (rembg) - falls back to a saliency crop that finds
   the object against a roughly uniform background and crops to it.
3. Pad to square so aspect ratio is preserved for the backbone.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageOps


def decode_image(data: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(data))
    img.load()
    return ImageOps.exif_transpose(img).convert("RGB")


def foreground_mask(
    arr: np.ndarray, min_area: float = 0.01, max_area: float = 0.95
) -> np.ndarray | None:
    """Pixels that deviate from a *planar* background model.

    The plane (colour = a + b*x + c*y per channel) is fitted on the four corner
    patches plus a thin border, then re-fitted after discarding outliers, so an
    object that touches the image edge (a tightly cropped photo) does not pollute
    the background estimate. Fitting a plane instead of a constant makes lighting
    gradients part of the background. A morphological closing bridges thin gaps
    (thread lines, glints) and only the largest connected blob is kept. Returns
    None when nothing stands out. ``arr`` is an (H, W, 3) float array; keep it
    small (<= 128 px) for speed.
    """
    h, w, _ = arr.shape
    yy, xx = np.mgrid[0:h, 0:w]
    cw, ch = max(2, int(w * 0.12)), max(2, int(h * 0.12))
    sample = np.zeros((h, w), dtype=bool)
    sample[:ch, :cw] = sample[:ch, -cw:] = sample[-ch:, :cw] = sample[-ch:, -cw:] = True
    sample[0], sample[-1], sample[:, 0], sample[:, -1] = True, True, True, True

    def fit(sel: np.ndarray) -> np.ndarray:
        a = np.stack([np.ones(sel.sum()), xx[sel] / w, yy[sel] / h], axis=1)
        coef, *_ = np.linalg.lstsq(a, arr[sel], rcond=None)
        return coef[0] + coef[1] * (xx / w)[..., None] + coef[2] * (yy / h)[..., None]

    plane = fit(sample)
    res = np.linalg.norm(arr - plane, axis=-1)
    # robust re-fit: drop sample pixels that are clearly not background
    med = np.median(res[sample])
    mad = np.median(np.abs(res[sample] - med)) + 1e-3
    inlier = sample & (res <= med + 4.0 * mad)
    if inlier.sum() >= 12:
        plane = fit(inlier)
        res = np.linalg.norm(arr - plane, axis=-1)
        spread = float(np.std(res[inlier]))
    else:
        spread = float(np.std(res[sample]))
    mask = res > max(12.0, 4.0 * spread)
    mask = _close(mask, iterations=2)
    frac = mask.mean()
    if frac < min_area or frac > max_area:
        return None
    return largest_component(mask)


def _dilate(m: np.ndarray) -> np.ndarray:
    out = m.copy()
    out[1:] |= m[:-1]
    out[:-1] |= m[1:]
    out[:, 1:] |= m[:, :-1]
    out[:, :-1] |= m[:, 1:]
    return out


def _erode(m: np.ndarray) -> np.ndarray:
    out = m.copy()
    out[1:] &= m[:-1]
    out[:-1] &= m[1:]
    out[:, 1:] &= m[:, :-1]
    out[:, :-1] &= m[:, 1:]
    return out


def _close(m: np.ndarray, iterations: int = 1) -> np.ndarray:
    for _ in range(iterations):
        m = _dilate(m)
    for _ in range(iterations):
        m = _erode(m)
    return m


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep the largest 4-connected blob (pure numpy/python; masks are <= 128x128)."""
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    best_label, best_size, current = 0, 0, 0
    ys, xs = np.nonzero(mask)
    for sy, sx in zip(ys.tolist(), xs.tolist(), strict=True):
        if labels[sy, sx]:
            continue
        current += 1
        stack = [(sy, sx)]
        labels[sy, sx] = current
        size = 0
        while stack:
            y, x = stack.pop()
            size += 1
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not labels[ny, nx]:
                    labels[ny, nx] = current
                    stack.append((ny, nx))
        if size > best_size:
            best_label, best_size = current, size
    return labels == best_label


def saliency_crop(img: Image.Image, margin: float = 0.2) -> Image.Image:
    """Crop to the object that stands out from the background (a part on a bench, a
    table, or a sheet of paper). Returns the original if nothing stands out."""
    small = img.copy()
    small.thumbnail((128, 128))
    arr = np.asarray(small).astype(np.float32)
    mask = foreground_mask(arr, min_area=0.005, max_area=0.9)
    if mask is None:
        return img
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    sx, sy = img.width / w, img.height / h
    mw, mh = max(1.0, (xs.max() - xs.min()) * margin), max(1.0, (ys.max() - ys.min()) * margin)
    box = (
        max(0, int((xs.min() - mw) * sx)),
        max(0, int((ys.min() - mh) * sy)),
        min(img.width, int((xs.max() + 1 + mw) * sx)),
        min(img.height, int((ys.max() + 1 + mh) * sy)),
    )
    if box[2] - box[0] < 8 or box[3] - box[1] < 8:
        return img
    return img.crop(box)


def remove_background(img: Image.Image) -> Image.Image:
    """Use rembg when installed; otherwise return the input unchanged."""
    try:
        from rembg import remove  # type: ignore
    except ImportError:
        return img
    cut = remove(img).convert("RGBA")
    white = Image.new("RGBA", cut.size, (255, 255, 255, 255))
    white.alpha_composite(cut)
    return white.convert("RGB")


def border_color(img: Image.Image) -> tuple[int, int, int]:
    """Median colour of the outermost pixels: a cheap background estimate."""
    arr = np.asarray(img.convert("RGB"))
    border = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]])
    return tuple(int(v) for v in np.median(border, axis=0))


def pad_to_square(img: Image.Image, fill: tuple[int, int, int] | None = None) -> Image.Image:
    """Pad to a square canvas. The padding colour defaults to the image's own border
    colour so that a photo on a dark bench does not acquire a white frame, which
    would otherwise confuse background estimation downstream."""
    w, h = img.size
    if w == h:
        return img
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), fill or border_color(img))
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))
    return canvas


def preprocess(
    img: Image.Image, size: int = 224, crop: bool = True, segment: bool = False
) -> Image.Image:
    if segment:
        img = remove_background(img)
    if crop:
        img = saliency_crop(img)
    img = pad_to_square(img)
    return img.resize((size, size), Image.Resampling.LANCZOS)


def preprocess_catalog(img: Image.Image, size: int = 224) -> Image.Image:
    """Gallery-side normalisation. Must match :func:`preprocess` (minus segmentation) so
    that catalog vectors and query vectors live in the same distribution."""
    return preprocess(img.convert("RGB"), size=size, crop=True, segment=False)
