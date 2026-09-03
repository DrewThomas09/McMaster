"""Catalog-to-photo augmentation.

Catalog imagery is clean: white background, studio lighting, canonical pose. User
photos are not. The domain gap is the single biggest source of retrieval error, so
training (and the demo query generator) applies a "make it look like a phone photo"
augmentation chain implemented with Pillow + numpy only, so it runs everywhere.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, fields

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

_BACKGROUNDS = [
    (232, 226, 214),  # workbench
    (120, 120, 125),  # concrete
    (180, 150, 110),  # wood
    (40, 40, 45),  # dark mat
    (200, 205, 210),  # steel
    (245, 245, 245),  # paper
]


@dataclass
class AugmentConfig:
    rotate_deg: float = 180.0
    scale_range: tuple[float, float] = (0.55, 1.0)
    translate_frac: float = 0.15
    perspective: float = 0.12
    brightness: tuple[float, float] = (0.6, 1.4)
    contrast: tuple[float, float] = (0.7, 1.3)
    color_temp: float = 0.12
    blur_prob: float = 0.3
    blur_radius: tuple[float, float] = (0.3, 1.8)
    noise_prob: float = 0.5
    noise_sigma: float = 6.0
    jpeg_prob: float = 0.5
    jpeg_quality: tuple[int, int] = (35, 90)
    background_prob: float = 0.85
    shadow_prob: float = 0.5
    occlusion_prob: float = 0.2
    grayscale_prob: float = 0.05
    backgrounds: list[tuple[int, int, int]] = field(default_factory=lambda: list(_BACKGROUNDS))

    @classmethod
    def gallery(cls) -> AugmentConfig:
        """Mild variants used to *augment the index* (database-side augmentation):
        backgrounds and shadows but no rotation, so every backbone sees catalog
        parts the way phones see them without exploding the index size."""
        return cls(
            rotate_deg=0.0,
            scale_range=(0.8, 1.0),
            translate_frac=0.0,
            perspective=0.03,
            brightness=(0.85, 1.15),
            contrast=(0.9, 1.1),
            color_temp=0.04,
            blur_prob=0.2,
            blur_radius=(0.3, 0.8),
            noise_sigma=3.0,
            jpeg_prob=0.3,
            occlusion_prob=0.0,
            grayscale_prob=0.0,
        )

    @classmethod
    def interpolate(cls, mild: AugmentConfig, harsh: AugmentConfig, t: float) -> AugmentConfig:
        """Linear blend of two configs (t=0 -> mild, t=1 -> harsh) for curricula."""
        t = max(0.0, min(1.0, t))
        kw = {}
        for f in fields(cls):
            a, b = getattr(mild, f.name), getattr(harsh, f.name)
            if isinstance(a, bool) or f.name == "backgrounds":
                kw[f.name] = b if t >= 0.5 else a
            elif isinstance(a, tuple):
                kw[f.name] = tuple(type(x)(x + (y - x) * t) for x, y in zip(a, b, strict=True))
            elif isinstance(a, (int, float)):
                kw[f.name] = type(a)(a + (b - a) * t)
            else:
                kw[f.name] = b
        return cls(**kw)

    @classmethod
    def evaluation(cls) -> AugmentConfig:
        """A *typical* phone photo rather than the worst case: used by ``mcv evaluate``
        and the demo so reported recall reflects realistic queries. Training keeps the
        harsher defaults on purpose."""
        return cls(
            perspective=0.06,
            brightness=(0.75, 1.25),
            contrast=(0.8, 1.2),
            color_temp=0.06,
            blur_radius=(0.3, 1.2),
            noise_sigma=4.0,
            jpeg_quality=(55, 92),
            occlusion_prob=0.0,
            grayscale_prob=0.0,
        )


class PhotoAugmenter:
    def __init__(self, config: AugmentConfig | None = None, seed: int | None = None):
        self.cfg = config or AugmentConfig()
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

    def reseed(self, seed: int | None) -> None:
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

    # ------------------------------------------------------------ helpers
    def _white_to_alpha(self, img: Image.Image, threshold: int = 248) -> Image.Image:
        """Turn the white studio background transparent so we can composite it."""
        rgba = img.convert("RGBA")
        arr = np.asarray(rgba).astype(np.int16)
        white = (arr[..., :3] > threshold).all(axis=-1)
        arr[white, 3] = 0
        return Image.fromarray(arr.astype(np.uint8), "RGBA")

    def _background(self, size: tuple[int, int]) -> Image.Image:
        base = self.rng.choice(self.cfg.backgrounds)
        w, h = size
        # Gentle gradient + speckle texture.
        yy, xx = np.mgrid[0:h, 0:w]
        grad = (xx / max(w - 1, 1) * 0.5 + yy / max(h - 1, 1) * 0.5) - 0.5
        strength = self.rng.uniform(-40, 40)
        noise = self.np_rng.normal(0, 8, size=(h, w, 1))
        arr = np.array(base, dtype=np.float32)[None, None, :] + grad[..., None] * strength + noise
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")

    def _shadow(
        self, canvas: Image.Image, alpha_mask: Image.Image, offset: tuple[int, int]
    ) -> None:
        shadow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        dark = Image.new("RGBA", alpha_mask.size, (0, 0, 0, int(self.rng.uniform(70, 140))))
        shadow.paste(dark, offset, alpha_mask)
        shadow = shadow.filter(ImageFilter.GaussianBlur(self.rng.uniform(3, 9)))
        canvas.alpha_composite(shadow)

    def _perspective(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        p = self.cfg.perspective
        jit = lambda: self.rng.uniform(-p, p)  # noqa: E731
        src = [(0, 0), (w, 0), (w, h), (0, h)]
        dst = [
            (w * jit(), h * jit()),
            (w * (1 + jit()), h * jit()),
            (w * (1 + jit()), h * (1 + jit())),
            (w * jit(), h * (1 + jit())),
        ]
        coeffs = _find_coeffs(dst, src)
        return img.transform(
            img.size, Image.Transform.PERSPECTIVE, coeffs, Image.Resampling.BICUBIC
        )

    # --------------------------------------------------------------- main
    def __call__(self, img: Image.Image, out_size: int | None = None) -> Image.Image:
        cfg, rng = self.cfg, self.rng
        img = ImageOps.exif_transpose(img).convert("RGB")
        out_size = out_size or max(img.size)

        fg = self._white_to_alpha(img)
        scale = rng.uniform(*cfg.scale_range)
        angle = rng.uniform(-cfg.rotate_deg, cfg.rotate_deg)
        fg = fg.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)
        fw = max(8, int(out_size * scale * fg.width / max(fg.size)))
        fh = max(8, int(out_size * scale * fg.height / max(fg.size)))
        fg = fg.resize((fw, fh), Image.Resampling.LANCZOS)
        if cfg.perspective > 0:
            fg = self._perspective(fg)

        use_bg = rng.random() < cfg.background_prob
        canvas = (
            self._background((out_size, out_size)).convert("RGBA")
            if use_bg
            else Image.new("RGBA", (out_size, out_size), (255, 255, 255, 255))
        )
        tx = int(rng.uniform(-cfg.translate_frac, cfg.translate_frac) * out_size)
        ty = int(rng.uniform(-cfg.translate_frac, cfg.translate_frac) * out_size)
        off = ((out_size - fw) // 2 + tx, (out_size - fh) // 2 + ty)
        alpha = fg.split()[-1]
        if use_bg and rng.random() < cfg.shadow_prob:
            self._shadow(canvas, alpha, (off[0] + rng.randint(2, 12), off[1] + rng.randint(2, 12)))
        canvas.paste(fg, off, alpha)

        if rng.random() < cfg.occlusion_prob:
            ow, oh = (
                rng.randint(out_size // 8, out_size // 3),
                rng.randint(out_size // 8, out_size // 3),
            )
            ox, oy = rng.randint(0, out_size - ow), rng.randint(0, out_size - oh)
            patch = Image.new("RGBA", (ow, oh), (*rng.choice(cfg.backgrounds), 255))
            canvas.paste(patch, (ox, oy))

        out = canvas.convert("RGB")
        out = ImageEnhance.Brightness(out).enhance(rng.uniform(*cfg.brightness))
        out = ImageEnhance.Contrast(out).enhance(rng.uniform(*cfg.contrast))
        if cfg.color_temp > 0:
            arr = np.asarray(out).astype(np.float32)
            t = rng.uniform(-cfg.color_temp, cfg.color_temp)
            arr[..., 0] *= 1 + t
            arr[..., 2] *= 1 - t
            out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        if rng.random() < cfg.blur_prob:
            out = out.filter(ImageFilter.GaussianBlur(rng.uniform(*cfg.blur_radius)))
        if rng.random() < cfg.noise_prob:
            arr = np.asarray(out).astype(np.float32)
            arr += self.np_rng.normal(0, cfg.noise_sigma, arr.shape)
            out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        if rng.random() < cfg.grayscale_prob:
            out = ImageOps.grayscale(out).convert("RGB")
        if rng.random() < cfg.jpeg_prob:
            out = _jpeg_roundtrip(out, rng.randint(*cfg.jpeg_quality))
        return out


def _jpeg_roundtrip(img: Image.Image, quality: int) -> Image.Image:
    import io

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _find_coeffs(pa: list[tuple[float, float]], pb: list[tuple[float, float]]) -> list[float]:
    matrix = []
    for p1, p2 in zip(pa, pb, strict=True):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])
    a = np.array(matrix, dtype=np.float64)
    b = np.array(pb, dtype=np.float64).reshape(8)
    res = np.linalg.lstsq(a, b, rcond=None)[0]
    return [float(x) for x in res]
