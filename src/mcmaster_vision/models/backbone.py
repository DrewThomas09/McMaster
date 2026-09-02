"""Image embedding backbones.

Every backbone exposes the same tiny interface so the pipeline, index builder, and
trainer are backbone-agnostic:

    backbone.embed(list_of_PIL_images) -> float32 array (N, dim), L2-normalised
    backbone.dim, backbone.name

``HashBackbone`` is a dependency-free numpy embedder used for CI and for wiring up
the system before GPUs are available. ``OpenCLIPBackbone`` / ``DINOv2Backbone`` are
the production choices and import torch lazily.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from mcmaster_vision.config import Settings


def l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)


class Backbone(ABC):
    name: str = "base"
    dim: int = 0

    @abstractmethod
    def embed(self, images: Sequence[Image.Image]) -> np.ndarray: ...

    def embed_one(self, image: Image.Image) -> np.ndarray:
        return self.embed([image])[0]

    @property
    def version(self) -> str:
        return self.name


# ---------------------------------------------------------------------------
# Dependency-free fallback
# ---------------------------------------------------------------------------
class HashBackbone(Backbone):
    """Hand-crafted global descriptor that needs only numpy + Pillow.

    1. Foreground mask from a plane-fit background model (shared with the
       preprocessing crop), largest blob only.
    2. Rotation-invariant core: the grayscale image and the silhouette are sampled
       on polar rings around the mask centroid (radius normalised by the mask
       extent), and the magnitude of the Fourier transform along the angle is kept
       for each ring. Magnitudes are invariant to rotation, and the centroid /
       radius normalisation gives translation and scale invariance.
    3. Orientation-normalised thumbnails (principal axis made horizontal) add
       detail for elongated parts; they are weighted by the shape's anisotropy so
       they do not add noise for round / hexagonal parts where the axis is undefined.
    4. Illumination-tolerant colour (chromaticity histogram), Hu moments, and a
       rotation-invariant gradient-orientation spectrum.

    The neural backbones below are the production path; this one exists so the
    demo, tests, and CI run without model weights or a GPU.
    """

    name = "hash"
    GROUP_ORDER = (
        "polar_gray",
        "polar_mask",
        "ring_gray",
        "ring_mask",
        "thumb_gray",
        "thumb_mask",
        "chroma_hist",
        "chroma_mean",
        "hu",
        "grad_spec",
    )
    # Tuned by coordinate ascent on synthetic photo-style queries (scripts/tune_hash_weights.py).
    DEFAULT_WEIGHTS = {
        "polar_gray": 1.5,
        "polar_mask": 1.5,
        "ring_gray": 0.25,
        "ring_mask": 1.5,
        "thumb_gray": 0.5,
        "thumb_mask": 2.5,
        "chroma_hist": 1.5,
        "chroma_mean": 0.5,
        "hu": 0.25,
        "grad_spec": 1.5,
    }

    def __init__(
        self,
        thumb: int = 12,
        rings: int = 12,
        angles: int = 36,
        harmonics: int = 8,
        chroma_bins: int = 4,
        orientations: int = 12,
        weights: dict[str, float] | None = None,
    ):
        self.thumb = thumb
        self.rings = rings
        self.angles = angles
        self.harmonics = harmonics
        self.chroma_bins = chroma_bins
        self.orientations = orientations
        self.weights = dict(weights or self.DEFAULT_WEIGHTS)
        self.dim = (
            rings * harmonics * 2  # polar spectra (gray + silhouette)
            + rings * 2  # ring means (gray + silhouette)
            + thumb * thumb * 2  # oriented thumbnails
            + chroma_bins * chroma_bins
            + 2  # colour
            + 7  # Hu moments
            + orientations // 2
            + 1  # gradient-orientation spectrum
        )

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _moments(mask: np.ndarray) -> tuple[float, float, float, float, float]:
        ys, xs = np.nonzero(mask)
        if len(xs) < 8:
            h, w = mask.shape
            return (w - 1) / 2, (h - 1) / 2, 0.0, 0.0, 1.0
        cx, cy = xs.mean(), ys.mean()
        x, y = xs - cx, ys - cy
        mu20, mu02, mu11 = (x * x).mean(), (y * y).mean(), (x * y).mean()
        angle = float(np.degrees(0.5 * np.arctan2(2 * mu11, mu20 - mu02)))
        aniso = float(np.hypot(mu20 - mu02, 2 * mu11) / (mu20 + mu02 + 1e-6))
        radius = float(np.percentile(np.hypot(x, y), 97)) * 1.05
        return float(cx), float(cy), angle, aniso, radius

    @staticmethod
    def _hu_moments(mask: np.ndarray) -> np.ndarray:
        m = mask.astype(np.float64)
        total = m.sum()
        if total < 1:
            return np.zeros(7, np.float32)
        ys, xs = np.mgrid[0 : m.shape[0], 0 : m.shape[1]]
        cx, cy = (xs * m).sum() / total, (ys * m).sum() / total
        x, y = xs - cx, ys - cy

        def mu(p, q):
            return ((x**p) * (y**q) * m).sum() / total ** (1 + (p + q) / 2)

        n20, n02, n11 = mu(2, 0), mu(0, 2), mu(1, 1)
        n30, n03, n21, n12 = mu(3, 0), mu(0, 3), mu(2, 1), mu(1, 2)
        hu = np.array(
            [
                n20 + n02,
                (n20 - n02) ** 2 + 4 * n11**2,
                (n30 - 3 * n12) ** 2 + (3 * n21 - n03) ** 2,
                (n30 + n12) ** 2 + (n21 + n03) ** 2,
                (n30 - 3 * n12) * (n30 + n12) * ((n30 + n12) ** 2 - 3 * (n21 + n03) ** 2)
                + (3 * n21 - n03) * (n21 + n03) * (3 * (n30 + n12) ** 2 - (n21 + n03) ** 2),
                (n20 - n02) * ((n30 + n12) ** 2 - (n21 + n03) ** 2)
                + 4 * n11 * (n30 + n12) * (n21 + n03),
                (3 * n21 - n03) * (n30 + n12) * ((n30 + n12) ** 2 - 3 * (n21 + n03) ** 2)
                - (n30 - 3 * n12) * (n21 + n03) * (3 * (n30 + n12) ** 2 - (n21 + n03) ** 2),
            ]
        )
        return (-np.sign(hu) * np.log10(np.abs(hu) + 1e-12) / 12.0).astype(np.float32)

    def _polar(self, values: np.ndarray, cx: float, cy: float, radius: float) -> np.ndarray:
        """Sample ``values`` on (rings x angles) polar grid with bilinear interpolation."""
        h, w = values.shape
        r = (np.arange(self.rings) + 0.5) / self.rings * radius
        t = np.arange(self.angles) / self.angles * 2 * np.pi
        xs = cx + r[:, None] * np.cos(t)[None, :]
        ys = cy + r[:, None] * np.sin(t)[None, :]
        x0 = np.clip(np.floor(xs).astype(int), 0, w - 2)
        y0 = np.clip(np.floor(ys).astype(int), 0, h - 2)
        fx = np.clip(xs - x0, 0, 1)
        fy = np.clip(ys - y0, 0, 1)
        v = (
            values[y0, x0] * (1 - fx) * (1 - fy)
            + values[y0, x0 + 1] * fx * (1 - fy)
            + values[y0 + 1, x0] * (1 - fx) * fy
            + values[y0 + 1, x0 + 1] * fx * fy
        )
        outside = (xs < 0) | (xs > w - 1) | (ys < 0) | (ys > h - 1)
        return np.where(outside, 0.0, v)

    def _ring_spectrum(self, polar: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        spec = np.abs(np.fft.rfft(polar, axis=1))[:, : self.harmonics]  # (rings, harmonics)
        means = polar.mean(axis=1)
        spec = spec / (np.abs(spec).sum() + 1e-6)
        return spec.ravel().astype(np.float32), means.astype(np.float32)

    def _oriented_thumb(
        self, img: Image.Image, mask: Image.Image, angle: float, bg: tuple
    ) -> tuple[np.ndarray, np.ndarray]:
        rot = img.rotate(angle, resample=Image.Resampling.BILINEAR, expand=True, fillcolor=bg)
        rmask = mask.rotate(angle, resample=Image.Resampling.NEAREST, expand=True, fillcolor=0)
        m = np.asarray(rmask) > 127
        if m.any():
            ys, xs = np.nonzero(m)
            box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
            rot, rmask = rot.crop(box), rmask.crop(box)
        w, h = rot.size
        side = max(w, h, 1)
        canvas = Image.new("RGB", (side, side), bg)
        canvas.paste(rot, ((side - w) // 2, (side - h) // 2))
        mcanvas = Image.new("L", (side, side), 0)
        mcanvas.paste(rmask, ((side - w) // 2, (side - h) // 2))
        g = (
            np.asarray(
                canvas.convert("L").resize((self.thumb, self.thumb), Image.Resampling.BOX),
                np.float32,
            )
            / 255.0
        )
        s = (
            np.asarray(mcanvas.resize((self.thumb, self.thumb), Image.Resampling.BOX), np.float32)
            / 255.0
        )
        g = (g - g.mean()) / (g.std() + 1e-6)
        return g.ravel(), (s - s.mean()).ravel()

    def _features(self, img: Image.Image) -> dict[str, np.ndarray]:
        from mcmaster_vision.pipeline.preprocess import foreground_mask

        img = ImageOps.exif_transpose(img).convert("RGB")
        img.thumbnail((96, 96))
        arr = np.asarray(img).astype(np.float32)
        mask = foreground_mask(arr)
        if mask is None:
            mask = np.ones(arr.shape[:2], dtype=bool)
        border = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]])
        bg = tuple(int(v) for v in np.median(border, axis=0))
        cx, cy, angle, aniso, radius = self._moments(mask)
        radius = max(radius, 4.0)

        arr01 = arr / 255.0
        gray = arr01.mean(axis=-1)
        fg_mean = gray[mask].mean()
        gray_fg = np.where(mask, gray, fg_mean)
        gray_fg = (gray_fg - fg_mean) / (gray[mask].std() + 1e-3)

        # rotation-invariant polar spectra
        spec_g, mean_g = self._ring_spectrum(self._polar(gray_fg, cx, cy, radius))
        spec_m, mean_m = self._ring_spectrum(self._polar(mask.astype(np.float32), cx, cy, radius))

        # oriented thumbnails, weighted by anisotropy
        thumb_g, thumb_m = self._oriented_thumb(
            img, Image.fromarray(mask.astype(np.uint8) * 255), angle, bg
        )
        w_thumb = 0.15 + 0.85 * min(1.0, aniso * 2.0)

        # colour: chromaticity histogram + mean chroma of the foreground
        px = arr01[mask]
        chroma = px / (px.sum(axis=1, keepdims=True) + 1e-6)
        hist = np.histogram2d(
            chroma[:, 0], chroma[:, 1], bins=self.chroma_bins, range=[[0.2, 0.5], [0.2, 0.5]]
        )[0]
        hist = hist.ravel().astype(np.float32) / (hist.sum() + 1e-6)
        mean_col = chroma[:, :2].mean(axis=0) - 1 / 3

        hu = self._hu_moments(mask)

        # rotation-invariant gradient-orientation spectrum
        gy, gx = np.gradient(gray)
        mag = np.hypot(gx, gy) * mask
        ang = (np.arctan2(gy, gx) + np.pi) % np.pi
        ohist = np.histogram(ang, bins=self.orientations, range=(0, np.pi), weights=mag)[0].astype(
            np.float32
        )
        ohist /= ohist.sum() + 1e-6
        ospec = np.abs(np.fft.rfft(ohist)).astype(np.float32)

        return {
            "polar_gray": spec_g,
            "polar_mask": spec_m,
            "ring_gray": mean_g,
            "ring_mask": mean_m,
            "thumb_gray": thumb_g * w_thumb,
            "thumb_mask": thumb_m * w_thumb,
            "chroma_hist": hist,
            "chroma_mean": mean_col,
            "hu": hu,
            "grad_spec": ospec,
        }

    def feature_groups(self, img: Image.Image) -> dict[str, np.ndarray]:
        """Unweighted feature groups (useful for diagnostics and weight tuning)."""
        return self._features(img)

    def _vector(self, groups: dict[str, np.ndarray]) -> np.ndarray:
        parts = []
        for name in self.GROUP_ORDER:
            g = groups[name]
            # each group is L2-normalised so that the weight is its total contribution
            parts.append(self.weights.get(name, 0.0) * g / (np.linalg.norm(g) + 1e-6))
        return np.concatenate(parts).astype(np.float32)

    def embed(self, images: Sequence[Image.Image]) -> np.ndarray:
        if not images:
            return np.zeros((0, self.dim), dtype=np.float32)
        return l2_normalize(np.stack([self._vector(self._features(im)) for im in images]))


# ---------------------------------------------------------------------------
# Neural backbones (torch imported lazily)
# ---------------------------------------------------------------------------
def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class TorchBackbone(Backbone):
    """Shared batching / device logic for torch-based backbones."""

    def __init__(self, device: str = "auto", batch_size: int = 64):
        try:
            import torch  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise ImportError("This backbone needs torch: pip install 'mcmaster-vision[ml]'") from e
        self.device = _resolve_device(device)
        self.batch_size = batch_size
        self.model = None
        self.preprocess = None
        self.projection = None  # optional fine-tuned head loaded from checkpoint

    def _forward(self, batch):  # -> torch.Tensor (B, dim)
        raise NotImplementedError

    def load_checkpoint(self, path: str | Path) -> None:
        """Load weights produced by ``mcv train`` (backbone + optional projection head)."""
        import torch

        state = torch.load(path, map_location=self.device)
        if "backbone" in state:
            self.model.load_state_dict(state["backbone"], strict=False)
        if state.get("projection") is not None:
            from mcmaster_vision.models.heads import ProjectionHead

            proj = ProjectionHead(state["projection_in"], state["projection_out"])
            proj.load_state_dict(state["projection"])
            self.projection = proj.to(self.device).eval()
            self.dim = state["projection_out"]
        self._checkpoint = str(path)

    @property
    def version(self) -> str:
        ck = getattr(self, "_checkpoint", None)
        return f"{self.name}@{Path(ck).stem}" if ck else self.name

    def embed(self, images: Sequence[Image.Image]) -> np.ndarray:
        import torch

        if not images:
            return np.zeros((0, self.dim), dtype=np.float32)
        outs: list[np.ndarray] = []
        with torch.inference_mode():
            for i in range(0, len(images), self.batch_size):
                chunk = images[i : i + self.batch_size]
                batch = torch.stack([self.preprocess(im.convert("RGB")) for im in chunk]).to(
                    self.device
                )
                feats = self._forward(batch)
                if self.projection is not None:
                    feats = self.projection(feats)
                feats = torch.nn.functional.normalize(feats.float(), dim=-1)
                outs.append(feats.cpu().numpy())
        return np.concatenate(outs).astype(np.float32)


class OpenCLIPBackbone(TorchBackbone):
    """CLIP / SigLIP image tower via ``open_clip``. Strong zero-shot starting point;
    fine-tune with ``mcv train`` for fine-grained part discrimination."""

    name = "openclip"

    def __init__(self, model_name: str = "ViT-B-16", pretrained: str = "laion2b_s34b_b88k", **kw):
        super().__init__(**kw)
        import open_clip

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.model.eval()
        self.model_name = model_name
        self.dim = int(self.model.visual.output_dim)
        self.name = f"openclip:{model_name}:{pretrained}"

    def _forward(self, batch):
        return self.model.encode_image(batch)

    def trainable_module(self):
        return self.model.visual


class DINOv2Backbone(TorchBackbone):
    """Self-supervised DINOv2 features via ``timm``; excellent for instance-level retrieval."""

    name = "dinov2"

    def __init__(self, model_name: str = "vit_base_patch14_dinov2.lvd142m", **kw):
        super().__init__(**kw)
        import timm

        self.model = (
            timm.create_model(model_name, pretrained=True, num_classes=0).to(self.device).eval()
        )
        cfg = timm.data.resolve_data_config({}, model=self.model)
        self.preprocess = timm.data.create_transform(**cfg)
        self.dim = int(self.model.num_features)
        self.name = f"dinov2:{model_name}"

    def _forward(self, batch):
        return self.model(batch)

    def trainable_module(self):
        return self.model


def load_backbone(settings: Settings) -> Backbone:
    """Instantiate the backbone named in settings (and load a fine-tuned checkpoint if set)."""
    if settings.backbone == "hash":
        return HashBackbone()
    if settings.backbone == "openclip":
        bb: TorchBackbone = OpenCLIPBackbone(
            settings.backbone_model, settings.backbone_pretrained, device=settings.device
        )
    elif settings.backbone == "dinov2":
        bb = DINOv2Backbone(settings.backbone_model, device=settings.device)
    else:  # pragma: no cover - validated by pydantic Literal
        raise ValueError(f"unknown backbone {settings.backbone}")
    if settings.backbone_checkpoint:
        bb.load_checkpoint(settings.backbone_checkpoint)
    return bb
