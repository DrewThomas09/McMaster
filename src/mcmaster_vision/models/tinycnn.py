"""TinyCNN: a compact residual network that can be trained from scratch on CPU.

Why it exists: the production backbones (CLIP / DINOv2) need pretrained weights
from the internet and a GPU to fine-tune comfortably. TinyCNN (~1.6M params,
96 px input) trains in minutes on a laptop, so the *whole* learning loop -
augmentation, hard-negative mining, metric losses, calibration - can be
developed, tested, and demonstrated offline. On synthetic catalogs it easily
beats the hand-crafted descriptor; on real photos it is a stepping stone.
"""

from __future__ import annotations

from mcmaster_vision.models.backbone import TorchBackbone

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

if nn is not None:

    class _Block(nn.Module):
        def __init__(self, cin: int, cout: int, stride: int):
            super().__init__()
            self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(cout)
            self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(cout)
            self.skip = (
                nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))
                if stride != 1 or cin != cout
                else nn.Identity()
            )

        def forward(self, x):
            y = F.relu(self.bn1(self.conv1(x)))
            y = self.bn2(self.conv2(y))
            return F.relu(y + self.skip(x))

    class GeM(nn.Module):
        """Generalised-mean pooling: standard for image retrieval."""

        def __init__(self, p: float = 3.0):
            super().__init__()
            self.p = nn.Parameter(torch.tensor(p))

        def forward(self, x):
            x = x.clamp(min=1e-6).pow(self.p)
            return F.adaptive_avg_pool2d(x, 1).pow(1.0 / self.p).flatten(1)

    class TinyCNN(nn.Module):
        def __init__(self, width: int = 32, out_dim: int = 256):
            super().__init__()
            w = width
            self.stem = nn.Sequential(
                nn.Conv2d(3, w, 3, 2, 1, bias=False), nn.BatchNorm2d(w), nn.ReLU(inplace=True)
            )
            self.layers = nn.Sequential(
                _Block(w, w, 1),
                _Block(w, 2 * w, 2),
                _Block(2 * w, 2 * w, 1),
                _Block(2 * w, 4 * w, 2),
                _Block(4 * w, 4 * w, 1),
                _Block(4 * w, 8 * w, 2),
                _Block(8 * w, 8 * w, 1),
            )
            self.pool = GeM()
            self.fc = nn.Linear(8 * w, out_dim)
            self.out_dim = out_dim

        def forward(self, x):
            x = self.layers(self.stem(x))
            return self.fc(self.pool(x))


class TinyCNNBackbone(TorchBackbone):
    name = "tinycnn"

    def __init__(self, width: int = 24, out_dim: int = 128, image_size: int = 96, **kw):
        super().__init__(**kw)
        from torchvision import transforms

        self.model = TinyCNN(width, out_dim).to(self.device).eval()
        self.image_size = image_size
        self.preprocess = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        self.dim = out_dim
        self.name = f"tinycnn:w{width}:d{out_dim}:{image_size}px"

    def _forward(self, batch):
        return self.model(batch)

    def trainable_module(self):
        return self.model
