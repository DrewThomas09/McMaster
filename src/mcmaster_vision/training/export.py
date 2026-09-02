"""Export a fine-tuned backbone to ONNX / TorchScript for serving without Python
training dependencies."""

from __future__ import annotations

from pathlib import Path


def export_onnx(backbone, out_path: str | Path, image_size: int = 224, opset: int = 17) -> Path:
    import torch

    class Wrapper(torch.nn.Module):
        def __init__(self, bb):
            super().__init__()
            self.bb = bb

        def forward(self, x):
            feats = self.bb._forward(x)
            if self.bb.projection is not None:
                feats = self.bb.projection(feats)
            return torch.nn.functional.normalize(feats, dim=-1)

    model = Wrapper(backbone).eval().to(backbone.device)
    dummy = torch.randn(1, 3, image_size, image_size, device=backbone.device)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(out),
        opset_version=opset,
        input_names=["image"],
        output_names=["embedding"],
        dynamic_axes={"image": {0: "batch"}, "embedding": {0: "batch"}},
    )
    return out


def export_torchscript(backbone, out_path: str | Path, image_size: int = 224) -> Path:
    import torch

    dummy = torch.randn(1, 3, image_size, image_size, device=backbone.device)
    traced = torch.jit.trace(backbone.model, dummy)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(out))
    return out
