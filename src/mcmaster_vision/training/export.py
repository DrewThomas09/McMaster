"""Export a fine-tuned backbone to ONNX / TorchScript for serving without Python
training dependencies (``pip install 'mcmaster-vision[export]'``).

Both exporters wrap the backbone so the graph maps a normalised image batch
``(B, 3, H, W)`` straight to L2-normalised embeddings ``(B, dim)``.
"""

from __future__ import annotations

from pathlib import Path


def _wrapper(backbone):
    import torch

    class EmbeddingWrapper(torch.nn.Module):
        def __init__(self, bb):
            super().__init__()
            self.bb = bb
            self.model = bb.model
            self.projection = bb.projection

        def forward(self, x):
            feats = self.bb._forward(x)
            if self.projection is not None:
                feats = self.projection(feats)
            return torch.nn.functional.normalize(feats.float(), dim=-1)

    return EmbeddingWrapper(backbone).eval().to(backbone.device)


def export_onnx(backbone, out_path: str | Path, image_size: int = 224, opset: int = 18) -> Path:
    import torch

    model = _wrapper(backbone)
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
        dynamic_shapes={"x": {0: torch.export.Dim("batch", min=1, max=4096)}},
    )
    return out


def export_torchscript(backbone, out_path: str | Path, image_size: int = 224) -> Path:
    import torch

    model = _wrapper(backbone)
    dummy = torch.randn(1, 3, image_size, image_size, device=backbone.device)
    with torch.inference_mode():
        traced = torch.jit.trace(model, dummy)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(out))
    return out
