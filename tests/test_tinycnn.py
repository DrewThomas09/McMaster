from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from mcmaster_vision.config import Settings  # noqa: E402
from mcmaster_vision.models.backbone import load_backbone  # noqa: E402
from mcmaster_vision.models.tinycnn import TinyCNNBackbone  # noqa: E402


def test_tinycnn_embeds_and_normalises():
    bb = TinyCNNBackbone(width=8, out_dim=16, image_size=32, device="cpu")
    v = bb.embed([Image.new("RGB", (50, 70), (200, 20, 20)), Image.new("RGB", (64, 64))])
    assert v.shape == (2, 16)
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5)


def test_train_smoke_and_checkpoint_roundtrip(store, tmp_path):
    from mcmaster_vision.training.train import load_train_config, train

    cfg = load_train_config("configs/train_tinycnn.yaml")
    cfg.update(
        {
            "epochs": 1,
            "batch_size": 4,
            "max_parts": 8,
            "num_workers": 0,
            "warmup_steps": 1,
            "hard_negative_mining": False,
            "val_frac": 0.25,
            "output_dir": str(tmp_path / "m"),
            "torch_threads": 2,
        }
    )
    ckpt = train(store, cfg)
    assert ckpt.exists() and (tmp_path / "m" / "history.json").exists()
    bb = load_backbone(Settings(backbone="tinycnn", backbone_checkpoint=ckpt, device="cpu"))
    assert bb.dim == cfg["embedding_dim"]
    assert "best" in bb.version
    v = bb.embed([Image.new("RGB", (96, 96), (90, 90, 90))])
    assert v.shape == (1, cfg["embedding_dim"])


def test_train_init_checkpoint_continues_from_saved_weights(store, tmp_path):
    import torch

    from mcmaster_vision.training.train import load_train_config, train

    cfg = load_train_config("configs/train_tinycnn.yaml")
    cfg.update(
        {
            "epochs": 1,
            "batch_size": 4,
            "max_parts": 8,
            "num_workers": 0,
            "warmup_steps": 1,
            "hard_negative_mining": False,
            "val_frac": 0.25,
            "output_dir": str(tmp_path / "a"),
            "torch_threads": 2,
        }
    )
    first = train(store, cfg)
    # a second run starts from the first's weights: after one small epoch its backbone is
    # still within a hair of what it loaded, where a fresh initialisation differs by tenths
    cfg.update(
        {
            "output_dir": str(tmp_path / "b"),
            "init_checkpoint": str(first),
            "freeze_backbone_epochs": 1,
        }
    )
    second = train(store, cfg)
    a = torch.load(first, map_location="cpu", weights_only=False)
    b = torch.load(tmp_path / "b" / "last.pt", map_location="cpu", weights_only=False)
    for k, v in a["backbone"].items():
        if v.dtype.is_floating_point and "running" not in k:
            assert (v - b["backbone"][k]).abs().max() < 0.05, k
    assert second.exists()
