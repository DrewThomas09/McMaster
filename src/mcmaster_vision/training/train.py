"""Fine-tune a backbone for part retrieval (requires torch: pip install .[ml]).

Recipe (see configs/train_openclip.yaml):
  * start from CLIP/SigLIP/DINOv2 weights
  * projection head -> 512-d embedding
  * supervised-contrastive loss with 2 augmented views per catalog image
    (optionally ArcFace over family labels as an auxiliary loss)
  * hard-negative mining refreshed every epoch
  * validation = Recall@1/5 on family-held-out SKUs with photo-style augmentation

Run: mcv train --config configs/train_openclip.yaml
"""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from mcmaster_vision.catalog.store import CatalogStore
from mcmaster_vision.data.augment import AugmentConfig, PhotoAugmenter
from mcmaster_vision.data.splits import split_by_family
from mcmaster_vision.schemas import Part
from mcmaster_vision.training.mining import hard_batch_sampler

log = logging.getLogger(__name__)

DEFAULTS: dict[str, Any] = {
    "backbone": "openclip",
    "backbone_model": "ViT-B-16",
    "backbone_pretrained": "laion2b_s34b_b88k",
    "image_size": 224,
    "embedding_dim": 512,
    "loss": "supcon",  # supcon | arcface (= supcon + ArcFace auxiliary head)
    "arcface_labels": "family",  # family | part : label space of the ArcFace head
    "arcface_margin": 0.3,
    "arcface_scale": 30.0,
    "arcface_weight": 1.0,
    "temperature": 0.07,
    "epochs": 10,
    "batch_size": 128,
    "lr": 1e-5,
    "head_lr": 1e-3,
    "weight_decay": 0.05,
    "warmup_steps": 500,
    "freeze_backbone_epochs": 1,
    "hard_negative_mining": True,
    "mining_refresh_every": 1,
    "num_workers": 4,
    "amp": True,
    "seed": 42,
    "val_frac": 0.1,
    "max_parts": None,
    "output_dir": "./data/models/finetuned",
    # Augmentation curriculum: blend from the mild evaluation preset to the full
    # training preset over the first ``curriculum_epochs`` epochs.
    "augment_curriculum": True,
    "curriculum_epochs": 4,
    "views": 2,
    "val_max_parts": 500,
    # Cached-view training (fast, CPU-friendly): pre-render ``cache_views`` augmented
    # views per catalog image every ``cache_refresh_epochs`` epochs and train on them.
    "cache_views": 0,
    "cache_refresh_epochs": 8,
    "cache_workers": 4,
    # Auxiliary part-classification (cross-entropy) loss weight - a strong early
    # training signal for from-scratch models. 0 disables.
    "ce_weight": 0.0,
    "label_smoothing": 0.1,
    "head_dropout": 0.0,
}


def load_train_config(path: str | Path | None) -> dict[str, Any]:
    cfg = dict(DEFAULTS)
    if path:
        with open(path, encoding="utf-8") as fh:
            cfg.update(yaml.safe_load(fh) or {})
    return cfg


def _mean_part_embeddings(backbone, parts: list[Part]) -> np.ndarray:
    from PIL import Image

    vecs = []
    for p in parts:
        imgs = [Image.open(x).convert("RGB") for x in p.image_paths[:2]]
        vecs.append(backbone.embed(imgs).mean(axis=0))
    v = np.stack(vecs)
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)


def with_extra_images(parts: list[Part], extra: dict[str, list[str]] | None) -> list[Part]:
    """Append labelled real photos (from the feedback store / ``--query-dir``) to the
    matching parts' image lists so they become training views."""
    if not extra:
        return parts
    out = []
    for p in parts:
        more = extra.get(p.part_number)
        out.append(p.model_copy(update={"image_paths": p.image_paths + more}) if more else p)
    return out


def train(
    store: CatalogStore, cfg: dict[str, Any], extra_images: dict[str, list[str]] | None = None
) -> Path:
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as e:  # pragma: no cover
        raise ImportError("Training needs torch: pip install 'mcmaster-vision[ml]'") from e

    from mcmaster_vision.config import Settings
    from mcmaster_vision.data.dataset import make_contrastive_dataset, worker_init_fn
    from mcmaster_vision.models.backbone import load_backbone
    from mcmaster_vision.models.heads import ArcFaceHead, ProjectionHead
    from mcmaster_vision.models.losses import arcface_loss, supcon_loss
    from mcmaster_vision.training.mining import mine_hard_negatives

    torch.manual_seed(cfg["seed"])
    if cfg.get("torch_threads"):
        torch.set_num_threads(int(cfg["torch_threads"]))
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    parts = list(store.iter_parts(with_images_only=True))
    if cfg.get("max_parts"):
        parts = parts[: int(cfg["max_parts"])]
    parts = with_extra_images(parts, extra_images)
    train_parts, val_parts = split_by_family(parts, cfg["val_frac"])
    log.info("train parts=%d val parts=%d", len(train_parts), len(val_parts))

    settings = Settings(
        backbone=cfg["backbone"],
        backbone_model=cfg["backbone_model"],
        backbone_pretrained=cfg["backbone_pretrained"],
        image_size=cfg["image_size"],
    )
    backbone = load_backbone(settings)
    device = backbone.device
    net = backbone.trainable_module()
    head = ProjectionHead(
        backbone.dim, cfg["embedding_dim"], dropout=float(cfg.get("head_dropout", 0.0))
    ).to(device)
    arc = None
    if cfg.get("arcface_labels", "family") == "part":
        fam_ids = sorted(p.part_number for p in train_parts)
    else:
        fam_ids = sorted({p.family_id or p.part_number for p in train_parts})
    fam_index = {f: i for i, f in enumerate(fam_ids)}

    def arc_label(part: Part) -> int:
        key = (
            part.part_number
            if cfg.get("arcface_labels", "family") == "part"
            else (part.family_id or part.part_number)
        )
        return fam_index[key]

    if cfg["loss"] == "arcface":
        arc = ArcFaceHead(
            cfg["embedding_dim"], len(fam_ids), cfg["arcface_scale"], cfg["arcface_margin"]
        ).to(device)

    if int(cfg.get("cache_views", 0)) > 0:
        return _train_cached(
            cfg, out_dir, backbone, net, head, arc, train_parts, val_parts, fam_index, arc_label
        )

    augmenter = PhotoAugmenter(seed=cfg["seed"])
    dataset, label_map = make_contrastive_dataset(
        train_parts,
        backbone.preprocess,
        augmenter,
        views=int(cfg["views"]),
        image_size=cfg["image_size"],
    )
    eval_augmenter = PhotoAugmenter(AugmentConfig.evaluation(), seed=cfg["seed"] + 1)
    pn_by_label = {v: k for k, v in label_map.items()}
    part_by_pn = {p.part_number: p for p in train_parts}

    params = [
        {"params": head.parameters(), "lr": cfg["head_lr"]},
        {"params": net.parameters(), "lr": cfg["lr"]},
    ]
    if arc is not None:
        params.append({"params": arc.parameters(), "lr": cfg["head_lr"]})
    opt = torch.optim.AdamW(params, weight_decay=cfg["weight_decay"])
    steps_per_epoch = max(1, len(dataset) // cfg["batch_size"])
    total_steps = steps_per_epoch * cfg["epochs"]
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: (
            min(1.0, (s + 1) / cfg["warmup_steps"])
            * 0.5
            * (1 + math.cos(math.pi * min(1.0, s / max(1, total_steps))))
        ),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["amp"] and device == "cuda")

    hard: dict[str, list[str]] = {}
    step = 0
    best_val = -1.0
    history: list[dict[str, Any]] = []

    for epoch in range(cfg["epochs"]):
        freeze = epoch < cfg["freeze_backbone_epochs"]
        augmenter.reseed(cfg["seed"] + 1000 * (epoch + 1))
        if cfg["augment_curriculum"]:
            t_cur = min(1.0, epoch / max(1, int(cfg["curriculum_epochs"])))
            augmenter.cfg = AugmentConfig.interpolate(
                AugmentConfig.evaluation(), AugmentConfig(), t_cur
            )
        for p in net.parameters():
            p.requires_grad_(not freeze)

        if cfg["hard_negative_mining"] and epoch % cfg["mining_refresh_every"] == 0 and epoch > 0:
            backbone.projection = head.eval()
            emb = _mean_part_embeddings(backbone, train_parts)
            hard = mine_hard_negatives(train_parts, emb)
            backbone.projection = None

        if hard:
            batches = _hard_batches(
                train_parts,
                hard,
                min(int(cfg["batch_size"]), len(train_parts)),
                steps_per_epoch,
                cfg["seed"] + epoch,
            )
            loader = DataLoader(
                dataset,
                batch_sampler=batches,
                num_workers=cfg["num_workers"],
                worker_init_fn=worker_init_fn,
            )
        else:
            loader = DataLoader(
                dataset,
                batch_size=cfg["batch_size"],
                shuffle=True,
                drop_last=True,
                num_workers=cfg["num_workers"],
                worker_init_fn=worker_init_fn,
            )

        net.train(not freeze or cfg["freeze_backbone_epochs"] == 0)
        head.train()
        t0 = time.time()
        running = 0.0
        for views, labels in loader:
            views, labels = views.to(device), labels.to(device)
            b, v = views.shape[:2]
            with torch.autocast(
                device_type="cuda" if device == "cuda" else "cpu", enabled=scaler.is_enabled()
            ):
                feats = backbone._forward(views.flatten(0, 1))
                emb = head(feats).reshape(b, v, -1)
                loss = supcon_loss(emb, labels, cfg["temperature"])
                if arc is not None:
                    fam_labels = torch.tensor(
                        [arc_label(part_by_pn[pn_by_label[int(lbl)]]) for lbl in labels],
                        device=device,
                    ).repeat_interleave(v)
                    loss = loss + float(cfg.get("arcface_weight", 1.0)) * arcface_loss(
                        arc(emb.flatten(0, 1), fam_labels), fam_labels
                    )
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(list(net.parameters()) + list(head.parameters()), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += float(loss.detach())
            step += 1

        # validation: Recall@1 with augmented val queries against the val gallery
        backbone.projection = head
        val_r1 = (
            _validate(backbone, val_parts, eval_augmenter, int(cfg["val_max_parts"]))
            if val_parts
            else float("nan")
        )
        backbone.projection = None
        rec = {
            "epoch": epoch,
            "loss": running / max(1, steps_per_epoch),
            "val_recall@1": val_r1,
            "seconds": round(time.time() - t0, 1),
        }
        history.append(rec)
        log.info("%s", rec)

        ckpt = {
            "backbone": net.state_dict(),
            "projection": head.state_dict(),
            "projection_in": backbone.dim,
            "projection_out": cfg["embedding_dim"],
            "config": cfg,
            "epoch": epoch,
            "val_recall@1": val_r1,
        }
        torch.save(ckpt, out_dir / "last.pt")
        if val_r1 != val_r1 or val_r1 > best_val:  # nan-safe
            best_val = val_r1 if val_r1 == val_r1 else best_val
            torch.save(ckpt, out_dir / "best.pt")
        (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    return out_dir / "best.pt"


def _train_cached(
    cfg, out_dir, backbone, net, head, arc, train_parts, val_parts, fam_index, arc_label
) -> Path:
    """Fast path: train on pre-augmented cached views (see training/cached.py)."""
    import torch
    import torch.nn.functional as F

    from mcmaster_vision.models.losses import arcface_loss, supcon_loss
    from mcmaster_vision.training.cached import build_view_cache, to_tensor
    from mcmaster_vision.training.mining import hard_batch_sampler, mine_hard_negatives

    device = backbone.device
    k = int(cfg["cache_views"])
    epochs = int(cfg["epochs"])
    n_parts = len(train_parts)
    bs = max(2, min(int(cfg["batch_size"]), n_parts))  # a batch cannot hold more parts than exist
    clf = (
        torch.nn.Linear(cfg["embedding_dim"], n_parts).to(device)
        if float(cfg.get("ce_weight", 0)) > 0
        else None
    )
    params = [
        {"params": head.parameters(), "lr": cfg["head_lr"]},
        {"params": net.parameters(), "lr": cfg["lr"]},
    ]
    if arc is not None:
        params.append({"params": arc.parameters(), "lr": cfg["head_lr"]})
    if clf is not None:
        params.append({"params": clf.parameters(), "lr": cfg["head_lr"]})
    opt = torch.optim.AdamW(params, weight_decay=cfg["weight_decay"])
    n_images = sum(len(p.image_paths) for p in train_parts)
    steps_per_epoch = max(1, (n_images * k) // (bs * 2))
    total_steps = steps_per_epoch * epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda st: (
            min(1.0, (st + 1) / cfg["warmup_steps"])
            * 0.5
            * (1 + math.cos(math.pi * min(1.0, st / max(1, total_steps))))
        ),
    )
    eval_augmenter = PhotoAugmenter(AugmentConfig.evaluation(), seed=cfg["seed"] + 1)
    rng = np.random.default_rng(cfg["seed"])
    part_labels = torch.arange(n_parts)
    arc_labels_all = torch.tensor([arc_label(p) for p in train_parts])

    x_u8: np.ndarray | None = None
    by_part: list[np.ndarray] = []
    hard: dict[str, list[str]] = {}
    history: list[dict[str, Any]] = []
    best_val = -1.0
    step = 0
    for epoch in range(epochs):
        if x_u8 is None or epoch % int(cfg["cache_refresh_epochs"]) == 0:
            t_cache = time.time()
            mild = cfg["augment_curriculum"] and epoch == 0 and epochs > 1
            aug_cfg = (
                AugmentConfig.interpolate(AugmentConfig.evaluation(), AugmentConfig(), 0.5)
                if mild
                else AugmentConfig()
            )
            x_u8, owners = build_view_cache(
                train_parts,
                k,
                cfg["image_size"],
                cfg=aug_cfg,
                seed=cfg["seed"] + 7919 * epoch,
                workers=int(cfg["cache_workers"]),
            )
            # keep the cache as uint8; batches are converted to float on the fly
            # (a float32 copy of a 70k-view cache would be ~7.5 GB)
            by_part = [np.nonzero(owners == i)[0] for i in range(n_parts)]
            log.info(
                "view cache: %d views (%.0fs, %s)",
                len(x_u8),
                time.time() - t_cache,
                "mild" if mild else "full",
            )
        if cfg["hard_negative_mining"] and epoch > 0 and epoch % cfg["mining_refresh_every"] == 0:
            backbone.projection = head
            hard = mine_hard_negatives(train_parts, _mean_part_embeddings(backbone, train_parts))
            backbone.projection = None
        sampler = (
            hard_batch_sampler(train_parts, hard, bs, seed=cfg["seed"] + epoch) if hard else None
        )

        net.train()
        head.train()
        t0 = time.time()
        running = 0.0
        for _ in range(steps_per_epoch):
            sel = (
                np.array(next(sampler))
                if sampler is not None
                else rng.choice(n_parts, bs, replace=False)
            )
            idx = np.array(
                [rng.choice(by_part[p], 2, replace=len(by_part[p]) < 2) for p in sel]
            ).reshape(-1)
            x = to_tensor(x_u8[np.sort(idx)][np.argsort(np.argsort(idx))]).to(device)
            labels = part_labels[sel].to(device)
            feats = backbone._forward(x)
            emb = head(feats).reshape(bs, 2, -1)
            loss = supcon_loss(emb, labels, cfg["temperature"])
            if clf is not None:
                loss = loss + float(cfg["ce_weight"]) * F.cross_entropy(
                    clf(emb.flatten(0, 1)),
                    labels.repeat_interleave(2),
                    label_smoothing=float(cfg["label_smoothing"]),
                )
            if arc is not None:
                al = arc_labels_all[sel].to(device).repeat_interleave(2)
                loss = loss + float(cfg.get("arcface_weight", 1.0)) * arcface_loss(
                    arc(emb.flatten(0, 1), al), al
                )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(net.parameters()) + list(head.parameters()), 5.0)
            opt.step()
            sched.step()
            running += float(loss.detach())
            step += 1

        backbone.projection = head
        val_r1 = (
            _validate(backbone, val_parts, eval_augmenter, int(cfg["val_max_parts"]))
            if val_parts
            else float("nan")
        )
        backbone.projection = None
        rec = {
            "epoch": epoch,
            "loss": running / steps_per_epoch,
            "val_recall@1": val_r1,
            "seconds": round(time.time() - t0, 1),
            "steps": step,
        }
        history.append(rec)
        log.info("%s", rec)
        ckpt = {
            "backbone": net.state_dict(),
            "projection": head.state_dict(),
            "projection_in": backbone.dim,
            "projection_out": cfg["embedding_dim"],
            "config": cfg,
            "epoch": epoch,
            "val_recall@1": val_r1,
        }
        torch.save(ckpt, out_dir / "last.pt")
        if val_r1 != val_r1 or val_r1 >= best_val:
            best_val = val_r1 if val_r1 == val_r1 else best_val
            torch.save(ckpt, out_dir / "best.pt")
        (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return out_dir / "best.pt"


def _hard_batches(
    train_parts: list[Part], hard: dict[str, list[str]], batch_size: int, steps: int, seed: int
) -> list[list[int]]:
    """Dataset items are per-image; translate hard-negative *part* batches into image-index batches."""
    img_idx_by_part: dict[str, list[int]] = {}
    i = 0
    for part in train_parts:
        for _ in part.image_paths:
            img_idx_by_part.setdefault(part.part_number, []).append(i)
            i += 1
    rng = np.random.default_rng(seed)
    sampler = hard_batch_sampler(train_parts, hard, batch_size, seed=seed)
    out: list[list[int]] = []
    for _ in range(steps):
        part_ids = next(sampler)
        out.append([int(rng.choice(img_idx_by_part[train_parts[j].part_number])) for j in part_ids])
    return out


def _validate(
    backbone, val_parts: list[Part], augmenter: PhotoAugmenter, max_parts: int = 500
) -> float:
    from PIL import Image

    parts = val_parts[:max_parts]
    gallery = _mean_part_embeddings(backbone, parts)
    queries = backbone.embed([augmenter(Image.open(p.image_paths[0])) for p in parts])
    sims = queries @ gallery.T
    return float((sims.argmax(axis=1) == np.arange(len(parts))).mean())
