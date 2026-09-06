"""Deployment manifest: what is built, from what, and how well it measures.

``data/manifest.json`` is written by ``mcv bootstrap`` / ``mcv build-index`` and
read by ``mcv status`` and ``GET /status`` so operators can see at a glance
whether the served index matches the catalog and the model.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mcmaster_vision import __version__
from mcmaster_vision.config import Settings


def manifest_path(settings: Settings) -> Path:
    return settings.data_dir / "manifest.json"


def read_manifest(settings: Settings) -> dict[str, Any]:
    p = manifest_path(settings)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def update_manifest(settings: Settings, **fields: Any) -> dict[str, Any]:
    m = read_manifest(settings)
    m.update(fields)
    m["version"] = __version__
    m["updated_at"] = datetime.now(timezone.utc).isoformat()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    manifest_path(settings).write_text(json.dumps(m, indent=2, default=str), encoding="utf-8")
    return m


def status(settings: Settings) -> dict[str, Any]:
    """Live view: catalog counts, index meta, calibration, feedback, manifest."""
    from mcmaster_vision.catalog.store import CatalogStore
    from mcmaster_vision.pipeline.feedback import FeedbackStore

    out: dict[str, Any] = {
        "manifest": read_manifest(settings),
        "settings": {
            "backbone": settings.backbone,
            "backbone_checkpoint": str(settings.backbone_checkpoint)
            if settings.backbone_checkpoint
            else None,
            "index_backend": settings.index_backend,
            "gallery_augment": settings.index_gallery_augment,
            "query_expansion_k": settings.query_expansion_k,
            "rerank_llm_enabled": settings.rerank_llm_enabled,
            "ocr_enabled": settings.ocr_enabled,
        },
    }
    if settings.catalog_db.exists():
        with CatalogStore(settings.catalog_db) as store:
            n = store.count()
            with_images = store.count(with_images_only=True)
            out["catalog"] = {
                "parts": n,
                "parts_with_images": with_images,
                "index_backbone": store.get_meta("index_backbone"),
                "updated_at": store.get_meta("updated_at"),
            }
    else:
        out["catalog"] = None
    meta = settings.index_path / "meta.json"
    out["index"] = json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else None
    cal = settings.model_dir / "calibration.json"
    out["calibration"] = json.loads(cal.read_text(encoding="utf-8")) if cal.exists() else None
    out["feedback"] = (
        FeedbackStore(settings.queries_dir).stats() if settings.queries_dir.exists() else None
    )
    out["ready"] = bool(out["catalog"] and out["index"])
    # the catalog changed after the index was built -> parts missing from search
    cat_updated = out["catalog"].get("updated_at") if out.get("catalog") else None
    built = out["index"].get("built_at") if out.get("index") else None
    out["index_stale"] = bool(cat_updated and built and cat_updated > built)
    return out
