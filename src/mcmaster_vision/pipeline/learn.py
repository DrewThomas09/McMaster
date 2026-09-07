"""The recursive part of the loop: confirmations and purchases flow back into the model.

Two speeds, both driven by what customers actually confirmed or bought:

* ``learn_index`` (minutes): rebuild the vector index with every confirmed photo as a
  gallery image next to the catalog renders. A part photographed once is found again
  from that angle, and the usage prior already leans on purchases through
  :meth:`FeedbackStore.confirmation_counts`. The running API picks the new index up
  by itself. Writes ``learned_at`` to the manifest.
* a full retrain (hours): once enough *new* confirmations have arrived since the last
  one (``Settings.learn_retrain_after``), ``mcv learn`` runs ``mcv retrain`` with the
  purchase-weighted photos. Writes ``retrained_at``.

``mcv learn`` does the right one; ``POST /admin/learn`` does only the fast one.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

from mcmaster_vision.config import Settings
from mcmaster_vision.pipeline.feedback import FeedbackStore
from mcmaster_vision.pipeline.manifest import read_manifest, update_manifest

log = logging.getLogger(__name__)


def learning_state(settings: Settings, feedback: FeedbackStore | None = None) -> dict[str, Any]:
    """What has been learned and what is waiting: counts of confirmations since the last
    index rebuild and since the last retrain, and whether a retrain is due."""
    feedback = feedback or FeedbackStore(settings.queries_dir)
    m = read_manifest(settings)
    learned_at = m.get("learned_at")
    retrained_at = m.get("retrained_at")
    fresh = [f for f in feedback.entries_since(learned_at) if f.part_number]
    since_retrain = [f for f in feedback.entries_since(retrained_at) if f.part_number]
    return {
        "learned_at": learned_at,
        "learned_photos": m.get("learned_photos"),
        "new_confirmations": len(fresh),
        "new_purchases": len([f for f in fresh if f.source == "checkout"]),
        "since_retrain": len(since_retrain),
        "retrain_threshold": settings.learn_retrain_after,
        "retrain_due": len(since_retrain) >= settings.learn_retrain_after,
        "last_retrain": retrained_at,
    }


def learn_index(settings: Settings, *, force: bool = False) -> dict[str, Any]:
    """Fold every confirmed photo into the gallery, with the deployment's own embedder so
    the API accepts the result. Incremental when the live index was built with that
    embedder (only the photos it does not hold yet are embedded: seconds), a full
    rebuild otherwise. Skips when nothing is new unless ``force``."""
    from mcmaster_vision.catalog import CatalogStore
    from mcmaster_vision.index import add_photos, build_index, load_index
    from mcmaster_vision.models import PartEmbedder, load_backbone

    feedback = FeedbackStore(settings.queries_dir)
    state = learning_state(settings, feedback)
    if not state["new_confirmations"] and not force:
        return {"action": "none", "reason": "no new confirmations", **state}
    # the index max-pools over a part's rows, so a photo repeated by weight would add
    # nothing here: the weight matters for training and the usage prior
    labelled = feedback.labelled_images()
    n_photos = sum(len(v) for v in labelled.values())
    t = time.time()
    embedder = PartEmbedder(load_backbone(settings))
    how = "rebuild"
    added = 0
    with CatalogStore(settings.catalog_db) as store:
        idx = None
        if (settings.index_path / "meta.json").exists():
            try:
                idx = load_index(settings.index_path)
            except Exception as e:  # unreadable: rebuild below
                log.warning("could not load the index for incremental learning: %s", e)
        incremental = (
            idx is not None
            and idx.meta.get("backbone") == embedder.version
            and int(idx.meta.get("gallery_augment", -1)) == settings.index_gallery_augment
            and int(idx.meta.get("image_size", -1)) == settings.image_size
            # an older --with-feedback index does not say which photos it holds
            and (idx.meta.get("learned_paths") is not None or not idx.meta.get("extra_images"))
        )
        if incremental:
            how = "incremental"
            added = add_photos(
                idx,
                store,
                embedder,
                labelled,
                image_size=settings.image_size,
                gallery_augment=settings.index_gallery_augment,
                out_path=settings.index_path,
            )
        else:
            idx = build_index(
                store,
                embedder,
                settings.index_backend,
                out_path=settings.index_path,
                image_size=settings.image_size,
                gallery_augment=settings.index_gallery_augment,
                extra_images=labelled or None,
            )
            added = n_photos
    now = datetime.now(timezone.utc).isoformat()
    update_manifest(
        settings,
        index=idx.stats().model_dump(mode="json"),
        index_build_seconds=round(time.time() - t, 1),
        index_path=str(settings.index_path),
        index_with_feedback=True,
        learned_at=now,
        learned_photos=n_photos,
        learned_parts=len(labelled),
        learned_how=how,
    )
    log.info("learned %d photos (%d new) of %d parts, %s", n_photos, added, len(labelled), how)
    return {
        "action": "index",
        "how": how,
        "photos": n_photos,
        "added": added,
        "parts": len(labelled),
        "new_confirmations": state["new_confirmations"],
        "new_purchases": state["new_purchases"],
        "seconds": round(time.time() - t, 1),
        "learned_at": now,
        "retrain_due": state["retrain_due"],
        "since_retrain": state["since_retrain"],
    }


def mark_retrained(settings: Settings, **fields: Any) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    return update_manifest(settings, retrained_at=now, learned_at=now, **fields)
