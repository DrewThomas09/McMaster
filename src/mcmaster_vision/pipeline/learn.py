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
from pathlib import Path
from typing import Any

from mcmaster_vision.config import Settings
from mcmaster_vision.pipeline.feedback import FeedbackStore
from mcmaster_vision.pipeline.manifest import read_manifest, update_manifest

log = logging.getLogger(__name__)


class LearnBusy(RuntimeError):
    """Another learn / retrain holds the lock (cron and the dashboard button overlap)."""


class _Lock:
    """An exclusive file lock on ``data/learn.lock`` held for the duration of a learn."""

    def __init__(self, settings: Settings):
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        self.path = settings.data_dir / "learn.lock"
        self.fh = None

    def __enter__(self):
        import fcntl

        self.fh = open(self.path, "a+")
        try:
            fcntl.flock(self.fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            self.fh.close()
            self.fh = None
            raise LearnBusy("a learn or retrain is already running") from e
        return self

    def __exit__(self, *exc):
        import fcntl

        if self.fh is not None:
            fcntl.flock(self.fh, fcntl.LOCK_UN)
            self.fh.close()


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

    with _Lock(settings):
        return _learn_index(settings, force=force)


def _learn_index(settings: Settings, *, force: bool) -> dict[str, Any]:
    from mcmaster_vision.catalog import CatalogStore
    from mcmaster_vision.index import add_photos, build_index, load_index
    from mcmaster_vision.models import PartEmbedder, load_backbone

    feedback = FeedbackStore(settings.queries_dir)
    state = learning_state(settings, feedback)
    if not state["new_confirmations"] and not force:
        return {"action": "none", "reason": "no new confirmations", **state}
    # stamped before the work: a confirmation that lands during the rebuild is still
    # "new" next time instead of being lost behind a later timestamp
    now = datetime.now(timezone.utc).isoformat()
    manifest = read_manifest(settings)
    # photos a retrain held out for its evaluation stay out of the gallery, or the
    # manifest's retrain_eval would stop describing the served index
    held_out = set(manifest.get("held_out_paths") or [])
    # the index max-pools over a part's rows, so a photo repeated by weight would add
    # nothing here: the weight matters for training and the usage prior
    labelled = {
        pn: [x for x in paths if x not in held_out]
        for pn, paths in feedback.labelled_images().items()
    }
    labelled = {pn: v for pn, v in labelled.items() if v}
    n_photos = sum(len(v) for v in labelled.values())
    t = time.time()
    embedder = PartEmbedder(load_backbone(settings))
    how = "rebuild"
    added = 0
    calibrated: dict[str, Any] = {}
    samples: list[dict] = []
    with CatalogStore(settings.catalog_db) as store:
        idx = None
        if (settings.index_path / "meta.json").exists():
            try:
                idx = load_index(settings.index_path)
            except Exception as e:  # unreadable: rebuild below
                log.warning("could not load the index for incremental learning: %s", e)
        learned_before = list((idx.meta.get("learned_paths") or []) if idx is not None else [])
        incremental = (
            idx is not None
            and idx.meta.get("backbone") == embedder.version
            and int(idx.meta.get("gallery_augment", -1)) == settings.index_gallery_augment
            and int(idx.meta.get("image_size", -1)) == settings.image_size
            # an older --with-feedback index does not say which photos it holds, and one
            # without category counts would let a few photos overwrite the category prior
            and (idx.meta.get("learned_paths") is not None or not idx.meta.get("extra_images"))
            and (idx.meta.get("category_counts") is not None or not idx.category_names)
            # a corrected label deletes the photo under the old part: only a rebuild
            # removes its row from the index
            and all(Path(x).exists() for x in learned_before)
        )
        if incremental:
            how = "incremental"
            # score the new photos *before* they join the gallery: each confirmed photo is
            # one honest calibration sample, and enough of them refit the thresholds on
            # what customers actually bought instead of on synthetic renders
            samples = calibration_samples(settings, store, idx, embedder, labelled)
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
    if samples:  # written only once the new index is safely on disk
        calibrated = calibrate_from_samples(settings, samples)
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
        **({"calibration_from_purchases": calibrated} if calibrated else {}),
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
        "calibration": calibrated or None,
    }


CALIBRATION_SAMPLES = "calibration_samples.jsonl"
MIN_CALIBRATION_SAMPLES = 30
MIN_WRONG_SAMPLES = 5


def calibration_samples(
    settings: Settings, store, index, embedder, labelled: dict[str, list[str]]
) -> list[dict]:
    """Identify every confirmed photo the index does not hold yet, without the usage
    prior (its own confirmation would flatter it) and with the serving default
    test-time augmentation, and return one calibration sample per photo."""
    from PIL import Image

    from mcmaster_vision.pipeline.calibration import Calibration
    from mcmaster_vision.pipeline.identify import Identifier

    known = set(index.meta.get("learned_paths") or [])
    fresh = [
        (pn, x)
        for pn, paths in labelled.items()
        if store.get(pn) is not None  # a photo of an unknown part is never learned either
        for x in paths
        if x not in known
    ]
    if not fresh:
        return []
    ident = Identifier(
        store,
        index,
        embedder,
        top_k=settings.index_top_k,
        qe_k=settings.query_expansion_k,
        calibration=Calibration.load(settings.model_dir / "calibration.json"),
        image_size=settings.image_size,
        feedback=None,
    )
    rows: list[dict] = []
    for pn, x in fresh:
        try:
            res = ident.identify(Image.open(x), top_n=10, tta="full")
        except Exception as e:  # one bad photo must not stop learning
            log.warning("calibration sample skipped for %s: %s", x, e)
            continue
        ranked = [c.part_number for c in res.candidates]
        rows.append(
            {
                "scores": [round(c.score, 4) for c in res.candidates],
                "correct": ranked.index(pn) if pn in ranked else -1,
                "tier": res.tier.value,
                "model": embedder.version,
                "at": datetime.now(timezone.utc).isoformat(),
            }
        )
    return rows


def calibrate_from_samples(settings: Settings, rows: list[dict]) -> dict[str, Any]:
    """Append the samples to ``models/calibration_samples.jsonl`` and, once
    ``MIN_CALIBRATION_SAMPLES`` exist with at least ``MIN_WRONG_SAMPLES`` wrong answers
    among them (a refit on all-correct samples would only sharpen the softmax), refit
    the temperature and the exact / likely thresholds."""
    import json

    from mcmaster_vision.pipeline.calibration import Calibration

    path = settings.model_dir / CALIBRATION_SAMPLES
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    # only samples scored by the model being served count: after a retrain the old
    # model's score lists would refit the new model's thresholds
    model = rows[0].get("model") if rows else None
    samples = []
    for ln in path.read_text(encoding="utf-8").splitlines()[-2000:]:
        try:
            r = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if model is None or r.get("model") == model:
            samples.append(r)
    wrong = sum(int(x.get("correct", -1)) != 0 for x in samples)
    out = {"new_samples": len(rows), "samples": len(samples), "wrong": wrong, "refit": False}
    if len(samples) < MIN_CALIBRATION_SAMPLES or wrong < MIN_WRONG_SAMPLES:
        return out
    score_lists = [x["scores"] for x in samples]
    correct = [int(x["correct"]) for x in samples]
    cal = Calibration.fit_temperature(score_lists, correct).fit_thresholds(score_lists, correct)
    cal.save(settings.model_dir / "calibration.json")
    out.update(
        refit=True,
        temperature=cal.temperature,
        exact_threshold=cal.exact_threshold,
        likely_threshold=cal.likely_threshold,
        top1=round(sum(c == 0 for c in correct) / len(correct), 3),
    )
    log.info("calibration refit on %d confirmed photos: %s", len(samples), out)
    return out


def mark_retrained(
    settings: Settings, *, started_at: str | None = None, learned: bool = True, **fields: Any
) -> dict[str, Any]:
    """Stamp the manifest after a retrain; ``started_at`` (taken before training) keeps
    confirmations that arrived during the run counted as new. ``learned=False`` when the
    retrain went to sibling directories: the served index has not seen those photos, so
    ``learned_at`` stays where it was and the next ``mcv learn`` still adds them."""
    now = started_at or datetime.now(timezone.utc).isoformat()
    stamp = {"retrained_at": now, **({"learned_at": now} if learned else {})}
    return update_manifest(settings, **stamp, **fields)
