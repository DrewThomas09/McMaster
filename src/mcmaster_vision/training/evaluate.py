"""Retrieval evaluation: Recall@K, MRR, and family-level recall.

Works with any backbone (numpy only). Queries are catalog images pushed through the
photo augmenter (a proxy for real user photos) unless a directory of real labelled
query photos is supplied (``<dir>/<part_number>/*.jpg``).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from PIL import Image

from mcmaster_vision.catalog.store import CatalogStore
from mcmaster_vision.data.augment import AugmentConfig, PhotoAugmenter
from mcmaster_vision.pipeline.identify import Identifier
from mcmaster_vision.schemas import Part

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass
class EvalReport:
    queries: int = 0
    recall_at: dict[int, float] = field(default_factory=dict)
    family_recall_at: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    tier_counts: dict[str, int] = field(default_factory=dict)
    tier_precision: dict[str, float] = field(default_factory=dict)
    mean_latency_ms: float = 0.0
    score_lists: list[list[float]] = field(default_factory=list, repr=False)
    correct_idx: list[int] = field(default_factory=list, repr=False)

    def to_json(self) -> str:
        d = asdict(self)
        d.pop("score_lists")
        d.pop("correct_idx")
        return json.dumps(d, indent=2)


def synthetic_queries(parts: Iterable[Part], per_part: int = 1, seed: int = 1, size: int = 224):
    aug = PhotoAugmenter(AugmentConfig.evaluation(), seed=seed)
    for p in parts:
        for i, path in enumerate(p.image_paths):
            if i >= per_part:
                break
            yield p, aug(Image.open(path), out_size=size)


def real_queries(query_dir: str | Path):
    for folder in sorted(Path(query_dir).iterdir()):
        if folder.is_dir():
            for f in sorted(folder.iterdir()):
                if f.suffix.lower() in IMAGE_EXTS:
                    yield folder.name, Image.open(f)


def evaluate_retrieval(
    identifier: Identifier,
    store: CatalogStore,
    parts: Sequence[Part] | None = None,
    *,
    ks: Sequence[int] = (1, 5, 10, 50),
    per_part: int = 1,
    query_dir: str | Path | None = None,
    query_items: Sequence[tuple[str, str | Path]] | None = None,
    max_queries: int | None = None,
    use_llm: bool = False,
) -> EvalReport:
    ks = sorted(ks)
    top_n = max(ks)
    hits = {k: 0 for k in ks}
    fam_hits = {k: 0 for k in ks}
    rr_sum = 0.0
    latency = 0.0
    tier_counts: dict[str, int] = {}
    tier_correct: dict[str, int] = {}
    report = EvalReport()

    if query_items:
        stream = ((store.get(pn), Image.open(path)) for pn, path in query_items)
    elif query_dir:
        stream = ((store.get(pn), img) for pn, img in real_queries(query_dir))
    else:
        parts = list(parts if parts is not None else store.iter_parts(with_images_only=True))
        stream = synthetic_queries(parts, per_part=per_part)

    n = 0
    for part, img in stream:
        if part is None:
            continue
        if max_queries and n >= max_queries:
            break
        res = identifier.identify(img, top_n=top_n, use_llm=use_llm)
        ranked = [c.part_number for c in res.candidates]
        looked = store.get_many(ranked[: max(ks)])
        rank = ranked.index(part.part_number) + 1 if part.part_number in ranked else 0
        fam_ranked = [
            (looked.get(pn).family_id if looked.get(pn) else None) for pn in ranked[: max(ks)]
        ]
        fam_rank = (
            (fam_ranked.index(part.family_id) + 1)
            if part.family_id and part.family_id in fam_ranked
            else 0
        )
        for k in ks:
            hits[k] += int(0 < rank <= k)
            fam_hits[k] += int(0 < fam_rank <= k)
        rr_sum += 1.0 / rank if rank else 0.0
        latency += res.timings_ms.get("total", 0.0)
        tier = res.tier.value
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
        if res.best and res.best.part_number == part.part_number:
            tier_correct[tier] = tier_correct.get(tier, 0) + 1
        report.score_lists.append([c.score for c in res.candidates])
        report.correct_idx.append(rank - 1)
        n += 1

    report.queries = n
    if n:
        report.recall_at = {k: round(hits[k] / n, 4) for k in ks}
        report.family_recall_at = {k: round(fam_hits[k] / n, 4) for k in ks}
        report.mrr = round(rr_sum / n, 4)
        report.mean_latency_ms = round(latency / n, 2)
        report.tier_counts = tier_counts
        report.tier_precision = {
            t: round(tier_correct.get(t, 0) / c, 4) for t, c in tier_counts.items()
        }
    return report
