"""End-to-end identification."""

from __future__ import annotations

import hashlib
import logging
import re
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path

import numpy as np
from PIL import Image

from mcmaster_vision.catalog.store import CatalogStore
from mcmaster_vision.config import Settings
from mcmaster_vision.index.base import VectorIndex, load_index
from mcmaster_vision.models.backbone import backbone_matches, load_backbone
from mcmaster_vision.models.embedder import PartEmbedder
from mcmaster_vision.pipeline.calibration import Calibration
from mcmaster_vision.pipeline.feedback import FeedbackStore
from mcmaster_vision.pipeline.measure import Measurement, erase_reference, measure
from mcmaster_vision.pipeline.ocr import OCREngine
from mcmaster_vision.pipeline.preprocess import decode_image, preprocess
from mcmaster_vision.pipeline.reference import find_coin
from mcmaster_vision.pipeline.rerank import ClaudeVisionReranker, FusionReranker, Scored
from mcmaster_vision.pipeline.retrieve import Retriever
from mcmaster_vision.pipeline.threads import measure_thread_pitch
from mcmaster_vision.schemas import (
    Candidate,
    ExtractedAttributes,
    FamilyHint,
    IdentificationResult,
    MatchTier,
    Part,
    spec_key,
)

log = logging.getLogger(__name__)


def merge_equivalents(parts: list[Part], probs: list[float]) -> tuple[list[float], list[str]]:
    """Fold candidates that are the same spec as the best one into it: returns the
    probabilities to judge the tier on (the best's plus its twins', then the rest) and
    the twins' part numbers."""
    if not parts or not probs:
        return list(probs), []
    key = spec_key(parts[0])
    merged = probs[0]
    twins: list[str] = []
    rest: list[float] = []
    for part, p in zip(parts[1:], probs[1:], strict=False):
        if spec_key(part) == key:
            merged += p
            twins.append(part.part_number)
        else:
            rest.append(p)
    return [min(1.0, merged), *rest], twins


class Identifier:
    def __init__(
        self,
        store: CatalogStore,
        index: VectorIndex,
        embedder: PartEmbedder,
        *,
        top_k: int = 50,
        calibration: Calibration | None = None,
        ocr: OCREngine | None = None,
        llm_reranker: ClaudeVisionReranker | None = None,
        image_size: int = 224,
        segment: bool = False,
        qe_k: int = 0,
        feedback: FeedbackStore | None = None,
    ):
        self.store = store
        self.feedback = feedback
        self._pop: dict[str, int] = {}
        self._pop_mtime = -1.0
        self.index = index
        self.embedder = embedder
        self.retriever = Retriever(index, top_k=top_k, qe_k=qe_k)
        self.fusion = FusionReranker()
        self.calibration = calibration or Calibration()
        self.ocr = ocr
        self.llm = llm_reranker
        self.image_size = image_size
        self.segment = segment
        # query-embedding cache keyed by (photo hash, tta mode): constraint refinements and
        # "add another angle" re-send the same photos, so their vectors are reused
        self._qcache: OrderedDict[tuple[str, str], np.ndarray] = OrderedDict()
        self._qcache_max = 256

    # ------------------------------------------------------------ helpers
    def _popularity(self) -> dict[str, int]:
        """Confirmation counts from the feedback store, re-read when the log changes."""
        if self.feedback is None:
            return {}
        m = self.feedback.mtime()
        if m != self._pop_mtime:
            self._pop = self.feedback.confirmation_counts()
            self._pop_mtime = m
        return self._pop

    def _timer(self, timings: dict[str, float], key: str, start: float) -> float:
        timings[key] = round((time.perf_counter() - start) * 1000, 2)
        return time.perf_counter()

    def _to_candidates(self, scored: list[Scored], probs: list[float]) -> list[Candidate]:
        return [
            Candidate(
                part_number=s.part.part_number,
                name=s.part.name,
                category_path=s.part.category_path,
                attributes=s.part.attributes,
                image_path=s.part.image_paths[0] if s.part.image_paths else None,
                similarity=round(s.similarity, 4),
                score=round(s.score, 4),
                confidence=round(p, 4),
                reasons=s.reasons,
            )
            for s, p in zip(scored, probs, strict=True)
        ]

    def _embed_cached(self, image: Image.Image, tta: str, key: str | None) -> np.ndarray:
        if key is None:
            return self.embedder.embed_query(image, tta=tta)
        k = (key, tta)
        hit = self._qcache.get(k)
        if hit is not None:
            self._qcache.move_to_end(k)
            return hit
        vec = self.embedder.embed_query(image, tta=tta)
        self._qcache[k] = vec
        if len(self._qcache) > self._qcache_max:
            self._qcache.popitem(last=False)
        return vec

    @staticmethod
    def _family_hint(
        scored: list[Scored], probs: list[float], min_mass: float = 0.6
    ) -> FamilyHint | None:
        """If most of the probability mass sits on several SKUs of one family, report the
        family and the attributes that differ between them (the question to ask the user)."""
        mass: dict[str, float] = {}
        members: dict[str, list[Scored]] = {}
        for s, p in zip(scored, probs, strict=True):
            fam = s.part.family_id
            if not fam:
                continue
            mass[fam] = mass.get(fam, 0.0) + p
            members.setdefault(fam, []).append(s)
        if not mass:
            return None
        fam = max(mass, key=mass.get)  # type: ignore[arg-type]
        if mass[fam] < min_mass or len(members[fam]) < 2:
            return None
        parts = [m.part for m in members[fam]]
        differing: dict[str, list[str]] = {}
        keys = {k for p in parts for k in p.attributes}
        for k in sorted(keys):
            values = [str(p.attributes.get(k, "?")) for p in parts]
            if len(set(values)) > 1:
                differing[k] = sorted(set(values))
        return FamilyHint(
            family_id=fam,
            name=parts[0].name,
            part_numbers=[p.part_number for p in parts],
            probability=round(mass[fam], 4),
            distinguishing_attributes=differing,
        )

    # ------------------------------------------------------------ public
    def identify_bytes(
        self, data: bytes, *, top_n: int = 5, use_llm: bool | None = None
    ) -> IdentificationResult:
        return self.identify(decode_image(data), top_n=top_n, use_llm=use_llm)

    def identify_path(self, path: str | Path, **kw) -> IdentificationResult:
        return self.identify_bytes(Path(path).read_bytes(), **kw)

    def identify_many_bytes(self, blobs: list[bytes], **kw) -> IdentificationResult:
        keys = [hashlib.sha1(b).hexdigest() for b in blobs]
        return self.identify([decode_image(b) for b in blobs], cache_keys=keys, **kw)

    def identify(
        self,
        image: Image.Image | list[Image.Image],
        *,
        top_n: int = 5,
        use_llm: bool | None = None,
        constraints: dict[str, str] | None = None,
        tta: str = "full",
        cache_keys: list[str] | None = None,
        mm_per_px: float | None = None,
        reference: tuple[float, float, float, float] | None = None,
        suggest_reference: bool = False,
        customer_prior: dict[str, float] | Callable[[list[str]], dict[str, float]] | None = None,
    ) -> IdentificationResult:
        """Identify one photo, or several photos of the *same* part (different angles):
        every photo's TTA variants are searched and each catalog part keeps its best score.

        ``constraints`` are attribute filters the user already knows (``{"thread_size": "M6"}``):
        candidates whose attribute value differs are dropped before calibration, which is how
        a family question ("which length?") is answered in one tap.

        ``mm_per_px`` is the scale of the first photo (the user marked a coin, a card or a
        ruler): the object's extent is measured and compared with each candidate's catalog
        dimensions, which separates look-alikes that share one catalog image. ``reference``
        is the segment (x1, y1, x2, y2, uploaded pixels) drawn across the reference object,
        so its blob is not mistaken for the part."""
        images = image if isinstance(image, list) else [image]
        if not images:
            raise ValueError("no images")
        timings: dict[str, float] = {}
        t = time.perf_counter()
        request_id = uuid.uuid4().hex[:12]
        notes: list[str] = []

        # 1. preprocess (the reference object, when marked, is erased before embedding:
        # a coin next to the part sets the scale and must not vote on looks)
        image = images[0]
        embed_first = erase_reference(image, reference) if reference else image
        query_imgs = [
            preprocess(im, size=self.image_size, segment=self.segment)
            for im in [embed_first, *images[1:]]
        ]
        query_img = query_imgs[0]
        size: Measurement | None = None
        coin_hint: dict[str, float] | None = None
        if mm_per_px:
            size = measure(image, mm_per_px, reference)
            if size is None:
                notes.append("could not find the object outline to measure it; size not used")
            else:
                tp = measure_thread_pitch(image, mm_per_px, reference)
                if tp is not None:
                    size.pitch_mm = tp.pitch_mm
        elif suggest_reference:
            coin = find_coin(image)
            if coin is not None:  # report it in *uploaded* pixels, like mm_per_px / ref
                k = float(image.info.get("upload_scale", 1.0) or 1.0)
                coin_hint = {
                    "cx": round(coin.cx * k, 1),
                    "cy": round(coin.cy * k, 1),
                    "diameter_px": round(coin.diameter_px * k, 1),
                }
        t = self._timer(timings, "preprocess", t)

        # 2. OCR (may short-circuit)
        ocr_pns: list[str] = []
        if self.ocr is not None and self.ocr.available:
            ocr_pns = [pn for pn in self.ocr.read(image).part_numbers if self.store.get(pn)]
            t = self._timer(timings, "ocr", t)

        # 3. embed + retrieve (all photos' TTA variants stacked into one multi-query)
        # the first photo's embedding differs with and without a reference (it is erased
        # before embedding), so the cache key must say which one this is
        ref_tag = f"|ref={tuple(round(v, 1) for v in reference)}" if reference else ""
        qvec = np.concatenate(
            [
                self._embed_cached(
                    q,
                    tta,
                    (cache_keys[i] + (ref_tag if i == 0 else ""))
                    if cache_keys and i < len(cache_keys)
                    else None,
                )
                for i, q in enumerate(query_imgs)
            ],
            axis=0,
        )
        t = self._timer(timings, "embed", t)
        hits = self.retriever.retrieve(qvec)
        for pn in ocr_pns:  # make sure OCR'd parts are in the candidate pool
            if all(h.part_number != pn for h in hits):
                from mcmaster_vision.pipeline.retrieve import Hit

                hits.append(Hit(pn, 0.0, 0))
        parts = self.store.get_many(h.part_number for h in hits)
        self.retriever.apply_category_prior(
            hits, qvec, {pn: p.category_path for pn, p in parts.items()}
        )
        t = self._timer(timings, "retrieve", t)

        # 4. fuse (first pass); a callable prior is asked only about the candidates
        pop = self._popularity()
        if callable(customer_prior):
            customer_prior = customer_prior([h.part_number for h in hits])
        scored = self.fusion.rerank(
            hits,
            parts,
            ocr_part_numbers=ocr_pns,
            popularity=pop,
            size=size,
            customer_prior=customer_prior,
        )

        # 5. optional vision-LLM rerank on the short list
        extracted: ExtractedAttributes | None = None
        none_match = False
        run_llm = self.llm is not None if use_llm is None else (use_llm and self.llm is not None)
        if run_llm and scored:
            ranking, extracted, none_match = self.llm.rerank(query_img, scored)
            scored = self.fusion.rerank(
                hits,
                parts,
                extracted=extracted,
                ocr_part_numbers=ocr_pns,
                llm_ranking=ranking,
                popularity=pop,
                size=size,
                customer_prior=customer_prior,
            )
            t = self._timer(timings, "llm_rerank", t)

        # 5b. attribute constraints (case-insensitive equality on stringified values)
        constraints = {k: str(v) for k, v in (constraints or {}).items() if str(v).strip()}
        if constraints:

            def _ok(part) -> bool:
                return all(
                    _norm_attr(part.attributes.get(k, "")) == _norm_attr(v)
                    for k, v in constraints.items()
                )

            filtered = [s for s in scored if _ok(s.part)]
            if filtered:
                scored = filtered
            else:
                notes.append(
                    "no retrieved candidate matches "
                    + ", ".join(f"{k}={v}" for k, v in constraints.items())
                    + "; showing unfiltered results"
                )
                constraints = {}

        # 6. calibrate
        top = scored[:top_n]
        probs = self.calibration.probabilities([s.score for s in scored])[:top_n]
        candidates = self._to_candidates(top, probs)
        best_sim = top[0].similarity if top else 0.0
        # two listings of one spec are one answer: their probability adds up and the
        # runner-up for the margin is the first candidate that is a different thing
        tier_probs, also_sold_as = merge_equivalents([s.part for s in top], probs)
        tier = self.calibration.tier(
            tier_probs,
            best_sim,
            ocr_hit=bool(ocr_pns) and bool(top) and top[0].part.part_number in ocr_pns,
        )
        if none_match and tier != MatchTier.EXACT:
            tier = MatchTier.UNKNOWN
        self._timer(timings, "calibrate", t)
        timings["total"] = round(sum(timings.values()), 2)

        return IdentificationResult(
            request_id=request_id,
            tier=tier,
            best=candidates[0] if candidates and tier != MatchTier.UNKNOWN else None,
            candidates=candidates,
            also_sold_as=also_sold_as if tier != MatchTier.UNKNOWN else [],
            family=self._family_hint(top, probs) if tier != MatchTier.UNKNOWN else None,
            category_guess=sorted(
                self.retriever.category_prior(qvec).items(), key=lambda kv: -kv[1]
            )[:3],
            constraints=constraints,
            notes=notes,
            photos=len(images),
            measured=size.as_dict() if size else None,
            coin_hint=coin_hint,
            ocr_part_numbers=ocr_pns,
            extracted=extracted,
            timings_ms=timings,
            model_version=self.embedder.version,
        )


def _norm_attr(value) -> str:
    """Loose attribute equality: case, whitespace, quote marks and unit words ignored,
    so ``1/4"-20`` == ``1/4-20`` == ``1/4 in - 20``."""
    v = str(value).lower().strip()
    v = v.replace("\u201d", "").replace("\u2033", "").replace('"', "").replace("'", "")
    v = re.sub(r"\b(inch|inches|in\.?)\b", "", v)
    v = re.sub(r"\s+", "", v)
    return v.replace("–", "-").replace("—", "-")


def load_identifier(settings: Settings) -> Identifier:
    """Assemble an Identifier from settings + artifacts on disk."""
    store = CatalogStore(settings.catalog_db)
    index = load_index(settings.index_path)
    backbone = load_backbone(settings)
    embedder = PartEmbedder(backbone)
    indexed_with = index.meta.get("backbone")
    if indexed_with and not backbone_matches(indexed_with, embedder.version):
        # serving a mismatched pair would 500 on every photo (or silently degrade when the
        # dimensions happen to agree); refuse, and say exactly what to change
        raise RuntimeError(
            f"index at {settings.index_path} was built with backbone {indexed_with!r} but the "
            f"configured backbone is {embedder.version!r}: set MCV_BACKBONE / "
            "MCV_BACKBONE_CHECKPOINT to match, or run `mcv build-index`"
        )
    if index.dim != embedder.dim:
        raise RuntimeError(
            f"index dimension {index.dim} != backbone dimension {embedder.dim}; rebuild the index"
        )
    calibration = Calibration.load(settings.model_dir / "calibration.json")
    ocr = OCREngine() if settings.ocr_enabled else None
    llm = (
        ClaudeVisionReranker(settings.rerank_llm_model, settings.rerank_llm_candidates)
        if settings.rerank_llm_enabled
        else None
    )
    return Identifier(
        store,
        index,
        embedder,
        top_k=settings.index_top_k,
        qe_k=settings.query_expansion_k,
        calibration=calibration,
        ocr=ocr,
        llm_reranker=llm,
        image_size=settings.image_size,
        feedback=FeedbackStore(settings.queries_dir),
    )
