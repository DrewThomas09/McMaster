"""End-to-end identification."""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

from mcmaster_vision.catalog.store import CatalogStore
from mcmaster_vision.config import Settings
from mcmaster_vision.index.base import VectorIndex, load_index
from mcmaster_vision.models.backbone import load_backbone
from mcmaster_vision.models.embedder import PartEmbedder
from mcmaster_vision.pipeline.calibration import Calibration
from mcmaster_vision.pipeline.ocr import OCREngine
from mcmaster_vision.pipeline.preprocess import decode_image, preprocess
from mcmaster_vision.pipeline.rerank import ClaudeVisionReranker, FusionReranker, Scored
from mcmaster_vision.pipeline.retrieve import Retriever
from mcmaster_vision.schemas import (
    Candidate,
    ExtractedAttributes,
    FamilyHint,
    IdentificationResult,
    MatchTier,
)

log = logging.getLogger(__name__)


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
    ):
        self.store = store
        self.index = index
        self.embedder = embedder
        self.retriever = Retriever(index, top_k=top_k, qe_k=qe_k)
        self.fusion = FusionReranker()
        self.calibration = calibration or Calibration()
        self.ocr = ocr
        self.llm = llm_reranker
        self.image_size = image_size
        self.segment = segment

    # ------------------------------------------------------------ helpers
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
        return self.identify([decode_image(b) for b in blobs], **kw)

    def identify(
        self,
        image: Image.Image | list[Image.Image],
        *,
        top_n: int = 5,
        use_llm: bool | None = None,
    ) -> IdentificationResult:
        """Identify one photo, or several photos of the *same* part (different angles):
        every photo's TTA variants are searched and each catalog part keeps its best score."""
        images = image if isinstance(image, list) else [image]
        if not images:
            raise ValueError("no images")
        timings: dict[str, float] = {}
        t = time.perf_counter()
        request_id = uuid.uuid4().hex[:12]

        # 1. preprocess
        query_imgs = [preprocess(im, size=self.image_size, segment=self.segment) for im in images]
        query_img = query_imgs[0]
        image = images[0]
        t = self._timer(timings, "preprocess", t)

        # 2. OCR (may short-circuit)
        ocr_pns: list[str] = []
        if self.ocr is not None and self.ocr.available:
            ocr_pns = [pn for pn in self.ocr.read(image).part_numbers if self.store.get(pn)]
            t = self._timer(timings, "ocr", t)

        # 3. embed + retrieve (all photos' TTA variants stacked into one multi-query)
        qvec = np.concatenate([self.embedder.embed_query(q) for q in query_imgs], axis=0)
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

        # 4. fuse (first pass)
        scored = self.fusion.rerank(hits, parts, ocr_part_numbers=ocr_pns)

        # 5. optional vision-LLM rerank on the short list
        extracted: ExtractedAttributes | None = None
        none_match = False
        run_llm = self.llm is not None if use_llm is None else (use_llm and self.llm is not None)
        if run_llm and scored:
            ranking, extracted, none_match = self.llm.rerank(query_img, scored)
            scored = self.fusion.rerank(
                hits, parts, extracted=extracted, ocr_part_numbers=ocr_pns, llm_ranking=ranking
            )
            t = self._timer(timings, "llm_rerank", t)

        # 6. calibrate
        top = scored[:top_n]
        probs = self.calibration.probabilities([s.score for s in scored])[:top_n]
        candidates = self._to_candidates(top, probs)
        best_sim = top[0].similarity if top else 0.0
        tier = self.calibration.tier(
            probs,
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
            family=self._family_hint(top, probs) if tier != MatchTier.UNKNOWN else None,
            photos=len(images),
            ocr_part_numbers=ocr_pns,
            extracted=extracted,
            timings_ms=timings,
            model_version=self.embedder.version,
        )


def load_identifier(settings: Settings) -> Identifier:
    """Assemble an Identifier from settings + artifacts on disk."""
    store = CatalogStore(settings.catalog_db)
    index = load_index(settings.index_path)
    backbone = load_backbone(settings)
    embedder = PartEmbedder(backbone)
    indexed_with = index.meta.get("backbone")
    if indexed_with and indexed_with != embedder.version:
        log.warning(
            "index was built with %s but backbone is %s; rebuild the index",
            indexed_with,
            embedder.version,
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
    )
