"""Rerankers refine the top-K from vector search.

* ``FusionReranker`` (always on): combines visual similarity, the category prior,
  hit count, OCR evidence, and attribute consistency into one score.
* ``ClaudeVisionReranker`` (optional): shows the query photo and the top candidate
  images/specs to Claude and asks for a ranked verdict with extracted attributes.
  This is where fine-grained distinctions (drive style, thread pitch visible on the
  shank, plating colour) get resolved.
"""

from __future__ import annotations

import base64
import io
import logging
import math
from dataclasses import dataclass, field

from PIL import Image
from pydantic import BaseModel, Field

from mcmaster_vision.pipeline.attributes import attribute_consistency
from mcmaster_vision.pipeline.measure import Measurement, size_consistency
from mcmaster_vision.pipeline.retrieve import Hit
from mcmaster_vision.schemas import ExtractedAttributes, Part

log = logging.getLogger(__name__)


@dataclass
class Scored:
    part: Part
    similarity: float
    score: float
    reasons: list[str] = field(default_factory=list)


class FusionReranker:
    def __init__(
        self,
        w_sim: float = 1.0,
        w_cat: float = 0.15,
        w_hits: float = 0.03,
        w_attr: float = 0.12,
        w_ocr: float = 0.5,
        w_pop: float = 0.02,
        w_size: float = 0.2,
        w_exact: float = 0.35,
        exact_similarity: float = 0.99,
    ):
        self.w_sim, self.w_cat, self.w_hits, self.w_attr, self.w_ocr = (
            w_sim,
            w_cat,
            w_hits,
            w_attr,
            w_ocr,
        )
        # usage prior: parts people confirmed before are a little more likely again.
        # log1p keeps it a tie-breaker (10 confirmations ~ +0.05, a fraction of one
        # similarity step), never a substitute for visual evidence.
        self.w_pop = w_pop
        # a measured dimension is strong evidence between look-alikes: a full
        # "off" verdict costs as much as 0.2 of cosine similarity
        self.w_size = w_size
        # a near-identical gallery image (in practice: this photo, or one like it, was
        # confirmed or bought before and learned) outweighs every prior: the bonus ramps
        # from 0 at ``exact_similarity`` to ``w_exact`` at 1.0, more than category prior,
        # hits and usage prior can add together
        self.w_exact = w_exact
        self.exact_similarity = exact_similarity

    def rerank(
        self,
        hits: list[Hit],
        parts: dict[str, Part],
        *,
        extracted: ExtractedAttributes | None = None,
        ocr_part_numbers: list[str] | None = None,
        llm_ranking: dict[str, float] | None = None,
        popularity: dict[str, int] | None = None,
        size: Measurement | None = None,
    ) -> list[Scored]:
        ocr = set(ocr_part_numbers or [])
        out: list[Scored] = []
        for h in hits:
            part = parts.get(h.part_number)
            if part is None:
                continue
            reasons = [f"visual similarity {h.similarity:.2f}"]
            score = self.w_sim * h.similarity + self.w_hits * min(h.hits - 1, 3)
            if self.w_exact and h.similarity > self.exact_similarity:
                ramp = min(
                    1.0, (h.similarity - self.exact_similarity) / (1 - self.exact_similarity)
                )
                score += self.w_exact * ramp
                reasons.append("near-identical to a gallery photo")
            if h.category_prior:
                score += self.w_cat * h.category_prior
                if h.category_prior > 0.5:
                    reasons.append(f"category prior {h.category_prior:.2f} ({part.category})")
            attr_score, attr_reasons = attribute_consistency(extracted, part)
            score += self.w_attr * attr_score
            reasons.extend(attr_reasons)
            if part.part_number in ocr:
                score += self.w_ocr
                reasons.append("part number read by OCR")
            if llm_ranking and part.part_number in llm_ranking:
                score += 0.3 * llm_ranking[part.part_number]
                reasons.append(f"vision-LLM rank score {llm_ranking[part.part_number]:.2f}")
            if size is not None:
                size_score, size_reasons = size_consistency(size, part)
                score += self.w_size * size_score
                reasons.extend(size_reasons)
            n_conf = (popularity or {}).get(part.part_number, 0)
            if n_conf > 0 and self.w_pop:
                # n_conf is weighted evidence (tap 2, purchase 3); /2 keeps the prior on
                # the scale it was tuned for (one confirmation ~ one unit)
                score += self.w_pop * math.log1p(min(n_conf / 2, 50))
                # weighted evidence: a purchase counts 3, a tap 2 (FEEDBACK_WEIGHTS)
                reasons.append(f"confirmed or bought before (evidence {n_conf})")
            out.append(Scored(part, h.similarity, float(score), reasons))
        out.sort(key=lambda s: -s.score)
        return out


# ---------------------------------------------------------------------------
# Claude vision reranker
# ---------------------------------------------------------------------------
class _RankedCandidate(BaseModel):
    part_number: str
    match_probability: float = Field(description="0 (certainly not) to 1 (certainly this part)")
    reason: str


class _RerankVerdict(BaseModel):
    ranking: list[_RankedCandidate]
    extracted: ExtractedAttributes
    none_match: bool = Field(description="True if the photo shows none of the candidates")


def _strict_schema(model: type[BaseModel]) -> dict:
    """A JSON schema structured outputs accept: refs inlined, every object closed
    (``additionalProperties: false`` with all properties required) and numeric bounds /
    defaults removed (they are rejected); bounds are enforced client-side instead."""
    raw = model.model_json_schema()
    defs = raw.pop("$defs", {})
    dropped = {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "default", "title"}

    def walk(node):
        if isinstance(node, list):
            return [walk(n) for n in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            name = node["$ref"].rsplit("/", 1)[-1]
            return walk(defs[name])
        out = {k: walk(v) for k, v in node.items() if k not in dropped}
        if out.get("type") == "object" and "properties" in out:
            out["additionalProperties"] = False
            out["required"] = list(out["properties"])
        return out

    return walk(raw)


_SYSTEM_PROMPT = (
    "You identify industrial hardware from photographs against McMaster-Carr catalog entries. "
    "You are given one query photo and several candidate catalog entries, each with a part number, "
    "specs, and a catalog image. Compare geometry, head/drive style, thread visibility, material "
    "colour and finish, proportions, and any visible text or markings. Rank every candidate with a "
    "probability that it is the pictured part; probabilities need not sum to one. Also report the "
    "attributes you can read directly from the photo. If none of the candidates match, say so."
)


def _b64(img: Image.Image, max_side: int = 768) -> str:
    img = img.copy()
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=88)
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")


class ClaudeVisionReranker:
    def __init__(self, model: str = "claude-opus-5", max_candidates: int = 8, client=None):
        self.model = model
        self.max_candidates = max_candidates
        self._client = client

    @property
    def client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError as e:  # pragma: no cover
                raise ImportError("pip install 'mcmaster-vision[llm]'") from e
            self._client = anthropic.Anthropic()
        return self._client

    def build_messages(self, query: Image.Image, candidates: list[Scored]) -> list[dict]:
        content: list[dict] = [
            {"type": "text", "text": "Query photo:"},
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": _b64(query)},
            },
        ]
        for i, c in enumerate(candidates[: self.max_candidates], 1):
            content.append(
                {
                    "type": "text",
                    "text": f"Candidate {i}: {c.part.spec_text()} (visual similarity {c.similarity:.2f})",
                }
            )
            if c.part.image_paths:
                try:
                    img = Image.open(c.part.image_paths[0])
                    content.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": _b64(img, 384),
                            },
                        }
                    )
                except OSError:
                    pass
        content.append(
            {"type": "text", "text": "Rank the candidates and extract the photo's attributes."}
        )
        return [{"role": "user", "content": content}]

    def rerank(
        self, query: Image.Image, candidates: list[Scored]
    ) -> tuple[dict[str, float], ExtractedAttributes | None, bool]:
        """Return (part_number -> probability, extracted attributes, none_match)."""
        if not candidates:
            return {}, None, False
        try:
            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=4096,
                system=_SYSTEM_PROMPT,
                messages=self.build_messages(query, candidates),
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                output_config={
                    "format": {"type": "json_schema", "schema": _strict_schema(_RerankVerdict)},
                    "effort": "medium",
                },
            )
        except Exception as e:  # network / auth / quota: degrade gracefully
            log.warning("vision reranker unavailable: %s", e)
            return {}, None, False
        if response.stop_reason == "refusal":
            log.warning("vision reranker refused the request")
            return {}, None, False
        text = "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        )
        try:
            verdict = _RerankVerdict.model_validate_json(text)
        except Exception as e:
            log.warning("could not parse reranker output: %s", e)
            return {}, None, False
        allowed = {c.part.part_number for c in candidates}
        ranking = {
            r.part_number: min(1.0, max(0.0, r.match_probability))
            for r in verdict.ranking
            if r.part_number in allowed
        }
        return ranking, verdict.extracted, verdict.none_match
