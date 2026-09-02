"""Pydantic models shared by the catalog, pipeline, and API layers."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Part(BaseModel):
    """One catalog SKU. ``part_number`` is the McMaster-Carr part number (e.g. ``91251A537``)."""

    part_number: str = Field(..., min_length=1, description="McMaster-Carr part number")
    name: str = Field(..., description="Short product name, e.g. 'Alloy Steel Socket Head Screw'")
    category_path: list[str] = Field(
        default_factory=list,
        description="Taxonomy path, e.g. ['Fastening & Joining', 'Screws & Bolts', 'Socket Head Screws']",
    )
    description: str = ""
    attributes: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured specs: thread_size, length, material, finish, head_type, drive_style ...",
    )
    image_paths: list[str] = Field(
        default_factory=list, description="Catalog image files for this SKU"
    )
    family_id: str | None = Field(
        default=None,
        description="Groups visually identical SKUs that differ only in a non-visual spec (e.g. length)",
    )
    url: str | None = None

    @property
    def category(self) -> str:
        return self.category_path[-1] if self.category_path else ""

    def spec_text(self) -> str:
        """Compact, deterministic text rendering used for text embeddings / reranking prompts."""
        attrs = ", ".join(f"{k}={v}" for k, v in sorted(self.attributes.items()))
        cat = " > ".join(self.category_path)
        return f"{self.part_number}: {self.name}. {cat}. {attrs}. {self.description}".strip()


class MatchTier(str, Enum):
    EXACT = "exact"  # OCR/serial read or extremely confident visual match
    LIKELY = "likely"  # confident visual match, one dominant candidate
    CANDIDATE = "candidate"  # plausible, needs human confirmation
    UNKNOWN = "unknown"  # nothing in the catalog looks like this


class Candidate(BaseModel):
    part_number: str
    name: str
    category_path: list[str] = Field(default_factory=list)
    attributes: dict[str, Any] = Field(default_factory=dict)
    image_path: str | None = None
    similarity: float = Field(..., description="Cosine similarity from the vector index (0..1)")
    score: float = Field(..., description="Final fused score after reranking (0..1)")
    confidence: float = Field(
        ..., description="Calibrated probability that this is the part (0..1)"
    )
    reasons: list[str] = Field(default_factory=list, description="Human-readable evidence")


class ExtractedAttributes(BaseModel):
    """Attributes a vision model can read directly from the photo."""

    part_type: str | None = None
    material_guess: str | None = None
    finish_guess: str | None = None
    head_type: str | None = None
    drive_style: str | None = None
    visible_text: list[str] = Field(default_factory=list)
    notes: str | None = None


class IdentificationResult(BaseModel):
    request_id: str
    tier: MatchTier
    best: Candidate | None
    candidates: list[Candidate]
    ocr_part_numbers: list[str] = Field(default_factory=list)
    extracted: ExtractedAttributes | None = None
    timings_ms: dict[str, float] = Field(default_factory=dict)
    model_version: str = "unknown"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class IndexStats(BaseModel):
    backend: str
    vectors: int
    dim: int
    parts: int
    built_at: datetime | None = None
    backbone: str = "unknown"
