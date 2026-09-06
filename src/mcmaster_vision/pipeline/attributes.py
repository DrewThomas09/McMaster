"""Attribute-consistency scoring.

Compares attributes a vision model read from the photo (material, head type,
drive style, visible text) against each candidate's catalog specs. Cheap, and
often decisive between look-alike SKUs (zinc vs. stainless, hex vs. socket head).
"""

from __future__ import annotations

from mcmaster_vision.schemas import ExtractedAttributes, Part

_MATERIAL_SYNONYMS = {
    "stainless": ["stainless", "18-8", "316", "304"],
    "steel": ["steel", "zinc", "black-oxide", "alloy"],
    "brass": ["brass"],
    "aluminum": ["aluminum", "aluminium"],
    "nylon": ["nylon", "plastic", "polymer"],
    "copper": ["copper"],
    "bronze": ["bronze"],
    "titanium": ["titanium"],
}


def _norm(s: str | None) -> str:
    return (s or "").lower().strip()


def _material_match(guess: str, catalog: str) -> float | None:
    g, c = _norm(guess), _norm(catalog)
    if not g or not c:
        return None
    for _, syns in _MATERIAL_SYNONYMS.items():
        if any(s in g for s in syns):
            return 1.0 if any(s in c for s in syns) else -1.0
    return None


def attribute_consistency(
    extracted: ExtractedAttributes | None, part: Part
) -> tuple[float, list[str]]:
    """Return (score in [-1, 1], reasons). 0 means "no evidence either way"."""
    if extracted is None:
        return 0.0, []
    votes: list[float] = []
    reasons: list[str] = []
    attrs = {k.lower(): str(v) for k, v in part.attributes.items()}
    haystack = " ".join(
        [part.name, part.description, " ".join(part.category_path), *attrs.values()]
    ).lower()

    m = _material_match(extracted.material_guess or "", attrs.get("material", part.name))
    if m is not None:
        votes.append(m)
        reasons.append(
            f"material {'matches' if m > 0 else 'conflicts'}: photo={extracted.material_guess}, catalog={attrs.get('material', part.name)}"
        )

    for field in ("head_type", "drive_style", "part_type"):
        g = _norm(getattr(extracted, field))
        if g:
            hit = any(tok in haystack for tok in g.split() if len(tok) > 2)
            votes.append(1.0 if hit else -0.5)
            if hit:
                reasons.append(f"{field.replace('_', ' ')} '{g}' found in catalog entry")

    for text in extracted.visible_text:
        t = _norm(text)
        if len(t) >= 3 and (t in haystack or t in part.part_number.lower()):
            votes.append(1.0)
            reasons.append(f"visible text '{text}' matches catalog")

    if not votes:
        return 0.0, reasons
    return max(-1.0, min(1.0, sum(votes) / len(votes))), reasons
