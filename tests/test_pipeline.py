from __future__ import annotations

import io

import numpy as np
from PIL import Image

from mcmaster_vision.data import PhotoAugmenter
from mcmaster_vision.pipeline.attributes import attribute_consistency
from mcmaster_vision.pipeline.calibration import Calibration
from mcmaster_vision.pipeline.ocr import extract_part_numbers
from mcmaster_vision.pipeline.preprocess import decode_image, pad_to_square, saliency_crop
from mcmaster_vision.pipeline.rerank import ClaudeVisionReranker, FusionReranker, Scored
from mcmaster_vision.pipeline.retrieve import Hit
from mcmaster_vision.schemas import ExtractedAttributes, MatchTier, Part
from mcmaster_vision.training import evaluate_retrieval


def test_identify_clean_catalog_image_is_top1(identifier, store):
    part = next(store.iter_parts(with_images_only=True))
    res = identifier.identify(Image.open(part.image_paths[0]))
    assert res.candidates[0].part_number == part.part_number
    assert res.tier in {MatchTier.EXACT, MatchTier.LIKELY, MatchTier.CANDIDATE}
    assert res.best is not None
    assert "total" in res.timings_ms
    assert sum(c.confidence for c in res.candidates) <= 1.0 + 1e-6
    assert res.candidates[0].confidence == max(c.confidence for c in res.candidates)
    assert res.candidates[0].similarity > 0.99


def test_identify_bytes_roundtrip(identifier, store):
    part = list(store.iter_parts(with_images_only=True))[5]
    buf = io.BytesIO()
    Image.open(part.image_paths[1]).save(buf, format="JPEG", quality=80)
    res = identifier.identify_bytes(buf.getvalue(), top_n=3)
    assert len(res.candidates) == 3
    assert part.part_number in [c.part_number for c in res.candidates]


def test_augmented_queries_recall_is_reasonable(identifier, store):
    report = evaluate_retrieval(identifier, store, max_queries=40, ks=(1, 5, 10))
    assert report.queries == 40
    # The dependency-free hash backbone is deliberately simple; on photo-style
    # queries it should still land the right SKU in the top-10 well above chance
    # (chance here is 10/40 = 0.25). Real accuracy comes from the neural backbones.
    assert report.recall_at[10] >= 0.45
    assert report.recall_at[1] >= 0.15
    assert report.recall_at[1] <= report.recall_at[5] <= report.recall_at[10]
    assert report.mrr > 0


def test_augmenter_output_shape_and_variety():
    img = Image.new("RGB", (120, 60), (255, 255, 255))
    Image.Image.paste(img, Image.new("RGB", (40, 20), (90, 90, 90)), (40, 20))
    aug = PhotoAugmenter(seed=1)
    a, b = aug(img, out_size=96), aug(img, out_size=96)
    assert a.size == (96, 96) and b.size == (96, 96)
    assert not np.array_equal(np.asarray(a), np.asarray(b))


def test_preprocess_helpers():
    img = Image.new("RGB", (200, 100), (240, 240, 240))
    img.paste(Image.new("RGB", (40, 40), (20, 20, 20)), (80, 30))
    crop = saliency_crop(img)
    assert crop.size[0] < 200 and crop.size[1] <= 100
    sq = pad_to_square(crop)
    assert sq.size[0] == sq.size[1]
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    assert decode_image(buf.getvalue()).size == (200, 100)


def test_ocr_regex():
    assert extract_part_numbers(["Part 91251A537 qty 100", "9452K21", "hello"]) == [
        "91251A537",
        "9452K21",
    ]
    assert extract_part_numbers(["9125lA537"]) == []


def test_calibration_tiers_and_fit():
    cal = Calibration(temperature=0.05)
    p = cal.probabilities([0.9, 0.5, 0.4])
    assert abs(sum(p) - 1) < 1e-6 and p[0] > 0.99
    assert cal.tier(p, best_similarity=0.9) == MatchTier.EXACT
    assert cal.tier(cal.probabilities([0.6, 0.58, 0.57]), 0.6) in {
        MatchTier.CANDIDATE,
        MatchTier.LIKELY,
    }
    assert cal.tier(p, best_similarity=0.1) == MatchTier.UNKNOWN
    assert cal.tier([], 0.9, ocr_hit=True) == MatchTier.EXACT
    fitted = Calibration.fit_temperature([[0.9, 0.2], [0.7, 0.65]], [0, 1])
    assert fitted.temperature > 0


def test_attribute_consistency():
    part = Part(
        part_number="1",
        name="18-8 Stainless Steel Socket Head Screw",
        attributes={"material": "18-8 Stainless Steel", "drive_style": "Hex Socket"},
    )
    good = ExtractedAttributes(
        material_guess="stainless", head_type="socket head", visible_text=["18-8"]
    )
    bad = ExtractedAttributes(material_guess="brass")
    assert attribute_consistency(good, part)[0] > 0.5
    assert attribute_consistency(bad, part)[0] < 0
    assert attribute_consistency(None, part) == (0.0, [])


def test_fusion_reranker_uses_ocr_and_llm():
    parts = {
        "A": Part(part_number="A", name="a", category_path=["x", "y"]),
        "B": Part(part_number="B", name="b", category_path=["x", "z"]),
    }
    hits = [Hit("A", 0.8, 1, 0.1), Hit("B", 0.7, 1, 0.1)]
    plain = FusionReranker().rerank(hits, parts)
    assert [s.part.part_number for s in plain] == ["A", "B"]
    with_ocr = FusionReranker().rerank(hits, parts, ocr_part_numbers=["B"])
    assert with_ocr[0].part.part_number == "B" and "OCR" in " ".join(with_ocr[0].reasons)
    with_llm = FusionReranker().rerank(hits, parts, llm_ranking={"B": 0.95, "A": 0.05})
    assert with_llm[0].part.part_number == "B"


class _FakeBlock:
    type = "text"

    def __init__(self, text):
        self.text = text


class _FakeResponse:
    stop_reason = "end_turn"

    def __init__(self, text):
        self.content = [_FakeBlock(text)]


class _FakeMessages:
    def __init__(self, text):
        self.text = text
        self.calls = []

    def create(self, **kw):
        self.calls.append(kw)
        return _FakeResponse(self.text)


class _FakeClient:
    def __init__(self, text):
        self.beta = type("B", (), {})()
        self.beta.messages = _FakeMessages(text)


def test_claude_reranker_parses_structured_output(tmp_path):
    img = Image.new("RGB", (64, 64), (200, 200, 200))
    parts = [Part(part_number=pn, name=pn, image_paths=[]) for pn in ("A", "B")]
    scored = [Scored(p, 0.8, 0.8) for p in parts]
    verdict = {
        "ranking": [
            {"part_number": "B", "match_probability": 0.9, "reason": "hex head"},
            {"part_number": "ZZZ", "match_probability": 0.9, "reason": "not a candidate"},
        ],
        "extracted": {"material_guess": "brass", "visible_text": []},
        "none_match": False,
    }
    import json

    client = _FakeClient(json.dumps(verdict))
    rr = ClaudeVisionReranker(client=client)
    ranking, extracted, none = rr.rerank(img, scored)
    assert ranking == {"B": 0.9}
    assert extracted.material_guess == "brass" and none is False
    call = client.beta.messages.calls[0]
    assert call["model"] == "claude-opus-5"
    assert call["fallbacks"] == "default"
    assert call["output_config"]["format"]["type"] == "json_schema"
    assert call["messages"][0]["content"][1]["type"] == "image"


def test_claude_reranker_degrades_on_garbage():
    client = _FakeClient("not json")
    rr = ClaudeVisionReranker(client=client)
    scored = [Scored(Part(part_number="A", name="a"), 0.5, 0.5)]
    assert rr.rerank(Image.new("RGB", (8, 8)), scored) == ({}, None, False)
    assert rr.rerank(Image.new("RGB", (8, 8)), []) == ({}, None, False)
