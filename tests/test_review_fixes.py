"""Regressions for the bug-hunt round: catalog, index, model and pipeline edge cases."""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from mcmaster_vision.catalog import CatalogStore
from mcmaster_vision.catalog.sources import CSVSource, open_source
from mcmaster_vision.catalog.web import McMasterParser, RobotsPolicy, WebImporter
from mcmaster_vision.index.base import load_index
from mcmaster_vision.index.numpy_index import NumpyIndex
from mcmaster_vision.pipeline.calibration import Calibration
from mcmaster_vision.pipeline.feedback import FeedbackStore
from mcmaster_vision.pipeline.preprocess import preprocess
from mcmaster_vision.schemas import Part


def _part(pn, **kw):
    base = dict(name=f"part {pn}", category_path=["Fasteners", "Screws"], attributes={})
    base.update(kw)
    return Part(part_number=pn, **base)


def test_search_survives_quote_only_and_prefix_wildcards(tmp_path):
    st = CatalogStore(tmp_path / "c.sqlite")
    st.upsert([_part("91251A537"), _part("A_B1"), _part("AXB1")])
    assert st.search_text('"') == []
    assert st.search_text('" "') == []
    # LIKE metacharacters in the prefix search are literal
    assert [p.part_number for p in st.search_text("A_B")] == ["A_B1"]
    assert st.search_text("%") == []


def test_fts_delete_uses_the_index_and_keeps_other_rows(tmp_path):
    st = CatalogStore(tmp_path / "c.sqlite")
    st.upsert([_part("1/4-20X1", name="hex bolt"), _part("1/4-20X2", name="hex bolt long")])
    st.upsert([_part("1/4-20X1", name="hex bolt renamed")])  # re-upsert: delete + insert
    assert {p.part_number for p in st.search_text("hex bolt")} == {"1/4-20X1", "1/4-20X2"}
    assert st.get("1/4-20X1").name == "hex bolt renamed"
    n = st._conn.execute("SELECT COUNT(*) FROM parts_fts").fetchone()[0]
    assert n == 2  # no duplicate FTS rows after the re-upsert


def test_merge_upsert_keeps_images_and_name(tmp_path):
    st = CatalogStore(tmp_path / "c.sqlite")
    img = tmp_path / "a.png"
    Image.new("RGB", (8, 8)).save(img)
    st.upsert([_part("X1", name="Hex nut", image_paths=[str(img)], attributes={"a": "1"})])
    st.upsert([_part("X1", name="X1", attributes={"b": "2"})], merge=True)
    p = st.get("X1")
    assert p.image_paths == [str(img)] and p.name == "Hex nut"
    assert p.attributes == {"a": "1", "b": "2"}
    st.upsert([_part("X1", name="X1")])  # plain upsert still replaces
    assert st.get("X1").image_paths == []


def test_csv_with_bom_and_ragged_rows(tmp_path):
    csv = tmp_path / "parts.csv"
    csv.write_bytes(
        b"\xef\xbb\xbfpart_number,name,category,length\r\n"
        b'91251a537,Socket screw,Fasteners > Screws,1"\r\n'
        b'92196A542,Flat washer,Fasteners > Washers,"1/2""",extra,fields\r\n'
    )
    parts = list(CSVSource(csv))
    assert [p.part_number for p in parts] == ["91251A537", "92196A542"]
    assert parts[0].attributes["length"] == '1"' and parts[0].category_path == [
        "Fasteners",
        "Screws",
    ]
    assert None not in parts[1].attributes


def test_flat_folder_with_clutter_is_still_flat(tmp_path):
    root = tmp_path / "drop"
    root.mkdir()
    (root / "__MACOSX").mkdir()
    (root / ".cache").mkdir()
    Image.new("RGB", (8, 8)).save(root / "91251A537.png")
    src = open_source(root)
    parts = list(src)
    assert [p.part_number for p in parts] == ["91251A537"]


def test_robots_wildcards_and_stacked_agents():
    pol = RobotsPolicy.parse(
        "User-agent: googlebot\nUser-agent: *\nDisallow: /*?\nDisallow: /search$\nDisallow: /cart\n"
    )
    assert not pol.allowed("https://x.com/p/1?q=1")
    assert pol.allowed("https://x.com/p/1")
    assert not pol.allowed("https://x.com/search")
    assert pol.allowed("https://x.com/searching")
    assert not pol.allowed("https://x.com/cart/2")
    assert RobotsPolicy.parse("User-agent: other\nDisallow: /\n").allowed("https://x.com/a")


def test_jsonld_shapes_that_used_to_crash():
    html = """<html><head><script type="application/ld+json">
    {"@type": "Product", "name": "Hex Nut", "sku": "90480A005",
     "image": {"@type": "ImageObject", "url": "/img/nut.jpg"}}
    </script><script type="application/ld+json">
    {"@type": "BreadcrumbList", "itemListElement": ["junk", {"position": "2", "name": "Nuts"},
     {"position": 1, "item": {"name": "Fasteners"}}]}
    </script></head><body></body></html>"""
    data = McMasterParser().parse("https://www.mcmaster.com/90480A005/", html)
    assert data.image_urls == ["https://www.mcmaster.com/img/nut.jpg"]
    assert data.category_path == ["Fasteners", "Nuts"]


def test_importer_refuses_unsafe_part_numbers(tmp_path):
    class Resp:
        status_code = 200
        headers = {"content-type": "text/html"}

        def __init__(self, url):
            self.url = url
            self.content = (
                b'<html><head><script type="application/ld+json">{"@type":"Product",'
                b'"name":"Evil","sku":"../../outside"}</script></head></html>'
            )

    class Client:
        def get(self, url):
            return Resp(url)

    imp = WebImporter(tmp_path / "img", delay_s=0, client=Client(), respect_robots=False)
    assert imp.import_one("https://www.mcmaster.com/evil/") is None
    assert not (tmp_path / "outside_0.jpg").exists() and not list(tmp_path.glob("**/outside*"))


def test_load_refuses_inconsistent_index(tmp_path):
    idx = NumpyIndex(4)
    idx.add(["a", "b", "c"], np.eye(4, dtype=np.float32)[:3])
    idx.save(tmp_path / "parts")
    (tmp_path / "parts" / "ids.json").write_text(json.dumps(["a", "b"]))
    with pytest.raises(ValueError, match="inconsistent"):
        load_index(tmp_path / "parts")


def test_faiss_empty_index_returns_nothing():
    pytest.importorskip("faiss")
    from mcmaster_vision.index.faiss_index import FaissIndex

    idx = FaissIndex(8)
    scores, rows = idx.search(np.zeros((1, 8), np.float32), 5)
    assert scores.shape == (1, 0) and rows.shape == (1, 0)
    assert idx.search_ids(np.zeros(8, np.float32), 5) == []
    with pytest.raises(ValueError):
        idx.add(["a", "b"], np.zeros((3, 8), np.float32))


def test_ensemble_rejects_recursion_and_weight_mismatch():
    from mcmaster_vision.config import Settings
    from mcmaster_vision.models.backbone import load_backbone

    with pytest.raises(ValueError):
        load_backbone(Settings(backbone="ensemble", ensemble_members="ensemble,hash"))
    with pytest.raises(ValueError):
        load_backbone(
            Settings(backbone="ensemble", ensemble_members="hash,hash", ensemble_weights="1")
        )


def test_calibration_keeps_likely_below_exact():
    cal = Calibration(exact_threshold=0.9, likely_threshold=0.6)
    # the exact search succeeds at a low threshold; nothing is found for likely
    scores = [[1.0, 0.2, 0.1]] * 12
    new = cal.fit_thresholds(scores, [0] * 12, exact_precision=0.5, likely_precision=1.01)
    assert new.likely_threshold <= new.exact_threshold


def test_grayscale_query_and_lowercase_feedback_folders(tmp_path):
    out = preprocess(Image.new("L", (120, 90), 128), size=64)
    assert out.mode == "RGB" and out.size == (64, 64)
    fs = FeedbackStore(tmp_path / "q")
    (tmp_path / "q" / "91251a537").mkdir()
    Image.new("RGB", (8, 8)).save(tmp_path / "q" / "91251a537" / "hand.jpg")
    assert list(fs.labelled_images()) == ["91251A537"]


def test_synthetic_rejects_unknown_kinds():
    from mcmaster_vision.data import SyntheticCatalog

    with pytest.raises(ValueError, match="unknown part kinds"):
        SyntheticCatalog(n_parts=2, kinds=["hex_nut", "unobtainium"])
