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


def test_pipe_and_page_review_regressions(tmp_path):
    from mcmaster_vision.catalog.pages import parse_page, read_pages
    from mcmaster_vision.pipeline.measure import Measurement, size_consistency
    from mcmaster_vision.pipeline.pipe import normalise_pipe_size

    # gender: "female" must not read as male; a 1/2 female coupling is not a 3/8 one
    m = Measurement(40.0, 28.0, 0.1)  # a 1/2 female coupling body
    fem14 = _part("F14", name="Coupling, female NPT", attributes={"pipe_size": "1/4"})
    fem12 = _part("F12", name="Coupling, female NPT", attributes={"pipe_size": "1/2"})
    assert size_consistency(m, fem12)[0] == 1.0 and size_consistency(m, fem14)[0] < 0.5
    assert "female" in size_consistency(m, fem12)[1][0]
    # parser guards
    assert normalise_pipe_size("1/0") is None and normalise_pipe_size("10/0") is None
    assert normalise_pipe_size("1/2 NPTF") == "1/2" and normalise_pipe_size("1/2 in.") == "1/2"
    hdr = "Pipe Size   90° Elbows   45° Elbows   Tees\nType 304 Stainless Steel\n"
    # a wrapped row: the continuation starts with a price, which is not a size
    parts = {
        p.part_number: p
        for p in parse_page(
            hdr + "1/8 ..... 4464K11 ... $4.90   4464K35 ...\n6.94   4464K23 ... $6.48\n"
        )
    }
    assert "4464K23" not in parts and parts["4464K11"].attributes["pipe_size"] == "1/8"
    # interleaved two-column rows keep their own sizes
    parts = {
        p.part_number: p
        for p in parse_page(hdr + "1/8 ..... 4464K11 ... $4.90   1/4 ..... 4464K12 ... $4.90\n")
    }
    assert parts["4464K12"].attributes["pipe_size"] == "1/4"
    # a placeholder cell does not shift later columns
    parts = {
        p.part_number: p
        for p in parse_page(hdr + "1/8 ..... 4464K11 ... $4.90   —   4464K47 ... $6.91\n")
    }
    assert parts["4464K47"].attributes["fitting_type"] == "Tees"
    # a dimension is not a price
    parts = {p.part_number: p for p in parse_page(hdr + "1/8 ..... 4464K11 ... 0.675\n")}
    assert not parts
    # (cont.) tables keep the family and the previous header
    text = (
        hdr
        + "1/8 ..... 4464K11 ... $4.90\nElbows, Tees, Crosses, and Unions (cont.)\n1/4 ..... 4464K12 ... $4.90\n"
    )
    parts = {p.part_number: p for p in parse_page("Elbows, Tees, Crosses, and Unions\n" + text)}
    assert parts["4464K11"].family_id == parts["4464K12"].family_id
    assert parts["4464K12"].attributes["fitting_type"] == "90° Elbows"
    # a page title starting with a material is a heading, not a material
    parts = {
        p.part_number: p
        for p in parse_page("Brass Pipe Fittings\n" + hdr + "1/8 ..... 4464K11 ... $4.90\n")
    }
    assert parts["4464K11"].attributes["material"] == "Type 304 Stainless Steel"
    # a header without "Pipe Size" is still a header
    parts = {
        p.part_number: p
        for p in parse_page(
            "Thread   90° Elbows   Tees\n1/8 ..... 4464K11 ... $4.90   4464K47 ... $6.91\n"
        )
    }
    assert parts["4464K47"].attributes.get("fitting_type") == "Tees"
    # one file per page: page numbers run on, duplicates across files are dropped
    (tmp_path / "p4.txt").write_text(hdr + "1/8 ..... 4464K11 ... $4.90\n")
    (tmp_path / "p5.txt").write_text(
        hdr + "1/8 ..... 4464K11 ... $4.90   1/4 ..... 4464K12 ... $4.90\n"
    )
    got = list(read_pages([tmp_path / "p4.txt", tmp_path / "p5.txt"], first_page=4))
    assert [p.part_number for p in got] == ["4464K11", "4464K12"]
    assert got[1].attributes["catalog_page"] == "5"


def test_round_part_alone_is_not_offered_as_a_coin():
    from PIL import Image, ImageDraw

    from mcmaster_vision.pipeline.reference import find_coin

    im = Image.new("RGB", (640, 480), (238, 236, 230))
    ImageDraw.Draw(im).ellipse((240, 160, 400, 320), fill=(150, 150, 155))  # a washer, alone
    assert find_coin(im) is None
    ImageDraw.Draw(im).rectangle((40, 220, 200, 260), fill=(60, 60, 65))  # now with a part
    assert find_coin(im) is not None
