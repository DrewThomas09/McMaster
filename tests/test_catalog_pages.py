from __future__ import annotations

from mcmaster_vision.catalog import CatalogStore, ingest
from mcmaster_vision.catalog.pages import parse_catalog_text, parse_page, split_pages
from mcmaster_vision.catalog.sources import CatalogSource

# OCR-style text modelled on the printed "Stainless Steel Pipe Fittings" page
PAGE = """CAD For technical drawings and 3-D models, go to mcmaster.com.   Stainless Steel Pipe Fittings
How to Measure Male Threaded Pipe and Fittings - Example shows pipe size 3/8.
Low-Pressure Stainless Steel Threaded Pipe Fittings
Elbows, Tees, Crosses, and Unions
The 90° female x male elbows are also known as street elbows.
Pipe Size     90° Elbows     90° Elbows, Female x Male     45° Elbows     Tees     Crosses     Unions
Type 304 Stainless Steel
1/8 ........ 4464K11 ... $4.90   4464K35 ... $6.94   4464K23 ... $6.48   4464K47 ... $6.91   4464K311 ... $11.98   4464K483 ... $13.57
1/4 ........ 4464K12 ..... 4.90  4464K36 ..... 6.94  4464K24 ..... 6.48  4464K49 ..... 6.91  4464K312 ..... 11.98  4464K484 ..... 14.12
1 1/4 ...... 4464K17 .... 24.32  4464K42 .... 34.29  4464K29 .... 23.98  4464K54 .... 37.14  4464K317 .... 52.95  4464K489 .... 49.59
Type 316 Stainless Steel
1/8 ........ 4452K411 ... 5.99   4452K471 ... 8.15   4452K421 ... 7.70   4452K431 ... 7.98   4452K481 ... 14.45   4452K222 ... 15.39
Couplings, Caps, Plugs, and Locknuts
Half couplings are partially threaded; one end has an unthreaded portion that can be welded.
Pipe Size     Couplings     Half Couplings     Caps     Square-Head Plugs     Hex-Head Plugs     Locknuts
Type 304 Stainless Steel
1/8 ........ 4464K351 ... $2.86  4464K364 ... $2.09  4464K496 ... $1.99  4464K231 ... $1.84  4464K251 ... $2.00  4464K581 ... $4.82
4
McMASTER-CARR
"""

NIPPLES = """Stainless Steel Pipe Nipples & Pipe
Standard-Wall Type 304/304L Stainless Steel Threaded Pipe Nipples and Pipe
FULLY THREADED
Pipe Size  Lg.
1/8 ....... 3/4" ...... 4830K111 ... $1.32
1/4 ....... 7/8" ...... 4830K131 ..... 1.50
THREADED ON BOTH ENDS
Pipe Size     1 1/2" Lengths     2" Lengths     2 1/2" Lengths
1/8 ........ 4830K112 ... $1.68   4830K113 ... $1.96   4830K114 ... $2.25
6
McMASTER-CARR
"""


def test_parse_fittings_page():
    parts = list(parse_page(PAGE, page_label="4"))
    by_pn = {p.part_number: p for p in parts}
    assert len(parts) == 30
    elbow = by_pn["4464K11"]
    assert elbow.attributes["pipe_size"] == "1/8" and elbow.attributes["price_usd"] == "4.90"
    assert elbow.attributes["material"] == "Type 304 Stainless Steel"
    assert elbow.attributes["fitting_type"] == "90° Elbows"
    assert elbow.category_path == [
        "Stainless Steel Pipe Fittings",
        "Elbows, Tees, Crosses, and Unions",
    ]
    assert by_pn["4464K483"].attributes["fitting_type"] == "Unions"
    assert by_pn["4464K17"].attributes["pipe_size"] == "1-1/4"
    assert by_pn["4452K411"].attributes["material"] == "Type 316 Stainless Steel"
    # look-alikes of one fitting type and material share a family; sizes differ
    assert elbow.family_id == by_pn["4464K12"].family_id == by_pn["4464K17"].family_id
    assert elbow.family_id != by_pn["4452K411"].family_id
    # the second table on the page has its own columns and section
    cap = by_pn["4464K496"]
    assert cap.attributes["fitting_type"] == "Caps"
    assert cap.category_path[1] == "Couplings, Caps, Plugs, and Locknuts"
    assert cap.attributes["catalog_page"] == "4"


def test_parse_nipples_and_whole_dump():
    parts = list(parse_page(NIPPLES))
    by_pn = {p.part_number: p for p in parts}
    assert by_pn["4830K111"].attributes["length"] == '3/4"'
    assert by_pn["4830K111"].attributes["pipe_size"] == "1/8"
    assert by_pn["4830K113"].attributes["length"] == '2"'
    assert by_pn["4830K113"].name.startswith("Threaded On Both Ends")
    assert by_pn["4830K111"].family_id == "Fully Threaded"
    dump = PAGE + "\f" + NIPPLES + "\f" + PAGE  # a repeated page must not duplicate
    assert len(split_pages(dump)) == 3
    all_parts = list(parse_catalog_text(dump, first_page=4))
    assert len(all_parts) == 30 + 5
    assert {p.attributes["catalog_page"] for p in all_parts} == {"4", "5"}


def test_import_pages_into_store_and_cli(tmp_path):
    txt = tmp_path / "catalog.txt"
    txt.write_text(PAGE + "\f" + NIPPLES, encoding="utf-8")
    from mcmaster_vision.catalog.pages import read_pages

    class Src(CatalogSource):
        def __iter__(self):
            return read_pages([txt])

        def __len__(self):
            return 35

    st = CatalogStore(tmp_path / "c.sqlite")
    stats = ingest(Src(), st)
    assert stats["parts"] == 35 and st.count() == 35
    hits = st.search_text("304 tees 1/8")
    assert any(p.part_number == "4464K47" for p in hits)
    from typer.testing import CliRunner

    from mcmaster_vision.cli import app

    env = {"MCV_CATALOG_DB": str(tmp_path / "c2.sqlite"), "MCV_DATA_DIR": str(tmp_path)}
    r = CliRunner().invoke(app, ["import-pages", str(txt), "--first-page", "4"], env=env)
    assert r.exit_code == 0, r.output
    assert "35" in r.output
    with CatalogStore(tmp_path / "c2.sqlite") as st2:
        assert st2.count() == 35 and st2.get("4830K111").attributes["catalog_page"] == "5"


def test_ocr_noise_is_normalised():
    from mcmaster_vision.catalog.pages import normalise_ocr

    assert normalise_ocr("⅛ ........ 4464K11 ... $4.90") == "1/8 ........ 4464K11 ... $4.90"
    assert normalise_ocr("1¼ ...... 4464K17 .... 24.32").startswith("1-1/4 ")
    assert normalise_ocr("3/8 ...... 4464KI3 .... S6.00") == "3/8 ...... 4464K13 .... $6.00"
    assert normalise_ocr("44O4Kl1 $1.00") == "4404K11 $1.00"
    assert normalise_ocr("Stainless Steel") == "Stainless Steel"  # words are left alone
    header = "Pipe Size  90° Elbows  Tees\nType 304 Stainless Steel\n"
    rows = "⅛ .. 4464K11 .. S4.90  4464K47 .. $6.91\n1¼ .. 4464KI7 .. 24.32  4464K54 .. 37.14\n"
    parts = {p.part_number: p for p in parse_page(header + rows)}
    assert parts["4464K11"].attributes["pipe_size"] == "1/8"
    assert parts["4464K17"].attributes["pipe_size"] == "1-1/4"
    assert parts["4464K54"].attributes["fitting_type"] == "Tees"


def test_parsers_never_crash_on_junk():
    import random
    import string

    from mcmaster_vision.pipeline.measure import parse_length_mm
    from mcmaster_vision.pipeline.pipe import normalise_pipe_size

    rng = random.Random(1)
    alphabet = string.printable + "⅛¼⅜½¾⅝″”°×–—"
    for _ in range(400):
        s = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 80)))
        parse_length_mm(s)
        normalise_pipe_size(s)
        list(parse_page(s))
