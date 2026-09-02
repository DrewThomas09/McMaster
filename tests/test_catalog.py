from __future__ import annotations

import csv

from mcmaster_vision.catalog import CatalogStore, CSVSource, JSONLSource, open_source
from mcmaster_vision.catalog.sources import part_from_record
from mcmaster_vision.schemas import Part


def test_jsonl_source_roundtrip(jsonl_path):
    parts = list(JSONLSource(jsonl_path))
    assert len(parts) == 40
    assert all(p.image_paths for p in parts)
    assert parts[0].category_path[0] in {
        "Fastening & Joining",
        "Power Transmission",
        "Hardware",
        "Sealing",
    }


def test_csv_source_extra_columns_become_attributes(tmp_path):
    path = tmp_path / "parts.csv"
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["part_number", "name", "category_path", "thread_size", "image_paths"])
        w.writerow(
            ["91251A537", "Socket Head Screw", "Fastening & Joining > Screws", '1/4"-20', ""]
        )
    parts = list(open_source(path))
    assert parts[0].part_number == "91251A537"
    assert parts[0].attributes["thread_size"] == '1/4"-20'
    assert parts[0].category_path == ["Fastening & Joining", "Screws"]
    assert isinstance(CSVSource(path), CSVSource)


def test_part_from_record_parses_json_attributes():
    p = part_from_record(
        {"part_number": "1", "name": "x", "attributes": '{"a": 1}', "images": "a.jpg;b.jpg"}
    )
    assert p.attributes == {"a": 1}
    assert len(p.image_paths) == 2


def test_store_get_many_search_and_family(store):
    parts = list(store.iter_parts())
    assert store.count() == 40
    sample = parts[:3]
    got = store.get_many(p.part_number for p in sample)
    assert set(got) == {p.part_number for p in sample}
    fam = store.family(sample[0].family_id)
    assert sample[0].part_number in {p.part_number for p in fam}
    hits = store.search_text(sample[0].part_number)
    assert any(h.part_number == sample[0].part_number for h in hits)
    hits = store.search_text("Hex Nut")
    assert all("Hex Nut" in h.name for h in hits) or hits == []


def test_store_upsert_replaces():
    st = CatalogStore(":memory:")
    st.upsert([Part(part_number="A1", name="one")])
    st.upsert([Part(part_number="A1", name="two")])
    assert st.count() == 1
    assert st.get("A1").name == "two"
    assert st.get("nope") is None


def test_taxonomy(store):
    tax = store.taxonomy()
    assert len(tax) > 0
    assert tax.children()  # top-level
    for leaf in tax.leaves():
        assert tax.count(leaf) >= 1
