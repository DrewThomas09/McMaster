from __future__ import annotations

import io
import json

import httpx
from PIL import Image
from typer.testing import CliRunner

from mcmaster_vision.catalog.intake import (
    fetch_image_urls,
    normalise_parts,
    prepare_image,
    read_records,
    validate_source,
    write_jsonl,
)
from mcmaster_vision.cli import app
from mcmaster_vision.index.builder import choose_backend
from mcmaster_vision.schemas import Part


def _png(size=(300, 200), color=(120, 120, 120), mode="RGB") -> bytes:
    buf = io.BytesIO()
    Image.new(mode, size, color).save(buf, format="PNG")
    return buf.getvalue()


def test_validate_source_reports_problems(tmp_path, jsonl_path):
    rep = validate_source(jsonl_path)
    assert rep.ok() and rep.parts == 40 and rep.with_images == 40 and rep.images == 80
    assert rep.families > 0 and rep.categories > 0
    bad = tmp_path / "bad.jsonl"
    (tmp_path / "corrupt.png").write_bytes(b"not an image")
    (tmp_path / "tiny.png").write_bytes(_png((20, 20)))
    rows = [
        {
            "part_number": "A1",
            "name": "A1",
            "image_paths": [str(tmp_path / "missing.png"), str(tmp_path / "corrupt.png")],
        },
        {"part_number": "A1", "name": "dup", "image_paths": [str(tmp_path / "tiny.png")]},
        {"part_number": "A2", "name": "A2", "category_path": [], "image_paths": []},
    ]
    bad.write_text("\n".join(json.dumps(r) for r in rows))
    rep = validate_source(bad)
    assert not rep.ok()
    assert (
        rep.duplicate_part_numbers == 1
        and rep.missing_files == 1
        and rep.unreadable == 1
        and rep.tiny_images == 1
    )
    assert rep.without_images == 2 and rep.without_name == 2 and rep.without_category == 3
    assert rep.examples["missing_files"]


def test_prepare_image_and_normalise(tmp_path):
    src = tmp_path / "big.png"
    src.write_bytes(_png((3000, 1500), (200, 30, 30, 255), mode="RGBA"))
    out = prepare_image(src, tmp_path / "out", stem="X")
    assert out is not None and out.suffix == ".jpg"
    with Image.open(out) as im:
        assert max(im.size) == 1024 and im.mode == "RGB"
    assert prepare_image(tmp_path / "nope.png", tmp_path / "out", stem="Y") is None
    dup = tmp_path / "dup.png"
    dup.write_bytes(src.read_bytes())
    parts = list(
        normalise_parts(
            [
                Part(
                    part_number="P1",
                    name="p",
                    image_paths=[str(src), str(dup), str(tmp_path / "missing.png")],
                )
            ],
            tmp_path / "img",
        )
    )
    assert len(parts[0].image_paths) == 1  # duplicate and missing dropped


def test_fetch_image_urls_with_mock_client(tmp_path):
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if request.url.path.endswith("bad.png"):
            return httpx.Response(404)
        return httpx.Response(200, content=_png(), headers={"content-type": "image/png"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    records = [
        {
            "part_number": "91251A537",
            "name": "screw",
            "image_urls": "https://x/a.png;https://x/bad.png",
        },
        {"part_number": "9452K21", "name": "nut", "image_urls": ["https://x/b.jpg"]},
    ]
    out = list(fetch_image_urls(records, tmp_path / "img", client=client, delay_s=0))
    assert len(out[0]["image_paths"]) == 1 and len(out[1]["image_paths"]) == 1
    assert "image_urls" not in out[0]
    n = len(calls)
    list(
        fetch_image_urls(records, tmp_path / "img", client=client, delay_s=0)
    )  # resumable: cached files
    assert len(calls) == n + 1  # only the 404 is retried
    path = tmp_path / "out.jsonl"
    assert write_jsonl(out, path) == 2 and [r["part_number"] for r in read_records(path)] == [
        "91251A537",
        "9452K21",
    ]


def test_choose_backend():
    assert choose_backend(10, None) == "numpy"
    assert choose_backend(10, "faiss") == "faiss"
    assert choose_backend(10_000_000, "numpy") == "numpy"
    assert choose_backend(10_000_000, "auto") in ("faiss", "numpy")


def test_bootstrap_status_and_incremental_index(tmp_path, jsonl_path):
    runner = CliRunner()
    env = {
        "MCV_DATA_DIR": str(tmp_path),
        "MCV_CATALOG_DB": str(tmp_path / "catalog.sqlite"),
        "MCV_INDEX_DIR": str(tmp_path / "index"),
        "MCV_MODEL_DIR": str(tmp_path / "models"),
        "MCV_QUERIES_DIR": str(tmp_path / "queries"),
        "MCV_BACKBONE": "hash",
        "MCV_INDEX_GALLERY_AUGMENT": "0",
    }
    r = runner.invoke(app, ["validate", str(jsonl_path), "--no-image-check"])
    assert r.exit_code == 0, r.output
    r = runner.invoke(app, ["bootstrap", str(jsonl_path), "--max-eval-queries", "10"], env=env)
    assert r.exit_code == 0, r.output
    assert "Recall@1" in r.output and (tmp_path / "manifest.json").exists()
    assert (tmp_path / "models" / "calibration.json").exists()
    assert any((tmp_path / "images" / "catalog").iterdir())
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["parts"] == 40 and manifest["index"]["parts"] == 40

    r = runner.invoke(app, ["status"], env=env)
    assert r.exit_code == 0 and '"ready": true' in r.output

    # add one part, extend the index incrementally
    from mcmaster_vision.catalog import CatalogStore
    from mcmaster_vision.index import load_index

    src = tmp_path / "images" / "catalog"
    first = next(src.iterdir())
    with CatalogStore(tmp_path / "catalog.sqlite") as store:
        store.upsert(
            [
                Part(
                    part_number="NEW1",
                    name="new",
                    category_path=["Hardware"],
                    image_paths=[str(next(first.iterdir()))],
                )
            ]
        )
    before = len(load_index(tmp_path / "index" / "parts"))
    r = runner.invoke(app, ["build-index", "--only-new"], env=env)
    assert r.exit_code == 0, r.output
    idx = load_index(tmp_path / "index" / "parts")
    assert len(idx) == before + 1 and "NEW1" in idx.ids
