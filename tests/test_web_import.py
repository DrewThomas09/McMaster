from __future__ import annotations

import io
from pathlib import Path

import httpx
from PIL import Image

from mcmaster_vision.catalog import CatalogStore, FlatImageSource, ingest, open_source
from mcmaster_vision.catalog.web import (
    McMasterParser,
    RobotsPolicy,
    WebImporter,
    WebSource,
    read_items,
)

FIXTURE = Path(__file__).parent / "fixtures" / "mcmaster_product.html"


def _png_bytes(seed: int = 0) -> bytes:
    import numpy as np

    buf = io.BytesIO()
    arr = np.random.default_rng(seed).integers(0, 255, size=(96, 96, 3), dtype=np.uint8)
    Image.fromarray(arr).save(
        buf, format="PNG"
    )  # noisy so it is well above the tracking-pixel size limit
    return buf.getvalue()


def test_parser_extracts_product_fields():
    data = McMasterParser().parse("https://www.mcmaster.com/91251A537/", FIXTURE.read_text())
    assert data.part_number == "91251A537"
    assert data.name == "Alloy Steel Socket Head Screw"
    assert data.category_path == ["Fastening & Joining", "Screws & Bolts", "Socket Head Screws"]
    assert data.attributes["Thread Size"] == '1/4"-20'
    assert data.attributes["Material"] == "Black-Oxide Alloy Steel"  # from the spec table
    assert data.image_urls[0].endswith("636909677602138926.png")
    assert any(u.endswith("91251A537p2.png") for u in data.image_urls)
    assert not any("logo" in u for u in data.image_urls)
    assert len(data.image_urls) == len(set(data.image_urls))


def test_parser_falls_back_to_title_and_url():
    html = "<html><head><title>9452K21 | Hex Nut | McMaster-Carr</title></head><body></body></html>"
    data = McMasterParser().parse("https://www.mcmaster.com/9452K21/", html)
    assert data.part_number == "9452K21" and data.name == "9452K21 | Hex Nut"


def test_robots_policy():
    pol = RobotsPolicy.parse(
        "User-agent: *\nDisallow: /private/\nDisallow: /cart\n\nUser-agent: other\nDisallow: /\n"
    )
    assert (
        pol.allowed("https://x/91251A537/")
        and not pol.allowed("https://x/private/a")
        and not pol.allowed("https://x/cart")
    )


def _mock_client(calls: list[str]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        path = request.url.path
        if path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nDisallow: /cart\n")
        if path.startswith("/91251A537"):
            return httpx.Response(200, text=FIXTURE.read_text())
        if path.startswith("/NOPE"):
            return httpx.Response(404)
        if path.endswith(".png"):
            return httpx.Response(200, content=_png_bytes(), headers={"content-type": "image/png"})
        return httpx.Response(500)

    return httpx.Client(transport=httpx.MockTransport(handler), base_url="https://www.mcmaster.com")


def test_importer_end_to_end(tmp_path):
    calls: list[str] = []
    imp = WebImporter(tmp_path / "img", delay_s=0, client=_mock_client(calls))
    part = imp.import_one("91251A537")
    assert part is not None and part.part_number == "91251A537"
    assert len(part.image_paths) == 2 and all(Path(p).exists() for p in part.image_paths)
    assert part.url == "https://www.mcmaster.com/91251A537/"
    assert imp.import_one("NOPE1234") is None
    # second import is served from cache: no new page/image requests
    n = len(calls)
    imp.import_one("https://www.mcmaster.com/91251A537/")
    assert len(calls) == n

    store = CatalogStore(":memory:")
    stats = ingest(WebSource(imp, ["91251A537", "# comment", "NOPE1234"]), store)
    assert (
        stats["written"] == 1 and store.get("91251A537").category_path[-1] == "Socket Head Screws"
    )


def test_importer_respects_robots(tmp_path):
    calls: list[str] = []
    imp = WebImporter(tmp_path / "img", delay_s=0, client=_mock_client(calls))
    assert imp.fetch_page("https://www.mcmaster.com/cart") is None
    assert not any(u.endswith("/cart") for u in calls)


def test_read_items_and_flat_image_source(tmp_path):
    (tmp_path / "list.txt").write_text("91251A537\n# note\nhttps://www.mcmaster.com/9452K21/\n")
    assert read_items(tmp_path / "list.txt") == ["91251A537", "https://www.mcmaster.com/9452K21/"]
    shots = tmp_path / "shots"
    shots.mkdir()
    for name in (
        "91251A537.png",
        "91251A537_2.png",
        "9452K21 screenshot.png",
        "notes.txt",
        "random.png",
    ):
        (shots / name).write_bytes(_png_bytes() if name.endswith(".png") else b"x")
    (shots / "meta.jsonl").write_text(
        '{"part_number": "9452K21", "name": "Hex Nut", "category_path": "Fastening & Joining > Nuts"}\n'
    )
    src = open_source(shots)
    assert isinstance(src, FlatImageSource) and len(src) == 2
    parts = {p.part_number: p for p in src}
    assert len(parts["91251A537"].image_paths) == 2
    assert parts["9452K21"].name == "Hex Nut" and parts["9452K21"].category_path == [
        "Fastening & Joining",
        "Nuts",
    ]
