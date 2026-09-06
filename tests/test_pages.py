from __future__ import annotations

from fastapi.testclient import TestClient

from mcmaster_vision.api import create_app
from mcmaster_vision.config import Settings


def _client(identifier, tmp_path, **kw):
    return TestClient(
        create_app(
            Settings(data_dir=tmp_path, queries_dir=tmp_path / "q", **kw), identifier=identifier
        )
    )


def test_theme_and_layout_served(identifier, tmp_path):
    client = _client(identifier, tmp_path)
    css = client.get("/static/theme.css")
    assert css.status_code == 200 and "--green" in css.text
    home = client.get("/").text
    assert (
        'href="/static/theme.css"' in home
        and "mc-header" in home
        and "not affiliated" in home.lower()
    )
    assert "Browse" in home and "Dashboard" in home


def test_browse_pages(identifier, store, tmp_path):
    client = _client(identifier, tmp_path)
    top = client.get("/browse")
    assert top.status_code == 200 and "Browse the catalog" in top.text and "chip" in top.text
    part = next(store.iter_parts(with_images_only=True))
    cat = " > ".join(part.category_path[:2])
    page = client.get("/browse", params={"category": cat}).text
    assert part.part_number in page or "next" in page
    deep = client.get("/browse", params={"category": " > ".join(part.category_path)}).text
    assert part.part_number in deep and f"/part/{part.part_number}" in deep


def test_part_page_and_family(identifier, store, tmp_path):
    client = _client(identifier, tmp_path, demo_mode=True)
    part = next(store.iter_parts(with_images_only=True))
    r = client.get(f"/part/{part.part_number}")
    assert r.status_code == 200
    assert (
        part.part_number in r.text
        and "Specifications" in r.text
        and f"/parts/{part.part_number}/thumb" in r.text
    )
    assert "Identify a photo-style render" in r.text  # demo mode link
    fam = store.family(part.family_id)
    if len(fam) > 1:
        assert "Look-alike SKUs" in r.text
    assert client.get("/part/NOPE").status_code == 404


def test_dashboard_page(identifier, store, tmp_path):
    import io

    from PIL import Image

    client = _client(identifier, tmp_path)
    part = next(store.iter_parts(with_images_only=True))
    buf = io.BytesIO()
    Image.open(part.image_paths[0]).save(buf, format="PNG")
    assert (
        client.post("/identify", files={"file": ("a.png", buf.getvalue(), "image/png")}).status_code
        == 200
    )
    page = client.get("/dashboard").text
    assert (
        "Dashboard" in page
        and "identifications" in page
        and part.part_number in page
        and "Recent identifications" in page
    )


def test_search_category_filter(identifier, store, tmp_path):
    client = _client(identifier, tmp_path)
    part = next(store.iter_parts(with_images_only=True))
    cat = " > ".join(part.category_path)
    r = client.get("/search", params={"category": cat, "limit": 5})
    assert (
        r.status_code == 200
        and all(p["category_path"] == part.category_path for p in r.json())
        and r.json()
    )
    r = client.get("/search", params={"q": part.part_number, "category": "Nonexistent"})
    assert r.status_code == 200 and r.json() == []
    assert client.get("/search").status_code == 400


def test_pages_when_not_ready(tmp_path):
    client = TestClient(
        create_app(Settings(data_dir=tmp_path, queries_dir=tmp_path / "q", warm_up=False))
    )
    assert "Nothing is built yet" in client.get("/browse").text
    assert client.get("/dashboard").status_code == 200
