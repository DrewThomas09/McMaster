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


def test_dashboard_storage_and_backup_button(identifier, store, tmp_path):
    client = _client(identifier, tmp_path, api_token="s3cret")
    page = client.get("/dashboard").text
    assert "Storage" in page and "no backup yet" in page and "backupbtn" in page
    assert client.post("/admin/backup").status_code == 401
    r = client.post("/admin/backup", headers={"X-API-Token": "s3cret"})
    assert r.status_code == 200 and "queries" in r.json()["components"]
    assert client.get("/admin/backups", headers={"X-API-Token": "s3cret"}).json()[0]["bytes"] > 0
    st = client.get("/status").json()["storage"]
    assert st["last_backup"] is not None and st["backups"] == 1
    assert "last backup" in client.get("/dashboard").text


def test_service_worker_is_network_first(identifier, tmp_path):
    """A cached shell must never pin an old UI or a stale dashboard on the phone."""
    client = _client(identifier, tmp_path)
    r = client.get("/sw.js")
    assert r.status_code == 200 and r.headers["service-worker-allowed"] == "/"
    js = r.text
    assert "fetch(e.request).then" in js and "catch(() => caches.match" in js
    assert "startsWith('/static/')" in js and "/dashboard" not in js


def test_part_page_shows_pipe_dimensions_and_compatibility(identifier, store, tmp_path):
    from mcmaster_vision.schemas import Part

    store.upsert(
        [
            Part(
                part_number="4464K13",
                name="Type 304 Stainless Steel, 90° Elbows, 3/8 pipe size, NPT",
                category_path=["Pipe Fittings"],
                attributes={"pipe_size": "3/8", "material": "Type 304 Stainless Steel"},
            )
        ]
    )
    client = _client(identifier, tmp_path)
    page = client.get("/part/4464K13").text
    assert "0.675 in" in page and "0.493 in" in page and "18 NPT" in page and "19 BSP" in page
    assert "NPSM" in page  # NPT male fits female NPSM per the compatibility table
