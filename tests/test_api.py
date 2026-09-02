from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from mcmaster_vision.api import create_app
from mcmaster_vision.config import Settings


@pytest.fixture(scope="module")
def client(identifier):
    app = create_app(Settings(max_upload_mb=1), identifier=identifier)
    return TestClient(app)


def test_health_and_stats(client):
    assert client.get("/health").json()["status"] == "ok"
    st = client.get("/stats").json()
    assert st["parts"] == 40 and st["backend"] == "numpy"


def test_ui_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "McMaster-Vision" in r.text


def test_identify_endpoint(client, store):
    part = next(store.iter_parts(with_images_only=True))
    buf = io.BytesIO()
    Image.open(part.image_paths[0]).save(buf, format="PNG")
    r = client.post("/identify?top_n=3", files={"file": ("q.png", buf.getvalue(), "image/png")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["candidates"][0]["part_number"] == part.part_number
    assert len(body["candidates"]) == 3


def test_identify_bad_upload(client):
    r = client.post("/identify", files={"file": ("q.png", b"garbage", "image/png")})
    assert r.status_code == 400
    r = client.post("/identify", files={"file": ("q.png", b"", "image/png")})
    assert r.status_code == 400
    r = client.post("/identify", files={"file": ("q.png", b"x" * (2 * 1024 * 1024), "image/png")})
    assert r.status_code == 413


def test_parts_and_search(client, store):
    part = next(store.iter_parts())
    r = client.get(f"/parts/{part.part_number}")
    assert r.status_code == 200 and r.json()["name"] == part.name
    assert client.get(f"/parts/{part.part_number}/image").status_code == 200
    assert client.get("/parts/NOPE").status_code == 404
    r = client.get("/search", params={"q": part.part_number})
    assert r.status_code == 200 and any(p["part_number"] == part.part_number for p in r.json())
