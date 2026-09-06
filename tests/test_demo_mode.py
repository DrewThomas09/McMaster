from __future__ import annotations

from fastapi.testclient import TestClient
from PIL import Image

from mcmaster_vision.api import create_app
from mcmaster_vision.config import Settings


def _client(identifier, tmp_path, demo=True):
    return TestClient(
        create_app(
            Settings(data_dir=tmp_path, queries_dir=tmp_path / "q", demo_mode=demo),
            identifier=identifier,
        )
    )


def test_demo_endpoints(identifier, tmp_path):
    client = _client(identifier, tmp_path)
    assert client.get("/health").json()["demo_mode"] is True
    samples = client.get("/demo/samples?n=6&seed=1").json()
    assert 1 <= len(samples) <= 6 and all(s["thumb"].startswith("/parts/") for s in samples)
    assert len({s["part_number"] for s in samples}) == len(samples)
    pn = samples[0]["part_number"]
    r = client.get(f"/demo/query/{pn}?seed=3")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    import io

    assert Image.open(io.BytesIO(r.content)).size == (512, 512)
    r = client.post(f"/demo/try/{pn}?seed=3&top_n=5&tta=fast")
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["truth"] == pn and (d["rank"] is None or 1 <= d["rank"] <= 5)
    assert d["result"]["candidates"] and d["query_image"].endswith("seed=3")
    assert client.post("/demo/try/NOPE1").status_code == 404
    sheet = client.get("/demo/sheet?n=4&seed=2").text
    assert (
        "demo sheet" in sheet.lower()
        and sheet.count("<figure>") == 4
        and samples[0]["part_number"] in client.get("/demo/sheet?n=6&seed=1").text
    )
    assert "&nbsp;" in client.get("/demo/sheet?n=4&key=false").text


def test_demo_endpoints_off_by_default(identifier, tmp_path):
    client = _client(identifier, tmp_path, demo=False)
    assert client.get("/demo/samples").status_code == 404
    assert client.get("/health").json()["demo_mode"] is False


def test_connect_page(identifier, tmp_path):
    client = _client(identifier, tmp_path)
    r = client.get("/connect")
    assert (
        r.status_code == 200
        and "Open on your phone" in r.text
        and ("<svg" in r.text or "qrcode" in r.text)
    )


def test_cors_when_configured(identifier, tmp_path):
    client = TestClient(
        create_app(
            Settings(
                data_dir=tmp_path, queries_dir=tmp_path / "q", cors_origins="https://app.example"
            ),
            identifier=identifier,
        )
    )
    r = client.options(
        "/health", headers={"Origin": "https://app.example", "Access-Control-Request-Method": "GET"}
    )
    assert r.headers.get("access-control-allow-origin") == "https://app.example"
