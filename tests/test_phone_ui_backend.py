from __future__ import annotations

import io
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from mcmaster_vision.api import create_app
from mcmaster_vision.api.app import lan_urls, self_signed_cert
from mcmaster_vision.config import Settings


def test_thumbs_images_and_tta_modes(identifier, store, tmp_path):
    client = TestClient(
        create_app(Settings(data_dir=tmp_path, queries_dir=tmp_path / "q"), identifier=identifier)
    )
    part = next(store.iter_parts(with_images_only=True))
    pn = part.part_number
    r = client.get(f"/parts/{pn}/thumb?size=96")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert Image.open(io.BytesIO(r.content)).size == (96, 96)
    assert (tmp_path / "cache" / "thumbs" / f"{pn}_0_96.jpg").exists()  # cached on disk
    assert "max-age" in r.headers.get("cache-control", "")
    lst = client.get(f"/parts/{pn}/images").json()
    assert (
        lst["count"] == len(part.image_paths) and client.get(lst["images"][-1]).status_code == 200
    )
    assert client.get(f"/parts/{pn}/image?i=99").status_code == 404
    assert client.get("/parts/NOPE/images").status_code == 404

    buf = io.BytesIO()
    Image.open(part.image_paths[0]).save(buf, format="PNG")
    for mode in ("fast", "none", "full"):
        r = client.post(
            f"/identify?top_n=3&tta={mode}", files={"file": ("a.png", buf.getvalue(), "image/png")}
        )
        assert r.status_code == 200, r.text
        assert r.json()["candidates"][0]["part_number"] == pn
    assert (
        client.post(
            "/identify?tta=bogus", files={"file": ("a.png", buf.getvalue(), "image/png")}
        ).status_code
        == 422
    )
    # gzip for larger JSON bodies
    r = client.get("/search?q=Steel&limit=100", headers={"Accept-Encoding": "gzip"})
    assert r.status_code == 200


def test_lan_urls_and_self_signed_cert(tmp_path):
    urls = lan_urls(8123)
    assert urls and all(u.startswith("http://") and u.endswith(":8123/") for u in urls)
    assert lan_urls(8443, "https")[0].startswith("https://")
    crt, key = self_signed_cert(tmp_path / "certs", ["192.168.1.20"])
    assert crt.exists() and key.exists() and b"BEGIN CERTIFICATE" in crt.read_bytes()
    assert self_signed_cert(tmp_path / "certs", ["10.0.0.1"]) == (
        crt,
        key,
    )  # reused, not regenerated
    assert Path(crt).stat().st_size > 500
