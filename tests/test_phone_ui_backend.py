from __future__ import annotations

import io
from pathlib import Path

import pytest
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
    assert list((tmp_path / "cache" / "thumbs").glob("*_0_96.jpg"))  # cached on disk (hashed name)
    # sizes snap to a few fixed values so the cache cannot be flooded
    r = client.get(f"/parts/{pn}/thumb?size=97")
    assert Image.open(io.BytesIO(r.content)).size == (96, 96)
    assert len(list((tmp_path / "cache" / "thumbs").glob("*.jpg"))) == 1
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


def test_live_frames_have_their_own_rate_budget(identifier, tmp_path):
    import io

    from fastapi.testclient import TestClient
    from PIL import Image

    from mcmaster_vision.api import create_app
    from mcmaster_vision.config import Settings

    client = TestClient(
        create_app(
            Settings(data_dir=tmp_path, queries_dir=tmp_path / "q", rate_limit_per_minute=3),
            identifier=identifier,
        )
    )
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), "gray").save(buf, "PNG")
    png = buf.getvalue()
    for _ in range(3):  # a burst of live-preview frames
        r = client.post(
            "/identify?top_n=1&tta=none&log=false", files={"file": ("l.png", png, "image/png")}
        )
        assert r.status_code == 200
    r = client.post(
        "/identify?top_n=1&tta=none&log=false", files={"file": ("l.png", png, "image/png")}
    )
    assert r.status_code == 429  # the preview budget is spent ...
    r = client.post("/identify?top_n=1&tta=none", files={"file": ("p.png", png, "image/png")})
    assert r.status_code == 200  # ... but a real photo still goes through


def test_phone_page_scripts_parse():
    """Every inline script on the phone page must parse: a template-literal slip breaks the
    whole page, and only the browser tests would notice, slowly."""
    import re
    import shutil
    import subprocess
    from pathlib import Path

    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")
    html = Path("src/mcmaster_vision/api/static/index.html").read_text(encoding="utf-8")
    scripts = re.findall(r"<script>(.*?)</script>", html, flags=re.S)
    assert scripts
    for body in scripts:
        r = subprocess.run(
            [node, "-e", "new Function(require('fs').readFileSync(0, 'utf8'))"],
            input=body,
            text=True,
            capture_output=True,
            timeout=30,
        )
        assert r.returncode == 0, r.stderr[-600:]
