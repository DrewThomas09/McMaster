from __future__ import annotations

from fastapi.testclient import TestClient

from mcmaster_vision.api import create_app
from mcmaster_vision.config import Settings


def test_status_and_reload(identifier, demo_dir, tmp_path):
    identifier.index.save(tmp_path / "index" / "parts")
    s = Settings(
        catalog_db=demo_dir / "catalog.sqlite",
        index_dir=tmp_path / "index",
        model_dir=tmp_path / "models",
        data_dir=tmp_path,
        queries_dir=tmp_path / "q",
        api_token="secret",
    )
    client = TestClient(create_app(s, identifier=identifier))
    st = client.get("/status").json()
    assert st["loaded"] is True and st["catalog"]["parts"] == 40 and st["ready"] is True
    assert client.post("/admin/reload").status_code == 401
    r = client.post("/admin/reload", headers={"X-API-Token": "secret"})
    assert r.status_code == 200, r.text
    assert r.json()["reloaded"] is True and r.json()["index"]["parts"] == 40
