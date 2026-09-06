from __future__ import annotations

import io
import json

from fastapi.testclient import TestClient
from PIL import Image

from mcmaster_vision.api import create_app
from mcmaster_vision.config import Settings


def _png(part) -> bytes:
    buf = io.BytesIO()
    Image.open(part.image_paths[0]).save(buf, format="PNG")
    return buf.getvalue()


def test_constraints_filter_candidates(identifier, store):
    parts = list(store.iter_parts(with_images_only=True))
    part = parts[0]
    res = identifier.identify(Image.open(part.image_paths[0]), top_n=10)
    assert res.category_guess and 0 < res.category_guess[0][1] <= 1
    material = part.attributes["material"]
    res_c = identifier.identify(
        Image.open(part.image_paths[0]), top_n=10, constraints={"material": material}
    )
    assert res_c.constraints == {"material": material}
    assert all(c.attributes.get("material") == material for c in res_c.candidates)
    assert res_c.candidates[0].part_number == part.part_number
    # an impossible constraint falls back to the unfiltered list rather than returning nothing
    res_x = identifier.identify(
        Image.open(part.image_paths[0]), top_n=3, constraints={"material": "Unobtainium"}
    )
    assert len(res_x.candidates) == 3


def test_metrics_batch_and_request_log(identifier, store, tmp_path):
    app = create_app(Settings(data_dir=tmp_path, queries_dir=tmp_path / "q"), identifier=identifier)
    client = TestClient(app)
    parts = list(store.iter_parts(with_images_only=True))[:3]
    r = client.post(
        "/identify?top_n=3",
        files={"file": ("a.png", _png(parts[0]), "image/png")},
        params={"constraints": json.dumps({"material": parts[0].attributes["material"]})},
    )
    assert r.status_code == 200, r.text
    assert r.json()["constraints"] == {"material": parts[0].attributes["material"]}
    assert (
        client.post(
            "/identify",
            files={"file": ("a.png", _png(parts[0]), "image/png")},
            params={"constraints": "[1,2]"},
        ).status_code
        == 400
    )
    r = client.post(
        "/identify/batch?top_n=2",
        files=[("files", (f"{p.part_number}.png", _png(p), "image/png")) for p in parts]
        + [("files", ("bad.png", b"junk", "image/png"))],
    )
    assert r.status_code == 200, r.text
    rows = r.json()
    assert (
        len(rows) == 4
        and rows[-1]["error"]
        and all(row["best"] == row["file"].split(".")[0] for row in rows[:3])
    )
    m = client.get("/metrics").json()
    assert (
        m["requests_total"] == 4
        and m["latency_ms"]["p50"] is not None
        and sum(m["tiers"].values()) == 4
    )
    log = (tmp_path / "logs" / "requests.jsonl").read_text().strip().splitlines()
    assert len(log) == 4 and json.loads(log[0])["request_id"]
