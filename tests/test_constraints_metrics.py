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


def test_unsatisfiable_constraints_are_reported_not_claimed(identifier, store):
    part = next(store.iter_parts(with_images_only=True))
    res = identifier.identify(
        Image.open(part.image_paths[0]), top_n=3, constraints={"material": "Unobtainium"}
    )
    assert res.constraints == {} and res.notes and "Unobtainium" in res.notes[0]


def test_batch_rejects_oversized_and_empty_files(identifier, tmp_path):
    client = TestClient(
        create_app(
            Settings(data_dir=tmp_path, queries_dir=tmp_path / "q", max_upload_mb=1),
            identifier=identifier,
        )
    )
    r = client.post(
        "/identify/batch",
        files=[
            ("files", ("big.png", b"x" * (2 * 1024 * 1024), "image/png")),
            ("files", ("empty.png", b"", "image/png")),
        ],
    )
    assert r.status_code == 200
    errs = [row["error"] for row in r.json()]
    assert any("exceeds" in e for e in errs) and any("empty" in e for e in errs)


def test_constraint_matching_is_loose():
    from mcmaster_vision.pipeline.identify import _norm_attr

    assert _norm_attr('1/4"-20') == _norm_attr("1/4-20") == _norm_attr(" 1/4 in - 20 ")
    assert _norm_attr("M6") == _norm_attr("m6") and _norm_attr("Brass") != _norm_attr("Bronze")


def test_query_embedding_cache_speeds_refinement(identifier, store):
    import time

    part = next(store.iter_parts(with_images_only=True))
    blob = _png(part)
    identifier._qcache.clear()
    t = time.perf_counter()
    r1 = identifier.identify_many_bytes([blob], top_n=3)
    t1 = time.perf_counter() - t
    t = time.perf_counter()
    r2 = identifier.identify_many_bytes(
        [blob], top_n=3, constraints={"material": part.attributes["material"]}
    )
    t2 = time.perf_counter() - t
    assert r1.candidates[0].part_number == r2.candidates[0].part_number == part.part_number
    assert "embed" in r2.timings_ms and r2.timings_ms["embed"] <= r1.timings_ms["embed"]
    assert len(identifier._qcache) == 1 and t2 <= t1 * 1.5
