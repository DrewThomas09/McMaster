"""Awkward inputs and concurrent operations must never produce a 5xx or corrupt state."""

from __future__ import annotations

import io
import json
import threading

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from mcmaster_vision.api import create_app
from mcmaster_vision.config import Settings
from mcmaster_vision.index import build_index

OK = (200, 400, 404, 413, 422, 429)


def _img(size=(64, 64), mode="RGB", fmt="JPEG") -> bytes:
    b = io.BytesIO()
    Image.new(mode, size, "gray").save(b, format=fmt)
    return b.getvalue()


@pytest.fixture()
def client(identifier, tmp_path):
    s = Settings(data_dir=tmp_path, queries_dir=tmp_path / "q", demo_mode=True)
    return TestClient(create_app(s, identifier=identifier), raise_server_exceptions=False)


@pytest.mark.parametrize(
    "q",
    [
        '1/4"-20',
        "M6 x 1",
        "AND",
        "OR NOT",
        '"unbalanced',
        "(",
        "*",
        "a OR",
        "NEAR/3",
        "-",
        " ",
        "x" * 500,
        'hex "nut"',
        "it's",
        "col:val",
        "^caret",
        "nut*",
    ],
)
def test_search_accepts_any_text(client, q):
    r = client.get("/search", params={"q": q})
    assert r.status_code in OK, r.text


def test_awkward_paths_and_params(client, store):
    pn = next(store.iter_parts(with_images_only=True)).part_number
    gets = [
        ("/search", {"category": "../../etc"}),
        ("/search", {"q": "nut", "offset": 10**6}),
        ("/parts/../etc/passwd", {}),
        ("/parts/" + "A" * 300, {}),
        (f"/parts/{pn}/thumb", {"i": 1000}),
        (f"/parts/{pn}/image", {"i": -1}),
        (f"/parts/{pn.lower()}", {}),
        ("/part/<script>alert(1)</script>", {}),
        ("/browse", {"category": "<b>x</b>", "offset": -5}),
        ("/browse", {"offset": 10**9}),
        ("/categories", {"depth": 4}),
        ("/demo/query/../../etc/passwd", {}),
        ("/demo/sheet", {"n": 100000}),
        ("/demo/samples", {"n": -1}),
        ("/static/../app.py", {}),
    ]
    for path, params in gets:
        r = client.get(path, params=params)
        assert r.status_code in OK, (path, params, r.status_code, r.text[:200])
    posts = [
        ("/identify", {}, None),
        ("/identify", {}, {"file": ("a.jpg", b"", "image/jpeg")}),
        ("/identify", {}, {"file": ("a.jpg", b"notanimage" * 100, "image/jpeg")}),
        ("/identify", {}, {"file": ("a.png", _img((64, 64), "RGBA", "PNG"), "image/png")}),
        ("/identify", {}, {"file": ("a.jpg", _img((1, 1)), "image/jpeg")}),
        ("/identify", {}, {"file": ("a.jpg", _img((80, 80), "L"), "image/jpeg")}),
        ("/identify", {"constraints": "[1,2]"}, {"file": ("a.jpg", _img(), "image/jpeg")}),
        ("/identify", {"constraints": '{"a":{"b":1}}'}, {"file": ("a.jpg", _img(), "image/jpeg")}),
        ("/identify", {"top_n": 0}, {"file": ("a.jpg", _img(), "image/jpeg")}),
        ("/identify", {"tta": "lots"}, {"file": ("a.jpg", _img(), "image/jpeg")}),
        ("/identify", {"mm_per_px": 1, "ref": "1,2,3"}, {"file": ("a.jpg", _img(), "image/jpeg")}),
        (
            "/identify",
            {"mm_per_px": 1, "ref": "1e9,1e9,-1e9,-1e9"},
            {"file": ("a.jpg", _img(), "image/jpeg")},
        ),
        ("/identify", {"mm_per_px": 1e30}, {"file": ("a.jpg", _img(), "image/jpeg")}),
        ("/identify/batch", {}, None),
        ("/feedback", {}, None),
        ("/demo/try/NOPE", {}, None),
        ("/demo/try/..%2F..%2Fetc", {}, None),
    ]
    for path, params, files in posts:
        r = client.post(path, params=params, files=files)
        assert r.status_code in OK, (path, params, r.status_code, r.text[:200])
    # nothing built under this data dir: reload says so instead of crashing
    assert client.post("/admin/reload").status_code == 503
    r = client.post("/identify", files=[("files", ("a.jpg", _img(), "image/jpeg"))] * 7)
    assert r.status_code == 400
    rid = client.post("/identify", files={"file": ("a.jpg", _img(), "image/jpeg")}).json()[
        "request_id"
    ]
    for data in (
        {"request_id": "../x", "part_number": pn},
        {"request_id": rid, "part_number": "../../x"},
        {"request_id": "zzzz", "part_number": pn},
        {"request_id": "r" * 200, "part_number": pn},
    ):
        r = client.post("/feedback", data=data)
        assert r.status_code in (400, 404, 422), (data, r.status_code, r.text)
    # lower-case part numbers are accepted and normalised
    assert (
        client.post("/feedback", data={"request_id": rid, "part_number": pn.lower()}).status_code
        == 200
    )
    assert (tmp_path_of(client) / "q" / pn / f"{rid}.jpg").exists()


def tmp_path_of(client: TestClient):
    return client.app.state.settings.data_dir


def test_parallel_identify_feedback_during_reload_backup_rebuild(store, embedder, tmp_path):
    build_index(store, embedder, "numpy", out_path=tmp_path / "index" / "parts")
    s = Settings(
        data_dir=tmp_path,
        queries_dir=tmp_path / "q",
        catalog_db=store.path,
        index_dir=tmp_path / "index",
        backbone="hash",
        max_concurrency=3,
        rate_limit_per_minute=10**6,
    )
    app = create_app(s)
    photo = io.BytesIO()
    Image.open(next(store.iter_parts()).image_paths[0]).convert("RGB").save(photo, format="JPEG")
    blob = photo.getvalue()
    errors: list = []
    with TestClient(app, raise_server_exceptions=False) as c:

        def worker(n: int) -> None:
            for _ in range(n):
                r = c.post("/identify?tta=none", files={"file": ("a.jpg", blob, "image/jpeg")})
                if r.status_code != 200:
                    errors.append(("identify", r.status_code, r.text[:120]))
                    continue
                res = r.json()
                r2 = c.post(
                    "/feedback",
                    data={
                        "request_id": res["request_id"],
                        "part_number": res["candidates"][0]["part_number"],
                        "predicted": res["candidates"][0]["part_number"],
                    },
                )
                if r2.status_code != 200:
                    errors.append(("feedback", r2.status_code, r2.text[:120]))

        def churn() -> None:
            for i in range(3):
                for path in ("/admin/reload", "/admin/backup"):
                    r = c.post(path)
                    if r.status_code != 200:
                        errors.append((path, r.status_code, r.text[:120]))
                build_index(
                    store,
                    embedder,
                    "numpy",
                    out_path=tmp_path / "index" / "parts",
                    gallery_augment=i % 2,
                )
                app.state.last_index_check = 0
                if c.get("/metrics").status_code != 200:
                    errors.append(("metrics", 0, ""))

        threads = [threading.Thread(target=worker, args=(8,)) for _ in range(4)]
        threads.append(threading.Thread(target=churn))
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors, errors[:5]
        m = c.get("/metrics").json()
        assert m["requests_total"] == 32 and m["feedback"]["confirmed"] == 32
        assert m["confirmed_top1_rate"] == 1.0
    lines = (tmp_path / "logs" / "requests.jsonl").read_text().splitlines()
    assert len(lines) == 32 and all(json.loads(ln) for ln in lines)
    fb = (tmp_path / "q" / "feedback.jsonl").read_text().splitlines()
    assert len(fb) == 32 and all(json.loads(ln) for ln in fb)
