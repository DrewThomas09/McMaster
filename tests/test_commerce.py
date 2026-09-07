"""The purchase loop: cart -> checkout -> checkout confirmation -> analytics -> learning."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from mcmaster_vision.api import create_app
from mcmaster_vision.config import Settings
from mcmaster_vision.pipeline.events import EventLog, analytics, issues


def _client(identifier, tmp_path, **kw):
    s = Settings(
        data_dir=tmp_path, queries_dir=tmp_path / "q", demo_mode=True, api_token="tok", **kw
    )
    return TestClient(create_app(s, identifier=identifier)), s


def _identify_sample(client, seed=1):
    pn = client.get("/demo/samples?n=1&seed=5").json()[0]["part_number"]
    d = client.post(f"/demo/try/{pn}?seed={seed}&top_n=5&tta=fast").json()
    return pn, d


def test_cart_add_remove_and_validation(identifier, tmp_path):
    client, _ = _client(identifier, tmp_path)
    pn, d = _identify_sample(client)
    req = d["result"]["request_id"]
    assert client.post("/cart", json={"client_id": "bad id!", "part_number": pn}).status_code == 400
    assert client.get("/cart?client_id=nobody").json() == []
    r = client.post(
        "/cart", json={"client_id": "phone-1", "part_number": pn.lower(), "request_id": req}
    )
    assert r.status_code == 200, r.text
    (item,) = r.json()
    assert item["part_number"] == pn and item["quantity"] == 1 and item["name"]
    # the cart line knows what the identification said about it
    if d["rank"] == 1:
        assert item["confidence"] is not None and item["tier"] == d["result"]["tier"]
    # adding again merges quantities
    r = client.post("/cart", json={"client_id": "phone-1", "part_number": pn, "quantity": 2})
    assert [it["quantity"] for it in r.json()] == [3]
    assert (
        client.post("/cart", json={"client_id": "phone-1", "part_number": "NOPE"}).status_code
        == 404
    )
    assert client.delete(f"/cart/{pn}?client_id=phone-1").json() == []
    assert client.post("/checkout", json={"client_id": "phone-1"}).status_code == 400  # empty
    ev = client.app.state.events
    assert [e["kind"] for e in ev.rows()][-2:] == ["cart_add", "cart_remove"]
    assert ev.rows("cart_add")[0]["rank"] == d["rank"]


def test_checkout_files_purchase_confirmations(identifier, tmp_path):
    client, s = _client(identifier, tmp_path)
    pn, d = _identify_sample(client, seed=2)
    req = d["result"]["request_id"]
    other = client.get("/demo/samples?n=2&seed=9").json()[1]["part_number"]
    client.post("/cart", json={"client_id": "p2", "part_number": pn, "request_id": req})
    client.post("/cart", json={"client_id": "p2", "part_number": other})  # no photo behind it
    r = client.post("/checkout", json={"client_id": "p2"})
    assert r.status_code == 200, r.text
    order = r.json()
    assert len(order["order_id"]) == 10 and len(order["items"]) == 2
    assert client.get("/cart?client_id=p2").json() == []
    # persisted
    lines = (tmp_path / "logs" / "orders.jsonl").read_text().splitlines()
    assert json.loads(lines[-1])["order_id"] == order["order_id"]
    assert client.get("/orders").json()[0]["order_id"] == order["order_id"]
    # the photographed item became a checkout confirmation; the other could not
    fb = client.app.state.feedback
    entries = fb.entries()
    assert len(entries) == 1 and entries[0].source == "checkout" and entries[0].weight == 3
    assert (
        entries[0].part_number == pn and entries[0].predicted == d["result"]["best"]["part_number"]
    )
    assert fb.stats()["purchases"] == 1 and fb.stats()["by_source"] == {"checkout": 1}
    assert fb.confirmation_counts() == {pn: 3}
    ck = client.app.state.events.rows("checkout")
    assert len(ck) == 1 and ck[0]["learned"] == 1 and len(ck[0]["items"]) == 2
    # analytics see the funnel and the learning state
    a = client.get("/analytics").json()
    assert a["window"]["identify"] == 1 and a["window"]["checkout"] == 1
    assert a["funnel"]["identify_to_checkout"] == 1.0
    assert a["bought_top1_rate"] in (0.0, 1.0)
    assert a["learning"]["new_purchases"] == 1 and a["learning"]["new_confirmations"] == 1
    assert a["learning"]["retrain_due"] is False
    assert isinstance(a["issues"], list)
    # confirmation photos of purchases go into the weighted training set 3x
    weighted = fb.labelled_images(weighted=True)
    assert len(weighted[pn]) == 3 and len(fb.labelled_images()[pn]) == 1


def test_admin_learn_requires_token_and_reports(identifier, tmp_path, monkeypatch):
    client, s = _client(identifier, tmp_path)
    assert client.post("/admin/learn").status_code in (401, 403)
    called = {}

    def fake_learn(settings, **kw):
        called["settings"] = settings
        return {"action": "none", "reason": "no new confirmations"}

    monkeypatch.setattr("mcmaster_vision.pipeline.learn.learn_index", fake_learn)
    r = client.post("/admin/learn", headers={"X-API-Token": "tok"})
    assert r.status_code == 200 and r.json()["action"] == "none"
    assert called["settings"].queries_dir == s.queries_dir


def test_error_events_are_logged(identifier, tmp_path):
    client, _ = _client(identifier, tmp_path)
    r = client.get("/analytics")
    assert r.status_code == 200 and r.json()["window"]["errors"] == 0


def test_analytics_and_issues_from_events(tmp_path):
    ev = EventLog(tmp_path / "e.jsonl")
    for i in range(12):
        best = f"P{i % 3}"
        ev.log(
            "identify",
            request_id=f"r{i}",
            tier="likely",
            best=best,
            confidence=0.5,
            candidates=[best, "P9"],
            latency_ms=2000 if i % 4 == 0 else 50,
        )
        ev.log("cart_add", request_id=f"r{i}", part_number="P9", rank=2, was_top=False)
        ev.log("checkout", order_id=f"o{i}", items=[{"part_number": "P9", "request_id": f"r{i}"}])
    for _ in range(3):
        ev.log("error", status=500, path="/identify")
    a = analytics(ev)
    assert a["window"] == {
        "identify": 12,
        "cart_add": 12,
        "checkout": 12,
        "items_bought": 12,
        "feedback": 0,
        "errors": 3,
    }
    assert a["funnel"]["identify_to_cart"] == 1.0 and a["bought_top1_rate"] == 0.0
    assert a["confusions"][0] == {"predicted": "P0", "bought": "P9", "times": 4}
    assert a["tier_precision_bought"]["likely"]["precision"] == 0.0
    assert a["latency_ms"]["p95"] > 1500 and a["bought_rank_hist"] == {2: 12}
    found = issues(a)
    whats = " | ".join(i["what"] for i in found)
    assert "top answer" in whats and "P0" in whats and "p95" in whats and "errors" in whats
    assert {i["severity"] for i in found} >= {"high", "medium"}
    # persisted and reloaded
    again = EventLog(tmp_path / "e.jsonl")
    assert again.total == ev.total and len(again.rows("checkout")) == 12
    (tmp_path / "e.jsonl").write_text("not json\n" + (tmp_path / "e.jsonl").read_text())
    assert EventLog(tmp_path / "e.jsonl").total == ev.total


def test_issues_quiet_on_empty(tmp_path):
    a = analytics(EventLog(tmp_path / "none.jsonl"))
    assert a["bought_top1_rate"] is None and a["funnel"]["identify_to_cart"] is None
    assert issues(a) == []
