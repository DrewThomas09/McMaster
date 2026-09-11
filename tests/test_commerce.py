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
    # adding again merges quantities; set_quantity replaces the line
    r = client.post("/cart", json={"client_id": "phone-1", "part_number": pn, "quantity": 2})
    assert [it["quantity"] for it in r.json()] == [3]
    body = {"client_id": "phone-1", "part_number": pn, "quantity": 5, "set_quantity": True}
    assert [it["quantity"] for it in client.post("/cart", json=body).json()] == [5]
    # the cart is on disk: a second app (another worker) sees it
    other = TestClient(create_app(client.app.state.settings, identifier=identifier))
    assert [it["quantity"] for it in other.get("/cart?client_id=phone-1").json()] == [5]
    assert (
        client.post("/cart", json={"client_id": "phone-1", "part_number": "NOPE"}).status_code
        == 404
    )
    # an unknown request id ties the line to no photo
    r = client.post("/cart", json={"client_id": "p9", "part_number": pn, "request_id": "nope"})
    assert r.json()[0]["request_id"] is None
    assert client.delete(f"/cart/{pn}?client_id=phone-1").json() == []
    assert client.post("/checkout", json={"client_id": "phone-1"}).status_code == 400  # empty
    ev = client.app.state.events
    kinds = [e["kind"] for e in ev.rows()]
    assert kinds.count("cart_add") == 3 and kinds[-1] == "cart_remove"  # set_quantity logs nothing
    assert ev.rows("cart_add")[0]["rank"] == d["rank"]
    assert ev.identify_row(req)["request_id"] == req and ev.identify_row("nope") is None
    # the cart add with a photo behind it already filed weak (weight 1) evidence
    fb = client.app.state.feedback
    assert [x.source for x in fb.entries()] == ["cart"] and fb.entries()[0].weight == 1


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
    assert client.get("/cart?client_id=p2").json() == [] and order["learned"] == 1
    # persisted
    lines = (tmp_path / "logs" / "orders.jsonl").read_text().splitlines()
    assert json.loads(lines[-1])["order_id"] == order["order_id"]
    # a phone sees its own orders; the full list needs the token
    assert client.get("/orders").status_code in (401, 403)
    assert client.get("/orders?client_id=p2").json()[0]["order_id"] == order["order_id"]
    assert client.get("/orders?client_id=someone-else").json() == []
    assert client.get("/orders", headers={"X-API-Token": "tok"}).json()[0]["client_id"] == "p2"
    # the photographed item went cart (weight 1) -> checkout (weight 3); the other could not
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


def test_error_events_are_logged(identifier, tmp_path, monkeypatch):
    client, _ = _client(identifier, tmp_path)
    assert client.get("/analytics").json()["window"]["errors"] == 0
    monkeypatch.setattr(identifier, "identify", lambda *a, **k: 1 / 0)
    client = TestClient(client.app, raise_server_exceptions=False)
    r = client.post("/identify", files={"file": ("a.png", _png(), "image/png")})
    assert r.status_code == 500
    a = client.get("/analytics").json()
    assert a["window"]["errors"] == 1 and "500 /identify" in a["errors"]


def _png() -> bytes:
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (64, 64), "gray").save(buf, "PNG")
    return buf.getvalue()


def test_event_log_survives_odd_rows(tmp_path):
    path = tmp_path / "e.jsonl"
    path.write_text('{"foo": 1}\n[]\n{"kind": "identify", "request_id": "r1", "best": "P"}\n')
    ev = EventLog(path)
    assert ev.rows() == [{"kind": "identify", "request_id": "r1", "best": "P"}]
    assert ev.identify_row("r1")["best"] == "P"
    # a row another worker wrote is found in the file
    with open(path, "a") as fh:
        fh.write('{"kind": "identify", "request_id": "r2", "best": "Q"}\n')
    assert ev.identify_row("r2")["best"] == "Q"
    # the file is compacted when it is far longer than the window
    small = EventLog(tmp_path / "s.jsonl", keep=5)
    for i in range(20):
        small.log("identify", request_id=f"x{i}")
    again = EventLog(tmp_path / "s.jsonl", keep=5)
    assert again.total == 20 and len(again.rows()) == 5
    assert len((tmp_path / "s.jsonl").read_text().splitlines()) == 5


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
        "none_of_these": 0,
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


def test_none_of_these_issue(tmp_path):
    ev = EventLog(tmp_path / "e.jsonl")
    for i in range(20):
        ev.log("identify", request_id=f"r{i}", tier="candidate", best="P1", candidates=["P1"])
        if i % 2 == 0:
            ev.log("feedback", request_id=f"r{i}", part_number=None, source="tap", correct=False)
    a = analytics(ev)
    assert a["window"]["none_of_these"] == 10 and a["funnel"]["none_of_these"] == 0.5
    assert any("found nothing" in i["what"] for i in issues(a))


def test_two_parts_from_one_photo_teach_nothing(identifier, tmp_path):
    client, _ = _client(identifier, tmp_path)
    pn, d = _identify_sample(client, seed=4)
    req = d["result"]["request_id"]
    other = [c["part_number"] for c in d["result"]["candidates"] if c["part_number"] != pn]
    other = other[0] if other else client.get("/demo/samples?n=2&seed=3").json()[1]["part_number"]
    client.post("/cart", json={"client_id": "cmp", "part_number": pn, "request_id": req})
    fb = client.app.state.feedback
    assert [x.source for x in fb.entries()] == ["cart"]
    # a second, different part from the same photo: the customer is comparing
    client.post("/cart", json={"client_id": "cmp", "part_number": other, "request_id": req})
    assert len(fb.entries()) == 1 and fb.entries()[0].part_number == pn  # not relabelled
    order = client.post("/checkout", json={"client_id": "cmp"}).json()
    assert order["learned"] == 0 and len(order["items"]) == 2
    assert len(fb.entries()) == 1 and fb.entries()[0].source == "cart"
    # the cart budget is separate from the photo budget
    for _ in range(5):
        assert client.get("/cart?client_id=cmp").status_code == 200


def test_confusion_pairs_know_the_catalog(tmp_path, store):
    from mcmaster_vision.pipeline.events import enrich_confusions

    parts = list(store.iter_parts(with_images_only=True))
    same = [p for p in parts if p.family_id == parts[0].family_id]
    a, b = parts[0], (same[1] if len(same) > 1 else parts[1])
    ev = EventLog(tmp_path / "e.jsonl")
    for i in range(2):
        ev.log("identify", request_id=f"r{i}", best=a.part_number, candidates=[a.part_number])
        ev.log(
            "checkout",
            order_id=f"o{i}",
            items=[{"part_number": b.part_number, "request_id": f"r{i}"}],
        )
    an = enrich_confusions(analytics(ev), store)
    c = an["confusions"][0]
    assert (
        c["predicted"] == a.part_number and "differ_by" in c and isinstance(c["same_family"], bool)
    )
    found = [i for i in issues(an) if a.part_number in i["what"]]
    assert found and (
        "identical" in found[0]["do"]
        or "differ by" in found[0]["do"]
        or "look-alikes" in found[0]["do"]
    )
    if c["differ_by"] == []:
        assert "identical specifications" in found[0]["do"]


def test_daily_trend_lines_up_learning_with_accuracy(tmp_path):
    ev = EventLog(tmp_path / "e.jsonl")
    for i in range(4):
        ev.log("identify", request_id=f"r{i}", best="P" if i % 2 else "Q", candidates=["P", "Q"])
        ev.log("checkout", order_id=f"o{i}", items=[{"part_number": "P", "request_id": f"r{i}"}])
    ev.log("learn", how="incremental", added=4)
    ev.log("learn", how="retrain", served=True)
    (day,) = analytics(ev)["daily"]
    assert day["identify"] == 4 and day["bought"] == 4 and day["bought_top1_rate"] == 0.5
    assert day["learns"] == 1 and day["retrains"] == 1


def test_category_precision_and_issue(tmp_path):
    ev = EventLog(tmp_path / "e.jsonl")
    for i in range(6):
        ev.log(
            "identify",
            request_id=f"r{i}",
            best="P",
            candidates=["P", "Q"],
            category="Fastening > Nuts",
        )
        ev.log("checkout", order_id=f"o{i}", items=[{"part_number": "Q", "request_id": f"r{i}"}])
    ev.log("identify", request_id="x", best="Z", candidates=["Z"], category="Sealing > O-Rings")
    ev.log("checkout", order_id="ox", items=[{"part_number": "Z", "request_id": "x"}])
    a = analytics(ev)
    cats = a["category_precision_bought"]
    assert list(cats)[0] == "Fastening > Nuts" and cats["Fastening > Nuts"]["precision"] == 0.0
    assert cats["Sealing > O-Rings"]["precision"] == 1.0
    assert any("Fastening > Nuts" in i["what"] for i in issues(a))


def test_identify_event_carries_category(identifier, tmp_path):
    client, _ = _client(identifier, tmp_path)
    pn, d = _identify_sample(client, seed=6)
    row = client.app.state.events.identify_row(d["result"]["request_id"])
    assert row["category"] and " > " in row["category"]


def test_segment_issue_names_the_worst_served_shops(tmp_path):
    a = analytics(EventLog(tmp_path / "e.jsonl"))
    a["segment_precision_bought"] = {
        "Pipe > Fittings (#2)": {
            "label": "Pipe > Fittings",
            "segment": 2,
            "bought": 12,
            "top1_right": 4,
            "precision": 0.333,
        }
    }
    found = issues(a)
    assert any("Pipe > Fittings" in i["what"] for i in found)


def test_recommendation_take_rate_counts_new_parts(tmp_path):
    from mcmaster_vision.pipeline.events import recommendation_take

    log = EventLog(tmp_path / "e.jsonl")
    log.log("checkout", client_id="s1", items=[{"part_number": "A"}])
    log.log("recommend_shown", client_id="s1", parts=["A", "B"])
    log.log("checkout", client_id="s1", items=[{"part_number": "A"}])  # took a re-order
    log.log("recommend_shown", client_id="s1", parts=["A", "B"])
    log.log("checkout", client_id="s1", items=[{"part_number": "B"}])  # took something new
    log.log("recommend_shown", client_id="s2", parts=["C"])
    log.log("checkout", client_id="s2", items=[{"part_number": "D"}])  # ignored the strip
    log.log("checkout", client_id="s3", items=[{"part_number": "C"}])  # never shown one
    take = recommendation_take(log.rows())
    assert take == {"orders_after_strip": 3, "take_rate": 0.667, "new_part_take_rate": 0.333}
    a = analytics(log)
    assert a["recommendations"] == take


def test_search_events_do_not_evict_the_journey(tmp_path):
    log = EventLog(tmp_path / "e.jsonl", keep=50)
    for i in range(10):
        log.log("identify", request_id=f"r{i}", best="A")
    for _ in range(300):
        log.log("search", q="hex nut", results=3)
    log.log("checkout", client_id="s1", items=[{"part_number": "A", "request_id": "r3"}])
    assert len(log.rows("identify")) == 10 and len(log.rows("checkout")) == 1
    assert len(log.rows("search")) == 300 and log.identify_row("r3") is not None
    assert len(log.rows()) == 311  # everything, in time order
    # compaction and a fresh boot keep the journey rows too
    log2 = EventLog(tmp_path / "e.jsonl", keep=50)
    assert len(log2.rows("identify")) == 10 and len(log2.rows("checkout")) == 1
    lines = (tmp_path / "e.jsonl").read_text().splitlines()
    assert 11 <= len(lines) <= 50 + 1000
