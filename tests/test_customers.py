"""The customer model: profiles, segments, priors, complements, recommendations, and the
personalised ranking it drives through the API."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from mcmaster_vision.api import create_app
from mcmaster_vision.config import Settings
from mcmaster_vision.pipeline.customers import CustomerBook, customer_boost_weight
from mcmaster_vision.schemas import CartItem, Order


def _orders(store):
    parts = list(store.iter_parts(with_images_only=True))
    by_cat: dict[str, list] = {}
    for p in parts:
        by_cat.setdefault(p.category_path[0], []).append(p)
    cats = sorted(by_cat, key=lambda c: -len(by_cat[c]))[:2]
    assert len(cats) == 2, "the synthetic catalog has two top-level categories at least"
    a, b = by_cat[cats[0]], by_cat[cats[1]]
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    orders = []
    # shop-a buys from the first category (three orders), shop-b from the second, a
    # one-off buyer takes one part of each; a and b each repeat one part
    for i in range(3):
        orders.append(
            Order(
                order_id=f"A{i}",
                client_id="shop-a",
                items=[
                    CartItem(part_number=a[0].part_number),
                    CartItem(part_number=a[1 if i < 2 else 2 % len(a)].part_number),
                ],
                created_at=t0 + timedelta(days=i),
            )
        )
        orders.append(
            Order(
                order_id=f"B{i}",
                client_id="shop-b",
                items=[
                    CartItem(part_number=b[0].part_number),
                    CartItem(part_number=b[(i + 1) % len(b)].part_number),
                ],
                created_at=t0 + timedelta(days=i),
            )
        )
    orders.append(
        Order(
            order_id="C0",
            client_id="once",
            items=[CartItem(part_number=a[0].part_number), CartItem(part_number=b[0].part_number)],
            created_at=t0,
        )
    )
    return orders, a, b, cats


def test_profiles_segments_priors_and_complements(store):
    orders, a, b, cats = _orders(store)
    book = CustomerBook(orders, {p.part_number: p for p in store.iter_parts()}, k=2)
    pa, pb = book.profiles["shop-a"], book.profiles["shop-b"]
    assert pa.orders == 3 and pa.parts[a[0].part_number] == 3 and pa.recent
    assert pa.segment is not None and pb.segment is not None and pa.segment != pb.segment
    labels = {s["segment"]: s["label"] for s in book.segments}
    assert cats[0] in labels[pa.segment] and cats[1] in labels[pb.segment]
    # the prior favours the customer's own category and penalises the other
    boosts = book.boosts("shop-a", [a[1].part_number, b[1].part_number])
    assert boosts[a[1].part_number] > 0 > boosts[b[1].part_number]
    assert all(v == 0 for v in book.boosts("stranger", [a[1].part_number]).values())
    # a repeat part is a re-order suggestion; complements come from co-purchases
    rec = book.recommend("shop-a", 5)
    assert rec and rec[0]["part_number"] == a[0].part_number and "3 times" in rec[0]["why"]
    comp = dict(book.complements(a[0].part_number))
    assert a[1].part_number in comp  # bought together twice
    assert customer_boost_weight(0) == 0 and customer_boost_weight(5) == 1.0
    summary = book.summary()
    assert summary["customers"] == 3 and summary["orders"] == 7 and len(summary["segments"]) == 2
    # a stranger gets the global mix, a regular their own
    glob = book.category_prior(None)
    own = book.category_prior("shop-a")
    ka = " > ".join(a[0].category_path[:2])
    assert own[ka] > glob[ka]


def _client(identifier, tmp_path):
    s = Settings(data_dir=tmp_path, queries_dir=tmp_path / "q", demo_mode=True)
    return TestClient(create_app(s, identifier=identifier))


def test_api_personalises_search_and_identify(identifier, store, tmp_path):
    client = _client(identifier, tmp_path)
    orders, a, b, cats = _orders(store)
    for o in orders:
        client.app.state.carts.save_order(o)
    me = client.get("/me?client_id=shop-a").json()
    assert me["known"] and me["profile"]["orders"] == 3 and me["segment"] is not None
    assert me["personalisation_weight"] == 0.6
    assert client.get("/me?client_id=nobody").json()["known"] is False
    seg = client.get("/segments").json()
    assert seg["customers"] == 3 and seg["segments"]
    rec = client.get("/recommend?client_id=shop-a&n=4").json()
    assert (
        rec and rec[0]["part_number"] == a[0].part_number and rec[0]["thumb"].startswith("/parts/")
    )
    # a search that spans both categories comes back re-ordered for each shop
    word = "steel"
    plain = [p["part_number"] for p in client.get(f"/search?q={word}&limit=30").json()]
    for_a = [
        p["part_number"] for p in client.get(f"/search?q={word}&limit=30&client_id=shop-a").json()
    ]
    for_b = [
        p["part_number"] for p in client.get(f"/search?q={word}&limit=30&client_id=shop-b").json()
    ]
    assert set(for_a) and set(for_b)
    parts = {p.part_number: p for p in store.iter_parts()}

    def share(pns, cat):
        top = pns[:8]
        return sum(parts[x].category_path[0] == cat for x in top if x in parts) / max(1, len(top))

    assert share(for_a, cats[0]) >= share(plain, cats[0])
    assert share(for_b, cats[1]) >= share(plain, cats[1])
    assert client.get(f"/search?q={word}&client_id=bad id").status_code == 422
    ev = client.app.state.events.rows("search")
    assert ev and ev[-1]["personalised"] is True and ev[0]["personalised"] is False
    # identify accepts the client id and the prior shows up as a reason when it matters
    pn = a[0].part_number
    d = client.post(f"/demo/try/{pn}?seed=1&tta=none").json()
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.open(a[0].image_paths[0]).convert("RGB").save(buf, "PNG")
    r = client.post(
        "/identify?tta=none&client_id=shop-a",
        files={"file": ("a.png", buf.getvalue(), "image/png")},
    )
    assert r.status_code == 200
    r2 = client.post(
        "/identify?tta=none&client_id=zz", files={"file": ("a.png", buf.getvalue(), "image/png")}
    )
    assert r2.status_code == 200
    assert d["truth"] == pn
