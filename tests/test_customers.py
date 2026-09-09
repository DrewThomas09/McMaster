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
    # one-off buyer takes a's pair and one of b's; a and b each repeat one part
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
            items=[
                CartItem(part_number=a[0].part_number),
                CartItem(part_number=a[1].part_number),
                CartItem(part_number=b[0].part_number),
            ],
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
    assert rec and rec[0]["part_number"] == a[0].part_number
    assert "3 times" in rec[0]["why"] or "usually every" in rec[0]["why"]  # a staple, due again
    comp = dict(book.complements(a[0].part_number))
    assert a[1].part_number in comp  # bought together by two different shops
    assert b[0].part_number not in comp  # one shop's one-off is not a complement
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
    # three shops cannot fill a public segment: the view folds them into one line
    assert len(seg["segments"]) == 1 and "small segments" in seg["segments"][0]["label"]
    assert seg["segments"][0]["customers"] == 3
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


def test_usual_values_and_reorder_due(store):
    orders, a, b, cats = _orders(store)
    book = CustomerBook(orders, {p.part_number: p for p in store.iter_parts()}, k=2)
    usual = book.usual_values("shop-a")
    mat = a[0].attributes.get("material")
    if mat:
        assert usual.get("material") == mat  # bought three times, the shop's habit
    assert book.usual_values("nobody") == {}
    # a[0] was bought on three consecutive days: a day later it is due again
    due = book.due("shop-a", now=datetime(2026, 1, 5, tzinfo=timezone.utc))
    assert due and due[0]["part_number"] == a[0].part_number and due[0]["every_days"] == 1.0
    assert book.due("shop-a", now=datetime(2026, 1, 3, 1, tzinfo=timezone.utc)) == []
    rec = book.recommend("shop-a", 3)
    assert rec[0]["part_number"] == a[0].part_number


def test_family_answer_marks_the_usual_size(identifier, store, tmp_path):
    client = _client(identifier, tmp_path)
    orders, a, b, cats = _orders(store)
    for o in orders:
        client.app.state.carts.save_order(o)
    seg = client.get("/analytics").json()
    assert "segment_precision_bought" in seg
    page = client.get("/dashboard").text
    assert "Customers and segments" in page and "photo top-1 when bought" in page


def test_rerank_within_tiers_keeps_text_order_across_tiers():
    from types import SimpleNamespace as P

    from mcmaster_vision.pipeline.customers import rerank_within_tiers

    a, b, c, d, e = (P(part_number=x) for x in "ABCDE")
    # A..C match the words equally well (bm25 within 10%); D, then E, match them less well
    scored = [(a, -10.0), (b, -9.8), (c, -9.5), (d, -8.0), (e, -6.0)]
    out = rerank_within_tiers(scored, {"C": 0.8, "E": 1.0, "A": -0.5})
    assert [p.part_number for p in out] == ["C", "A", "B", "D", "E"]
    # a history below the threshold does not move anything; ties keep the text order
    assert [p.part_number for p in rerank_within_tiers(scored, {"B": 0.02})] == list("ABCDE")
    # part-number matches stay first in their own order, whatever the boosts say
    pinned = [(e, -1e9), (d, -1e9)] + scored[:2]
    assert [p.part_number for p in rerank_within_tiers(pinned, {"D": 1.0, "B": 1.0})] == list(
        "EDBA"
    )
    # the LIKE fallback scores every hit 0: one tier, the history orders it
    flat = [(a, 0.0), (b, 0.0), (c, 0.0)]
    assert [p.part_number for p in rerank_within_tiers(flat, {"C": 0.3})] == list("CAB")


def test_search_text_scored_matches_search_text(store):
    parts = store.search_text("Threaded Pipe Nipple", 8)
    scored = store.search_text_scored("Threaded Pipe Nipple", 8)
    assert [p.part_number for p, _ in scored] == [p.part_number for p in parts]
    assert all(sc < 0 for _, sc in scored)  # bm25: stronger is more negative
    assert scored[0][1] <= scored[-1][1]
    pn = parts[0].part_number
    pinned = store.search_text_scored(pn[:6], 5)
    assert pinned and pinned[0][1] == store.PINNED_SCORE


def test_recommend_tiers_never_let_popularity_outrank_the_shop_history(store):
    orders, a, b, cats = _orders(store)
    # a crowd of shops in shop-b's category all buy one part shop-b never did: popular
    # with shops like it, but not personal
    mine = {it.part_number for o in orders if o.client_id == "shop-b" for it in o.items}
    extra = next(p for p in b if p.part_number not in mine)
    crowd = [
        Order(
            order_id=f"X{i}",
            client_id=f"crowd-{i}",
            items=[CartItem(part_number=b[0].part_number), CartItem(part_number=extra.part_number)],
            created_at=datetime(2026, 2, 1, tzinfo=timezone.utc),
        )
        for i in range(120)
    ]
    book = CustomerBook(orders + crowd, {p.part_number: p for p in store.iter_parts()}, k=2)
    rec = book.recommend("shop-b", 10)
    why = [r["why"] for r in rec]
    assert rec[0]["part_number"] == b[0].part_number  # its own staple, bought three times
    guess = next(r for r in rec if r["part_number"] == extra.part_number)
    # a complement of its staple or a segment favourite: a guess, never above 0.75
    assert ("often bought with" in guess["why"] or "shops like yours" in guess["why"]) and guess[
        "score"
    ] <= 0.75
    first_guess = why.index(guess["why"])
    own = ("ordered", "usually every")
    assert first_guess >= 4 and all(any(k in w for k in own) for w in why[:first_guess])


def test_personalised_search_pages_never_overlap_or_skip(identifier, store, tmp_path):
    client = _client(identifier, tmp_path)
    orders, a, b, cats = _orders(store)
    for o in orders:
        client.app.state.carts.save_order(o)
    word = "steel"
    pages = []
    for off in range(0, 120, 7):
        rows = client.get(f"/search?q={word}&limit=7&offset={off}&client_id=shop-a").json()
        pages.extend(p["part_number"] for p in rows)
        if len(rows) < 7:
            break
    full = [
        p["part_number"] for p in client.get(f"/search?q={word}&limit=100&client_id=shop-a").json()
    ]
    assert len(pages) == len(set(pages))  # no duplicates across pages
    assert pages == full[: len(pages)]  # and the same order as one long page


def test_recommend_keeps_a_slot_for_something_new(store):
    orders, a, b, cats = _orders(store)
    parts = {p.part_number: p for p in store.iter_parts()}
    book = CustomerBook(orders, parts, k=2)
    by_cat = lambda cat: [p for p in parts.values() if " > ".join(p.category_path[:2]) == cat]  # noqa: E731
    rec = book.recommend("shop-a", 4, browse=by_cat)
    mine = {it.part_number for o in orders if o.client_id == "shop-a" for it in o.items}
    fresh = [r for r in rec if r["part_number"] not in mine]
    assert len(rec) == 4 and fresh and fresh[0]["why"].startswith("new in")
    assert rec[-1] is fresh[-1]  # the new thing sits after the re-orders
    assert all(r["part_number"] in mine for r in rec[:-1])
    assert parts[fresh[0]["part_number"]].category_path[0] == a[0].category_path[0]
    # no slot is wasted when there is nothing new to say
    assert len(book.recommend("shop-a", 4)) == 4
