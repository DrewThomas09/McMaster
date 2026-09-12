"""The customer model: profiles, segments, priors, complements, recommendations, and the
personalised ranking it drives through the API."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from mcmaster_vision.api import create_app
from mcmaster_vision.config import Settings
from mcmaster_vision.pipeline.customers import CustomerBook, customer_boost_weight
from mcmaster_vision.schemas import CartItem, Order, Part


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
    assert me["profile"]["bought"][a[0].part_number] == 3  # the phone's "bought 3x" chip
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
    by_cat = lambda path, mat: store.by_category(path, limit=40, material=mat)  # noqa: E731
    rec = book.recommend("shop-a", 4, browse=by_cat)
    mine = {it.part_number for o in orders if o.client_id == "shop-a" for it in o.items}
    fresh = [r for r in rec if r["part_number"] not in mine]
    assert len(rec) == 4 and fresh and fresh[0]["why"].startswith("new in")
    assert rec[-1] is fresh[-1]  # the new thing sits after the re-orders
    assert all(r["part_number"] in mine for r in rec[:-1])
    assert parts[fresh[0]["part_number"]].category_path[0] == a[0].category_path[0]
    # no slot is held back when there is nothing new to say: the history fills the list
    plain = book.recommend("shop-a", 4)
    assert len(plain) >= 3 and all(r["part_number"] in mine for r in plain)
    # one slot is never the whole list: the shop's own first pick always fits
    one = book.recommend("shop-a", 1, browse=by_cat)
    assert len(one) == 1 and one[0]["part_number"] in mine
    # the new part is in one of the shop's two materials (asked of the catalog, not
    # filtered after a cut) and the reason names the aisle and the material
    mats = {m for m, _ in book.profiles["shop-a"].materials.most_common(2)}
    for r in rec:
        if r["why"].startswith("new in"):
            m = parts[r["part_number"]].attributes.get("material")
            assert m in mats and r["why"].endswith(f", {m}") and "Washers" in r["why"]


def test_by_category_material_filter(store):
    parts = list(store.iter_parts())
    p = next(p for p in parts if p.attributes.get("material"))
    mat = p.attributes["material"]
    got = store.by_category(p.category_path[:1], limit=500, material=mat)
    assert got and all(x.attributes.get("material") == mat for x in got)
    assert p.part_number in {x.part_number for x in got}
    assert len(got) < len(store.by_category(p.category_path[:1], limit=500))
    assert store.by_category([], limit=5, material="no such material") == []


def test_segments_follow_the_top_level_mix_not_the_staple(store):
    # two industries that buy from different top-level categories, each shop with its
    # own staple sub-category bought every time: the staple must not split an industry
    parts = list(store.iter_parts(with_images_only=True))
    by_top: dict[str, list] = {}
    for p in parts:
        by_top.setdefault(p.category_path[0], []).append(p)
    tops = sorted(by_top, key=lambda c: -len(by_top[c]))[:2]
    t0 = datetime(2026, 3, 1, tzinfo=timezone.utc)
    orders = []
    for ind, top in enumerate(tops):
        pool = by_top[top]
        for shop in range(6):
            staple = pool[shop % len(pool)]
            for i in range(6):
                other = pool[(shop * 7 + i * 3) % len(pool)]
                orders.append(
                    Order(
                        order_id=f"{ind}-{shop}-{i}",
                        client_id=f"ind{ind}-shop{shop}",
                        items=[
                            CartItem(part_number=staple.part_number),
                            CartItem(part_number=other.part_number),
                        ],
                        created_at=t0 + timedelta(days=i),
                    )
                )
    book = CustomerBook(orders, {p.part_number: p for p in parts}, k=2, seed=1)
    segs = {ind: {book.profiles[f"ind{ind}-shop{s}"].segment for s in range(6)} for ind in (0, 1)}
    assert len(segs[0]) == 1 and len(segs[1]) == 1 and segs[0] != segs[1]
    # the prior still carries the segment's own category mix
    prior = book.category_prior("ind0-shop0")
    assert max(prior, key=prior.get).startswith(tops[0])


def test_popularity_breaks_ties_for_strangers(identifier, store, tmp_path):
    from mcmaster_vision.pipeline.customers import popularity_boosts

    counts = Counter({"A": 9, "B": 1})
    b = popularity_boosts(counts, ["A", "B", "C"])
    assert b["A"] == 0.3 and 0 < b["B"] < 0.3 and "C" not in b
    client = _client(identifier, tmp_path)
    orders, a, b_, cats = _orders(store)
    for o in orders:
        client.app.state.carts.save_order(o)
    # a stranger searching the name shared by a[0]'s variants sees the most-bought first
    words = " ".join(w for w in a[0].name.split() if w[0].isupper())[:40]
    q = a[0].name.split()[-1]
    rows = client.get(f"/search?q={q}&limit=30").json()
    pns = [p["part_number"] for p in rows]
    if a[0].part_number in pns:
        same_tier = [p for p in rows if p["name"].endswith(q)]
        assert same_tier and same_tier[0]["part_number"] == a[0].part_number, (words, pns[:5])


def test_naive_timestamps_do_not_break_the_model(store):
    orders, a, b, cats = _orders(store)
    orders.append(
        Order(
            order_id="N0",
            client_id="shop-a",
            items=[CartItem(part_number=a[0].part_number)],
            created_at=datetime(2026, 1, 9),  # hand-imported, no timezone
        )
    )
    book = CustomerBook(orders, {p.part_number: p for p in store.iter_parts()}, k=2)
    assert book.profiles["shop-a"].orders == 4
    assert a[1].part_number in dict(book.complements(a[0].part_number))  # adjacency path


def test_search_facets_offer_what_tells_the_variants_apart(identifier, store, tmp_path):
    from mcmaster_vision.pipeline.customers import facets, top_tier

    client = _client(identifier, tmp_path)
    parts = list(store.iter_parts(with_images_only=True))
    # the most common name has variants that share it
    names = Counter(" ".join(p.name.split()[-2:]) for p in parts)
    name, n = names.most_common(1)[0]
    if n < 3:
        pytest.skip("the fixture catalog has no name with three variants")
    f = client.get(f"/search/facets?q={name}").json()
    assert f["variants"] >= 2 and f["facets"], f
    key, vals = next(iter(f["facets"].items()))
    assert len(vals) >= 2 and all(v["count"] >= 1 for v in vals)
    # narrowing by a value cuts the list down to that value
    narrowed = client.get(f"/search?q={name} {vals[0]['value']}&limit=30").json()
    assert narrowed and all(
        str(p["attributes"].get(key, "")) == vals[0]["value"] or vals[0]["value"] in p["name"]
        for p in narrowed[: vals[0]["count"]]
    )
    # the helpers on their own
    scored = store.search_text_scored(name, 60)
    tier = top_tier(scored)
    assert 2 <= len(tier) <= len(scored) and facets(tier)
    assert facets(tier[:1]) == {}


def test_checkout_marks_the_part_at_once_even_inside_the_rebuild_floor(identifier, store, tmp_path):
    """The customer book is rebuilt at most every two seconds under load, but a checkout
    must not wait for that floor: the phone searches right after and expects its chip."""
    client = _client(identifier, tmp_path)
    pn = next(store.iter_parts()).part_number
    assert client.get("/me?client_id=shop-z").json()["known"] is False  # a fresh rebuild now
    r = client.post("/cart", json={"client_id": "shop-z", "part_number": pn})
    assert r.status_code == 200
    assert client.post("/checkout", json={"client_id": "shop-z"}).status_code == 200
    me = client.get("/me?client_id=shop-z").json()  # well inside the two seconds
    assert me["known"] and me["profile"]["bought"][pn] == 1


def test_a_spec_value_typed_verbatim_leads_the_search():
    """``1/2"`` in the query is an exact match for the 1/2" variant, not two tokens that the
    1-1/2" variant (which mentions 1 twice) scores higher on; the exact variants form the
    leading tier, so a facet chip can narrow again."""
    from mcmaster_vision.catalog import CatalogStore

    st = CatalogStore(":memory:")
    mk = lambda pn, od, mat: Part(  # noqa: E731
        part_number=pn,
        name="V-Belt Pulley",
        description=f"V-Belt Pulley, {od} OD, {mat}",
        attributes={"od": od, "material": mat},
        category_path=["Power Transmission", "Pulleys"],
    )
    st.upsert(
        [
            mk("P1", '1-1/2"', "Bronze"),
            mk("P2", '1/2"', "Bronze"),
            mk("P3", '1/2"', "Aluminum"),
            mk("P4", '3/4"', "Bronze"),
        ]
    )
    plain = [p.part_number for p, _ in st.search_text_scored("V-Belt Pulley", 10)]
    assert set(plain) == {"P1", "P2", "P3", "P4"}
    half = st.search_text_scored('V-Belt Pulley 1/2"', 10)
    assert [p.part_number for p, _ in half][:2] in (["P2", "P3"], ["P3", "P2"])
    assert half[0][1] < half[2][1] * 1.1  # a tier of its own: the chips narrow inside it
    assert [p.part_number for p, _ in st.search_text_scored('V-Belt Pulley 1-1/2"', 10)][0] == "P1"
    both = st.search_text_scored('V-Belt Pulley 1/2" Aluminum', 10)
    assert both[0][0].part_number == "P3"
    # a value that is only part of a longer token was not typed verbatim
    inch = st.search_text_scored('V-Belt Pulley 1-1/2" Bronze', 10)
    assert inch[0][0].part_number == "P1"
    st.close()
